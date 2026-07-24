from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE_ROOT = "data/raw/xBD"
DEFAULT_OUTPUT_ROOT = "data/xBD"
SPLITS = ("train", "val", "test")
PRE_SUFFIX = "_pre_disaster.png"
POST_SUFFIX = "_post_disaster.png"
POST_RGB_SUFFIX = "_post_disaster_rgb.png"
TARGET_SUFFIX = "_building_damage.png"
IMAGE_DIR = "images"
MASK_DIR = "masks"
OUTPUT_PRE_DIR = "pre-event"
OUTPUT_POST_DIR = "post-event"
OUTPUT_TARGET_DIR = "target"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    disaster: str
    tier: str
    pre_path: Path
    post_path: Path
    label_path: Path


class _Progress:
    def __call__(self, iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split xBD samples, crop 1024x1024 pre/post images and post-disaster "
            "damage masks into 256x256 patches, and drop empty-label patches."
        )
    )
    parser.add_argument("--src-root", default=DEFAULT_SOURCE_ROOT, help="Raw xBD root containing tier directories.")
    parser.add_argument("--dst-root", default=DEFAULT_OUTPUT_ROOT, help="Output dataset root.")
    parser.add_argument("--tiers", default="tier3", help="Comma-separated xBD tiers to process, e.g. tier1,tier2,tier3.")
    parser.add_argument("--ratio", default="4,1,1", help="Train/val/test split ratio.")
    parser.add_argument("--split-mode", choices=("sample", "stratified-disaster", "disaster"), default="stratified-disaster")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-size", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument(
        "--label-ignore-values",
        default="5",
        help=(
            "Additional comma-separated label values excluded from foreground and class counts. "
            "Use 'none' to disable. xBD masks commonly use 5 for un-classified."
        ),
    )
    parser.add_argument("--min-foreground-pixels", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Only count patches; do not write files.")
    parser.add_argument("--overwrite", action="store_true", help="Remove dst-root before writing if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()
    tiers = [item.strip() for item in str(args.tiers).split(",") if item.strip()]
    ratio = parse_ratio(args.ratio)
    ignore_values = {int(args.ignore_index), *parse_int_list(args.label_ignore_values)}

    if not src_root.exists():
        raise FileNotFoundError(src_root)
    if not tiers:
        raise ValueError("--tiers must contain at least one tier name")
    if args.source_size <= 0 or args.patch_size <= 0:
        raise ValueError("--source-size and --patch-size must be positive")
    if args.source_size % args.patch_size != 0:
        raise ValueError("--source-size must be divisible by --patch-size")
    expected_patches = (args.source_size // args.patch_size) ** 2

    Image = import_pillow_image()
    tqdm = import_tqdm()

    samples, scan_stats = collect_samples(src_root, tiers)
    if not samples:
        raise RuntimeError(f"No paired xBD samples found under {src_root} for tiers={tiers}")

    partitions = split_samples(samples, ratio=ratio, seed=int(args.seed), mode=str(args.split_mode))

    stats: dict[str, Any] = {
        "source_root": str(src_root),
        "output_root": str(dst_root),
        "tiers": tiers,
        "ratio": list(ratio),
        "split_mode": str(args.split_mode),
        "seed": int(args.seed),
        "source_size": int(args.source_size),
        "patch_size": int(args.patch_size),
        "expected_patches_per_sample": int(expected_patches),
        "ignore_values": sorted(ignore_values),
        "min_foreground_pixels": int(args.min_foreground_pixels),
        "label_source": "post-disaster masks, excluding *_post_disaster_rgb.png",
        "output_layout": {
            "pre_event": OUTPUT_PRE_DIR,
            "post_event": OUTPUT_POST_DIR,
            "target": OUTPUT_TARGET_DIR,
            "pre_suffix": PRE_SUFFIX,
            "post_suffix": POST_SUFFIX,
            "target_suffix": TARGET_SUFFIX,
        },
        "scan": scan_stats,
        "splits": {},
        "total_source_samples": len(samples),
        "total_candidate_patches": 0,
        "total_kept_patches": 0,
        "total_dropped_background_patches": 0,
        "total_skipped_invalid_size": 0,
    }

    if not args.dry_run:
        prepare_output_root(dst_root, overwrite=bool(args.overwrite))
        for split in SPLITS:
            for subdir in (OUTPUT_PRE_DIR, OUTPUT_POST_DIR, OUTPUT_TARGET_DIR):
                (dst_root / split / subdir).mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_stats = crop_partition(
            samples=partitions[split],
            split=split,
            dst_root=dst_root,
            image_cls=Image,
            patch_size=int(args.patch_size),
            source_size=int(args.source_size),
            ignore_values=ignore_values,
            min_foreground_pixels=int(args.min_foreground_pixels),
            dry_run=bool(args.dry_run),
            tqdm=tqdm,
        )
        stats["splits"][split] = split_stats
        stats["total_candidate_patches"] += split_stats["candidate_patches"]
        stats["total_kept_patches"] += split_stats["kept_patches"]
        stats["total_dropped_background_patches"] += split_stats["dropped_background_patches"]
        stats["total_skipped_invalid_size"] += split_stats["skipped_invalid_size"]

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if not args.dry_run:
        (dst_root / "xbd1_crop256_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"xBD1 written to: {dst_root}")


def import_pillow_image() -> Any:
    try:
        from PIL import Image

        return Image
    except ImportError as exc:
        raise ImportError("Pillow is required. Install project dependencies with `pip install -r requirements.txt`.") from exc


def import_tqdm() -> Any:
    try:
        from tqdm import tqdm

        return tqdm
    except ImportError:
        return _Progress()


def parse_ratio(text: str) -> tuple[float, float, float]:
    parts = [item.strip() for item in text.replace(":", ",").split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("--ratio must contain exactly three numbers, e.g. 4,1,1")
    ratio = tuple(float(item) for item in parts)
    if any(item < 0 for item in ratio) or sum(ratio) <= 0:
        raise ValueError("--ratio values must be non-negative and sum to a positive value")
    return ratio  # type: ignore[return-value]


def parse_int_list(text: str) -> set[int]:
    if not text or text.strip().lower() in {"none", "null", "false"}:
        return set()
    return {int(item.strip()) for item in text.replace(":", ",").split(",") if item.strip()}


def collect_samples(src_root: Path, tiers: list[str]) -> tuple[list[Sample], dict[str, Any]]:
    samples: list[Sample] = []
    scan_stats: dict[str, Any] = {"tiers": {}, "paired_samples": 0, "missing_samples": 0}
    for tier in tiers:
        tier_root = src_root / tier
        image_root = tier_root / IMAGE_DIR
        mask_root = tier_root / MASK_DIR
        if not image_root.exists():
            raise FileNotFoundError(image_root)
        if not mask_root.exists():
            raise FileNotFoundError(mask_root)

        pre_map = strip_suffix_map(image_root.glob(f"*{PRE_SUFFIX}"), PRE_SUFFIX)
        post_map = strip_suffix_map(image_root.glob(f"*{POST_SUFFIX}"), POST_SUFFIX)
        label_map = strip_suffix_map(
            (path for path in mask_root.glob(f"*{POST_SUFFIX}") if not path.name.endswith(POST_RGB_SUFFIX)),
            POST_SUFFIX,
        )
        paired_ids = sorted(set(pre_map) & set(post_map) & set(label_map))
        all_ids = set(pre_map) | set(post_map) | set(label_map)
        missing = sorted(all_ids - set(paired_ids))
        for sample_id in paired_ids:
            samples.append(
                Sample(
                    sample_id=sample_id,
                    disaster=disaster_name(sample_id),
                    tier=tier,
                    pre_path=pre_map[sample_id],
                    post_path=post_map[sample_id],
                    label_path=label_map[sample_id],
                )
            )
        scan_stats["tiers"][tier] = {
            "pre_images": len(pre_map),
            "post_images": len(post_map),
            "post_masks": len(label_map),
            "paired_samples": len(paired_ids),
            "missing_samples": len(missing),
            "missing_examples": missing[:10],
        }
        scan_stats["paired_samples"] += len(paired_ids)
        scan_stats["missing_samples"] += len(missing)
    return samples, scan_stats


def strip_suffix_map(paths: Iterable[Path], suffix: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in paths:
        name = path.name
        if name.endswith(suffix):
            mapping[name[: -len(suffix)]] = path
    return mapping


def disaster_name(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0] if "_" in sample_id else sample_id


def split_samples(
    samples: list[Sample],
    ratio: tuple[float, float, float],
    seed: int,
    mode: str,
) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    if mode == "sample":
        shuffled = list(samples)
        rng.shuffle(shuffled)
        return partition_list(shuffled, ratio)
    if mode == "stratified-disaster":
        by_disaster: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            by_disaster[sample.disaster].append(sample)
        partitions = {split: [] for split in SPLITS}
        for disaster in sorted(by_disaster):
            group = list(by_disaster[disaster])
            rng.shuffle(group)
            group_partition = partition_list(group, ratio)
            for split in SPLITS:
                partitions[split].extend(group_partition[split])
        for split in SPLITS:
            partitions[split].sort(key=lambda sample: (sample.disaster, sample.sample_id))
        return partitions
    if mode == "disaster":
        by_disaster = defaultdict(list)
        for sample in samples:
            by_disaster[sample.disaster].append(sample)
        disasters = sorted(by_disaster)
        rng.shuffle(disasters)
        disaster_partitions = partition_list(disasters, ratio)
        return {
            split: sorted(
                [sample for disaster in disaster_partitions[split] for sample in by_disaster[disaster]],
                key=lambda sample: (sample.disaster, sample.sample_id),
            )
            for split in SPLITS
        }
    raise ValueError(f"Unsupported split mode: {mode}")


def partition_list(items: list[Any], ratio: tuple[float, float, float]) -> dict[str, list[Any]]:
    total = len(items)
    ratio_sum = float(sum(ratio))
    train_end = int(total * ratio[0] / ratio_sum)
    val_end = train_end + int(total * ratio[1] / ratio_sum)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def prepare_output_root(dst_root: Path, overwrite: bool) -> None:
    if dst_root.exists():
        if overwrite:
            shutil.rmtree(dst_root)
        elif any(dst_root.iterdir()):
            raise FileExistsError(f"{dst_root} already exists and is not empty. Use --overwrite to replace it.")
    dst_root.mkdir(parents=True, exist_ok=True)


def crop_partition(
    samples: list[Sample],
    split: str,
    dst_root: Path,
    image_cls: Any,
    patch_size: int,
    source_size: int,
    ignore_values: set[int],
    min_foreground_pixels: int,
    dry_run: bool,
    tqdm: Any,
) -> dict[str, Any]:
    split_stats: dict[str, Any] = {
        "source_samples": len(samples),
        "candidate_patches": 0,
        "kept_patches": 0,
        "dropped_background_patches": 0,
        "skipped_invalid_size": 0,
        "class_pixel_counts": {},
        "disasters": dict(sorted(disaster_counts(samples).items())),
    }
    for sample in tqdm(samples, desc=f"Processing {split}", unit="sample"):
        with image_cls.open(sample.pre_path) as pre_img, image_cls.open(sample.post_path) as post_img, image_cls.open(sample.label_path) as label_img:
            if not valid_sizes([pre_img.size, post_img.size, label_img.size], source_size):
                split_stats["skipped_invalid_size"] += 1
                continue
            for row in range(source_size // patch_size):
                for col in range(source_size // patch_size):
                    left = col * patch_size
                    upper = row * patch_size
                    box = (left, upper, left + patch_size, upper + patch_size)
                    label_patch = label_img.crop(box)
                    split_stats["candidate_patches"] += 1
                    patch_counts = count_label_pixels(label_patch, ignore_values)
                    foreground_pixels = sum(count for value, count in patch_counts.items() if value > 0)
                    if foreground_pixels < min_foreground_pixels:
                        split_stats["dropped_background_patches"] += 1
                        continue
                    merge_counts(split_stats["class_pixel_counts"], patch_counts)
                    split_stats["kept_patches"] += 1
                    if dry_run:
                        continue

                    patch_id = f"{sample.sample_id}_r{row:02d}_c{col:02d}"
                    pre_img.crop(box).save(dst_root / split / OUTPUT_PRE_DIR / f"{patch_id}{PRE_SUFFIX}")
                    post_img.crop(box).save(dst_root / split / OUTPUT_POST_DIR / f"{patch_id}{POST_SUFFIX}")
                    label_patch.save(dst_root / split / OUTPUT_TARGET_DIR / f"{patch_id}{TARGET_SUFFIX}")
    return split_stats


def valid_sizes(sizes: list[tuple[int, int]], source_size: int) -> bool:
    expected = (source_size, source_size)
    return all(size == expected for size in sizes)


def count_label_pixels(label_patch: Any, ignore_values: set[int]) -> dict[int, int]:
    if label_patch.mode not in ("1", "L", "P", "I"):
        label_patch = label_patch.convert("L")
    histogram = label_patch.histogram()
    counts: dict[int, int] = {}
    for value, count in enumerate(histogram):
        if count and value not in ignore_values:
            counts[int(value)] = int(count)
    return counts


def merge_counts(target: dict[str, int], patch_counts: dict[int, int]) -> None:
    for value, count in patch_counts.items():
        key = str(int(value))
        target[key] = int(target.get(key, 0)) + int(count)


def disaster_counts(samples: list[Sample]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[sample.disaster] += 1
    return counts


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
