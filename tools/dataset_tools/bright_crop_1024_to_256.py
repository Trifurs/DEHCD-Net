from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.disaster_dataset import index_files, natural_key


SPLITS = ("train", "val", "test")
OPTICAL_DIR = "pre-event"
SAR_DIR = "post-event"
LABEL_DIR = "target"
OPTICAL_SUFFIX = "_pre_disaster.tif"
SAR_SUFFIX = "_post_disaster.tif"
LABEL_SUFFIX = "_building_damage.tif"
EXTENSIONS = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop BRIGHT1 1024x1024 tiles into foreground 256x256 patches.")
    parser.add_argument("--root", default="data/BRIGHT")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--replace-root", action="store_true", help="Replace root with cropped dataset after creating a backup.")
    parser.add_argument("--dry-run", action="store_true", help="Only count candidate/kept patches; do not write files.")
    parser.add_argument("--backup-suffix", default=None, help="Optional suffix for the backup directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    patch_size = int(args.patch_size)
    if not root.exists():
        raise FileNotFoundError(root)
    if patch_size <= 0:
        raise ValueError("--patch-size must be positive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = root.parent / f"{root.name}_crop{patch_size}_tmp_{timestamp}"
    backup_suffix = args.backup_suffix or f"1024_backup_{timestamp}"
    backup_root = root.parent / f"{root.name}_{backup_suffix}"

    stats: dict[str, Any] = {
        "source_root": str(root),
        "patch_size": patch_size,
        "ignore_index": int(args.ignore_index),
        "splits": {},
        "total_source_samples": 0,
        "total_candidate_patches": 0,
        "total_kept_patches": 0,
        "total_dropped_background_patches": 0,
    }

    import rasterio
    from rasterio.windows import Window

    if not args.dry_run:
        if output_root.exists():
            raise FileExistsError(output_root)
        output_root.mkdir(parents=True)

    try:
        for split in SPLITS:
            split_stats = crop_split(
                root=root,
                output_root=output_root,
                split=split,
                patch_size=patch_size,
                ignore_index=int(args.ignore_index),
                rasterio_module=rasterio,
                window_cls=Window,
                dry_run=bool(args.dry_run),
            )
            stats["splits"][split] = split_stats
            stats["total_source_samples"] += split_stats["source_samples"]
            stats["total_candidate_patches"] += split_stats["candidate_patches"]
            stats["total_kept_patches"] += split_stats["kept_patches"]
            stats["total_dropped_background_patches"] += split_stats["dropped_background_patches"]

        print(json.dumps(stats, ensure_ascii=False, indent=2))

        if args.dry_run:
            return

        write_json(output_root / "crop256_stats.json", stats)

        if args.replace_root:
            if backup_root.exists():
                raise FileExistsError(backup_root)
            shutil.move(str(root), str(backup_root))
            shutil.move(str(output_root), str(root))
            print(f"Replaced {root}")
            print(f"Backup saved at {backup_root}")
        else:
            print(f"Cropped dataset written to {output_root}")
            print("Run again with --replace-root to make it the active BRIGHT1 directory.")
    except Exception:
        if not args.dry_run and output_root.exists() and not args.replace_root:
            shutil.rmtree(output_root, ignore_errors=True)
        raise


def crop_split(
    root: Path,
    output_root: Path,
    split: str,
    patch_size: int,
    ignore_index: int,
    rasterio_module: Any,
    window_cls: Any,
    dry_run: bool,
) -> dict[str, Any]:
    split_root = root / split
    optical_root = split_root / OPTICAL_DIR
    sar_root = split_root / SAR_DIR
    label_root = split_root / LABEL_DIR
    for required in (optical_root, sar_root, label_root):
        if not required.exists():
            raise FileNotFoundError(required)

    optical_map = index_files(optical_root, OPTICAL_SUFFIX, EXTENSIONS)
    sar_map = index_files(sar_root, SAR_SUFFIX, EXTENSIONS)
    label_map = index_files(label_root, LABEL_SUFFIX, EXTENSIONS)
    sample_ids = sorted(set(optical_map) & set(sar_map) & set(label_map), key=natural_key)
    if not sample_ids:
        raise RuntimeError(f"No paired BRIGHT samples found for split {split}")

    if not dry_run:
        for subdir in (OPTICAL_DIR, SAR_DIR, LABEL_DIR):
            (output_root / split / subdir).mkdir(parents=True, exist_ok=True)

    split_stats = {
        "source_samples": len(sample_ids),
        "candidate_patches": 0,
        "kept_patches": 0,
        "dropped_background_patches": 0,
        "skipped_invalid_size": 0,
        "class_pixel_counts": {},
    }

    for sample_id in sample_ids:
        with rasterio_module.open(label_map[sample_id]) as label_src:
            height, width = int(label_src.height), int(label_src.width)
            if height % patch_size != 0 or width % patch_size != 0:
                split_stats["skipped_invalid_size"] += 1
                continue
            rows = height // patch_size
            cols = width // patch_size
            if rows * cols != 16:
                raise ValueError(
                    f"{label_map[sample_id]} yields {rows * cols} patches, expected exactly 16. "
                    f"Size={width}x{height}, patch_size={patch_size}."
                )

            for row in range(rows):
                for col in range(cols):
                    window = window_cls(col * patch_size, row * patch_size, patch_size, patch_size)
                    label_patch = label_src.read(1, window=window)
                    split_stats["candidate_patches"] += 1
                    foreground_mask = (label_patch != ignore_index) & (label_patch > 0)
                    if not bool(foreground_mask.any()):
                        split_stats["dropped_background_patches"] += 1
                        continue
                    update_class_counts(split_stats["class_pixel_counts"], label_patch, ignore_index)
                    split_stats["kept_patches"] += 1
                    if dry_run:
                        continue

                    patch_id = f"{sample_id}_r{row:02d}_c{col:02d}"
                    write_window(
                        src_path=optical_map[sample_id],
                        dst_path=output_root / split / OPTICAL_DIR / f"{patch_id}{OPTICAL_SUFFIX}",
                        window=window,
                        rasterio_module=rasterio_module,
                    )
                    write_window(
                        src_path=sar_map[sample_id],
                        dst_path=output_root / split / SAR_DIR / f"{patch_id}{SAR_SUFFIX}",
                        window=window,
                        rasterio_module=rasterio_module,
                    )
                    write_window(
                        src_path=label_map[sample_id],
                        dst_path=output_root / split / LABEL_DIR / f"{patch_id}{LABEL_SUFFIX}",
                        window=window,
                        rasterio_module=rasterio_module,
                    )
    return split_stats


def write_window(src_path: Path, dst_path: Path, window: Any, rasterio_module: Any) -> None:
    with rasterio_module.open(src_path) as src:
        patch = src.read(window=window)
        profile = src.profile.copy()
        profile.update(
            height=int(window.height),
            width=int(window.width),
            transform=src.window_transform(window),
        )
        with rasterio_module.open(dst_path, "w", **profile) as dst:
            dst.write(patch)


def update_class_counts(counter: dict[str, int], label_patch: Any, ignore_index: int) -> None:
    import numpy as np

    valid = label_patch[label_patch != ignore_index]
    values, counts = np.unique(valid, return_counts=True)
    for value, count in zip(values.tolist(), counts.tolist()):
        key = str(int(value))
        counter[key] = int(counter.get(key, 0)) + int(count)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
