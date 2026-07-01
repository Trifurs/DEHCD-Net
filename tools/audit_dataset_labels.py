from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_dataset
from datasets.disaster_dataset import prepare_label
from utils.config import XMLConfigParser
from utils.raster import read_raster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit label balance for configured disaster CD datasets.")
    parser.add_argument("--configs", nargs="+", required=True, help="XML config files to audit.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Dataset splits to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for config_path in args.configs:
        config = XMLConfigParser(config_path).parse()
        task_cfg = config.get("task", {})
        num_classes = int(task_cfg.get("num_classes", 2))
        ignore_index = int(task_cfg.get("ignore_index", 255))
        dataset_name = str(config.get("dataset", {}).get("name", Path(config_path).stem))
        print(f"\n{config_path} ({dataset_name}, classes={num_classes})")
        for split in args.splits:
            dataset = build_dataset(config, split=split, training=False)
            counts: Counter[int] = Counter()
            present_images: Counter[int] = Counter()
            foreground_ratios: list[float] = []
            class_ratios: dict[int, list[float]] = {idx: [] for idx in range(num_classes)}

            for sample in dataset.samples:
                if hasattr(dataset, "load_label_for_stats"):
                    label = dataset.load_label_for_stats(sample).numpy()
                else:
                    label_np, _ = read_raster(sample["label"])
                    label = torch.from_numpy(label_np[0] if label_np.ndim == 3 else label_np)
                    label = prepare_label(
                        label,
                        mode=dataset.label_mode,
                        ignore_index=ignore_index,
                        num_classes=num_classes,
                        extra_ignore_values=dataset.label_ignore_values,
                    ).numpy()
                valid = label != ignore_index
                valid_pixels = max(int(valid.sum()), 1)
                values, value_counts = np.unique(label[valid], return_counts=True)
                local = {int(value): int(count) for value, count in zip(values, value_counts)}
                counts.update(local)

                foreground_count = sum(local.get(idx, 0) for idx in range(1, num_classes))
                foreground_ratios.append(foreground_count / valid_pixels)
                for class_idx in range(num_classes):
                    ratio = local.get(class_idx, 0) / valid_pixels
                    class_ratios[class_idx].append(ratio)
                    if local.get(class_idx, 0) > 0:
                        present_images[class_idx] += 1

            total = max(sum(counts.values()), 1)
            pixel_ratio = [counts.get(idx, 0) / total for idx in range(num_classes)]
            fg = np.asarray(foreground_ratios, dtype=np.float32)
            print(f"  {split}: n={len(dataset.samples)} pixel_ratio={[round(x, 4) for x in pixel_ratio]}")
            print(
                "    image_fg_ratio mean/median/p10/p90/max="
                f"{[round(float(x), 4) for x in [fg.mean(), np.median(fg), np.percentile(fg, 10), np.percentile(fg, 90), fg.max()]]}"
            )
            print(f"    present_images={dict((idx, present_images[idx]) for idx in range(num_classes))}")
            for class_idx in range(1, num_classes):
                ratios = np.asarray(class_ratios[class_idx], dtype=np.float32)
                nonzero = ratios[ratios > 0]
                if nonzero.size == 0:
                    continue
                print(
                    f"    class{class_idx}_ratio_nonzero mean/median/p90/max="
                    f"{[round(float(x), 5) for x in [nonzero.mean(), np.median(nonzero), np.percentile(nonzero, 90), nonzero.max()]]}"
                )


if __name__ == "__main__":
    main()
