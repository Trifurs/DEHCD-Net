from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import XMLConfigParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save quick optical/SAR/label sample panels.")
    parser.add_argument("--config", default="configs/5090_x1/config.xml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--save-dir", default="outputs/sample_panels")
    return parser.parse_args()


def normalize_preview(tensor):
    import numpy as np

    arr = tensor.detach().cpu().float().numpy()
    if arr.shape[0] >= 3:
        img = arr[:3].transpose(1, 2, 0)
    else:
        img = arr[0]
    low, high = np.percentile(img, [2, 98])
    return np.clip((img - low) / max(high - low, 1e-6), 0, 1)


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt

    from datasets import build_dataset

    config: Dict[str, Any] = XMLConfigParser(args.config).parse().as_dict()
    dataset = build_dataset(config, split=args.split, training=False)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min(args.count, len(dataset))):
        item = dataset[idx]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        axes[0].imshow(normalize_preview(item["optical"]))
        axes[0].set_title("Optical")
        axes[1].imshow(normalize_preview(item["sar"]), cmap="gray")
        axes[1].set_title("SAR")
        axes[2].imshow(item["label"].numpy(), cmap="gray")
        axes[2].set_title("Label")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(item["id"])
        fig.tight_layout()
        fig.savefig(save_dir / f"{item['id']}.png", dpi=150)
        plt.close(fig)
    print(f"Saved sample panels to {save_dir}")


if __name__ == "__main__":
    main()
