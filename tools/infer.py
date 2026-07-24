from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import XMLConfigParser
from utils.checkpoint import load_model_state
from utils.logger import setup_logger
from utils.run_manager import create_run_dir, save_config_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and save prediction maps.")
    parser.add_argument("--config", default="configs/config.xml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit inference samples; 0 disables.")
    return parser.parse_args()


def save_png(array, path: Path) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array).astype("uint8")).save(path)


def tensor_to_preview(tensor):
    import numpy as np

    arr = tensor.detach().cpu().float().numpy()
    if arr.shape[0] >= 3:
        img = arr[:3].transpose(1, 2, 0)
    else:
        img = arr[0]
    low, high = np.percentile(img, [2, 98])
    img = np.clip((img - low) / max(high - low, 1e-6), 0, 1)
    return img


def save_panel(
    sample_id: str,
    optical,
    sar,
    pred,
    save_path: Path,
    num_classes: int = 2,
    second_modality_title: str = "SAR",
    second_modality_rgb: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(tensor_to_preview(optical))
    axes[0].set_title("Optical")
    axes[1].imshow(tensor_to_preview(sar), cmap=None if second_modality_rgb else "gray")
    axes[1].set_title(second_modality_title)
    axes[2].imshow(pred, cmap="tab10" if num_classes > 2 else "gray", vmin=0, vmax=max(num_classes - 1, 1))
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(sample_id)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from datasets import build_dataset
    from models import build_model
    from utils.model_outputs import extract_logits

    config: Dict[str, Any] = XMLConfigParser(args.config).parse().as_dict()
    inference_cfg = config.get("inference", {})
    checkpoint_path = args.checkpoint or inference_cfg.get("checkpoint")
    split = args.split or inference_cfg.get("split", "test")
    root_dir = str(config.get("logging", {}).get("root_dir", "runs"))
    run_name = str(config.get("logging", {}).get("run_name", Path(args.config).stem))
    run_dir = create_run_dir(root_dir, "predict", f"{run_name}_{split}")
    save_dir_value = args.save_dir or inference_cfg.get("save_dir", "auto")
    save_dir = run_dir / "predictions" if str(save_dir_value).lower() in {"", "auto", "none"} else Path(save_dir_value)
    config.setdefault("inference", {})["run_dir"] = str(run_dir)
    config["inference"]["save_dir"] = str(save_dir)
    save_config_snapshot(config, run_dir)

    log_dir = config.get("logging", {}).get("log_dir")
    if str(log_dir).lower() in {"", "auto", "none"}:
        log_dir = str(run_dir / "logs")
    logger = setup_logger(log_dir=log_dir)
    logger.info("Prediction run directory: %s", run_dir)
    device_name = str(config.get("training", {}).get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    dataset = build_dataset(config, split=split, training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        config,
        optical_channels=int(checkpoint.get("optical_channels", dataset.num_optical_channels)),
        sar_channels=int(checkpoint.get("sar_channels", dataset.num_sar_channels)),
    ).to(device)
    load_model_state(model, checkpoint["model"])
    model.eval()

    threshold = float(inference_cfg.get("threshold", 0.5))
    save_visualization = bool(inference_cfg.get("save_visualization", True))
    num_classes = int(config.get("task", {}).get("num_classes", logits_classes_from_checkpoint(checkpoint, config)))
    dataset_cfg = config.get("dataset", {})
    second_modality_title = str(dataset_cfg.get("second_modality_name") or "SAR")
    second_modality_rgb = bool(dataset_cfg.get("second_modality_rgb", False))
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc=f"Infer {split}"), start=1):
            optical = batch["optical"].to(device)
            sar = batch["sar"].to(device)
            logits = extract_logits(model(optical, sar))
            prob = torch.softmax(logits, dim=1)
            if logits.shape[1] == 2:
                pred = (prob[:, 1] >= threshold).squeeze(0).detach().cpu().numpy().astype("uint8")
            else:
                pred = torch.argmax(prob, dim=1).squeeze(0).detach().cpu().numpy().astype("uint8")
            sample_id = batch["id"][0]
            scale = 255 if logits.shape[1] == 2 else 1
            save_png(pred * scale, save_dir / "masks" / f"{sample_id}.png")
            if save_visualization:
                try:
                    save_panel(
                        sample_id,
                        batch["optical"][0],
                        batch["sar"][0],
                        pred,
                        save_dir / "visualizations" / f"{sample_id}.png",
                        num_classes=num_classes,
                        second_modality_title=second_modality_title,
                        second_modality_rgb=second_modality_rgb,
                    )
                except Exception as exc:
                    logger.warning("Visualization skipped for %s: %s", sample_id, exc)
            if args.max_samples > 0 and step >= args.max_samples:
                break
    logger.info("Saved predictions to %s", save_dir)


def logits_classes_from_checkpoint(checkpoint: Dict[str, Any], config: Dict[str, Any]) -> int:
    return int(config.get("model", {}).get("num_classes", config.get("task", {}).get("num_classes", 2)))


if __name__ == "__main__":
    main()
