from __future__ import annotations

import argparse
import copy
import json
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
    parser = argparse.ArgumentParser(description="Evaluate heterogeneous CD checkpoint.")
    parser.add_argument("--config", default="configs/5090_x1/config.xml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=0, help="Limit evaluation batches; 0 disables.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from datasets import build_dataset
    from models import build_model
    from utils.losses import segmentation_loss
    from utils.metrics import ConfusionMatrixMeter, format_metrics, primary_metric_name
    from utils.model_outputs import extract_logits

    config: Dict[str, Any] = copy.deepcopy(XMLConfigParser(args.config).parse().as_dict())
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.patch_size is not None:
        config.setdefault("dataset", {})["patch_size"] = args.patch_size
        config["dataset"]["eval_full_image"] = False
    if args.num_workers is not None:
        config.setdefault("training", {})["num_workers"] = args.num_workers
    if args.no_amp:
        config.setdefault("training", {})["amp"] = False
    checkpoint_path = args.checkpoint or config.get("inference", {}).get("checkpoint")
    if not checkpoint_path:
        raise ValueError("Provide --checkpoint or inference.checkpoint in XML.")

    root_dir = str(config.get("logging", {}).get("root_dir", "runs"))
    run_name = str(config.get("logging", {}).get("run_name", Path(args.config).stem))
    run_dir = create_run_dir(root_dir, "evaluate", f"{run_name}_{args.split}")
    config.setdefault("evaluation", {})["run_dir"] = str(run_dir)
    save_config_snapshot(config, run_dir)
    log_dir = config.get("logging", {}).get("log_dir")
    if str(log_dir).lower() in {"", "auto", "none"}:
        log_dir = str(run_dir / "logs")
    logger = setup_logger(log_dir=log_dir)
    logger.info("Evaluation run directory: %s", run_dir)
    device_name = str(config.get("training", {}).get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    dataset = build_dataset(config, split=args.split, training=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 4)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        config,
        optical_channels=int(checkpoint.get("optical_channels", dataset.num_optical_channels)),
        sar_channels=int(checkpoint.get("sar_channels", dataset.num_sar_channels)),
    ).to(device)
    load_model_state(model, checkpoint["model"])
    model.eval()

    task_cfg = config.get("task", {})
    train_cfg = config.get("training", {})
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    num_classes = int(task_cfg.get("num_classes", 2))
    ignore_index = int(task_cfg.get("ignore_index", 255))
    meter = ConfusionMatrixMeter(num_classes=num_classes, ignore_index=ignore_index)
    total_loss = 0.0

    with torch.no_grad():
        seen_batches = 0
        for step, batch in enumerate(tqdm(loader, desc=f"Evaluate {args.split}"), start=1):
            optical = batch["optical"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                model_output = model(optical, sar)
                loss = segmentation_loss(
                    model_output,
                    label,
                    train_cfg,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )
            logits = extract_logits(model_output)
            total_loss += float(loss.item())
            seen_batches += 1
            meter.update(torch.argmax(logits, dim=1), label)
            if args.max_batches > 0 and step >= args.max_batches:
                break

    metrics = meter.compute()
    best_name = primary_metric_name(num_classes, config.get("training", {}).get("best_metric", "auto"))
    loss_value = total_loss / max(seen_batches, 1)
    logger.info(
        "Evaluation split=%s loss=%.4f %s best_metric=%s=%.4f",
        args.split,
        loss_value,
        format_metrics(metrics, num_classes),
        best_name,
        float(metrics.get(best_name, metrics.get("primary_score", 0.0))),
    )
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"split": args.split, "loss": loss_value, "best_metric": best_name, "metrics": metrics}, file, indent=2)
    print(format_metrics(metrics, num_classes))


if __name__ == "__main__":
    main()
