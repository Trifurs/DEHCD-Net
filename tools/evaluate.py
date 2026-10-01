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
from utils.model_metadata import model_metadata
from utils.damage_metrics import DamageHeadMeter, localization_is_supervised
from utils.prediction import (predict_outputs, resolve_tta, decode_predictions, prediction_rule,
                              check_evaluation_tensors, validate_evaluation_labels, evaluation_coverage)
from utils.protocol import verify_checkpoint_config, config_digest, file_digest, dataset_identity, atomic_json
from utils.logger import setup_logger
from utils.run_manager import create_run_dir, save_config_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heterogeneous CD checkpoint.")
    parser.add_argument("--config", default="configs/config.xml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=0, help="Limit evaluation batches; 0 disables.")
    parser.add_argument("--tta", choices=["none", "flips", "d4"], default=None)
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

    from utils.runtime import configure_runtime
    execution = configure_runtime(config.get("training", {}), device)
    dataset = build_dataset(config, split=args.split, training=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 4)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # Trusted local run checkpoint.
    verify_checkpoint_config(checkpoint, config)
    model = build_model(
        config,
        optical_channels=int(checkpoint.get("optical_channels", dataset.num_optical_channels)),
        sar_channels=int(checkpoint.get("sar_channels", dataset.num_sar_channels)),
        initialize_encoder=False,
    ).to(device)
    load_model_state(model, checkpoint["model"])
    model.eval()

    task_cfg = config.get("task", {})
    train_cfg = config.get("training", {})
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    num_classes = int(task_cfg.get("num_classes", 2))
    ignore_index = int(task_cfg.get("ignore_index", 255))
    meter = ConfusionMatrixMeter(num_classes=num_classes, ignore_index=ignore_index)
    damage_head_meter = DamageHeadMeter(num_classes, ignore_index,
        localization_supervised=localization_is_supervised(checkpoint.get("config", config)))
    total_loss = 0.0

    with torch.no_grad():
        seen_batches = 0
        seen_samples = 0
        for step, batch in enumerate(tqdm(loader, desc=f"Evaluate {args.split}"), start=1):
            optical = batch["optical"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            validate_evaluation_labels(label, num_classes, ignore_index)
            model_output = predict_outputs(model, optical, sar, amp=amp, tta_mode=resolve_tta(config, args.tta))
            loss = segmentation_loss(
                model_output,
                label,
                train_cfg,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            check_evaluation_tensors({"loss": loss})
            total_loss += float(loss.item())
            seen_batches += 1
            seen_samples += int(label.shape[0])
            meter.update(decode_predictions(model_output, config), label)
            damage_head_meter.update(model_output, label)
            if args.max_batches > 0 and step >= args.max_batches:
                break

    coverage = evaluation_coverage(seen_samples, len(dataset), limit=args.max_batches)
    metrics = meter.compute()
    if metrics["valid_pixels"] <= 0:
        raise ValueError("Evaluation contains no valid labelled pixels")
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
        json.dump({"schema_version": 2, **coverage, "split": args.split, "loss": loss_value, "best_metric": best_name, "metrics": metrics,
                   "confusion_matrix": meter.matrix.long().tolist(), "checkpoint": str(Path(checkpoint_path).resolve()),
                   "checkpoint_sha256": file_digest(checkpoint_path), "config_sha256": config_digest(checkpoint.get("config", config)),
                   "dataset_identity": dataset_identity(dataset), "seed": config.get("training", {}).get("seed"),
                   "damage_head_metrics": damage_head_meter.compute(), "implementation": model_metadata(model, device),
                   "runtime": {"execution": execution, "test_time_augmentation": resolve_tta(config, args.tta), "max_batches": args.max_batches, "prediction_rule": prediction_rule(config)},
                   "experiment": config.get("experiment", {})}, file, indent=2, allow_nan=False)
    print(format_metrics(metrics, num_classes))


if __name__ == "__main__":
    main()
