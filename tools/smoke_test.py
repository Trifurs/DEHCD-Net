from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import XMLConfigParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small data/model/loss/backward smoke test.")
    parser.add_argument("--config", default="configs/5090_x1/config.xml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument("--backward", action="store_true", help="Run one optimizer step.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from datasets import build_dataset
    from models import build_model
    from tools.train import maybe_wrap_data_parallel
    from utils.logger import setup_logger
    from utils.losses import segmentation_loss
    from utils.metrics import ConfusionMatrixMeter, format_metrics
    from utils.model_outputs import extract_logits

    config = XMLConfigParser(args.config).parse().as_dict()
    config = copy.deepcopy(config)
    config.setdefault("dataset", {})["patch_size"] = args.patch_size
    config["dataset"]["eval_full_image"] = False
    config.setdefault("training", {})["batch_size"] = args.batch_size
    config["training"]["num_workers"] = args.num_workers

    logger = setup_logger(log_dir=config.get("logging", {}).get("log_dir"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Smoke device: %s", device)

    dataset = build_dataset(config, split=args.split, training=args.split == "train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    batch = next(iter(loader))
    optical = batch["optical"].to(device)
    sar = batch["sar"].to(device)
    label = batch["label"].to(device)
    logger.info(
        "Batch ids=%s optical=%s sar=%s label=%s label_values=%s",
        list(batch["id"]),
        tuple(optical.shape),
        tuple(sar.shape),
        tuple(label.shape),
        torch.unique(label.detach().cpu()).tolist(),
    )

    model = build_model(
        config,
        optical_channels=dataset.num_optical_channels,
        sar_channels=dataset.num_sar_channels,
    ).to(device)
    model = maybe_wrap_data_parallel(model, device, config.get("training", {}), logger)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    amp = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start = time.time()
    num_classes = int(config.get("task", {}).get("num_classes", 2))
    ignore_index = int(config.get("task", {}).get("ignore_index", 255))
    with torch.amp.autocast("cuda", enabled=amp):
        model_output = model(optical, sar)
        loss = segmentation_loss(
            model_output,
            label,
            config.get("training", {}),
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
    logits = extract_logits(model_output)
    elapsed = time.time() - start
    metrics = ConfusionMatrixMeter(num_classes=num_classes, ignore_index=ignore_index).update(torch.argmax(logits, dim=1), label).compute()
    logger.info(
        "Forward logits=%s loss=%.6f %s amp=%s time=%.3fs",
        tuple(logits.shape),
        float(loss.item()),
        format_metrics(metrics, num_classes),
        amp,
        elapsed,
    )

    if args.backward:
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        logger.info("Backward+step completed in %.3fs", time.time() - start)


if __name__ == "__main__":
    main()
