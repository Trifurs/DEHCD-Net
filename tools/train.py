from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import XMLConfigParser
from utils.checkpoint import load_model_state, save_checkpoint
from utils.logger import ExperimentWriters, setup_logger
from utils.metrics import format_metrics, primary_metric_name
from utils.run_manager import create_run_dir, save_config_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heterogeneous optical-SAR CD model.")
    parser.add_argument("--config", default="configs/5090_x1/config.xml", help="XML configuration path.")
    parser.add_argument("--resume", default=None, help="Checkpoint path overriding XML training.resume.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs for quick runs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--patch-size", type=int, default=None, help="Override dataset.patch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers.")
    parser.add_argument("--device", default=None, help="Override training.device.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches per epoch; 0 disables.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit val batches per validation; 0 disables.")
    parser.add_argument("--checkpoint-dir", default=None, help="Override training.checkpoint_dir.")
    parser.add_argument("--log-dir", default=None, help="Override logging.log_dir.")
    parser.add_argument("--run-name", default=None, help="Override logging.run_name.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_dataloaders(config: Dict[str, Any], device):
    from torch.utils.data import DataLoader

    from datasets import build_dataset

    train_dataset = build_dataset(config, split="train", training=True)
    val_dataset = build_dataset(config, split="val", training=False)
    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 4))
    persistent_workers = bool(train_cfg.get("persistent_workers", True)) and num_workers > 0
    pin_memory = device.type == "cuda"
    num_classes = int(config.get("task", {}).get("num_classes", 2))
    ignore_index = int(config.get("task", {}).get("ignore_index", 255))
    sampler = build_class_balanced_sampler(train_dataset, train_cfg, num_classes, ignore_index)
    train_dataset.sampler_info = getattr(sampler, "info", None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    return train_loader, val_loader, train_dataset


def compute_label_distribution(dataset, num_classes: int, ignore_index: int, max_samples: int = 0) -> Dict[str, Any]:
    import numpy as np
    import torch

    from datasets.disaster_dataset import prepare_label
    from utils.raster import read_raster

    labels = getattr(dataset, "samples", None)
    if not labels:
        return {"sample_counts": [], "counts": np.zeros(num_classes, dtype=np.float64), "class_frequencies": {}, "foreground_ratio": 0.0}

    samples = labels if max_samples <= 0 else labels[:max_samples]
    sample_counts = []
    global_counts = np.zeros(num_classes, dtype=np.float64)
    for sample in samples:
        if hasattr(dataset, "load_label_for_stats"):
            label = dataset.load_label_for_stats(sample).numpy()
        else:
            label_np, _ = read_raster(sample["label"])
            label = label_np[0] if label_np.ndim == 3 else label_np
            label = prepare_label(
                torch.from_numpy(label),
                mode=getattr(dataset, "label_mode", "multiclass_damage"),
                ignore_index=ignore_index,
                num_classes=num_classes,
                extra_ignore_values=getattr(dataset, "label_ignore_values", []),
            ).numpy()
        valid = (label != ignore_index) & (label >= 0) & (label < num_classes)
        counts = np.bincount(label[valid].astype(np.int64).reshape(-1), minlength=num_classes).astype(np.float64)
        sample_counts.append(counts)
        global_counts += counts
    total = max(float(global_counts.sum()), 1.0)
    frequencies = global_counts / total
    return {
        "sample_counts": sample_counts,
        "counts": global_counts,
        "class_frequencies": {idx: float(frequencies[idx]) for idx in range(num_classes)},
        "foreground_ratio": float(frequencies[1:].sum()) if num_classes > 1 else float(frequencies[0]),
        "sample_count": len(samples),
    }


def format_label_distribution(stats: Dict[str, Any], num_classes: int) -> str:
    freqs = stats.get("class_frequencies", {})
    class_text = ", ".join(f"{idx}:{float(freqs.get(idx, 0.0)):.4f}" for idx in range(num_classes))
    return f"samples={int(stats.get('sample_count', 0))} foreground_ratio={float(stats.get('foreground_ratio', 0.0)):.4f} class_ratios={{ {class_text} }}"


def build_class_balanced_sampler(dataset, train_cfg: Dict[str, Any], num_classes: int, ignore_index: int):
    import torch
    from torch.utils.data import WeightedRandomSampler

    if not bool(train_cfg.get("class_balanced_sampler", False)):
        return None

    max_samples = int(train_cfg.get("sampler_stat_samples", 0) or 0)
    stats = compute_label_distribution(dataset, num_classes, ignore_index, max_samples=max_samples)
    sample_counts = stats["sample_counts"]
    global_counts = stats["counts"]
    if not sample_counts or global_counts.sum() <= 0:
        return None

    freqs = global_counts / max(global_counts.sum(), 1.0)
    rare_threshold = float(train_cfg.get("sampler_rare_class_threshold", 0.03))
    target_classes = train_cfg.get("sampler_target_classes")
    if target_classes:
        rare_classes = [int(item) for item in target_classes if 0 < int(item) < num_classes]
    else:
        rare_classes = [idx for idx in range(1, num_classes) if freqs[idx] <= rare_threshold]
    if not rare_classes:
        return None

    power = float(train_cfg.get("sampler_power", 1.0))
    max_weight = float(train_cfg.get("sampler_max_weight", 8.0))
    min_pixels = int(train_cfg.get("sampler_min_pixels", 16))
    weights = []
    for counts in sample_counts:
        weight = 1.0
        for class_idx in rare_classes:
            if counts[class_idx] >= min_pixels:
                class_boost = (rare_threshold / max(freqs[class_idx], 1e-12)) ** power
                weight += class_boost
        weights.append(min(weight, max_weight))

    if max_samples > 0 and len(weights) < len(dataset):
        weights.extend([1.0] * (len(dataset) - len(weights)))
    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)
    sampler.info = {
        "rare_classes": rare_classes,
        "class_frequencies": {idx: float(freqs[idx]) for idx in range(num_classes)},
        "weight_min": float(weights_tensor.min().item()),
        "weight_mean": float(weights_tensor.mean().item()),
        "weight_max": float(weights_tensor.max().item()),
    }
    return sampler


def get_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def maybe_wrap_data_parallel(model, device, train_cfg: Dict[str, Any], logger):
    import torch

    if device.type != "cuda":
        return model
    mode = str(train_cfg.get("multi_gpu", "auto")).lower()
    if mode in {"false", "none", "off", "single"}:
        return model

    visible = torch.cuda.device_count()
    configured_ids = train_cfg.get("gpu_ids") or list(range(visible))
    gpu_ids = [int(idx) for idx in configured_ids if int(idx) < visible]
    if len(gpu_ids) <= 1:
        logger.info("Single GPU training on cuda:%s", gpu_ids[0] if gpu_ids else 0)
        return model
    logger.info("Using DataParallel on GPUs: %s", gpu_ids)
    return torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    train_cfg = config.setdefault("training", {})
    dataset_cfg = config.setdefault("dataset", {})
    log_cfg = config.setdefault("logging", {})
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.patch_size is not None:
        dataset_cfg["patch_size"] = args.patch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.max_train_batches is not None:
        train_cfg["max_train_batches"] = args.max_train_batches
    if args.max_val_batches is not None:
        train_cfg["max_val_batches"] = args.max_val_batches
    if args.checkpoint_dir is not None:
        train_cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.log_dir is not None:
        log_cfg["log_dir"] = args.log_dir
    if args.run_name is not None:
        log_cfg["run_name"] = args.run_name
    if args.no_amp:
        train_cfg["amp"] = False
    return config


def log_cuda_environment(device, logger) -> None:
    import torch

    if device.type != "cuda":
        return
    logger.info("CUDA runtime reported by PyTorch: %s", torch.version.cuda)
    logger.info("CUDA devices visible to PyTorch: %s", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        logger.info(
            "cuda:%s %s | capability=%s.%s | memory=%.1f GB",
            idx,
            props.name,
            props.major,
            props.minor,
            props.total_memory / (1024 ** 3),
        )


def build_scheduler(optimizer, train_cfg: Dict[str, Any], epochs: int):
    import math
    import torch

    scheduler_name = str(train_cfg.get("scheduler", "cosine_warmup")).lower()
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5) or 0)
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.01) or 0.0)
    if scheduler_name in {"none", "off"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if scheduler_name in {"cosine_warmup", "warmup_cosine", "cosine"}:
        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return max((epoch + 1) / warmup_epochs, min_lr_ratio)
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if scheduler_name in {"poly", "poly_warmup", "warmup_poly"}:
        power = float(train_cfg.get("poly_power", 0.9) or 0.9)

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return max((epoch + 1) / warmup_epochs, min_lr_ratio)
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            factor = (1.0 - min(progress, 1.0)) ** power
            return min_lr_ratio + (1.0 - min_lr_ratio) * factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if scheduler_name in {"warm_restarts", "cosine_restarts"}:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(int(train_cfg.get("restart_period", 10) or 10), 1),
            T_mult=max(int(train_cfg.get("restart_mult", 2) or 2), 1),
            eta_min=float(train_cfg.get("min_lr", 1e-6)),
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_optimizer(model, train_cfg: Dict[str, Any]):
    import torch

    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    betas = train_cfg.get("optimizer_betas", train_cfg.get("betas", [0.9, 0.999]))
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError(f"optimizer_betas must be a two-value list, got {betas!r}")
    eps = float(train_cfg.get("optimizer_eps", train_cfg.get("eps", 1e-8)))
    params = model.parameters()

    if name in {"adamw", "adamw_amsgrad"}:
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(float(betas[0]), float(betas[1])),
            eps=eps,
            amsgrad=bool(train_cfg.get("amsgrad", name == "adamw_amsgrad")),
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(float(betas[0]), float(betas[1])),
            eps=eps,
            amsgrad=bool(train_cfg.get("amsgrad", False)),
        )
    if name == "radam":
        return torch.optim.RAdam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(float(betas[0]), float(betas[1])),
            eps=eps,
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(train_cfg.get("momentum", 0.9)),
            nesterov=bool(train_cfg.get("nesterov", True)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def train_one_epoch(model, loader, optimizer, scaler, device, config, epoch: int) -> Tuple[float, Dict[str, float]]:
    import torch
    from tqdm import tqdm

    from utils.losses import segmentation_loss
    from utils.metrics import ConfusionMatrixMeter, primary_metric_name
    from utils.model_outputs import extract_logits

    train_cfg = config.get("training", {})
    task_cfg = config.get("task", {})
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0) or 0.0)
    max_batches = int(train_cfg.get("max_train_batches", 0) or 0)
    accumulation_steps = max(int(train_cfg.get("gradient_accumulation_steps", 1) or 1), 1)
    num_classes = int(task_cfg.get("num_classes", 2))
    ignore_index = int(task_cfg.get("ignore_index", 255))
    progress_metric = primary_metric_name(num_classes, "auto")
    meter = ConfusionMatrixMeter(num_classes=num_classes, ignore_index=ignore_index)

    model.train()
    total_loss = 0.0
    seen_batches = 0
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    progress = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        optical = batch["optical"].to(device, non_blocking=True)
        sar = batch["sar"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        is_accumulation_boundary = step % accumulation_steps == 0
        is_last_step = step == len(loader) or (max_batches > 0 and step >= max_batches)
        with torch.amp.autocast("cuda", enabled=amp):
            model_output = model(optical, sar)
            raw_loss = segmentation_loss(
                model_output,
                label,
                train_cfg,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            loss = raw_loss / accumulation_steps
        logits = extract_logits(model_output)
        scaler.scale(loss).backward()
        if is_accumulation_boundary or is_last_step:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scale_before = float(scaler.get_scale()) if amp else 1.0
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale()) if amp else 1.0
            if amp and scale_after < scale_before:
                skipped_optimizer_steps += 1
            else:
                optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(raw_loss.item())
        seen_batches += 1
        meter.update(torch.argmax(logits.detach(), dim=1), label)
        metrics = meter.compute()
        progress.set_postfix(loss=f"{raw_loss.item():.4f}", score=f"{metrics.get(progress_metric, 0.0):.4f}")
        if max_batches > 0 and step >= max_batches:
            break

    metrics = meter.compute()
    metrics["optimizer_steps"] = float(optimizer_steps)
    metrics["skipped_optimizer_steps"] = float(skipped_optimizer_steps)
    metrics["amp_scale"] = float(scaler.get_scale()) if amp else 0.0
    return total_loss / max(seen_batches, 1), metrics


def validate(model, loader, device, config) -> Tuple[float, Dict[str, float]]:
    import torch
    from tqdm import tqdm

    from utils.losses import segmentation_loss
    from utils.metrics import ConfusionMatrixMeter
    from utils.model_outputs import extract_logits

    train_cfg = config.get("training", {})
    task_cfg = config.get("task", {})
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    max_batches = int(train_cfg.get("max_val_batches", 0) or 0)
    num_classes = int(task_cfg.get("num_classes", 2))
    ignore_index = int(task_cfg.get("ignore_index", 255))
    meter = ConfusionMatrixMeter(num_classes=num_classes, ignore_index=ignore_index)
    total_loss = 0.0
    seen_batches = 0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Val", leave=False), start=1):
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
            if max_batches > 0 and step >= max_batches:
                break
    return total_loss / max(seen_batches, 1), meter.compute()


def log_validation_images(model, loader, device, config, writers: ExperimentWriters, epoch: int) -> None:
    import torch

    from utils.model_outputs import extract_logits
    from utils.visualization import make_validation_grid

    log_cfg = config.get("logging", {})
    max_items = int(log_cfg.get("validation_images", 2) or 0)
    if max_items <= 0 or (writers.tb_writer is None and writers.wandb is None):
        return

    task_cfg = config.get("task", {})
    train_cfg = config.get("training", {})
    num_classes = int(task_cfg.get("num_classes", 2))
    ignore_index = int(task_cfg.get("ignore_index", 255))
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    was_training = model.training
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        if was_training:
            model.train()
        return

    with torch.no_grad():
        optical = batch["optical"].to(device, non_blocking=True)
        sar = batch["sar"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = extract_logits(model(optical, sar))

    grid = make_validation_grid(
        optical.detach().cpu(),
        sar.detach().cpu(),
        label.detach().cpu(),
        logits.detach().cpu(),
        num_classes=num_classes,
        ignore_index=ignore_index,
        max_items=max_items,
    )
    writers.log_image("val/optical_sar_label_prediction", grid, epoch)
    if was_training:
        model.train()


def main() -> None:
    args = parse_args()
    import torch

    from models import build_model

    config = XMLConfigParser(args.config).parse().as_dict()
    config = apply_cli_overrides(config, args)
    train_cfg = config.get("training", {})
    log_cfg = config.get("logging", {})
    task_cfg = config.get("task", {})
    num_classes = int(task_cfg.get("num_classes", config.get("model", {}).get("num_classes", 2)))
    ignore_index = int(task_cfg.get("ignore_index", 255))
    best_name = primary_metric_name(num_classes, train_cfg.get("best_metric", "auto"))
    train_cfg["best_metric_resolved"] = best_name
    root_dir = str(log_cfg.get("root_dir", "runs"))
    run_name = str(log_cfg.get("run_name", Path(args.config).stem))
    run_dir = create_run_dir(root_dir, "train", run_name)
    log_cfg["run_dir"] = str(run_dir)
    if str(log_cfg.get("log_dir", "auto")).lower() in {"", "auto", "none"}:
        log_cfg["log_dir"] = str(run_dir / "logs")
    if str(train_cfg.get("checkpoint_dir", "auto")).lower() in {"", "auto", "none"}:
        train_cfg["checkpoint_dir"] = str(run_dir / "checkpoints")
    save_config_snapshot(config, run_dir)
    logger = setup_logger(log_dir=log_cfg.get("log_dir"))
    logger.info("Run directory: %s", run_dir)

    set_seed(int(train_cfg.get("seed", 42)))
    device = get_device(str(train_cfg.get("device", "auto")))
    logger.info("Using device: %s", device)
    log_cuda_environment(device, logger)

    train_loader, val_loader, train_dataset = build_dataloaders(config, device)
    model = build_model(
        config,
        optical_channels=train_dataset.num_optical_channels,
        sar_channels=train_dataset.num_sar_channels,
    ).to(device)
    logger.info(
        "Model channels: optical=%s, sar=%s",
        train_dataset.num_optical_channels,
        train_dataset.num_sar_channels,
    )
    label_stats = compute_label_distribution(
        train_dataset,
        num_classes=num_classes,
        ignore_index=ignore_index,
        max_samples=int(train_cfg.get("label_stat_samples", 0) or 0),
    )
    logger.info("Train label distribution: %s", format_label_distribution(label_stats, num_classes))
    if getattr(train_dataset, "sampler_info", None):
        logger.info("Class-balanced sampler: %s", train_dataset.sampler_info)
    logger.info("Best-checkpoint metric: %s", best_name)

    start_epoch = 1
    best_metric = -1.0
    checkpoint_dir = Path(str(train_cfg.get("checkpoint_dir", "checkpoints")))
    resume_path = args.resume or train_cfg.get("resume")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        load_model_state(model, checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        logger.info("Resumed from %s at epoch %s", resume_path, start_epoch)

    model = maybe_wrap_data_parallel(model, device, train_cfg, logger)
    optimizer = build_optimizer(model, train_cfg)
    epochs = int(train_cfg.get("epochs", 100))
    scheduler = build_scheduler(optimizer, train_cfg, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    early_stop_patience = int(train_cfg.get("early_stop_patience", 0) or 0)
    early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 0.0) or 0.0)
    epochs_without_improvement = 0

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])

    writers = ExperimentWriters(
        log_dir=str(log_cfg.get("log_dir", "runs/bright1")),
        enable_tensorboard=bool(log_cfg.get("tensorboard", True)),
        enable_wandb=bool(log_cfg.get("wandb", False)),
        wandb_project=log_cfg.get("wandb_project"),
        run_name=log_cfg.get("run_name"),
        config=config,
    )

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, config, epoch)
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        logger.info(
            "Epoch %03d train loss=%.4f %s lr=%.6g",
            epoch,
            train_loss,
            format_metrics(train_metrics, num_classes),
            current_lr,
        )
        skipped_steps = int(train_metrics.get("skipped_optimizer_steps", 0.0))
        optimizer_steps = int(train_metrics.get("optimizer_steps", 0.0))
        if skipped_steps > 0:
            logger.warning(
                "Epoch %03d skipped %s optimizer steps because AMP found non-finite gradients; final GradScaler scale=%.1f",
                epoch,
                skipped_steps,
                float(train_metrics.get("amp_scale", 0.0)),
            )
        if bool(train_cfg.get("amp", True)) and device.type == "cuda" and optimizer_steps == 0:
            raise RuntimeError(
                "No optimizer steps completed this epoch. AMP likely skipped every step because gradients were "
                "non-finite. Set training.amp=false for this config or reduce the unstable loss/model settings."
            )
        writers.log_scalar("train/loss", train_loss, epoch)
        writers.log_scalar("train/lr", current_lr, epoch)
        for key, value in train_metrics.items():
            writers.log_scalar(f"train/{key}", value, epoch)

        should_validate = epoch % int(train_cfg.get("val_interval", 1)) == 0
        if should_validate:
            val_loss, val_metrics = validate(model, val_loader, device, config)
            logger.info(
                "Epoch %03d val loss=%.4f %s",
                epoch,
                val_loss,
                format_metrics(val_metrics, num_classes),
            )
            writers.log_scalar("val/loss", val_loss, epoch)
            for key, value in val_metrics.items():
                writers.log_scalar(f"val/{key}", value, epoch)
            image_interval = max(int(log_cfg.get("validation_image_interval", train_cfg.get("val_interval", 1)) or 1), 1)
            if epoch % image_interval == 0:
                log_validation_images(model, val_loader, device, config, writers, epoch)

            current = float(val_metrics.get(best_name, val_metrics.get("primary_score", val_metrics["f1"])))
            if current > best_metric + early_stop_min_delta:
                best_metric = current
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_dir / "best.pth",
                    model,
                    epoch=epoch,
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict(),
                    best_metric=best_metric,
                    best_metric_name=best_name,
                    config=config,
                    optical_channels=train_dataset.num_optical_channels,
                    sar_channels=train_dataset.num_sar_channels,
                )
                logger.info("Saved new best checkpoint: %s=%.4f", best_name, best_metric)
            else:
                epochs_without_improvement += 1
                if early_stop_patience > 0:
                    logger.info(
                        "No %s improvement for %s/%s validation checks (min_delta=%.6g)",
                        best_name,
                        epochs_without_improvement,
                        early_stop_patience,
                        early_stop_min_delta,
                    )

        if epoch % int(train_cfg.get("save_interval", 5)) == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pth",
                model,
                epoch=epoch,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                best_metric=best_metric,
                best_metric_name=best_name,
                config=config,
                optical_channels=train_dataset.num_optical_channels,
                sar_channels=train_dataset.num_sar_channels,
            )

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            logger.info(
                "Early stopping at epoch %s after %s validation checks without %s improvement. Best %.4f.",
                epoch,
                epochs_without_improvement,
                best_name,
                best_metric,
            )
            break

    writers.close()
    logger.info("Training completed. Best %s=%.4f", best_name, best_metric)


if __name__ == "__main__":
    main()
