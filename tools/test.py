from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved training runs on a test split.")
    parser.add_argument("--train-root", default="runs/train", help="Directory containing training run folders.")
    parser.add_argument("--result-file", default=None, help="Exact result JSON path for a single run; artifacts go directly under output-root.")
    parser.add_argument("--output-root", default="runs/test", help="Directory for per-run test result files.")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional run names, paths, or glob patterns under --train-root. Omit to test all runs.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint selector: best, last/latest, a checkpoint filename under checkpoints/, or a .pth path.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--device", default=None, help="Override training.device from the run snapshot.")
    parser.add_argument("--max-batches", type=int, default=0, help="Limit test batches; 0 disables.")
    parser.add_argument(
        "--tta",
        default=None,
        choices=["none", "flips", "d4"],
        help="Test-time augmentation. Defaults to the explicit config value, otherwise none.",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with an existing result JSON.")
    parser.add_argument("--no-profile", action="store_true", help="Skip parameter and FLOPs/MACs profiling.")
    parser.add_argument(
        "--profile-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Resize one sample to this size before FLOPs profiling. Defaults to the test sample size.",
    )
    parser.add_argument("--save-predictions", action="store_true", help="Save predicted label masks for the whole split.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save paper-ready panels, probability heatmaps, and error maps.",
    )
    parser.add_argument(
        "--vis-count",
        type=int,
        default=0,
        help="Number of samples to visualize. If --visualize is set and this is 0, saves 16 samples.",
    )
    parser.add_argument(
        "--heatmap-classes",
        default="foreground",
        help="Comma-separated probability heatmaps to save: foreground, all, or class ids such as 1,2,3.",
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save compressed per-pixel probability arrays. With no prediction/visualization filter, saves all samples.",
    )
    parser.add_argument("--paper-dpi", type=int, default=220, help="DPI for saved visualization figures.")
    parser.add_argument("--no-confusion-plot", action="store_true", help="Skip confusion matrix CSV/PNG artifacts.")
    parser.add_argument(
        "--save-sample-metrics",
        action="store_true",
        help="Save per-sample metrics CSV for error analysis; qualitative comparison samples must be fixed before inspecting scores.",
    )
    return parser.parse_args()


def discover_runs(train_root: Path, patterns: Iterable[str] | None) -> List[Path]:
    train_root = train_root.expanduser()
    if not train_root.exists():
        raise FileNotFoundError(f"Training root not found: {train_root}")

    all_runs = sorted(
        [path.parent for path in train_root.rglob("config_snapshot.json")]
        if not patterns or any(p.lower() in {"latest", "last"} for p in patterns) else [],
        key=lambda path: str(path),
    )
    if not patterns:
        return all_runs

    selected: list[Path] = []
    for pattern in patterns:
        if pattern.lower() in {"latest", "last"}:
            if all_runs:
                selected.append(max(all_runs, key=lambda path: path.stat().st_mtime))
            continue

        candidate = Path(pattern).expanduser()
        if candidate.exists():
            selected.append(candidate)
            continue

        exact = train_root / pattern
        if exact.exists():
            selected.append(exact)
            continue

        selected.extend(sorted(train_root.glob(pattern), key=lambda path: path.name))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in selected:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.is_dir() or not (path / "config_snapshot.json").exists():
            raise FileNotFoundError(f"Not a training run directory with config_snapshot.json: {path}")
        unique.append(path)
    if not unique:
        raise RuntimeError(f"No training runs matched: {list(patterns)}")
    return unique


def resolve_checkpoint(run_dir: Path, selector: str) -> Path:
    value = str(selector or "best")
    checkpoint_dir = run_dir / "checkpoints"
    path = Path(value).expanduser()
    if path.exists():
        return path

    lowered = value.lower()
    if lowered == "best":
        path = checkpoint_dir / "best.pth"
    elif lowered in {"last", "latest"}:
        path = checkpoint_dir / "last.pth"
        if not path.exists():
            path = latest_epoch_checkpoint(checkpoint_dir)
    else:
        path = checkpoint_dir / value

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {run_dir.name}: {path}")
    return path


def latest_epoch_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("epoch_*.pth"), key=checkpoint_epoch)
    if candidates:
        return candidates[-1]
    best = checkpoint_dir / "best.pth"
    if best.exists():
        return best
    raise FileNotFoundError(f"No epoch_*.pth or best.pth found under {checkpoint_dir}")


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else -1


def load_config_snapshot(run_dir: Path) -> Dict[str, Any]:
    with (run_dir / "config_snapshot.json").open("r", encoding="utf-8") as file:
        return copy.deepcopy(json.load(file))


def get_device(config: Dict[str, Any], override: str | None):
    import torch

    device_name = override or str(config.get("training", {}).get("device", "auto"))
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def apply_runtime_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    train_cfg = config.setdefault("training", {})
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.no_amp:
        train_cfg["amp"] = False
    return config


from utils.model_metadata import model_metadata
from utils.damage_metrics import DamageHeadMeter, localization_is_supervised
from utils.prediction import (predict_outputs, resolve_tta, decode_predictions, prediction_rule,
                              check_evaluation_tensors, validate_evaluation_labels, evaluation_coverage)
from utils.protocol import config_digest, file_digest, verify_checkpoint_config, dataset_identity


def evaluate_run(run_dir: Path, args: argparse.Namespace, artifact_dir: Path) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from datasets import build_dataset
    from models import build_model
    from utils.checkpoint import load_model_state
    from utils.losses import segmentation_loss
    from utils.metrics import ConfusionMatrixMeter, format_metrics, primary_metric_name

    config = apply_runtime_overrides(load_config_snapshot(run_dir), args)
    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint)
    device = get_device(config, args.device)
    train_cfg = config.get("training", {})
    from utils.runtime import configure_runtime
    execution = configure_runtime(train_cfg, device)
    inference_cfg = config.get("inference", {})
    task_cfg = config.get("task", {})
    num_classes = int(task_cfg.get("num_classes", config.get("model", {}).get("num_classes", 2)))
    ignore_index = int(task_cfg.get("ignore_index", 255))
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 4))
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    tta_mode = resolve_tta(config, args.tta)

    dataset = build_dataset(config, split=args.split, training=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # Trusted local run checkpoint.
    for unused in ("optimizer", "scheduler", "scaler", "rng_state"):
        checkpoint.pop(unused, None)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain a model state: {checkpoint_path}")

    verify_checkpoint_config(checkpoint, config)
    model = build_model(
        config,
        optical_channels=int(checkpoint.get("optical_channels", dataset.num_optical_channels)),
        sar_channels=int(checkpoint.get("sar_channels", dataset.num_sar_channels)),
        initialize_encoder=False,
    ).to(device)
    load_model_state(model, checkpoint["model"])
    model.eval()

    artifact_dir.mkdir(parents=True, exist_ok=True)
    visual_style = build_visual_style(config, num_classes)
    save_json(visual_style, artifact_dir / "visual_style.json")
    profile = None
    if not args.no_profile:
        try:
            profile = profile_model(model, dataset, device=device, args=args)
            save_json(profile, artifact_dir / "model_profile.json")
        except Exception as exc:
            profile = {"profile_error": str(exc)}
            save_json(profile, artifact_dir / "model_profile.json")

    meter = ConfusionMatrixMeter(num_classes=num_classes, ignore_index=ignore_index)
    damage_head_meter = DamageHeadMeter(num_classes, ignore_index,
        localization_supervised=localization_is_supervised(checkpoint.get("config", config)))
    total_loss = 0.0
    seen_batches = 0
    visualized = 0
    visualize_count = int(args.vis_count)
    if args.visualize and visualize_count <= 0:
        visualize_count = 16
    save_visuals = bool(args.visualize or visualize_count > 0)
    heatmap_specs = parse_heatmap_specs(args.heatmap_classes, num_classes)
    sample_metric_rows: list[Dict[str, Any]] = []
    seen_samples = 0
    sample_counts = []
    event_meters = {}
    sample_info = {s["id"]: s for s in dataset.samples}

    from utils.progress import RunProgress
    reporter = RunProgress(artifact_dir / "progress.json", stage="test")
    total_batches = min(len(loader), args.max_batches) if args.max_batches > 0 else len(loader)
    reporter.phase("test", total_batches)
    with torch.no_grad():
        progress = tqdm(loader, total=total_batches, desc=f"Test {run_dir.name}", leave=False,
                        dynamic_ncols=True, mininterval=1.0, disable=not sys.stderr.isatty())
        for step, batch in enumerate(progress, start=1):
            optical = batch["optical"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            validate_evaluation_labels(label, num_classes, ignore_index)
            model_output = predict_outputs(model, optical, sar, amp=amp, tta_mode=tta_mode)
            logits = model_output["logits"]
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
            pred = decode_predictions(model_output, config)
            damage_head_meter.update(model_output, label)
            if args.save_sample_metrics:
                pred_cpu, label_cpu = pred.detach().cpu(), batch["label"]
                for i, sample_id in enumerate(batch["id"]):
                    local = ConfusionMatrixMeter(num_classes, ignore_index).update(pred_cpu[i], label_cpu[i])
                    meter.matrix += local.matrix
                    sample_metric_rows.append({"id": sample_id, **local.compute()})
                    info = sample_info[sample_id]
                    sample_counts.append({"id": sample_id, "group": info.get("group"), "event": info.get("event"),
                                          "confusion_matrix": local.matrix.long().tolist()})
                    event = info.get("event")
                    if event:
                        event_meters.setdefault(event, ConfusionMatrixMeter(num_classes, ignore_index)).matrix += local.matrix
            else:
                meter.update(pred, label)
            reporter.batch(step)
            if args.save_predictions or save_visuals or args.save_probabilities:
                visualized = save_batch_artifacts(
                    batch=batch,
                    logits=logits,
                    pred=pred,
                    artifact_dir=artifact_dir,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    class_names=visual_style["class_names"],
                    class_colors=visual_style["class_colors"],
                    second_modality_title=visual_style["second_modality_title"],
                    second_modality_rgb=bool(visual_style["second_modality_rgb"]),
                    save_predictions=bool(args.save_predictions),
                    save_probabilities=bool(args.save_probabilities),
                    save_visuals=save_visuals,
                    visualized=visualized,
                    visualize_count=visualize_count,
                    heatmap_specs=heatmap_specs,
                    dpi=int(args.paper_dpi),
                )
            if args.max_batches > 0 and step >= args.max_batches:
                break

    coverage = evaluation_coverage(seen_samples, len(dataset), limit=args.max_batches)
    metrics = meter.compute()
    if metrics["valid_pixels"] <= 0:
        raise ValueError("Evaluation contains no valid labelled pixels")
    if args.save_sample_metrics:
        save_json(sample_counts, artifact_dir / "sample_confusion_matrices.json")
        save_json({e: {"confusion_matrix": m.matrix.long().tolist(), "metrics": m.compute()}
                   for e, m in event_meters.items()}, artifact_dir / "per_event_metrics.json")
    if args.save_sample_metrics and sample_metric_rows:
        write_sample_metrics(sample_metric_rows, artifact_dir / "sample_metrics.csv", num_classes=num_classes)
    confusion_plot_error = None
    if not args.no_confusion_plot:
        try:
            save_confusion_matrix_artifacts(
                meter.matrix,
                artifact_dir=artifact_dir,
                class_names=visual_style["class_names"],
            )
        except Exception as exc:
            confusion_plot_error = str(exc)
            warnings.warn(f"Confusion matrix plot skipped for {run_dir.name}: {exc}", stacklevel=2)
    best_name = primary_metric_name(num_classes, train_cfg.get("best_metric_resolved", train_cfg.get("best_metric", "auto")))
    result = {
        "schema_version": 2,
        **coverage,
        "config_sha256": config_digest(checkpoint.get("config", config)),
        "checkpoint_sha256": file_digest(checkpoint_path),
        "dataset_identity": dataset_identity(dataset),
        "seed": config.get("training", {}).get("seed"),
        "experiment": config.get("experiment", {}),
        "train_run": run_identifier(run_dir, config),
        "train_run_dir": str(run_dir),
        "artifact_dir": str(artifact_dir),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_selector": args.checkpoint,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_metric_name": checkpoint.get("best_metric_name"),
        "checkpoint_best_metric": float(checkpoint.get("best_metric", 0.0)),
        "split": args.split,
        "loss": total_loss / max(seen_batches, 1),
        "best_metric": best_name,
        "best_metric_value": float(metrics.get(best_name, metrics.get("primary_score", 0.0))),
        "metrics": metrics,
        "confusion_matrix": meter.matrix.long().tolist(),
        "damage_head_metrics": damage_head_meter.compute(),
        "model_profile": profile,
        "dataset": {
            "type": config.get("dataset", {}).get("type"),
            "name": config.get("dataset", {}).get("name"),
            "root": config.get("dataset", {}).get("root"),
            "size": len(dataset),
            "optical_channels": int(dataset.num_optical_channels),
            "sar_channels": int(dataset.num_sar_channels),
        },
        "implementation": model_metadata(model, device),
        "runtime": {
            "execution": execution,
            "device": str(device),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "amp": amp,
            "test_time_augmentation": tta_mode,
            "prediction_rule": prediction_rule(config),
            "max_batches": int(args.max_batches),
        },
        "artifacts": {
            "save_predictions": bool(args.save_predictions),
            "save_visualizations": bool(save_visuals),
            "visualized_samples": int(visualized),
            "heatmap_specs": heatmap_specs,
            "save_probabilities": bool(args.save_probabilities),
            "save_sample_metrics": bool(args.save_sample_metrics),
            "confusion_plot": not bool(args.no_confusion_plot),
            "confusion_plot_error": confusion_plot_error,
            "visual_style": visual_style,
        },
    }
    print(
        f"{run_dir.name}: loss={result['loss']:.4f} "
        f"{format_metrics(metrics, num_classes)} "
        f"{best_name}={result['best_metric_value']:.4f}"
    )
    reporter.finish("complete" if coverage["complete"] else "partial")
    return result


def run_identifier(run_dir, config):
    if run_dir.name.startswith("seed_") and config.get("experiment", {}).get("id"):
        return f"{config['experiment']['id']}__{run_dir.name}"
    return run_dir.name


def _is_haiti_config(config: Dict[str, Any]) -> bool:
    dataset_cfg = config.get("dataset", {})
    values = [
        dataset_cfg.get("type", ""),
        dataset_cfg.get("name", ""),
        dataset_cfg.get("root", ""),
    ]
    return any("haiti" in str(value).lower() for value in values)


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)


def profile_model(model, dataset, device, args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from utils.model_outputs import extract_logits

    sample = dataset[0]
    optical = sample["optical"].unsqueeze(0).to(device)
    sar = sample["sar"].unsqueeze(0).to(device)
    if args.profile_size is not None:
        size = tuple(int(v) for v in args.profile_size)
        optical = F.interpolate(optical, size=size, mode="bilinear", align_corners=False)
        sar = F.interpolate(sar, size=size, mode="bilinear", align_corners=False)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    param_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())

    macs_by_type: Dict[str, int] = {}
    macs_by_module: Dict[str, int] = {}
    hooks = []

    def add_macs(module_type: str, module_name: str, value: int) -> None:
        macs_by_type[module_type] = macs_by_type.get(module_type, 0) + int(value)
        macs_by_module[module_name] = macs_by_module.get(module_name, 0) + int(value)

    def conv_hook(module_name: str, module_type: str):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor) or not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or output.ndim < 4:
                return
            kernel_h, kernel_w = module.kernel_size
            in_channels = int(module.in_channels)
            groups = int(module.groups)
            out_elements = int(output.numel())
            macs_per_element = (in_channels // max(groups, 1)) * kernel_h * kernel_w
            add_macs(module_type, module_name, out_elements * macs_per_element)

        return hook

    def linear_hook(module_name: str):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            add_macs("Linear", module_name, int(output.numel()) * int(module.in_features))

        return hook

    def norm_hook(module_name: str, module_type: str):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                add_macs(module_type, module_name, int(output.numel()) * 2)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook(name, "Conv2d")))
        elif isinstance(module, nn.ConvTranspose2d):
            hooks.append(module.register_forward_hook(conv_hook(name, "ConvTranspose2d")))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook(name)))
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm)):
            hooks.append(module.register_forward_hook(norm_hook(name, module.__class__.__name__)))

    try:
        with torch.no_grad():
            logits = extract_logits(model(optical, sar))
    finally:
        for hook in hooks:
            hook.remove()

    total_macs = int(sum(macs_by_type.values()))
    top_modules = sorted(macs_by_module.items(), key=lambda item: item[1], reverse=True)[:30]
    return {
        "input_shape": {
            "optical": list(optical.shape),
            "sar": list(sar.shape),
            "logits": list(logits.shape),
        },
        "parameters": {
            "total": int(total_params),
            "trainable": int(trainable_params),
            "total_millions": float(total_params / 1e6),
            "trainable_millions": float(trainable_params / 1e6),
            "parameter_memory_mb": float(param_bytes / (1024 ** 2)),
            "buffer_memory_mb": float(buffer_bytes / (1024 ** 2)),
        },
        "complexity": {
            "macs": None,
            "partial_module_operation_estimate": total_macs,
            "gmacs": None,
            "flops_2x_macs": None,
            "gflops_2x_macs": None,
            "complete": False,
            "macs_by_type": {key: int(value) for key, value in sorted(macs_by_type.items())},
            "top_modules_by_macs": [
                {"module": name, "macs": int(value), "gmacs": float(value / 1e9)}
                for name, value in top_modules
            ],
            "note": "PARTIAL module-operation estimate, not total MACs/FLOPs. Excludes selective SSM scan, Swin functional attention/projections, einsum, grid_sample, HOG, interpolation and softmax. Norm estimates are mixed operations. Do not use it to rank architectures; measure same-device latency with tools/benchmark.py.",
        },
    }


def parse_heatmap_specs(value: str, num_classes: int) -> list[str]:
    specs: list[str] = []
    for item in str(value or "foreground").split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "all":
            specs.extend([f"class_{idx}" for idx in range(num_classes)])
        elif item in {"foreground", "fg", "change"}:
            specs.append("foreground")
        elif item.isdigit():
            idx = int(item)
            if 0 <= idx < num_classes:
                specs.append(f"class_{idx}")
        elif item.startswith("class_") and item[6:].isdigit():
            idx = int(item[6:])
            if 0 <= idx < num_classes:
                specs.append(f"class_{idx}")
    deduped: list[str] = []
    for item in specs:
        if item not in deduped:
            deduped.append(item)
    return deduped or ["foreground"]


def save_batch_artifacts(
    batch: Dict[str, Any],
    logits,
    pred,
    artifact_dir: Path,
    num_classes: int,
    ignore_index: int,
    class_names: list[str],
    class_colors: list[list[int]],
    second_modality_title: str,
    second_modality_rgb: bool,
    save_predictions: bool,
    save_probabilities: bool,
    save_visuals: bool,
    visualized: int,
    visualize_count: int,
    heatmap_specs: list[str],
    dpi: int,
) -> int:
    import numpy as np
    import torch

    prob = torch.softmax(logits.detach().float(), dim=1).cpu()
    pred_cpu = pred.detach().cpu()
    ids = batch_ids(batch, pred_cpu.shape[0])
    for idx, sample_id in enumerate(ids):
        safe_id = sanitize_filename(sample_id)
        pred_i = pred_cpu[idx]
        prob_i = prob[idx]
        label_i = batch["label"][idx].detach().cpu()
        optical_i = batch["optical"][idx].detach().cpu()
        sar_i = batch["sar"][idx].detach().cpu()

        if save_predictions:
            save_prediction_mask(pred_i, artifact_dir / "predictions" / "masks" / f"{safe_id}.png", num_classes)
            save_rgb_image(
                colorize_mask_array(
                    pred_i,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    class_colors=class_colors,
                ),
                artifact_dir / "predictions" / "colored_masks" / f"{safe_id}.png",
            )

        should_visualize = save_visuals and visualized < visualize_count
        if save_probabilities and (save_predictions or should_visualize or visualize_count <= 0):
            prob_path = artifact_dir / "probabilities" / f"{safe_id}.npz"
            prob_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(prob_path, prob=prob_i.numpy(), pred=pred_i.numpy())

        if not should_visualize:
            continue

        try:
            save_visual_panel(
                sample_id=sample_id,
                optical=optical_i,
                sar=sar_i,
                label=label_i,
                pred=pred_i,
                prob=prob_i,
                num_classes=num_classes,
                ignore_index=ignore_index,
                class_names=class_names,
                class_colors=class_colors,
                second_modality_title=second_modality_title,
                second_modality_rgb=second_modality_rgb,
                path=artifact_dir / "visualizations" / "panels" / f"{safe_id}.png",
                dpi=dpi,
            )
            save_rgb_image(
                error_map_array(pred_i, label_i, ignore_index=ignore_index),
                artifact_dir / "visualizations" / "errors" / f"{safe_id}.png",
            )
            for spec in heatmap_specs:
                heatmap = probability_heatmap(prob_i, spec)
                save_heatmap(
                    heatmap,
                    artifact_dir / "visualizations" / "heatmaps" / f"{safe_id}_{spec}.png",
                    title=f"{sample_id} {spec}",
                    dpi=dpi,
                )
            visualized += 1
        except Exception as exc:
            warnings.warn(f"Visualization skipped for {sample_id}: {exc}", stacklevel=2)
    return visualized


def batch_ids(batch: Dict[str, Any], batch_size: int) -> list[str]:
    ids = batch.get("id")
    if isinstance(ids, (list, tuple)):
        return [str(item) for item in ids]
    if ids is None:
        return [f"sample_{idx:05d}" for idx in range(batch_size)]
    return [str(ids)] if batch_size == 1 else [f"{ids}_{idx:03d}" for idx in range(batch_size)]


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._")
    return cleaned or "sample"


def save_prediction_mask(mask, path: Path, num_classes: int) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = mask.detach().cpu().numpy().astype("uint8")
    if int(num_classes) == 2:
        arr = arr * 255
    Image.fromarray(arr).save(path)


def save_rgb_image(array, path: Path) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def tensor_preview_array(tensor, sar: bool = False):
    import numpy as np

    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if sar:
        image = arr.mean(axis=0)
    elif arr.shape[0] >= 3:
        image = np.transpose(arr[:3], (1, 2, 0))
    else:
        image = arr[0]
    finite = np.isfinite(image)
    if not bool(finite.any()):
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(image[finite], [2, 98])
    return np.clip((image - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def colorize_mask_array(mask, num_classes: int, ignore_index: int = 255, class_colors: list[list[int]] | None = None):
    import numpy as np

    palette = np.asarray(class_colors or default_palette(num_classes), dtype=np.uint8)
    if num_classes > palette.shape[0]:
        rng = np.random.default_rng(7)
        extra = rng.integers(0, 255, size=(num_classes - palette.shape[0], 3), dtype=np.uint8)
        palette = np.concatenate([palette, extra], axis=0)
    arr = mask.detach().cpu().long().numpy()
    clean = np.clip(arr, 0, max(num_classes - 1, 0))
    color = palette[clean]
    if ignore_index is not None:
        color[arr == int(ignore_index)] = np.asarray([140, 140, 140], dtype=np.uint8)
    return color


def error_map_array(pred, label, ignore_index: int = 255):
    import numpy as np

    pred_np = pred.detach().cpu().long().numpy()
    label_np = label.detach().cpu().long().numpy()
    valid = label_np != int(ignore_index)
    pred_fg = pred_np > 0
    label_fg = label_np > 0
    error = np.zeros((*label_np.shape, 3), dtype=np.uint8)
    error[valid & ~pred_fg & ~label_fg] = np.asarray([20, 20, 20], dtype=np.uint8)
    error[valid & pred_fg & label_fg & (pred_np == label_np)] = np.asarray([42, 180, 88], dtype=np.uint8)
    error[valid & pred_fg & ~label_fg] = np.asarray([230, 42, 42], dtype=np.uint8)
    error[valid & ~pred_fg & label_fg] = np.asarray([48, 111, 230], dtype=np.uint8)
    error[valid & pred_fg & label_fg & (pred_np != label_np)] = np.asarray([250, 205, 40], dtype=np.uint8)
    error[~valid] = np.asarray([140, 140, 140], dtype=np.uint8)
    return error


def probability_heatmap(prob, spec: str):
    heatmap = prob[1:].sum(dim=0) if spec == "foreground" and prob.shape[0] > 1 else prob[0]
    if spec.startswith("class_"):
        idx = int(spec.split("_", 1)[1])
        heatmap = prob[idx]
    return heatmap.detach().cpu().float().clamp(0, 1).numpy()


def save_heatmap(heatmap, path: Path, title: str, dpi: int) -> None:
    plt = get_pyplot()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(4.5, 4.0))
    image = axis.imshow(heatmap, cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_visual_panel(
    sample_id: str,
    optical,
    sar,
    label,
    pred,
    prob,
    num_classes: int,
    ignore_index: int,
    class_names: list[str],
    class_colors: list[list[int]],
    second_modality_title: str,
    second_modality_rgb: bool,
    path: Path,
    dpi: int,
) -> None:
    plt = get_pyplot()

    path.parent.mkdir(parents=True, exist_ok=True)
    foreground_prob = probability_heatmap(prob, "foreground")
    panels = [
        ("Optical", tensor_preview_array(optical, sar=False), None, None),
        (
            second_modality_title,
            tensor_preview_array(sar, sar=not bool(second_modality_rgb)),
            None if second_modality_rgb else "gray",
            None,
        ),
        (
            "Ground truth",
            colorize_mask_array(
                label,
                num_classes=num_classes,
                ignore_index=ignore_index,
                class_colors=class_colors,
            ),
            None,
            None,
        ),
        (
            "Prediction",
            colorize_mask_array(
                pred,
                num_classes=num_classes,
                ignore_index=ignore_index,
                class_colors=class_colors,
            ),
            None,
            None,
        ),
        ("Foreground prob.", foreground_prob, "magma", (0.0, 1.0)),
        ("Error", error_map_array(pred, label, ignore_index=ignore_index), None, None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.4))
    for axis, (title, image, cmap, limits) in zip(axes, panels):
        if limits is None:
            axis.imshow(image, cmap=cmap)
        else:
            axis.imshow(image, cmap=cmap, vmin=limits[0], vmax=limits[1])
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(sample_id)
    add_class_legend(fig, class_names, class_colors, num_classes)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.92))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def compute_sample_metric_rows(batch: Dict[str, Any], pred, label, num_classes: int, ignore_index: int) -> list[Dict[str, Any]]:
    from utils.metrics import ConfusionMatrixMeter

    rows = []
    ids = batch_ids(batch, pred.shape[0])
    pred_cpu = pred.detach().cpu()
    label_cpu = label.detach().cpu()
    for idx, sample_id in enumerate(ids):
        meter = ConfusionMatrixMeter(num_classes=num_classes, ignore_index=ignore_index)
        meter.update(pred_cpu[idx], label_cpu[idx])
        metrics = meter.compute()
        row: Dict[str, Any] = {"id": sample_id}
        row.update(metrics)
        rows.append(row)
    return rows


def write_sample_metrics(rows: list[Dict[str, Any]], path: Path, num_classes: int) -> None:
    if not rows:
        return
    keys = ["id", *sorted({key for row in rows for key in row.keys() if key != "id"})]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_confusion_matrix_artifacts(matrix, artifact_dir: Path, class_names: list[str]) -> None:
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    plt = get_pyplot()

    artifact_dir.mkdir(parents=True, exist_ok=True)
    matrix_np = matrix.detach().cpu().numpy().astype(np.float64)
    row_count, col_count = matrix_np.shape
    row_labels = [class_names[idx] if idx < len(class_names) else f"class_{idx}" for idx in range(row_count)]
    col_labels = [class_names[idx] if idx < len(class_names) else f"class_{idx}" for idx in range(col_count)]
    csv_path = artifact_dir / "confusion_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["target/pred", *col_labels])
        for idx, row in enumerate(matrix_np):
            writer.writerow([row_labels[idx], *[int(value) for value in row]])

    row_sum = matrix_np.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix_np, row_sum, out=np.zeros_like(matrix_np), where=row_sum > 0)
    fig_width = max(4.8, min(9.0, 1.05 * col_count + 2.0))
    fig_height = max(4.4, min(8.5, 0.95 * row_count + 1.7))
    font_size = 13 if max(row_count, col_count) <= 5 else 11 if max(row_count, col_count) <= 8 else 9
    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.edgecolor": "#2f3437",
        "axes.linewidth": 1.1,
        "xtick.color": "#1f2933",
        "ytick.color": "#1f2933",
    }
    with plt.rc_context(style):
        fig, axis = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        fig.patch.set_facecolor("white")
        axis.set_facecolor("white")
        image = axis.imshow(normalized, cmap="YlGnBu", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.set_xticks(np.arange(col_count))
        axis.set_yticks(np.arange(row_count))
        axis.set_xticklabels(col_labels, rotation=28, ha="right", rotation_mode="anchor")
        axis.set_yticklabels(row_labels)
        axis.set_xlabel("Predicted", labelpad=8)
        axis.set_ylabel("Target", labelpad=8)
        axis.tick_params(axis="both", which="major", length=0, pad=6, labelsize=font_size)
        axis.set_xticks(np.arange(-0.5, col_count, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, row_count, 1), minor=True)
        axis.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            label.set_fontweight("bold")

        for y in range(row_count):
            for x in range(col_count):
                if matrix_np[y, x] <= 0 or row_sum[y, 0] <= 0:
                    continue
                value = float(normalized[y, x])
                percent = value * 100.0
                text = "100%" if percent >= 99.95 else "<0.1%" if percent < 0.05 else f"{percent:.1f}%"
                axis.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    fontweight="bold",
                    color="white" if value >= 0.52 else "#1f2933",
                )

        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=0, labelsize=max(font_size - 1, 8))
        for label in colorbar.ax.get_yticklabels():
            label.set_fontweight("bold")

        fig.savefig(artifact_dir / "confusion_matrix.png", dpi=600, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def add_class_legend(fig, class_names: list[str], class_colors: list[list[int]], num_classes: int) -> None:
    from matplotlib.patches import Patch

    handles = []
    for idx in range(min(num_classes, len(class_names), len(class_colors))):
        color = [channel / 255.0 for channel in class_colors[idx]]
        handles.append(Patch(facecolor=color, edgecolor="none", label=f"{idx}: {class_names[idx]}"))
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.01),
        )


def build_visual_style(config: Dict[str, Any], num_classes: int) -> Dict[str, Any]:
    dataset = dataset_key(config)
    class_names = default_class_names(config, num_classes, dataset)
    class_colors = default_class_colors(num_classes, dataset)
    return {
        "dataset": dataset,
        "num_classes": int(num_classes),
        "class_names": class_names,
        "class_colors": class_colors,
        "class_colors_hex": [rgb_to_hex(color) for color in class_colors[:num_classes]],
        "second_modality_title": str(config.get("dataset", {}).get("second_modality_name") or "SAR"),
        "second_modality_rgb": bool(config.get("dataset", {}).get("second_modality_rgb", False)),
        "ignore_color": [140, 140, 140],
        "error_legend": {
            "true_negative": [20, 20, 20],
            "true_positive_correct_class": [42, 180, 88],
            "false_positive": [230, 42, 42],
            "false_negative": [48, 111, 230],
            "foreground_class_confusion": [250, 205, 40],
            "ignore": [140, 140, 140],
        },
    }


def dataset_key(config: Dict[str, Any]) -> str:
    dataset_cfg = config.get("dataset", {})
    task_cfg = config.get("task", {})
    logging_cfg = config.get("logging", {})
    text = " ".join(
        str(value).lower()
        for value in [
            dataset_cfg.get("type", ""),
            dataset_cfg.get("name", ""),
            dataset_cfg.get("root", ""),
            task_cfg.get("name", ""),
            task_cfg.get("type", ""),
            logging_cfg.get("run_name", ""),
        ]
    )
    if "cau" in text or "flood" in text:
        return "cau_binary_flood"
    if "xbd" in text:
        return "xbd_multiclass_damage"
    if "haiti" in text:
        return "haiti_multiclass_change"
    if "bright" in text or "building_damage" in text:
        return "bright_multiclass_damage"
    return "binary_change" if int(config.get("task", {}).get("num_classes", 2)) == 2 else "multiclass"


def default_class_names(config: Dict[str, Any], num_classes: int, dataset: str | None = None) -> list[str]:
    dataset = dataset or dataset_key(config)
    if dataset == "bright_multiclass_damage" and int(num_classes) == 4:
        return ["Background", "Intact", "Damaged", "Destroyed"]

    task_cfg = config.get("task", {})
    configured = task_cfg.get("class_names") or config.get("dataset", {}).get("class_names")
    if isinstance(configured, list) and len(configured) >= num_classes:
        return [str(item) for item in configured[:num_classes]]
    if dataset == "cau_binary_flood":
        names = ["non-flood", "flood"]
    elif int(num_classes) == 2:
        names = ["background", "change"]
    elif dataset == "xbd_multiclass_damage":
        names = ["background", "no damage", "minor damage", "major damage", "destroyed"]
    elif dataset == "haiti_multiclass_change":
        names = ["background", "change class 1", "change class 2", "change class 3"]
    else:
        names = [f"class_{idx}" for idx in range(num_classes)]
    if len(names) < num_classes:
        names.extend(f"class_{idx}" for idx in range(len(names), num_classes))
    return names[:num_classes]


def default_class_colors(num_classes: int, dataset: str | None = None) -> list[list[int]]:
    if dataset == "cau_binary_flood":
        colors = [
            [30, 30, 30],
            [31, 119, 180],
        ]
    elif dataset == "bright_multiclass_damage":
        colors = [
            [25, 25, 25],
            [255, 216, 77],
            [245, 130, 48],
            [214, 39, 40],
        ]
    elif dataset == "xbd_multiclass_damage":
        colors = [
            [25, 25, 25],
            [55, 180, 85],
            [255, 216, 77],
            [245, 130, 48],
            [214, 39, 40],
        ]
    elif dataset == "haiti_multiclass_change":
        colors = [
            [25, 25, 25],
            [46, 204, 113],
            [52, 152, 219],
            [155, 89, 182],
        ]
    else:
        colors = default_palette(num_classes)
    if len(colors) < num_classes:
        colors.extend(default_palette(num_classes)[len(colors) : num_classes])
    return colors[:num_classes]


def default_palette(num_classes: int) -> list[list[int]]:
    colors = [
        [0, 0, 0],
        [230, 26, 26],
        [250, 158, 31],
        [250, 224, 46],
        [51, 115, 242],
        [38, 179, 89],
        [179, 64, 230],
        [26, 191, 191],
    ]
    if len(colors) < num_classes:
        import numpy as np

        rng = np.random.default_rng(7)
        extra = rng.integers(0, 255, size=(num_classes - len(colors), 3), dtype=np.uint8).tolist()
        colors.extend(extra)
    return colors[:num_classes]


def rgb_to_hex(color: list[int]) -> str:
    return "#" + "".join(f"{int(channel):02x}" for channel in color[:3])


def save_result(result: Dict[str, Any], output_root: Path, destination=None) -> Path:
    from utils.protocol import atomic_json
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = Path(destination) if destination else output_root / f"{result['train_run']}.json"
    atomic_json(output_path, result)
    return output_path


def write_summary(results: List[Dict[str, Any]], output_root: Path) -> None:
    if not results:
        return
    metric_keys = sorted({key for result in results for key in result.get("metrics", {})})
    fixed_fields = [
        "train_run",
        "split",
        "status",
        "complete",
        "evaluated_samples",
        "expected_samples",
        "artifact_dir",
        "checkpoint",
        "checkpoint_epoch",
        "checkpoint_best_metric_name",
        "checkpoint_best_metric",
        "loss",
        "best_metric",
        "best_metric_value",
        "params_total_m",
        "params_trainable_m",
        "gmacs",
        "gflops_2x_macs",
    ]
    fields = fixed_fields + [key for key in metric_keys if key not in fixed_fields]
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field) for field in fixed_fields}
            profile = result.get("model_profile") or {}
            params = profile.get("parameters") if isinstance(profile, dict) else {}
            complexity = profile.get("complexity") if isinstance(profile, dict) else {}
            row["params_total_m"] = (params or {}).get("total_millions")
            row["params_trainable_m"] = (params or {}).get("trainable_millions")
            row["gmacs"] = (complexity or {}).get("gmacs")
            row["gflops_2x_macs"] = (complexity or {}).get("gflops_2x_macs")
            row.update(result.get("metrics", {}))
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    train_root = Path(args.train_root)
    output_root = Path(args.output_root)
    runs = discover_runs(train_root, args.runs)
    if args.result_file and len(runs) != 1:
        raise ValueError("--result-file requires exactly one run")
    results: list[Dict[str, Any]] = []

    print(f"Discovered {len(runs)} training run(s). Saving test results to {output_root}")
    for run_dir in runs:
        identifier = run_identifier(run_dir, load_config_snapshot(run_dir))
        output_path = Path(args.result_file) if args.result_file else output_root / f"{identifier}.json"
        if args.skip_existing and output_path.exists():
            from tools.validate_results import validate_result
            existing = json.loads(output_path.read_text())
            effective = apply_runtime_overrides(load_config_snapshot(run_dir), args)
            try:
                validate_result(existing, expected_config=config_digest(effective), verify_files=True)
                runtime = existing.get("runtime", {})
                if (runtime.get("test_time_augmentation") != resolve_tta(effective, args.tta)
                        or runtime.get("prediction_rule") != prediction_rule(effective)
                        or existing.get("checkpoint_sha256") != file_digest(resolve_checkpoint(run_dir, args.checkpoint))):
                    raise ValueError("Existing result uses a different checkpoint or decoding rule")
            except (KeyError, ValueError, OSError) as exc:
                print(f"Re-evaluate existing result: {output_path} ({exc})")
            else:
                print(f"Skip verified complete result: {output_path}")
                continue
        try:
            result = evaluate_run(run_dir, args, artifact_dir=output_root if args.result_file else output_root / identifier)
        except Exception as exc:
            print(f"{run_dir.name}: FAILED - {exc}", file=sys.stderr)
            raise
        if args.result_file:
            write_summary([result], output_root)
        saved_path = save_result(result, output_root, destination=output_path)
        results.append(result)
        print(f"Saved {saved_path}")

    if not args.result_file:
        write_summary(results, output_root)
    if results:
        print(f"Saved summary: {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
