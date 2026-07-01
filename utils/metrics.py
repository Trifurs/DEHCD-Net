from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch


def _valid_mask(target: torch.Tensor, ignore_index: Optional[int]) -> torch.Tensor:
    if ignore_index is None:
        return torch.ones_like(target, dtype=torch.bool)
    return target != ignore_index


def compute_change_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: Optional[int] = 255,
) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=1)
    return ConfusionMatrixMeter(num_classes, ignore_index).update(preds, target).compute()


@dataclass
class ConfusionMatrixMeter:
    num_classes: int = 2
    ignore_index: Optional[int] = 255

    def __post_init__(self) -> None:
        self.matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)

    def reset(self) -> None:
        self.matrix.zero_()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> "ConfusionMatrixMeter":
        preds = preds.detach().cpu().long().view(-1)
        target = target.detach().cpu().long().view(-1)
        mask = _valid_mask(target, self.ignore_index)
        preds = preds[mask]
        target = target[mask]
        class_mask = (target >= 0) & (target < self.num_classes)
        preds = preds[class_mask].clamp(0, self.num_classes - 1)
        target = target[class_mask]
        if target.numel() == 0:
            return self
        indices = target * self.num_classes + preds
        counts = torch.bincount(indices, minlength=self.num_classes ** 2).double()
        self.matrix += counts.reshape(self.num_classes, self.num_classes)
        return self

    def compute(self) -> Dict[str, float]:
        cm = self.matrix
        total = cm.sum().clamp(min=1.0)
        oa = torch.diag(cm).sum() / total

        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        union = tp + fp + fn
        valid_iou = union > 0

        precision = tp / (tp + fp).clamp(min=1.0)
        recall = tp / (tp + fn).clamp(min=1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
        iou = torch.where(valid_iou, tp / union.clamp(min=1.0), torch.zeros_like(tp))

        foreground_indices = list(range(1, self.num_classes)) if self.num_classes > 1 else [0]
        foreground = torch.tensor(foreground_indices, dtype=torch.long)
        foreground_iou = _mean_valid(iou[foreground], valid_iou[foreground])
        foreground_hmean_iou = _harmonic_mean_valid(iou[foreground], valid_iou[foreground])
        foreground_min_iou = _min_valid(iou[foreground], valid_iou[foreground])
        foreground_precision = _mean_valid(precision[foreground], valid_iou[foreground])
        foreground_recall = _mean_valid(recall[foreground], valid_iou[foreground])
        foreground_f1 = _mean_valid(f1[foreground], valid_iou[foreground])
        mean_iou = _mean_valid(iou, valid_iou)
        binary_foreground = _binary_foreground_metrics(cm)

        if self.num_classes == 2:
            primary_iou = float(iou[1].item()) if valid_iou[1] else 0.0
            primary_f1 = float(f1[1].item()) if valid_iou[1] else 0.0
            primary_precision = float(precision[1].item()) if valid_iou[1] else 0.0
            primary_recall = float(recall[1].item()) if valid_iou[1] else 0.0
        else:
            primary_iou = foreground_iou
            primary_f1 = foreground_f1
            primary_precision = foreground_precision
            primary_recall = foreground_recall

        metrics = {
            "oa": float(oa.item()),
            "precision": primary_precision,
            "recall": primary_recall,
            "f1": primary_f1,
            "iou": primary_iou,
            "change_iou": float(iou[1].item()) if self.num_classes > 1 and valid_iou[1] else 0.0,
            "change_f1": float(f1[1].item()) if self.num_classes > 1 and valid_iou[1] else 0.0,
            "foreground_miou": foreground_iou,
            "foreground_hmean_iou": foreground_hmean_iou,
            "foreground_min_iou": foreground_min_iou,
            "foreground_f1": foreground_f1,
            "foreground_precision": foreground_precision,
            "foreground_recall": foreground_recall,
            "binary_foreground_iou": binary_foreground["iou"],
            "binary_foreground_f1": binary_foreground["f1"],
            "binary_foreground_precision": binary_foreground["precision"],
            "binary_foreground_recall": binary_foreground["recall"],
            "mean_iou": mean_iou,
            "miou": mean_iou,
            "macro_precision": _mean_valid(precision, valid_iou),
            "macro_recall": _mean_valid(recall, valid_iou),
            "macro_f1": _mean_valid(f1, valid_iou),
            "primary_score": primary_iou,
            "valid_pixels": float(total.item()),
        }
        pred_total = cm.sum(dim=0).clamp(min=0.0)
        target_total = cm.sum(dim=1).clamp(min=0.0)
        if self.num_classes > 1:
            metrics["pred_foreground_ratio"] = float((pred_total[1:].sum() / total).item())
            metrics["target_foreground_ratio"] = float((target_total[1:].sum() / total).item())
        for idx in range(self.num_classes):
            metrics[f"class_{idx}_precision"] = float(precision[idx].item())
            metrics[f"class_{idx}_recall"] = float(recall[idx].item())
            metrics[f"class_{idx}_f1"] = float(f1[idx].item())
            metrics[f"class_{idx}_iou"] = float(iou[idx].item())
            metrics[f"pred_class_{idx}_ratio"] = float((pred_total[idx] / total).item())
            metrics[f"target_class_{idx}_ratio"] = float((target_total[idx] / total).item())
        return metrics


def _mean_valid(values: torch.Tensor, valid: torch.Tensor) -> float:
    valid_values = values[valid]
    if valid_values.numel() == 0:
        return 0.0
    return float(valid_values.mean().item())


def _harmonic_mean_valid(values: torch.Tensor, valid: torch.Tensor) -> float:
    valid_values = values[valid]
    if valid_values.numel() == 0:
        return 0.0
    if torch.any(valid_values <= 0):
        return 0.0
    return float(valid_values.numel() / torch.sum(1.0 / valid_values.clamp(min=1e-6)).item())


def _min_valid(values: torch.Tensor, valid: torch.Tensor) -> float:
    valid_values = values[valid]
    if valid_values.numel() == 0:
        return 0.0
    return float(valid_values.min().item())


def _binary_foreground_metrics(cm: torch.Tensor) -> Dict[str, float]:
    if cm.shape[0] <= 1:
        tp = cm[0, 0]
        pred = cm[:, 0].sum()
        target = cm[0, :].sum()
    else:
        tp = cm[1:, 1:].sum()
        pred = cm[:, 1:].sum()
        target = cm[1:, :].sum()
    fp = pred - tp
    fn = target - tp
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    iou = tp / (tp + fp + fn).clamp(min=1.0)
    return {
        "iou": float(iou.item()),
        "f1": float(f1.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
    }


def primary_metric_name(num_classes: int, configured: Optional[str] = None) -> str:
    configured_name = str(configured or "auto").strip().lower()
    if configured_name not in {"", "auto", "default"}:
        return configured_name
    return "iou" if int(num_classes) == 2 else "foreground_miou"


def display_metric_keys(num_classes: int) -> list[str]:
    if int(num_classes) == 2:
        return ["iou", "f1", "precision", "recall", "oa"]
    keys = ["foreground_miou", "foreground_hmean_iou", "binary_foreground_iou"]
    keys.extend([f"class_{idx}_iou" for idx in range(1, min(int(num_classes), 6))])
    keys.extend(["foreground_f1", "pred_foreground_ratio", "target_foreground_ratio", "oa"])
    return keys


def format_metrics(metrics: Dict[str, float], num_classes: int, keys: Optional[Iterable[str]] = None) -> str:
    selected = list(keys) if keys is not None else display_metric_keys(num_classes)
    return " ".join(f"{key}={float(metrics.get(key, 0.0)):.4f}" for key in selected)
