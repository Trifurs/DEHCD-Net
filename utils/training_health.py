"""Fail before corrupting weights, and distinguish foreground collapse from NaNs."""
from __future__ import annotations

import torch


def gradient_statistics(model, diagnostic_threshold=100.0, foreach=False):
    gradients = [(name, p.grad.detach().float()) for name, p in model.named_parameters() if p.grad is not None]
    norms = (torch._foreach_norm([g for _, g in gradients]) if foreach and gradients
             else [torch.linalg.vector_norm(g) for _, g in gradients])
    entries = [(name, norm) for (name, _), norm in zip(gradients, norms)]
    if not entries:
        return torch.tensor(0.0), []
    total = torch.linalg.vector_norm(torch.stack([norm for _, norm in entries]).double()).cpu()
    details = []
    if not torch.isfinite(total) or total > diagnostic_threshold:
        values = torch.stack([norm for _, norm in entries]).cpu().tolist()
        import math
        details = [{"parameter": name, "norm": value if math.isfinite(value) else None}
                   for (name, _), value in zip(entries, values)]
        details.sort(key=lambda item: float("inf") if item["norm"] is None else item["norm"], reverse=True)
    return total, details[:10]


def check_finite(tensor, name, epoch, step, sample_ids):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(
            f"Non-finite {name}; epoch={epoch} batch={step} samples={sample_ids}. "
            "No optimizer update was made for this batch; inspect failure.json and last.pth.")


def check_finite_many(tensors, epoch, step, sample_ids):
    """Retain every check, synchronize once per group, identify failures precisely."""
    tensors = list(tensors)
    if not tensors:
        return
    flags = torch.stack([torch.isfinite(value).all() for _, value in tensors]).cpu()
    if not bool(flags.all()):
        for (name, value), finite in zip(tensors, flags):
            if not bool(finite):
                check_finite(value, name, epoch, step, sample_ids)


class CollapseMonitor:
    """Flag foreground collapse or a prolonged all-background stall, not ordinary noise."""
    def __init__(self, config, count=0):
        self.patience = int(config.get("collapse_patience", 0) or 0)
        self.warmup = int(config.get("collapse_warmup_epochs", 20))
        self.min_best = float(config.get("collapse_min_best", 0.1))
        self.relative_score = float(config.get("collapse_score_fraction", 0.02))
        self.relative_foreground = float(config.get("collapse_foreground_fraction", 0.01))
        self.count = count
        self.stall_patience = int(config.get("foreground_stall_patience", 0) or 0)
        self.stall_warmup = int(config.get("foreground_stall_warmup_epochs", 40))
        self.stall_count = 0

    def update(self, epoch, best, score, metrics):
        target = metrics.get("target_foreground_ratio", 0.0)
        pred = metrics.get("pred_foreground_ratio", 0.0)
        collapsed = (epoch >= self.warmup and best >= self.min_best and target > 0
                     and score <= best * self.relative_score
                     and pred <= target * self.relative_foreground)
        self.count = self.count + 1 if collapsed else 0
        stalled = (epoch >= self.stall_warmup and target > 0
                   and pred <= target * self.relative_foreground)
        self.stall_count = self.stall_count + 1 if stalled else 0
        if self.stall_patience > 0 and self.stall_count >= self.stall_patience:
            raise RuntimeError(
                f"Foreground learning stalled for {self.stall_count} validation checks: "
                f"predicted_foreground={pred:.6g}, target_foreground={target:.6g}, best={best:.6g}. "
                "Training stopped even though no strong foreground score was previously reached. "
                "Check foreground/background loss balance, masks and sampling; finite gradients can still yield all-background predictions.")
        if self.patience > 0 and self.count >= self.patience:
            raise RuntimeError(
                f"Foreground collapse for {self.count} validation checks: score={score:.6g}, "
                f"best={best:.6g}, predicted_foreground={pred:.6g}, target_foreground={target:.6g}. "
                "Training stopped; best/last checkpoints are retained. This is not proof of gradient explosion.")
