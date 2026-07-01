from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from .model_outputs import extract_aux_logits, extract_feature_pairs, extract_logits


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: Optional[int] = 255,
    eps: float = 1e-6,
    present_only: bool = False,
    exclude_background: bool = False,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index
    safe_target = target.clone()
    safe_target[~valid] = 0

    one_hot = F.one_hot(safe_target.long(), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1).float()
    probs = probs * valid
    one_hot = one_hot * valid

    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)
    target_area = torch.sum(one_hot, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    class_mask = torch.ones(num_classes, device=logits.device, dtype=torch.bool)
    if present_only:
        class_mask &= target_area > 0
    if exclude_background and num_classes > 1:
        class_mask[0] = False
    if not bool(class_mask.any()):
        return logits.new_tensor(0.0)
    return 1.0 - dice[class_mask].mean()


def foreground_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: Optional[int] = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Binary foreground Dice over all non-background classes.

    Multiclass disaster labels often have tiny foreground areas. This auxiliary
    term first stabilizes foreground localization while CE/Dice still learn the
    individual damage classes.
    """

    if logits.shape[1] <= 1:
        return logits.new_tensor(0.0)

    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index
    if not bool(valid.any()):
        return logits.new_tensor(0.0)

    probs = torch.softmax(logits, dim=1)
    foreground_prob = probs[:, 1:].sum(dim=1)
    foreground_target = ((target > 0) & valid).to(dtype=foreground_prob.dtype)
    valid_float = valid.to(dtype=foreground_prob.dtype)
    foreground_prob = foreground_prob * valid_float

    target_area = foreground_target.sum()
    if target_area <= 0:
        return logits.new_tensor(0.0)
    intersection = (foreground_prob * foreground_target).sum()
    cardinality = foreground_prob.sum() + target_area
    return 1.0 - (2.0 * intersection + eps) / (cardinality + eps)


def hierarchical_change_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    config: Dict[str, Any],
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Decouple change localization from foreground-class discrimination.

    Background-heavy multiclass change labels can make every foreground class
    compete with the background at every background pixel, which often shows up
    as unstable foreground overprediction. This loss first learns
    background-vs-any-change over all valid pixels, then learns the foreground
    subclass only on pixels that are labelled foreground.
    """

    if num_classes <= 2:
        return _standard_segmentation_loss_tensor(logits, target, config, num_classes, ignore_index)

    valid = target != ignore_index
    if not bool(valid.any()):
        return logits.new_tensor(0.0)

    background_logit = logits[:, :1]
    foreground_logit = torch.logsumexp(logits[:, 1:], dim=1, keepdim=True)
    foreground_offset = config.get("hier_foreground_logit_offset")
    if foreground_offset in (None, "", "auto"):
        foreground_offset = -math.log(max(num_classes - 1, 1))
    foreground_logit = foreground_logit + float(foreground_offset)
    binary_logits = torch.cat([background_logit, foreground_logit], dim=1)
    binary_target = ((target > 0) & valid).long()
    binary_target = torch.where(valid, binary_target, torch.full_like(binary_target, ignore_index))

    binary_weights = make_class_weights(
        config.get("hier_binary_class_weights", config.get("binary_class_weights", [1.0, 2.5])),
        logits,
        2,
    )
    loss = logits.new_tensor(0.0)

    binary_ce_weight = float(config.get("hier_binary_ce_weight", 1.0) or 0.0)
    if binary_ce_weight > 0:
        loss = loss + binary_ce_weight * F.cross_entropy(
            binary_logits,
            binary_target,
            weight=binary_weights,
            ignore_index=ignore_index,
            label_smoothing=float(config.get("hier_binary_label_smoothing", 0.0) or 0.0),
        )

    binary_dice_weight = float(config.get("hier_binary_dice_weight", 0.7) or 0.0)
    if binary_dice_weight > 0:
        loss = loss + binary_dice_weight * dice_loss(
            binary_logits,
            binary_target,
            num_classes=2,
            ignore_index=ignore_index,
            present_only=False,
            exclude_background=False,
        )

    binary_focal_weight = float(config.get("hier_binary_focal_weight", 0.0) or 0.0)
    if binary_focal_weight > 0:
        loss = loss + binary_focal_weight * focal_loss(
            binary_logits,
            binary_target,
            alpha=binary_weights,
            gamma=float(config.get("hier_binary_focal_gamma", 2.0)),
            ignore_index=ignore_index,
        )

    foreground_mask = (target > 0) & valid
    if bool(foreground_mask.any()):
        foreground_logits = logits[:, 1:]
        foreground_target = (target - 1).clamp(min=0)
        subclass_weights = make_class_weights(
            config.get("hier_subclass_weights", config.get("subclass_weights")),
            logits,
            num_classes - 1,
        )
        subclass_ce_weight = float(config.get("hier_subclass_ce_weight", 0.8) or 0.0)
        if subclass_ce_weight > 0:
            loss = loss + subclass_ce_weight * masked_cross_entropy(
                foreground_logits,
                foreground_target,
                foreground_mask,
                weight=subclass_weights,
                label_smoothing=float(config.get("hier_subclass_label_smoothing", 0.02) or 0.0),
            )

        subclass_dice_weight = float(config.get("hier_subclass_dice_weight", 0.2) or 0.0)
        if subclass_dice_weight > 0:
            loss = loss + subclass_dice_weight * masked_dice_loss(
                foreground_logits,
                foreground_target,
                foreground_mask,
                num_classes=num_classes - 1,
                present_only=True,
            )

        subclass_focal_weight = float(config.get("hier_subclass_focal_weight", 0.0) or 0.0)
        if subclass_focal_weight > 0:
            loss = loss + subclass_focal_weight * masked_focal_loss(
                foreground_logits,
                foreground_target,
                foreground_mask,
                alpha=subclass_weights,
                gamma=float(config.get("hier_subclass_focal_gamma", 2.0)),
            )

    prior_weight = float(config.get("hier_foreground_prior_weight", 0.0) or 0.0)
    if prior_weight > 0:
        foreground_prob = torch.softmax(binary_logits, dim=1)[:, 1]
        valid_float = valid.to(dtype=foreground_prob.dtype)
        pred_ratio = (foreground_prob * valid_float).sum() / valid_float.sum().clamp(min=1.0)
        target_ratio = ((target > 0) & valid).to(dtype=foreground_prob.dtype).sum() / valid_float.sum().clamp(min=1.0)
        loss = loss + prior_weight * torch.abs(pred_ratio - target_ratio.detach())

    return loss


def ce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = 255,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    class_weights: Optional[Sequence[float] | torch.Tensor] = None,
) -> torch.Tensor:
    ce_class_weights = make_class_weights(class_weights, logits, num_classes)
    ce = F.cross_entropy(
        logits,
        target.long(),
        weight=ce_class_weights,
        ignore_index=ignore_index,
    )
    dl = dice_loss(logits, target, num_classes=num_classes, ignore_index=ignore_index)
    return ce_weight * ce + dice_weight * dl


def segmentation_loss(
    model_output: Any,
    target: torch.Tensor,
    config: Dict[str, Any],
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    logits = extract_logits(model_output)
    loss = _segmentation_loss_tensor(logits, target, config, num_classes, ignore_index)
    aux_logits = extract_aux_logits(model_output)
    aux_weight = float(config.get("aux_loss_weight", config.get("deep_supervision_weight", 0.0)) or 0.0)
    if aux_weight > 0 and aux_logits:
        aux_losses = []
        for aux in aux_logits:
            if aux.shape[-2:] != target.shape[-2:]:
                aux = F.interpolate(aux, size=target.shape[-2:], mode="bilinear", align_corners=False)
            aux_losses.append(_segmentation_loss_tensor(aux, target, config, num_classes, ignore_index))
        if aux_losses:
            loss = loss + aux_weight * torch.stack(aux_losses).mean()

    feature_pair_weight = float(
        config.get("feature_pair_loss_weight", config.get("modality_alignment_loss_weight", 0.0)) or 0.0
    )
    feature_pairs = extract_feature_pairs(model_output)
    if feature_pair_weight > 0 and feature_pairs:
        loss = loss + feature_pair_weight * feature_pair_alignment_loss(feature_pairs)
    return loss


def feature_pair_alignment_loss(feature_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    losses = []
    for left, right in feature_pairs:
        if left.ndim != 4 or right.ndim != 4 or left.shape[1] != right.shape[1]:
            continue
        if right.shape[-2:] != left.shape[-2:]:
            right = F.interpolate(right, size=left.shape[-2:], mode="bilinear", align_corners=False)
        left_mean = left.mean(dim=(2, 3))
        right_mean = right.mean(dim=(2, 3))
        left_std = left.var(dim=(2, 3), unbiased=False).clamp_min(1e-6).sqrt()
        right_std = right.var(dim=(2, 3), unbiased=False).clamp_min(1e-6).sqrt()
        losses.append(F.smooth_l1_loss(left_mean, right_mean) + 0.5 * F.smooth_l1_loss(left_std, right_std))
    if not losses:
        first = feature_pairs[0][0]
        return first.new_tensor(0.0)
    return torch.stack(losses).mean()


def _segmentation_loss_tensor(
    logits: torch.Tensor,
    target: torch.Tensor,
    config: Dict[str, Any],
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    loss_name = str(config.get("loss", "ce_dice")).strip().lower()
    if loss_name in {"hierarchical_change", "hierarchical", "change_hierarchy", "binary_subclass"}:
        return hierarchical_change_loss(logits, target, config, num_classes, ignore_index)
    return _standard_segmentation_loss_tensor(logits, target, config, num_classes, ignore_index)


def _standard_segmentation_loss_tensor(
    logits: torch.Tensor,
    target: torch.Tensor,
    config: Dict[str, Any],
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    class_weights = config.get("class_weights")
    ce_weight = float(config.get("ce_weight", 1.0))
    dice_weight = float(config.get("dice_weight", 1.0))
    focal_weight = float(config.get("focal_weight", 0.0) or 0.0)
    foreground_dice_weight = float(
        config.get("foreground_dice_weight", config.get("binary_foreground_dice_weight", 0.0)) or 0.0
    )
    lovasz_weight = float(config.get("lovasz_weight", 0.0) or 0.0)
    tversky_weight = float(config.get("tversky_weight", 0.0) or 0.0)
    label_smoothing = float(config.get("label_smoothing", 0.0) or 0.0)
    dice_present_only = bool(config.get("dice_present_only", num_classes > 2))
    dice_exclude_background = bool(config.get("dice_exclude_background", num_classes > 2))

    ce_class_weights = make_class_weights(class_weights, logits, num_classes)
    loss = logits.new_tensor(0.0)
    if ce_weight > 0:
        loss = loss + ce_weight * F.cross_entropy(
            logits,
            target.long(),
            weight=ce_class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
    if dice_weight > 0:
        loss = loss + dice_weight * dice_loss(
            logits,
            target,
            num_classes=num_classes,
            ignore_index=ignore_index,
            present_only=dice_present_only,
            exclude_background=dice_exclude_background,
        )
    if focal_weight > 0:
        loss = loss + focal_weight * focal_loss(
            logits,
            target,
            alpha=ce_class_weights,
            gamma=float(config.get("focal_gamma", 2.0)),
            ignore_index=ignore_index,
        )
    if foreground_dice_weight > 0 and num_classes > 2:
        loss = loss + foreground_dice_weight * foreground_dice_loss(
            logits,
            target,
            ignore_index=ignore_index,
        )
    if lovasz_weight > 0:
        loss = loss + lovasz_weight * lovasz_softmax_loss(
            torch.softmax(logits, dim=1),
            target,
            ignore_index=ignore_index,
            exclude_background=bool(config.get("lovasz_exclude_background", num_classes > 2)),
        )
    if tversky_weight > 0:
        loss = loss + tversky_weight * tversky_loss(
            logits,
            target,
            num_classes=num_classes,
            ignore_index=ignore_index,
            alpha=float(config.get("tversky_alpha", 0.3)),
            beta=float(config.get("tversky_beta", 0.7)),
            present_only=bool(config.get("tversky_present_only", True)),
            exclude_background=bool(config.get("tversky_exclude_background", num_classes > 2)),
        )
    return loss


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    logits_flat = logits.permute(0, 2, 3, 1)[mask]
    target_flat = target[mask].long()
    if target_flat.numel() == 0:
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits_flat, target_flat, weight=weight, label_smoothing=label_smoothing)


def masked_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    present_only: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    safe_target = target.clamp(min=0, max=num_classes - 1).long()
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    mask_float = mask.unsqueeze(1).to(dtype=probs.dtype)
    probs = probs * mask_float
    one_hot = one_hot * mask_float
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)
    target_area = torch.sum(one_hot, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    class_mask = torch.ones(num_classes, device=logits.device, dtype=torch.bool)
    if present_only:
        class_mask &= target_area > 0
    if not bool(class_mask.any()):
        return logits.new_tensor(0.0)
    return 1.0 - dice[class_mask].mean()


def masked_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    logits_flat = logits.permute(0, 2, 3, 1)[mask]
    target_flat = target[mask].long()
    if target_flat.numel() == 0:
        return logits.new_tensor(0.0)
    ce = F.cross_entropy(logits_flat, target_flat, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt).pow(gamma) * ce).mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    ignore_index: int = 255,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target.long(), weight=alpha, ignore_index=ignore_index, reduction="none")
    valid = target != ignore_index
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    pt = torch.exp(-ce[valid])
    return ((1.0 - pt).pow(gamma) * ce[valid]).mean()


def lovasz_softmax_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: Optional[int] = 255,
    exclude_background: bool = False,
) -> torch.Tensor:
    losses = []
    for prob, label in zip(probs, labels):
        flat_probs, flat_labels = flatten_probs(prob.unsqueeze(0), label.unsqueeze(0), ignore_index)
        if flat_labels.numel() == 0:
            continue
        losses.append(lovasz_softmax_flat(flat_probs, flat_labels, exclude_background=exclude_background))
    if not losses:
        return probs.new_tensor(0.0)
    return torch.stack(losses).mean()


def flatten_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, probs.shape[1])
    labels = labels.contiguous().view(-1)
    if ignore_index is None:
        return probs, labels
    valid = labels != ignore_index
    return probs[valid], labels[valid]


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor, exclude_background: bool = False) -> torch.Tensor:
    num_classes = probs.shape[1]
    losses = []
    start_class = 1 if exclude_background and num_classes > 1 else 0
    for class_idx in range(start_class, num_classes):
        fg = (labels == class_idx).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, class_idx]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if not losses:
        return probs.new_tensor(0.0)
    return torch.stack(losses).mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = 255,
    alpha: float = 0.3,
    beta: float = 0.7,
    present_only: bool = True,
    exclude_background: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index
    safe_target = target.clone()
    safe_target[~valid] = 0
    one_hot = F.one_hot(safe_target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1).float()
    probs = probs * valid
    one_hot = one_hot * valid

    dims = (0, 2, 3)
    tp = torch.sum(probs * one_hot, dims)
    fp = torch.sum(probs * (1.0 - one_hot), dims)
    fn = torch.sum((1.0 - probs) * one_hot, dims)
    target_area = torch.sum(one_hot, dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    class_mask = torch.ones(num_classes, device=logits.device, dtype=torch.bool)
    if present_only:
        class_mask &= target_area > 0
    if exclude_background and num_classes > 1:
        class_mask[0] = False
    if not bool(class_mask.any()):
        return logits.new_tensor(0.0)
    return 1.0 - tversky[class_mask].mean()


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def make_class_weights(
    class_weights: Optional[Sequence[float] | torch.Tensor],
    reference: torch.Tensor,
    num_classes: int,
) -> Optional[torch.Tensor]:
    if class_weights is None:
        return None
    if isinstance(class_weights, torch.Tensor):
        weight = class_weights.to(device=reference.device, dtype=reference.dtype)
    else:
        if len(class_weights) == 0:
            return None
        weight = torch.as_tensor(class_weights, device=reference.device, dtype=reference.dtype)
    if weight.numel() != num_classes:
        raise ValueError(f"class_weights length must equal num_classes={num_classes}, got {weight.numel()}")
    return weight
