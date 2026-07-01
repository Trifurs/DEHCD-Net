from __future__ import annotations

import torch
import torch.nn.functional as F


_PALETTE = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.90, 0.10, 0.10],
        [0.98, 0.62, 0.12],
        [0.98, 0.88, 0.18],
        [0.20, 0.45, 0.95],
        [0.15, 0.70, 0.35],
        [0.70, 0.25, 0.90],
        [0.10, 0.75, 0.75],
    ],
    dtype=torch.float32,
)


def tensor_to_preview(tensor: torch.Tensor, sar: bool = False) -> torch.Tensor:
    image = tensor.detach().float().cpu()
    if image.ndim == 4:
        image = image[0]
    if sar:
        image = image.mean(dim=0, keepdim=True)
        image = image.repeat(3, 1, 1)
    elif image.shape[0] >= 3:
        image = image[:3]
    else:
        image = image[:1].repeat(3, 1, 1)
    return robust_minmax(image)


def robust_minmax(image: torch.Tensor) -> torch.Tensor:
    flat = image.flatten()
    if flat.numel() == 0:
        return image.clamp(0, 1)
    low = torch.quantile(flat, 0.02)
    high = torch.quantile(flat, 0.98)
    return ((image - low) / (high - low).clamp(min=1e-6)).clamp(0, 1)


def colorize_mask(mask: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    mask = mask.detach().long().cpu()
    if mask.ndim == 3:
        mask = mask[0]
    palette = _PALETTE
    if num_classes > palette.shape[0]:
        extra = torch.rand(num_classes - palette.shape[0], 3, generator=torch.Generator().manual_seed(7))
        palette = torch.cat([palette, extra], dim=0)
    clean = mask.clamp(0, max(num_classes - 1, 0))
    color = palette[clean].permute(2, 0, 1)
    if ignore_index is not None:
        ignore = mask == int(ignore_index)
        color[:, ignore] = 0.55
    return color.clamp(0, 1)


def make_validation_grid(
    optical: torch.Tensor,
    sar: torch.Tensor,
    label: torch.Tensor,
    logits: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    max_items: int = 2,
) -> torch.Tensor:
    pred = torch.argmax(logits.detach().cpu(), dim=1)
    panels = []
    count = min(int(max_items), optical.shape[0])
    for idx in range(count):
        parts = [
            tensor_to_preview(optical[idx], sar=False),
            tensor_to_preview(sar[idx], sar=True),
            colorize_mask(label[idx], num_classes=num_classes, ignore_index=ignore_index),
            colorize_mask(pred[idx], num_classes=num_classes, ignore_index=ignore_index),
        ]
        panels.append(join_with_separator(parts, dim=2))
    return join_with_separator(panels, dim=1) if panels else torch.zeros(3, 8, 8)


def join_with_separator(images: list[torch.Tensor], dim: int, value: float = 1.0, width: int = 4) -> torch.Tensor:
    if not images:
        return torch.zeros(3, 8, 8)
    normalized = [resize_to_match(image, images[0].shape[-2:]) for image in images]
    separator_shape = list(normalized[0].shape)
    separator_shape[dim] = width
    separator = torch.full(separator_shape, value, dtype=normalized[0].dtype)
    pieces = []
    for idx, image in enumerate(normalized):
        if idx > 0:
            pieces.append(separator)
        pieces.append(image)
    return torch.cat(pieces, dim=dim)


def resize_to_match(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape[-2:]) == tuple(size):
        return image
    return F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
