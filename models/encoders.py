from __future__ import annotations

from typing import Tuple

import torch.nn as nn

from .backbones import DEHCDBackboneEncoder, ModalityEncoder, resolve_dehcd_variant


class EncoderSpec(nn.Module):
    out_channels: list[int]


def build_encoder(
    backbone: str,
    in_channels: int,
    base_channels: int,
    dropout: float,
    norm_type: str,
    norm_groups: int,
    block_type: str,
    pretrained: bool = False,
) -> Tuple[nn.Module, list[int]]:
    name = str(backbone).lower()
    if name in {"lightweight", "lightweight_convnext", "custom", "hetero"}:
        encoder = ModalityEncoder(
            in_channels,
            base_channels=base_channels,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
            block_type=block_type,
        )
        return encoder, [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
    if name in {
        "dehcd_s",
        "dehcd_m",
        "dehcd_l",
    }:
        encoder = DEHCDBackboneEncoder(
            in_channels=in_channels,
            variant=resolve_dehcd_variant(name),
            base_channels=base_channels,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )
        return encoder, encoder.out_channels
    raise ValueError(f"Unsupported backbone: {backbone}. Use dehcd_s, dehcd_m, dehcd_l, or lightweight.")
