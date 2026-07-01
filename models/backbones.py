from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def build_norm_layer(channels: int, norm_type: str = "batch", norm_groups: int = 8) -> nn.Module:
    norm_type = str(norm_type).lower()
    if norm_type in {"batch", "batchnorm", "bn"}:
        return nn.BatchNorm2d(channels)
    if norm_type in {"group", "groupnorm", "gn"}:
        groups = min(int(norm_groups), channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm_type in {"instance", "instancenorm", "in"}:
        return nn.InstanceNorm2d(channels, affine=True)
    if norm_type in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dropout: float = 0.0,
        norm_type: str = "batch",
        norm_groups: int = 8,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            build_norm_layer(out_channels, norm_type=norm_type, norm_groups=norm_groups),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
        norm_type: str = "batch",
        norm_groups: int = 8,
    ):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
            ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=1,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        norm_type: str = "batch",
        norm_groups: int = 8,
    ):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(
            channels,
            channels,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )
        self.conv2 = DepthwiseSeparableConv(
            channels,
            channels,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class GRN(nn.Module):
    """Global response normalization from ConvNeXt V2."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style large-kernel block with GRN, adapted for dense prediction."""

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        dropout: float = 0.0,
        norm_type: str = "group",
        norm_groups: int = 8,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            build_norm_layer(channels, norm_type=norm_type, norm_groups=norm_groups),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            GRN(hidden),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer_scale * self.block(x)


class DEHCDBlock(nn.Module):
    """Multi-scale ConvNeXt-V2 style block for high-resolution heterogeneous CD.

    It keeps the efficient depthwise-convolution path of modern ConvNets, adds
    dilated branches for larger receptive fields, and uses GRN to improve
    channel competition without relying on custom CUDA ops.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        dropout: float = 0.0,
        norm_type: str = "group",
        norm_groups: int = 8,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        hidden = channels * expansion
        self.dw_local = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.dw_mid = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
        )
        self.dw_large = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=3,
            dilation=3,
            groups=channels,
            bias=False,
        )
        self.branch_weight = nn.Parameter(torch.ones(3))
        self.norm = build_norm_layer(channels, norm_type=norm_type, norm_groups=norm_groups)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            GRN(hidden),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        gate_hidden = max(channels // 4, 8)
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, gate_hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.softmax(self.branch_weight, dim=0)
        mixed = weight[0] * self.dw_local(x) + weight[1] * self.dw_mid(x) + weight[2] * self.dw_large(x)
        mixed = self.mlp(self.norm(mixed))
        mixed = mixed * self.context_gate(x)
        return x + self.layer_scale * mixed


def make_context_block(
    channels: int,
    block_type: str = "residual",
    dropout: float = 0.0,
    norm_type: str = "batch",
    norm_groups: int = 8,
) -> nn.Module:
    block_type = str(block_type).lower()
    if block_type in {"residual", "res", "depthwise"}:
        return ResidualBlock(channels, dropout=dropout, norm_type=norm_type, norm_groups=norm_groups)
    if block_type in {"convnext", "convnextv2", "modern"}:
        return ConvNeXtBlock(channels, dropout=dropout, norm_type=norm_type, norm_groups=norm_groups)
    if block_type in {"dehcd", "multiscale", "scale_aware"}:
        return DEHCDBlock(channels, dropout=dropout, norm_type=norm_type, norm_groups=norm_groups)
    if block_type in {"hybrid", "res_convnext"}:
        return nn.Sequential(
            ResidualBlock(channels, dropout=dropout, norm_type=norm_type, norm_groups=norm_groups),
            ConvNeXtBlock(channels, dropout=dropout, norm_type=norm_type, norm_groups=norm_groups),
        )
    raise ValueError(f"Unsupported block_type: {block_type}")


class ModalityEncoder(nn.Module):
    """Lightweight modality-specific encoder for optical or SAR input."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        dropout: float = 0.0,
        norm_type: str = "batch",
        norm_groups: int = 8,
        block_type: str = "residual",
    ):
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        self.stem = nn.Sequential(
            ConvNormAct(
                in_channels,
                channels[0],
                kernel_size=3,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
            make_context_block(
                channels[0],
                block_type=block_type,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(
                channels[0],
                channels[1],
                stride=2,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
            make_context_block(channels[1], block_type=block_type, dropout=dropout, norm_type=norm_type, norm_groups=norm_groups),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(
                channels[1],
                channels[2],
                stride=2,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
            make_context_block(channels[2], block_type=block_type, dropout=dropout, norm_type=norm_type, norm_groups=norm_groups),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(
                channels[2],
                channels[3],
                stride=2,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
            make_context_block(channels[3], block_type=block_type, dropout=dropout, norm_type=norm_type, norm_groups=norm_groups),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return [x0, x1, x2, x3]


DEHCD_VARIANTS = {
    "s": {"base": 16, "depths": (1, 1, 2, 1), "expansion": 2, "multipliers": (1, 2, 4, 6)},
    "m": {"base": 24, "depths": (1, 2, 3, 2), "expansion": 2, "multipliers": (1, 2, 4, 6)},
    "l": {"base": 32, "depths": (2, 2, 4, 2), "expansion": 3, "multipliers": (1, 2, 4, 6)},
}


def resolve_dehcd_variant(backbone: str) -> str:
    name = str(backbone).lower().replace("-", "_")
    if name in {"dehcd_s", "dehcd_m", "dehcd_l"}:
        return name.rsplit("_", 1)[-1]
    if name in DEHCD_VARIANTS:
        return name
    raise ValueError(f"Unsupported DEHCD backbone variant: {backbone}. Use dehcd_s, dehcd_m, or dehcd_l.")


class DEHCDBackboneEncoder(nn.Module):
    """Difference-Enhanced Heterogeneous Change Detection backbone family.

    Variants ``dehcd_s``, ``dehcd_m`` and ``dehcd_l`` scale channel width and depth
    while keeping a four-level feature pyramid compatible with the project
    decoder and comparison pipeline.
    """

    def __init__(
        self,
        in_channels: int,
        variant: str = "s",
        base_channels: int = 0,
        dropout: float = 0.0,
        norm_type: str = "group",
        norm_groups: int = 8,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        if variant not in DEHCD_VARIANTS:
            raise ValueError(f"Unsupported DEHCD variant: {variant}")
        spec = DEHCD_VARIANTS[variant]
        base = max(int(base_channels or 0), int(spec["base"]))
        multipliers = tuple(int(item) for item in spec.get("multipliers", (1, 2, 4, 6)))
        self.out_channels = [base * item for item in multipliers]
        depths = tuple(int(item) for item in spec["depths"])
        expansion = int(spec["expansion"])

        self.stem = ConvNormAct(
            in_channels,
            self.out_channels[0],
            kernel_size=3,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )
        downsamples = [nn.Identity()]
        for in_ch, out_ch in zip(self.out_channels[:-1], self.out_channels[1:]):
            downsamples.append(
                ConvNormAct(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    dropout=dropout,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                )
            )
        self.downsamples = nn.ModuleList(downsamples)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        DEHCDBlock(
                            channels,
                            expansion=expansion,
                            dropout=dropout,
                            norm_type=norm_type,
                            norm_groups=norm_groups,
                        )
                        for _ in range(depth)
                    ]
                )
                for channels, depth in zip(self.out_channels, depths)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features: list[torch.Tensor] = []
        for downsample, stage in zip(self.downsamples, self.stages):
            x = downsample(x)
            x = self._maybe_checkpoint(stage, x)
            features.append(x)
        return features

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _maybe_checkpoint(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and x.requires_grad:
            return checkpoint(module, x)
        return module(x)
