from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ConvNormAct, build_norm_layer, make_context_block


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class DiffusionRefinementBlock(nn.Module):
    """Lightweight diffusion-inspired iterative denoising for fused features."""

    def __init__(
        self,
        channels: int,
        steps: int = 2,
        dropout: float = 0.0,
        norm_type: str = "group",
        norm_groups: int = 8,
    ):
        super().__init__()
        self.steps = max(int(steps), 0)
        self.denoiser = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            build_norm_layer(channels, norm_type=norm_type, norm_groups=norm_groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.step_scale = nn.Parameter(torch.full((self.steps,), 0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for step in range(self.steps):
            residual = torch.tanh(self.denoiser(x))
            x = x + self.step_scale[step] * residual
        return x


class HeterogeneousFusionBlock(nn.Module):
    """Efficient optical/SAR feature alignment with difference-gated fusion."""

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        norm_type: str = "batch",
        norm_groups: int = 8,
        block_type: str = "residual",
        align: bool = True,
        max_flow: float = 2.0,
        adaptive_modality_weight: bool = False,
    ):
        super().__init__()
        self.align = bool(align)
        self.max_flow = float(max_flow)
        self.adaptive_modality_weight = bool(adaptive_modality_weight)
        self.optical_proj = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )
        self.sar_proj = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )
        self.fuse = nn.Sequential(
            ConvNormAct(
                channels,
                channels,
                kernel_size=3,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
            make_context_block(
                channels,
                block_type=block_type,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
            ChannelGate(channels),
        )
        if self.align:
            self.flow_head = nn.Sequential(
                ConvNormAct(
                    channels,
                    channels,
                    kernel_size=3,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                ),
                nn.Conv2d(channels, 2, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.flow_head[-1].weight)
            nn.init.zeros_(self.flow_head[-1].bias)
        else:
            self.flow_head = None
        if self.adaptive_modality_weight:
            gate_hidden = max(channels // 4, 8)
            self.modality_gate = nn.Sequential(
                nn.Conv2d(channels * 3, gate_hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(gate_hidden, 2, kernel_size=1),
            )
            nn.init.zeros_(self.modality_gate[-1].weight)
            nn.init.zeros_(self.modality_gate[-1].bias)
        else:
            self.modality_gate = None
        self.change_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(channels, norm_type=norm_type, norm_groups=norm_groups),
            nn.Sigmoid(),
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        optical = self.optical_proj(optical)
        sar = self.sar_proj(sar)
        if self.flow_head is not None:
            flow = self.max_flow * torch.tanh(self.flow_head(optical + torch.abs(optical - sar)))
            sar = warp_with_flow(sar, flow)
        diff = torch.abs(optical - sar)
        if self.modality_gate is not None:
            modality_weight = torch.softmax(self.modality_gate(torch.cat([optical, sar, diff], dim=1)), dim=1)
            shared = modality_weight[:, :1] * optical + modality_weight[:, 1:] * sar
        else:
            shared = 0.5 * (optical + sar)
        interaction = shared + diff + 0.25 * (optical * torch.sigmoid(sar))
        fused = self.fuse(interaction)
        gate = self.change_gate(torch.cat([shared, diff], dim=1))
        return shared + gate * fused


# class HeterogeneousFusionBlock(nn.Module):
#     """Efficient optical/SAR feature alignment without difference-gated fusion."""

#     def __init__(
#         self,
#         channels: int,
#         dropout: float = 0.0,
#         norm_type: str = "batch",
#         norm_groups: int = 8,
#         block_type: str = "residual",
#         align: bool = True,
#         max_flow: float = 2.0,
#         adaptive_modality_weight: bool = False,
#     ):
#         super().__init__()
#         self.align = bool(align)
#         self.max_flow = float(max_flow)
#         self.adaptive_modality_weight = bool(adaptive_modality_weight)
#         self.optical_proj = ConvNormAct(
#             channels,
#             channels,
#             kernel_size=1,
#             dropout=dropout,
#             norm_type=norm_type,
#             norm_groups=norm_groups,
#         )
#         self.sar_proj = ConvNormAct(
#             channels,
#             channels,
#             kernel_size=1,
#             dropout=dropout,
#             norm_type=norm_type,
#             norm_groups=norm_groups,
#         )
#         self.fuse = nn.Sequential(
#             ConvNormAct(
#                 channels,
#                 channels,
#                 kernel_size=3,
#                 dropout=dropout,
#                 norm_type=norm_type,
#                 norm_groups=norm_groups,
#             ),
#             make_context_block(
#                 channels,
#                 block_type=block_type,
#                 dropout=dropout,
#                 norm_type=norm_type,
#                 norm_groups=norm_groups,
#             ),
#             ChannelGate(channels),
#         )
#         if self.align:
#             self.flow_head = nn.Sequential(
#                 ConvNormAct(
#                     channels,
#                     channels,
#                     kernel_size=3,
#                     norm_type=norm_type,
#                     norm_groups=norm_groups,
#                 ),
#                 nn.Conv2d(channels, 2, kernel_size=3, padding=1),
#             )
#             nn.init.zeros_(self.flow_head[-1].weight)
#             nn.init.zeros_(self.flow_head[-1].bias)
#         else:
#             self.flow_head = None
#         if self.adaptive_modality_weight:
#             gate_hidden = max(channels // 4, 8)
#             self.modality_gate = nn.Sequential(
#                 nn.Conv2d(channels * 3, gate_hidden, kernel_size=1),
#                 nn.SiLU(inplace=True),
#                 nn.Conv2d(gate_hidden, 2, kernel_size=1),
#             )
#             nn.init.zeros_(self.modality_gate[-1].weight)
#             nn.init.zeros_(self.modality_gate[-1].bias)
#         else:
#             self.modality_gate = None
#         self.change_gate = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
#             build_norm_layer(channels, norm_type=norm_type, norm_groups=norm_groups),
#             nn.Sigmoid(),
#         )

#     def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
#         optical = self.optical_proj(optical)
#         sar = self.sar_proj(sar)
#         if self.flow_head is not None:
#             flow = self.max_flow * torch.tanh(self.flow_head(optical + torch.abs(optical - sar)))
#             sar = warp_with_flow(sar, flow)

#         diff = torch.abs(optical - sar)
#         if self.modality_gate is not None:
#             modality_weight = torch.softmax(self.modality_gate(torch.cat([optical, sar, diff], dim=1)), dim=1)
#             shared = modality_weight[:, :1] * optical + modality_weight[:, 1:] * sar
#         else:
#             shared = 0.5 * (optical + sar)

#         interaction = shared + diff + 0.25 * (optical * torch.sigmoid(sar))
#         fused = self.fuse(interaction)

#         concat_fused = torch.cat([shared, fused], dim=1)
#         return self.change_gate[1](self.change_gate[0](concat_fused))


class CrossScaleMergeBlock(nn.Module):
    """Weighted adjacent-scale interaction without activation-heavy concatenation."""

    def __init__(
        self,
        channels: int,
        context_channels: int,
        dropout: float = 0.0,
        norm_type: str = "batch",
        norm_groups: int = 8,
        block_type: str = "residual",
    ):
        super().__init__()
        self.context_proj = ConvNormAct(
            context_channels,
            channels,
            kernel_size=1,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )
        self.weights = nn.Parameter(torch.ones(2))
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            build_norm_layer(channels, norm_type=norm_type, norm_groups=norm_groups),
            nn.Sigmoid(),
        )
        self.refine = make_context_block(
            channels,
            block_type=block_type,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = F.interpolate(context, size=x.shape[-2:], mode="bilinear", align_corners=False)
        context = self.context_proj(context)
        weights = torch.relu(self.weights)
        weights = weights / weights.sum().clamp(min=1e-6)
        merged = weights[0] * x + weights[1] * context
        gate = self.gate(torch.abs(x - context))
        return self.refine(x + gate * (merged - x))


class GlobalContextBridge(nn.Module):
    """Inject high-level global scene context into every fused feature scale."""

    def __init__(
        self,
        channels: list[int],
        reduction: int = 4,
        norm_type: str = "batch",
        norm_groups: int = 8,
    ):
        super().__init__()
        top_channels = channels[-1]
        hidden = max(top_channels // int(reduction), 8)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(top_channels, hidden, kernel_size=1, bias=False),
            build_norm_layer(hidden, norm_type="group", norm_groups=norm_groups),
            nn.SiLU(inplace=True),
        )
        self.gates = nn.ModuleList([nn.Conv2d(hidden, channel, kernel_size=1) for channel in channels])
        self.biases = nn.ModuleList([nn.Conv2d(hidden, channel, kernel_size=1) for channel in channels])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        context = self.context(features[-1])
        out = []
        for feature, gate_head, bias_head in zip(features, self.gates, self.biases):
            gate = torch.sigmoid(gate_head(context))
            bias = 0.1 * torch.tanh(bias_head(context))
            out.append(feature * (1.0 + gate) + bias)
        return out


class BidirectionalCrossScaleFusion(nn.Module):
    """BiFPN/SegFormer-inspired cross-scale fusion for already aligned features."""

    def __init__(
        self,
        channels: list[int],
        dropout: float = 0.0,
        norm_type: str = "batch",
        norm_groups: int = 8,
        block_type: str = "residual",
    ):
        super().__init__()
        self.top_down = nn.ModuleList(
            [
                CrossScaleMergeBlock(
                    channels[idx],
                    channels[idx + 1],
                    dropout=dropout,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                    block_type=block_type,
                )
                for idx in range(len(channels) - 1)
            ]
        )
        self.bottom_up = nn.ModuleList(
            [
                CrossScaleMergeBlock(
                    channels[idx + 1],
                    channels[idx],
                    dropout=dropout,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                    block_type=block_type,
                )
                for idx in range(len(channels) - 1)
            ]
        )
        self.outputs = nn.ModuleList(
            [
                make_context_block(
                    channel,
                    block_type=block_type,
                    dropout=dropout,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                )
                for channel in channels
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        top_features = list(features)
        for idx in reversed(range(len(features) - 1)):
            top_features[idx] = self.top_down[idx](top_features[idx], top_features[idx + 1])

        out_features = list(top_features)
        for idx in range(1, len(features)):
            out_features[idx] = self.bottom_up[idx - 1](top_features[idx], out_features[idx - 1])
        return [refine(feature) for refine, feature in zip(self.outputs, out_features)]


def warp_with_flow(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = x.shape
    y, x_coord = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([x_coord, y], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    norm = torch.tensor([max(width - 1, 1), max(height - 1, 1)], device=x.device, dtype=x.dtype).view(1, 2, 1, 1)
    flow_norm = (2.0 * flow / norm).permute(0, 2, 3, 1)
    return F.grid_sample(x, base_grid + flow_norm, mode="bilinear", padding_mode="border", align_corners=True)
