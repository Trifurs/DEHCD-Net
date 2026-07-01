from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .backbones import ConvNormAct, make_context_block
from .encoders import build_encoder
from .fusion import BidirectionalCrossScaleFusion, DiffusionRefinementBlock, GlobalContextBridge, HeterogeneousFusionBlock


MODEL_DISPLAY_NAME = "DEHCD-Net"


class HOGFeatureExtractor(nn.Module):
    """Differentiable HOG-like orientation histograms for optical/SAR edge priors."""

    def __init__(self, bins: int = 6, cell_size: int = 8, eps: float = 1e-6):
        super().__init__()
        self.bins = int(bins)
        self.cell_size = int(cell_size)
        self.eps = eps
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x.to(dtype=gray.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(dtype=gray.dtype), padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + self.eps)
        angle = torch.remainder(torch.atan2(gy, gx), torch.pi)
        scaled = angle * (self.bins / torch.pi)
        lower = torch.floor(scaled).long().clamp(0, self.bins - 1)
        upper = torch.remainder(lower + 1, self.bins)
        upper_weight = (scaled - lower.float()).clamp(0.0, 1.0)
        lower_weight = 1.0 - upper_weight

        hist = x.new_zeros((x.shape[0], self.bins, x.shape[2], x.shape[3]))
        hist.scatter_add_(1, lower, magnitude * lower_weight)
        hist.scatter_add_(1, upper, magnitude * upper_weight)
        if self.cell_size > 1:
            hist = F.avg_pool2d(hist, kernel_size=self.cell_size, stride=1, padding=self.cell_size // 2)
            hist = hist[..., : x.shape[2], : x.shape[3]]
        hist = hist / (hist.mean(dim=(2, 3), keepdim=True) + self.eps)
        return hist.clamp(max=10.0)


class HOGFeatureModulator(nn.Module):
    """Use HOG as an edge-aware modulation signal instead of encoder input."""

    def __init__(
        self,
        bins: int,
        channels: int,
        norm_type: str = "group",
        norm_groups: int = 8,
    ):
        super().__init__()
        hidden = max(int(channels) // 4, 4)
        self.net = nn.Sequential(
            ConvNormAct(
                bins,
                hidden,
                kernel_size=1,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
            nn.Conv2d(hidden, int(channels) * 2, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor, hog: torch.Tensor) -> torch.Tensor:
        if hog.shape[-2:] != feature.shape[-2:]:
            hog = F.interpolate(hog, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        scale, bias = torch.chunk(self.net(hog), chunks=2, dim=1)
        return feature * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * torch.tanh(bias)


class ModalityCalibrationStem(nn.Module):
    """Lightweight per-modality calibration before the encoder.

    Dataset preprocessing already normalizes each modality, but the three
    datasets still expose very different channel semantics: RGB-like optical,
    CAU's extra optical channel, single-band SAR, and Haiti's two-pass SAR stack.
    This residual stem lets each branch learn small channel/texture corrections
    while starting as an identity mapping.
    """

    def __init__(
        self,
        channels: int,
        norm_type: str = "group",
        norm_groups: int = 8,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            ConvNormAct(
                channels,
                channels,
                kernel_size=1,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        calibrated = x * self.scale + self.bias
        return calibrated + self.residual_scale * self.refine(calibrated)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        norm_type: str = "batch",
        norm_groups: int = 8,
        block_type: str = "residual",
    ):
        super().__init__()
        self.x_proj = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=1,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )
        self.skip_proj = (
            nn.Identity()
            if skip_channels == out_channels
            else ConvNormAct(
                skip_channels,
                out_channels,
                kernel_size=1,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            )
        )
        self.refine = make_context_block(
            out_channels,
            block_type=block_type,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.x_proj(x) + self.skip_proj(skip))


class DEHCDNet(nn.Module):
    """DEHCD-Net: difference-enhanced heterogeneous change detection CD network.

    The network keeps separate modality encoders, maps each scale into a shared
    latent space, aligns SAR features, fuses difference/product cues, exchanges
    information across feature scales, then restores full-resolution logits.
    """

    def __init__(
        self,
        optical_channels: int,
        sar_channels: int,
        num_classes: int = 2,
        base_channels: int = 32,
        dropout: float = 0.1,
        backbone: str = "lightweight_convnext",
        pretrained_backbone: bool = False,
        norm_type: str = "batch",
        norm_groups: int = 8,
        block_type: str = "residual",
        use_hog: bool = False,
        hog_bins: int = 6,
        hog_cell_size: int = 8,
        hog_modulation_levels: int = 2,
        diffusion_steps: int = 0,
        align_fusion: bool = True,
        align_start_level: int = 1,
        align_max_flow: float = 2.0,
        adaptive_modality_weight: bool = False,
        input_adaptation: bool = False,
        input_residual_scale: float = 0.1,
        cross_scale_fusion: bool = True,
        global_context: bool = True,
        share_encoder_from_level: int = 2,
        deep_supervision: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_hog = bool(use_hog)
        self.deep_supervision = bool(deep_supervision)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.optical_input = (
            ModalityCalibrationStem(
                optical_channels,
                norm_type=norm_type,
                norm_groups=norm_groups,
                residual_scale=input_residual_scale,
            )
            if bool(input_adaptation)
            else nn.Identity()
        )
        self.sar_input = (
            ModalityCalibrationStem(
                sar_channels,
                norm_type=norm_type,
                norm_groups=norm_groups,
                residual_scale=input_residual_scale,
            )
            if bool(input_adaptation)
            else nn.Identity()
        )
        self.hog = HOGFeatureExtractor(bins=hog_bins, cell_size=hog_cell_size) if self.use_hog else None
        encoder_optical_channels = optical_channels
        encoder_sar_channels = sar_channels
        self.optical_encoder, optical_channels_list = build_encoder(
            backbone,
            encoder_optical_channels,
            base_channels=base_channels,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
            block_type=block_type,
            pretrained=pretrained_backbone,
        )
        self.sar_encoder, sar_channels_list = build_encoder(
            backbone,
            encoder_sar_channels,
            base_channels=base_channels,
            dropout=dropout,
            norm_type=norm_type,
            norm_groups=norm_groups,
            block_type=block_type,
            pretrained=False,
        )
        for encoder in (self.optical_encoder, self.sar_encoder):
            if hasattr(encoder, "set_gradient_checkpointing"):
                encoder.set_gradient_checkpointing(self.gradient_checkpointing)
        _share_encoder_levels(self.optical_encoder, self.sar_encoder, int(share_encoder_from_level))
        if optical_channels_list != sar_channels_list:
            raise ValueError(f"Encoder channel mismatch: {optical_channels_list} vs {sar_channels_list}")
        if _uses_native_encoder_channels(backbone):
            channels = list(optical_channels_list)
        else:
            channels = [
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
            ]
        self.optical_adapters = nn.ModuleList(
            [
                nn.Identity()
                if in_ch == out_ch
                else ConvNormAct(in_ch, out_ch, kernel_size=1, norm_type=norm_type, norm_groups=norm_groups)
                for in_ch, out_ch in zip(optical_channels_list, channels)
            ]
        )
        self.sar_adapters = nn.ModuleList(
            [
                nn.Identity()
                if in_ch == out_ch
                else ConvNormAct(in_ch, out_ch, kernel_size=1, norm_type=norm_type, norm_groups=norm_groups)
                for in_ch, out_ch in zip(sar_channels_list, channels)
            ]
        )
        hog_levels = min(max(int(hog_modulation_levels), 0), len(channels)) if self.use_hog else 0
        self.optical_hog_modulators = nn.ModuleList(
            [
                HOGFeatureModulator(hog_bins, channels[idx], norm_type=norm_type, norm_groups=norm_groups)
                for idx in range(hog_levels)
            ]
        )
        self.sar_hog_modulators = nn.ModuleList(
            [
                HOGFeatureModulator(hog_bins, channels[idx], norm_type=norm_type, norm_groups=norm_groups)
                for idx in range(hog_levels)
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [
                HeterogeneousFusionBlock(
                    channel,
                    dropout=dropout,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                    block_type=block_type,
                    align=bool(align_fusion) and idx >= int(align_start_level),
                    max_flow=align_max_flow,
                    adaptive_modality_weight=adaptive_modality_weight,
                )
                for idx, channel in enumerate(channels)
            ]
        )
        self.global_context = (
            GlobalContextBridge(channels, norm_type=norm_type, norm_groups=norm_groups)
            if bool(global_context)
            else nn.Identity()
        )
        self.cross_scale_fusion = (
            BidirectionalCrossScaleFusion(
                channels,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
                block_type=block_type,
            )
            if bool(cross_scale_fusion)
            else nn.Identity()
        )
        self.bottleneck = nn.Sequential(
            ConvNormAct(
                channels[3],
                channels[3],
                kernel_size=3,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
            make_context_block(
                channels[3],
                block_type=block_type,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
        )
        self.diffusion_refine = (
            DiffusionRefinementBlock(
                channels[3],
                steps=diffusion_steps,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            )
            if int(diffusion_steps) > 0
            else nn.Identity()
        )
        self.dec2 = DecoderBlock(channels[3], channels[2], channels[2], dropout=dropout, norm_type=norm_type, norm_groups=norm_groups, block_type=block_type)
        self.dec1 = DecoderBlock(channels[2], channels[1], channels[1], dropout=dropout, norm_type=norm_type, norm_groups=norm_groups, block_type=block_type)
        self.dec0 = DecoderBlock(channels[1], channels[0], channels[0], dropout=dropout, norm_type=norm_type, norm_groups=norm_groups, block_type=block_type)
        self.aux_heads = (
            nn.ModuleList(
                [
                    self._make_aux_head(channels[2], num_classes),
                    self._make_aux_head(channels[1], num_classes),
                    self._make_aux_head(channels[0], num_classes),
                ]
            )
            if self.deep_supervision
            else nn.ModuleList()
        )
        self.head = nn.Sequential(
            ConvNormAct(
                channels[0],
                channels[0],
                kernel_size=3,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
            nn.Conv2d(channels[0], num_classes, kernel_size=1),
        )

    @staticmethod
    def _make_aux_head(channels: int, num_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor | list[torch.Tensor]]:
        input_size = optical.shape[-2:]
        optical = self.optical_input(optical)
        sar = self.sar_input(sar)
        optical_hog = self.hog(optical) if self.hog is not None else None
        sar_hog = self.hog(sar) if self.hog is not None else None
        optical_feats = self.optical_encoder(optical)
        sar_feats = self.sar_encoder(sar)
        optical_feats = [adapter(feat) for adapter, feat in zip(self.optical_adapters, optical_feats)]
        sar_feats = [adapter(feat) for adapter, feat in zip(self.sar_adapters, sar_feats)]
        if optical_hog is not None and sar_hog is not None:
            optical_feats = [
                modulator(feature, optical_hog)
                for modulator, feature in zip(self.optical_hog_modulators, optical_feats)
            ] + optical_feats[len(self.optical_hog_modulators) :]
            sar_feats = [
                modulator(feature, sar_hog)
                for modulator, feature in zip(self.sar_hog_modulators, sar_feats)
            ] + sar_feats[len(self.sar_hog_modulators) :]
        fused = [
            self._maybe_checkpoint(fusion, opt_feat, sar_feat)
            for fusion, opt_feat, sar_feat in zip(self.fusion_blocks, optical_feats, sar_feats)
        ]
        fused = self.global_context(fused)
        fused = self.cross_scale_fusion(fused)
        x = self._maybe_checkpoint(self.bottleneck, fused[3])
        x = self._maybe_checkpoint(self.diffusion_refine, x)
        d2 = self._maybe_checkpoint(self.dec2, x, fused[2])
        d1 = self._maybe_checkpoint(self.dec1, d2, fused[1])
        d0 = self._maybe_checkpoint(self.dec0, d1, fused[0])
        logits = self.head(d0)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        if self.deep_supervision and self.training:
            aux_logits = [
                F.interpolate(head(feat), size=input_size, mode="bilinear", align_corners=False)
                for head, feat in zip(self.aux_heads, [d2, d1, d0])
            ]
            return {"logits": logits, "aux_logits": aux_logits}
        return logits

    def _maybe_checkpoint(self, module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and any(inp.requires_grad for inp in inputs):
            return checkpoint(module, *inputs)
        return module(*inputs)


def build_model(
    config: Dict[str, Any],
    optical_channels: Optional[int] = None,
    sar_channels: Optional[int] = None,
) -> nn.Module:
    model_cfg = config.get("model", {})
    if optical_channels is None:
        optical_channels = _coerce_channels(model_cfg.get("optical_channels"), "optical_channels")
    if sar_channels is None:
        sar_channels = _coerce_channels(model_cfg.get("sar_channels"), "sar_channels")
    compare_name = model_cfg.get("compare_model")
    model_name = str(model_cfg.get("name", "DEHCDNet"))
    if compare_name or _is_compare_model_name(model_name):
        from compare import build_compare_model

        return build_compare_model(
            name=str(compare_name or model_name),
            optical_channels=optical_channels,
            sar_channels=sar_channels,
            num_classes=int(model_cfg.get("num_classes", config.get("task", {}).get("num_classes", 2))),
            base_channels=int(model_cfg.get("base_channels", 16)),
            target_channels=int(model_cfg.get("compare_target_channels", model_cfg.get("target_channels", 3))),
            adapt_batchnorm=bool(model_cfg.get("compare_adapt_batchnorm", True)),
            deep_supervision=bool(model_cfg.get("compare_deep_supervision", True)),
        )
    return DEHCDNet(
        optical_channels=optical_channels,
        sar_channels=sar_channels,
        num_classes=int(model_cfg.get("num_classes", config.get("task", {}).get("num_classes", 2))),
        base_channels=int(model_cfg.get("base_channels", 32)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        norm_type=str(model_cfg.get("norm_type", "batch")),
        norm_groups=int(model_cfg.get("norm_groups", 8)),
        block_type=str(model_cfg.get("block_type", "residual")),
        use_hog=bool(model_cfg.get("use_hog", False)),
        hog_bins=int(model_cfg.get("hog_bins", 6)),
        hog_cell_size=int(model_cfg.get("hog_cell_size", 8)),
        hog_modulation_levels=int(model_cfg.get("hog_modulation_levels", model_cfg.get("hog_levels", 2))),
        diffusion_steps=int(model_cfg.get("diffusion_steps", 0)),
        backbone=str(model_cfg.get("backbone", "lightweight_convnext")),
        pretrained_backbone=bool(model_cfg.get("pretrained_backbone", False)),
        align_fusion=bool(model_cfg.get("align_fusion", True)),
        align_start_level=int(model_cfg.get("align_start_level", 1)),
        align_max_flow=float(model_cfg.get("align_max_flow", 2.0)),
        adaptive_modality_weight=bool(model_cfg.get("adaptive_modality_weight", False)),
        input_adaptation=bool(model_cfg.get("input_adaptation", False)),
        input_residual_scale=float(model_cfg.get("input_residual_scale", 0.1)),
        cross_scale_fusion=bool(model_cfg.get("cross_scale_fusion", True)),
        global_context=bool(model_cfg.get("global_context", True)),
        share_encoder_from_level=int(model_cfg.get("share_encoder_from_level", 2)),
        deep_supervision=bool(model_cfg.get("deep_supervision", False)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
    )


def _coerce_channels(value: Any, name: str) -> int:
    if value in (None, "", "auto"):
        raise ValueError(f"{name} must be inferred from dataset or set explicitly in XML.")
    return int(value)


def _is_compare_model_name(name: str) -> bool:
    try:
        from compare import is_compare_model
    except ImportError:
        return False
    return is_compare_model(name)


def _uses_native_encoder_channels(backbone: str) -> bool:
    name = str(backbone).lower().replace("-", "_")
    return name in {"dehcd_s", "dehcd_m", "dehcd_l"}


def _share_encoder_levels(optical_encoder: nn.Module, sar_encoder: nn.Module, start_level: int) -> None:
    if int(start_level) < 0:
        return
    if not all(hasattr(encoder, "stages") and hasattr(encoder, "downsamples") for encoder in (optical_encoder, sar_encoder)):
        return
    stages_optical = getattr(optical_encoder, "stages")
    stages_sar = getattr(sar_encoder, "stages")
    downs_optical = getattr(optical_encoder, "downsamples")
    downs_sar = getattr(sar_encoder, "downsamples")
    for idx in range(int(start_level), min(len(stages_optical), len(stages_sar))):
        stages_sar[idx] = stages_optical[idx]
        if idx < len(downs_optical) and idx < len(downs_sar):
            downs_sar[idx] = downs_optical[idx]
