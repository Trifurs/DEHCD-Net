from __future__ import annotations

from typing import Callable

from torch import nn

from .official_adapters import (
    DMINetOfficial,
    HAFFOfficial,
    HFAPANetOfficial,
    HRSICDOfficial,
    ICIFNetOfficial,
    WaveHFGOfficial,
)


_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "icif": ICIFNetOfficial,
    "icif_net": ICIFNetOfficial,
    "icifnet": ICIFNetOfficial,
    "dminet": DMINetOfficial,
    "dmi_net": DMINetOfficial,
    "hfa_panet": HFAPANetOfficial,
    "hfapanet": HFAPANetOfficial,
    "hfa-panet": HFAPANetOfficial,
    "wavehfg": WaveHFGOfficial,
    "wave_hfg": WaveHFGOfficial,
    "hrsicd": HRSICDOfficial,
    "haff": HAFFOfficial,
}


def available_compare_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def build_compare_model(
    name: str,
    optical_channels: int,
    sar_channels: int,
    num_classes: int,
    base_channels: int = 16,
    target_channels: int = 3,
    adapt_batchnorm: bool = True,
    deep_supervision: bool = True,
) -> nn.Module:
    key = normalize_compare_model_name(name)
    if key not in _MODEL_REGISTRY:
        choices = ", ".join(available_compare_models())
        raise ValueError(f"Unknown compare model '{name}'. Available: {choices}")
    return _MODEL_REGISTRY[key](
        optical_channels=optical_channels,
        sar_channels=sar_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        target_channels=target_channels,
        adapt_batchnorm=adapt_batchnorm,
        deep_supervision=deep_supervision,
    )


def is_compare_model(name: str | None) -> bool:
    return normalize_compare_model_name(name or "") in _MODEL_REGISTRY


def normalize_compare_model_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")
