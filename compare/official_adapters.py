from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityInputAdapter(nn.Module):
    """Adapt project optical/SAR tensors to the 3-channel inputs used upstream.

    The official comparison models were mostly written for RGB/RGB image pairs.
    In this project the second branch can be one SAR band, two SAR passes, or a
    four-channel Haiti SAR stack. A small learnable stem gives each modality a
    chance to form a stable pseudo-RGB representation before entering the
    unmodified official core.
    """

    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if self.in_channels == self.out_channels:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
            self._init_projection(self.proj)
        groups = _best_group_count(self.out_channels)
        self.refine = nn.Sequential(
            nn.GroupNorm(groups, self.out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                groups=self.out_channels,
                bias=False,
            ),
            nn.GroupNorm(groups, self.out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        projected = self.proj(x)
        return projected + 0.1 * self.refine(projected)

    def _init_projection(self, proj: nn.Conv2d) -> None:
        if self.in_channels == 1:
            nn.init.ones_(proj.weight)
            return
        with torch.no_grad():
            proj.weight.zero_()
            for out_idx in range(self.out_channels):
                start = int(out_idx * self.in_channels / self.out_channels)
                end = int((out_idx + 1) * self.in_channels / self.out_channels)
                if end <= start:
                    start = min(out_idx, self.in_channels - 1)
                    end = start + 1
                end = min(end, self.in_channels)
                proj.weight[out_idx, start:end, 0, 0] = 1.0 / max(end - start, 1)


class OfficialCompareAdapter(nn.Module):
    """Project-level adapter around official comparison-model source code."""

    def __init__(
        self,
        optical_channels: int,
        sar_channels: int,
        num_classes: int,
        output_channels: int,
        target_channels: int = 3,
        resize_to: int | None = None,
        output_is_probability: bool = False,
        adapt_batchnorm: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.output_channels = int(output_channels)
        self.resize_to = int(resize_to) if resize_to else None
        self.output_is_probability = bool(output_is_probability)
        self.adapt_batchnorm = bool(adapt_batchnorm)
        self.deep_supervision = bool(deep_supervision)
        self.optical_adapter = ModalityInputAdapter(optical_channels, target_channels)
        self.sar_adapter = ModalityInputAdapter(sar_channels, target_channels)
        self.output_adapter = (
            nn.Identity()
            if self.output_channels == self.num_classes or (self.output_channels == 1 and self.num_classes == 2)
            else nn.Conv2d(self.output_channels, self.num_classes, kernel_size=1)
        )

    def _prepare_inputs(self, optical: torch.Tensor, sar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        output_size = optical.shape[-2:]
        optical = self.optical_adapter(optical)
        sar = self.sar_adapter(sar)
        optical = optical.float()
        sar = sar.float()
        if self.resize_to:
            target_size = (self.resize_to, self.resize_to)
            optical = F.interpolate(optical, size=target_size, mode="bilinear", align_corners=False)
            sar = F.interpolate(sar, size=target_size, mode="bilinear", align_corners=False)
        return optical, sar, output_size

    def _run_official_model(self, optical: torch.Tensor, sar: torch.Tensor) -> Any:
        # Most paper-code comparison models predate modern AMP and contain
        # float32-only ops such as wavelet filters, dynamic convolution kernels,
        # and positional encodings. Keep the official core in FP32 while the
        # project-level training loop can still use AMP around loss/logit work.
        amp_context = torch.amp.autocast("cuda", enabled=False) if optical.is_cuda else nullcontext()
        with amp_context:
            return self.model(optical.float(), sar.float())

    def _finish_output(self, output: Any, output_size: tuple[int, int]) -> torch.Tensor | dict[str, Any]:
        logits = _first_tensor(output)
        logits = self._finish_logits(logits, output_size)
        if not (self.training and self.deep_supervision):
            return logits

        primary_id = id(_first_tensor(output))
        aux_logits = []
        for aux in _prediction_tensors(output, self._accepted_output_channels()):
            if id(aux) == primary_id:
                continue
            aux_logits.append(self._finish_logits(aux, output_size))
        if not aux_logits:
            return logits
        return {"logits": logits, "aux_logits": aux_logits}

    def _finish_logits(self, logits: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected BCHW logits from comparison model, got shape={tuple(logits.shape)}")
        if self.output_is_probability:
            logits = torch.logit(logits.clamp(min=1e-4, max=1.0 - 1e-4))
        if logits.shape[1] == 1 and self.num_classes == 2:
            logits = torch.cat([-logits, logits], dim=1)
        elif logits.shape[1] == self.num_classes:
            pass
        elif logits.shape[1] == self.output_channels:
            logits = self.output_adapter(logits)
        else:
            raise ValueError(
                "Comparison model produced an unexpected prediction channel count: "
                f"{logits.shape[1]} not in {sorted(self._accepted_output_channels())}."
            )
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits

    def _accepted_output_channels(self) -> set[int]:
        accepted = {self.num_classes, self.output_channels}
        if self.num_classes == 2:
            accepted.add(1)
        return accepted

    def _finalize_official_model(self) -> None:
        if self.adapt_batchnorm:
            _replace_batchnorm2d(self.model)


class ICIFNetOfficial(OfficialCompareAdapter):
    def __init__(
        self,
        optical_channels: int,
        sar_channels: int,
        num_classes: int,
        base_channels: int = 16,
        target_channels: int = 3,
        adapt_batchnorm: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__(
            optical_channels,
            sar_channels,
            num_classes,
            output_channels=num_classes,
            target_channels=target_channels,
            adapt_batchnorm=adapt_batchnorm,
            deep_supervision=deep_supervision,
        )
        try:
            from .official.icif_net.ICIFNet import ICIFNet
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error("ICIF-Net", exc) from exc

        self.model = ICIFNet(num_classes=num_classes, pretrained=False)
        self._finalize_official_model()

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor | dict[str, Any]:
        optical, sar, output_size = self._prepare_inputs(optical, sar)
        return self._finish_output(self._run_official_model(optical, sar), output_size)


class DMINetOfficial(OfficialCompareAdapter):
    def __init__(
        self,
        optical_channels: int,
        sar_channels: int,
        num_classes: int,
        base_channels: int = 16,
        target_channels: int = 3,
        adapt_batchnorm: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__(
            optical_channels,
            sar_channels,
            num_classes,
            output_channels=num_classes,
            target_channels=target_channels,
            adapt_batchnorm=adapt_batchnorm,
            deep_supervision=deep_supervision,
        )
        try:
            from .official.dminet.DMINet import DMINet
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error("DMINet", exc) from exc

        self.model = DMINet(num_classes=num_classes, pretrained=False, show_Feature_Maps=False)
        self._finalize_official_model()

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor | dict[str, Any]:
        optical, sar, output_size = self._prepare_inputs(optical, sar)
        return self._finish_output(self._run_official_model(optical, sar), output_size)


class HFAPANetOfficial(OfficialCompareAdapter):
    def __init__(
        self,
        optical_channels: int,
        sar_channels: int,
        num_classes: int,
        base_channels: int = 16,
        target_channels: int = 3,
        adapt_batchnorm: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__(
            optical_channels,
            sar_channels,
            num_classes,
            output_channels=num_classes,
            target_channels=target_channels,
            adapt_batchnorm=adapt_batchnorm,
            deep_supervision=deep_supervision,
        )
        try:
            from .official.hfa_panet.MASNet2 import FPANet_NoSaim
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error("HFA-PANet", exc) from exc

        self.model = FPANet_NoSaim(pretrain=False, num_classes=num_classes)
        self._finalize_official_model()

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor | dict[str, Any]:
        optical, sar, output_size = self._prepare_inputs(optical, sar)
        output = self._run_official_model(optical, sar)
        finished = self._finish_output(output, output_size)
        if self.training and isinstance(output, (tuple, list)) and len(output) >= 3:
            feature_pairs = _feature_pairs(output[2])
            if feature_pairs:
                if isinstance(finished, dict):
                    finished["feature_pairs"] = feature_pairs
                else:
                    finished = {"logits": finished, "feature_pairs": feature_pairs}
        return finished


class WaveHFGOfficial(OfficialCompareAdapter):
    def __init__(
        self,
        optical_channels: int,
        sar_channels: int,
        num_classes: int,
        base_channels: int = 16,
        target_channels: int = 3,
        adapt_batchnorm: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__(
            optical_channels,
            sar_channels,
            num_classes,
            output_channels=num_classes,
            target_channels=target_channels,
            adapt_batchnorm=adapt_batchnorm,
            deep_supervision=deep_supervision,
        )
        try:
            from .official.wavehfg.WHFCE import WHFCE
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error("WaveHFG", exc) from exc

        self.model = WHFCE(num_classes=num_classes)
        self._finalize_official_model()

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor | dict[str, Any]:
        optical, sar, output_size = self._prepare_inputs(optical, sar)
        return self._finish_output(self._run_official_model(optical, sar), output_size)


class HRSICDOfficial(OfficialCompareAdapter):
    def __init__(
        self,
        optical_channels: int,
        sar_channels: int,
        num_classes: int,
        base_channels: int = 16,
        target_channels: int = 3,
        adapt_batchnorm: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__(
            optical_channels,
            sar_channels,
            num_classes,
            output_channels=num_classes if num_classes > 2 else 1,
            target_channels=target_channels,
            resize_to=64,
            output_is_probability=False,
            adapt_batchnorm=adapt_batchnorm,
            deep_supervision=deep_supervision,
        )
        try:
            from .official.hrsicd.HRSICD import HRSICD
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error("HRSICD", exc) from exc

        n_classes = num_classes if num_classes > 2 else 1
        self.model = HRSICD(n_channels=3, n_classes=n_classes, img_size=64, apply_sigmoid=False)
        self._finalize_official_model()

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor | dict[str, Any]:
        optical, sar, output_size = self._prepare_inputs(optical, sar)
        return self._finish_output(self._run_official_model(optical, sar), output_size)


class HAFFOfficial(OfficialCompareAdapter):
    def __init__(
        self,
        optical_channels: int,
        sar_channels: int,
        num_classes: int,
        base_channels: int = 16,
        target_channels: int = 3,
        adapt_batchnorm: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__(
            optical_channels,
            sar_channels,
            num_classes,
            output_channels=num_classes,
            target_channels=target_channels,
            adapt_batchnorm=adapt_batchnorm,
            deep_supervision=deep_supervision,
        )
        try:
            from .official.haff.modelMCD import ADVNets
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error("HAFF", exc) from exc

        self.model = ADVNets(input_nbr=3, label_nbr=num_classes)
        self._finalize_official_model()

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor | dict[str, Any]:
        optical, sar, output_size = self._prepare_inputs(optical, sar)
        return self._finish_output(self._run_official_model(optical, sar), output_size)


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "pred", "output"):
            if key in output:
                return _first_tensor(output[key])
        return _first_tensor(next(iter(output.values())))
    if isinstance(output, (tuple, list)):
        return _first_tensor(output[0])
    raise TypeError(f"Unsupported compare model output type: {type(output)!r}")


def _prediction_tensors(output: Any, accepted_channels: set[int]) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    _collect_prediction_tensors(output, accepted_channels, tensors)
    return tensors


def _collect_prediction_tensors(output: Any, accepted_channels: set[int], tensors: list[torch.Tensor]) -> None:
    if isinstance(output, torch.Tensor):
        if output.ndim == 4 and int(output.shape[1]) in accepted_channels:
            tensors.append(output)
        return
    if isinstance(output, dict):
        for value in output.values():
            _collect_prediction_tensors(value, accepted_channels, tensors)
        return
    if isinstance(output, (tuple, list)):
        for value in output:
            _collect_prediction_tensors(value, accepted_channels, tensors)


def _feature_pairs(output: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if not isinstance(output, (tuple, list)):
        return pairs
    for item in output:
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[0], torch.Tensor)
            and isinstance(item[1], torch.Tensor)
        ):
            pairs.append((item[0], item[1]))
    return pairs


def _replace_batchnorm2d(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            replacement = nn.GroupNorm(_best_group_count(child.num_features), child.num_features)
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            _replace_batchnorm2d(child)


def _best_group_count(channels: int, max_groups: int = 8) -> int:
    channels = int(channels)
    for groups in range(min(int(max_groups), channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _missing_dependency_error(model_name: str, exc: ModuleNotFoundError) -> ModuleNotFoundError:
    package = getattr(exc, "name", None) or str(exc)
    return ModuleNotFoundError(
        f"{model_name} official code requires optional dependency '{package}'. "
        "Install the updated requirements.txt before running this comparison model."
    )
