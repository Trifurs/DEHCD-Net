"""Dual-head building-damage baselines with explicit project adaptations."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

from .official_adapters import ModalityInputAdapter, _replace_batchnorm2d
from utils.protocol import file_digest


def _load_torchvision_encoder(encoder, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    # Only the discarded ImageNet classifier/final unused Swin norm may differ.
    state = {k: v for k, v in state.items() if not k.startswith(("fc.", "head.", "norm."))}
    encoder.load_state_dict(state, strict=True)


class DamageBaseline(nn.Module):
    """Returns damage and localization logits; the protocol selects supervision.

    The formal comparison supervises only the primary damage/change logits.
    The localization branch is retained for structure and checkpoint compatibility;
    its unsupervised outputs are diagnostic only. This is a common-protocol
    architecture adaptation, not a reproduction of official benchmark recipes.
    """
    def __init__(self, name, optical_channels, sar_channels, num_classes,
                 adapt_batchnorm=True, model_config=None, **unused):
        super().__init__()
        cfg = model_config or {}
        self.name = name
        self.core_precision_policy = "fp32"
        self.num_classes = int(num_classes)
        self.optical_adapter = ModalityInputAdapter(optical_channels, 3)
        self.sar_adapter = ModalityInputAdapter(sar_channels, 3)
        pretrained = cfg.get("encoder_checkpoint") if cfg.get("_initialize_encoder", True) else None
        if pretrained in (None, "", "none"):
            pretrained = None
        else:
            pretrained = str(Path(pretrained).expanduser().resolve(strict=True))
            expected = cfg.get("encoder_checkpoint_sha256")
            if expected and file_digest(pretrained) != expected:
                raise ValueError("Encoder checkpoint content changed after the experiment was configured")
        variant = cfg.get("changemamba_variant", "multimodal")
        if name == "changeos":
            from .official.changeos import ChangeOS
            self.model = ChangeOS(num_classes)
            if pretrained:
                _load_torchvision_encoder(self.model.encoder, pretrained)
        elif name == "damageformer":
            from .official.damageformer import DamageFormer
            self.model = DamageFormer(num_classes, share_encoder=bool(cfg.get("share_damage_encoder", False)))
            if pretrained:
                for encoder in {self.model.encoder_1, self.model.encoder_2}:
                    _load_torchvision_encoder(encoder.swin_transformer, pretrained)
        elif name == "changemamba":
            from .official.changemamba.scan_backend import configure_scan_backend
            if variant == "multimodal":
                from .official.changemamba.ChangeMambaMMBDA import ChangeMambaMMBDA as Core
            elif variant == "bda":
                from .official.changemamba.ChangeMambaBDA import ChangeMambaBDA as Core
            else:
                raise ValueError("changemamba_variant must be multimodal (BRIGHT) or bda (xBD)")
            self.model = Core(output_building=2, output_damage=num_classes, pretrained=pretrained,
                patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 4, 2], dims=96,
                ssm_d_state=1, ssm_ratio=2., ssm_rank_ratio=2., ssm_dt_rank="auto", ssm_act_layer="silu",
                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0., ssm_init="v0", forward_type="v3noz",
                mlp_ratio=4., mlp_act_layer="gelu", mlp_drop_rate=0., drop_path_rate=.2,
                patch_norm=True, norm_layer="ln", downsample_version="v3", patchembed_version="v2",
                gmlp=False, use_checkpoint=bool(cfg.get("gradient_checkpointing", False)))
            configure_scan_backend(self.model, str(cfg.get("selective_scan_backend", "auto")))
        else:
            raise ValueError(f"Unknown damage baseline: {name}")
        # Load pretrained BN statistics before an explicitly requested GN conversion.
        if adapt_batchnorm:
            _replace_batchnorm2d(self.model)
        self.implementation_info = {"model": name, "classes": num_classes,
            "variant": variant if name == "changemamba" else "resnet50_deep_heads" if name == "changeos" else "swin_t",
            "input_adapter": "learned_3channel", "batchnorm_replaced": bool(adapt_batchnorm),
            "encoder_checkpoint": pretrained, "encoder_checkpoint_sha256": file_digest(pretrained) if pretrained else None,
            "initialization": "encoder_pretrained" if pretrained else "scratch",
            "primary_prediction": "damage_argmax", "localization_target": "label_gt_zero_excluding_ignore"}
        if not cfg.get("_initialize_encoder", True):
            self.implementation_info.update(initialization="full_checkpoint_expected",
                encoder_checkpoint=cfg.get("encoder_checkpoint"),
                encoder_checkpoint_sha256=cfg.get("encoder_checkpoint_sha256"))

    def forward(self, optical, sar):
        if optical.shape[-2:] != sar.shape[-2:]:
            raise ValueError("Paired inputs must be spatially aligned and have equal size")
        h, w = optical.shape[-2:]
        # Official decoders assume pyramid dimensions divisible by 32.
        pad = (0, (-w) % 32, 0, (-h) % 32)
        optical = F.pad(self.optical_adapter(optical.float()), pad)
        sar = F.pad(self.sar_adapter(sar.float()), pad)
        with torch.autocast(device_type=optical.device.type, enabled=False):
            loc, damage = self.model(optical.float(), sar.float())
        loc, damage = loc[..., :h, :w], damage[..., :h, :w]
        if loc.shape[1] == 1:
            loc = torch.cat([torch.zeros_like(loc), loc], dim=1)
        return {"logits": damage, "localization_logits": loc}


class ChangeOSOfficial(DamageBaseline):
    def __init__(self, **kwargs):
        super().__init__("changeos", **kwargs)


class DamageFormerOfficial(DamageBaseline):
    def __init__(self, **kwargs):
        super().__init__("damageformer", **kwargs)


class ChangeMambaOfficial(DamageBaseline):
    def __init__(self, **kwargs):
        super().__init__("changemamba", **kwargs)
