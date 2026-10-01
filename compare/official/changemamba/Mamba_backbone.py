# Modified for DEHCD-Net: package imports, optional profilers, strict weight loading / current AMP.
import torch.nn as nn
from .checkpoints import format_checkpoint_load_report, load_encoder_pretrained_weights
from .vmamba import VSSM, LayerNorm2d


class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        load_info = load_encoder_pretrained_weights(self, ckpt)
        # Classification checkpoints omit only newly created output normalization.
        missing = [k for k in load_info["missing_keys"] if not k.startswith("outnorm")]
        if missing or load_info["mismatched_keys"] or not load_info["loaded_keys"]:
            raise ValueError(f"Incomplete VMamba backbone checkpoint: {load_info}")
        self.pretrained_load_report = load_info
        print(format_checkpoint_load_report(load_info, title="PRETRAIN Load"))

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x

        return outs
