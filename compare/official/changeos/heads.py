# Copyright (c) Zhuo Zheng and affiliates. All rights reserved.
# Modified: extracted ChangeOS-R50 fusion and deep heads; removed registry dependency.
# Licensed under Apache-2.0; see LICENSE.
from torch import nn
from .se_block import SEBlock

class FuseConv(nn.Sequential):
    def __init__(self, inchannels, outchannels):
        super(FuseConv, self).__init__(
            nn.Conv2d(inchannels, outchannels, kernel_size=1),
            nn.BatchNorm2d(outchannels),
        )
        self.relu = nn.ReLU(True)
        self.se = SEBlock(outchannels, 16)

    def forward(self, x):
        out = super(FuseConv, self).forward(x)
        residual = out
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class DeepHead(nn.Module):
    def __init__(self, in_channels, bottlneck_channels, num_blocks, num_classes, upsample_scale):
        super().__init__()
        assert num_blocks > 0
        self.relu = nn.ReLU(True)
        self.blocks = nn.ModuleList([nn.Sequential(
            # 1x1
            nn.Conv2d(in_channels, bottlneck_channels, 1),
            nn.BatchNorm2d(bottlneck_channels),
            nn.ReLU(True),
            # 3x3
            nn.Conv2d(bottlneck_channels, bottlneck_channels, 3, 1, 1),
            nn.BatchNorm2d(bottlneck_channels),
            # 1x1
            nn.Conv2d(bottlneck_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            SEBlock(in_channels, 16)
        ) for _ in range(num_blocks)])

        self.cls = nn.Conv2d(in_channels, num_classes, 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)

    def forward(self, x, upsample=True):
        indentity = x
        for m in self.blocks:
            x = m(x)
            x += indentity
            x = self.relu(x)
            indentity = x
        x = self.cls(x)
        if upsample:
            x = self.up(x)
        return x
