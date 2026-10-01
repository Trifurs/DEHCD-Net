# Modified: extracted EVER SEBlock; unrelated modules removed. See LICENSE-ever.
import torch
from torch import nn

GlobalAvgPool2D = lambda: nn.AdaptiveAvgPool2d(1)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction):
        super(SEBlock, self).__init__()
        self.gap = GlobalAvgPool2D()
        self.seq = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        v = self.gap(x)
        score = self.seq(v.view(v.size(0), v.size(1)))
        y = x * score.view(score.size(0), score.size(1), 1, 1)
        return y

