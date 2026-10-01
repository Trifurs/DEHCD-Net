# Modified: minimal EVER operations used by ChangeOS. See LICENSE-ever.
import torch
from torch import nn

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False,
                 bn=True,
                 relu=True,
                 init_fn=None):
        super(ConvBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, dilation, groups,
                      bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ReLU(True) if relu else nn.Identity()
        )
        if init_fn:
            self.apply(init_fn)

    @staticmethod
    def same_padding(kernel_size, dilation):
        return dilation * (kernel_size - 1) // 2

class Bf16compatible(nn.Module):
    def __init__(self, module):
        super().__init__()
        self._inner_module = module

    def forward(self, x):
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        x = self._inner_module(x)
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        return x

