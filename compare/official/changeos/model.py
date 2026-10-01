# Copyright (c) Zhuo Zheng and affiliates. All rights reserved.
# Modified for DEHCD-Net: standalone nn.Module port of TorChange cos_r50.
# Apache-2.0; see LICENSE and LICENSE-ever. Source commits: compare/UPSTREAM.md.
import torch
from torch import nn
from torchvision.models import resnet50

from .fpn import FPN, AssymetricDecoder, conv_bn_block
from .heads import FuseConv, DeepHead


class ChangeOS(nn.Module):
    """ChangeOS-R50: shared ResNet, two FPN decoders, SE fusion, two deep heads.

    Returns raw logits. Training losses and object voting are handled explicitly
    by the project, so evaluation never switches the model's forward contract.
    """

    def __init__(self, num_classes=5):
        super().__init__()
        self.encoder = resnet50(weights=None)
        self.encoder.fc = nn.Identity()
        channels = (256, 512, 1024, 2048)
        self.loc_neck = nn.Sequential(FPN(channels, 256, conv_bn_block), AssymetricDecoder(256, 256))
        self.dam_neck = nn.Sequential(FPN(channels, 256, conv_bn_block), AssymetricDecoder(256, 256))
        self.fuse_conv = FuseConv(512, 256)
        self.loc_cls = DeepHead(256, 128, 1, 1, 4.)
        self.dam_cls = DeepHead(256, 128, 1, num_classes, 4.)

    def features(self, x):
        e = self.encoder
        x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
        outputs = []
        for layer in (e.layer1, e.layer2, e.layer3, e.layer4):
            x = layer(x)
            outputs.append(x)
        return outputs

    def forward(self, optical, sar):
        # The official bitemporal_forward batches both dates through one encoder.
        features = self.features(torch.stack([optical, sar], dim=1).flatten(0, 1))
        pre = self.loc_neck([f[0::2] for f in features])
        post = self.dam_neck([f[1::2] for f in features])
        fused = self.fuse_conv(torch.cat([pre, post], dim=1))
        return self.loc_cls(pre), self.dam_cls(fused)
