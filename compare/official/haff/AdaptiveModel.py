import torch.nn as nn
import torch
import torch.nn.functional as F
"""
    dynamic Convolutional Module used in DMNet.
    Args:
        filter_size (int): The filter size of generated convolution kernel used in Dynamic Convolutional Module.
        fusion (bool): Add one conv to fuse DCM output feature.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
    """
class DCM(nn.Module):
    def __init__(self,filter_size, fusion, in_channels, channels):
        super().__init__()
        self.filter_size = filter_size
        self.fusion = fusion
        self.channels = channels
        self.filter_gen_conv = nn.Conv2d(in_channels, channels, 1)
        self.input_redu_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.norm =  nn.BatchNorm2d(channels)

        self.act = nn.ReLU(inplace=True)
        if self.fusion:
            self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):

        generated_filter = self.filter_gen_conv(F.adaptive_avg_pool2d(x, self.filter_size))
        x = self.input_redu_conv(x)
        b, c, h, w = x.shape
        x = x.reshape([1, b * c, h, w])
        generated_filter = generated_filter.reshape([b * c, 1, self.filter_size, self.filter_size])
        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            pad = (pad, pad, pad, pad)
        else:
            pad = (pad + 1, pad, pad + 1, pad)
        x = F.pad(x, pad, mode='constant', value=0)  # [1, b * c, h, w]
        output = F.conv2d(x, weight=generated_filter, groups=b * c)
        output = output.reshape([b, self.channels, h, w])
        output = self.norm(output)
        output = self.act(output)
        if self.fusion:
            output = self.fusion_conv(output)
        print(output.shape)
        return output

class attention2d(nn.Module):
    def __init__(self, in_channel, out_channel,):
        super(attention2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channel, out_channel, 1,)
        self.fc2 = nn.Conv2d(out_channel, out_channel, 1,)
    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x, 1)
class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, K, )

        self.weight = nn.Parameter(torch.Tensor(K, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None

    def forward(self, x):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        print( batch_size, in_planes, height, width)
        x = x.view(1, -1, height, width)  # 变化成一个维度进行组卷积
        print(x.shape)
        weight = self.weight.view(self.K, -1)
        print(weight)
        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(self.out_planes, -1, self.kernel_size,
                                                                    self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class DCMModle(nn.Module):
    def __init__(self, in_channels, channels, filter_size=1, fusion=True):
        super(DCMModle, self).__init__()
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.channels = channels
        self.fusion = fusion


        # Global Information vector
       # nn.Conv2d(self.in_channels, self.channels, 1)
        self.reduce_Conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(self.in_channels,self.channels//2,kernel_size=1, padding=1)
        )
        self.reduce_Conv2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(self.in_channels, self.channels//2 , kernel_size=1, padding=1)
        )
        self.reduce_Conv3 = nn.Conv2d(self.channels,self.channels,kernel_size=1)


        self.filter = nn.AdaptiveAvgPool2d(self.filter_size)

        self.filter_gen_conv = nn.Conv2d(self.in_channels, self.channels, 1, 1,0)

        self.residual_conv = nn.Conv2d(self.channels, self.channels, 1)
        self.global_info = nn.Conv2d(self.channels, self.channels, 1)
        self.gla = nn.Conv2d(self.channels, self.filter_size ** 2, 1, 1, 0)
        self.activate = nn.Sequential(
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        )
        if self.fusion:
            self.fusion_conv = nn.Conv2d(self.channels, self.channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        generted_filter = self.filter_gen_conv(self.filter(x)).view(b, self.channels, self.filter_size,self.filter_size)
    #*****************************************************************
        #从新导入特征
        x1 = self.reduce_Conv1(x) #上边的卷积1*1
        x2 = self.reduce_Conv2(x)

        x3 = torch.cat([x1, x2],1)
        x = self.reduce_Conv3(x3)

        #print(x4.shape)
    # *****************************************************************
        c = self.channels # 512

        # [1, b * c, h, w], c = self.channels
        x = x.view(1, b * c, h, w)
        # [b * c, 1, filter_size, filter_size]
        generted_filter = generted_filter.view(b * c, 1, self.filter_size,
                                               self.filter_size)

        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            p2d = (pad, pad, pad, pad)
        else:
            p2d = (pad + 1, pad, pad + 1, pad)

        x = F.pad(input=x, pad=p2d, mode='constant', value=0)

        # [1, b * c, h, w]
        output = nn.functional.conv2d(input=x, weight=generted_filter, groups=b * c)

        # [b, c, h, w]
        output = output.view(b, c, h, w)
        output = self.activate(output)
        if self.fusion:
            output = self.fusion_conv(output)
        return output

if __name__ == '__main__':
    filter_sizes = 7
    fusion = False
    #net = attention2d(3,5)
    net = DCMModle(1024,512)
    #net = Dynamic_conv2d(3,4,3)
    image1 = torch.randn(5, 1024, 64,64)
    out = net(image1)
