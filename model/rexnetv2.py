"""
ReXNet
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
from math import ceil

# Memory-efficient Siwsh using torch.jit.script borrowed from the code in (https://twitter.com/jeremyphoward/status/1188251041835315200)
# Currently use memory-efficient SiLU as default:
USE_MEMORY_EFFICIENT_SiLU = True

if USE_MEMORY_EFFICIENT_SiLU:
    @torch.jit.script
    def silu_fwd(x):
        return x.mul(torch.sigmoid(x))


    @torch.jit.script
    def silu_bwd(x, grad_output):
        x_sigmoid = torch.sigmoid(x)
        return grad_output * (x_sigmoid * (1. + x * (1. - x_sigmoid)))


    class SiLUJitImplementation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return silu_fwd(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            return silu_bwd(x, grad_output)


    def silu(x, inplace=False):
        return SiLUJitImplementation.apply(x)

else:
    def silu(x, inplace=False):
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class SiLU(nn.Module):
    def __init__(self, inplace=True):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return silu(x, self.inplace)


def ConvBNAct(out, in_channels, channels, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.append(nn.Conv2d(in_channels, channels, kernel,
                         stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))


def ConvBNSiLU(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    out.append(nn.Conv2d(in_channels, channels, kernel,
                         stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    out.append(SiLU(inplace=True))


class ECA(nn.Module):
    def __init__(self, in_channels, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SE(nn.Module):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, channels // se_ratio, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, channels, t, stride, use_channel_attn=True, channel_attn='eca', channel_attn_param=12, drop_path=0.0,
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels
        
        if drop_path > 0:
            self.drop_path = nn.Dropout2d(drop_path)
        else:
            self.drop_path = nn.Identity()

        out = []
        if t != 1:
            dw_channels = in_channels * t
            ConvBNSiLU(out, in_channels=in_channels, channels=dw_channels)
        else:
            dw_channels = in_channels

        ConvBNAct(out, in_channels=dw_channels, channels=dw_channels, kernel=3, stride=stride, pad=1,
                  num_group=dw_channels, active=False)

        if use_channel_attn:
            if channel_attn == 'se':
                out.append(SE(dw_channels, dw_channels, channel_attn_param))
            elif channel_attn == 'eca':
                out.append(ECA(dw_channels, dw_channels, channel_attn_param))
            else:
                raise Exception("Not implemented !")

        out.append(nn.ReLU6())
        ConvBNAct(out, in_channels=dw_channels, channels=channels, active=False, relu6=True)
        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out = self.drop_path(out)
            out[:, 0:self.in_channels] += x
        
        return out


class ReXNetV2(nn.Module):
    def __init__(self, input_ch=16, final_ch=180, width_mult=1.0, depth_mult=1.0, classes=1000,
                 use_channel_attn=True,
                 channel_attn='eca',
                 channel_attn_param=3,
                 dropout_ratio=0.2,
                 drop_path = 0.25,
                 bn_momentum=0.9):
        super(ReXNetV2, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_channel_attns = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1)
                       for idx, element in enumerate(strides)], [])
        if use_channel_attn:
            use_channel_attns = sum([[element] * layers[idx] for idx, element in enumerate(use_channel_attns)], [])
        else:
            use_channel_attns = [False] * sum(layers[:])

        # print(use_channel_attns)

        ts = [1] * layers[0] + [6] * sum(layers[1:])

        self.depth = sum(layers[:]) * 3
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))

        ConvBNSiLU(features, 3, int(round(stem_channel * width_mult)), kernel=3, stride=2, pad=1)

        for block_idx, (in_c, c, t, s, use_channel_attn) in enumerate(zip(in_channels_group, channels_group, ts, strides, use_channel_attns)):
            features.append(LinearBottleneck(in_channels=in_c,
                                             channels=c,
                                             t=t,
                                             stride=s,
                                             use_channel_attn=use_channel_attn,
                                             channel_attn=channel_attn, channel_attn_param=channel_attn_param, drop_path=drop_path))

        pen_channels = int(1280 * width_mult)
        ConvBNSiLU(features, c, pen_channels)

        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)
        self.out = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv2d(pen_channels, classes, 1, bias=True))
        
    def extract_features(self, x):
        return self.features[:-1](x)
    
    def forward(self, x):
        x = self.features(x)
        x = self.out(x).flatten(1)
        return x


if __name__ == '__main__':
    model = ReXNetV2(width_mult=1.0, classes=4)
    out = model(torch.randn(2, 3, 224, 224))
    print(out.size())
    loss = out.sum()
    loss.backward()
    print('Checked a single forward/backward iteration')