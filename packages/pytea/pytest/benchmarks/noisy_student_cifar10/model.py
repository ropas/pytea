"""
Resnet-X model with stochastic depth feature.

Mostly followed the SimCLR code practice.

Naming convension and implementation is from
shamangary/Pytorch-Stochastic-Depth-Resnet with github link:
https://github.com/shamangary/Pytorch-Stochastic-Depth-Resnet/blob/master/TYY_stodepth_lineardecay.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()
        self.out_channels = out_channels

    def forward(self, x):
        out = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.out_channels // 4, self.out_channels // 4), "constant", 0)
        return out


class StoDepth_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, prob, norm_layer, stride=1, downsample=None):
        super(StoDepth_BasicBlock, self).__init__()
        self.stride = stride
        self.norm_layer = norm_layer

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU()

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.relu2 = nn.ReLU()

        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(self.prob)

    def forward(self, x):
        residual = x

        if self.training:
            sample = self.m.sample().item()
            if sample > 0:
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu1(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    residual = self.downsample(x)
                out = out + residual
            else:
                if self.downsample is not None:
                    residual = self.downsample(x)
                out = residual
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)
            out = self.prob * out + residual

        out = self.relu2(out)
        return out


class StoDepth_ResNet(nn.Module):
    def __init__(
        self,
        num_layers,
        prob_0_L=(1.0, 0.8),
        dropout_prob=0.5,
        num_classes=10,
        width=1,
        norm_layer=nn.BatchNorm2d,
    ):
        super(StoDepth_ResNet, self).__init__()

        self.num_layers = num_layers
        assert (num_layers - 2) % 6 == 0
        self.N = (self.num_layers - 2) // 6

        self.prob = prob_0_L[0]
        self.prob_step = (prob_0_L[0] - prob_0_L[1]) / (self.N * 3)

        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.inplanes = 16 * width

        self.conv = nn.Conv2d(3, self.inplanes, 3, padding=1, bias=False)

        self.layer1 = self._make_layers(planes=16 * width, stride=1)
        self.layer2 = self._make_layers(planes=32 * width, stride=2)
        self.layer3 = self._make_layers(planes=64 * width, stride=2)

        self.bn = norm_layer(64 * width)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(8)
        self.drop = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(64 * width, num_classes)

        # weight initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, torch.sqrt(torch.Tensor([2.0 / n])).item())

    def _make_layers(self, planes, stride):
        if stride == 2:
            down_sample = IdentityPadding(self.inplanes, planes, stride)
        else:
            down_sample = None

        layers_list = nn.ModuleList(
            [
                StoDepth_BasicBlock(
                    self.inplanes,
                    planes,
                    self.prob,
                    self.norm_layer,
                    stride,
                    down_sample,
                )
            ]
        )
        self.inplanes = planes
        self.prob -= self.prob_step

        for _ in range(self.N - 1):
            layers_list.append(StoDepth_BasicBlock(planes, planes, self.prob, self.norm_layer))
            self.prob -= self.prob_step

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.drop(x)
        x = self.fc(x)
        return x
