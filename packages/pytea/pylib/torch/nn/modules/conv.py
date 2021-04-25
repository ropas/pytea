import LibCall
from .module import Module
from .... import torch
from .. import functional as F


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(Conv2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert LibCall.guard.require_eq(
            in_channels % groups,
            0,
            "from Conv2d: in_channels must be divisible by groups",
        )
        assert LibCall.guard.require_eq(
            out_channels % groups,
            0,
            "from Conv2d: out_channels must be divisible by groups",
        )

        self.weight = torch.rand(
            out_channels, in_channels // groups, kernel_size[0], kernel_size[1]
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = torch.rand(out_channels)
        else:
            self.bias = None

    def forward(self, input):
        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
    ):
        super(ConvTranspose2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)

        assert LibCall.guard.require_eq(
            in_channels % groups,
            0,
            "from ConvTranspose2d: in_channels must be divisible by groups",
        )
        assert LibCall.guard.require_eq(
            out_channels % groups,
            0,
            "from ConvTranspose2d: out_channels must be divisible by groups",
        )

        self.weight = torch.rand(
            in_channels, out_channels // groups, kernel_size[0], kernel_size[1]
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = torch.rand(out_channels)
        else:
            self.bias = None

    def forward(self, input):
        return F.conv_transpose2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

