import LibCall
from .module import Module
from .. import functional as F

class _AvgPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad']

    def extra_repr(self):
        return 'kernel_size={}, stride={}, padding={}'.format(
            self.kernel_size, self.stride, self.padding
        )

class AvgPool2d(_AvgPoolNd):
    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input):
        return F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode)


class AdaptiveAvgPool2d(_AvgPoolNd):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__()
        if isinstance(output_size, tuple) or isinstance(output_size, list):
            self.output_size = output_size
        else:
            self.output_size = (output_size, output_size)

    def forward(self, input):
        return LibCall.torch.adaptive(input, self.output_size)