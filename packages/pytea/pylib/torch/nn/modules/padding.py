import LibCall
from .module import Module
from .. import functional as F


class ReflectionPad2d(Module):
    def __init__(self, padding):
        super(ReflectionPad2d, self).__init__()
        if isinstance(padding, tuple) or isinstance(padding, list):
            self.padding = padding
        else:
            self.padding = (padding, padding, padding, padding)

    def forward(self, input):
        return F.pad(input, self.padding, "reflect")

