import LibCall
from .module import Module
from .. import functional as F


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self._modules = args

    def __len__(self):
        return len(self._modules)

    def forward(self, input):
        for module in self._modules:
            input = module(input)
        return input


# TODO: temporarily make it list gen.
def ModuleList(module=None):
    ret_val = []
    if module is None:
        return ret_val

    for m in module:
        ret_val.append(m)

    return ret_val

