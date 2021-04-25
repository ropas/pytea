import LibCall
from .... import torch
from .. import functional as F
from .module import Module


class LeakyReLU(Module):
    def __init__(self, inplace=False):
        super(LeakyReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class ReLU(Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Softmax(Module):
    def __init__(self, dim=None):
        super(Softmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input):
        return F.softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return "dim={dim}".format(dim=self.dim)


class Sigmoid(Module):
    def forward(self, input):
        return torch.sigmoid(input)


class Tanh(Module):
    def forward(self, input):
        return torch.tanh(input)
