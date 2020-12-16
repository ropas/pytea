import LibCall
from .module import Module


class Dropout2d(Module):
    def __init__(self, p=0.5, inplace=False):
        super(Module, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        return x
