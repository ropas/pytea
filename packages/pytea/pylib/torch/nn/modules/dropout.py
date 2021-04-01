import LibCall
from .module import Module
from .. import functional as F


class _DropoutNd(Module):
    def __init__(self, p=0.5, inplace=False):
        super(_DropoutNd, self).__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(
                "dropout probability has to be between 0 and 1, but got ..."
            )
        self.p = p
        self.inplace = inplace


class Dropout(_DropoutNd):
    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)


class Dropout2d(_DropoutNd):
    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)
