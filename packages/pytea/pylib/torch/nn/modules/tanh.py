import LibCall
from .module import Module
from .. import functional as F


class Tanh(Module):
    def forward(self, input):
        return F.tanh(input)