import LibCall
from .module import Module
from .. import functional as F

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = LibCall.torch.callTensor(out_features, in_features)
        if bias:
            self.bias = LibCall.torch.callTensor(out_features)
        else:
            self.bias = None

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
