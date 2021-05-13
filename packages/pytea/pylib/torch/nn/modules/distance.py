import LibCall
from .module import Module
from .. import functional as F

class CosineSimilarity(Module):
    def __init__(self, dim=1, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return LibCall.torch.reduce(x1 * x2, self.dim, False)