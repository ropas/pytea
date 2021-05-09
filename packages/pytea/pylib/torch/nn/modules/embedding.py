import LibCall
from .module import Module
from .. import functional as F
from ...tensor import Tensor, Size


class Embedding(Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
    ):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        if _weight:
            self.weight = _weight
        else:
            self.weight = Tensor(num_embeddings, embedding_dim)

    def forward(self, input):
        shape = LibCall.shape.repeat(input.shape, input.ndim, self.embedding_dim)
        return Tensor(Size(shape))
