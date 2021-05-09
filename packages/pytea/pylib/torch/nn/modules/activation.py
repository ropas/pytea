import LibCall
from .... import torch
from .. import functional as F
from .module import Module
from .linear import Linear


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


class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert LibCall.guard.require_eq(
            self.head_dim * num_heads,
            self.embed_dim,
            "embed_dim must be divisible by num_heads",
        )

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = torch.Tensor(embed_dim, embed_dim)
            self.k_proj_weight = torch.Tensor(embed_dim, self.kdim)
            self.v_proj_weight = torch.Tensor(embed_dim, self.vdim)
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = torch.empty(3 * embed_dim, embed_dim)
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = torch.empty(3 * embed_dim)
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

        if add_bias_kv:
            self.bias_k = torch.empty(1, 1, embed_dim)
            self.bias_v = torch.empty(1, 1, embed_dim)
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
    ):
        shape_q = query.shape
        shape_k = key.shape
        shape_v = value.shape

        assert LibCall.guard.require_eq(
            len(shape_q),
            3,
            "from 'torch.nn.MultiheadAttention': query shape rank should be 3.",
        )
        assert LibCall.guard.require_eq(
            len(shape_k),
            3,
            "from 'torch.nn.MultiheadAttention': key shape rank should be 3.",
        )
        L = shape_q[0]
        N = shape_q[1]
        E = shape_q[2]
        S = shape_k[0]

        assert LibCall.guard.require_shape_eq(
            shape_k,
            torch.Size((S, N, E)),
            "from 'torch.nn.MultiheadAttention': key shape mismatch.",
        )
        assert LibCall.guard.require_shape_eq(
            shape_v,
            torch.Size((S, N, E)),
            "from 'torch.nn.MultiheadAttention': value shape mismatch.",
        )
        if key_padding_mask is not None:
            assert LibCall.guard.require_shape_eq(
                key_padding_mask.shape,
                torch.Size((N, S)),
                "from 'torch.nn.MultiheadAttention': key_padding_mask shape mismatch.",
            )

        # TODO: check attn_mask

        return torch.rand(L, N, E), torch.rand(N, L, S)
