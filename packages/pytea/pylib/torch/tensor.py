from .. import LibCall
from collections import namedtuple

import torch
import numpy


class Tensor:
    def __init__(self, *args, dtype=None, **kwargs):
        LibCall.torch.tensorInit(self, args, kwargs)
        if dtype is None:
            self.dtype = torch.floatDefault
        else:
            self.dtype = dtype
        self.data = self

    # # TODO: make @staticmethod
    def __getattr__(self, attr):
        if attr == "ndim":
            return len(self.shape)

        return NotImplemented

    def new_tensor(self, data, dtype=None, device=None, requires_grad=False):
        return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

    def backward(self):
        return self

    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def dim(self):
        return len(self.shape)

    def matmul(self, other):
        return torch.matmul(self, other)

    def add(self, other):
        return torch._bop(self, other)

    def add_(self, other):
        torch._bop(self, other)
        return self

    def sub(self, other):
        return torch._bop(self, other)

    def sub_(self, other):
        torch._bop(self, other)
        return self

    def mul(self, other):
        return torch._bop(self, other)

    def mul_(self, other):
        torch._bop(self, other)
        return self

    def div(self, other):
        return torch._bop(self, other)

    def div_(self, other):
        torch._bop(self, other)
        return self

    def pow(self, exponent):
        return torch._bop(self, exponent)

    def pow_(self, other):
        torch._bop(self, other)
        return self

    def neg(self):
        return torch.neg(self)

    def neg_(self):
        return self

    def abs(self):
        return torch.abs(self)

    def abs_(self):
        return self

    def log(self):
        return torch.log(self)

    def log_(self):
        return self

    def log10(self):
        return torch.log10(self)

    def log10_(self):
        return self

    def log1p(self):
        return torch.log1p(self)

    def log1p_(self):
        return self

    def log2(self):
        return torch.log2(self)

    def log2(self):
        return self

    def exp(self):
        return torch.exp(self)

    def exp_(self):
        return self

    def expm1(self):
        return torch.exp(self)

    def expm1_(self):
        return self

    def fill_(self):
        return self

    def bernoulli(self, generator=None):
        return torch.bernoulli(self)

    def bernoulli_(self, p=0.5, generator=None):
        return self

    def normal_(self, *args, **kwargs):
        return self

    def uniform_(self, _from=0, to=1):
        # TODO: from is reserved keyword.
        return self

    def random_(self, _from=0, to=None, generator=None):
        # TODO: from is reserved keyword.
        return self

    def repeat(self, *sizes):
        tensor = LibCall.torch.repeat(self, sizes)
        tensor.dtype = self.dtype
        return tensor

    def transpose(self, dim0, dim1):
        return torch.transpose(self, dim0, dim1)

    def t(self):
        return torch.t(self)

    def argmax(self, dim=None, keepdim=False):
        return torch.argmax(self, dim, keepdim)

    def numpy(self):
        return numpy.ndarray(self.shape)

    # TODO: dim can be tuple. LibCall.torch.reduce must cover it.
    def sum(self, dim=None, keepdim=False, dtype=None):
        return torch.sum(self, dim, keepdim, dtype)

    def mean(self, dim=None, keepdim=False, dtype=None):
        return torch.mean(self, dim, keepdim, dtype)

    def det(self):
        return torch.det(self)

    def logdet(self):
        return torch.logdet(self)

    def item(self):
        return LibCall.torch.item(self)

    def eq(self, other):
        return torch._bop(self, other)

    def view(self, *shape):
        tensor = LibCall.torch.view(self, shape)
        tensor.dtype = self.dtype
        return tensor

    def view_as(self, other):
        return self.view(other.size())

    def reshape(self, *shape):
        tensor = LibCall.torch.view(self, shape)
        tensor.dtype = self.dtype
        return tensor

    def reshape_as(self, other):
        return self.view(other.size())

    def squeeze(self, dim=None):
        return torch.squeeze(self, dim)

    def squeeze_(self, dim=None):
        value = torch.squeeze(self, dim)
        self.shape = value.shape
        return self

    def unsqueeze(self, dim):
        return torch.unsqueeze(self, dim)

    def unsqueeze_(self, dim=None):
        value = torch.unsqueeze(self, dim)
        self.shape = value.shape
        return self

    def cuda(self, **kwargs):
        return self

    def requires_grad_(self, requires_grad=True):
        return self

    def mm(self, mat2):
        return torch.mm(self, mat2)

    def bmm(self, batch2):
        return torch.bmm(self, batch2)

    def topk(self, k, dim=None, largest=True, sorted=True, out=None):
        if dim is None:
            dim = len(self.shape) - 1
        return torch.topk(self, k, dim)

    def to(self, *args, **kwargs):
        firstArg = args[0]

        if isinstance(firstArg, Tensor):
            return self.type(firstArg.dtype)
        elif isinstance(firstArg, str):  # device
            tensor = LibCall.torch.identityShape(self)
            tensor.dtype = self.dtype
            tensor.device = firstArg
            return tensor
        elif isinstance(firstArg, torch.device):
            tensor = LibCall.torch.identityShape(self)
            tensor.dtype = self.dtype
            tensor.device = firstArg
            return tensor
        else:
            return self.type(firstArg)

        return NotImplemented

    def type(self, dtype=None, **kwargs):
        if dtype is None:
            return self.dtype
        elif self.dtype is dtype:
            return self
        else:
            tensor = LibCall.torch.identityShape(self)
            tensor.dtype = dtype
            return tensor

    def bool(self):
        return self.to(torch.bool)

    def float(self):
        return self.to(torch.float)

    def long(self, **kwargs):
        return self.to(torch.int64)

    def detach(self):
        tensor = LibCall.torch.identityShape(self)
        tensor.dtype = self.dtype
        return tensor

    def clone(self):
        tensor = LibCall.torch.identityShape(self)
        tensor.dtype = self.dtype
        return tensor

    def cpu(self):
        tensor = LibCall.torch.identityShape(self)
        tensor.dtype = self.dtype
        return tensor

    def flatten(self, start_dim=0, end_dim=-1):
        tensor = LibCall.torch.flatten(self, start_dim, end_dim)
        tensor.dtype = self.dtype
        return tensor

    def expand(self, *sizes):
        tensor = LibCall.torch.expand(self, sizes)
        tensor.dtype = self.dtype
        return tensor

    def expand_as(self, other):
        tensor = LibCall.torch.expand_as(self, other)
        tensor.dtype = self.dtype
        return tensor

    def device(self):
        return "cuda"

    def permute(self, *args):
        ndim = self.dim()
        if ndim != len(args):
            raise ValueError("permute shape mismatched")
        ret_shape = []
        for arg in args:
            # TODO: add duplicated indices assertion
            # currently, __setitem__ for list is not supported
            if arg < 0 or arg >= ndim:
                raise ValueError("permute invalid index!")
            ret_shape.append(self.shape[arg])

        tensor = self.view(*ret_shape)
        tensor.dtype = self.dtype
        return tensor

    def contiguous(self):
        return self

    def __len__(self):
        if len(self.shape) == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __getitem__(self, index):
        temp = self.shape

        if isinstance(index, tuple):
            idx_len = len(index)
            if idx_len > len(self.shape):
                raise IndexError("too many indices for tensor")
            for axis in range(idx_len - 1, -1, -1):
                temp = LibCall.shape.tensorGetItem(temp, axis, index[axis])
            return Tensor(temp)

        if len(temp) <= 0:
            raise IndexError(
                "invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number"
            )

        temp = LibCall.shape.tensorGetItem(temp, 0, index)
        return Tensor(temp)

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            sliced = self[key]
            assert LibCall.guard.require_broadcastable(
                sliced,
                value,
                "The expanded size of the assigned value must match the target size.",
            )

    def __neg__(self):
        return torch.neg(self)

    def __add__(self, other):
        return torch._bop(self, other)

    def __radd__(self, other):
        return torch._bop(self, other)

    def __sub__(self, other):
        return torch._bop(self, other)

    def __rsub__(self, other):
        return torch._bop(self, other)

    def __mul__(self, other):
        return torch._bop(self, other)

    def __rmul__(self, other):
        return torch._bop(self, other)

    def __truediv__(self, other):
        return torch._bop(self, other)

    def __rtruediv__(self, other):
        return torch._bop(self, other)

    def __floordiv__(self, other):
        return torch._bop(self, other)

    def __rfloordiv__(self, other):
        return torch._bop(self, other)

    def __lt__(self, other):
        tensor = torch._bop(self, other)
        tensor.dtype = torch.bool
        return tensor

    def __le__(self, other):
        tensor = torch._bop(self, other)
        tensor.dtype = torch.bool
        return tensor

    def __gt__(self, other):
        tensor = torch._bop(self, other)
        tensor.dtype = torch.bool
        return tensor

    def __ge__(self, other):
        tensor = torch._bop(self, other)
        tensor.dtype = torch.bool
        return tensor

    def __eq__(self, other):
        tensor = torch._bop(self, other)
        tensor.dtype = torch.bool
        return tensor

    def __ne__(self, other):
        tensor = torch._bop(self, other)
        tensor.dtype = torch.bool
        return tensor

    def __matmul__(self, other):
        tensor = LibCall.torch.matmul(self, other)
        tensor.dtype = self.dtype
        return tensor

    def __rmatmul__(self, other):
        tensor = LibCall.torch.matmul(other, self)
        tensor.dtype = self.dtype
        return tensor

    def __abs__(self):
        tensor = LibCall.torch.identityShape(self)
        tensor.dtype = self.dtype
        return tensor

    def __eq__(self, other):
        return torch._bop(self, other)

    def max(self, dim=None, keepdim=False):
        return torch.max(self, dim, keepdim)

    def sqrt(self):
        return torch.sqrt(self)


def FloatTensor(*args, **kwargs):
    return Tensor(*args, **kwargs)


def DoubleTensor(*args, **kwargs):
    kwargs["dtype"] = torch.float64
    return Tensor(*args, **kwargs)


def IntTensor(*args, **kwargs):
    kwargs["dtype"] = torch.int32
    return Tensor(*args, **kwargs)


def LongTensor(*args, **kwargs):
    kwargs["dtype"] = torch.int64
    return Tensor(*args, **kwargs)

