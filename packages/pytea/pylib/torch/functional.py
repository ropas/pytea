import LibCall
import torch
from collections import namedtuple
import numpy as np
from torch.tensor import Tensor


TVI = namedtuple("TorchValId", ["values", "indices"])


def tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    # TODO: infer dtype from data
    shape = LibCall.shape.extractShape(data)
    result = torch.Tensor(shape, dtype=dtype)
    return result


def rand(*size, out=None, dtype=None, **kwargs):
    if dtype is None:
        dtype = torch.floatDefault
    tensor = Tensor(*size, dtype=dtype)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def rand_like(input, dtype=None, **kwargs):
    if dtype is None:
        dtype = input.dtype
    tensor = LibCall.torch.identityShape(input)
    tensor.dtype = dtype
    return tensor


def empty(*size, out=None, dtype=None, **kwargs):
    if dtype is None:
        dtype = torch.floatDefault
    tensor = Tensor(*size, dtype=dtype)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def randn(*size, out=None, dtype=None, **kwargs):
    if dtype is None:
        dtype = torch.floatDefault
    tensor = Tensor(*size, dtype=dtype)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def randn_like(input, dtype=None, **kwargs):
    if dtype is None:
        dtype = input.dtype
    tensor = LibCall.torch.identityShape(input)
    tensor.dtype = dtype
    return tensor


# TODO: optional low
def randint(*args, **kwargs):
    dtype = torch.intDefault
    if "dtype" in kwargs:
        dtype = kwargs["dtype"]

    size = args[1]
    if len(args) > 2:
        size = args[2]
    tensor = Tensor(*size, dtype=dtype)
    return tensor


def randint_like(input, low=0, high=1, dtype=None, **kwargs):
    if dtype is None:
        dtype = input.dtype
    tensor = LibCall.torch.identityShape(input)
    tensor.dtype = dtype
    return tensor


def ones(*size, out=None, dtype=None, **kwargs):
    if dtype is None:
        dtype = torch.floatDefault
    tensor = Tensor(*size, dtype=dtype)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def ones_like(input, dtype=None, **kwargs):
    if dtype is None:
        dtype = input.dtype
    tensor = LibCall.torch.identityShape(input)
    tensor.dtype = dtype
    return tensor


def zeros(*size, out=None, dtype=None, **kwargs):
    if dtype is None:
        dtype = torch.floatDefault
    tensor = Tensor(*size, dtype=dtype)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def zeros_like(input, dtype=None, **kwargs):
    if dtype is None:
        dtype = input.dtype
    tensor = LibCall.torch.identityShape(input)
    tensor.dtype = dtype
    return tensor


def eye(n, m=None, out=None, dtype=None, **kwargs):
    if dtype is None:
        dtype = torch.floatDefault
    if m is None:
        m = n
    tensor = Tensor(n, m, dtype=dtype)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def cat(tensors, dim=0, out=None):
    tensor = LibCall.torch.cat(tensors, dim)
    tensor.dtype = torch.maxDtype(*tensors)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def squeeze(input, dim=None, out=None):
    dtype = input.dtype
    tensor = LibCall.torch.squeeze(input, dim)
    tensor.dtype = dtype
    return tensor


def unsqueeze(input, dim):
    dtype = input.dtype
    tensor = LibCall.torch.unsqueeze(input, dim)
    tensor.dtype = dtype
    return tensor


def diag(input, diagonal=0, out=None):
    dtype = input.dtype
    tensor = LibCall.torch.diag(input, diagonal)
    tensor.dtype = dtype
    LibCall.torch.copyOut(tensor, out)
    return tensor


def matmul(input, other, out=None):
    if not (isinstance(input, Tensor) and isinstance(other, Tensor)):
        raise TypeError("not a torch.Tensor object")
    if not (input.dtype == other.dtype):
        raise TypeError("Tensor dtype mismatch")
    dtype = input.dtype
    tensor = LibCall.torch.matmul(input, other)
    tensor.dtype = dtype
    LibCall.torch.copyOut(tensor, out)
    return tensor


def mm(input, mat2, out=None):
    if not (input.dtype == mat2.dtype):
        raise TypeError("Tensor dtype mismatch")
    dtype = input.dtype
    tensor = LibCall.torch.mm(input, mat2)
    tensor.dtype = dtype
    LibCall.torch.copyOut(tensor, out)
    return tensor


def bmm(input, mat2, deterministic=False, out=None):
    if not (input.dtype == mat2.dtype):
        raise TypeError("Tensor dtype mismatch")
    dtype = input.dtype
    tensor = LibCall.torch.bmm(input, mat2)
    tensor.dtype = dtype
    LibCall.torch.copyOut(tensor, out)
    return tensor


def topk(input, k, dim=None, largest=True, sorted=True, out=None):
    tensor = LibCall.torch.topk(input, k, dim)
    tensor.dtype = input.dtype
    index = zeros_like(tensor)
    index.dtype = torch.intDefault
    return TVI(tensor, index)


def transpose(input, dim0, dim1):
    dtype = input.dtype
    tensor = LibCall.torch.transpose(input, dim0, dim1)
    tensor.dtype = dtype
    return tensor


def t(input):
    rank = len(input.shape)
    if rank > 2:
        raise ValueError("t() expects a tensor with <= 2 dimensions")
    elif rank == 2:
        return input.transpose(1, 0)
    return input


def reshape(input, shape):
    dtype = input.dtype
    tensor = input.reshape(*shape)
    tensor.dtype = dtype
    return tensor


def argmax(input, dim=None, keepdim=False):
    tensor = LibCall.torch.reduce(input, dim, keepdim)
    tensor.dtype = torch.intDefault
    return tensor


def max(input, dim=None, keepdim=False, out=None):
    dtype = input.dtype
    if dim is None:
        tensor = LibCall.torch.reduce(input, dim, keepdim)
        tensor.dtype = dtype
        LibCall.torch.copyOut(tensor, out)
        return tensor
    else:
        tensor = LibCall.torch.reduce(input, dim, keepdim)
        tensor.dtype = dtype
        indice = LibCall.torch.reduce(input, dim, keepdim)
        indice.dtype = torch.intDefault
        if out is not None:
            LibCall.torch.copyOut(tensor, out[0])
            LibCall.torch.copyOut(indice, out[1])
        return tensor, indice


def mean(input, dim=None, keepdim=False, dtype=None, out=None):
    if dtype is None:
        dtype = input.dtype
    tensor = LibCall.torch.reduce(input, dim, keepdim)
    tensor.dtype = dtype
    return tensor


def sum(input, dim=None, keepdim=False, dtype=None):
    if dtype is None:
        if input.dtype == torch.bool:
            dtype = torch.intDefault
        else:
            dtype = input.dtype

    tensor = LibCall.torch.reduce(input, dim, keepdim)
    tensor.dtype = dtype
    return tensor


def det(input):
    dtype = input.dtype
    tensor = LibCall.torch.reduce(input, None, False)
    tensor.dtype = dtype
    return tensor


def logdet(input):
    dtype = input.dtype
    tensor = LibCall.torch.reduce(input, None, False)
    tensor.dtype = dtype
    return tensor


def mul(input, other, out=None):
    result = _bop(input, other)
    LibCall.torch.copyOut(result, out)
    return result


def pow(input, exponent, out=None):
    result = _bop(input, exponent)
    LibCall.torch.copyOut(result, out)
    return result


def abs(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def neg(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


negative = neg


def exp(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def expm1(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def log(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def log10(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def log1p(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def log2(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def bernoulli(input, generator=None, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def _bop(tensor, other):
    if isinstance(other, Tensor):
        dtype = torch.maxDtype(tensor, other)
        tensor = LibCall.torch.broadcast(tensor, other)
        tensor.dtype = dtype
        return tensor
    elif isinstance(other, np.ndarray):
        return LibCall.torch.broadcast(tensor, other)
    elif isinstance(other, int):
        dtype = tensor.dtype
        tensor = LibCall.torch.identityShape(tensor)
        tensor.dtype = dtype
        return tensor
    elif isinstance(other, float):
        if (
            tensor.dtype is torch.float64
            or tensor.dtype is torch.float32
            or tensor.dtype is torch.float16
        ):
            dtype = tensor.dtype
        else:
            dtype = torch.floatDefault
        tensor = LibCall.torch.identityShape(tensor)
        tensor.dtype = dtype
        return tensor
    else:
        return NotImplemented


def flatten(input, start_dim=0, end_dim=-1):
    dtype = input.dtype
    tensor = LibCall.torch.flatten(input, start_dim, end_dim)
    tensor.dtype = dtype
    return tensor


def sqrt(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def tanh(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def sigmoid(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def relu(input):
    result = LibCall.torch.identityShape(input)
    return result


def gelu(input):
    result = LibCall.torch.identityShape(input)
    return result


def softmax(input, dim=None, dtype=None):
    if not (input.dtype in torch.floatTypes):
        raise TypeError("Can only calculate the softmax of floating types")
    dtype = input.dtype
    tensor = LibCall.torch.identityShape(input)
    tensor.dtype = dtype
    return tensor


def arange(start, end=None, step=1, out=None, **kwargs):
    if end is None:  # arange(N)
        end = start
        start = 0
    if isinstance(start, int) and isinstance(end, int) and isinstance(step, int):
        dtype = torch.intDefault
    else:
        dtype = torch.floatDefault
    tensor = Tensor(int((end - start) / step))
    tensor.dtype = dtype
    LibCall.torch.copyOut(tensor, out)
    return tensor


def full(size, fill_value, out=None, dtype=None, **kwargs):
    if dtype is None:
        dtype = torch.floatDefault
    tensor = Tensor(*size, dtype=dtype)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def save(
    obj, f, pickle_module=None, pickle_protocol=2, _use_new_zipfile_serialization=True
):
    pass


def manual_seed(seed=0):
    pass
