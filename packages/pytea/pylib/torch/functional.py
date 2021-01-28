import LibCall
from .. import numpy as np
from torch.tensor import Tensor


def rand(*size, out=None, **kwargs):
    tensor = Tensor(*size)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def empty(*size, out=None, **kwargs):
    tensor = Tensor(*size)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def randn(*size, out=None, **kwargs):
    tensor = Tensor(*size)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def ones(*size, out=None, **kwargs):
    tensor = Tensor(*size)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def zeros(*size, out=None, **kwargs):
    tensor = Tensor(*size)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def eye(n, m=None, out=None, **kwargs):
    if m is None:
        m = n
    tensor = Tensor(n, m)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def cat(tensors, dim=0, out=None):
    tensor = LibCall.torch.cat(tensors, dim)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def unsqueeze(input, dim):
    return LibCall.torch.unsqueeze(input, dim)


def diag(input, diagonal=0, out=None):
    tensor = LibCall.torch.diag(input, diagonal)
    LibCall.torch.copyOut(tensor, out)
    return tensor


def matmul(input, other, out=None):
    if not (isinstance(input, Tensor) and isinstance(other, Tensor)):
        raise TypeError("not a torch.Tensor object")
    tensor = LibCall.torch.matmul(input, other)
    LibCall.torch.copyOut(tensor, out)
    return tensor

def mm(input, mat2, out=None):
    tensor = LibCall.torch.mm(input, mat2)
    LibCall.torch.copyOut(tensor, out)
    return tensor

def bmm(input, mat2, deterministic=False, out=None):
    tensor = LibCall.torch.bmm(input, mat2)
    LibCall.torch.copyOut(tensor, out)
    return tensor

def transpose(input, dim0, dim1):
    return LibCall.torch.transpose(input, dim0, dim1)


def argmax(input, dim=None, keepdim=False):
    return LibCall.torch.reduce(input, dim, keepdim)


def mean(input, dim=None, keepdim=False, out=None):
    return LibCall.torch.reduce(input, dim, keepdim)


def sum(input, dim=None, keepdim=False, dtype=None):
    return LibCall.torch.reduce(input, dim, keepdim)


def softmax(input, dim=None, dtype=None):
    return LibCall.torch.identityShape(input)


def mul(input, other, out=None):
    result = _bop(input, other)
    LibCall.torch.copyOut(result, out)
    return result


def pow(input, exponent, out=None):
    result = _bop(input, exponent)
    LibCall.torch.copyOut(result, out)
    return result


def _bop(tensor, other):
    if isinstance(other, Tensor) or isinstance(other, np.ndarray):
        return LibCall.torch.broadcast(tensor, other)
    elif isinstance(other, int) or isinstance(other, float):
        return LibCall.torch.identityShape(tensor)
    else:
        return NotImplemented


def flatten(input, start_dim=0, end_dim=-1):
    return LibCall.torch.flatten(input, start_dim, end_dim)


def sqrt(input, out=None):
    LibCall.torch.copyOut(input, out)
    return input


def manual_seed(seed=0):
    pass


def save(obj, f, pickle_module=None, pickle_protocol=2, _use_new_zipfile_serialization=True):
    pass
