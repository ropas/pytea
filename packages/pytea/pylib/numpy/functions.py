import LibCall
import torch
import numpy as np
from .ndarray import ndarray
from PIL import Image

# Assumption: obj is well shaped. (wrong)Array below will be parsed as (2, 3).
#             [[2, 3, 4],
#              [5, 6, 7, 8]]
def array(obj, dtype=None, **kwargs):
    if isinstance(obj, torch.Tensor):
        arr = ndarray(obj.shape)
    if isinstance(obj, Image.Image):
        if obj._channel == 1:
            arr = ndarray((obj.height, obj.width))
        else:
            arr = ndarray((obj.height, obj.width, obj._channel))
    else:
        arr = ndarray(LibCall.shape.extractShape(obj))
    arr.dtype = _parseDtype(obj)
    return arr


def _parseDtype(obj):
    if isinstance(obj, ndarray):
        return obj.dtype
    if isinstance(obj, torch.Tensor):
        return np.toNpdtype(obj.dtype)
    if isinstance(obj, Image.Image):
        return np.uint8
    if isinstance(obj, float):
        return np.floatDefault
    elif isinstance(obj, int):
        return np.intDefault

    if isinstance(obj, tuple) or isinstance(obj, list):
        dtypes = []
        for elem in obj:
            dtypes.append(_parseDtype(elem))
        return np.maxDtype(*dtypes)

    LibCall.builtins.warn("cannot infer dtype. fallback to float")
    return np.floatDefault


def zeros(shape, dtype=float, order="C"):
    if isinstance(shape, int):
        shape = [shape]
    return ndarray(shape, dtype=dtype, order=order)


def empty(shape, dtype=float, order="C"):
    if isinstance(shape, int):
        shape = [shape]
    return ndarray(shape, dtype=dtype, order=order)


def copy(a, order="K", subok=False):
    return ndarray(a.shape, dtype=a.dtype, order=order)


def eye(N, M=None, k=0, dtype=float, order="C", like=None):
    if M is None:
        return ndarray((N, N), dtype=dtype, order=order)
    else:
        return ndarray((N, M), dtype=dtype, order=order)


def matmul(x1, x2, out=None, casting="same_kind", order="K", dtype=None, subok=True):
    if not isinstance(x1, ndarray):
        x1 = array(x1)
    if not isinstance(x2, ndarray):
        x2 = array(x2)
    dtype = x1.dtype
    arr = LibCall.numpy.matmul(x1, x2)
    arr.dtype = dtype
    LibCall.numpy.copyOut(arr, out)
    return arr


def concatenate(seq, axis=0, out=None):
    dtype = np.maxDtype(*seq)
    arr = LibCall.numpy.concatenate(seq, axis)
    arr.dtype = dtype
    LibCall.numpy.copyOut(array, out)
    return arr


def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    if dtype is None:
        dtype = a.dtype
    arr = LibCall.numpy.reduce(a, axis, keepdims)
    arr.dtype = dtype
    LibCall.numpy.copyOut(arr, out)
    return arr


def average(a, axis=None, weights=None, returned=False):
    # TODO: implement returned
    if not isinstance(a, ndarray):
        a = array(a)
    arr = LibCall.numpy.reduce(a, axis, False)
    arr.dtype = a.dtype
    return arr


def mean(a, axis=None, dtype=None, out=None, keepdims=False, where=None):
    # TODO: implement `where` option
    if not isinstance(a, ndarray):
        a = array(a)
    arr = LibCall.numpy.reduce(a, axis, keepdims)
    LibCall.numpy.copyOut(arr, out)
    return arr


def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    dtype = a.dtype
    arr = LibCall.numpy.reduce(a, axis, keepdims)
    arr.dtype = dtype
    LibCall.numpy.copyOut(arr, out)
    return arr


def min(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    dtype = a.dtype
    arr = LibCall.numpy.reduce(a, axis, keepdims)
    arr.dtype = dtype
    LibCall.numpy.copyOut(arr, out)
    return arr


def argmax(a, axis=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    if (axis is not None) and (not isinstance(axis, int)):  # tuple axis is not allowed
        raise TypeError("axis must be an int")
    indexArray = LibCall.numpy.reduce(a, axis, False)
    indexArray.dtype = np.intDefault
    LibCall.numpy.copyOut(indexArray, out)
    return indexArray


# torch.Tensor (bop) numpy.ndarray -> allowed
# numpy.ndarray (bop) torch.Tensor -> not allowed
def _bop(array, other):
    if isinstance(other, ndarray):
        dtype = np.maxDtype(array.dtype, other.dtype)
        arr = LibCall.numpy.broadcast(array, other)
        arr.dtype = dtype
        return arr
    elif isinstance(other, int):
        dtype = array.dtype
        arr = LibCall.numpy.identityShape(array)
        arr.dtype = dtype
        return arr
    elif isinstance(other, float):
        if (
            (array.dtype is np.float64)
            or (array.dtype is np.float32)
            or (array.dtype is np.float16)
        ):
            dtype = array.dtype
        else:
            dtype = np.floatDefault
        arr = LibCall.numpy.identityShape(array)
        arr.dtype = dtype
        return arr
    elif isinstance(other, torch.Tensor):
        raise TypeError("unsupported operand type")
    else:
        return NotImplemented

