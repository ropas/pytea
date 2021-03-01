import LibCall
import torch
import numpy as np
from .ndarray import ndarray
from PIL import Image

# Assumption: arrObj is well shaped. (wrong)Array below will be parsed as (2, 3).
#             [[2, 3, 4],
#              [5, 6, 7, 8]]
def array(arrObj, dtype=None, **kwargs):
    if isinstance(arrObj, torch.Tensor):
        arr = ndarray(arrObj.shape)
    if isinstance(arrObj, Image.Image):
        arr = ndarray(())
        LibCall.numpy.fromImage(arr, arrObj)
    else:
        arr = _parseShape(arrObj, [])
    arr.dtype = _parseDtype(arrObj)
    return arr


def _parseShape(arrObj, size):
    if isinstance(arrObj, ndarray):
        pre = ndarray(size)
        return LibCall.shape.shapeConcat(pre, arrObj, ndarray(()))
    if isinstance(arrObj, float):
        return ndarray(size, dtype=float)
    elif isinstance(arrObj, int):
        return ndarray(size, dtype=int)

    if isinstance(arrObj, tuple) or isinstance(arrObj, list):
        if len(arrObj) == 0:
            return ndarray([0])
        else:
            size.append(len(arrObj))
            return _parseShape(arrObj[0], size)

    raise TypeError("invalid array")


def _parseDtype(arrObj, size):
    if isinstance(arrObj, ndarray):
        return arrObj.dtype
    if isinstance(arrObj, torch.Tensor):
        return np.toNpdtype(arrObj.dtype)
    if isinstance(arrObj, Image.Image):
        # TODO: dtype
        return np.floatDefault
    if isinstance(arrObj, float):
        return np.floatDefault
    elif isinstance(arrObj, int):
        return np.intDefault

    if isinstance(arrObj, tuple) or isinstance(arrObj, list):
        dtypes = []
        for elem in arrObj:
            dtypes.append(_parseDtype(elem))
        return np.maxDtype(*dtypes)

    raise TypeError("invalid array")



def zeros(shape, dtype=float, order="C"):
    if isinstance(shape, int):
        shape = [shape]
    return ndarray(shape, dtype=dtype, order=order)


def empty(shape, dtype=float, order="C"):
    if isinstance(shape, int):
        shape = [shape]
    return ndarray(shape, dtype=dtype, order=order)


def eye(N, M=None, k=0, dtype=float, order="C", like=None):
    if M is None:
        return ndarray((N, N), dtype=dtype, order=order)
    else:
        return ndarray((N, M), dtype=dtype, order=order)


def matmul(x1, x2, out=None, casting="same_kind", order="K", dtype=None, subok=True):
    if not (isinstance(x1, ndarray) and isinstance(x2, ndarray)):
        raise TypeError("not a numpy.ndarray object")
    dtype = x1.dtype
    array = LibCall.numpy.matmul(x1, x2)
    array.dtype = dtype
    LibCall.numpy.copyOut(array, out)
    return array


def concatenate(seq, axis=0, out=None):
    dtype = np.maxDtype(*seq)
    array = LibCall.numpy.concatenate(seq, axis)
    array.dtype
    LibCall.numpy.copyOut(array, out)
    return array


def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if dtype is None:
        dtype = a.dtype
    sumArray = LibCall.numpy.reduce(a, axis, out, keepdims)
    sumArray.dtype = dtype
    LibCall.numpy.copyOut(sumArray, out)
    return sumArray


def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    dtype = a.dtype
    maxArray = LibCall.numpy.reduce(a, axis, out, keepdims)
    maxArray.dtype = dtype
    LibCall.numpy.copyOut(maxArray, out)
    return maxArray


def argmax(a, axis=None, out=None):
    if (axis is not None) and (not isinstance(axis, int)):  # tuple axis is not allowed
        raise TypeError("axis must be an int")
    indexArray = LibCall.numpy.reduce(a, axis, out, False)
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


def mean(a, axis=None, dtype=None, out=None, keepdims=False, where=None):
    # TODO: implement `where` option
    array = LibCall.numpy.reduce(a, axis, out, keepdims)
    LibCall.numpy.copyOut(array, out)
    return array
