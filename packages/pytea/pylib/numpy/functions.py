import LibCall
import torch
from .ndarray import ndarray
from PIL import Image

def array(arrObj, dtype=None, **kwargs):
    if isinstance(arrObj, Image.Image):
        arr = ndarray(())
        LibCall.numpy.fromImage(arr, arrObj)
        return arr

    return NotImplemented

def zeros(shape, dtype=float, order="C"):
    return ndarray(shape, dtype=dtype, order=order)

def empty(shape, dtype=float, order="C"):
    return ndarray(shape, dtype=dtype, order=order)

def matmul(x1, x2, out=None, casting='same_kind', order='K', dtype=None, subok=True):
    if not (isinstance(x1, ndarray) and isinstance(x2, ndarray)):
        raise TypeError("not a numpy.ndarray object")
    array = LibCall.numpy.matmul(x1, x2)
    LibCall.numpy.copyOut(array, out)
    return array

def concatenate(seq, axis=0, out=None):
    array = LibCall.numpy.concatenate(seq, axis)
    LibCall.numpy.copyOut(array, out)
    return array

def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    maxArray = LibCall.numpy.reduce(a, axis, out, keepdims)
    LibCall.numpy.copyOut(maxArray, out)
    return maxArray

# TODO: handle dtype(should be int).
def argmax(a, axis=None, out=None):
    if (axis is not None) and (not isinstance(axis, int)): # tuple axis is not allowed
        raise TypeError("axis must be an int")
    indexArray = LibCall.numpy.reduce(a, axis, out, False)
    LibCall.numpy.copyOut(indexArray, out)
    return indexArray

# torch.Tensor (bop) numpy.ndarray -> allowed
# numpyp.ndarray (bop) torch.Tensor -> not allowed
def _bop(array, other):
    if isinstance(other, ndarray):
        return LibCall.numpy.broadcast(array, other)
    elif isinstance(other, int) or isinstance(other, float):
        return LibCall.numpy.identityShape(tensor)
    elif isinstance(other, torch.Tensor):
        raise TypeError("unsupported operand type")
    else:
        return NotImplemented
