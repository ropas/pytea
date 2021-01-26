import LibCall
from .ndarray import ndarray
from PIL import Image

def array(arrObj, dtype=None, **kwargs):
    if isinstance(arrObj, Image.Image):
        arr = ndarray(())
        LibCall.numpy.fromImage(arr, arrObj)
        return arr

    raise Exception("currently support only PIL.Image.Image")

def zeros(shape, dtype=float, order="C"):
    return ndarray(shape, dtype=dtype, order=order)

def empty(shape, dtype=float, order="C"):
    return ndarray(shape, dtype=dtype, order=order)


def concatenate(seq, axis=0, out=None):
    array = LibCall.numpy.concatenate(seq, axis)
    LibCall.numpy.copyOut(array, out)
    return array
