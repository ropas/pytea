import LibCall
import numpy

class ndarray:
    def __init__(
        self, shape, dtype=float, buffer=None, offset=0, strides=None, order=None
    ):
        LibCall.numpy.ndarrayInit(self, shape, dtype, buffer, offset, strides, order)
        self.dtype = dtype

    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        maxVal = LibCall.numpy.max(self, axis, out, keepdims)
        LibCall.numpy.copyOut(maxVal, out)
        return maxVal

    def argmax(axis=None, out=None):
        pass

    def __add__(self, other):
        return numpy._bop(self, other)

    def __radd__(self, other):
        return numpy._bop(self, other)

    def __sub__(self, other):
        return numpy._bop(self, other)

    def __rsub__(self, other):
        return numpy._bop(self, other)

    def __mul__(self, other):
        return numpy._bop(self, other)

    def __rmul__(self, other):
        return numpy._bop(self, other)

    def __truediv__(self, other):
        return numpy._bop(self, other)

    def __rtruediv__(self, other):
        return numpy._bop(self, other)

    def __floordiv__(self, other):
        return numpy._bop(self, other)

    def __rfloordiv__(self, other):
        return numpy._bop(self, other)

    def __matmul__(self, other):
        return LibCall.numpy.matmul(self, other)

    def __rmatmul__(self, other):
        return LibCall.numpy.matmul(other, self)