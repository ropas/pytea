import LibCall
import numpy


class ndarray:
    def __init__(
        self, shape, dtype=float, buffer=None, offset=0, strides=None, order=None
    ):
        LibCall.numpy.ndarrayInit(self, shape, dtype, buffer, offset, strides, order)
        self.dtype = dtype

    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        return numpy.max(self, axis, out, keepdims, initial, where)

    def argmax(self, axis=None, out=None):
        return numpy.argmax(self, axis, out)

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

    def __eq__(self, other):
        return numpy._bop(self, other)

    def __matmul__(self, other):
        return LibCall.numpy.matmul(self, other)

    def __rmatmul__(self, other):
        return LibCall.numpy.matmul(other, self)

    def __len__(self):
        if len(self.shape) == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def flatten(self, order="C"):
        return LibCall.numpy.flatten(self)
