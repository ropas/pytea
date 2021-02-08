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
            return ndarray(temp)

        if len(temp) <= 0:
            raise IndexError(
                "invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number"
            )

        temp = LibCall.shape.tensorGetItem(temp, 0, index)
        return ndarray(temp)

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