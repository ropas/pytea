import LibCall
import numpy as np


class ndarray:
    def __init__(
        self, shape, dtype=float, buffer=None, offset=0, strides=None, order=None
    ):
        LibCall.numpy.ndarrayInit(self, shape)
        if dtype is float:
            dtype = np.floatDefault
        elif dtype is int:
            dtype = np.intDefault
        elif dtype is bool:
            dtype = np.bool
        self.dtype = dtype

    def __len__(self):
        if len(self.shape) == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __setitem__(self, index, value):
        # TODO: boundary check / value should be int, float, or bool
        return self

    def __getitem__(self, index):
        temp = self.shape
        dtype = self.dtype

        if isinstance(index, tuple):
            if isinstance(index[0], np.ndarray) and (
                (index[0].dtype is np.int64)
                or (index[0].dtype is np.int32)
                or (index[0].dtype is np.int16)
                or (index[0].dtype is np.int8)
                or (index[0].dtype is np.uint8)
            ):
                arr = LibCall.numpy.indexIntarrays(temp, len(index), index)
                arr.dtype = dtype
                return arr

            idx_len = len(index)
            if idx_len > len(self.shape):
                raise IndexError("too many indices for tensor")
            for axis in range(idx_len - 1, -1, -1):
                temp = LibCall.shape.tensorGetItem(temp, axis, index[axis])
            arr = ndarray(temp)
            arr.dtype = dtype
            return arr
        elif isinstance(index, np.ndarray) and (
            (index.dtype is np.int64)
            or (index.dtype is np.int32)
            or (index.dtype is np.int16)
            or (index.dtype is np.int8)
            or (index.dtype is np.uint8)
        ):
            arr = LibCall.numpy.indexIntarrays(temp, 1, [index])
            arr.dtype = dtype
            return arr
        elif isinstance(index, np.ndarray) and index.dtype is np.bool:
            arr = LibCall.numpy.indexBoolarray(temp, index)
            arr.dtype = dtype
            return arr

        if len(temp) <= 0:
            raise IndexError(
                "invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number"
            )

        temp = LibCall.shape.tensorGetItem(temp, 0, index)
        arr = ndarray(temp)
        arr.dtype = dtype
        return arr

    def __add__(self, other):
        return np._bop(self, other)

    def __radd__(self, other):
        return np._bop(self, other)

    def __sub__(self, other):
        return np._bop(self, other)

    def __rsub__(self, other):
        return np._bop(self, other)

    def __mul__(self, other):
        return np._bop(self, other)

    def __rmul__(self, other):
        return np._bop(self, other)

    def __truediv__(self, other):
        return np._bop(self, other)

    def __rtruediv__(self, other):
        return np._bop(self, other)

    def __floordiv__(self, other):
        return np._bop(self, other)

    def __rfloordiv__(self, other):
        return np._bop(self, other)

    def __eq__(self, other):
        return np._bop(self, other)

    def __matmul__(self, other):
        dtype = self.dtype
        arr = LibCall.numpy.matmul(self, other)
        arr.dtype = dtype
        return arr

    def __rmatmul__(self, other):
        dtype = self.dtype
        arr = LibCall.numpy.matmul(other, self)
        arr.dtype = dtype
        return arr

    def sum(
        self, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True
    ):
        return np.sum(self, axis, dtype, out, keepdims, initial, where)

    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np.max(self, axis, out, keepdims, initial, where)

    def argmax(self, axis=None, out=None):
        return np.argmax(self, axis, out)

    def flatten(self, order="C"):
        return LibCall.numpy.flatten(self)

    def copy(self, order="C"):
        return np.copy(self, order=order)

    def item(self):
        return LibCall.torch.item(self)

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        if dtype is float:
            dtype = np.floatDefault
        elif dtype is int:
            dtype = np.intDefault
        elif dtype is bool:
            dtype = np.bool

        if copy is True:
            copied = np.copy(self, order=order, subok=subok)
            copied.dtype = dtype
            return copied

        self.dtype = dtype
        return self
