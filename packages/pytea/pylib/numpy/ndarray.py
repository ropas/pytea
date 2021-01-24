import LibCall

class ndarray:
    def __init__(
        self, shape, dtype=float, buffer=None, offset=0, strides=None, order=None
    ):
        LibCall.numpy.ndarrayInit(self, shape, dtype, buffer, offset, strides, order)
        self.dtype = dtype

    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        pass

    def argmax(axis=None, out=None):
        pass
