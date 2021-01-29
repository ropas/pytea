from .. import torch


class ndarray:
    def __init__(
        self, shape, dtype=float, buffer=None, offset=0, strides=None, order=None
    ):
        self.shape = shape
        self.dtype = dtype

    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        pass

    def argmax(axis=None, out=None):
        pass
