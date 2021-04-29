import numpy as np
from .ndarray import ndarray
import LibCall


def seed(seed=None):
    pass


def rand(*shape):
    if isinstance(shape, int):
        shape = [shape]
    return ndarray(shape, dtype=float, order=None)


# random integer [low, high)
def randint(low, high=None, size=None, dtype=int):
    if high is None:
        a = 0
        b = low
    else:
        a = low
        b = high

    if size is None:
        # range of randInt(a, b) = [a, b] (not [a, b))
        return LibCall.builtins.randInt(a, b - 1, "np_randint")
    if isinstance(size, int):
        size = [size]
    # TODO: Set range of each dim
    return ndarray(size, dtype=dtype, order=None)
