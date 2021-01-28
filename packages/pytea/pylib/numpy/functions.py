from .. import LibCall


def array(object, dtype=None, **kwargs):
    return NotImplemented


def empty(shape, dtype=float, order="C"):
    return NotImplemented


def concatenate(seq, axis=0, out=None):
    return NotImplemented


def argmax(a, axis=None, out=None):
    if axis is None:
        return 0
    else:
        result = LibCall.torch.reduce(a, axis, False)
        LibCall.torch.copyOut(result, out)
        return result


def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    if axis is None:
        return 0
    else:
        result = LibCall.torch.mean(a, axis, keepdims)
        LibCall.torch.copyOut(result, out)
        return result
