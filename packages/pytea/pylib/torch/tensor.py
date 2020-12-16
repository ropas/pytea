from .. import LibCall
import torch


class Tensor:
    def __init__(self, *args, **kwargs):
        LibCall.torch.tensorInit(self, args, kwargs)

    # # TODO: make @staticmethod
    # def __getattr__(self, attr):
    #     if attr == 'ndim':
    #         return len(self.shape)

    #     return NotImplemented

    def backward(self):
        return self

    def size(self):
        return self.shape

    def matmul(self, other):
        return LibCall.torch.matmul(self, other)

    def mul(self, other):
        return torch._bop(self, other)

    def pow(self, exponent):
        return torch._bop(self, exponent)

    def normal_(self, *args, **kwargs):
        return self

    def backward(self):
        return self

    def repeat(self, *sizes):
        return LibCall.torch.repeat(self, sizes)

    def transpose(self, dim0, dim1):
        return LibCall.torch.transpose(self, dim0, dim1)

    def argmax(self, dim=None, keepdim=False):
        return LibCall.torch.reduce(self, dim, keepdim)

    # TODO: dim can be tuple. LibCall.torch.reduce must cover it.
    def sum(self, dim=None, keepdim=False, dtype=None):
        return LibCall.torch.reduce(self, dim, keepdim)

    def item(self):
        return LibCall.torch.item(self)

    def eq(self, other):
        return torch._bop(self, other)

    def view(self, *shape):
        return LibCall.torch.view(self, shape)

    def view_as(self, other):
        return self.view(other.size())

    def unsqueeze(self, dim):
        return torch.unsqueeze(self, dim)

    def cuda(self, **kwargs):
        return self

    #TODO: Behavior of functions like to, type, long, item is dependent on dtype.
    #      They should be fixed if info of ExpShape extends.
    def to(self, *args, **kwargs):
        return self

    def type(self, dtype=None, **kwargs):
        if dtype is None:
            return 'UnknownTensorType'
        else:
            return self

    def long(self, **kwargs):
        return self.to(self, kwargs)

    def __len__(self):
        if len(self.shape) == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __getitem__(self, index):
        return LibCall.torch.getItem(self, index)

    def __add__(self, other):
        return torch._bop(self, other)

    def __radd__(self, other):
        return torch._bop(self, other)

    def __sub__(self, other):
        return torch._bop(self, other)

    def __rsub__(self, other):
        return torch._bop(self, other)

    def __mul__(self, other):
        return torch._bop(self, other)

    def __rmul__(self, other):
        return torch._bop(self, other)

    def __truediv__(self, other):
        return torch._bop(self, other)

    def __rtruediv__(self, other):
        return torch._bop(self, other)

    def __floordiv__(self, other):
        return torch._bop(self, other)

    def __rfloordiv__(self, other):
        return torch._bop(self, other)

    def __matmul__(self, other):
        return LibCall.torch.matmul(self, other)

    def __rmatmul__(self, other):
        return LibCall.torch.matmul(other, self)
