from .. import LibCall
import torch

class Tensor:
    def __init__(self, *args, **kwargs):
        LibCall.torch.tensorInit(self, args, kwargs)
        self.data = self

    # # TODO: make @staticmethod
    def __getattr__(self, attr):
        if attr == 'ndim':
            return len(self.shape)

        return NotImplemented

    def backward(self):
        return self

    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def matmul(self, other):
        return torch.matmul(self, other)

    def add(self, other):
        return torch._bop(self, other)

    def add_(self, other):
        torch._bop(self, other)
        return self

    def sub(self, other):
        return torch._bop(self, other)

    def sub_(self, other):
        torch._bop(self, other)
        return self

    def mul(self, other):
        return torch._bop(self, other)

    def mul_(self, other):
        torch._bop(self, other)
        return self

    def div(self, other):
        return torch._bop(self, other)

    def div_(self, other):
        torch._bop(self, other)
        return self

    def pow(self, exponent):
        return torch._bop(self, exponent)

    def pow_(self, other):
        torch._bop(self, other)
        return self

    def normal_(self, *args, **kwargs):
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

    def mm(self, mat2):
        return torch.mm(self, mat2)

    def bmm(self, batch2):
        return torch.bmm(self, batch2)

    # TODO: Behavior of functions like to, type, long, item is dependent on dtype.
    #      They should be fixed if info of ExpShape extends.
    def to(self, *args, **kwargs):
        return self

    def type(self, dtype=None, **kwargs):
        if dtype is None:
            # TODO: return dtype of self
            return "UnknownTensorType"
        else:
            return self

    def long(self, **kwargs):
        return self.to(self, kwargs)

    def __len__(self):
        if len(self.shape) == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __getitem__(self, index):
        temp = self.shape

        if isinstance(index, tuple):
            idx_len = len(index)
            if len(self.shape) > idx_len:
                raise IndexError("too many indices for tensor")
            for axis in range(idx_len - 1, -1, -1):
                temp = LibCall.shape.tensorGetItem(temp, axis, index[axis])
            return Tensor(temp)

        if len(temp) <= 0:
            raise IndexError(
                "invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number"
            )

        temp = LibCall.shape.tensorGetItem(temp, 0, index)
        return Tensor(temp)

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
