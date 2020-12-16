import LibCall
from .module import Module
from .. import functional as F

class ReLU(Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class Softmax(Module):
    def __init__(self, dim=None):
        super(Softmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        return F.softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)