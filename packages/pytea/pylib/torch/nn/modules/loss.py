import LibCall
from .module import Module
from .. import functional as F
from .. import _reduction as _Reduction

class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        # self.register_buffer('weight', weight)

class CrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return F.cross_entropy(input, target, None,
                               ignore_index=self.ignore_index, reduction=self.reduction)

class NLLLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(NLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return F.nll_loss(input, target, None,
                               ignore_index=self.ignore_index, reduction=self.reduction)

class MSELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction)

class BCELoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(BCELoss, self).__init__(size_average, reduce, reduction)
        self.weight = weight

    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, self.weight, self.size_average, self.reduce, self.reduction)