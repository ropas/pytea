import LibCall
from .module import Module
from ...tensor import Tensor


class _InstanceNorm(Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
    ):
        super(_InstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentup = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        # simple tensor copy
        return LibCall.torch.identityShape(input)


class InstanceNorm2d(_InstanceNorm):
    pass
