import LibCall
from .module import Module
from .. import functional as F


class _NormBase(Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats


class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )


class BatchNorm2d(_BatchNorm):
    def forward(self, input):
        if len(input.shape) != 4:
            raise ValueError("rank must be 4")
        if input.shape[1] != self.num_features:
            raise ValueError("channel size mismatch")

        return LibCall.torch.identityShape(input)
