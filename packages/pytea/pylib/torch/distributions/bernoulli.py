import LibCall
import torch
from .distribution import Distribution


class Bernoulli(Distribution):
    def __init__(self, probs=None, logits=None, validate_args=None):
        if probs is not None:
            self.is_scalar = isinstance(probs, float) or isinstance(probs, int)
            self._param = probs
        elif logits is not None:
            self.is_scalar = isinstance(logits, float) or isinstance(logits, int)
            self._param = logits
        else:
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

        if self.is_scalar:
            empty_tensor = torch.Tensor()
            batch_shape = empty_tensor.shape
        else:
            batch_shape = self._param.shape
        self._batch_shape = batch_shape

        super(Bernoulli, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape):
        if self.is_scalar:
            sample = LibCall.builtins.randFloat(0, 1, "ThBer")
            if sample > self._param:
                return Sample(1.0)
            else:
                return Sample(0.0)

        return torch.rand(self._batch_shape)


# temporary class to be used like Bernoulli(p).sample().item()
# in reality, Bernoulli(p).sample() returns tensor.
class Sample:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value
