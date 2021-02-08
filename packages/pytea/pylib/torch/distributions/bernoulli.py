import torch
from .distribution import Distribution


class Bernoulli(Distribution):
    def __init__(self, probs=None, logits=None, validate_args=None):
        if probs is not None:
            is_scalar = isinstance(probs, float) or isinstance(probs, int)
            self._param = probs
        elif logits is not None:
            is_scalar = isinstance(logits, float) or isinstance(logits, int)
            self._param = logits
        else:
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")

        if is_scalar:
            empty_tensor = torch.Tensor()
            batch_shape = empty_tensor.shape
        else:
            batch_shape = self._param.shape
        self._batch_shape = batch_shape
        
        super(Bernoulli, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape):
        return torch.random(self._batch_shape)