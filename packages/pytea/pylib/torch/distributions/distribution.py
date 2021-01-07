import torch


class Distribution:
    def __init__(self, batch_shape=None, event_shape=None, validate_args=None):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
