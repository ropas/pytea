class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        self.optimizer = optimizer
        self.last_epoch = last_epoch

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def step(self, epoch=None):
        pass


class StepLR(_LRScheduler):
    pass


class MultiStepLR(_LRScheduler):
    pass


class LambdaLR(_LRScheduler):
    pass


class CosineAnnealingLR(_LRScheduler):
    pass
