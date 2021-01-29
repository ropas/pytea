class Optimizer:
    def __init__(self, params, *args, **kwargs):
        self.param_groups = []
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def zero_grad(self):
        pass

    def cuda(self):
        pass

    def cpu(self):
        pass

    def add_param_group(self):
        pass


class _RequiredParameter:
    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter
