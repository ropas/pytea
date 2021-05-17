from ... import LibCall


class Module:
    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def modules(self):
        def modules_(this, moduleList):
            moduleList.append(this)
            for value in this.__dict__.values():
                if isinstance(value, Module):
                    moduleList = modules_(value, moduleList)
            return moduleList

        return modules_(self, [])

    def forward(self, *args, **kwargs):
        pass

    def train(self, mode=True):
        if self.training != mode:
            self.training = mode
        # for module in self.modules():
        #    module.train()
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        # TODO: return something meaningful
        return []

    def cuda(self):
        return self

    def cpu(self):
        return self

    def to(self, *args, **kwargs):
        # ignore it.
        return self

    def zero_grad(self, set_to_none=False):
        return None

    def register_buffer(self, name, tensor, persistent=True):
        setattr(self, name, tensor)

    def register_parameter(self, name, param):
        setattr(self, name, param)

    def state_dict(self, destination=None, prefix=None, keep_vars=None):
        # TODO: make key-value pair
        return {}

    def share_memory(self):
        return self

    def apply(self, fn):
        # TODO: recursively add apply fn in children
        return self

    def float(self):
        # TODO: make float recursively
        return self

    def double(self):
        return self

    def half(self):
        return self

    def bfloat16(self):
        return self

    def load_state_dict(state_dict, strict=True):
        # TODO: generally, we cannot infer state_dict statically.
        # this means we had no choice but to ignore the state_dict
        pass
