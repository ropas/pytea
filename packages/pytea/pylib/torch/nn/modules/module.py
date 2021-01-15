class Module:
    def __init__(self):
        self.training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def modules(self):
        def modules_(this, moduleList):
            moduleList.append(this)
            for key, value in this.__dict__.items():
                if isinstance(value, Module):
                    moduleList = modules_(value, moduleList)
            return moduleList

        return modules_(self, [])

    def forward(self, *args, **kwargs):
        pass

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        # TODO: return something meaningful
        return []

    def cuda(self):
        return self

    def to(self, *args, **kwargs):
        # ignore it.
        return self

    def zero_grad(self, set_to_none=False):
        return None
