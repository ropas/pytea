class Module:
    def __init__(self):
        self.is_training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def modules(self):
        def modules_(this, moduleList):
            if this not in moduleList:
                moduleList.append(this)
            for key, value in this.__dict__.items():
                if isinstance(value, Module):
                    moduleList = modules_(value, moduleList)
            return moduleList

        return modules_(self, [])

    def forward(self, *args, **kwargs):
        pass

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def parameters(self):
        # TODO: return something
        return None

    def cuda(self):
        return self

