class Module:
    def __init__(self):
        self.training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        pass

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        # TODO: return something
        return None

    def cuda(self):
        return self

