class Module:
    def __init__(self):
        self.is_training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

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

