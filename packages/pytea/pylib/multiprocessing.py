def set_start_method(method):
    pass


class Process:
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, daemon=None
    ):
        self.group = group
        self.target = target
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.daemon = daemon

    def start(self):
        self.target(*self.args, **self.kwargs)

    def join(self):
        pass
