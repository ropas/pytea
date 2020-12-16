import LibCall


class Namespace:
    def __init__(self):
        pass


class ArgumentParser:
    def __init__(self, *args, **kwargs):
        self.parsed = Namespace()

    def add_argument(self, *args, **kwargs):
        LibCall.argparse.inject_argument(self.parsed, args, kwargs)

    def parse_args(self, *args, **kwargs):
        # TODO: parse explicit argument.
        return self.parsed
