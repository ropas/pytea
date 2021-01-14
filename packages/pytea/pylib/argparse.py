import LibCall


class Namespace:
    def __init__(self):
        pass


class ArgumentParser:
    def __init__(self, *args, **kwargs):
        self.parsed = Namespace()
        self._subcommand = None
        self._valid = True

    def add_argument(self, *args, **kwargs):
        if self._valid:
            LibCall.argparse.inject_argument(self.parsed, args, kwargs)

    def parse_args(self, *args, **kwargs):
        # TODO: parse explicit argument.
        return self.parsed

    def add_subparsers(self, **kwargs):
        # set self._subcommand
        self._subcommand = LibCall.argparse.set_subcommand(self, kwargs)
        return self

    def add_parser(self, name, **kwargs):
        if self._subcommand == name:
            return self
        else:
            dummy = ArgumentParser()
            dummy._valid = False
            return dummy
