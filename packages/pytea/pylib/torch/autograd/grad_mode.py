class no_grad:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return True

    def __call__(self, func):
        return self


class enable_grad:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return True

    def __call__(self, func):
        return self

