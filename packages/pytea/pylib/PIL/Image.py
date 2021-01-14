import LibCall
import random


class Image:
    def __init__(self):
        self._channel = 1
        self.size = (0, 0)
        self.height = 0
        self.width = 0
        self.mode = "L"

    def _setSize(self, channel, width, height):
        self._channel = channel
        self.width = width
        self.height = height
        LibCall.builtins.setSize(self, (channel, width, height))

    def copy(self):
        im = Image()
        im._setSize(self._channel, self.width, self.height)
        im.mode = self.mode
        return im

    def convert(self, mode=None, *args, **kwargs):
        if mode is None:
            return self
        elif len(mode) == 1:
            im = Image()
            im._setSize(1, self.width, self.height)
            return im
        elif mode == "RGBA" or mode == "CMYK":
            im = Image()
            im._setSize(4, self.width, self.height)
            return im
        else:
            im = Image()
            im._setSize(3, self.width, self.height)
            return im


def open(fp, mode="r"):
    im = Image()
    # make symbolic image
    im._setSize(
        random.randint(1, 4), random.randint(1, 10000), random.randint(1, 10000)
    )
    return im


NEAREST = 0
NONE = 0
BOX = 4
BILINEAR = 2
LINEAR = 2
HAMMING = 5
BICUBIC = 3
CUBIC = 3
LANCZOS = 1
ANTIALIAS = 1
AFFINE = 0

