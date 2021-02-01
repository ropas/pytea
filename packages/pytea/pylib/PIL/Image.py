import LibCall
import random
import numpy as np


class Image:
    def __init__(self):
        self._channel = 1
        self.size = (0, 0)
        self.height = 0
        self.width = 0

    def _setSize(self, channel, width, height):
        self._channel = channel
        self.width = width
        self.height = height
        self.size = (width, height)
        LibCall.builtins.setSize(self, (channel, width, height))

    def copy(self):
        im = Image()
        im._setSize(self._channel, self.width, self.height)
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

    def transform(self, size, method, data=None, resample=0, fill=1, fillcolor=None):
        if (
            method is not EXTENT
            and method is not AFFINE
            and method is not PERSPECTIVE
            and method is not QUAD
            and method is not MESH
            and not isinstance(method, ImageTransformHandler)
            and not hasattr(method, "getdata")
        ):
            raise Exception("unknown method type")

        im = Image()
        im._setSize(self._channel, size[0], size[1])
        return im

    def resize(self, size, resample=3, box=None, reducing_gap=None):
        im = Image()
        im._setSize(im._channel, size[0], size[1])
        return im


def new(mode, size, color=0):
    if len(mode) == 1:
        im = Image()
        im._setSize(1, size[0], size[1])
        return im
    elif mode == "RGBA" or mode == "CMYK":
        im = Image()
        im._setSize(4, size[0], size[1])
        return im
    else:
        im = Image()
        im._setSize(3, size[0], size[1])
        return im


class ImageTransformHandler:
    pass


def open(fp, mode="r"):
    # TODO: image size range and target range settings in pyteaconfig.json
    im = Image()
    # make symbolic image
    im._setSize(
        LibCall.builtins.randInt(1, 4, 'PILImgC'),
        LibCall.builtins.randInt(24, 4096, 'PILImgW'),
        LibCall.builtins.randInt(24, 4096, 'PILImgH')
    )
    return im


def blend(im1, im2, alpha):
    LibCall.PIL.blend(im1, im2, alpha)  # just adds constraints, doesn't return obj.
    im = im1.copy()
    return im


def fromarray(obj, mode=None):
    if isinstance(obj, np.ndarray):
        im = Image()
        return LibCall.PIL.fromarray(im, obj, mode)

    return NotImplemented


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

# transforms
AFFINE = 0
EXTENT = 1
PERSPECTIVE = 2
QUAD = 3
MESH = 4
