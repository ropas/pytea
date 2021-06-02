import LibCall

"""
Image.size() -> (W, H)
torchvision.ToTensor -> (C, H, W)
numpy.asarray() -> (H, W, C)
"""


class Image:
    def __init__(self):
        self._channel = 1
        self.width = 0
        self.height = 0
        self.size = (0, 0)

    def _setSize(self, width, height, channel):
        self._channel = channel
        self.width = width
        self.height = height
        self.size = (width, height)

    def copy(self):
        im = Image()
        im._setSize(self.width, self.height, self._channel)
        return im

    def convert(self, mode=None, *args, **kwargs):
        if mode is None:
            return self
        elif len(mode) == 1:
            im = Image()
            im._setSize(self.width, self.height, 1)
            return im
        elif mode == "RGBA" or mode == "CMYK":
            im = Image()
            im._setSize(self.width, self.height, 4)
            return im
        else:
            im = Image()
            im._setSize(self.width, self.height, 3)
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
        im._setSize(size[0], size[1], self._channel)
        return im

    def resize(self, size, resample=3, box=None, reducing_gap=None):
        im = Image()
        im._setSize(size[0], size[1], self._channel)
        return im

    def split(self):
        ret_val = [self.convert("L")]
        for _ in range(1, self._channel):
            ret_val.append(self.convert("L"))

        return tuple(ret_val)

    def save(self, fp, mode=None, **kwargs):
        pass


def new(mode, size, color=0):
    if len(mode) == 1:
        im = Image()
        im._setSize(size[0], size[1], 1)
        return im
    elif mode == "RGBA" or mode == "CMYK":
        im = Image()
        im._setSize(size[0], size[1], 4)
        return im
    else:
        im = Image()
        im._setSize(size[0], size[1], 3)
        return im


class ImageTransformHandler:
    pass


def open(fp, mode="r"):
    # TODO: image size range and target range settings in pyteaconfig.json
    im = Image()
    # make symbolic image
    im._setSize(
        LibCall.builtins.randInt(24, 4096, "PILImgW"),
        LibCall.builtins.randInt(24, 4096, "PILImgH"),
        LibCall.builtins.randInt(1, 4, "PILImgC"),
    )
    return im


def blend(im1, im2, alpha):
    LibCall.guard.require_eq(
        im1.height, im2.height, "from 'PIL.Image.blend': height mismatch"
    )
    LibCall.guard.require_eq(
        im1.width, im2.width, "from 'PIL.Image.blend': width mismatch"
    )
    LibCall.guard.require_eq(
        im1._channel, im2._channel, "from 'PIL.Image.blend': channel mismatch"
    )
    im = im1.copy()
    return im


def fromarray(obj, mode=None):
    size = LibCall.shape.extractShape(obj)
    if mode is None:
        im = Image()
        if len(size) <= 2:
            im._setSize(size[0], size[1], 1)
        else:
            im._setSize(size[0], size[1], size[2])
        return im
    else:
        return new(mode, size[0], size[1])


def save(fp, mode=None, **kwargs):
    pass


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
