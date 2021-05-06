import LibCall
from . import Image


def autocontrast(image, cutoff=0, ignore=None, mask=None):
    im = image.copy()
    return im


def colorize(image, block, white, mid=None, blockpoint=0, whitepoint=255, midpoint=127):
    if image.mode is not "L":
        raise Exception('mode must be "L"')
    im = Image.Image()
    im._setSize(image.width, image.height, 3)
    im.mode = "RGB"
    return im


def equalize(image, mask=None):
    im = image.copy()
    return im


def invert(image):
    im = image.copy()
    return im


def mirror(image):
    im = image.copy()
    return im


def posterize(image, bits):
    im = image.copy()
    return im


def solarize(image, threshold=128):
    im = image.copy()
    return im
