import LibCall
from . import Image


class ImageDraw:
    def __init__(self, im, mode=None):
        pass

    def arc(xy, start, end, fill=None, width=0):
        pass

    def bitmap(xy, bitmap, fill=None):
        pass

    def ellipse(xy, fill=None, outline=None, width=1):
        pass

    def line(xy, fill=None, width=0, joint=None):
        pass

    def pieslice(xy, start, end, fill=None, outline=None, width=1):
        pass

    def point(xy, fill=None):
        pass

    def polygon(xy, fill=None, outline=None):
        pass

    def regular_polygon(bounding_circle, n_sides, rotation=0, fill=None, outline=None):
        pass

    def rectangle(xy, fill=None, outline=None, width=1):
        pass


def Draw(im, mode=None):
    return ImageDraw(im, mode)
