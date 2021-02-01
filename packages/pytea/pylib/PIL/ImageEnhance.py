import LibCall
from . import Image


class _Enhance:
    def enhance(self, factor):
        return self.image.copy()


class Color(_Enhance):
    def __init__(self, image):
        self.image = image


class Contrast(_Enhance):
    def __init__(self, image):
        self.image = image


class Brightness(_Enhance):
    def __init__(self, image):
        self.image = image


class Sharpness(_Enhance):
    def __init__(self, image):
        self.image = image
