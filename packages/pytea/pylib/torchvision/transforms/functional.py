import torch
from PIL import Image


def pad(img, padding, fill=0, padding_mode="constant"):
    return NotImplemented


def crop(img, top, left, height, width):
    if not isinstance(img, Image):
        raise TypeError("img should be PIL Image.")

    return img.crop((left, top, left + width, top + height))
