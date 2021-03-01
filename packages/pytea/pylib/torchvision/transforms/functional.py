import LibCall
import torch
import numpy as np
from PIL import Image


def pad(img, padding, fill=0, padding_mode="constant"):
    return NotImplemented


def crop(img, top, left, height, width):
    if not isinstance(img, Image):
        raise TypeError("img should be PIL Image.")

    return img.crop((left, top, left + width, top + height))


def to_pil_image(pic, mode=None):
    if isinstance(pic, np.ndarray):
        return Image.fromarray(pic, mode)
    if isinstance(pic, torch.Tensor):
        im = Image.Image()
        return LibCall.torchvision.to_pil_image(im, pic, mode)

    raise TypeError("pic should be Tensor or ndarray")
