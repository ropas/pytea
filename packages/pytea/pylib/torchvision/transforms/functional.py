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


def normalize(tensor, mean, std, inplace=False):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor should be a torch tensor.")
    meanLen = len(mean) if isinstance(mean, list) or isinstance(mean, tuple) else 1
    stdLen = len(std) if isinstance(std, list) or isinstance(std, tuple) else 1
    return LibCall.torchvision.normalize(tensor, meanLen, stdLen)

