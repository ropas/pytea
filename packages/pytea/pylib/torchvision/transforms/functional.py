import LibCall
import torch
import numpy as np
from PIL import Image


def pad(img, padding, fill=0, padding_mode="constant"):
    return NotImplemented


def crop(img, top, left, height, width):
    if not isinstance(img, Image.Image):
        raise TypeError("img should be PIL Image.")

    return img.crop((left, top, left + width, top + height))


def to_pil_image(pic, mode=None):
    """
    Image.size() -> (W, H)
    torchvision.ToTensor -> (C, H, W)
    numpy.asarray() -> (H, W, C)
    """
    shape = pic.shape
    rank = len(shape)
    assert LibCall.guard.require_lte(2, rank, "input rank should be 2 or 3")
    assert LibCall.guard.require_lte(rank, 3, "input rank should be 2 or 3")

    channel = 1

    if isinstance(pic, np.ndarray):
        if mode is None:
            if rank == 3:
                channel = shape[2]
            im = Image.Image()
            im._setSize(shape[1], shape[0], channel)
        else:
            im = Image.new(mode, (shape[1], shape[0]))

        return im
    elif isinstance(pic, torch.Tensor):
        if mode is None:
            if rank == 3:
                channel = shape[0]
            im = Image.Image()
            im._setSize(shape[2], shape[1], channel)
        else:
            im = Image.new(mode, (shape[2], shape[1]))

        return im

    raise TypeError("pic should be Tensor or ndarray")


def normalize(tensor, mean, std, inplace=False):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor should be a torch tensor.")
    meanLen = len(mean) if isinstance(mean, list) or isinstance(mean, tuple) else 1
    stdLen = len(std) if isinstance(std, list) or isinstance(std, tuple) else 1
    return LibCall.torchvision.normalize(tensor, meanLen, stdLen)

