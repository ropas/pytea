import LibCall
import torch
from torch import Tensor
from PIL import Image
import numpy as np
from . import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor:
    def __call__(self, pic):
        # TODO: add numpy
        if isinstance(pic, Image.Image):
            return Tensor(pic._channel, pic.height, pic.width)
        elif isinstance(pic, Tensor):
            return pic
        else:
            return LibCall.torch.warnTensorWithMsg("pic should be PIL Image or ndarray")


class ToPILImage:
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        return F.to_pil_image(pic, self.mode)


class Normalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std)


class RandomCrop:
    def __init__(
        self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"
    ):
        if isinstance(size, tuple) or isinstance(size, list):
            self.size = size
        else:
            self.size = (int(size), int(size))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        if isinstance(img, Image.Image):
            # TODO: require le size
            image = Image.Image()
            image._setSize(self.size[1], self.size[0], img._channel)
            return image
        else:
            return LibCall.torchvision.crop(img, self.size[0], self.size[1])


class RandomResizedCrop:
    def __init__(
        self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=2
    ):
        if isinstance(size, tuple) or isinstance(size, list):
            self.size = size
        else:
            self.size = (int(size), int(size))

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        if isinstance(img, Image.Image):
            image = Image.Image()
            image._setSize(self.size[1], self.size[0], img._channel)
            return image
        else:
            return LibCall.torchvision.crop(img, self.size[0], self.size[1])


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, tuple) or isinstance(size, list):
            self.size = size
        else:
            self.size = (int(size), int(size))

    def __call__(self, img):
        if isinstance(img, Image.Image):
            image = Image.Image()
            image._setSize(self.size[1], self.size[0], img._channel)
            return image
        else:
            return LibCall.torchvision.crop(img, self.size[0], self.size[1])


class Resize:
    def __init__(self, size, interpolation=2):
        if isinstance(size, tuple) or isinstance(size, list):
            self.size = size
        else:
            self.size = (int(size), int(size))

        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(img, Image.Image):
            image = Image.Image()
            image._setSize(self.size[1], self.size[0], img._channel)
            return image
        else:
            return LibCall.torchvision.crop(img, self.size[0], self.size[1])


class Lambda:
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        pass

    def forward(self, img):
        if isinstance(img, Image.Image) or isinstance(img, Tensor):
            return img
        raise TypeError("img should be PIL Image or Tensor")


class RandomRotation(torch.nn.Module):
    def __init__(
        self,
        degrees,
        interpolation=None,
        expand=False,
        center=None,
        fill=0,
        resample=None,
    ):
        pass

    def forward(self, img):
        # TODO: when expand=True: (C, H, W) -> (C, ?, ?)
        if isinstance(img, Image.Image) or isinstance(img, Tensor):
            return img
        raise TypeError("img should be PIL Image or Tensor")


class RandomAffine(torch.nn.Module):
    def __init__(
        self,
        degrees,
        translate=None,
        scale=None,
        shear=None,
        interpolation=None,
        fill=0,
        fillcolor=None,
        resample=None,
    ):
        pass

    def forward(self, img):
        if isinstance(img, Image.Image) or isinstance(img, Tensor):
            return img
        raise TypeError("img should be PIL Image or Tensor")


class ColorJitter(torch.nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        pass

    def forward(self, img):
        if isinstance(img, Image.Image) or isinstance(img, Tensor):
            return img
        raise TypeError("img should be PIL Image or Tensor")
