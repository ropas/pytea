import LibCall
from torch import Tensor
from PIL import Image


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
            return Tensor(pic)
        elif isinstance(pic, Tensor):
            return pic
        else:
            return LibCall.torch.warnTensorWithMsg("pic should be PIL Image or ndarray")


class Normalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return tensor


class RandomResizedCrop:
    def __init__(
        self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=2
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        if isinstance(img, Image.Image):
            image = Image.Image()
            image.mode = img.mode
            image._setSize(img._channel, self.size[0], self.size[1])
            return image
        else:
            return LibCall.torchvision.crop(img, self.size[0], self.size[1])
