import torch
import torch.utils.data as data
import random
from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=None,
        is_valid_file=None,
    ):
        super(ImageFolder, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.len = LibCall.builtins.randInt(1000, 2000, "ImageFolder_Length")

    def __getitem__(self, index):
        img = Image.open("RANDOM_IMAGE").convert("RGB")
        target = LibCall.builtins.randInt(0, 9, "ImageFolder_Class")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.len
