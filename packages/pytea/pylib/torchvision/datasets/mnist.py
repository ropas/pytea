import torch
import torch.utils.data as data
import random
from PIL import Image


class MNIST(data.Dataset):
    def __init__(
        self,
        root,
        train=True,
        transforms=None,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(MNIST, self).__init__()
        self.root = root
        self.transform = transform
        self.transforms = transforms
        self.target_transform = target_transform

        if train:
            self._len = 60000
        else:
            self._len = 10000

    def __getitem__(self, index):
        img = Image.Image()
        img._setSize(1, 28, 28)
        target = LibCall.builtins.randInt(0, 9, "MNIST_Class")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        # box constant value to prevent constant iteration by for-loop
        # to prevent boxing, set 'boxDataLoader' option to false in 'pyteaconfig.json'
        return LibCall.builtins.box(self._len)
