import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


# hparams settings: list of (BATCH_SIZE, TOTAL_EPOCHS, TEMPERATURE) tuples.
hparams_settings = ((256, 1, 0.07), (64, 1, 0.07))

X1 = 0
X2 = 0
X3 = 0
X4 = 0
i = 0

for B, E, T in hparams_settings:
    # change global variables.
    # error was caused since I didn't change BATCH_SIZE for the train_loader,
    # so the tensor size for NTXentLoss mismatched.
    BATCH_SIZE = B
    train_transform = torchvision.transforms.Compose(
        [transforms.RandomResizedCrop((32, 32)), transforms.ToTensor(),]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=".", train=True, download=True, transform=train_transform
    )
    # train_loader should be updated after changing the BATCH_SIZE
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        num_workers=4,
        shuffle=True,
        drop_last=True,  # *ERROR*: drop_last=True is essential
    )

    for p, k in train_loader:
        if i == 0:
            X1 = p
        elif i == 1:
            X2 = p
        elif i == 2:
            X3 = p
        else:
            X4 = p
        i += 1
