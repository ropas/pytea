import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

dataroot = "../../../data"

transform = transforms.Compose(
    [transforms.ToTensor(), lambda x: torch.reshape(x, (784,)),]
)

# mnist = datasets.CIFAR10(root=dataroot, train=True, transform=None, download=True)
train_set, valid_set = train_test_split(
    datasets.MNIST(root=dataroot, train=True, transform=transform, download=True),
    test_size=0.1,
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=True, drop_last=True)

tdata = train_loader[0][0]
tlabel = train_loader[0][1]
vdata = valid_loader[0][0]
vlabel = valid_loader[0][1]
