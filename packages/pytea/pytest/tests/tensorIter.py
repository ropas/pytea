import torch
from torchvision import datasets
import torchvision.transforms as transforms

dataroot = "../../../data"

transform = transforms.Compose(
    [transforms.ToTensor(), lambda x: torch.reshape(x, (784,)),]
)

mnist = datasets.CIFAR10(root=dataroot, train=True, transform=transform, download=True)

t = torch.rand(7, 3, 4)
t0 = 1
arr = []
for array in t:
    t0 = array
    arr.append(1)

