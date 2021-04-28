import torch
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


dataroot = "."

transform = transforms.Compose(
    [transforms.ToTensor(), lambda x: torch.reshape(x, (784,)),]
)

cfar = datasets.CIFAR10(root=dataroot, train=True, transform=None, download=True)
train_set, valid_set = train_test_split(
    datasets.MNIST(root=dataroot, train=True, transform=transform, download=True),
    test_size=0.1,
)
print(len(train_set))
print(len(valid_set))
print(train_set[0])
print(valid_set[0])

"""
print(cfar)
print("----------------")
arr = []
for array in cfar:
    print(array)
    print(len(array))
    print(array[1])
    arr.append(1)
    break
"""
