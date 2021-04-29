import torch
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


dataroot = "~/torchData"

transform = transforms.Compose(
    [transforms.ToTensor(), lambda x: torch.reshape(x, (784,)),]
)

cfar = datasets.MNIST(root=dataroot, train=True, transform=transform, download=True)
mn1 = cfar
mn2 = cfar
mn1._len = 54000
mn2._len = 6000
print("cfar:")
print(type(cfar))
print(cfar)
train_set, valid_set = train_test_split(
    cfar,
    test_size=0.1,
)
print("type of train_set: ", type(train_set))
print("type of valid_set: ", type(valid_set))
#print(train_set)
#print(len(train_set))
#print(len(valid_set))
#print(train_set[0])
#print(valid_set[0])

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
