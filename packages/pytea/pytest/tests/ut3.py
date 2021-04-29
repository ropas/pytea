import torch
from torchvision import datasets, transforms

"""
transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ])
trainset = datasets.MNIST('~/torchData', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
t = trainloader[0]
"""

trans = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
t = torch.rand(3,26)
print(t.shape)
t2 = trans(t)
print(t2.shape)