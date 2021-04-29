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
trans0 = transforms.ToPILImage()
trans1 = transforms.Resize((32, 32))
trans2 = transforms.RandomHorizontalFlip()
trans3 = transforms.RandomRotation(30)
trans4 = transforms.RandomAffine(0, translate=(0.5, 0.5), shear=10, scale=(0.8, 1.2))
trans5 = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
"""
t = torch.rand(3, 32, 32)
print(t.shape)
t0 = trans0(t)
print(t0)
t0.show()
t1 = trans1(t0)
print(t1)
t1.show()
t2 = trans2(t1)
print(t2)
t2.show()
t3 = trans3(t2)
print(t3)
t3.show()
"""
from PIL import Image

im = Image.open("./cat.jfif")
print(im)
# im.show()
im = trans2(im)
print(im)
# im.show()
im = trans3(im)
print(im)
im.show()
im = trans4(im)
print(im)
im.show()
im = trans5(im)
print(im)
# im.show()
