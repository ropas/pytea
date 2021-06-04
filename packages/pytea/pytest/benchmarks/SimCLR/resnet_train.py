import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms


TOTAL_EPOCHS = 200
BATCH_SIZE = 256
TEMPERATURE = 0.07


# same to torchvision.models.resnet.conv3x3; to minimize import
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride

        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        residual = x
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        if self.downsample is not None:
            x = self.downsample(x)
        return x + residual


class Downsample(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        # return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)
        return torch.cat((x, x.mul(0).repeat(1, self.expand_ratio - 1, 1, 1)), 1)


class ResNetCifar(nn.Module):
    def __init__(
        self, depth, width=1, classes=10, channels=3, norm_layer=nn.BatchNorm2d
    ):
        assert (depth - 2) % 6 == 0  # depth is 6N+2
        self.N = (depth - 2) // 6
        super(ResNetCifar, self).__init__()

        # Following the Wide ResNet convention, we fix the very first convolution
        self.conv1 = nn.Conv2d(
            channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.inplanes = 16
        self.layer1 = self._make_layer(norm_layer, 16 * width)
        self.layer2 = self._make_layer(norm_layer, 32 * width, stride=2)
        self.layer3 = self._make_layer(norm_layer, 64 * width, stride=2)
        self.bn = norm_layer(64 * width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, torch.sqrt(torch.Tensor([2.0 / n])).item())

    def _make_layer(self, norm_layer, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Downsample(self.inplanes, planes, stride)
        layers = [BasicBlock(self.inplanes, planes, norm_layer, stride, downsample)]
        self.inplanes = planes
        # list append libcall should be implemented
        for i in range(self.N - 1):
            layers.append(BasicBlock(self.inplanes, planes, norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class SimCLRHead(nn.Module):
    def __init__(self, width, emb_dim):
        super(SimCLRHead, self).__init__()

        self.fc1 = nn.Linear(64 * width, 64 * width)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64 * width, emb_dim)
        self.norm = Normalize()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x


class SimCLRNet(nn.Module):
    def __init__(self, depth, width=1, num_classes=10, emb_dim=32):
        super(SimCLRNet, self).__init__()

        self.num_classes = num_classes

        self.feat = ResNetCifar(depth=depth, width=width, classes=num_classes)
        self.head = SimCLRHead(width, emb_dim)
        self.classifier = nn.Linear(64 * width, num_classes)
        self.norm = Normalize()

    def forward(self, x, norm_feat=False):
        feat = self.feat(x)
        if norm_feat:
            feat = self.norm(feat)

        emb = self.head(feat)
        logit = self.classifier(feat)

        return feat, emb, logit


class DuplicatedCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        for t in self.transforms:
            img1 = t(img1)
            img2 = t(img2)
        return img1, img2


class NTXentLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self):
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        return self._cosine_simililarity

    def _get_correlated_mask(self):
        diag = torch.eye(2 * self.batch_size)
        l1 = torch.diag(torch.ones(self.batch_size), -self.batch_size)
        l2 = torch.diag(torch.ones(self.batch_size), self.batch_size)
        mask = diag + l1 + l2
        mask = (1 - mask).type(torch.bool)
        return mask.cuda()

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        # challenging part: tensor shape's requirement condition depends on the tensor data
        # negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
        #     2 * self.batch_size, -1
        # )
        negatives = torch.rand(
            2 * self.batch_size,
            2 * (self.batch_size - 1),
            dtype=similarity_matrix.dtype,
        )
        logits = torch.cat((positives, negatives), dim=1)

        logits = logits / self.temperature

        labels = torch.zeros(2 * self.batch_size).cuda().long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


def train(net, loader):
    losses = []
    loss_fn = NTXentLoss(batch_size=BATCH_SIZE, temperature=TEMPERATURE)

    optimizer = optim.Adam(net.parameters(), lr=0.001 * BATCH_SIZE / 256)

    for epoch in range(1, TOTAL_EPOCHS + 1):
        train_loss = 0
        net.train()

        for idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()

            xi, xj, target = data[0].cuda(), data[1].cuda(), target.cuda()

            _, zis, _ = net(xi)
            _, zjs, _ = net(xj)

            loss = loss_fn(zis, zjs)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= idx + 1
        losses.append(train_loss)
        print("Epoch\t", epoch, "\tLoss\t", train_loss)

    print("Finished training.")
    return losses


### Main function of this script
def main():
    global BATCH_SIZE
    global TOTAL_EPOCHS
    global TEMPERATURE
    # data loader
    img_size = (32, 32)

    train_transform = DuplicatedCompose(
        [transforms.RandomResizedCrop(img_size), transforms.ToTensor(),]
    )

    train_dataset = datasets.CIFAR10(
        root=".", train=True, download=True, transform=train_transform
    )

    # hparams settings: list of (BATCH_SIZE, TOTAL_EPOCHS, TEMPERATURE) tuples.
    hparams_settings = ((256, 1, 0.07), (64, 1, 0.07))

    for B, E, T in hparams_settings:
        # change global variables.
        # error was caused since I didn't change BATCH_SIZE for the train_loader,
        # so the tensor size for NTXentLoss mismatched.
        BATCH_SIZE = B
        TOTAL_EPOCHS = E
        TEMPERATURE = T

        # train_loader should be updated after changing the BATCH_SIZE
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            shuffle=True,
            # drop_last=True,  # *ERROR*: drop_last=True is essential
        )

        net = SimCLRNet(26, 1, 10, 32)
        net.cuda()

        print("========================================")
        print("Pre-training with {},{},{} begins".format(B, E, T))
        print("========================================")
        losses = train(net, train_loader)

        del net


if __name__ == "__main__":
    main()
