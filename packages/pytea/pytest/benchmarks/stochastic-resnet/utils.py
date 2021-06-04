import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer, required

import numpy as np
from PIL import Image

import torchvision
import torchvision.transforms as transforms
import torch.utils.data

from torch.utils.data import Dataset, DataLoader

from model import StoDepth_ResNet


############################################################
####################### For training #######################
############################################################


def make_model(
    num_layers=20, width=1, prob_0_L=(1, 0.8), dropout_prob=0.5, num_classes=10
):
    return StoDepth_ResNet(
        num_layers,
        width=width,
        prob_0_L=prob_0_L,
        dropout_prob=dropout_prob,
        num_classes=num_classes,
    )


def train_model(
    model,
    dataloader_train,
    dataloader_test,
    device,
    optimizer,
    lr_scheduler=None,
    save_path=None,
    epochs=100,
    onehot=True,
):
    if onehot:
        criterion = lambda output, target: torch.mean(
            torch.sum(-target * F.log_softmax(output, dim=1), dim=1)
        )
        to_target = lambda target: target.argmax(dim=1)
    else:
        criterion = nn.CrossEntropyLoss()
        to_target = lambda target: target

    best_test_acc = 0.0
    train_dataset_len = len(dataloader_train.dataset)
    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, target in dataloader_train:
            data, target = data.to(device), target.to(device=device, dtype=torch.long)

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * target.size(0)
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(to_target(target)).sum().item()

            print(
                "Epoch {:0>3}/{:0>3}: ({:0>5}/{:0>5}) train loss = {:.4f}, acc = {:.4f}\r".format(
                    epoch,
                    epochs,
                    train_total,
                    train_dataset_len,
                    train_loss / train_total,
                    train_correct / train_total,
                ),
                end="",
            )

        train_loss /= train_total
        train_acc = train_correct / train_total

        test_loss, test_acc = test_model(model, dataloader_test, device, onehot=False)

        if lr_scheduler is not None:
            lr_scheduler.step()

        print(
            "Epoch {:0>3}/{:0>3}: train loss = {:.4f}, acc = {:.4f}, test loss = {:.4f}, acc = {:.4f}".format(
                epoch, epochs, train_loss, train_acc, test_loss, test_acc,
            )
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if save_path is not None:
                torch.save(model.state_dict(), save_path)


class DatasetFromTeacher(torch.utils.data.Dataset):
    def __init__(
        self,
        teacher_model,
        dataset_labeled,
        dataset_unlabeled,
        transform_test,
        transform_noisy,
        device,
        num_classes=10,
        confidence_threshold=0.8,
        generated_batch_size=128,
    ):
        super(DatasetFromTeacher, self).__init__()
        self.transform_test = transform_test
        self.transform_noisy = transform_noisy
        self.data, self.label = [], []
        self.device = device
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold

        self.images_per_class, self.onehot_labels_per_class = {}, {}
        for y in range(num_classes):
            self.images_per_class[y], self.onehot_labels_per_class[y] = [], []

        # add labeled image
        for image, y in dataset_labeled:
            onehot_label = np.zeros(num_classes)
            onehot_label[y] = 1.0
            self.images_per_class[y].append(np.array(transforms.ToPILImage()(image)))
            self.onehot_labels_per_class[y].append(onehot_label)

        # add unlabeled image with pseudo-labels from the teacher model
        teacher_model.eval()
        generated_batch = []
        for image, _ in dataset_unlabeled:
            generated_batch.append(image)
            if len(generated_batch) == generated_batch_size:
                self._add_label_from_generated_batch(teacher_model, generated_batch)
                generated_batch = []

        if len(generated_batch) > 0:
            self._add_label_from_generated_batch(teacher_model, generated_batch)
            generated_batch = []

        # to make collected data into numpy array
        for y in range(num_classes):
            self.images_per_class[y] = np.array(self.images_per_class[y])
            self.onehot_labels_per_class[y] = np.array(self.onehot_labels_per_class[y])

        # dataset generation
        for y in range(num_classes):
            self.data.append(self.images_per_class[y])
            self.label.append(self.onehot_labels_per_class[y])

        self.data, self.label = np.concatenate(self.data), np.concatenate(self.label)

    def _add_label_from_generated_batch(self, teacher_model, generated_batch):
        images = np.array(
            [np.array(transforms.ToPILImage()(b)) for b in generated_batch]
        )

        input_batch = [self.transform_test(b).unsqueeze(0) for b in generated_batch]
        input_batch = torch.Tensor(torch.cat(input_batch)).to(self.device)

        with torch.no_grad():
            teacher_model.eval()
            probs = torch.softmax(teacher_model(input_batch), dim=1).cpu().numpy()

        confidences = probs.max(axis=1)
        survival_indices = confidences > self.confidence_threshold

        probs = probs[survival_indices, :]
        images = images[survival_indices, :, :, :]

        n = len(probs)

        # why 0.99? to make confidence priority to the labeled one
        mxindex = probs.argmax(axis=1)
        onehot_labels = np.zeros((n, self.num_classes))
        onehot_labels[np.arange(n), mxindex] = 0.99

        for image, onehot_label in zip(images, onehot_labels):
            y = onehot_label.argmax().item()
            self.images_per_class[y].append(image)
            self.onehot_labels_per_class[y].append(onehot_label)

    def num_images_per_label(self):
        return [len(self.images_per_class[i]) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.transform_noisy(self.data[index]),
            self.label[index],
        )


class DatasetApplyTransform(torch.utils.data.Dataset):
    def __init__(
        self, dataset_labeled, transform, num_classes=10,
    ):
        super(DatasetApplyTransform, self).__init__()
        self.transform = transform
        self.data, self.label = [], []
        self.num_classes = num_classes

        for image, y in dataset_labeled:
            self.data.append(np.array(transforms.ToPILImage()(image)))
            self.label.append(y)

        self.data, self.label = np.array(self.data), np.array(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.transform(self.data[index]),
            self.label[index],
        )


def to_onehot(label, num_classes=10):
    return torch.eye(num_classes)[label]


def test_model(model, dataloader_test, device, onehot=False):
    if onehot:
        criterion = lambda output, target: torch.mean(
            torch.sum(-target * F.log_softmax(output, dim=1), dim=1)
        )
        to_target = lambda target: target.argmax(dim=1)
    else:
        criterion = nn.CrossEntropyLoss()
        to_target = lambda target: target

    model.eval()

    test_loss = 0.0
    test_total = 0
    test_correct = 0

    for batch in dataloader_test:
        data, target = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        test_loss += loss.item() * target.size(0)
        _, predicted = output.max(1)
        test_total += target.size(0)
        test_correct += predicted.eq(to_target(target)).sum().item()

    test_loss /= test_total
    test_acc = test_correct / test_total

    return test_loss, test_acc
