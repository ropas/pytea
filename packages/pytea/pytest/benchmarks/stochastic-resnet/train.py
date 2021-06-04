import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import augmentations

from utils import *
from math import ceil

import argparse


def main():
    """ Basic preparation """
    parser = argparse.ArgumentParser(description="CIFAR10 noisy student ST model")
    parser.add_argument(
        "--lr", default=0.1, help="learning rate for training step", type=float
    )
    parser.add_argument(
        "--momentum", default=0.9, help="factor for training step", type=float
    )
    parser.add_argument(
        "--weight_decay", default=1e-4, help="factor for training step", type=float
    )
    parser.add_argument(
        "--batch_size", default=256, help="batch size for training", type=int
    )
    parser.add_argument(
        "--batch_size_test", default=256, help="batch size for testing", type=int
    )
    parser.add_argument(
        "--num_workers", default=4, help="number of cpu workers", type=int
    )
    parser.add_argument(
        "--randaugment_magnitude", default=27, help="magnitude of randaugment", type=int
    )
    parser.add_argument(
        "--stochastic_depth_0_prob",
        default=1.0,
        help="stochastic depth prob of the first resnet layer",
        type=float,
    )
    parser.add_argument(
        "--stochastic_depth_L_prob",
        default=0.8,
        help="stochastic depth prob of the final resnet layer",
        type=float,
    )
    parser.add_argument(
        "--dropout_prob", default=0.2, help="dropout probability for fc", type=float
    )
    parser.add_argument(
        "--device", default="auto", help="device to run the model", type=str
    )
    parser.add_argument(
        "--ratio_labeled",
        default=0.1,
        help="ratio of labeled training data",
        type=float,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.8,
        help="minimum confidence level of unlabeled data from the teacher model",
        type=float,
    )
    parser.add_argument(
        "--teacher_width", default=1, help="resnet width of teacher model", type=int
    )
    parser.add_argument(
        "--teacher_epochs", default=10, help="total epochs of teacher model", type=int,
    )
    parser.add_argument(
        "--student_width", default=1, help="resnet width of student model", type=int
    )
    parser.add_argument(
        "--student_num_learning_images",
        default=150000,
        help="the number of images required to train student",
        type=int,
    )
    parser.add_argument(
        "--only_train_teacher",
        default=False,
        help="only trains the teacher model w/o ST",
        type=bool,
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device {} has found".format(device))
    else:
        device = args.device

    """ Datasets Setting """
    labels = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    cifar10_mean, cifar10_std = [0.4913, 0.4821, 0.4465], [0.2470, 0.2434, 0.2615]

    transform_common = transforms.ToTensor()
    transform_noisy = transforms.Compose(
        [
            transforms.ToPILImage(),
            augmentations.RandAugment(2, args.randaugment_magnitude),  ### challenge
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        ]
    )

    print("Loading CIFAR10 dataset...")
    dataset_train_full = CIFAR10(
        root="./data", train=True, transform=transform_common, download=True
    )

    print("Separating labeled & unlabeled training set...")
    # torch.manual_seed(0)  # to deterministically divide splits
    num_labeled = int(args.ratio_labeled * len(dataset_train_full))
    dataset_train_labeled, dataset_train_unlabeled = torch.utils.data.random_split(
        dataset_train_full,
        [num_labeled, len(dataset_train_full) - num_labeled,],  # 50  # 450
    )
    print("> Number of labeled training images: {}".format(len(dataset_train_labeled)))
    print(
        "> Number of unlabeled training images: {}".format(len(dataset_train_unlabeled))
    )
    # torch.manual_seed(torch.initial_seed())

    print("Loading labeled test image for CIFAR10 dataset...")
    dataset_test = CIFAR10(
        root="./data", train=False, transform=transform_test, download=True
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.num_workers,
    )

    """ Teacher model preparation """

    print("Creating and begining to train the teacher model with resnet {}".format(20))

    teacher_model = make_model(
        num_layers=20,
        width=args.teacher_width,
        prob_0_L=(1.0, 1.0),
        dropout_prob=0.0,
        num_classes=10,
    ).to(device)
    teacher_model.train()

    dataset_train_teacher = DatasetApplyTransform(
        dataset_train_labeled, transform_noisy, device
    )
    dataloader_train_teacher = DataLoader(
        dataset_train_teacher,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    ys = np.eye(10)[dataset_train_teacher.label].sum(axis=0)
    for i in range(len(labels)):
        print("> Number of image {}: {}".format(labels[i].ljust(10), int(ys[i])))

    teacher_epochs = args.teacher_epochs

    optimizer = optim.SGD(
        teacher_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.5 * teacher_epochs), int(0.75 * teacher_epochs)],
        gamma=0.1,
    )

    train_model(
        teacher_model,
        dataloader_train=dataloader_train_teacher,
        dataloader_test=dataloader_test,
        device=device,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        epochs=teacher_epochs,
        onehot=False,
    )

    test_loss, test_acc = test_model(
        teacher_model, dataloader_test, device, onehot=False
    )

    print(
        "Test loss and accuarcy for the teacher: {}, {}".format(
            round(test_loss, 4), round(test_acc, 4)
        )
    )

    if args.only_train_teacher:
        exit(0)


if __name__ == "__main__":
    main()
