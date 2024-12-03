"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 01:07:15
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 01:07:15
"""
from __future__ import print_function


from typing import Tuple

from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from .vowel import VowelRecognition
from .cars import StanfordCars
from .dogs import StanfordDogs
from .flowers import OxfordFlowers
from .tinyimagenet import TinyImageNet

__all__ = ["get_dataset"]


def get_dataset(
    dataset: str,
    img_height: int,
    img_width: int,
    dataset_dir: str = "./data",
    transform: str = "basic",
) -> Tuple[Dataset, Dataset]:
    if dataset == "mnist":
        t = []
        if (img_height, img_width) != (28, 28):
            t.append(transforms.Resize((img_height, img_width), interpolation=2))
        transform_test = transform_train = transforms.Compose(t + [transforms.ToTensor()])
        train_dataset = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform_train)

        validation_dataset = datasets.MNIST(dataset_dir, train=False, transform=transform_test)
    elif dataset == "fashionmnist":
        t = []
        if (img_height, img_width) != (28, 28):
            t.append(transforms.Resize((img_height, img_width), interpolation=2))
        transform_test = transform_train = transforms.Compose(t + [transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(
            dataset_dir, train=True, download=True, transform=transform_train
        )

        validation_dataset = datasets.FashionMNIST(dataset_dir, train=False, transform=transform_test)
    elif dataset == "cifar10":
        if transform == "basic":
            t = []
            if (img_height, img_width) != (32, 32):
                t.append(transforms.Resize((img_height, img_width), interpolation=2))
            transform_test = transform_train = transforms.Compose(t + [transforms.ToTensor()])

        else:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize((img_height, img_width), interpolation=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.Resize((img_height, img_width), interpolation=2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        train_dataset = datasets.CIFAR10(dataset_dir, train=True, download=True, transform=transform_train)

        validation_dataset = datasets.CIFAR10(dataset_dir, train=False, transform=transform_test)
    elif dataset == "cifar100":
        if transform == "basic":
            t = []
            if (img_height, img_width) != (28, 28):
                t.append(transforms.Resize((img_height, img_width), interpolation=2))
            transform_test = transform_train = transforms.Compose(t + [transforms.ToTensor()])
        else:
            CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize((img_height, img_width), interpolation=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.Resize((img_height, img_width), interpolation=2),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
                ]
            )
        train_dataset = datasets.CIFAR100(dataset_dir, train=True, download=True, transform=transform_train)

        validation_dataset = datasets.CIFAR100(dataset_dir, train=False, transform=transform_test)
    elif dataset == "svhn":
        if transform == "basic":
            t = []
            if (img_height, img_width) != (28, 28):
                t.append(transforms.Resize((img_height, img_width), interpolation=2))
            transform_test = transform_train = transforms.Compose(t + [transforms.ToTensor()])

        else:
            SVHN_TRAIN_MEAN = (0.4377, 0.4438, 0.4728)
            SVHN_TRAIN_STD = (0.1980, 0.2010, 0.1970)
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize((img_height, img_width), interpolation=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.Resize((img_height, img_width), interpolation=2),
                    transforms.ToTensor(),
                    transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD),
                ]
            )
        train_dataset = datasets.SVHN(dataset_dir, split="train", download=True, transform=transform_train)

        validation_dataset = datasets.SVHN(dataset_dir, split="test", download=True, transform=transform_test)
    elif dataset == "dogs":
        # this is imagenet-style transform
        # input_transforms = transforms.Compose([
        # transforms.RandomResizedCrop((img_height, img_width), ratio=(1, 1.3)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor()])

        # this is blueprint conv style transform [CVPR 2020]
        DOGS_TRAIN_MEAN = (0.485, 0.456, 0.406)
        DOGS_TRAIN_STD = (0.229, 0.224, 0.225)
        transform_train = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.RandomCrop((img_height, img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4),
                transforms.ToTensor(),
                transforms.Normalize(DOGS_TRAIN_MEAN, DOGS_TRAIN_STD),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize(DOGS_TRAIN_MEAN, DOGS_TRAIN_STD),
            ]
        )

        train_dataset = StanfordDogs(dataset_dir, train=True, download=True, transform=transform_train)

        validation_dataset = StanfordDogs(dataset_dir, train=False, download=True, transform=transform_test)

    elif dataset == "cars":
        # this is imagenet-style transform
        # input_transforms = transforms.Compose([
        # transforms.RandomResizedCrop((img_height, img_width), ratio=(1, 1.3)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor()])

        # this is blueprint conv style transform [CVPR 2020]
        CARS_TRAIN_MEAN = (0.4707, 0.4602, 0.4550)
        CARS_TRAIN_STD = (0.2899, 0.2890, 0.2975)
        transform_train = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.RandomCrop((img_height, img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4),
                transforms.ToTensor(),
                transforms.Normalize(CARS_TRAIN_MEAN, CARS_TRAIN_STD),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize(CARS_TRAIN_MEAN, CARS_TRAIN_STD),
            ]
        )

        train_dataset = StanfordCars(dataset_dir, train=True, download=True, transform=transform_train)

        validation_dataset = StanfordCars(dataset_dir, train=False, download=True, transform=transform_test)

    elif dataset == "flowers":
        FLOWERS_TRAIN_MEAN = (0.4330, 0.3819, 0.2964)
        FLOWERS_TRAIN_STD = (0.2929, 0.2445, 0.2718)
        if transform == "basic":
            transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop((img_height, img_width), ratio=(1, 1.3)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.RandomResizedCrop((img_height, img_width), ratio=(1, 1.3)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )

        else:
            transform_train = transforms.Compose(
                [
                    transforms.Resize(size=(256, 256)),
                    transforms.RandomCrop((img_height, img_width)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(FLOWERS_TRAIN_MEAN, FLOWERS_TRAIN_STD),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize(size=(256, 256)),
                    transforms.CenterCrop((img_height, img_width)),
                    transforms.ToTensor(),
                    transforms.Normalize(FLOWERS_TRAIN_MEAN, FLOWERS_TRAIN_STD),
                ]
            )
        train_dataset = OxfordFlowers(dataset_dir, train=True, download=True, transform=transform_train)

        validation_dataset = OxfordFlowers(dataset_dir, train=False, download=True, transform=transform_test)

    elif dataset == "tinyimagenet":
        TINY_TRAIN_MEAN = (0.4802, 0.4481, 0.3975)
        TINY_TRAIN_STD = (0.2770, 0.2691, 0.2821)
        if transform == "basic":
            transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop((img_height, img_width), ratio=(1, 1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.RandomResizedCrop((img_height, img_width), ratio=(1, 1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )

        else:
            transform_train = transforms.Compose(
                [
                    transforms.Resize(size=(64, 64)),
                    transforms.RandomCrop((img_height, img_width)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(TINY_TRAIN_MEAN, TINY_TRAIN_STD),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize(size=(64, 64)),
                    transforms.CenterCrop((img_height, img_width)),
                    transforms.ToTensor(),
                    transforms.Normalize(TINY_TRAIN_MEAN, TINY_TRAIN_STD),
                ]
            )
        train_dataset = TinyImageNet(dataset_dir, train=True, download=True, transform=transform_train)

        validation_dataset = TinyImageNet(dataset_dir, train=False, download=True, transform=transform_test)

    elif "vowel" in dataset:
        ## vowel_4_4: 4 features and 4 labels
        n_features, n_labels = [int(i) for i in dataset[5:].split("_")]
        train_dataset = VowelRecognition(
            root=dataset_dir,
            train=True,
            transform=None,
            target_transform=None,
            n_features=n_features,
            n_labels=n_labels,
            train_ratio=0.7,
            download=True,
        )
        validation_dataset = VowelRecognition(
            root=dataset_dir,
            train=False,
            transform=None,
            target_transform=None,
            n_features=n_features,
            n_labels=n_labels,
            train_ratio=0.7,
            download=True,
        )
    else:
        raise NotImplementedError

    return train_dataset, validation_dataset
