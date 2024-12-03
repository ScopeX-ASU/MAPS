"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-03-05 04:41:33
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-03-07 22:05:56
"""

import torch
import numpy as np
from pyutils.datasets import StanfordCars, OxfordFlowers, TinyImageNet, get_dataset


def stanfordcars():
    train_set = StanfordCars("./data", download=True)
    loader = torch.utils.data.DataLoader(train_set, batch_size=100, num_workers=2, shuffle=False)

    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset) * images.size(-1)))
    print(mean, std)


def oxfordflowers():
    train_set = OxfordFlowers("./data", download=True)
    loader = torch.utils.data.DataLoader(train_set, batch_size=100, num_workers=2, shuffle=False)
    print(len(loader.dataset))

    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset) * images.size(-1)))
    print(mean, std)


def tinyimagenet():
    train_set = TinyImageNet("./data", download=True)
    loader = torch.utils.data.DataLoader(train_set, batch_size=100, num_workers=2, shuffle=False)
    print(len(loader.dataset))

    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset) * images.size(-1)))
    print(mean, std)


def svhn():
    train_set, _ = get_dataset("svhn", 32, 32, "./data")
    loader = torch.utils.data.DataLoader(train_set, batch_size=100, num_workers=2, shuffle=False)
    print(len(loader.dataset))

    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1).cuda()
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1).cuda()
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset) * images.size(-1)))
    print(mean, std)


if __name__ == "__main__":
    svhn()
    # stanfordcars()
    # oxfordflowers()
    # tinyimagenet()
