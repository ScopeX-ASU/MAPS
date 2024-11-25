'''
Date: 2024-08-24 21:37:48
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-08-24 21:40:12
FilePath: /Metasurface-Opt/core/models/activation.py
'''
from torch import nn

ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}
