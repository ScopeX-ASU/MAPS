"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2025-10-12 00:24:38
LastEditors: Jiaqi Gu (jiaqigu@asu.edu)
LastEditTime: 2025-10-29 17:40:35
FilePath: /MAPS/core/train/models/layers/activation.py
"""

import torch
from mmengine.registry import MODELS
from torch import nn

__all__ = ["SIREN", "CustomTanh01", "ScaledTanh"]


@MODELS.register_module()
class SIREN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sin()


@MODELS.register_module()
class SinusoidActivation(nn.Module):
    def forward(self, input):
        return input.sin().mul(2)


@MODELS.register_module()
class Tanh3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.tanh(input) * 3


@MODELS.register_module()
class CustomTanh01(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        ## range is [0,1]
        ## in the center, the gradient is 1
        return (torch.tanh(2 * input) + 1) / 2


@MODELS.register_module()
class ScaledTanh(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        return torch.tanh(input) * self.scale

    def extra_repr(self):
        s = super().extra_repr()
        s += f"scale={self.scale}"
        return s


# ACTIVATION_REGISTRY = {
#     "relu": nn.ReLU(),
#     "silu": nn.SiLU(),
#     "gelu": nn.GELU(),
#     "tanh": nn.Tanh(),
#     "sigmoid": nn.Sigmoid(),
# }
