"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-20 16:58:26
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-20 16:58:26
"""

from torch import nn
import torch
from mmengine.registry import MODELS

__all__ = ["SIREN"]


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

# ACTIVATION_REGISTRY = {
#     "relu": nn.ReLU(),
#     "silu": nn.SiLU(),
#     "gelu": nn.GELU(),
#     "tanh": nn.Tanh(),
#     "sigmoid": nn.Sigmoid(),
# }