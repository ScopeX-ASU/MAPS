'''
this is just a simple model to build the training flow
'''

import copy
import math
from collections import OrderedDict
from functools import lru_cache
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from pyutils.torch_train import set_torch_deterministic
from timm.models.layers import DropPath
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.types import Device

__all__ = ["simpleCNN"]

class convBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int = 3, 
            stride: int = 1, 
            padding: int = 1,
            enable_residual: bool = True,
            enable_bn: bool = True,
            enable_act: bool = True,
        ):
        super(convBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.enable_residual = enable_residual
        self.enable_bn = enable_bn
        self.enable_act = enable_act

    def forward(self, x):
        residual = x.clone()
        x = self.conv(x)
        if self.enable_bn:
            x = self.bn(x)
        if self.enable_act:
            x = self.relu(x)
        if self.enable_residual:
            x = x + residual
        return x


class simpleCNN(nn.Module):
    '''
    this model takes a two channel input and outputs a single channel output
    two channel input is the eps and the adjoint source, maybe three channel since the adjoint source is complex
    '''
    def __init__(self, in_channels: int = 3, out_channels: int = 1, device: Device = torch.device("cuda:0")):
        super(simpleCNN, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.build_layers()

    def build_layers(self):
        self.conv1 = convBlock(self.in_channels, 64, enable_residual=False)
        self.conv2 = convBlock(64, 128, enable_residual=False)
        self.conv3 = convBlock(128, 128)
        self.conv4 = convBlock(128, 128)
        self.conv5 = convBlock(128, 1, enable_act=False, enable_residual=False)

        self.predictor = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
        )

    def forward(
            self, 
            eps_map, 
            adj_src, 
        ):
        adj_src = adj_src.permute(0, 4, 2, 3, 1).squeeze(-1)
        x = torch.cat([eps_map, adj_src], dim=1)
        predict_grad = self.predictor(x)
        return predict_grad