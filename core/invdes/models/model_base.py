"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-25 22:47:23
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-12-25 22:55:39
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.activation import Swish
from timm.models.layers import DropPath
from torch import nn
from torch.functional import Tensor
from torch.types import Device
from neuralop.models import FNO
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.spectral_convolution import SpectralConv
import matplotlib.pyplot as plt
from zmq import has
from core.models.layers.utils import (
    Si_eps,
    SiO2_eps,
)
import copy

# from .layers.local_fno import FNO
from .constant import *

# from .layers.fno_conv2d import FNOConv2d
import torch.nn.functional as F
from functools import lru_cache
from pyutils.torch_train import set_torch_deterministic
from core.utils import resize_to_targt_size
from core.models.fdfd.fdfd import fdfd_ez
from ceviche.constants import *
from einops import rearrange
from core.utils import print_stat

__all__ = ["LearnableFourierFeatures",
           "SpatialInterpolater",
           "ConvBlock",
           "LayerNorm",
           ]


class LearnableFourierFeatures(nn.Module):
    def __init__(self, pos_dim, f_dim, h_dim, d_dim, g_dim=1, gamma=1.0):
        super(LearnableFourierFeatures, self).__init__()
        assert (
            f_dim % 2 == 0
        ), "number of fourier feature dimensions must be divisible by 2."
        assert (
            d_dim % g_dim == 0
        ), "number of D dimension must be divisible by the number of G dimension."
        enc_f_dim = int(f_dim / 2)
        dg_dim = int(d_dim / g_dim)
        self.Wr = nn.Parameter(torch.randn([enc_f_dim, pos_dim]) * (gamma**2))
        self.mlp = nn.Sequential(
            nn.Linear(f_dim, h_dim), nn.GELU(), nn.Linear(h_dim, dg_dim)
        )
        self.div_term = np.sqrt(f_dim)

    def forward(self, pos):
        # input pos dim: (B L G M)
        # output dim: (B L D)
        # L stands for sequence length. all dimensions must be flattened to a single dimension.
        XWr = torch.matmul(pos, self.Wr.T)
        F = torch.cat([torch.cos(XWr), torch.sin(XWr)], dim=-1) / self.div_term
        Y = self.mlp(F)
        pos_enc = rearrange(Y, "b l g d -> b l (g d)")

        return pos_enc


class SpatialInterpolater(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=(2, 2), mode="bilinear", align_corners=False)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        stride: int = 1,
        ln: bool = True,
        act_func: Optional[str] = "GELU",
        device: Device = torch.device("cuda:0"),
        groups: int = 1,
        skip: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode="replicate",
            stride=stride,
            groups=groups,
        )
        if ln:
            self.ln = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        else:
            self.ln = None
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()
        self.skip = skip

    def forward(self, x: Tensor) -> Tensor:
        y = x = self.conv(x)
        if self.ln is not None:
            x = self.ln(x)
        if self.act_func is not None:
            x = self.act_func(x)
        if self.skip:
            x = x + y
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm implementation used in ConvNeXt
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape,
        dim=2,
        eps=1e-6,
        data_format="channels_last",
        reshape_last_to_first=False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first
        self.dim = dim

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.dim == 3:
                x = (
                    self.weight[:, None, None, None] * x
                    + self.bias[:, None, None, None]
                )  # add one extra dimension to match conv2d but not 2d
            elif self.dim == 2:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

