"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-10 01:25:27
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-10 01:37:06
"""


from turtle import pos
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
from neuralop.layers.mlp import MLP
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.spectral_convolution import SpectralConv
import matplotlib.pyplot as plt
from core.utils import (
    Si_eps,
    SiO2_eps,
)
import copy
from mmengine.registry import MODELS

import torch.nn.functional as F
from functools import lru_cache
from pyutils.torch_train import set_torch_deterministic
from core.train.utils import resize_to_targt_size
from core.fdfd.fdfd import fdfd_ez
from thirdparty.ceviche.ceviche.constants import *
from einops import rearrange
from .model_base import ModelBase, ConvBlock, LinearBlock
from .fno_cnn import LearnableFourierFeatures

__all__ = ["UNet"]


def double_conv(in_channels, hidden, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
    )

@MODELS.register_module()
class UNet(ModelBase):
    """
    Frequency-domain scattered electric field envelop predictor
    Assumption:
    (1) TE10 mode, i.e., Ey(r, omega) = Ez(r, omega) = 0
    (2) Fixed wavelength. wavelength currently not being modeled
    (3) Only predict Ex_scatter(r, omega)

    Args:
        PDE_NN_BASE ([type]): [description]
    """

    default_cfgs = dict(
        train_field="fwd",
        in_channels=3,
        out_channels=2,
        dim=32,
        act_func="GELU",
        img_size=512,  # image size
        img_res=50,  # image resolution
        pml_width=0.5,
        pml_permittivity=0 + 0j,
        buffer_width=0.5,
        buffer_permittivity=-1e-10 + 0j,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        device=torch.device("cuda"),
        eps_min=2.085136,
        eps_max=12.3,
        aux_head=False,
        aux_head_idx=1,
        pos_encoding="none",
        with_cp=False,
        fourier_feature="none",
        mapping_size=2,
        norm_cfg=dict(type="LayerNorm", data_format="channels_first"),
        act_cfg=dict(type="GELU"),

        wl=1.55,
        temp=300,
        mode=1,

        conv_cfg=dict(type="Conv2d", padding_mode="replicate"),
        linear_cfg=dict(type="Linear"),
        incident_field_fwd=False,
    )

    def __init__(
        self,
        **cfgs,
    ):
        super().__init__()
        self.load_cfgs(**cfgs)

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.build_layers()
        self.build_pml_mask()
        self.build_sim()

    def load_cfgs(
        self,
        **cfgs,
    ) -> None:
        super().load_cfgs(**self.default_cfgs)
        super().load_cfgs(**cfgs)

        assert self.train_field in {
            "fwd",
            "adj",
        }, f"train_field must be fwd or adj, but got {self.train_field}"

        assert (
            self.out_channels % 2 == 0
        ), f"The output channels must be even number larger than 2, but got {self.out_channels}"

        match self.pos_encoding:
            case "none":
                pass
            case "linear":
                self.in_channels += 2
            case "exp":
                self.in_channels += 4
            case "exp3":
                self.in_channels += 6
            case "exp4":
                self.in_channels += 8
            case "exp_full", "exp_full_r":
                self.in_channels += 7
            case _:
                raise ValueError(
                    f"pos_encoding only supports linear and exp, but got {self.pos_encoding}"
                )

        if self.fourier_feature == "basic":
            self.B = torch.eye(2, device=self.device)
        elif self.fourier_feature.startswith("gauss"):  # guass_10
            scale = eval(self.fourier_feature.split("_")[-1])
            self.B = torch.randn((self.mapping_size, 1), device=self.device) * scale
            self.in_channels = self.in_channels - 1 + 2 * self.mapping_size
        elif self.fourier_feature == "learnable":
            self.LFF = LearnableFourierFeatures(
                pos_dim=2, f_dim=2 * self.mapping_size, h_dim=64, d_dim=64
            )
            self.in_channels = self.in_channels + 64
        elif self.fourier_feature == "none":
            pass
        else:
            raise ValueError("fourier_feature only supports basic and gauss")

    def build_layers(self):
        dim = self.dim
        self.dconv_down1 = double_conv(self.in_channels, dim, dim)
        self.dconv_down2 = double_conv(dim, dim * 2, dim * 2)
        self.dconv_down3 = double_conv(dim * 2, dim * 4, dim * 4)
        self.dconv_down4 = double_conv(dim * 4, dim * 8, dim * 8)

        # self.maxpool = nn.MaxPool2d(2)
        self.maxpool = nn.AvgPool2d(2)

        self.upsample1 = nn.ConvTranspose2d(dim * 8, dim * 8, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(dim * 4, dim * 4, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(dim * 2, dim * 2, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(dim * 12, dim * 8, dim * 4)
        self.dconv_up2 = double_conv(dim * 6, dim * 4, dim * 2)
        self.dconv_up1 = double_conv(dim * 3, dim * 2, dim)
        self.drop_out = nn.Dropout2d(self.dropout_rate)
        self.conv_last = nn.Conv2d(dim, self.out_channels, 1)

    def set_trainable_permittivity(self, mode: bool = True) -> None:
        self.trainable_permittivity = mode


    def requires_network_params_grad(self, mode: float = True) -> None:
        params = self.parameters()
        for p in params:
            p.requires_grad_(mode)

    def fourier_feature_mapping(self, x: Tensor) -> Tensor:
        if self.fourier_feature == "none":
            return x
        else:
            x = x.permute(0, 2, 3, 1)  # B, H, W, 1
            x_proj = (
                2.0 * torch.pi * x
            ) @ self.B.T  # Matrix multiplication and scaling # B, H, W, mapping_size
            x_proj = torch.cat(
                [torch.sin(x_proj), torch.cos(x_proj)], dim=-1
            )  # B, H, W, 2 * mapping_size
            x_proj = x_proj.permute(0, 3, 1, 2)  # B, 2 * mapping_size, H, W
            return x_proj

    def forward(
        self,
        eps,
        src,
    ):
        if self.incident_field_fwd:
            incident_field_fwd = self.incident_field_from_src(src)
            incident_field_fwd = torch.view_as_real(incident_field_fwd).permute(
                0, 3, 1, 2
            )  # B, 2, H, W
            incident_field_fwd = incident_field_fwd / (
                torch.abs(incident_field_fwd).amax(dim=(1, 2, 3), keepdim=True) + 1e-6
            )
        src = torch.view_as_real(src.resolve_conj()).permute(0, 3, 1, 2)  # B, 2, H, W
        src = src / (torch.abs(src).amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

        eps = eps / 12.11
        eps = eps.unsqueeze(1)  # B, 1, H, W

        if self.fourier_feature == "learnable":
            H = eps.shape[-2]
            W = eps.shape[-1]
            bs = eps.shape[0]
            y = torch.linspace(-1, 1, H, device=eps.device)
            x = torch.linspace(-1, 1, W, device=eps.device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack((grid_x, grid_y), dim=-1)  # Shape (H, W, 2)
            grid_flat = rearrange(
                grid, "h w d -> (h w) d"
            )  # Flatten spatial to shape (H*W, 2)
            pos = grid_flat.unsqueeze(0).unsqueeze(2).expand(bs, H * W, 1, 2)
            enc_fwd = self.LFF(pos).permute(0, 2, 1).reshape(bs, -1, H, W)
            eps_enc_fwd = torch.cat((eps, enc_fwd), dim=1)
        else:
            enc_fwd = self.fourier_feature_mapping(eps)
            eps_enc_fwd = torch.cat((eps, enc_fwd), dim=1) if self.fourier_feature != "none" else eps

        if self.incident_field_fwd:
            x = torch.cat((eps_enc_fwd, incident_field_fwd), dim=1)
        else:
            x = torch.cat((eps_enc_fwd, src), dim=1)

        # positional encoding
        grid = self.get_grid(
            x.shape,
            x.device,
            mode=self.pos_encoding,
            epsilon=eps * 12.11,
            wavelength=self.wl,
            grid_step=1 / self.img_res,
        )  # [bs, 2 or 4 or 8, h, w] real
        if grid is not None:
            x = torch.cat((x, grid), dim=1)  # [bs, inc*2+4, h, w] real
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)  # [bs, outc, h, w] real
        
        return x
