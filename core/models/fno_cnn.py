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
# from .layers.local_fno import FNO
from .constant import *
# from .layers.fno_conv2d import FNOConv2d
import torch.nn.functional as F
from functools import lru_cache
from pyutils.torch_train import set_torch_deterministic
from core.utils import resize_to_targt_size
__all__ = ["FNO3d"]

class SpatialInterpolater(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=False)
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.ln is not None:
            x = self.ln(x)
        if self.act_func is not None:
            x = self.act_func(x)
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
        dim = 2,
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
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None] # add one extra dimension to match conv2d but not 2d
            elif self.dim == 2:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FNO3d(nn.Module):
    """
    Frequency-domain scattered electric field envelop predictor
    Assumption:
    (1) TE10 mode, i.e., Ey(r, omega) = Ez(r, omega) = 0
    (2) Fixed wavelength. wavelength currently not being modeled
    (3) Only predict Ex_scatter(r, omega)

    Args:
        PDE_NN_BASE ([type]): [description]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        mode_list: List[Tuple[int]] = [(20, 20)],
        act_func: Optional[str] = "GELU",
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        aux_head: bool = False,
        aux_head_idx: int = 1,
        pos_encoding: str = "none",
        with_cp: bool = False,
        mode1: int = 20,
        mode2: int = 20,
    ):
        super().__init__()

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode1 = mode1
        self.mode2 = mode2
        assert (
            out_channels % 2 == 0
        ), f"The output channels must be even number larger than 2, but got {out_channels}"
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.hidden_list = hidden_list
        self.mode_list = mode_list
        self.act_func = act_func
      
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate

        self.aux_head = aux_head
        self.aux_head_idx = aux_head_idx
        self.pos_encoding = pos_encoding
        self.with_cp = with_cp
        if pos_encoding == "none":
            pass
        elif pos_encoding == "linear":
            self.in_channels += 2
        elif pos_encoding == "exp":
            self.in_channels += 4
        elif pos_encoding == "exp3":
            self.in_channels += 6
        elif pos_encoding == "exp4":
            self.in_channels += 8
        elif pos_encoding in {"exp_full", "exp_full_r"}:
            self.in_channels += 7
        else:
            raise ValueError(f"pos_encoding only supports linear and exp, but got {pos_encoding}")

        self.device = device

        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()

    def build_layers(self):
        self.stem = nn.Sequential(
            ConvBlock(
                self.in_channels,
                self.hidden_list[0],
                kernel_size=5,
                padding=2,
                stride=2,
                act_func=self.act_func,
                device=self.device,
            ),
            ConvBlock(
                self.hidden_list[0],
                self.hidden_list[0],
                kernel_size=5,
                padding=2,
                stride=2,
                act_func=None,
                device=self.device,
            ),
        )
        print("this is the mode to pass to the FNO", self.mode1, self.mode2, flush=True)
        self.head = FNO(
            n_modes=(self.mode1, self.mode2),
            # in_channels=self.in_channels,
            in_channels=self.hidden_list[0],
            out_channels=self.out_channels,
            lifting_channels=self.hidden_list[-1],
            projection_channels=96,
            hidden_channels=self.hidden_list[-1],
            n_layers=4,
            norm=None,
            factorization=None,
        )

        if self.aux_head:
            hidden_list = [self.kernel_list[self.aux_head_idx]] + self.hidden_list
            head = [
                nn.Sequential(
                    ConvBlock(
                        inc, outc, kernel_size=1, padding=0, act_func=self.act_func, device=self.device
                    ),
                    nn.Dropout2d(self.dropout_rate),
                )
                for inc, outc in zip(hidden_list[:-1], hidden_list[1:])
            ]
            # 2 channels as real and imag part of the TE field
            head += [
                ConvBlock(
                    hidden_list[-1],
                    self.out_channels // 2,
                    kernel_size=1,
                    padding=0,
                    act_func=None,
                    device=self.device,
                )
            ]

            self.aux_head = nn.Sequential(*head)
        else:
            self.aux_head = None

    def set_trainable_permittivity(self, mode: bool = True) -> None:
        self.trainable_permittivity = mode

    def requires_network_params_grad(self, mode: float = True) -> None:
        params = (
            self.stem.parameters()
            + self.features.parameters()
            + self.head.parameters()
            + self.full_field_head.parameters()
        )
        for p in params:
            p.requires_grad_(mode)

    @lru_cache(maxsize=16)
    def _get_linear_pos_enc(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.arange(0, size_x, device=device)
        gridy = torch.arange(0, size_y, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy)
        mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
        return mesh

    def forward(
        self,
        eps,
        src,
        adj_src,
        incident_field, 
    ):
        # src and adj_src are all complex numbers tensor
        src = src["source_profile-wl-1.55-port-in_port_1-mode-1"]
        adj_src = adj_src["adj_src-wl-1.55-port-in_port_1-mode-1"]
        incident_field = incident_field["incident_field-wl-1.55-port-in_port_1-mode-1"]
        src = torch.view_as_real(src).permute(0, 3, 1, 2) # B, 2, H, W
        src = src / (torch.abs(src).amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
        adj_src = torch.view_as_real(adj_src).permute(0, 3, 1, 2) # B, 2, H, W
        adj_src = adj_src / (torch.abs(adj_src).amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
        incident_field = torch.view_as_real(incident_field).permute(0, 3, 1, 2) # B, 2, H, W
        incident_field = incident_field / (torch.abs(incident_field).amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

        # plt.figure()
        # plt.imshow(incident_field[0][0].detach().cpu().numpy())
        # plt.savefig("./figs/incident_field_real.png")
        # plt.close()

        # plt.figure()
        # plt.imshow(incident_field[0][1].detach().cpu().numpy())
        # plt.savefig("./figs/incident_field_imag.png")
        # plt.close()
        # quit()

        eps_copy = eps.clone()
        eps = 1 / eps # take the inverse of the permittivity to easy the training difficulty
        eps = eps.unsqueeze(1) # B, 1, H, W

        # -----------this is before ----------------
        # eps = torch.cat((eps, eps), dim=0)
        # sources = torch.cat((src, adj_src), dim=0)
        # x = torch.cat((eps, sources), dim=1) # 2B, 3, H, W
        # x = self.stem(x)
        # feature = self.features(x)
        # forward_Ez_field = self.head(feature)[:feature.size(0) // 2]
        # adjoint_Ez_field = self.head(feature)[feature.size(0) // 2:]
        # ------------------------------------------

        # -----------this is after ----------------
        x = torch.cat((eps, incident_field), dim=1)
        x = self.stem(x)
        forward_Ez_field = self.head(x)
        forward_Ez_field = resize_to_targt_size(forward_Ez_field, (src.shape[-2], src.shape[-1]))
        if len(forward_Ez_field.shape) == 3:
            forward_Ez_field = forward_Ez_field.unsqueeze(0)
        # feature = self.features(x)
        # forward_Ez_field = self.head(feature)
        # ------------------------------------------

        # calculate the hx and hy from the Ez field
        # forward_Hx_field, forward_Hy_field = self.from_Ez_to_Hx_Hy(eps_copy, forward_Ez_field)

        # forward_field = torch.cat((forward_Hx_field, forward_Hy_field, forward_Ez_field), dim=1)
        adjoint_field = None

        return forward_Ez_field, adjoint_field
