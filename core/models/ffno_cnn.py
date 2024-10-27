"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-03 01:17:52
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-05 03:25:43
"""

from typing import List, Optional, Tuple

import numpy as np
from sympy import Identity
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.activation import Swish
from timm.models.layers import DropPath, to_2tuple
from torch import nn
from torch.functional import Tensor
from torch.types import Device
from torch.utils.checkpoint import checkpoint
from .constant import *
from .layers.activation import SIREN
from .layers.ffno_conv2d import FFNOConv2d
from torch.types import _size
from .layers.layer_norm import MyLayerNorm
from core.utils import resize_to_targt_size

__all__ = ["FFNO2d"]

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

# class ConvBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int = 1,
#         padding: int = 0,
#         act_func: Optional[str] = "GELU",
#         device: Device = torch.device("cuda:0"),
#     ) -> None:
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
#         if act_func is None:
#             self.act_func = None
#         elif act_func.lower() == "siren":
#             self.act_func = SIREN()
#         elif act_func.lower() == "swish":
#             self.act_func = Swish()
#         else:
#             self.act_func = getattr(nn, act_func)()

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.conv(x)
#         if self.act_func is not None:
#             x = self.act_func(x)
#         return x


class BSConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size = 3,
        stride: _size = 1,
        dilation: _size = 1,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [(dilation[i] * (kernel_size[i] - 1) + 1) // 2 for i in range(len(kernel_size))]
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=bias,
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ResStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size = 3,
        stride: _size = 1,
        dilation: _size = 1,
        norm: str = "ln",
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [(dilation[i] * (kernel_size[i] - 1) + 1) // 2 for i in range(len(kernel_size))]

        # self.conv1 = nn.Conv2d(
        #     in_channels,
        #     out_channels // 2,
        #     kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     groups=groups,
        #     bias=bias,
        # )
        self.conv1 = BSConv2d(
            in_channels,
            out_channels // 2,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        if norm == "bn":
            self.norm1 = nn.BatchNorm2d(out_channels // 2)
        elif norm == "ln":
            self.norm1 = MyLayerNorm(out_channels // 2, data_format="channels_first")
        else:
            raise ValueError(f"Norm type {norm} not supported")
        self.act1 = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv2d(
        #     out_channels // 2,
        #     out_channels,
        #     kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     groups=groups,
        #     bias=bias,
        # )
        self.conv2 = BSConv2d(
            out_channels // 2,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        if self.norm == "bn":
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif self.norm == "ln":
            self.norm2 = MyLayerNorm(out_channels, data_format="channels_first")
        else:
            raise ValueError(f"Norm type {norm} not supported")
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
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

class FFNO2dBlock(nn.Module):
    expansion = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "GELU",
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        with_cp=False,
        ffn: bool = True,
        ffn_dwconv: bool = True,
        aug_path: bool = True,
        norm: str = "ln",
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        # self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.f_conv = FFNOConv2d(in_channels, out_channels, n_modes, device=device)
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
            self.pre_norm = nn.BatchNorm2d(in_channels)
        elif norm == "ln":
            self.norm = MyLayerNorm(out_channels, data_format="channels_first")
            self.pre_norm = MyLayerNorm(in_channels, data_format="channels_first")
        self.with_cp = with_cp
        # self.norm.weight.data.zero_()
        if ffn:
            if ffn_dwconv:
                self.ff = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels * self.expansion, 1),
                    nn.Conv2d(
                        out_channels * self.expansion,
                        out_channels * self.expansion,
                        3,
                        groups=out_channels * self.expansion,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels * self.expansion) if norm == "bn" else MyLayerNorm(out_channels * self.expansion, data_format="channels_first"),
                    nn.GELU(),
                    nn.Conv2d(out_channels * self.expansion, out_channels, 1),
                )
            else:
                self.ff = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels * self.expansion, 1),
                    nn.BatchNorm2d(out_channels * self.expansion) if norm == "bn" else MyLayerNorm(out_channels * self.expansion, data_format="channels_first"),
                    nn.GELU(),
                    nn.Conv2d(out_channels * self.expansion, out_channels, 1),
                )
        else:
            self.ff = None
        if aug_path:
            self.aug_path = nn.Sequential(BSConv2d(in_channels, out_channels, 3), nn.GELU())
        else:
            self.aug_path = None
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "siren":
            self.act_func = SIREN()
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        def _inner_forward(x):
            y = x
            # x = self.norm(self.ff(self.f_conv(self.pre_norm(x))))
            if self.ff is not None:
                x = self.norm(self.ff(self.pre_norm(self.f_conv(x))))
                # x = self.norm(self.ff(self.act_func(self.pre_norm(self.f_conv(x)))))
                # x = self.norm(self.ff(self.act_func(self.pre_norm(self.f_conv(x)))))
                x = self.drop_path(x) + y
            else:
                x = self.act_func(self.drop_path(self.norm(self.f_conv(x))) + y)
            if self.aug_path is not None:
                x = x + self.aug_path(y)
            return x

        # def _inner_forward(x):
        #     x = self.drop_path(self.pre_norm(self.f_conv(x))) + self.aug_path(x)
        #     y = x
        #     x = self.norm(self.ff(x))
        #     x = self.drop_path2(x) + y
        #     return x

        if x.requires_grad and self.with_cp:
            return checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)


class FFNO2d(nn.Module):
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
        dim: int = 16,
        kernel_list: List[int] = [72, 72, 72, 72],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [512],
        mode_list: List[Tuple[int]] = [(128, 129), (128, 129), (128, 129), (128, 129)],
        act_func: Optional[str] = "GELU",
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        aux_head: bool = False,
        aux_head_idx: int = 1,
        with_cp=False,
        conv_stem: bool = False,
        aug_path: bool = True,
        ffn: bool = True,
        ffn_dwconv: bool = True,
        **kwargs,
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
        assert (
            out_channels % 2 == 0
        ), f"The output channels must be even number larger than 2, but got {out_channels}"
        self.dim = dim
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
        self.with_cp = with_cp
        self.conv_stem = conv_stem
        self.aug_path = aug_path
        self.ffn = ffn
        self.ffn_dwconv = ffn_dwconv

        self.device = device
        self.pos_encoding = "none"
        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()

        self.permittivity_encoder = None
        self.set_trainable_permittivity(False)

    def build_layers(self):
        if self.conv_stem:
            self.stem = nn.Sequential(
                ConvBlock(
                    self.in_channels,
                    self.dim,
                    kernel_size=5,
                    padding=2,
                    stride=2,
                    act_func=self.act_func,
                    device=self.device,
                ),
                ConvBlock(
                    self.dim,
                    self.dim,
                    kernel_size=5,
                    padding=2,
                    stride=2,
                    act_func=None,
                    device=self.device,
                ),
            )
        else:
            self.stem = nn.Conv2d(self.in_channels, self.dim, 1)
        kernel_list = [self.dim] + self.kernel_list
        drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))
        print("this is the mode list: ", self.mode_list, flush=True)
        features = [
            FFNO2dBlock(
                inc,
                outc,
                n_modes,
                kernel_size,
                padding,
                act_func=self.act_func,
                drop_path_rate=drop,
                device=self.device,
                with_cp=self.with_cp,
                aug_path=self.aug_path,
                ffn=self.ffn,
                ffn_dwconv=self.ffn_dwconv,
            )
            for inc, outc, n_modes, kernel_size, padding, drop in zip(
                kernel_list[:-1],
                kernel_list[1:],
                self.mode_list,
                self.kernel_size_list,
                self.padding_list,
                drop_path_rates,
            )
        ]
        self.features = nn.Sequential(*features)
        hidden_list = [self.kernel_list[-1]] + self.hidden_list
        head = [
            nn.Sequential(
                ConvBlock(inc, outc, kernel_size=1, padding=0, act_func=self.act_func, device=self.device),
                nn.Dropout2d(self.dropout_rate),
            )
            for inc, outc in zip(hidden_list[:-1], hidden_list[1:])
        ]
        # 2 channels as real and imag part of the TE field
        head += [
            ConvBlock(
                hidden_list[-1],
                self.out_channels,
                kernel_size=1,
                padding=0,
                act_func=None,
                device=self.device,
            )
        ]

        self.head = nn.Sequential(*head)

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
        x = self.features(x)
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
