"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-03 01:17:52
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-05 03:25:43
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.activation import Swish
from timm.models.layers import DropPath, to_2tuple
from torch import nn
from torch.functional import Tensor
from torch.types import Device, _size
from torch.utils.checkpoint import checkpoint
from pyutils.torch_train import set_torch_deterministic
from mmengine.registry import MODELS
from .constant import *
from .layers.neurolight_conv2d import NeurOLightConv2d
from .pde_base import PDE_NN_BASE
from .layers.layer_norm import MyLayerNorm
from core.utils import print_stat, resize_to_targt_size
from core.models.fdfd.fdfd import fdfd_ez
from ceviche.constants import *
import matplotlib.pyplot as plt

__all__ = ["NeurOLight2d"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "GELU",
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x


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
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [(dilation[i] * (kernel_size[i] - 1) + 1) // 2 for i in range(len(kernel_size))]

        self.conv1 = BSConv2d(
            in_channels,
            out_channels // 2,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.bn1 = MyLayerNorm(out_channels // 2, data_format = "channels_first")
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = BSConv2d(
            out_channels // 2,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.bn2 = MyLayerNorm(out_channels, data_format = "channels_first")
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class NeurOLight2dBlock(nn.Module):
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
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.f_conv = NeurOLightConv2d(in_channels, out_channels, n_modes, device=device)
        self.pre_norm = MyLayerNorm(in_channels, data_format = "channels_first")
        self.norm = MyLayerNorm(out_channels, data_format = "channels_first")
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
                    MyLayerNorm(out_channels * self.expansion, data_format = "channels_first"),
                    nn.GELU(),
                    nn.Conv2d(out_channels * self.expansion, out_channels, 1),
                )
            else:
                self.ff = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels * self.expansion, 1),
                    MyLayerNorm(out_channels * self.expansion, data_format = "channels_first"),
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
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        def _inner_forward(x):
            y = x
            if self.ff is not None:
                x = self.norm(self.ff(self.pre_norm(self.f_conv(x))))
                x = self.drop_path(x) + y
            else:
                x = self.act_func(self.drop_path(self.norm(self.f_conv(x))) + y)
            if self.aug_path is not None:
                x = x + self.aug_path(y)
            return x

        if x.requires_grad and self.with_cp:
            return checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)


class NeurOLight2d(nn.Module):
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
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        mode_list: List[Tuple[int]] = [(20, 20), (20, 20), (20, 20), (20, 20)],
        act_func: Optional[str] = "GELU",
        domain_size: Tuple[float] = [20, 100],  # computation domain in unit of um
        grid_step: float = 1.550 / 20,  # grid step size in unit of um, typically 1/20 or 1/30 of the wavelength
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        eps_min: float = 2.085136,
        eps_max: float = 12.3,
        aux_head: bool = False,
        aux_head_idx: int = 1,
        conv_stem: bool = True,
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
        assert out_channels % 2 == 0, f"The output channels must be even number larger than 2, but got {out_channels}"
        self.dim = dim
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.hidden_list = hidden_list
        self.mode_list = mode_list
        self.act_func = act_func
        self.domain_size = domain_size
        self.grid_step = grid_step
        self.domain_size_pixel = [round(i / grid_step) for i in domain_size]
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.aux_head = aux_head
        self.aux_head_idx = aux_head_idx
        self.conv_stem = conv_stem
        self.aug_path = aug_path
        self.ffn = ffn
        self.ffn_dwconv = ffn_dwconv
        self.with_cp = False
        self.device = device

        with MODELS.switch_scope_and_registry(None) as registry:
            self._conv = tuple(
                set(
                    [
                        registry.get("Conv2d"),
                    ]
                )
            )

        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()
        self.reset_parameters()
        self.set_trainable_permittivity(False)

    def reset_parameters(self, random_state: Optional[int] = None):
        for name, m in self.named_modules():
            if isinstance(m, self._conv) and hasattr(m, "reset_parameters"):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()

    def build_layers(self):
        if self.conv_stem:
            self.stem = ResStem(
                self.in_channels,
                self.dim,
                kernel_size=3,
                stride=1,
            )
        else:
            self.stem = nn.Conv2d(self.in_channels, self.dim, 1)
        kernel_list = [self.dim] + self.kernel_list
        drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))

        features = [
            NeurOLight2dBlock(
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

            self.aux_head = nn.Sequential(*head)
        else:
            self.aux_head = None

        self.sim = None

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

    def observe_waveprior(self, x: Tensor, wavelength: Tensor, grid_step: Tensor):
        epsilon = x[:, 0:1] * (self.eps_max - self.eps_min) + self.eps_min  # this is de-normalized permittivity

        # convert complex permittivity/mode to real numbers
        x = torch.view_as_real(x).permute(0, 1, 4, 2, 3).flatten(1, 2)  # [bs, inc*2, h, w] real

        # encoding
        grid = self.get_grid(
            x.shape,
            x.device,
            mode=self.pos_encoding,
            epsilon=epsilon,
            wavelength=wavelength,
            grid_step=grid_step,
        )
        return grid

    def observe_stem_output(self, x: Tensor, wavelength: Tensor, grid_step: Tensor):
        epsilon = x[:, 0:1] * (self.eps_max - self.eps_min) + self.eps_min  # this is de-normalized permittivity

        # convert complex permittivity/mode to real numbers
        x = torch.view_as_real(x).permute(0, 1, 4, 2, 3).flatten(1, 2)  # [bs, inc*2, h, w] real

        # encoding
        grid = self.get_grid(
            x.shape,
            x.device,
            mode=self.pos_encoding,
            epsilon=epsilon,
            wavelength=wavelength,
            grid_step=grid_step,
        )  # [bs, 2 or 4 or 8, h, w] real

        if grid is not None:
            x = torch.cat((x, grid), dim=1)  # [bs, inc*2+4, h, w] real
        return self.stem(x)

    def from_Ez_to_Hx_Hy(self, eps: Tensor, Ez: Tensor) -> None:
        # eps b, h, w
        # Ez b, 2, h, w
        eps = resize_to_targt_size(eps, (600, 900))
        if len(eps.shape) == 2:
            eps = eps.unsqueeze(0)
        Ez = resize_to_targt_size(Ez, (600, 900))
        if len(Ez.shape) == 3:
            Ez = Ez.unsqueeze(0)
        Ez = Ez.permute(0, 2, 3, 1).contiguous()
        Ez = torch.view_as_complex(Ez)
        omega = 2 * np.pi * C_0 / (1.55 * 1e-6)
        Hx = []
        Hy = []
        for i in range(Ez.size(0)):
            sim = fdfd_ez(
                omega=omega,
                dL=2e-8,
                eps_r=eps[i],
                npml=(50, 50),
            )
            Hx_vec, Hy_vec = sim._Ez_to_Hx_Hy(Ez[i].flatten())
            Hx.append(torch.view_as_real(Hx_vec.reshape(Ez[i].shape)).permute(2, 0, 1))
            Hy.append(torch.view_as_real(Hy_vec.reshape(Ez[i].shape)).permute(2, 0, 1))
        Hx = resize_to_targt_size(torch.stack(Hx, 0), (200, 300))
        if len(Hx.shape) == 3:
            Hx = Hx.unsqueeze(0)
        Hy = resize_to_targt_size(torch.stack(Hy, 0), (200, 300))
        if len(Hy.shape) == 3:
            Hy = Hy.unsqueeze(0)
        return Hx, Hy

    def forward(
        self,
        eps,
        src,
        adj_src,
        incident_field, 
    ) -> Tensor:
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
        feature = self.features(x)
        forward_Ez_field = self.head(feature)
        # ------------------------------------------

        # calculate the hx and hy from the Ez field
        # forward_Hx_field, forward_Hy_field = self.from_Ez_to_Hx_Hy(eps_copy, forward_Ez_field)

        # forward_field = torch.cat((forward_Hx_field, forward_Hy_field, forward_Ez_field), dim=1)
        adjoint_field = None

        return forward_Ez_field, adjoint_field