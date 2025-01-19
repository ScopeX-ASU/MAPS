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
from thirdparty.ceviche.ceviche.constants import *
from .layers.neurolight_conv2d import NeurOLightConv2d
from .pde_base import PDE_NN_BASE
from .layers.layer_norm import MyLayerNorm
from core.utils import print_stat, resize_to_targt_size
from core.fdfd.fdfd import fdfd_ez
from ceviche.constants import *
import matplotlib.pyplot as plt
from .model_base import ModelBase, ConvBlock, LinearBlock
from .fno_cnn import LearnableFourierFeatures
from einops import rearrange
from core.utils import (
    Si_eps,
    SiO2_eps,
)
from mmcv.cnn.bricks import build_activation_layer, build_conv_layer, build_norm_layer
__all__ = ["NeurOLight2d"]


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
        norm_cfg: dict | None = dict(),
        act_cfg: dict | None = dict(type="ReLU"),
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
        if norm_cfg is not None:
            _, self.norm1 = build_norm_layer(norm_cfg, out_channels // 2)
        else:
            self.norm1 = None
        if act_cfg is not None:
            act_cfg_copy = act_cfg.copy()
            act_cfg_copy["inplace"] = True
            self.act1 = build_activation_layer(act_cfg_copy) if act_cfg['type'].lower == "relu" else build_activation_layer(act_cfg)
        else:
            self.act1 = None

        self.conv2 = BSConv2d(
            out_channels // 2,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        if norm_cfg is not None:
            _, self.norm2 = build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm2 = None
        if act_cfg is not None:
            act_cfg_copy = act_cfg.copy()
            act_cfg_copy["inplace"] = True
            self.act2 = build_activation_layer(act_cfg_copy) if act_cfg['type'].lower == "relu" else build_activation_layer(act_cfg)
        else:
            self.act2 = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        if self.act1 is not None:
            x = self.act1(x)
        x = self.conv2(x)
        if self.norm2 is not None:
            x = self.norm2(x)
        if self.act2 is not None:
            x = self.act2(x)
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
        act_cfg: dict | None = dict(type="GELU"),
        norm_cfg: dict | None = dict(type="LayerNorm", data_format="channels_first"),
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

        if norm_cfg is not None:
            _, self.pre_norm = build_norm_layer(norm_cfg, in_channels)
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
            _, self.ffn_norm = build_norm_layer(norm_cfg, out_channels * self.expansion) if ffn else (None, None)
        else:
            self.pre_norm = None
            self.norm = None

        if act_cfg is not None:
            self.act_func = build_activation_layer(act_cfg)
        else:
            self.act_func = None

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
                    # nn.BatchNorm2d(out_channels * self.expansion),
                    # nn.GELU(),
                    self.ffn_norm,
                    self.act_func,
                    nn.Conv2d(out_channels * self.expansion, out_channels, 1),
                )
            else:
                self.ff = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels * self.expansion, 1),
                    # nn.BatchNorm2d(out_channels * self.expansion),
                    # nn.GELU(),
                    self.ffn_norm,
                    self.act_func,
                    nn.Conv2d(out_channels * self.expansion, out_channels, 1),
                )
        else:
            self.ff = None
        if aug_path:
            self.aug_path = nn.Sequential(BSConv2d(in_channels, out_channels, 3), self.act_func)
        else:
            self.aug_path = None

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

@MODELS.register_module()
class NeurOLight2d(ModelBase):
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
        in_channels=1,
        out_channels=2,
        dim=16,
        kernel_list=[16, 16, 16, 16],
        kernel_size_list=[1, 1, 1, 1],
        padding_list=[0, 0, 0, 0],
        hidden_list=[128],
        mode_list=[(20, 20), (20, 20), (20, 20), (20, 20)],
        domain_size=(20, 100),
        grid_step=1.550 / 20,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        eps_min=2.085136,
        eps_max=12.3,
        aux_head=False,
        aux_head_idx=1,
        conv_stem=True,
        aug_path=True,
        ffn=True,
        ffn_dwconv=True,
        mapping_size=2,
        output_sparam=False,
        norm_cfg=dict(type="LayerNorm", data_format="channels_first"),
        act_cfg=dict(type="GELU"),
        fourier_feature="none",
        pos_encoding="none",
        device=torch.device("cuda"),
        with_cp=False,
        img_size=512,  # image size
        img_res=50,  # image resolution
        pml_width=0.5,  # PML width
        wl=1.55,
        temp=300,
        mode=1,
        conv_cfg=dict(type="Conv2d", padding_mode="replicate"),
        linear_cfg=dict(type="Linear"),
        incident_field=False,
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
        self.set_trainable_permittivity(False)


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
            raise ValueError("fourier_feature only supports basic and gauss or none")

    def build_layers(self):
        if self.conv_stem:
            self.stem = ResStem(
                self.in_channels,
                self.dim,
                kernel_size=3,
                stride=1,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg,
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
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg,
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
                ConvBlock(inc, outc, kernel_size=1, padding=0, act_cfg=self.act_cfg, device=self.device),
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
                act_cfg=None,
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

        if self.output_sparam:
            self.build_sparam_head(self.kernel_list[-1])

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

    def forward(
        self,
        eps,
        src,
        monitor_slices=None,
        monitor_slice_list=None,
        in_slice_name=None,
        wl=None,
        temp=None,
    ):
        src = src / (torch.abs(src).amax(dim=(1, 2), keepdim=True) + 1e-6) # B, H, W
        if self.incident_field:
            incident_field = self.calculate_incident_light_field(
                                                        source=src,
                                                        monitor_slices=monitor_slices,
                                                        monitor_slice_list=monitor_slice_list,
                                                        in_slice_name=in_slice_name,
                                                        wl=wl,
                                                        temp=temp,
                                                    ) # Bs, 2, H, W
        temp_multiplier = self._get_temp_multiplier(temp).unsqueeze(-1).unsqueeze(-1)  # B, 1, 1
        eps_min = torch.amin(eps, dim=(-1, -2), keepdim=True)
        eps = (eps - torch.amin(eps, dim=(-1, -2), keepdim=True)) / (torch.amax(eps, dim=(-1, -2), keepdim=True) - torch.amin(eps, dim=(-1, -2), keepdim=True))
        # now it is normalized to [0, 1]
        eps = (eps * temp_multiplier + eps_min) / 12.11
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

        if self.incident_field:
            x = torch.cat((eps_enc_fwd, incident_field), dim=1)
        else:
            src = torch.view_as_real(src).permute(0, 3, 1, 2) # B, 2, H, W
            x = torch.cat((eps_enc_fwd, src), dim=1)

        # positional encoding
        grid = self.get_grid(
            x.shape,
            x.device,
            mode=self.pos_encoding,
            epsilon=eps * 12.11,
            wavelength=wl.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 1e-6, # B, 1, 1, 1
            grid_step=1 / self.img_res * 1e-6,
        )  # [bs, 2 or 4 or 8, h, w] real
        if grid is not None:
            x = torch.cat((x, grid), dim=1)  # [bs, inc*2+4, h, w] real

        x = self.stem(x)
        x = self.features(x)

        forward_Ez_field = self.head(x)  # 1x1 conv
        
        if self.output_sparam:
            assert hasattr(self, "sparam_head"), "sparam_head is not defined"
            s_parameter = self.sparam_head(forward_Ez_field)
            return forward_Ez_field, s_parameter

        return forward_Ez_field