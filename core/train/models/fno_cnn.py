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

__all__ = ["FNO2d"]


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
@MODELS.register_module()
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
            elif self.dim == 1:
                # print("this is the shape of x", x.shape)
                # print("this is the shape of weight", self.weight.shape)
                x = self.weight[None, :] * x + self.bias[None, :]
            return x


@MODELS.register_module()
class FNO2d(ModelBase):
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
        kernel_list=[16, 16, 16, 16],
        kernel_size_list=[1, 1, 1, 1],
        padding_list=[0, 0, 0, 0],
        hidden_list=[128],
        mode_list=[(20, 20)],
        dropout_rate=0.0,
        drop_path_rate=0.0,
        aux_head=False,
        aux_head_idx=1,
        pos_encoding="none",
        with_cp=False,
        mode1=20,
        mode2=20,
        fourier_feature="none",
        mapping_size=2,
        fno_block_only=False,
        conv_cfg=dict(type="Conv2d", padding_mode="replicate"),
        layer_cfg=dict(type="Conv2d", padding_mode="replicate"),
        linear_cfg=dict(type="Linear"),
        # norm_cfg=dict(type="MyLayerNorm", data_format="channels_first"),
        norm_cfg=dict(type="LayerNorm", data_format="channels_first"),
        act_cfg=dict(type="GELU"),
        device=torch.device("cuda"),
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

        omega = 2 * np.pi * C_0 / (1.55 * 1e-6)
        self.sim = fdfd_ez(
            omega=omega,
            dL=2e-8,
            eps_r=torch.randn((260, 260), device=self.device),  # random permittivity
            npml=(25, 25),
        )
        self.padding = 9  # pad the domain if input is non-periodic

    def _build_s_param_head(self):
        s_param_head = nn.Sequential(
            ConvBlock(
                128,
                256,
                kernel_size=5,
                padding=2,
                stride=2,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg,
                device=self.device,
            ),
            ConvBlock(
                256,
                256,
                kernel_size=5,
                padding=2,
                stride=2,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg,
                device=self.device,
            ),
            ConvBlock(
                256,
                256,
                kernel_size=5,
                padding=2,
                stride=2,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg,
                device=self.device,
            ),
            ConvBlock(
                256,
                256*2,
                kernel_size=5,
                padding=2,
                stride=2,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg,
                device=self.device,
            ),
            ConvBlock(
                256*2,
                256*2,
                kernel_size=5,
                padding=2,
                stride=2,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg,
                device=self.device,
            ),
            ConvBlock(
                256*2,
                256*2,
                kernel_size=5,
                padding=2,
                stride=2,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg,
                device=self.device,
            ),
            nn.Flatten(),
            LinearBlock(
                12800,
                128,
                bias=True,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                dropout=0.3,
                device=self.device,
            ),
            LinearBlock(
                128,
                64,
                bias=True,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                dropout=0.3,
                device=self.device,
            ),
            LinearBlock(
                64,
                2,
                bias=True,
                norm_cfg=None,
                act_cfg=None,
                dropout=0.3,
                device=self.device,
            ),
        )

        self.s_param_head = s_param_head

    def _build_layers(self):
        hidden_dim = self.hidden_list[-1]
        lifting = MLP(
            in_channels=self.in_channels,
            out_channels=hidden_dim,
            hidden_channels=hidden_dim,
            n_layers=1,
            n_dim=2,
        )
        fno_blocks = nn.ModuleList()
        fno_blocks.append(
            FNOBlocks(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                n_modes=(self.mode1 * 2, self.mode2 * 2),
                output_scaling_factor=None,
                use_mlp=False,
                mlp_dropout=0,
                mlp_expansion=0.5,
                non_linearity=F.gelu,
                stabilizer=None,
                norm=None,
                preactivation=False,
                fno_skip="linear",
                mlp_skip="soft-gating",
                max_n_modes=None,
                fno_block_precision="full",
                rank=1.0,
                fft_norm="forward",
                fixed_rank_modes=False,
                implementation="factorized",
                separable=False,
                factorization=None,
                decomposition_kwargs=dict(),
                joint_factorization=False,
                SpectralConv=SpectralConv,
                n_layers=4,
            )
        )

        head = MLP(
            in_channels=hidden_dim,
            out_channels=self.out_channels,
            hidden_channels=hidden_dim,
            n_layers=2,
            n_dim=2,
        )

        # head = ConvBlock(
        #     hidden_dim // 2 + hidden_dim,
        #     self.out_channels,
        #     kernel_size=1,
        #     padding=0,
        #     # norm_cfg=self.norm_cfg,
        #     norm_cfg=None,
        #     act_cfg=None,
        #     device=self.device,
        # )

        return lifting, fno_blocks, head

    def build_layers(self):
        self.lifting, self.fno_blocks, self.head = self._build_layers()
        # print("if nn have eps branch", hasattr(self, "has_eps_branch"), flush=True)

        if getattr(self, "s_param_only", False):
            self._build_s_param_head()

        if self.aux_head:
            hidden_list = [self.kernel_list[self.aux_head_idx]] + self.hidden_list
            head = [
                nn.Sequential(
                    ConvBlock(
                        inc,
                        outc,
                        kernel_size=1,
                        padding=0,
                        act_cfg=self.act_cfg,
                        device=self.device,
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
                    act_cfg=None,
                    device=self.device,
                )
            ]

            self.aux_head = nn.Sequential(*head)
        else:
            self.aux_head = None

        # Simulation grid size
        grid_size = (260, 260)  # Adjust to your simulation grid size
        pml_thickness = 25

        self.pml_mask = torch.ones(grid_size).to(self.device)

        # Define the damping factor for exponential decay
        damping_factor = torch.tensor(
            [
                0.05,
            ],
            device=self.device,
        )  # adjust this to control decay rate

        # Apply exponential decay in the PML regions
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Calculate distance from each edge
                dist_to_left = max(0, pml_thickness - i)
                dist_to_right = max(0, pml_thickness - (grid_size[0] - i - 1))
                dist_to_top = max(0, pml_thickness - j)
                dist_to_bottom = max(0, pml_thickness - (grid_size[1] - j - 1))

                # Calculate the damping factor based on the distance to the nearest edge
                dist = max(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
                if dist > 0:
                    self.pml_mask[i, j] = torch.exp(-damping_factor * dist)

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

    def incident_field_from_src(self, src: Tensor) -> Tensor:
        if self.train_field == "fwd":
            mode = src[:, int(0.4 * src.shape[-2] / 2), :]
            mode = mode.unsqueeze(1).repeat(1, src.shape[-2], 1)
            source_index = int(0.4 * src.shape[-2] / 2)
            resolution = (
                2e-8  # hardcode here since the we are now using resolution of 50px/um
            )
            epsilon = Si_eps(1.55)
            lambda_0 = (
                1.55e-6  # wavelength is hardcode here since we are now using 1.55um
            )
            k = (2 * torch.pi / lambda_0) * torch.sqrt(torch.tensor(epsilon)).to(
                src.device
            )
            x_coords = torch.arange(src.shape[-2]).float().to(src.device)
            distances = torch.abs(x_coords - source_index) * resolution
            phase_shifts = (k * distances).unsqueeze(1)
            mode = mode * torch.exp(1j * phase_shifts)

        elif self.train_field == "adj":
            # in the adjoint mode, there are two sources and we need to calculate the incident field for each of them
            # then added together as the incident field
            mode_x = src[:, int(0.41 * src.shape[-2] / 2), :]
            mode_x = mode_x.unsqueeze(1).repeat(1, src.shape[-2], 1)
            source_index = int(0.41 * src.shape[-2] / 2)
            resolution = (
                2e-8  # hardcode here since the we are now using resolution of 50px/um
            )
            epsilon = Si_eps(1.55)
            lambda_0 = (
                1.55e-6  # wavelength is hardcode here since we are now using 1.55um
            )
            k = (2 * torch.pi / lambda_0) * torch.sqrt(torch.tensor(epsilon)).to(
                src.device
            )
            x_coords = torch.arange(src.shape[-2]).float().to(src.device)
            distances = torch.abs(x_coords - source_index) * resolution
            phase_shifts = (k * distances).unsqueeze(1)
            mode_x = mode_x * torch.exp(1j * phase_shifts)

            mode_y = src[
                :, :, -int(0.4 * src.shape[-1] / 2)
            ]  # not quite sure with this index, need to plot it out to check
            mode_y = mode_y.unsqueeze(-1).repeat(1, 1, src.shape[-1])
            source_index = src.shape[-1] - int(0.4 * src.shape[-1] / 2)
            resolution = 2e-8
            epsilon = Si_eps(1.55)
            lambda_0 = 1.55e-6
            k = (2 * torch.pi / lambda_0) * torch.sqrt(torch.tensor(epsilon)).to(
                src.device
            )
            y_coords = torch.arange(src.shape[-1]).float().to(src.device)
            distances = torch.abs(y_coords - source_index) * resolution
            phase_shifts = (k * distances).unsqueeze(0)
            mode_y = mode_y * torch.exp(1j * phase_shifts)

            mode = mode_x + mode_y  # superposition of two sources
        return mode

    def forward(
        self,
        eps,
        src,
    ):
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

        ## eps_branch
        eps_1 = eps_2 = eps_0 = None

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
            eps_enc_fwd = torch.cat((eps, enc_fwd), dim=1)

        x_fwd = torch.cat((eps_enc_fwd, incident_field_fwd), dim=1)

        x_fwd = self.lifting(x_fwd)  # conv2d downsample

        if eps_0 is not None:
            x_fwd = x_fwd * eps_0

        x1_fwd = self.fno_blocks[0](x_fwd)  # fno block

        if hasattr(self, "s_param_head"):
            s_param = self.s_param_head(x1_fwd) * 1e-8
            return s_param
        forward_Ez_field = self.head(x1_fwd)  # 1x1 conv
        if len(forward_Ez_field.shape) == 3:
            forward_Ez_field = forward_Ez_field.unsqueeze(0)

        return forward_Ez_field
