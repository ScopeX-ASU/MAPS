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

__all__ = ["FNO3d"]


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
        train_field: str = "fwd",
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
        fourier_feature: str = "none",
        mapping_size: int = 2,
        err_correction: bool = False,
        fno_block_only: bool = False,
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
        assert train_field in {
            "fwd",
            "adj",
        }, f"train_field must be fwd or adj, but got {train_field}"
        self.train_field = train_field
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
        self.err_correction = err_correction
        self.fno_block_only = fno_block_only
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
            raise ValueError(
                f"pos_encoding only supports linear and exp, but got {pos_encoding}"
            )
        self.fouier_feature = fourier_feature
        self.mapping_size = mapping_size
        if self.fouier_feature == "basic":
            self.B = torch.eye(2).to(device)
        elif self.fouier_feature.startswith("gauss"):  # guass_10
            scale = eval(self.fouier_feature.split("_")[-1])
            self.B = torch.randn((mapping_size, 1)).to(device) * scale
            self.in_channels = self.in_channels - 1 + 2 * mapping_size
        elif self.fouier_feature == "learnable":
            self.LFF = LearnableFourierFeatures(
                pos_dim=2, f_dim=2 * mapping_size, h_dim=64, d_dim=64
            )
            self.in_channels = self.in_channels + mapping_size
        elif self.fouier_feature == "none":
            pass
        else:
            raise ValueError("fourier_feature only supports basic and gauss")

        self.device = device
        omega = 2 * np.pi * C_0 / (1.55 * 1e-6)
        self.sim = fdfd_ez(
            omega=omega,
            dL=2e-8,
            eps_r=torch.randn((260, 260)).to(device),  # random permittivity
            npml=(25, 25),
        )
        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()
        self._build_eps_layers()

    def _build_eps_layers(self):
        self.eps_stem = nn.Sequential(
            ConvBlock(
                1,
                self.hidden_list[0] // 4,
                kernel_size=7,
                padding=3,
                stride=1,
                act_func=self.act_func,
                device=self.device,
            ),
            ConvBlock(
                self.hidden_list[0] // 4,
                self.hidden_list[0] // 2,
                kernel_size=7,
                padding=3,
                stride=2,
                act_func=None,
                device=self.device,
            ),
        )
        stages = nn.ModuleList()
        hidden_dim = self.hidden_list[-1] // 2
        stages.append(
            nn.Sequential(
                *[
                    ConvBlock(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        act_func=self.act_func,
                        device=self.device,
                    )
                    for _ in range(2)
                ]
            )
        )

        hidden_dim = self.hidden_list[-1]
        stages.append(
            nn.Sequential(
                ConvBlock(
                    hidden_dim // 2,
                    hidden_dim,
                    kernel_size=5,
                    padding=2,
                    stride=2,
                    act_func=self.act_func,
                    device=self.device,
                ),
                *[
                    ConvBlock(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        act_func=self.act_func,
                        device=self.device,
                    )
                    for _ in range(2)
                ],
            )
        )
        self.eps_stages = stages

    def _build_layers(self):
        stem = nn.Sequential(
            ConvBlock(
                self.in_channels,
                self.hidden_list[0] // 4,
                kernel_size=3,
                padding=1,
                stride=1,
                act_func=self.act_func,
                device=self.device,
            ),
            ConvBlock(
                self.hidden_list[0] // 4,
                self.hidden_list[0] // 2,
                kernel_size=5,
                padding=2,
                stride=2,
                act_func=None,
                device=self.device,
            ),
        )
        stages = nn.ModuleList()
        hidden_dim = self.hidden_list[-1] // 2
        stages.append(
            FNO(
                n_modes=(self.mode1 * 2, self.mode2 * 2),
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                lifting_channels=hidden_dim,
                projection_channels=hidden_dim,
                hidden_channels=hidden_dim,
                n_layers=2,
                norm=None,
                factorization=None,
            )
            if not self.fno_block_only
            else FNOBlocks(
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
                n_layers=2,
            )
        )
        hidden_dim = self.hidden_list[-1]
        stages.append(
            nn.Sequential(
                ConvBlock(
                    hidden_dim // 2,
                    hidden_dim,
                    kernel_size=5,
                    padding=2,
                    stride=2,
                    act_func=None,
                    device=self.device,
                ),
                FNO(
                    n_modes=(self.mode1, self.mode2),
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    lifting_channels=hidden_dim,
                    projection_channels=hidden_dim,
                    hidden_channels=hidden_dim,
                    n_layers=2,
                    norm=None,
                    factorization=None,
                )
                if not self.fno_block_only
                else FNOBlocks(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    n_modes=(self.mode1, self.mode2),
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
                    n_layers=2,
                ),
            )
        )

        head = ConvBlock(
            hidden_dim // 2 + hidden_dim,
            self.out_channels,
            kernel_size=1,
            padding=0,
            ln=False,
            act_func=None,
            device=self.device,
        )
        if self.err_correction:
            stem_err_corr = nn.Sequential(
                ConvBlock(
                    self.in_channels + 2,
                    self.hidden_list[0] // 4,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    act_func=self.act_func,
                    device=self.device,
                ),
                ConvBlock(
                    self.hidden_list[0] // 4,
                    self.hidden_list[0] // 2,
                    kernel_size=5,
                    padding=2,
                    stride=2,
                    act_func=None,
                    device=self.device,
                ),
            )
            stages_err_corr = nn.ModuleList()
            hidden_dim = self.hidden_list[-1] // 2
            stages_err_corr.append(
                FNO(
                    n_modes=(self.mode1 * 2, self.mode2 * 2),
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    lifting_channels=hidden_dim,
                    projection_channels=hidden_dim,
                    hidden_channels=hidden_dim,
                    n_layers=2,
                    norm=None,
                    factorization=None,
                )
                if not self.fno_block_only
                else FNOBlocks(
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
                    n_layers=2,
                )
            )
            hidden_dim = self.hidden_list[-1]
            stages_err_corr.append(
                nn.Sequential(
                    ConvBlock(
                        hidden_dim // 2,
                        hidden_dim,
                        kernel_size=2,
                        padding=1,
                        stride=2,
                        act_func=None,
                        device=self.device,
                    ),
                    FNO(
                        n_modes=(self.mode1, self.mode2),
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        lifting_channels=hidden_dim,
                        projection_channels=hidden_dim,
                        hidden_channels=hidden_dim,
                        n_layers=2,
                        norm=None,
                        factorization=None,
                    )
                    if not self.fno_block_only
                    else FNOBlocks(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        n_modes=(self.mode1, self.mode2),
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
                        n_layers=2,
                    ),
                )
            )

            head_err_corr = ConvBlock(
                hidden_dim // 2 + hidden_dim,
                self.out_channels,
                kernel_size=1,
                padding=0,
                ln=False,
                act_func=None,
                device=self.device,
            )

        if not self.err_correction:
            return stem, stages, head
        else:
            return stem, stages, head, stem_err_corr, stages_err_corr, head_err_corr

    def build_layers(self):
        if self.err_correction:
            (
                self.stem,
                self.stages,
                self.head,
                self.stem_err_corr,
                self.stages_err_corr,
                self.head_err_corr,
            ) = self._build_layers()
        else:
            self.stem, self.stages, self.head = self._build_layers()

        # self.stem = nn.Sequential(
        #     ConvBlock(
        #         self.in_channels,
        #         self.hidden_list[0] // 4,
        #         kernel_size=3,
        #         padding=1,
        #         stride=1,
        #         act_func=self.act_func,
        #         device=self.device,
        #     ),
        #     ConvBlock(
        #         self.hidden_list[0] // 4,
        #         self.hidden_list[0] // 2,
        #         kernel_size=5,
        #         padding=2,
        #         stride=2,
        #         act_func=None,
        #         device=self.device,
        #     ),
        # )
        # self.stages = nn.ModuleList()
        # hidden_dim = self.hidden_list[-1] // 2
        # self.stages.append(
        #     FNO(
        #         n_modes=(self.mode1*2, self.mode2*2),
        #         in_channels=hidden_dim,
        #         out_channels=hidden_dim,
        #         lifting_channels=hidden_dim,
        #         projection_channels=hidden_dim,
        #         hidden_channels=hidden_dim,
        #         n_layers=2,
        #         norm=None,
        #         factorization=None,
        #     ) if not self.fno_block_only else FNOBlocks(
        #         in_channels=hidden_dim,
        #         out_channels=hidden_dim,
        #         n_modes=(self.mode1*2, self.mode2*2),
        #         output_scaling_factor=None,
        #         use_mlp=False,
        #         mlp_dropout=0,
        #         mlp_expansion=0.5,
        #         non_linearity=F.gelu,
        #         stabilizer=None,
        #         norm=None,
        #         preactivation=False,
        #         fno_skip="linear",
        #         mlp_skip="soft-gating",
        #         max_n_modes=None,
        #         fno_block_precision="full",
        #         rank=1.0,
        #         fft_norm="forward",
        #         fixed_rank_modes=False,
        #         implementation="factorized",
        #         separable=False,
        #         factorization=None,
        #         decomposition_kwargs=dict(),
        #         joint_factorization=False,
        #         SpectralConv=SpectralConv,
        #         n_layers=2,
        #     )
        # )
        # hidden_dim = self.hidden_list[-1]
        # self.stages.append(
        #     nn.Sequential(
        #         ConvBlock(
        #         hidden_dim // 2,
        #         hidden_dim,
        #         kernel_size=2,
        #         padding=1,
        #         stride=2,
        #         act_func=None,
        #         device=self.device,
        #     ),
        #     FNO(
        #         n_modes=(self.mode1, self.mode2),
        #         in_channels=hidden_dim,
        #         out_channels=hidden_dim,
        #         lifting_channels=hidden_dim,
        #         projection_channels=hidden_dim,
        #         hidden_channels=hidden_dim,
        #         n_layers=2,
        #         norm=None,
        #         factorization=None,
        #     ) if not self.fno_block_only else FNOBlocks(
        #         in_channels=hidden_dim,
        #         out_channels=hidden_dim,
        #         n_modes=(self.mode1, self.mode2),
        #         output_scaling_factor=None,
        #         use_mlp=False,
        #         mlp_dropout=0,
        #         mlp_expansion=0.5,
        #         non_linearity=F.gelu,
        #         stabilizer=None,
        #         norm=None,
        #         preactivation=False,
        #         fno_skip="linear",
        #         mlp_skip="soft-gating",
        #         max_n_modes=None,
        #         fno_block_precision="full",
        #         rank=1.0,
        #         fft_norm="forward",
        #         fixed_rank_modes=False,
        #         implementation="factorized",
        #         separable=False,
        #         factorization=None,
        #         decomposition_kwargs=dict(),
        #         joint_factorization=False,
        #         SpectralConv=SpectralConv,
        #         n_layers=2,
        #     ),
        #     )
        # )

        # self.head = ConvBlock(
        #     hidden_dim // 2+hidden_dim,
        #     self.out_channels,
        #     kernel_size=1,
        #     padding=0,
        #     ln=False,
        #     act_func=None,
        #     device=self.device,
        # )
        # if self.err_correction:
        #     self.stem_err_corr = nn.Sequential(
        #         ConvBlock(
        #             self.in_channels + 2,
        #             self.hidden_list[0] // 4,
        #             kernel_size=3,
        #             padding=1,
        #             stride=1,
        #             act_func=self.act_func,
        #             device=self.device,
        #         ),
        #         ConvBlock(
        #             self.hidden_list[0] // 4,
        #             self.hidden_list[0] // 2,
        #             kernel_size=5,
        #             padding=2,
        #             stride=2,
        #             act_func=None,
        #             device=self.device,
        #         ),
        #     )
        #     self.stages_err_corr = nn.ModuleList()
        #     hidden_dim = self.hidden_list[-1] // 2
        #     self.stages_err_corr.append(
        #         FNO(
        #             n_modes=(self.mode1*2, self.mode2*2),
        #             in_channels=hidden_dim,
        #             out_channels=hidden_dim,
        #             lifting_channels=hidden_dim,
        #             projection_channels=hidden_dim,
        #             hidden_channels=hidden_dim,
        #             n_layers=2,
        #             norm=None,
        #             factorization=None,
        #         ) if not self.fno_block_only else FNOBlocks(
        #             in_channels=hidden_dim,
        #             out_channels=hidden_dim,
        #             n_modes=(self.mode1*2, self.mode2*2),
        #             output_scaling_factor=None,
        #             use_mlp=False,
        #             mlp_dropout=0,
        #             mlp_expansion=0.5,
        #             non_linearity=F.gelu,
        #             stabilizer=None,
        #             norm=None,
        #             preactivation=False,
        #             fno_skip="linear",
        #             mlp_skip="soft-gating",
        #             max_n_modes=None,
        #             fno_block_precision="full",
        #             rank=1.0,
        #             fft_norm="forward",
        #             fixed_rank_modes=False,
        #             implementation="factorized",
        #             separable=False,
        #             factorization=None,
        #             decomposition_kwargs=dict(),
        #             joint_factorization=False,
        #             SpectralConv=SpectralConv,
        #             n_layers=2,
        #         )
        #     )
        #     hidden_dim = self.hidden_list[-1]
        #     self.stages_err_corr.append(
        #         nn.Sequential(
        #             ConvBlock(
        #             hidden_dim // 2,
        #             hidden_dim,
        #             kernel_size=2,
        #             padding=1,
        #             stride=2,
        #             act_func=None,
        #             device=self.device,
        #         ),
        #         FNO(
        #             n_modes=(self.mode1, self.mode2),
        #             in_channels=hidden_dim,
        #             out_channels=hidden_dim,
        #             lifting_channels=hidden_dim,
        #             projection_channels=hidden_dim,
        #             hidden_channels=hidden_dim,
        #             n_layers=2,
        #             norm=None,
        #             factorization=None,
        #         ) if not self.fno_block_only else FNOBlocks(
        #             in_channels=hidden_dim,
        #             out_channels=hidden_dim,
        #             n_modes=(self.mode1, self.mode2),
        #             output_scaling_factor=None,
        #             use_mlp=False,
        #             mlp_dropout=0,
        #             mlp_expansion=0.5,
        #             non_linearity=F.gelu,
        #             stabilizer=None,
        #             norm=None,
        #             preactivation=False,
        #             fno_skip="linear",
        #             mlp_skip="soft-gating",
        #             max_n_modes=None,
        #             fno_block_precision="full",
        #             rank=1.0,
        #             fft_norm="forward",
        #             fixed_rank_modes=False,
        #             implementation="factorized",
        #             separable=False,
        #             factorization=None,
        #             decomposition_kwargs=dict(),
        #             joint_factorization=False,
        #             SpectralConv=SpectralConv,
        #             n_layers=2,
        #         ),
        #         )
        #     )

        #     self.head_err_corr = ConvBlock(
        #         hidden_dim // 2+hidden_dim,
        #         self.out_channels,
        #         kernel_size=1,
        #         padding=0,
        #         ln=False,
        #         act_func=None,
        #         device=self.device,
        #     )

        # if self.train_field == "both" or self.train_field == "adj":
        #     self.stem_adj = copy.deepcopy(self.stem)
        #     self.stages_adj = copy.deepcopy(self.stages)
        #     self.head_adj = copy.deepcopy(self.head)

        #     if self.err_correction:
        #         self.stem_err_corr_adj = copy.deepcopy(self.stem_err_corr)
        #         self.stages_err_corr_adj = copy.deepcopy(self.stages_err_corr)
        #         self.head_err_corr_adj = copy.deepcopy(self.head_err_corr)

        if self.aux_head:
            hidden_list = [self.kernel_list[self.aux_head_idx]] + self.hidden_list
            head = [
                nn.Sequential(
                    ConvBlock(
                        inc,
                        outc,
                        kernel_size=1,
                        padding=0,
                        act_func=self.act_func,
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
                    act_func=None,
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

    def fouier_feature_mapping(self, x: Tensor) -> Tensor:
        if self.fouier_feature == "none":
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

    @lru_cache(maxsize=16)
    def _get_linear_pos_enc(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.arange(0, size_x, device=device)
        gridy = torch.arange(0, size_y, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy)
        mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
        return mesh

    def from_Ez_to_Hx_Hy(self, eps: Tensor, Ez: Tensor) -> None:
        # eps b, h, w
        # Ez b, 2, h, w
        Ez = Ez.permute(0, 2, 3, 1).contiguous()
        Ez = torch.view_as_complex(Ez)
        Hx = []
        Hy = []
        for i in range(Ez.size(0)):
            self.sim.eps_r = eps[i]
            Hx_vec, Hy_vec = self.sim._Ez_to_Hx_Hy(Ez[i].flatten())
            Hx.append(torch.view_as_real(Hx_vec.reshape(Ez[i].shape)).permute(2, 0, 1))
            Hy.append(torch.view_as_real(Hy_vec.reshape(Ez[i].shape)).permute(2, 0, 1))
        Hx = torch.stack(Hx, 0)
        Hy = torch.stack(Hy, 0)
        return Hx, Hy

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

            # plt.figure()
            # plt.imshow(torch.abs(src[0]).detach().cpu().numpy())
            # plt.savefig("./figs/src.png")
            # plt.close()

            # plt.figure()
            # plt.imshow(mode[0].real.detach().cpu().numpy())
            # plt.savefig("./figs/incident_field_real.png")
            # plt.close()

            # plt.figure()
            # plt.imshow(mode[0].imag.detach().cpu().numpy())
            # plt.savefig("./figs/incident_field_imag.png")
            # plt.close()
            # quit()
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
            # plt.figure()
            # plt.imshow(torch.abs(src[0]).detach().cpu().numpy())
            # plt.savefig("./figs/src.png")
            # plt.close()

            # plt.figure()
            # plt.imshow(mode[0].real.detach().cpu().numpy())
            # plt.savefig("./figs/incident_field_real.png")
            # plt.close()

            # plt.figure()
            # plt.imshow(mode[0].imag.detach().cpu().numpy())
            # plt.savefig("./figs/incident_field_imag.png")
            # plt.close()
            # quit()
        return mode

    def forward(
        self,
        eps,
        src,
    ):
        # src and adj_src are all complex numbers tensor
        # if self.train_field == "fwd":
        #     src = src["source_profile-wl-1.55-port-in_port_1-mode-1"]
        # elif self.train_field == "adj":
        #     src = src["adj_src-wl-1.55-port-in_port_1-mode-1"]
        # the incident field should calculate in model using src
        incident_field_fwd = self.incident_field_from_src(src)
        incident_field_fwd = torch.view_as_real(incident_field_fwd).permute(
            0, 3, 1, 2
        )  # B, 2, H, W
        incident_field_fwd = incident_field_fwd / (
            torch.abs(incident_field_fwd).amax(dim=(1, 2, 3), keepdim=True) + 1e-6
        )
        src = torch.view_as_real(src).permute(0, 3, 1, 2)  # B, 2, H, W
        src = src / (torch.abs(src).amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

        # plt.figure()
        # plt.imshow(incident_field[0][0].detach().cpu().numpy())
        # plt.savefig("./figs/incident_field_real.png")
        # plt.close()

        # plt.figure()
        # plt.imshow(incident_field[0][1].detach().cpu().numpy())
        # plt.savefig("./figs/incident_field_imag.png")
        # plt.close()
        # quit()

        # eps_copy = eps.clone()
        # eps = (
        #     1 / eps
        # )  # take the inverse of the permittivity to easy the training difficulty
        eps = eps / 12.11
        eps = eps.unsqueeze(1)  # B, 1, H, W

        ## eps_branch
        if hasattr(self, "eps_stem"):
            eps_0 = self.eps_stem(eps)
            eps_1 = self.eps_stages[0](eps_0)
            eps_2 = self.eps_stages[1](eps_1)
        else:
            eps_1 = eps_2 = eps_0 = None

        if self.fouier_feature == "learnable":
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
            enc_fwd = self.fouier_feature_mapping(eps)
            eps_enc_fwd = torch.cat((eps, enc_fwd), dim=1)

        x_fwd = torch.cat((eps_enc_fwd, incident_field_fwd), dim=1)

        x_fwd = self.stem(x_fwd)  # conv2d downsample

        if eps_0 is not None:
            x_fwd = x_fwd * eps_0

        x1_fwd = self.stages[0](x_fwd)  # fno block

        if eps_1 is not None:
            x1_fwd = x1_fwd * eps_1

        x2_fwd = self.stages[1](x1_fwd)  # sequential conv2d downsample + fno block

        if eps_2 is not None:
            x2_fwd = x2_fwd * eps_2

        x1_fwd = resize_to_targt_size(x1_fwd, (src.shape[-2], src.shape[-1]))
        if len(x1_fwd.shape) == 3:
            x1_fwd = x1_fwd.unsqueeze(0)
        x2_fwd = resize_to_targt_size(x2_fwd, (src.shape[-2], src.shape[-1]))
        if len(x2_fwd.shape) == 3:
            x2_fwd = x2_fwd.unsqueeze(0)
        x_fwd = torch.cat((x1_fwd, x2_fwd), dim=1)
        forward_Ez_field = self.head(x_fwd)  # 1x1 conv
        if len(forward_Ez_field.shape) == 3:
            forward_Ez_field = forward_Ez_field.unsqueeze(0)

        if self.err_correction:
            forward_Ez_field_copy = forward_Ez_field.detach()
            x = torch.cat((eps, incident_field_fwd, forward_Ez_field_copy), dim=1)
            x = self.stem_err_corr(x)
            x1 = self.stages_err_corr[0](x)
            x2 = self.stages_err_corr[1](x1)
            x1 = resize_to_targt_size(x1, (src.shape[-2], src.shape[-1]))
            if len(x1.shape) == 3:
                x1 = x1.unsqueeze(0)
            x2 = resize_to_targt_size(x2, (src.shape[-2], src.shape[-1]))
            if len(x2.shape) == 3:
                x2 = x2.unsqueeze(0)
            x = torch.cat((x1, x2), dim=1)
            forward_Ez_field_err_corr = self.head_err_corr(x) + forward_Ez_field_copy
        # ------------------------------------------
        if self.err_correction:
            return forward_Ez_field, forward_Ez_field_err_corr
        else:
            return forward_Ez_field
        # calculate the hx and hy from the Ez field
        forward_Hx_field, forward_Hy_field = self.from_Ez_to_Hx_Hy(
            eps_copy, forward_Ez_field
        )
        forward_field = torch.cat(
            (forward_Hx_field, forward_Hy_field, forward_Ez_field), dim=1
        )

        if self.err_correction:
            forward_Hx_field_err_corr, forward_Hy_field_err_corr = (
                self.from_Ez_to_Hx_Hy(eps_copy, forward_Ez_field_err_corr)
            )
            forward_field_err_corr = torch.cat(
                (
                    forward_Hx_field_err_corr,
                    forward_Hy_field_err_corr,
                    forward_Ez_field_err_corr,
                ),
                dim=1,
            )
            forward_field_err_corr = (
                self.pml_mask.unsqueeze(0).unsqueeze(0) * forward_field_err_corr
            )

        forward_field = self.pml_mask.unsqueeze(0).unsqueeze(0) * forward_field
