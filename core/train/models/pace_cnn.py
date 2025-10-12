"""
 # @ Author: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Create Time: 2024-03-26 16:05:02
 # @ Modified by: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Modified time: 2024-10-21 02:29:59
 # @ Description: Add hirechy information into the device simulation.
 """

# Standard Library Imports
from typing import Dict, Optional, Tuple

import numpy as np

# Third-party Imports
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

# Local Application Imports
from pyutils.activation import Swish
from timm.models.layers import DropPath, to_2tuple
from torch import nn
from torch.functional import Tensor
from torch.types import Device, _size
from torch.utils.checkpoint import checkpoint

from .constant import *
from .layers.activation import SIREN
from .layers.neurolight_conv2d import NeurOLightConv2d as FFNOConv2d
from .layers.pace_conv2d import PACEConv2d_Split2x, PACEConv2d_Split4x
from .pde_base import PDE_NN_BASE

__all__ = ["PACE2d"]


def add_prefix_to_keys(config: Dict[str, any], prefix: str = "aux_") -> Dict[str, any]:
    """
    Add the specified prefix to the keys in a dictionary if they don't already have it.

    Args:
        config (Dict[str, any]): The configuration dictionary with keys to process.
        prefix (str): The prefix to add to the keys. Default is "aux_".

    Returns:
        Dict[str, any]: A new dictionary with the prefix added to the keys.
    """
    new_config = {}
    for key, value in config.items():
        # Add prefix if the key doesn't already have it
        if not key.startswith(prefix):
            new_key = prefix + key
        else:
            new_key = key  # Retain the key if it already has the prefix
        new_config[new_key] = value
    return new_config


def del_prefix_to_keys(config: Dict[str, any], prefix: str = "aux_") -> Dict[str, any]:
    """
    Remove the specified prefix from the keys in a dictionary if they have it.

    Args:
        config (Dict[str, any]): The configuration dictionary with keys to process.
        prefix (str): The prefix to remove from the keys. Default is "aux_".

    Returns:
        Dict[str, any]: A new dictionary with the prefix removed from the keys.
    """
    new_config = {}
    for key, value in config.items():
        # Remove prefix if the key starts with the specified prefix
        if key.startswith(prefix):
            new_key = key[len(prefix) :]  # Remove the prefix by slicing
        else:
            new_key = key  # Retain the key if it doesn't have the prefix
        new_config[new_key] = value
    return new_config


class DWConvFFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        expansion=2,
        act_func="gelu",
        norm_func="bn",
        ffn_inverted=False,
    ):
        super().__init__()
        self.expansion = expansion
        self.act_func = self._select_activation(act_func)

        # Define the convolutional layers
        if not ffn_inverted:
            self.conv1 = nn.Conv2d(in_channels, out_channels * self.expansion, 1)
            self.conv2 = nn.Conv2d(
                out_channels * self.expansion,
                out_channels * self.expansion,
                kernel_size,
                groups=out_channels * self.expansion,
                padding=kernel_size // 2,
            )
        else:
            raise NotImplementedError("Not implemented yet")

        self.conv3 = nn.Conv2d(out_channels * self.expansion, out_channels, 1)

        self.norm = nn.Sequential(
            *self._select_norm(norm_func, out_channels * self.expansion)
        )

    def _select_activation(self, act_func):
        if act_func.lower() == "gelu":
            return nn.GELU()
        elif act_func.lower() == "relu":
            return nn.ReLU()
        else:
            raise ValueError("Unsupported activation function: {}".format(act_func))

    def _select_norm(self, norm_func, channels):
        if norm_func is None:
            return (nn.Identity(),)
        elif norm_func.lower() == "bn":
            return (nn.BatchNorm2d(channels),)
        elif norm_func.lower() == "in":
            return (nn.InstanceNorm2d(channels),)
        elif norm_func.lower() == "ln":
            return (
                Rearrange("b c nx ny -> b nx ny c"),
                nn.LayerNorm(channels),
                Rearrange("b nx ny c -> b c nx ny"),
            )
        else:
            raise ValueError(f"Unsupported normalization function: {norm_func}")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.norm(x)

        x = self.act_func(x)
        x = self.conv3(x)

        return x


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
        elif act_func.lower() == "siren":
            self.act_func = SIREN()
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
        padding = [
            (dilation[i] * (kernel_size[i] - 1) + 1) // 2
            for i in range(len(kernel_size))
        ]
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


# NOTE(hqzhu): add choices for nonlinear function inside resstem
class ResStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size = 3,
        stride: _size = 1,
        dilation: _size = 1,
        groups: int = 1,
        act_func: Optional[str] = "GELU",
        norm_func: Optional[str] = "BN",
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [
            (dilation[i] * (kernel_size[i] - 1) + 1) // 2
            for i in range(len(kernel_size))
        ]

        self.conv1 = BSConv2d(
            in_channels,
            out_channels // 2,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        # self.bn1 = nn.BatchNorm2d(out_channels // 2)

        if norm_func is None:
            self.norm1 = None
        elif norm_func.lower() == "bn":
            self.norm1 = nn.BatchNorm2d(out_channels // 2)
        elif norm_func.lower() == "in":
            self.norm1 = nn.InstanceNorm2d(out_channels // 2)
        elif norm_func.lower() == "ln":
            self.norm1 = nn.LayerNorm(out_channels // 2)
        else:
            raise ValueError(f"Unsupported normalization function: {norm_func}")

        if act_func is None:
            self.act1 = None
        elif act_func.lower() == "siren":
            self.act1 = SIREN()
        elif act_func.lower() == "swish":
            self.act1 = Swish()
        else:
            self.act1 = getattr(nn, act_func)()

        self.conv2 = BSConv2d(
            out_channels // 2,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        # self.bn2 = nn.BatchNorm2d(out_channels)
        if norm_func is None:
            self.norm2 = None
        elif norm_func.lower() == "bn":
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm_func.lower() == "in":
            self.norm2 = nn.InstanceNorm2d(out_channels)
        elif norm_func.lower() == "ln":
            self.norm2 = nn.LayerNorm(out_channels)
        else:
            raise ValueError(f"Unsupported normalization function: {norm_func}")

        # self.act2 = nn.ReLU(inplace=True)
        if act_func is None:
            self.act2 = None
        elif act_func.lower() == "siren":
            self.act2 = SIREN()
        elif act_func.lower() == "swish":
            self.act2 = Swish()
        else:
            self.act2 = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(self.norm1, nn.LayerNorm):
            x = self.conv1(x)
            b, inc, h, w = x.shape
            x = x.permute(0, 2, 3, 1)
            x = self.norm1(x)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act1(x)

            x = self.conv2(x)
            b, inc, h, w = x.shape
            x = x.permute(0, 2, 3, 1)
            x = self.norm2(x)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act2(x)

            # x = self.act2(self.norm2(x)).reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        else:
            x = self.act1(self.norm1(self.conv1(x)))
            x = self.act2(self.norm2(self.conv2(x)))
        return x


class ResStem2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: _size = 3,
        stride: _size = 1,
        dilation: _size = 1,
        groups: int = 1,
        act_func: Optional[str] = "GELU",
        norm_func: Optional[str] = "BN",
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [
            (dilation[i] * (kernel_size[i] - 1) + 1) // 2
            for i in range(len(kernel_size))
        ]

        self.conv1 = BSConv2d(
            in_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        # self.bn1 = nn.BatchNorm2d(out_channels // 2)

        if norm_func is None:
            self.norm1 = None
        elif norm_func.lower() == "bn":
            self.norm1 = nn.BatchNorm2d(mid_channels)
        elif norm_func.lower() == "in":
            self.norm1 = nn.InstanceNorm2d(mid_channels)
        elif norm_func.lower() == "ln":
            self.norm1 = nn.LayerNorm(mid_channels)
        else:
            raise ValueError(f"Unsupported normalization function: {norm_func}")

        if act_func is None:
            self.act1 = None
        elif act_func.lower() == "siren":
            self.act1 = SIREN()
        elif act_func.lower() == "swish":
            self.act1 = Swish()
        else:
            self.act1 = getattr(nn, act_func)()

        self.conv2 = BSConv2d(
            mid_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        # self.bn2 = nn.BatchNorm2d(out_channels)
        if norm_func is None:
            self.norm2 = None
        elif norm_func.lower() == "bn":
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm_func.lower() == "in":
            self.norm2 = nn.InstanceNorm2d(out_channels)
        elif norm_func.lower() == "ln":
            self.norm2 = nn.LayerNorm(out_channels)
        else:
            raise ValueError(f"Unsupported normalization function: {norm_func}")

        # self.act2 = nn.ReLU(inplace=True)
        if act_func is None:
            self.act2 = None
        elif act_func.lower() == "siren":
            self.act2 = SIREN()
        elif act_func.lower() == "swish":
            self.act2 = Swish()
        else:
            self.act2 = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(self.norm1, nn.LayerNorm):
            x = self.conv1(x)
            b, inc, h, w = x.shape
            x = x.permute(0, 2, 3, 1)
            x = self.act1(self.norm1(x)).permute(0, 3, 1, 2).contiguous()

            x = self.conv2(x)
            b, inc, h, w = x.shape
            x = x.permute(0, 2, 3, 1)
            x = self.act2(self.norm2(x)).permute(0, 3, 1, 2).contiguous()
        else:
            x = self.act1(self.norm1(self.conv1(x)))
            x = self.act2(self.norm2(self.conv2(x)))
        return x


class PACE2dLayer(nn.Module):
    def __init__(self, dim, n_modes=(40, 70), act_func="GELU", module_type="CFNO"):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)  # Projection layer
        self.act_func = self._select_activation(
            act_func
        )  # Select the activation function

        # Conditional module initialization based on module_type
        if module_type.lower() == "pace_4x":
            self.pace_conv2d = PACEConv2d_Split4x(dim, dim, n_modes)
        elif module_type.lower() == "pace_2x":
            self.pace_conv2d = PACEConv2d_Split2x(dim, dim, n_modes)
        else:
            raise ValueError(f"Unsupported module type: {module_type}")

        self.conv2 = nn.Conv2d(dim, dim, 1)  # Second convolutional layer

        # Third sequential convolution with a sigmoid function
        self.conv3 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def _select_activation(self, act_func):
        """Helper function to select the activation function."""
        if act_func is None:
            return None
        elif act_func.lower() == "siren":
            return SIREN()  # Custom activation (assumed to be defined elsewhere)
        elif act_func.lower() == "swish":
            return Swish()  # Custom activation (assumed to be defined elsewhere)
        elif hasattr(
            nn, act_func
        ):  # Check if the activation is part of PyTorch's nn module
            return getattr(nn, act_func)()
        else:
            raise ValueError(f"Unsupported activation function: {act_func}")

    def forward(self, x):
        """Forward pass for the PACE2dBlock."""
        # Initial projection and optional activation
        x = self.proj_1(x)
        if self.act_func:
            x = self.act_func(x)

        u = x  # Save a copy of the projected input

        # Apply PACE Conv2D layer
        f1 = self.pace_conv2d(x)

        # Apply second convolutional layer
        f2 = self.conv2(f1)

        # Apply element-wise multiplication with sigmoid output
        y = f2 * self.conv3(u)

        return y


class PACE2dBlock(nn.Module):
    expansion = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_func: Optional[str] = "RELU",
        norm_func: Optional[str] = "BN",
        kernel_size: int = 3,
        drop_path_rate: float = 0.0,
        n_modes: Tuple[int] = (40, 70),
        device: Device = torch.device("cuda:0"),
        with_cp=False,
        ffn: bool = True,
        ffn_dwconv: bool = True,
        aug_path: bool = False,
        fno_bias: bool = False,
        module_type: str = "pace_4x",
        block_skip: bool = False,
        layer_skip: bool = True,
        pre_norm_fno: bool = False,
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )

        # skip connection definition
        self.block_skip = block_skip
        self.layer_skip = layer_skip

        self.pace_block = PACE2dLayer(
            dim=in_channels,
            n_modes=n_modes,
            module_type=module_type,
        )

        ## three normalization layers choices
        ## pre_norm_fno: put norm before fno
        self.pre_norm_fno = (
            nn.Sequential(*self._select_norm(norm_func, out_channels))
            if pre_norm_fno
            else nn.Identity()
        )
        self.pre_norm = nn.Sequential(*self._select_norm(norm_func, in_channels))
        self.norm = nn.Sequential(*self._select_norm(norm_func, out_channels))

        self.with_cp = with_cp

        if ffn:
            if ffn_dwconv:
                # inverted or not, large kernel if it is inverted
                self.ff = DWConvFFN(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    expansion=self.expansion,
                    act_func=act_func,
                    norm_func=norm_func,
                )
            else:
                self.ff = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels * self.expansion, 1),
                    (
                        nn.BatchNorm2d(out_channels * self.expansion)
                        if norm_func.lower() == "bn"
                        else nn.InstanceNorm2d(out_channels * self.expansion)
                    ),
                    self._select_activation(act_func),
                    nn.Conv2d(out_channels * self.expansion, out_channels, 1),
                )
        else:
            self.ff = None
        if aug_path:
            self.aug_path = nn.Sequential(
                BSConv2d(in_channels, out_channels, 3), nn.GELU()
            )
        else:
            self.aug_path = None

        self.act_func = self._select_activation(act_func)

    def _select_activation(self, act_func):
        if act_func is None:
            return nn.Identity()
        elif act_func.lower() == "siren":
            return SIREN()
        elif act_func.lower() == "swish":
            return Swish()
        return getattr(nn, act_func)()

    def _select_norm(self, norm_func, channels):
        if norm_func is None:
            return (nn.Identity(),)
        elif norm_func.lower() == "bn":
            return (nn.BatchNorm2d(channels),)
        elif norm_func.lower() == "in":
            return (nn.InstanceNorm2d(channels),)
        elif norm_func.lower() == "ln":
            return (
                Rearrange("b c nx ny -> b nx ny c"),
                nn.LayerNorm(channels),
                Rearrange("b nx ny c -> b c nx ny"),
            )
        else:
            raise ValueError(f"Unsupported normalization function: {norm_func}")

    def forward(self, x: Tensor) -> Tensor:
        def _inner_forward(x):
            y = x
            if self.ff is not None:
                x = self.pre_norm_fno(x)
                x = self.pace_block(x)  # operator block

                if self.block_skip:
                    x += y

                x = self.pre_norm(x)
                x = self.norm(self.ff(x))
                if self.layer_skip:
                    if self.aug_path is not None:
                        x = self.drop_path(x) + self.aug_path(y)
                    else:
                        x = self.drop_path(x) + y
            else:
                x = self.act_func(self.drop_path(self.norm(self.conv_attention(x))) + y)

            return x

        if x.requires_grad and self.with_cp:
            return checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)


class FFNO2dBlock(nn.Module):
    expansion = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        kernel_size: int = 1,  # size of depthwise conv
        padding: int = 0,
        act_func: Optional[str] = "RELU",
        norm_func: Optional[str] = "BN",
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        with_cp=False,
        ffn: bool = True,
        ffn_dwconv: bool = True,
        aug_path: bool = False,
        fno_bias: bool = False,
        block_skip: bool = False,
        layer_skip: bool = True,
        pre_norm_fno: bool = False,
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )
        self.block_skip = block_skip
        self.layer_skip = layer_skip

        self.f_conv = FFNOConv2d(in_channels, out_channels, n_modes, device=device)

        self.pre_norm = nn.Sequential(*self._select_norm(norm_func, in_channels))
        self.norm = nn.Sequential(*self._select_norm(norm_func, out_channels))
        self.pre_norm_fno = (
            nn.Sequential(*self._select_norm(norm_func, out_channels))
            if pre_norm_fno
            else nn.Identity()
        )

        self.with_cp = with_cp
        if ffn:
            if ffn_dwconv:
                self.ff = DWConvFFN(
                    in_channels,
                    out_channels,
                    kernel_size,
                    expansion=self.expansion,
                    norm_func=norm_func,
                    act_func=act_func,
                )
            else:
                self.ff = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels * self.expansion, 1),
                    (
                        nn.BatchNorm2d(out_channels * self.expansion)
                        if norm_func.lower() == "bn"
                        else nn.InstanceNorm2d(out_channels * self.expansion)
                    ),
                    # nn.GELU(),
                    self._select_activation(act_func),
                    nn.Conv2d(out_channels * self.expansion, out_channels, 1),
                )
        else:
            self.ff = None

        if aug_path:
            self.aug_path = nn.Sequential(
                BSConv2d(in_channels, out_channels, 3), nn.GELU()
            )
        else:
            self.aug_path = None

        self.act_func = self._select_activation(act_func)

    def _select_activation(self, act_func):
        if act_func is None:
            return nn.Identity()
        elif act_func.lower() == "siren":
            return SIREN()
        elif act_func.lower() == "swish":
            return Swish()
        return getattr(nn, act_func)()

    def _select_norm(self, norm_func, channels):
        if norm_func is None:
            return (nn.Identity(),)
        elif norm_func.lower() == "bn":
            return (nn.BatchNorm2d(channels),)
        elif norm_func.lower() == "in":
            return (nn.InstanceNorm2d(channels),)
        elif norm_func.lower() == "ln":
            return (
                Rearrange("b c nx ny -> b nx ny c"),
                nn.LayerNorm(channels),
                Rearrange("b nx ny c -> b c nx ny"),
            )
        else:
            raise ValueError(f"Unsupported normalization function: {norm_func}")

    def forward(self, x: Tensor) -> Tensor:
        def _inner_forward(x):
            y = x
            if self.ff is not None:
                x = self.pre_norm_fno(x)
                x = self.f_conv(x)  # operator block

                if self.block_skip:
                    x += y

                x = self.pre_norm(x)
                x = self.norm(self.ff(x))

                if self.layer_skip:
                    if self.aug_path is not None:
                        x = self.drop_path(x) + self.aug_path(y)
                    else:
                        x = self.drop_path(x) + y
            else:
                x = self.act_func(self.drop_path(self.norm(self.f_conv(x))) + y)
            return x

        if x.requires_grad and self.with_cp:
            return checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)


class PACE2d(PDE_NN_BASE):
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
        # PACE-I model
        in_channels: int = 1,
        out_channels: int = 2,
        # unique setting for field
        domain_size: Tuple[float] = [20, 100],  # computation domain in unit of um
        grid_step: float = 1.550
        / 20,  # grid step size in unit of um, typically 1/20 or 1/30 of the wavelength
        pml_width: float = 0,
        pml_permittivity: complex = 0 + 0j,
        buffer_width: float = 0.5,
        buffer_permittivity: complex = -1e-10 + 0j,
        eps_min: float = 1.0,
        eps_max: float = 12.3,
        pos_encoding: str = "exp",
        with_cp=False,
        # PACE-I config
        pace_config: Optional[dict] = None,
        # PACE-II model (auxiliary PACE head)
        aux_pace: bool = False,
        aux_pace_learn_residual: bool = False,
        aux_pace_aug_input: bool = False,
        aux_pace_aug_feature: bool = False,
        aux_pace_aug_feature_enhance: bool = True,
        aux_pace_config: Optional[dict] = None,
        device: torch.device = torch.device("cuda:0"),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert (
            out_channels % 2 == 0
        ), f"The output channels must be even number larger than 2, but got {out_channels}"

        self.ori_in_channels = in_channels

        if pos_encoding == "none":
            pass
        elif pos_encoding == "linear":
            self.in_channels += 2
        elif pos_encoding == "exp":
            self.in_channels += 4  # new wave prior right?
        elif pos_encoding == "exp_noeps":
            self.in_channels += 2
        elif pos_encoding == "exp3":
            self.in_channels += 6
        elif pos_encoding == "exp4":
            self.in_channels += 8
        elif pos_encoding in {"exp_full", "exp_full_r"}:
            self.in_channels += 7
        elif pos_encoding == "raw":
            self.in_channels += 3
        else:
            raise ValueError(
                f"pos_encoding only supports linear and exp, but got {pos_encoding}"
            )

        self.device = device

        self.padding = 9  # pad the domain if input is non-periodic

        # initialize the PACE-I model args
        self._initialize_pace_config(pace_config)

        # initialize the PACE-II model args
        self.aux_pace = aux_pace
        self.aux_pace_learn_residual = aux_pace_learn_residual
        self.aux_pace_aug_input = aux_pace_aug_input
        self.aux_pace_aug_feature = aux_pace_aug_feature
        self.aux_pace_aug_feature_enhance = aux_pace_aug_feature_enhance

        if self.aux_pace:
            self._initialize_aux_pace_config(aux_pace_config)

        self.domain_size = domain_size
        self.grid_step = grid_step
        self.domain_size_pixel = [round(i / grid_step) for i in domain_size]
        self.buffer_width = buffer_width
        self.buffer_permittivity = buffer_permittivity
        self.pml_width = pml_width
        self.pml_permittivity = pml_permittivity
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.pos_encoding = pos_encoding
        self.with_cp = with_cp

        self.build_layers()
        self.reset_parameters()

        self.permittivity_encoder = None
        self.set_trainable_permittivity(False)
        self.set_linear_probing_mode(False)

    def _initialize_pace_config(self, config: Optional[Dict]):
        """
        Initialize main PACE-specific configurations and assign them to `self`.
        """
        # Default main PACE configuration (without `pace_` prefix)
        default_pace_config = {
            "dim": 64,
            "kernel_list": [16, 16, 16, 16],
            "kernel_size_list": [1, 1, 1, 1],
            "padding_list": [0, 0, 0, 0],
            "hidden_list": [128],
            "pos": [5, 7, 9],
            "mode_list": [(20, 20), (20, 20), (20, 20), (20, 20)],
            "act_func": "GELU",
            "norm_func": "BN",
            "dropout_rate": 0.0,
            "drop_path_rate": 0.0,
            "conv_stem": True,
            "aug_path": True,
            "block_type": "ffno",
            "ffn": True,
            "ffn_dwconv": True,
            "fno_bias": False,
            "module_type": "pace_4x",
            "block_skip": True,
            "layer_skip": True,
            "pre_norm_fno": True,
        }

        # Update with any provided config values
        if config:
            default_pace_config.update(config)

        self.pace_config = default_pace_config

        # Assign each configuration option as a class attribute
        for key, value in default_pace_config.items():
            setattr(self, key, value)

    def _initialize_aux_pace_config(self, config: Optional[Dict]):
        """
        Initialize auxiliary PACE-specific configurations and assign them to `self`.
        """
        # Default auxiliary configuration
        default_aux_config = {
            "aux_pace_dim": 64,
            "aux_pace_kernel_list": [16, 16],
            "aux_pace_kernel_size_list": [1, 1],
            "aux_pace_padding_list": [0, 0],
            "aux_pace_hidden_list": [128],
            "aux_pace_pos": [5, 7, 9],
            "aux_pace_mode_list": [(20, 20), (20, 20)],
            "aux_pace_act_func": "GELU",
            "aux_pace_norm_func": "BN",
            "aux_pace_dropout_rate": 0.0,
            "aux_pace_drop_path_rate": 0.0,
            "aux_pace_conv_steam": True,
            "aux_pace_aug_path": False,
            "aux_pace_block_type": "ffno",
            "aux_pace_ffn": True,
            "aux_pace_ffn_dwconv": True,
            "aux_pace_fno_bias": False,
            "aux_pace_module_type": "pace_4x",
            "aux_pace_block_skip": False,
            "aux_pace_layer_skip": True,
        }

        # Update with any provided config values (without prefix)
        if config:
            prefixed_config = add_prefix_to_keys(config, prefix="aux_")
            default_aux_config.update(prefixed_config)

        self.default_aux_config = default_aux_config

        # Assign each configuration option as a class attribute
        for key, value in default_aux_config.items():
            setattr(self, key, value)

    def build_layers(self):
        if self.conv_stem:
            self.stem = ResStem(
                self.in_channels,
                self.dim,
                kernel_size=3,
                stride=1,
                act_func=self.act_func,
                norm_func=self.norm_func,
            )
        else:
            self.stem = nn.Conv2d(self.in_channels, self.dim, 1)
        kernel_list = [self.dim] + self.kernel_list
        drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))

        features = []
        fft_module_cnt = 0
        for i, (inc, outc, n_modes, kernel_size, padding, drop) in enumerate(
            zip(
                kernel_list[:-1],
                kernel_list[1:],
                self.mode_list,
                self.kernel_size_list,
                self.padding_list,
                drop_path_rates,
            )
        ):
            if i in self.pos:
                if n_modes[0] > 0:
                    features.append(
                        PACE2dBlock(
                            inc,
                            outc,
                            n_modes=n_modes,
                            kernel_size=kernel_size,
                            act_func=self.act_func,
                            norm_func=self.norm_func,
                            drop_path_rate=drop,
                            device=self.device,
                            with_cp=self.with_cp,
                            aug_path=self.aug_path,
                            ffn=self.ffn,
                            ffn_dwconv=self.ffn_dwconv,
                            module_type=self.module_type,
                            block_skip=self.block_skip,
                            layer_skip=self.layer_skip,
                            pre_norm_fno=self.pre_norm_fno,
                        )
                    )
                fft_module_cnt += 1
            else:
                features.append(
                    FFNO2dBlock(
                        inc,
                        outc,
                        n_modes,
                        kernel_size,
                        padding,
                        act_func=self.act_func,
                        norm_func=self.norm_func,
                        drop_path_rate=drop,
                        device=self.device,
                        with_cp=self.with_cp,
                        aug_path=self.aug_path,
                        ffn=self.ffn,
                        ffn_dwconv=self.ffn_dwconv,
                        fno_bias=self.fno_bias,
                        block_skip=self.block_skip,
                        layer_skip=self.layer_skip,
                        pre_norm_fno=self.pre_norm_fno,
                    )
                )

        self.features = nn.Sequential(*features)
        hidden_list = [self.kernel_list[-1]] + self.hidden_list
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
                self.out_channels,
                kernel_size=1,
                padding=0,
                act_func=None,
                device=self.device,
            )
        ]

        self.head = nn.Sequential(*head)
        self.aux_pace_mode = False
        if self.aux_pace:
            in_channels = self.out_channels

            self.aux_pace_model = PACE2d_refiner(
                in_channels,
                out_channels=self.out_channels,
                aug_feature=self.aux_pace_aug_feature,
                aug_input=self.aux_pace_aug_input,
                aug_feature_enhance=self.aux_pace_aug_feature_enhance,
                aug_input_dim=self.ori_in_channels // 2,
                aug_feature_dim=self.kernel_list[-1],
                learn_residual=self.aux_pace_learn_residual,
                pace_config=self.default_aux_config,
            )

            self.aug_feature = None
        else:
            self.aux_pace_model = None
            self.aug_feature = None

    def set_trainable_permittivity(self, mode: bool = True) -> None:
        self.trainable_permittivity = mode

    def set_aux_pace_mode(self, mode: bool = True) -> None:
        self.aux_pace_mode = mode

    def disable_aux_pace_params(self):
        if self.aux_pace_model:
            for param in self.aux_pace_model.parameters():
                param.requires_grad = False
        if self.aug_feature is not None:
            for param in self.aug_feature.parameters():
                param.requires_grad = False

    def enable_aux_pace_params(self):
        if self.aux_pace_model:
            for param in self.aux_pace_model.parameters():
                param.requires_grad = True
        if self.aug_feature is not None:
            for param in self.aug_feature.parameters():
                param.requires_grad = True

    def requires_network_params_grad(self, mode: float = True) -> None:
        params = (
            self.stem.parameters()
            + self.features.parameters()
            + self.head.parameters()
            + self.full_field_head.parameters()
        )
        for p in params:
            p.requires_grad_(mode)

    def forward(self, x: Tensor, wavelength: Tensor, grid_step: Tensor):
        # x [bs, inc, h, w] complex
        # wavelength [bs, 1] real
        # grid_step [bs, 2] real

        epsilon = (
            x[:, 0:1] * (self.eps_max - self.eps_min) + self.eps_min
        )  # this is de-normalized permittivity

        # convert complex permittivity/mode to real numbers
        if "noeps" in self.pos_encoding:  # no epsilon
            x = (
                torch.view_as_real(x[:, 1:]).permute(0, 1, 4, 2, 3).flatten(1, 2)
            )  # [bs, inc*2, h, w] real
        else:
            x = (
                torch.view_as_real(x).permute(0, 1, 4, 2, 3).flatten(1, 2)
            )  # [bs, inc*2, h, w] real

        # NOTE(hqzhu): this is used for wave prior
        grid = self.get_grid(
            x.shape,
            x.device,
            mode=self.pos_encoding,
            epsilon=epsilon,
            wavelength=wavelength,
            grid_step=grid_step,
        )  # [bs, 2 or 4 or 8, h, w] real

        y = x[:, 0:2]  # only pass eps to next stage, no input hints

        if grid is not None:
            x = torch.cat((x, grid), dim=1)  # [bs, inc*2+4, h, w] real

        if self.linear_probing_mode and self.aux_pace_mode:
            with torch.no_grad():
                # DNN-based electric field envelop prediction
                x = self.stem(x)
                x = self.features(x)
        else:
            x = self.stem(x)
            x = self.features(x)

        if self.aux_pace_mode and self.aux_pace_aug_feature:
            aux_x = x

        x = self.head(x)  # [bs, outc, h, w] real

        # go to PACE-II model
        if self.aux_pace_mode:
            main_x = x  # save the main output for the PACE-I model
            if self.aux_pace_aug_input and self.aux_pace_aug_feature:
                x = self.aux_pace_model(x, aux_x, input=y)
            elif self.aux_pace_aug_input and not self.aux_pace_aug_feature:
                x = self.aux_pace_model(x, None, input=y)
            elif not self.aux_pace_aug_input and self.aux_pace_aug_feature:
                x = self.aux_pace_model(x, aux_x, input=None)
            else:
                x = self.aux_pace_model(x, None, input=None)

            main_x = torch.view_as_complex(
                main_x.view(main_x.size(0), -1, 2, main_x.size(-2), main_x.size(-1))
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            return (x, main_x)

        # convert to complex frequency-domain electric field envelops
        x = torch.view_as_complex(
            x.view(x.size(0), -1, 2, x.size(-2), x.size(-1))
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # [bs, outc/2, h, w] complex

        return x


class PACE2d_refiner(PDE_NN_BASE):
    """Define the module to refine the residual between real output and the predicted output
    contains steam and features and new head. same as ffno2d block.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        aug_feature: bool = False,
        aug_input: bool = False,
        aug_feature_enhance: bool = True,
        aug_input_dim: int = 1,
        aug_feature_dim: int = 16,
        learn_residual: bool = False,
        # PACE-I model config
        pace_config: Optional[dict] = None,
        device: torch.device = torch.device("cuda:0"),
    ):
        super(PACE2d_refiner, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # initialize the PACE-I model args
        self._initialize_pace_config(pace_config)

        self.aug_feature = aug_feature
        self.aug_input = aug_input
        self.aug_feature_enhance = aug_feature_enhance
        self.learn_residual = learn_residual
        self.device = device

        if self.aug_feature and not self.aug_feature_enhance:
            self.in_channels += aug_feature_dim
        if self.aug_input:
            self.in_channels += aug_input_dim

        self.with_cp = False
        self.linear_probing_mode = False

        self.build_layers()
        self.reset_parameters()

    def _initialize_pace_config(self, config: Optional[Dict]):
        """
        Initialize main PACE-specific configurations and assign them to `self`.
        """
        # Default main PACE configuration (without `pace_` prefix)
        default_pace_config = {
            "dim": 64,
            "kernel_list": [16, 16, 16, 16],
            "kernel_size_list": [1, 1, 1, 1],
            "padding_list": [0, 0, 0, 0],
            "hidden_list": [128],
            "pos": [5, 7, 9],
            "mode_list": [(20, 20), (20, 20), (20, 20), (20, 20)],
            "act_func": "GELU",
            "norm_func": "BN",
            "dropout_rate": 0.0,
            "drop_path_rate": 0.0,
            "conv_stem": True,
            "aug_path": True,
            "block_type": "ffno",
            "ffn": True,
            "ffn_dwconv": True,
            "fno_bias": False,
            "module_type": "pace_4x",
            "block_skip": True,
            "layer_skip": True,
            "pre_norm_fno": True,
        }

        # Update with any provided config values
        if config:
            prefixed_config = del_prefix_to_keys(config, prefix="aux_")
            default_pace_config.update(prefixed_config)

        self.pace_config = default_pace_config

        # Assign each configuration option as a class attribute
        for key, value in default_pace_config.items():
            setattr(self, key, value)

    def _select_activation(self, act_func):
        if act_func is None:
            return None
        elif act_func.lower() == "siren":
            return SIREN()
        elif act_func.lower() == "swish":
            return Swish()
        return getattr(nn, act_func)()

    def build_layers(self):

        if self.aug_feature_enhance:
            # conv1x1 plus gelu to get high freq information to guide the information distillation
            self.aug_feature_enhancer = nn.Sequential(
                nn.Conv2d(self.dim, self.dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.aug_feature_enhancer = None

        if self.conv_stem:
            self.stem = ResStem2(
                in_channels=self.in_channels,
                mid_channels=self.dim // 2 if self.in_channels < self.dim else self.dim,
                out_channels=self.dim,
                kernel_size=3,
                stride=1,
                act_func=self.act_func,
                norm_func=self.norm_func,
            )
        else:
            self.stem = nn.Conv2d(self.in_channels, self.dim, 1)
        kernel_list = [self.dim] + self.kernel_list
        drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))

        features = []
        fft_module_cnt = 0
        for i, (inc, outc, n_modes, kernel_size, padding, drop) in enumerate(
            zip(
                kernel_list[:-1],
                kernel_list[1:],
                self.mode_list,
                self.kernel_size_list,
                self.padding_list,
                drop_path_rates,
            )
        ):
            # features.append(DilatedConv(num_channels=outc))
            # add the attention block here to enable global information exchangement
            # add into the later stage as in vit
            if i in self.pos:
                features.append(
                    PACE2dBlock(
                        inc,
                        outc,
                        n_modes=n_modes,
                        kernel_size=kernel_size,
                        act_func=self.act_func,
                        norm_func=self.norm_func,
                        drop_path_rate=drop,
                        device=self.device,
                        with_cp=self.with_cp,
                        aug_path=self.aug_path,
                        ffn=self.ffn,
                        ffn_dwconv=self.ffn_dwconv,
                        module_type=self.module_type,
                        block_skip=self.block_skip,
                        layer_skip=self.layer_skip,
                        pre_norm_fno=self.pre_norm_fno,
                    )
                )
                fft_module_cnt += 1
            else:
                features.append(
                    FFNO2dBlock(
                        inc,
                        outc,
                        n_modes,
                        kernel_size,
                        padding,
                        act_func=self.act_func,
                        norm_func=self.norm_func,
                        drop_path_rate=drop,
                        device=self.device,
                        with_cp=self.with_cp,
                        aug_path=self.aug_path,
                        ffn=self.ffn,
                        ffn_dwconv=self.ffn_dwconv,
                        fno_bias=self.fno_bias,
                        block_skip=self.block_skip,
                        layer_skip=self.layer_skip,
                        pre_norm_fno=self.pre_norm_fno,
                    )
                )

        self.features = nn.Sequential(*features)
        hidden_list = [self.kernel_list[-1]] + self.hidden_list
        # if we want to add the global feature to the head

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
                self.out_channels,
                kernel_size=1,
                padding=0,
                act_func=None,
                device=self.device,
            )
        ]
        self.head = nn.Sequential(*head)  # new head

    def forward(
        self, x: Tensor, aug_feature: Tensor = None, input: Tensor = None
    ) -> Tensor:
        """get input from
        x (output from last head)
        aug_feature: (output from the features)
        input: (input raw feature)
        """
        if x.is_complex():
            x = (
                torch.view_as_real(x).permute(0, 1, 4, 2, 3).flatten(1, 2)
            )  # [bs, inc*2, h, w] real

        y = x  # save for residual
        if self.aug_feature and not self.aug_feature_enhance:
            x = torch.cat([x, aug_feature], dim=1)
        if self.aug_input:
            x = torch.cat([x, input], dim=1)

        if self.linear_probing_mode:
            with torch.no_grad():
                x = self.stem(x)
                if self.aug_feature and self.aug_feature_enhance:
                    attn = self.aug_feature_enhancer(aug_feature)
                    x = x * attn
                x = self.features(x)
        else:
            x = self.stem(x)
            if self.aug_feature and self.aug_feature_enhance:
                attn = self.aug_feature_enhancer(aug_feature)
                x = x * attn
            x = self.features(x)

        x = self.head(x)

        if self.learn_residual:
            x += y

        x = torch.view_as_complex(
            x.view(x.size(0), -1, 2, x.size(-2), x.size(-1))
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # [bs, outc/2, h, w] complex

        return x
