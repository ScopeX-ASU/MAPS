"""
Date: 2024-04-12 23:48:37
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-12 23:48:37
FilePath: /NeurOLight_Local/core/models/layers/fourier_conv2d.py
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import einsum
from mmengine.registry import MODELS
from torch.functional import Tensor
from torch.nn.modules.utils import _ntuple
from torch.types import Device

__all__ = ["FourierConv2d"]


class _FourierConv(nn.Module):
    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    padding: int
    weight: Tensor
    miniblock: int
    mode: str
    enable_padding: bool
    _mask_options: str = ["cone", "cylinder", "block"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,  # unused
        padding: int = 0,  # unused, always assume "same" padding
        groups: int = 1,  # unused
        dilation: int = 1,  # unused
        enable_padding: bool = True,
        is_causal: bool = True,
        mask_shape: str = "cone",
        bias: bool = False,
        ndim: int = 2,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _ntuple(ndim)(kernel_size)
        self.groups = groups
        assert (
            self.kernel_size[-1] == self.kernel_size[-2]
        ), f"Only support square kernel in spatial dimension, but got {self.kernel_size}"
        self.enable_padding = enable_padding
        self.is_causal = is_causal
        self.mask_shape = mask_shape
        assert (
            mask_shape in self._mask_options
        ), f"Only support {self._mask_options}, but got {mask_shape}"

        ## after padding, input shape changes from [t, h, w] to [t+k1-1, h+k2-1, w+k3-1] assume t, h, w are even number, k1/k2/k3 are odd number
        self.input_padding = [
            (k // 2, k // 2) for k in self.kernel_size
        ]  # left right, up down padding

        ## kernel need right/bottom padding. kernel shape changes from [k1, k2, k3] to
        self.ndim = ndim
        self.device = device

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.build_parameters()

    def build_parameters(self):
        self.weight = nn.Parameter(
            [nn.Conv1d, nn.Conv2d, nn.Conv3d][self.ndim - 1](
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                groups=self.groups,
                padding=0,
                bias=False,
            ).weight.data.to(self.device)
        )
        self.mask = torch.zeros(*self.kernel_size, device=self.device)

    def reset_parameters(self):
        time_center = self.kernel_size[0] // 2
        space_center = self.kernel_size[1] // 2
        X, Y = torch.meshgrid(
            torch.arange(self.kernel_size[1], device=self.device),
            torch.arange(self.kernel_size[2], device=self.device),
        )
        dist_to_center = (
            X.sub(space_center).square_().add(Y.sub(space_center).square_())[None, ...]
        )  # [1, k, k]

        if self.mask_shape == "cone":
            radius = (
                torch.arange(self.kernel_size[0], device=self.device)
                .sub_(time_center)
                .square_()[..., None, None]
            )  # [t, 1, 1]
            self.mask = dist_to_center <= radius
        elif self.mask_shape == "cylinder":
            radius = space_center
            self.mask = dist_to_center <= radius
        elif self.mask_shape == "block":
            self.mask.fill_(1)
        else:
            raise NotImplementedError

        if self.is_causal:
            self.mask[time_center + 1 :] = 0

        if self.bias is not None:
            init.zeros_(self.bias)

    def pad_input_weight(self, x, weight):
        if self.enable_padding:
            ## two-sided padding for input
            x = F.pad(
                x,
                tuple(p for pad in self.input_padding[::-1] for p in pad),
            )

            ## first do one-sided padding for weight
            weight = F.pad(
                weight,  # [outc, inc, k1, k2, k3]
                tuple(
                    pad
                    for in_size, w_size in zip(
                        x.shape[-self.ndim :][::-1], weight.shape[-self.ndim :][::-1]
                    )
                    for pad in [0, in_size - w_size]
                ),
            )  # [outc, t1, h1, w1]
            ## then do roll
            dims = tuple(range(-self.ndim, 0))
            weight = torch.roll(
                weight, dims=dims, shifts=[-self.kernel_size[d] // 2 for d in dims]
            )

        return x, weight

    def unpad_input(self, x):
        ## two-sided unpad
        if self.enable_padding:
            crop_slices = [slice(None), slice(None)] + [
                slice(
                    self.weight.shape[i] // 2,
                    self.weight.shape[i] // 2 + (x.shape[i] - self.weight.shape[i] + 1),
                    1,
                )
                for i in range(2, x.ndim)
            ]
            x = x[crop_slices]

        return x

    def forward(self, x: Tensor) -> Tensor:  # bs f, 256, 256
        ## padding the input tensor
        # inplace zero out, because we do not need to apply mask to grad. this is faster than self.weight * self.mask
        if self.mask_shape == "block" and not self.is_causal:
            pass
        else:
            self.weight.data.mul_(self.mask)

        x, kernel = self.pad_input_weight(x, self.weight)
        x = torch.fft.rfftn(x, dim=tuple(range(-self.ndim, 0)), norm="ortho")

        kernel = torch.fft.rfftn(kernel, dim=tuple(range(-self.ndim, 0))).conj()

        ## convolution in the frequency domain
        ## equals to the element-wise multiplication in the frequency domain
        if self.in_channels == 1:
            x = x * kernel
        elif self.groups == 1:
            x = einsum(x, kernel, "b i ..., o i ... -> b o ...")
        else:
            x = x.reshape(x.shape[0], self.groups, -1, *x.shape[2:])
            kernel = kernel.reshape(self.groups, -1, *kernel.shape[1:])
            x = einsum(x, kernel, "b g i ..., g o i ... -> b g o ...", g=self.groups)
            x = x.flatten(1, 2)

        x = torch.fft.irfftn(x, dim=tuple(range(-self.ndim, 0)), norm="ortho")
        x = self.unpad_input(x)

        return x

    def extra_repr(self) -> str:
        s = ""
        s += f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
        s += f"kernel_size={self.kernel_size}, r={self.r}, "
        s += f"is_causal={self.is_causal}, mask_shape={self.mask_shape}, "
        s += f"enable_padding={self.enable_padding}"
        return s


@MODELS.register_module()
class FourierConv2d(_FourierConv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs,
            ndim=2,
        )

    def reset_parameters(self):
        space_center = self.kernel_size[1] // 2
        X, Y = torch.meshgrid(
            torch.arange(self.kernel_size[-2], device=self.device),
            torch.arange(self.kernel_size[-1], device=self.device),
        )
        dist_to_center = (
            X.sub(space_center).square_().add(Y.sub(space_center).square_())
        )  # [k, k]

        if self.mask_shape == "cone":
            raise ValueError("Not support cone mask in 2d")
        elif self.mask_shape == "cylinder":
            radius = space_center
            self.mask = dist_to_center <= radius
        elif self.mask_shape == "block":
            self.mask.fill_(1)
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.zeros_(self.bias)
