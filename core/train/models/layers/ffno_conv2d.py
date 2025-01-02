"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-02 21:12:39
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-12-24 03:32:13
"""

from typing import Tuple

import torch
from mmengine.registry import MODELS
from torch import nn
from torch.functional import Tensor
from torch.types import Device

__all__ = ["FFNOConv2d", "FFNOConv3d"]

@MODELS.register_module()
class FFNOConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        https://arxiv.org/pdf/2010.08895.pdf
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        (
            self.n_mode_1,
            self.n_mode_2,
        ) = n_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.device = device

        self.scale = 1 / (in_channels * out_channels)
        self.build_parameters()
        self.reset_parameters()

    def build_parameters(self) -> None:
        self.weight_1 = nn.Parameter(
            self.scale
            * torch.zeros(
                [self.in_channels // 2, self.out_channels // 2, self.n_modes[0]],
                dtype=torch.cfloat,
            )
        )
        self.weight_2 = nn.Parameter(
            self.scale
            * torch.zeros(
                [
                    self.in_channels - self.in_channels // 2,
                    self.out_channels - self.out_channels // 2,
                    self.n_modes[1],
                ],
                dtype=torch.cfloat,
            )
        )

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_1.real)
        nn.init.kaiming_normal_(self.weight_2.real)

    def get_zero_padding(self, size, device):
        return torch.zeros(*size, dtype=torch.cfloat, device=device)

    def _ffno_forward(self, x, dim=-2):
        if dim == -2:
            with torch.amp.autocast("cuda", enabled=False):
                x_ft = torch.fft.rfft(x.float(), norm="ortho", dim=-2)
            n_mode = self.n_mode_1
            if n_mode == x_ft.size(-2):  # full mode
                out_ft = torch.einsum("bixy,iox->boxy", x_ft, self.weight_1)
            else:
                out_ft = self.get_zero_padding(
                    [x.size(0), self.weight_1.size(1), x_ft.size(-2), x_ft.size(-1)],
                    x.device,
                )
                # print(x_ft.shape, self.weight_1.shape)
                out_ft[..., :n_mode, :] = torch.einsum(
                    "bixy,iox->boxy", x_ft[..., :n_mode, :], self.weight_1
                )
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.fft.irfft(
                    out_ft.to(torch.cfloat), n=x.size(-2), dim=-2, norm="ortho"
                )
        elif dim == -1:
            with torch.amp.autocast("cuda", enabled=False):
                x_ft = torch.fft.rfft(x.float(), norm="ortho", dim=-1)
            n_mode = self.n_mode_2
            if n_mode == x_ft.size(-1):
                out_ft = torch.einsum("bixy,ioy->boxy", x_ft, self.weight_2)
            else:
                out_ft = self.get_zero_padding(
                    [x.size(0), self.weight_2.size(1), x_ft.size(-2), x_ft.size(-1)],
                    x.device,
                )
                # print(x_ft.shape, self.weight_2.shape)
                out_ft[..., :n_mode] = torch.einsum(
                    "bixy,ioy->boxy", x_ft[..., :n_mode], self.weight_2
                )
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.fft.irfft(
                    out_ft.to(torch.cfloat), n=x.size(-1), dim=-1, norm="ortho"
                )
        return x

    def _ffno2_forward(self, x, dim=-2):
        if dim == -2:
            with torch.amp.autocast("cuda", enabled=False):
                x_ft = torch.fft.rfft2(x.transpose(-2, -1).float(), norm="ortho")
            out_ft = self.get_zero_padding(
                [x.size(0), self.weight_1.size(1), x_ft.size(-2), x_ft.size(-1)],
                x.device,
            )
            n_mode = self.n_mode_1
            # print(x_ft.shape, self.weight_1.shape)
            out_ft[..., :n_mode] = torch.einsum(
                "biyx,iox->boyx", x_ft[..., :n_mode], self.weight_1
            )
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.fft.irfft2(
                    out_ft.to(torch.cfloat), s=(x.size(-1), x.size(-2)), norm="ortho"
                ).transpose(-2, -1)
        elif dim == -1:
            with torch.amp.autocast("cuda", enabled=False):
                x_ft = torch.fft.rfft2(x.float(), norm="ortho")
            out_ft = self.get_zero_padding(
                [x.size(0), self.weight_2.size(1), x_ft.size(-2), x_ft.size(-1)],
                x.device,
            )
            n_mode = self.n_mode_2
            # print(x_ft.shape, self.weight_2.shape)
            out_ft[..., :n_mode] = torch.einsum(
                "bixy,ioy->boxy", x_ft[..., :n_mode], self.weight_2
            )
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.fft.irfft2(
                    out_ft.to(torch.cfloat), s=(x.size(-2), x.size(-1)), norm="ortho"
                )
        return x

    def forward(self, x: Tensor) -> Tensor:
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        xx, xy = x.chunk(2, dim=1)
        xx = self._ffno_forward(xx, dim=-1)
        xy = self._ffno_forward(xy, dim=-2)
        x = torch.cat([xx, xy], dim=1)
        return x

    # def forward(self, x: Tensor) -> Tensor:
    #     # Compute Fourier coeffcients up to factor of e^(- something constant)
    #     # print("1", x.shape)
    #     xx = self._ffno_forward(x, dim=-1)
    #     xy = self._ffno_forward(x, dim=-2)
    #     x = xx + xy
    #     # print("2", x.shape)
    #     return x


@MODELS.register_module()
class FFNOConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        https://arxiv.org/pdf/2010.08895.pdf
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        (
            self.n_mode_1,
            self.n_mode_2,
            self.n_mode_3,
        ) = n_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.device = device

        self.scale = 1 / (in_channels * out_channels)
        self.build_parameters()
        self.reset_parameters()

    def build_parameters(self) -> None:
        self.weight_1 = nn.Parameter(
            self.scale
            * torch.zeros(
                [self.in_channels // 3, self.out_channels // 3, self.n_modes[0]],
                dtype=torch.cfloat,
            )
        )
        self.weight_2 = nn.Parameter(
            self.scale
            * torch.zeros(
                [self.in_channels // 3, self.out_channels // 3, self.n_modes[1]],
                dtype=torch.cfloat,
            )
        )
        self.weight_3 = nn.Parameter(
            self.scale
            * torch.zeros(
                [
                    self.in_channels - 2 * (self.in_channels // 3),
                    self.out_channels - 2 * (self.out_channels // 3),
                    self.n_modes[2],
                ],
                dtype=torch.cfloat,
            )
        )

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_1.real)
        nn.init.kaiming_normal_(self.weight_2.real)
        nn.init.kaiming_normal_(self.weight_3.real)

    def get_zero_padding(self, size, device):
        return torch.zeros(*size, dtype=torch.cfloat, device=device)

    def _ffno_forward(self, x, dim=-2):
        if dim == -3:
            with torch.amp.autocast("cuda", enabled=False):
                x_ft = torch.fft.rfft(x.float(), norm="ortho", dim=-3)
            n_mode = self.n_mode_1
            if n_mode == x_ft.size(-3):  # full mode
                out_ft = torch.einsum("bitxy,iot->botxy", x_ft, self.weight_1)
            else:
                out_ft = self.get_zero_padding(
                    [
                        x.size(0),
                        self.weight_1.size(1),
                        x_ft.size(-3),
                        x_ft.size(-2),
                        x_ft.size(-1),
                    ],
                    x.device,
                )
                # print(x_ft.shape, self.weight_1.shape)
                out_ft[..., :n_mode, :, :] = torch.einsum(
                    "bitxy,iot->botxy", x_ft[..., :n_mode, :, :], self.weight_1
                )
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.fft.irfft(
                    out_ft.to(torch.cfloat), n=x.size(-3), dim=-3, norm="ortho"
                )
        if dim == -2:
            with torch.amp.autocast("cuda", enabled=False):
                x_ft = torch.fft.rfft(x.float(), norm="ortho", dim=-2)
            n_mode = self.n_mode_2
            if n_mode == x_ft.size(-2):  # full mode
                out_ft = torch.einsum("bitxy,iox->botxy", x_ft, self.weight_2)
            else:
                out_ft = self.get_zero_padding(
                    [
                        x.size(0),
                        self.weight_2.size(1),
                        x_ft.size(-3),
                        x_ft.size(-2),
                        x_ft.size(-1),
                    ],
                    x.device,
                )
                # print(x_ft.shape, self.weight_1.shape)
                out_ft[..., :, :n_mode, :] = torch.einsum(
                    "bitxy,iox->botxy", x_ft[..., :, :n_mode, :], self.weight_2
                )
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.fft.irfft(
                    out_ft.to(torch.cfloat), n=x.size(-2), dim=-2, norm="ortho"
                )
        elif dim == -1:
            with torch.amp.autocast("cuda", enabled=False):
                x_ft = torch.fft.rfft(x.float(), norm="ortho", dim=-1)
            n_mode = self.n_mode_3
            if n_mode == x_ft.size(-1):
                out_ft = torch.einsum("bitxy,ioy->botxy", x_ft, self.weight_3)
            else:
                out_ft = self.get_zero_padding(
                    [
                        x.size(0),
                        self.weight_3.size(1),
                        x_ft.size(-3),
                        x_ft.size(-2),
                        x_ft.size(-1),
                    ],
                    x.device,
                )
                # print(x_ft.shape, self.weight_2.shape)
                out_ft[..., :n_mode] = torch.einsum(
                    "bitxy,ioy->botxy", x_ft[..., :n_mode], self.weight_3
                )
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.fft.irfft(
                    out_ft.to(torch.cfloat), n=x.size(-1), dim=-1, norm="ortho"
                )
        return x

    def forward(self, x: Tensor) -> Tensor:
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        xt, xx, xy = x.chunk(3, dim=1)
        xt = self._ffno_forward(xx, dim=-3)
        xx = self._ffno_forward(xx, dim=-2)
        xy = self._ffno_forward(xy, dim=-1)
        x = torch.cat([xt, xx, xy], dim=1)
        return x
