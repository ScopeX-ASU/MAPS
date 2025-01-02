from typing import Tuple

import torch
from torch import nn
from torch.functional import Tensor
from torch.types import Device

__all__ = ["PACEConv2d_Split2x", "PACEConv2d_Split4x"]

class PACEConv2d_Split2x(nn.Module):
    """A Optimzied global conv module that split the input into four parts and apply the global conv separately, then concat them"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        conv_mutation: bool= True,
        bias: bool = False,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()

        """
        Global conv based on FFT
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_mode_1, self.n_mode_2 = n_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.device = device

        self.scale = 1 / (in_channels * out_channels)
        
        if bias:
            self.bias = nn.Parameter(
                torch.zeros([self.out_channels], dtype=torch.float32)
            )
        else:
            self.register_parameter("bias", None)
        
        self.build_parameters()
        self.reset_parameters()
    
    def build_parameters(self) -> None:
        # split into 4 pieces
        self.weight_1 = nn.Parameter(
            self.scale * torch.zeros([2, self.in_channels // 2, self.out_channels // 2, self.n_modes[0]], dtype=torch.cfloat)
        )
        self.weight_2 = nn.Parameter(
            self.scale * torch.zeros([2, self.in_channels // 2, self.out_channels // 2, self.n_modes[1]], dtype=torch.cfloat)
        )

    def reset_parameters(self) -> None:
        for i in range(2):
            nn.init.kaiming_normal_(self.weight_1[i])
            nn.init.kaiming_normal_(self.weight_2[i])
        
        if self.bias is not None:
            nn.init.uniform_(self.bias, 0, 0)

    def get_zero_padding(self, size, device):
        return torch.zeros(*size, dtype=torch.cfloat, device=device)

    def ffno_forward(self, x, dim = -2):
        return self._ffno_forward(x, dim)
    
    def _ffno_forward(self, x, dim = -2):
        bs, c, h, w = x.size()
        x = x.reshape(bs, 2, c // 2, h, w)
        if dim == -2:
            with torch.cuda.amp.autocast(enabled=False):
                x_ft = torch.fft.rfft(x, norm="ortho", dim=-2)
            n_mode = self.n_mode_1
            if n_mode == x_ft.size(-2): # full mode
                out_ft = torch.einsum("bgixy,giox->bgoxy", x_ft, self.weight_1)
            else:
                out_ft = self.get_zero_padding([bs, 2, c // 2, x_ft.size(-2), x_ft.size(-1)], x.device)
                out_ft[..., : n_mode, :] = torch.einsum(
                    "bgixy,giox->bgoxy", x_ft[..., : n_mode, :], self.weight_1
                )
            with torch.cuda.amp.autocast(enabled=False):
                x = torch.fft.irfft(out_ft, n=x.size(-2), dim=-2, norm="ortho")
        elif dim == -1:
            with torch.cuda.amp.autocast(enabled=False):
                x_ft = torch.fft.rfft(x, norm="ortho", dim=-1)
            n_mode = self.n_mode_2
            if n_mode == x_ft.size(-1):
                out_ft = torch.einsum("bgixy,gioy->bgoxy", x_ft, self.weight_2)
            else:
                out_ft = self.get_zero_padding([bs, 2, c //2, x_ft.size(-2), x_ft.size(-1)], x.device)
                out_ft[..., :n_mode] = torch.einsum(
                    "bgixy,gioy->bgoxy", x_ft[..., : n_mode], self.weight_2
                )
            with torch.cuda.amp.autocast(enabled=False):
                x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1, norm="ortho")
                
        x = x.reshape(bs, c, h, w)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._ffno_forward(x, dim=-1) # do 1d fft and then do group-wise multiplication
        x = self._ffno_forward(x, dim=-2) # do 1d fft and then do group-wise multiplication

        if self.bias is not None:
            x = x + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        return x



class PACEConv2d_Split4x(nn.Module):
    """A Optimzied global conv module that split the input into four parts and apply the global conv separately, then concat them"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        conv_mutation: bool= True,
        bias: bool = False,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()

        """
        Global conv based on FFT
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_mode_1, self.n_mode_2 = n_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.device = device

        self.scale = 1 / (in_channels * out_channels)
        
        if bias:
            self.bias = nn.Parameter(
                torch.zeros([self.out_channels], dtype=torch.float32)
            )
        else:
            self.register_parameter("bias", None)
        
        self.build_parameters()
        self.reset_parameters()
    
    def build_parameters(self) -> None:
        # split into 4 pieces
        self.weight_1 = nn.Parameter(
            self.scale * torch.zeros([4, self.in_channels // 4, self.out_channels // 4, self.n_modes[0]], dtype=torch.cfloat)
        )
        self.weight_2 = nn.Parameter(
            self.scale * torch.zeros([4, self.in_channels // 4, self.out_channels // 4, self.n_modes[1]], dtype=torch.cfloat)
        )

    def reset_parameters(self) -> None:
        for i in range(4):
            nn.init.kaiming_normal_(self.weight_1[i])
            nn.init.kaiming_normal_(self.weight_2[i])
        
        if self.bias is not None:
            nn.init.uniform_(self.bias, 0, 0)

    def get_zero_padding(self, size, device):
        return torch.zeros(*size, dtype=torch.cfloat, device=device)

    def ffno_forward(self, x, dim = -2):
        return self._ffno_forward(x, dim)
    
    def _ffno_forward(self, x, dim = -2):
        bs, c, h, w = x.size()
        x = x.reshape(bs, 4, c // 4, h, w)
        if dim == -2:
            with torch.cuda.amp.autocast(enabled=False):
                x_ft = torch.fft.rfft(x, norm="ortho", dim=-2)
            n_mode = self.n_mode_1
            if n_mode == x_ft.size(-2): # full mode
                out_ft = torch.einsum("bgixy,giox->bgoxy", x_ft, self.weight_1)
            else:
                out_ft = self.get_zero_padding([bs, 4, c // 4, x_ft.size(-2), x_ft.size(-1)], x.device)
                out_ft[..., : n_mode, :] = torch.einsum(
                    "bgixy,giox->bgoxy", x_ft[..., : n_mode, :], self.weight_1
                )
            with torch.cuda.amp.autocast(enabled=False):
                x = torch.fft.irfft(out_ft, n=x.size(-2), dim=-2, norm="ortho")
        elif dim == -1:
            with torch.cuda.amp.autocast(enabled=False):
                x_ft = torch.fft.rfft(x, norm="ortho", dim=-1)
            n_mode = self.n_mode_2
            if n_mode == x_ft.size(-1):
                out_ft = torch.einsum("bgixy,gioy->bgoxy", x_ft, self.weight_2)
            else:
                out_ft = self.get_zero_padding([bs, 4, c //4, x_ft.size(-2), x_ft.size(-1)], x.device)
                out_ft[..., :n_mode] = torch.einsum(
                    "bgixy,gioy->bgoxy", x_ft[..., : n_mode], self.weight_2
                )
            with torch.cuda.amp.autocast(enabled=False):
                x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1, norm="ortho")
                
        x = x.reshape(bs, c, h, w)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._ffno_forward(x, dim=-1) # do 1d fft and then do group-wise multiplication
        x = self._ffno_forward(x, dim=-2) # do 1d fft and then do group-wise multiplication

        if self.bias is not None:
            x = x + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        return x
