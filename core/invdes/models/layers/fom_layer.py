from typing import Callable, List

import torch
from torch import Tensor, nn

from .utils import *

__all__ = [
    "SimulatedFoM",
    "BinaryProjectionLayer",
    "HeavisideProjection",
    "ClipLayer",
    "heightProjection",
    "InsensitivePeriod",
]


class SimulatedFoM(nn.Module):
    def __init__(
        self, cal_obj_and_grad_fn: Callable, adjoint_mode: str = "fdtd"
    ) -> None:
        super().__init__()
        self.cal_obj_and_grad_fn = cal_obj_and_grad_fn
        self.adjoint_mode = adjoint_mode

    def forward(
        self, permittivity_list: List[Tensor], resolution: int | None = None, custom_source: Tensor | None = None
    ) -> Tensor:
        if self.adjoint_mode == "ceviche":
            ## we need to use autograd, jacobian to calculate gradients
            raise NotImplementedError("ceviche adjoint mode is not supported")
            fom = AdjointGradient.apply(
                self.cal_obj_and_grad_fn,
                self.adjoint_mode,
                resolution,
                *permittivity_list,
            )
        elif self.adjoint_mode == "ceviche_torch":
            ## this function is completely differentiable, so we only need forward for torch
            fom = self.cal_obj_and_grad_fn(
                adjoint_mode=self.adjoint_mode,
                need_item="need_value",
                resolution=resolution,
                permittivity_list=permittivity_list,
                custom_source=custom_source,
            )
        return fom

    def extra_repr(self) -> str:
        return f"adjoint_mode={self.adjoint_mode}"


class ClipLayer(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: Tensor, upper_limit=None, lower_limit=None) -> Tensor:
        if self.mode == "lower_limit":
            assert upper_limit is None and lower_limit is not None
            x = ApplyLowerLimit.apply(x, lower_limit)
        elif self.mode == "upper_limit":
            assert upper_limit is not None and lower_limit is None
            x = ApplyUpperLimit.apply(x, upper_limit)
        elif self.mode == "both":
            assert upper_limit is not None and lower_limit is not None
            x = ApplyBothLimit.apply(x, upper_limit, lower_limit)
        else:
            raise ValueError("Invalid mode")
        return x

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


class BinaryProjectionLayer(nn.Module):
    def __init__(self, threshold: float = 0.05) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: Tensor, t_bny: float) -> Tensor:
        permittivity = BinaryProjection.apply(x, t_bny, self.threshold)
        return permittivity

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


class HeavisideProjection(nn.Module):
    def __init__(self, threshold: float = 10) -> None:
        super().__init__()
        self.threshold = threshold  # leave threshold here for future use STE

    def forward(self, x: Tensor, beta, eta) -> Tensor:
        permittivity = HeavisideProjectionLayer.apply(
            x, beta, eta, self.threshold
        )  # STE
        return permittivity


class InsensitivePeriod(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, i: int):
        x = InsensitivePeriodLayer.apply(x, i)  # STE
        return x


class heightProjection(nn.Module):
    def __init__(self, threshold: float = 10, height_max: float = 1.0) -> None:
        super().__init__()
        self.threshold = threshold  # leave threshold here for future use STE
        self.height_max = height_max

    def forward(self, ridge_height, gratings: Tensor, sharpness, resolution) -> Tensor:
        height_mask = torch.linspace(
            0, self.height_max, self.height_max * resolution + 1
        ).to(gratings.device)
        height_mask = heightProjectionLayer.apply(
            ridge_height, height_mask, sharpness, self.threshold
        )  # STE
        gratings = gratings * height_mask.unsqueeze(1)
        return gratings
