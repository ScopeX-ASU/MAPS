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
    """Bridge a differentiable simulation/objective callback into the PyTorch graph.

    Args:
        cal_obj_and_grad_fn (Callable): Objective callback invoked with solver mode,
            requested output type, resolution, permittivity tensors, and optional source.
        adjoint_mode (str): Backend selector. Supported runtime paths are ``"ceviche_torch"``
            and ``"fdtdx"``; ``"ceviche"`` raises ``NotImplementedError``. Defaults to ``"fdtd"``.
    """

    def __init__(
        self, cal_obj_and_grad_fn: Callable, adjoint_mode: str = "fdtd"
    ) -> None:
        """Initialize the differentiable FoM wrapper.

        Args:
            cal_obj_and_grad_fn (Callable): Objective/simulation callback.
            adjoint_mode (str): Backend/adjoint selector. Defaults to ``"fdtd"``.
        """
        super().__init__()
        self.cal_obj_and_grad_fn = cal_obj_and_grad_fn
        self.adjoint_mode = adjoint_mode

    def forward(
        self,
        permittivity_list: List[Tensor],
        resolution: int | None = None,
        custom_source: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the figure of merit from one or more permittivity tensors.

        Args:
            permittivity_list (list[Tensor]): Ordered simulation permittivity tensors.
                Spatial shape and units must match the selected solver/backend contract.
            resolution (int | None): Optional simulation resolution forwarded to the
                objective callback. Defaults to None.
            custom_source (Tensor | None): Optional source tensor forwarded unchanged to
                the objective callback. Defaults to None.

        Returns:
            Tensor: Scalar differentiable figure-of-merit value.

        Raises:
            NotImplementedError: If ``adjoint_mode`` is ``"ceviche"``.
        """
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
        elif self.adjoint_mode == "fdtdx":
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
    """Apply differentiable lower, upper, or two-sided clipping.

    Args:
        mode (str): Limit mode: ``"lower_limit"``, ``"upper_limit"``, or ``"both"``.
    """

    def __init__(self, mode: str) -> None:
        """Initialize the clipping mode.

        Args:
            mode (str): Limit mode. No default is defined.
        """
        super().__init__()
        self.mode = mode

    def forward(self, x: Tensor, upper_limit=None, lower_limit=None) -> Tensor:
        """Apply the configured clipping rule.

        Args:
            x (Tensor): Input tensor.
            upper_limit (Tensor | float | None): Upper bound required by ``"upper_limit"``
                and ``"both"`` modes. Defaults to None.
            lower_limit (Tensor | float | None): Lower bound required by ``"lower_limit"``
                and ``"both"`` modes. Defaults to None.

        Returns:
            Tensor: Clipped tensor with the same shape as ``x``.

        Raises:
            AssertionError: If bounds are inconsistent with the selected mode.
            ValueError: If ``mode`` is unsupported.
        """
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
    """Apply the custom differentiable binary projection.

    Args:
        threshold (float): Backward temperature threshold used by the underlying
            binary projection function. Defaults to 0.05.
    """

    def __init__(self, threshold: float = 0.05) -> None:
        """Initialize the binary projection layer.

        Args:
            threshold (float): Backward temperature threshold. Defaults to 0.05.
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, x: Tensor, t_bny: float) -> Tensor:
        """Project a continuous tensor toward binary values.

        Args:
            x (Tensor): Continuous normalized density/permittivity tensor.
            t_bny (float): Binary projection temperature/broadening parameter.

        Returns:
            Tensor: Projected tensor with the same shape as ``x``.
        """
        permittivity = BinaryProjection.apply(x, t_bny, self.threshold)
        return permittivity

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


class HeavisideProjection(nn.Module):
    """Apply a differentiable straight-through Heaviside projection.

    Args:
        threshold (float): Forward threshold stored by the custom layer. Defaults to 10.
    """

    def __init__(self, threshold: float = 10) -> None:
        """Initialize the Heaviside projection layer.

        Args:
            threshold (float): Forward threshold used by the custom autograd operator.
                Defaults to 10.
        """
        super().__init__()
        self.threshold = threshold  # leave threshold here for future use STE

    def forward(self, x: Tensor, beta, eta) -> Tensor:
        """Apply the custom straight-through Heaviside projection.

        Args:
            x (Tensor): Continuous normalized density/permittivity tensor.
            beta (float | Tensor): Projection sharpness parameter.
            eta (float | Tensor): Projection threshold/midpoint.

        Returns:
            Tensor: Projected tensor with the same shape as ``x``.
        """
        permittivity = HeavisideProjectionLayer.apply(
            x, beta, eta, self.threshold
        )  # STE
        return permittivity


class InsensitivePeriod(nn.Module):
    """Apply the custom insensitive-period straight-through transformation."""

    def __init__(self) -> None:
        """Initialize the insensitive-period layer."""
        super().__init__()

    def forward(self, x: Tensor, i: int):
        """Apply the insensitive-period transformation.

        Args:
            x (Tensor): Input design tensor.
            i (int): Period/index parameter controlling the insensitive interval.

        Returns:
            Tensor: Transformed tensor with the same shape as ``x``.
        """
        x = InsensitivePeriodLayer.apply(x, i)  # STE
        return x


class heightProjection(nn.Module):
    """Apply a differentiable ridge-height mask to a grating tensor.

    Args:
        threshold (float): Forward threshold passed to ``heightProjectionLayer``.
            Defaults to 10.
        height_max (float): Maximum sampled ridge height. Defaults to 1.0.
    """

    def __init__(self, threshold: float = 10, height_max: float = 1.0) -> None:
        """Initialize the ridge-height projection module.

        Args:
            threshold (float): Forward threshold. Defaults to 10.
            height_max (float): Maximum height sampled by ``forward``. Defaults to 1.0.
        """
        super().__init__()
        self.threshold = threshold  # leave threshold here for future use STE
        self.height_max = height_max

    def forward(self, ridge_height, gratings: Tensor, sharpness, resolution) -> Tensor:
        """Build a height-dependent mask and multiply it into the grating tensor.

        Args:
            ridge_height (float | Tensor): Ridge-height parameter consumed by the
                custom height projection operator.
            gratings (Tensor): Grating tensor whose second dimension aligns with the
                sampled height dimension.
            sharpness (float | Tensor): Projection sharpness passed to the custom operator.
            resolution (int | float): Samples per unit height used to construct the mask.

        Returns:
            Tensor: ``gratings`` multiplied by the projected height mask.
        """
        height_mask = torch.linspace(
            0, self.height_max, self.height_max * resolution + 1
        ).to(gratings.device)
        height_mask = heightProjectionLayer.apply(
            ridge_height, height_mask, sharpness, self.threshold
        )  # STE
        gratings = gratings * height_mask.unsqueeze(1)
        return gratings
