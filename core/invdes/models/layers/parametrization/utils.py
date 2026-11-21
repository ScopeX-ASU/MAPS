"""
Date: 2024-10-05 02:05:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-05 02:05:18
FilePath: /Metasurface-Opt/core/models/parametrization/utils.py
"""

import torch
from torch import Tensor, nn


class ClipLayer(nn.Module):
    """Apply a configurable lower, upper, or two-sided limit while preserving gradients.

    Args:
        mode (str): Limit mode. Supported values are ``"lower_limit"``, ``"upper_limit"``,
            and ``"both"``.
    """

    def __init__(self, mode: str) -> None:
        """Initialize the clipping mode.

        Args:
            mode (str): Limit mode. Supported values are ``"lower_limit"``,
                ``"upper_limit"``, and ``"both"``.
        """
        super().__init__()
        self.mode = mode

    def forward(self, x: Tensor, upper_limit=None, lower_limit=None) -> Tensor:
        """Clip ``x`` using the mode-selected limit(s).

        Args:
            x (Tensor): Input tensor to limit.
            upper_limit (Tensor | float | None): Upper bound required for ``"upper_limit"``
                and ``"both"`` modes. Defaults to None.
            lower_limit (Tensor | float | None): Lower bound required for ``"lower_limit"``
                and ``"both"`` modes. Defaults to None.

        Returns:
            Tensor: Clipped tensor with the same shape as ``x``.

        Raises:
            AssertionError: If the selected mode is given incompatible bound arguments.
            ValueError: If ``self.mode`` is unsupported.
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


class BinaryProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permittivity: Tensor, T_bny: float, T_threshold: float):
        ctx.T_bny = T_bny
        ctx.T_threshold = T_threshold
        ctx.save_for_backward(permittivity)
        result = (torch.tanh((0.5 - permittivity) / T_bny) + 1) / 2
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # if T_bny is larger than T_threshold, then use the automatic differentiation of the tanh function
        # if the T_bny is smaller than T_threshold, then use the gradient as if T_bny is T_threshold
        T_bny = ctx.T_bny
        T_threshold = ctx.T_threshold
        (permittivity,) = ctx.saved_tensors

        if T_bny > T_threshold:
            grad = (
                -grad_output
                * (1 - torch.tanh((0.5 - permittivity) / T_bny) ** 2)
                / T_bny
            )
        else:
            grad = (
                -grad_output
                * (1 - torch.tanh((0.5 - permittivity) / T_threshold) ** 2)
                / T_threshold
            )

        return grad, None, None


class BinaryProjectionLayer(nn.Module):
    """Module wrapper around the custom binary projection autograd function.

    Args:
        threshold (float): Backward-gradient temperature threshold used by
            ``BinaryProjection``. Defaults to 0.05.
    """

    def __init__(self, threshold: float = 0.05) -> None:
        """Initialize the binary projection layer.

        Args:
            threshold (float): Backward-gradient temperature threshold. Defaults to 0.05.
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, x: Tensor, t_bny: float) -> Tensor:
        """Project a normalized permittivity/density tensor toward binary values.

        Args:
            x (Tensor): Input normalized permittivity or density tensor.
            t_bny (float): Forward projection temperature/broadening parameter.

        Returns:
            Tensor: Projected tensor with the same shape as ``x``.
        """
        permittivity = BinaryProjection.apply(x, t_bny, self.threshold)
        return permittivity

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


class ApplyLowerLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lower_limit):
        ctx.save_for_backward(x)
        ctx.lower_limit = lower_limit
        return torch.maximum(x, lower_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors
        lower_limit = ctx.lower_limit

        # Compute gradient
        # If x > lower_limit, propagate grad_output normally
        # If x <= lower_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
        )  # None for lower_limit since it does not require gradients


class ApplyUpperLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, upper_limit):
        ctx.save_for_backward(x)
        ctx.upper_limit = upper_limit
        return torch.minimum(x, upper_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors
        upper_limit = ctx.upper_limit

        # Compute gradient
        # If x > upper_limit, propagate grad_output normally
        # If x <= upper_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
        )  # None for upper_limit since it does not require gradients


class ApplyBothLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, upper_limit, lower_limit):
        upper_t = torch.as_tensor(upper_limit, dtype=x.dtype, device=x.device)
        lower_t = torch.as_tensor(lower_limit, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x)
        ctx.upper_limit = upper_t
        ctx.lower_limit = lower_t
        return torch.minimum(torch.maximum(x, lower_t), upper_t)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors
        upper_limit = ctx.upper_limit
        lower_limit = ctx.lower_limit

        # Compute gradient
        # If x > upper_limit, propagate grad_output normally
        # If x <= upper_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
            None,
        )  # None for upper_limit and lower_limit since they do not require gradients


class HeavisideProjectionLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, eta, fw_threshold, bw_threshold):
        ctx.save_for_backward(x, eta)
        ctx.bw_threshold = bw_threshold
        ctx.beta = beta
        if (
            beta < fw_threshold
        ):  # over a large number we will treat this as a pure binary projection
            return (torch.tanh(beta * eta) + torch.tanh(beta * (x - eta))) / (
                torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
            )
        else:
            return torch.where(
                x < eta,
                torch.tensor(0, dtype=x.dtype, device=x.device),
                torch.tensor(1, dtype=x.dtype, device=x.device),
            )

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, eta = ctx.saved_tensors
        bw_threshold = ctx.bw_threshold
        beta = ctx.beta
        if beta > bw_threshold:
            grad = (
                grad_output
                * (bw_threshold * (1 - (torch.tanh(bw_threshold * (x - eta))) ** 2))
                / (
                    torch.tanh(bw_threshold * eta)
                    + torch.tanh(bw_threshold * (1 - eta))
                )
            )
            denominator = torch.tanh(bw_threshold * eta) + torch.tanh(
                bw_threshold * (1 - eta)
            )
            denominator_grad_eta = bw_threshold * (
                1 - (torch.tanh(bw_threshold * eta)) ** 2
            ) - bw_threshold * (1 - (torch.tanh(bw_threshold * (1 - eta))) ** 2)
            nominator = torch.tanh(bw_threshold * eta) + torch.tanh(
                bw_threshold * (x - eta)
            )
            nominator_grad_eta = bw_threshold * (
                1 - (torch.tanh(bw_threshold * eta)) ** 2
            ) - bw_threshold * (1 - (torch.tanh(bw_threshold * (x - eta))) ** 2)
        else:
            grad = (
                grad_output
                * (beta * (1 - (torch.tanh(beta * (x - eta))) ** 2))
                / (torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta)))
            )
            denominator = torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
            denominator_grad_eta = beta * (1 - (torch.tanh(beta * eta)) ** 2) - beta * (
                1 - (torch.tanh(beta * (1 - eta))) ** 2
            )
            nominator = torch.tanh(beta * eta) + torch.tanh(beta * (x - eta))
            nominator_grad_eta = beta * (1 - (torch.tanh(beta * eta)) ** 2) - beta * (
                1 - (torch.tanh(beta * (x - eta))) ** 2
            )
        grad_eta = (
            grad_output
            * (denominator * nominator_grad_eta - nominator * denominator_grad_eta)
            / (denominator**2)
        )

        return grad, None, grad_eta, None, None


class HeavisideProjection(nn.Module):
    """Apply regular or straight-through Heaviside projection and optionally clip to ``[0, 1]``.

    Args:
        fw_threshold (float): Forward beta threshold above which the custom layer
            uses hard binary output. Defaults to 200.
        bw_threshold (float): Backward beta threshold used for straight-through
            gradient stabilization. Defaults to 80.
        mode (str): Projection mode, typically ``"regular"`` or ``"ste"``.
            Defaults to ``"regular"``.
        clip (bool): Whether to clamp the projected output to ``[0, 1]`` using
            straight-through ``ClipLayer``. Defaults to True.
    """

    def __init__(
        self,
        fw_threshold: float = 200,
        bw_threshold: float = 80,
        mode: str = "regular",
        clip: bool = True,
    ) -> None:
        """Initialize a Heaviside projection module.

        Args:
            fw_threshold (float): Forward beta threshold. Defaults to 200.
            bw_threshold (float): Backward beta threshold. Defaults to 80.
            mode (str): Projection mode. Defaults to ``"regular"``.
            clip (bool): Whether to clamp output to ``[0, 1]``. Defaults to True.
        """
        super().__init__()
        self.fw_threshold = fw_threshold  # leave threshold here for future use STE
        self.bw_threshold = bw_threshold
        self.mode = mode
        self.clip = bool(clip)
        self.clip_layer = ClipLayer(mode="both")
        self.upper_limit = torch.tensor(1.0)
        self.lower_limit = torch.tensor(0.0)

    def forward(
        self,
        x: Tensor,
        beta: float | Tensor,
        eta: float | Tensor = 0.5,
        clip: bool | None = None,
        **kwargs,
    ) -> Tensor:
        """Project ``x`` using the configured regular or straight-through mode.

        Args:
            x (Tensor): Input normalized density/permittivity tensor.
            beta (float | Tensor): Projection sharpness parameter.
            eta (float | Tensor): Projection midpoint. Defaults to 0.5.
            clip (bool | None): Optional per-call override for output clamping to ``[0, 1]``.
            **kwargs (Any): Additional keyword arguments accepted for interface compatibility.

        Returns:
            Tensor: Projected tensor, optionally clipped to ``[0, 1]``.
        """
        do_clip = self.clip if clip is None else bool(clip)
        if self.mode == "regular":
            return tanh_projection(x, beta=beta, eta=eta, clip=do_clip)
        elif self.mode.lower() == "ste":
            beta_t = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
            eta_t = torch.as_tensor(eta, dtype=x.dtype, device=x.device)
            x = HeavisideProjectionLayer.apply(
                x, beta_t, eta_t, self.fw_threshold, self.bw_threshold
            )  # STE
        if do_clip:
            if self.upper_limit.device != x.device or self.upper_limit.dtype != x.dtype:
                self.upper_limit = self.upper_limit.to(device=x.device, dtype=x.dtype)
                self.lower_limit = self.lower_limit.to(device=x.device, dtype=x.dtype)
            x = self.clip_layer(
                x, upper_limit=self.upper_limit, lower_limit=self.lower_limit
            )
        return x


def tanh_projection(
    x: Tensor,
    beta: float | Tensor,
    eta: float | Tensor = 0.5,
    clip: bool = False,
) -> Tensor:
    """Compute a numerically safe tanh projection with explicit zero/inf beta handling.

    Args:
        x (Tensor): Input normalized density/permittivity tensor.
        beta (float | Tensor): Projection sharpness. Non-positive values are handled as a
            clipped identity limit and infinity as a hard threshold at ``eta``.
        eta (float | Tensor): Projection midpoint. Defaults to 0.5.
        clip (bool): If True, clamp the output to ``[0, 1]`` via straight-through
            ``ApplyBothLimit`` autograd Function. Defaults to False.

    Returns:
        Tensor: Projected tensor with the same shape, dtype, and device as ``x``.
    """
    beta_t = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
    eta_t = torch.as_tensor(eta, dtype=x.dtype, device=x.device)
    is_inf = torch.isinf(beta_t)
    is_zero = beta_t <= 0

    safe_beta = torch.where(
        is_inf | is_zero,
        torch.ones((), dtype=x.dtype, device=x.device),
        beta_t,
    )
    dividend = torch.tanh(safe_beta * eta_t) + torch.tanh(safe_beta * (x - eta_t))
    divisor = torch.tanh(safe_beta * eta_t) + torch.tanh(safe_beta * (1 - eta_t))
    projected = dividend / divisor

    inf_projected = torch.where(x > eta_t, torch.ones_like(x), torch.zeros_like(x))
    zero_projected = ApplyBothLimit.apply(x, 1.0, 0.0) if clip else x.clamp(0, 1)
    out = torch.where(
        is_zero, zero_projected, torch.where(is_inf, inf_projected, projected)
    )
    if clip:
        out = ApplyBothLimit.apply(out, 1.0, 0.0)
    return out


def _gradient_2d_by_index(x: Tensor) -> tuple[Tensor, Tensor]:
    """Compute a 2-D unit-spacing gradient using forward/backward edge differences.

    Args:
        x (Tensor): Exactly 2-D tensor. Unit spacing is assumed on both axes.

    Returns:
        tuple[Tensor, Tensor]: Gradients along the first and second tensor axes,
        each with the same shape, dtype, and device as ``x``.

    Raises:
        ValueError: If ``x`` is not 2-D.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(x.shape)}")

    grad0 = torch.empty_like(x)
    grad1 = torch.empty_like(x)

    if x.shape[0] == 1:
        grad0.zero_()
    else:
        grad0[0] = x[1] - x[0]
        grad0[-1] = x[-1] - x[-2]
        if x.shape[0] > 2:
            grad0[1:-1] = (x[2:] - x[:-2]) * 0.5

    if x.shape[1] == 1:
        grad1.zero_()
    else:
        grad1[:, 0] = x[:, 1] - x[:, 0]
        grad1[:, -1] = x[:, -1] - x[:, -2]
        if x.shape[1] > 2:
            grad1[:, 1:-1] = (x[:, 2:] - x[:, :-2]) * 0.5

    return grad0, grad1


def subpixel_smoothed_projection(
    rho_filtered: Tensor,
    beta: float | Tensor,
    eta: float | Tensor = 0.5,
    resolution: float | Tensor = 1.0,
) -> Tensor:
    """Apply a 2-D subpixel-smoothed tanh projection near material interfaces.

    Args:
        rho_filtered (Tensor): Exactly 2-D spatially filtered density tensor.
        beta (float | Tensor): Projection sharpness parameter; zero and infinity are
            handled by ``tanh_projection``.
        eta (float | Tensor): Projection midpoint. Defaults to 0.5.
        resolution (float | Tensor): Spatial resolution in pixels per physical unit,
            used to set the smoothing radius. Defaults to 1.0.

    Returns:
        Tensor: Projected density with local fill-fraction smoothing near interface
        crossings. The output has the same shape, dtype, and device as ``rho_filtered``.

    Raises:
        ValueError: If ``rho_filtered`` is not 2-D.
    """
    if rho_filtered.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(rho_filtered.shape)}")

    resolution_t = torch.as_tensor(
        resolution, dtype=rho_filtered.dtype, device=rho_filtered.device
    )
    eta_t = torch.as_tensor(eta, dtype=rho_filtered.dtype, device=rho_filtered.device)
    dx = 1 / resolution_t
    dy = dx
    smoothing_radius = 0.55 * dx

    rho_projected = tanh_projection(rho_filtered, beta=beta, eta=eta_t)

    grad0, grad1 = _gradient_2d_by_index(rho_filtered)
    grad_norm_sq = (grad0 / dx).square() + (grad1 / dy).square()
    nonzero_norm = grad_norm_sq.abs() > 0
    grad_norm = torch.sqrt(
        torch.where(nonzero_norm, grad_norm_sq, torch.ones_like(grad_norm_sq))
    )
    grad_norm_eff = torch.where(nonzero_norm, grad_norm, torch.ones_like(grad_norm))

    distance = (eta_t - rho_filtered) / grad_norm_eff
    needs_smoothing = nonzero_norm & (distance.abs() < smoothing_radius)

    d_r = distance / smoothing_radius
    safe_d_r = torch.where(needs_smoothing, d_r, torch.zeros_like(d_r))
    safe_d_r2 = safe_d_r.square()
    safe_d_r3 = safe_d_r2 * safe_d_r
    safe_d_r5 = safe_d_r3 * safe_d_r2

    fill = torch.where(
        needs_smoothing,
        0.5 - 15 / 16 * safe_d_r + 5 / 8 * safe_d_r3 - 3 / 16 * safe_d_r5,
        torch.ones_like(safe_d_r),
    )
    fill_minus = torch.where(
        needs_smoothing,
        0.5 + 15 / 16 * safe_d_r - 5 / 8 * safe_d_r3 + 3 / 16 * safe_d_r5,
        torch.ones_like(safe_d_r),
    )

    rho_minus = rho_filtered - smoothing_radius * grad_norm_eff * fill
    rho_plus = rho_filtered + smoothing_radius * grad_norm_eff * fill_minus
    rho_minus_projected = tanh_projection(rho_minus, beta=beta, eta=eta_t)
    rho_plus_projected = tanh_projection(rho_plus, beta=beta, eta=eta_t)
    rho_smoothed = (1 - fill) * rho_minus_projected + fill * rho_plus_projected
    return torch.where(needs_smoothing, rho_smoothed, rho_projected)


class SubpixelSmoothedProjection(nn.Module):
    """Module wrapper for the 2-D subpixel-smoothed projection function.

    The wrapper uses the same implementation for gradient and no-gradient paths,
    but switches the latter to ``torch.inference_mode`` and consumes ``x.data``.
    """

    def __init__(
        self,
        projection_midpoint: float = 0.5,
        resolution: float | None = None,
    ) -> None:
        """Initialize subpixel projection settings.

        Args:
            projection_midpoint (float): Default threshold midpoint used when ``eta`` is
                omitted in ``forward``. Defaults to 0.5.
            resolution (float | None): Default spatial resolution in pixels per physical
                unit used when ``resolution`` is omitted in ``forward``. Defaults to None.
        """
        super().__init__()
        self.projection_midpoint = projection_midpoint
        self.resolution = resolution
        self.subpixel_smoothed_projection_nograd = subpixel_smoothed_projection

        self.subpixel_smoothed_projection_grad = subpixel_smoothed_projection

    def forward(
        self,
        x: Tensor,
        beta: float | Tensor,
        eta: float | Tensor | None = None,
        resolution: float | Tensor | None = None,
    ) -> Tensor:
        """Project a 2-D tensor with subpixel interface smoothing.

        Args:
            x (Tensor): Exactly 2-D density tensor.
            beta (float | Tensor): Projection sharpness parameter.
            eta (float | Tensor | None): Projection midpoint. Defaults to None; the
                stored ``projection_midpoint`` is used.
            resolution (float | Tensor | None): Spatial resolution. Defaults to None;
                the stored ``resolution`` is used and must be non-None after fallback.

        Returns:
            Tensor: Subpixel-smoothed projected density.

        Raises:
            ValueError: If no resolution is available after applying the fallback.
        """
        eta = self.projection_midpoint if eta is None else eta
        resolution = self.resolution if resolution is None else resolution
        if resolution is None:
            raise ValueError("SubpixelSmoothedProjection requires a resolution")
        if x.requires_grad:
            x = self.subpixel_smoothed_projection_grad(
                x, beta=beta, eta=eta, resolution=resolution
            )
        else:
            with torch.inference_mode():
                x = self.subpixel_smoothed_projection_nograd(
                    x.data, beta=beta, eta=eta, resolution=resolution
                )

        return x

    def extra_repr(self) -> str:
        return f"projection_midpoint={self.projection_midpoint}, resolution={self.resolution}"
