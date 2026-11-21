"""Material-to-geometry projection helpers for digital primitives."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .utils import tanh_projection


class EffectiveMediumProjector:
    """Convert normalized effective material values into geometric fill fractions.

    The initial implementation uses a linear-permittivity effective-medium model.
    The host and inclusion endpoints are explicit so the same API can represent
    high-index posts in a low-index host or low-index holes in a high-index host.
    """

    def __init__(self, model: str = "linear_eps") -> None:
        """Initialize a projector.

        Args:
            model: Effective-medium model. ``"linear_eps"`` is currently supported.

        Raises:
            ValueError: If ``model`` is unsupported.
        """
        if model != "linear_eps":
            raise ValueError(
                f"Unsupported effective-medium model {model!r}; "
                "only 'linear_eps' is currently supported."
            )
        self.model = model

    def effective_eps_to_fill_fraction(
        self,
        eps_eff: Tensor | float,
        eps_host: Tensor | float,
        eps_inclusion: Tensor | float,
        *,
        clamp: bool = True,
    ) -> Tensor:
        """Invert the configured effective-medium relation.

        Args:
            eps_eff: Effective relative permittivity.
            eps_host: Relative permittivity of the geometric host material.
            eps_inclusion: Relative permittivity of the material occupying the
                primitive, if present.
            clamp: Clamp the returned fraction to ``[0, 1]``. Defaults to True.

        Returns:
            Tensor: Inclusion fill fraction with the broadcasted input shape.

        Raises:
            ValueError: If host and inclusion permittivities are equal.
        """
        eps_eff_t = torch.as_tensor(eps_eff)
        if not (eps_eff_t.is_floating_point() or eps_eff_t.is_complex()):
            eps_eff_t = eps_eff_t.to(torch.get_default_dtype())
        eps_host_t = torch.as_tensor(
            eps_host, dtype=eps_eff_t.dtype, device=eps_eff_t.device
        )
        eps_inclusion_t = torch.as_tensor(
            eps_inclusion, dtype=eps_eff_t.dtype, device=eps_eff_t.device
        )
        denominator = eps_inclusion_t - eps_host_t
        tolerance = torch.finfo(eps_eff_t.real.dtype).eps
        if bool(torch.any(torch.abs(denominator) <= tolerance)):
            raise ValueError("eps_host and eps_inclusion must be different")
        fill_fraction = (eps_eff_t - eps_host_t) / denominator
        return fill_fraction.clamp(0.0, 1.0) if clamp else fill_fraction

    def rho_to_fill_fraction(
        self,
        rho: Tensor,
        *,
        eps_low: Tensor | float,
        eps_high: Tensor | float,
        eps_host: Tensor | float,
        eps_inclusion: Tensor | float,
        beta: float | Tensor | None = None,
        eta: float | Tensor = 0.5,
    ) -> Tensor:
        """Map normalized material variables to inclusion fill fractions.

        Args:
            rho: Normalized material variable where zero and one denote the low
                and high permittivity endpoints, respectively.
            eps_low: Physical permittivity represented by ``rho=0``.
            eps_high: Physical permittivity represented by ``rho=1``.
            eps_host: Host permittivity used by the geometric fill model.
            eps_inclusion: Inclusion permittivity used by the geometric fill model.
            beta: Optional tanh-projection sharpness. If omitted, ``rho`` is only
                clipped to ``[0, 1]``.
            eta: Projection midpoint when ``beta`` is supplied.

        Returns:
            Tensor: Inclusion fill fraction in ``[0, 1]``.
        """
        rho = self.project_rho(rho, beta=beta, eta=eta)
        eps_eff = torch.as_tensor(eps_low, dtype=rho.dtype, device=rho.device) + rho * (
            torch.as_tensor(eps_high, dtype=rho.dtype, device=rho.device)
            - torch.as_tensor(eps_low, dtype=rho.dtype, device=rho.device)
        )
        return self.effective_eps_to_fill_fraction(
            eps_eff,
            eps_host,
            eps_inclusion,
        )

    @staticmethod
    def project_rho(
        rho: Tensor,
        *,
        beta: float | Tensor | None = None,
        eta: float | Tensor = 0.5,
    ) -> Tensor:
        """Project normalized material values toward binary endpoints.

        Args:
            rho: Normalized material values.
            beta: Optional tanh sharpness; omitted means clipping only.
            eta: Projection midpoint. Defaults to 0.5.

        Returns:
            Tensor: Projected values in ``[0, 1]``.
        """
        rho = rho.clamp(0.0, 1.0)
        if beta is not None:
            rho = tanh_projection(rho, beta=beta, eta=eta)
        return rho

    @staticmethod
    def fill_fraction_to_scale(
        fill_fraction: Tensor,
        mapping: Literal["isotropic", "x_only", "y_only"] = "isotropic",
    ) -> Tensor:
        """Convert area fill fraction to a box scale factor.

        Args:
            fill_fraction: Inclusion area fraction in ``[0, 1]``.
            mapping: ``"isotropic"`` uses ``sqrt(f)``; axis-specific modes use
                ``f`` for the selected axis and leave the other axis unchanged.

        Returns:
            Tensor: Scale factors with a trailing dimension of length two.

        Raises:
            ValueError: If ``mapping`` is unsupported.
        """
        fill_fraction = fill_fraction.clamp(0.0, 1.0)
        if mapping == "isotropic":
            scale = torch.sqrt(fill_fraction)
            return torch.stack((scale, scale), dim=-1)
        if mapping == "x_only":
            return torch.stack((fill_fraction, torch.ones_like(fill_fraction)), dim=-1)
        if mapping == "y_only":
            return torch.stack((torch.ones_like(fill_fraction), fill_fraction), dim=-1)
        raise ValueError(
            f"Unsupported box fill-fraction mapping {mapping!r}; expected "
            "'isotropic', 'x_only', or 'y_only'."
        )
