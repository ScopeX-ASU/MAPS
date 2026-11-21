"""Smooth adjoint current families with arbitrary complex frequency weights."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


def factor_current_profiles_torch(
    electric: Tensor | None,
    magnetic: Tensor | None,
    *,
    frequencies: Sequence[float],
    center_frequency: float,
    order: int,
) -> tuple[Tensor | None, Tensor | None, Tensor, dict[str, Any]]:
    """Factor current samples into smooth unit profiles and complex weights.

    Args:
        electric: Complex electric currents shaped ``(frequency, 3, ...)``.
        magnetic: Complex magnetic currents with the same spatial layout.
        frequencies: Target frequencies in Hz, in sample order.
        center_frequency: Carrier frequency for the Chebyshev coordinate.
        order: Number of Chebyshev terms used to measure profile fit quality.

    Returns:
        Electric and magnetic unit profiles, arbitrary complex sample weights,
        and shape-fit diagnostics. Multiplying profiles by weights reconstructs
        the input samples before approximation. Profile norms are algebraic
        scales for factorization, not modal power normalizations.

    Raises:
        ValueError: If profiles are absent or zero, the order is invalid, or
            the target frequency span vanishes.
    """
    if electric is None and magnetic is None:
        raise ValueError("At least one current profile is required.")
    spatial_shape = (electric if electric is not None else magnetic).shape[2:]
    parts = []
    sizes = []
    for profile in (electric, magnetic):
        if profile is not None:
            flat = profile.reshape(profile.shape[0], -1)
            parts.append(flat)
            sizes.append(flat.shape[1])
    if not parts:
        raise ValueError("No electric or magnetic current profiles were produced.")
    combined = torch.cat(parts, dim=1)
    num_frequencies = combined.shape[0]
    if order < 1 or order > num_frequencies:
        raise ValueError(
            f"Chebyshev order must be in [1, {num_frequencies}], got {order}."
        )
    norms = torch.linalg.vector_norm(combined, dim=1)
    if torch.any(norms <= torch.finfo(norms.dtype).tiny):
        raise ValueError("A broadband adjoint profile has zero norm.")
    normalized = combined / norms[:, None]

    reference_index = num_frequencies // 2
    aligned = normalized.clone()
    phases = torch.ones(num_frequencies, dtype=combined.dtype, device=combined.device)
    adjacent_overlaps = []
    for indices in (
        range(reference_index - 1, -1, -1),
        range(reference_index + 1, num_frequencies),
    ):
        previous = aligned[reference_index]
        for index in indices:
            overlap = torch.vdot(previous, normalized[index])
            adjacent_overlaps.append(float(torch.abs(overlap).detach().cpu()))
            phase = (
                torch.exp(-1j * torch.angle(overlap))
                if torch.abs(overlap) > 0
                else torch.ones((), dtype=combined.dtype, device=combined.device)
            )
            phases[index] = phase
            aligned[index] = normalized[index] * phase
            previous = aligned[index]

    frequency_tensor = torch.as_tensor(
        frequencies, dtype=torch.float64, device=combined.device
    )
    half_width = torch.max(torch.abs(frequency_tensor - center_frequency))
    if half_width <= 0:
        raise ValueError(
            "Broadband Chebyshev fitting requires a nonzero frequency span."
        )
    coordinate = (frequency_tensor - center_frequency) / half_width
    vandermonde = torch.ones(
        (num_frequencies, order), dtype=torch.float64, device=combined.device
    )
    if order > 1:
        vandermonde[:, 1] = coordinate
    for degree in range(2, order):
        vandermonde[:, degree] = (
            2 * coordinate * vandermonde[:, degree - 1] - vandermonde[:, degree - 2]
        )
    # Invert only the small frequency matrix; full-domain RHS least-squares
    # would allocate an unnecessarily large cuSOLVER workspace.
    coefficients = torch.linalg.pinv(vandermonde).to(combined.dtype) @ aligned
    reconstructed = vandermonde.to(combined.dtype) @ coefficients
    residuals = torch.linalg.vector_norm(reconstructed - aligned, dim=1)
    weights = norms.to(combined.dtype) / phases

    offset = 0
    electric_aligned = None
    magnetic_aligned = None
    if electric is not None:
        electric_aligned = aligned[:, offset : offset + sizes[0]].reshape(
            num_frequencies, 3, *spatial_shape
        )
        offset += sizes[0]
    if magnetic is not None:
        size = sizes[-1]
        magnetic_aligned = aligned[:, offset : offset + size].reshape(
            num_frequencies, 3, *spatial_shape
        )
    diagnostics = {
        "fit_relative_l2_max": float(torch.max(residuals).detach().cpu()),
        "adjacent_overlap_min": min(adjacent_overlaps, default=1.0),
        "polynomial_half_width_hz": float(half_width.detach().cpu()),
        "fit_condition": float(torch.linalg.cond(vandermonde).detach().cpu()),
        "coefficient_norm_sum": float(
            torch.linalg.vector_norm(coefficients, dim=1).sum().detach().cpu()
        ),
        "coefficient_norm_by_term": torch.linalg.vector_norm(coefficients, dim=1)
        .detach()
        .cpu()
        .tolist(),
    }
    return electric_aligned, magnetic_aligned, weights.to(torch.complex64), diagnostics
