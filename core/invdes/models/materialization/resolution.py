"""Torch-native resolution conversion for MAPS design materialization.

This module defines the canonical HR-to-LR tensor conversion used by the
inverse-design pipeline. The conversion operates on normalized density and,
when requested, performs downsampling in relative-permittivity space so that
the LR representation is the subpixel-accurate material realization of the
HR design rather than an independently generated LR geometry.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn import functional as F

__all__ = ["convert_resolution_tensor"]


def convert_resolution_tensor(
    x: torch.Tensor,
    source_resolution: Optional[float] = None,
    target_resolution: Optional[float] = None,
    intplt_mode: str = "nearest",
    subpixel_smoothing: bool = False,
    eps_r: Optional[float] = None,
    eps_bg: Optional[float] = None,
    target_size: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Convert a normalized design tensor between spatial resolutions.

    Args:
        x: Torch tensor with 2 or 3 dimensions. The final two dimensions are
            interpreted as spatial ``(nx, ny)`` dimensions. The leading
            dimension is preserved for 3-D inputs.
        source_resolution: Source grid resolution in pixels per micrometer.
            Required when ``target_size`` is omitted.
        target_resolution: Target grid resolution in pixels per micrometer.
            Required when ``target_size`` is omitted.
        intplt_mode: PyTorch interpolation mode used for ordinary resampling.
            The legacy MAPS default is ``"nearest"``.
        subpixel_smoothing: When ``True`` and both target dimensions are smaller
            than the source dimensions, convert normalized density to relative
            permittivity, apply adaptive area averaging, and convert back to
            normalized density.
        eps_r: Relative permittivity of the material endpoint used for
            permittivity-space subpixel smoothing.
        eps_bg: Relative permittivity of the background endpoint used for
            permittivity-space subpixel smoothing.
        target_size: Explicit target ``(nx, ny)`` shape. When provided, it
            takes precedence over ``source_resolution`` and
            ``target_resolution``.

    Returns:
        A Torch tensor with final spatial shape ``target_size``. If the input
        already has the requested shape, the original tensor object is returned
        unchanged. Gradients propagate through ordinary interpolation and
        permittivity-space subpixel averaging.

    Raises:
        TypeError: If ``x`` is not a Torch tensor.
        ValueError: If the input rank is unsupported, target information is
            missing, target dimensions are invalid, or subpixel smoothing lacks
            valid material endpoints.

    Invariants:
        - The operation remains Torch-native and differentiable.
        - LR density is derived from HR density; it is never independently
          parameterized.
        - With subpixel smoothing, averaging occurs in permittivity space before
          conversion back to normalized density.
        - The subpixel branch intentionally preserves the legacy arithmetic
          expression structure so existing MAPS gradient behavior is unchanged.
        - The function does not mutate ``x`` in place.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected a 2-D or 3-D tensor, got shape {tuple(x.shape)}")

    if target_size is None:
        if source_resolution is None or target_resolution is None:
            raise ValueError(
                "source_resolution and target_resolution are required when "
                "target_size is not provided"
            )
        target_size = tuple(
            max(1, int(round(size * target_resolution / source_resolution)))
            for size in x.shape[-2:]
        )
    else:
        target_size = tuple(int(size) for size in target_size)
        if len(target_size) != 2 or any(size <= 0 for size in target_size):
            raise ValueError(
                f"target_size must contain two positive integers, got {target_size}"
            )

    if x.shape[-2:] == target_size:
        return x

    downsampling = target_size[0] < x.shape[-2] and target_size[1] < x.shape[-1]
    if downsampling and subpixel_smoothing:
        if eps_r is None or eps_bg is None:
            raise ValueError(
                "eps_r and eps_bg are required for permittivity-space "
                "subpixel smoothing"
            )
        if eps_r == eps_bg:
            raise ValueError("eps_r and eps_bg must be different")

        # Keep the exact legacy arithmetic structure. This is intentionally not
        # algebraically factored into a cached denominator because the backward
        # graph is part of the regression contract.
        x = eps_bg + (eps_r - eps_bg) * x
        x = F.adaptive_avg_pool2d(
            x[None, None],
            output_size=target_size,
        )[0, 0]
        x = (x - eps_bg) / (eps_r - eps_bg)
        return x

    if x.ndim == 2:
        return F.interpolate(
            x.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode=intplt_mode,
        )[0, 0]

    return F.interpolate(
        x.unsqueeze(0),
        size=target_size,
        mode=intplt_mode,
    )[0]
