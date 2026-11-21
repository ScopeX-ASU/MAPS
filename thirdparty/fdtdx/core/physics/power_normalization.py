"""Install physical-area power normalization over the public FDTD-X package.

The public dependency normalizes detector-plane areas by their mean. That
convention loses the physical scale between differently oriented planes. MAPS
instead treats every supplied area weight as an area in square meters.
"""

from __future__ import annotations

import inspect
from functools import wraps
from importlib import import_module
from typing import Any

import fdtdx
import jax
import jax.numpy as jnp
from fdtdx.core.physics import metrics as public_metrics
from fdtdx.core.physics import modes as public_modes

_PUBLIC_COMPUTE_MODE = public_modes.compute_mode
_PUBLIC_COMPUTE_MODE_SYMMETRY_REDUCED = public_modes.compute_mode_symmetry_reduced


def compute_integrated_power(
    E: jax.Array,
    H: jax.Array,
    axis: int,
    area_weights: jax.Array | None = None,
) -> jax.Array:
    """Integrate absolute time-averaged power using physical plane areas.

    Args:
        E: Electric field with the component axis first.
        H: Magnetic field with the component axis first.
        axis: Physical propagation axis.
        area_weights: Physical cell-face areas in square meters, broadcastable
            to the selected Poynting-vector component. A value of ``None``
            preserves the public package's legacy unit weight per sample.

    Returns:
        Absolute integrated power.
    """
    poynting = public_metrics.compute_poynting_flux(E, H)
    flux_density = 0.5 * jnp.real(poynting[axis])
    if area_weights is not None:
        flux_density = flux_density * area_weights
    return jnp.abs(jnp.sum(flux_density))


def normalize_by_poynting_flux(
    E: jax.Array,
    H: jax.Array,
    axis: int,
    area_weights: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Scale electric and magnetic fields to unit physical power.

    Args:
        E: Electric field with the component axis first.
        H: Magnetic field with the component axis first.
        axis: Physical propagation axis.
        area_weights: Physical cell-face areas in square meters.

    Returns:
        Electric and magnetic fields scaled by the inverse square root of
        their integrated power.
    """
    power = compute_integrated_power(E, H, axis, area_weights)
    scale = jnp.sqrt(power)
    return E / scale, H / scale


def _uniform_transverse_coordinates(
    inv_permittivities: jax.Array,
    resolution: float,
) -> tuple[jax.Array, jax.Array]:
    """Build physical edge coordinates for a uniform mode cross-section."""
    transverse_sizes = [size for size in inv_permittivities.shape[1:] if size != 1]
    if len(transverse_sizes) != 2:
        raise ValueError(
            "Mode permittivity must have exactly two non-singleton spatial axes, "
            f"got shape {inv_permittivities.shape}."
        )
    return tuple(jnp.arange(size + 1) * resolution for size in transverse_sizes)


@wraps(_PUBLIC_COMPUTE_MODE)
def compute_mode(*args: Any, **kwargs: Any):
    """Call the public mode solver with physical coordinates on uniform grids."""
    bound = inspect.signature(_PUBLIC_COMPUTE_MODE).bind(*args, **kwargs)
    bound.apply_defaults()
    if bound.arguments["transverse_coords"] is None:
        resolution = bound.arguments["resolution"]
        if resolution is None:
            raise ValueError(
                "resolution is required when transverse_coords is not provided"
            )
        bound.arguments["transverse_coords"] = _uniform_transverse_coordinates(
            bound.arguments["inv_permittivities"], resolution
        )
    return _PUBLIC_COMPUTE_MODE(*bound.args, **bound.kwargs)


@wraps(_PUBLIC_COMPUTE_MODE_SYMMETRY_REDUCED)
def compute_mode_symmetry_reduced(*args: Any, **kwargs: Any):
    """Use physical coordinates for a uniform symmetry-reduced mode solve."""
    bound = inspect.signature(_PUBLIC_COMPUTE_MODE_SYMMETRY_REDUCED).bind(
        *args, **kwargs
    )
    bound.apply_defaults()
    if bound.arguments["transverse_coords"] is None:
        resolution = bound.arguments["resolution"]
        if resolution is None:
            raise ValueError(
                "resolution is required when transverse_coords is not provided"
            )
        bound.arguments["transverse_coords"] = _uniform_transverse_coordinates(
            bound.arguments["inv_permittivities"], resolution
        )
    return _PUBLIC_COMPUTE_MODE_SYMMETRY_REDUCED(*bound.args, **bound.kwargs)


def _replace_module_attributes(module_name: str, replacements: dict[str, Any]) -> None:
    """Replace imported FDTD-X symbols when the target module is available."""
    try:
        module = import_module(module_name)
    except ImportError:
        return
    for name, value in replacements.items():
        if hasattr(module, name):
            setattr(module, name, value)


def install_physical_power_normalization() -> None:
    """Register MAPS' physical-power convention on the public package."""
    metric_replacements = {
        "compute_integrated_power": compute_integrated_power,
        "normalize_by_poynting_flux": normalize_by_poynting_flux,
    }
    mode_replacements = {
        "compute_mode": compute_mode,
        "compute_mode_symmetry_reduced": compute_mode_symmetry_reduced,
    }

    _replace_module_attributes("fdtdx.core.physics.metrics", metric_replacements)
    _replace_module_attributes(
        "fdtdx.core.physics.modes",
        {
            **metric_replacements,
            **mode_replacements,
        },
    )
    _replace_module_attributes(
        "fdtdx.objects.sources.linear_polarization", metric_replacements
    )
    _replace_module_attributes("fdtdx.objects.sources.mode", mode_replacements)
    _replace_module_attributes(
        "fdtdx.objects.detectors.mode",
        {**metric_replacements, **mode_replacements},
    )

    for name, value in {**metric_replacements, **mode_replacements}.items():
        if hasattr(fdtdx, name):
            setattr(fdtdx, name, value)
