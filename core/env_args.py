"""Central registry for MAPS FDTD environment arguments.

Set variables before importing solver/device modules; those modules retain their
existing import-time snapshots. No simulation/configuration interface changes.
Run ``python -m core.env_args`` to list defaults, allowed values, and explanations.
Only MAPS FDTD switches live here, not CUDA/JAX/library-owned environment settings.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class EnvArg:
    """Definition of an environment argument, including its validation contract.

    Attributes:
        default: Default environment string in backend modules.
        kind: Conversion type: bool, int, float, or str.
        allowed: Human-readable domain shown in help and validation errors.
        description: Purpose and numerical/operational implications.
        choices: Optional accepted strings after lowercasing.
        minimum: Optional inclusive numeric lower bound.
        maximum: Optional inclusive numeric upper bound.
        device_default: Historical device-monitor default, if different.
    """

    default: str
    kind: type
    allowed: str
    description: str
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    device_default: str | None = None


ENV_ARGS = {
    "MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM": EnvArg(
        "1",
        bool,
        "0 or 1",
        "Compensate returned complex E/H fields using the sampled source DTFT.",
    ),
    "MAPS_FDTDX_ADJOINT_DEBUG": EnvArg(
        "0", bool, "0 or 1", "Write adjoint diagnostic plots."
    ),
    "MAPS_FDTDX_ADJOINT_DEBUG_DIR": EnvArg(
        "figs/fdtdx_adjoint_debug",
        str,
        "nonempty path",
        "Output directory for adjoint diagnostics; relative to the working directory.",
    ),
    "MAPS_ENABLE_BROADBAND_ADJOINT": EnvArg(
        "0",
        bool,
        "0 or 1",
        "Opt into guarded broadband adjoints with independent-frequency fallback.",
    ),
    "MAPS_BROADBAND_ADJOINT_METHOD": EnvArg(
        "auto",
        str,
        "auto, chebyshev, or legacy (case-insensitive)",
        "Select smooth-profile fitting or the legacy diagnostic source path.",
        choices=("auto", "chebyshev", "legacy"),
    ),
    "MAPS_BROADBAND_ADJOINT_PROFILE_ORDER": EnvArg(
        "3",
        int,
        "integer 0..20",
        "Chebyshev basis term count; 0 searches orders 1..min(5, sample count).",
        minimum=0,
        maximum=20,
    ),
    "MAPS_BROADBAND_ADJOINT_PROFILE_RTOL": EnvArg(
        "1e-3",
        float,
        "finite number >= 0",
        "Maximum relative spatial-profile fit residual before fallback.",
        minimum=0,
    ),
    "MAPS_BROADBAND_ADJOINT_MIN_OVERLAP": EnvArg(
        "0.5",
        float,
        "finite number in [0, 1]",
        "Minimum adjacent normalized-profile overlap before fallback.",
        minimum=0,
        maximum=1,
    ),
    "MAPS_BROADBAND_ADJOINT_MAX_COEFFICIENT_NORM": EnvArg(
        "100",
        float,
        "finite number >= 0",
        "Upper bound on coefficient-weighted temporal waveform amplification.",
        minimum=0,
    ),
    "MAPS_SPARSE_PHASOR": EnvArg(
        "0", bool, "0 or 1", "Record sparse monitor phasors instead of dense volumes."
    ),
    "MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES": EnvArg(
        "1",
        bool,
        "0 or 1",
        "Include visualization planes in sparse device monitor setup.",
    ),
    "MAPS_COMPARE_SPARSE_DENSE": EnvArg(
        "0", bool, "0 or 1", "Record a dense reference to diagnose sparse phasors."
    ),
    "MAPS_WRITE_SPARSE_DENSE_PLOTS": EnvArg(
        "0", bool, "0 or 1", "Write sparse-versus-dense comparison plots."
    ),
    "MAPS_SPARSE_ADJOINT_SOURCE": EnvArg(
        "0",
        bool,
        "0 or 1",
        "Use sparse current blocks in supported adjoint source paths.",
    ),
    "MAPS_ENABLE_PHASOR_H": EnvArg(
        "1",
        bool,
        "0 or 1",
        "Record native H phasors; backend otherwise derives H from E. Device monitor "
        "setup historically defaults to E-only; an explicit value controls both.",
        device_default="0",
    ),
    "MAPS_PHASOR_H_NATIVE_SLAB_CELLS": EnvArg(
        "3",
        int,
        "integer >= 1",
        "Thickness in cells of native-H source-plane slabs.",
        minimum=1,
    ),
    "MAPS_COMPARE_EXACT_INTERPOLATION": EnvArg(
        "0",
        bool,
        "0 or 1",
        "Compare Torch field colocation with exact detector interpolation.",
    ),
}


def get_env_arg(
    name: str, *, context: str = "backend", environ: Mapping[str, str] | None = None
) -> bool | int | float | str:
    """Read and validate a registered argument without caching its value.

    Args:
        name: Registered environment variable name.
        context: ``backend`` or ``device``; preserves historical H-monitor defaults.
        environ: Optional environment mapping for isolated tests; defaults to os.environ.

    Returns:
        The parsed value. Boolean switches accept exactly ``0`` and ``1``.

    Raises:
        KeyError: If name is not registered.
        ValueError: If context or the environment value is invalid.
    """
    if context not in ("backend", "device"):
        raise ValueError(f"Unknown environment argument context: {context!r}")
    spec = ENV_ARGS[name]
    default = spec.default
    if context == "device" and spec.device_default is not None:
        default = spec.device_default
    raw = (os.environ if environ is None else environ).get(name, default)
    try:
        if spec.kind is bool:
            if raw not in ("0", "1"):
                raise ValueError("invalid boolean")
            return raw == "1"
        value = spec.kind(raw)
        if spec.choices:
            value = value.lower()
            if value not in spec.choices:
                raise ValueError("invalid choice")
        if spec.kind in (int, float):
            if not math.isfinite(value):
                raise ValueError("nonfinite number")
            if spec.minimum is not None and value < spec.minimum:
                raise ValueError("below minimum")
            if spec.maximum is not None and value > spec.maximum:
                raise ValueError("above maximum")
        elif not value.strip():
            raise ValueError("empty value")
        return value
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name}={raw!r}: expected {spec.allowed}") from exc


def describe_env_args() -> str:
    """Return the registered defaults, allowed values, and usage explanations."""
    lines = [
        "MAPS FDTD environment arguments (set before importing simulation modules):"
    ]
    for name, spec in ENV_ARGS.items():
        default = repr(spec.default)
        if spec.device_default is not None:
            default += f" (device monitors: {spec.device_default!r})"
        lines.append(
            f"\n{name}\n  Default: {default}; allowed: {spec.allowed}\n  {spec.description}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_env_args())
