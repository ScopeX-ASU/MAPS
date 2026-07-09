from importlib.util import find_spec
from typing import Iterable, Tuple

import torch
from torch import Tensor


def ensure_heat_dependencies() -> None:
    required = {
        "jax": "jax",
        "jax_fem": "jax-fem",
        "meshio": "meshio",
        "gmsh": "gmsh",
        "basix": "fenics-basix",
    }
    missing = [
        pkg_name
        for module_name, pkg_name in required.items()
        if find_spec(module_name) is None
    ]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ImportError(
            "Differentiable heat solver requires extra dependencies that are not "
            f"installed: {missing_str}."
        )


def normalize_spacing(spacing, ndim):
    if isinstance(spacing, (int, float)):
        out = (float(spacing),) * ndim
    else:
        out = tuple(float(v) for v in spacing)
    if len(out) != ndim:
        raise ValueError(f"Expected {ndim} spacing values, but got {len(out)}.")
    if any(v <= 0 for v in out):
        raise ValueError("All spacing values must be positive.")
    return out


def infer_ndim(k_map, dimension):
    if dimension is None:
        ndim = k_map.ndim
    else:
        normalized = dimension.lower()
        if normalized not in {"2d", "3d"}:
            raise ValueError("dimension must be '2d', '3d', or None.")
        ndim = 2 if normalized == "2d" else 3
    if ndim not in (2, 3):
        raise ValueError("HeatSolver only supports 2D and 3D inputs.")
    if k_map.ndim != ndim:
        raise ValueError(
            f"k_map has ndim={k_map.ndim}, but solver is configured for {ndim}D."
        )
    return ndim


def validate_k_map(k_map: Tensor, *, ndim: int) -> None:
    if not isinstance(k_map, Tensor):
        raise TypeError("k_map must be a torch.Tensor.")
    if k_map.ndim != ndim:
        raise ValueError(f"Expected a {ndim}D conductivity map, got {k_map.ndim}D.")
    if not torch.is_floating_point(k_map):
        raise TypeError("k_map must have a floating-point dtype.")


def validate_q_map(q_map: Tensor, k_map: Tensor) -> None:
    if not isinstance(q_map, Tensor):
        raise TypeError("q_map must be a torch.Tensor.")
    if q_map.shape != k_map.shape:
        raise ValueError(
            f"q_map shape {tuple(q_map.shape)} must match k_map shape {tuple(k_map.shape)}."
        )
    if q_map.device != k_map.device:
        raise ValueError("q_map must be on the same device as k_map.")
    if q_map.dtype != k_map.dtype:
        raise ValueError("q_map must have the same dtype as k_map.")


def ensure_positive_conductivity(k_map: Tensor) -> None:
    if torch.any(k_map <= 0):
        raise ValueError("Thermal conductivity must be strictly positive everywhere.")
