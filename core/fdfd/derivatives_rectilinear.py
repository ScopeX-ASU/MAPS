"""Sparse FDFD derivative operators for a 2-D rectilinear grid.

The grid is cell-centered and uses the same ``(Nx, Ny)``/row-major
flattening convention as :mod:`core.fdfd.derivatives`.  ``dxs`` and ``dys``
contain the widths of the cells along the corresponding axis.  Thus, for
example, the forward x derivative in cell ``i`` is

    (u[i + 1] - u[i]) / dxs[i].

For a grid described by the native FDTDx metadata, use
``spacing_from_boundaries(metadata["boundaries"])``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from thirdparty.ceviche.constants import EPSILON_0, ETA_0

__all__ = [
    "spacing_from_boundaries",
    "compute_derivative_matrices",
    "createDws",
    "make_Dxf",
    "make_Dxb",
    "make_Dyf",
    "make_Dyb",
    "create_S_matrices",
]


def spacing_from_boundaries(boundaries: Sequence[np.ndarray]):
    """Return cell widths from native-grid boundary arrays.

    ``boundaries[0]`` and ``boundaries[1]`` must have lengths ``Nx + 1`` and
    ``Ny + 1`` respectively.  Additional axes are ignored deliberately: this
    module implements the 2-D FDFD operators only.
    """

    if len(boundaries) < 2:
        raise ValueError("At least x and y boundary arrays are required")
    widths = tuple(
        np.asarray(axis, dtype=np.float64)[1:] - np.asarray(axis, dtype=np.float64)[:-1]
        for axis in boundaries[:2]
    )
    for axis, values in zip("xy", widths):
        if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
            raise ValueError(f"Grid boundaries for {axis} must be strictly increasing")
    return widths


def _validate_spacing(values, expected, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size != expected:
        raise ValueError(f"{name} must have shape ({expected},), got {values.shape}")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{name} must contain finite positive cell widths")
    return values


def _spacing_from_args(shape, dxs, dys, boundaries):
    if boundaries is not None:
        if dxs is not None or dys is not None:
            raise ValueError("Pass either boundaries or dxs/dys, not both")
        dxs, dys = spacing_from_boundaries(boundaries)
    if dxs is None or dys is None:
        raise ValueError("dxs and dys, or boundaries, must be provided")
    return (
        _validate_spacing(dxs, shape[0], "dxs"),
        _validate_spacing(dys, shape[1], "dys"),
    )


def _sparse_1d(rows, cols, values, size, device):
    indices = torch.as_tensor(np.vstack((rows, cols)), dtype=torch.long, device=device)
    data = torch.as_tensor(values, dtype=torch.complex128, device=device)
    return torch.sparse_coo_tensor(
        indices, data, (size, size), device=device
    ).coalesce()


def _make_1d_forward(widths, bloch, device):
    n = len(widths)
    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=np.int64)
    cols[0::2] = np.arange(n)
    cols[1::2] = (np.arange(n) + 1) % n
    values = np.empty(2 * n, dtype=np.complex128)
    values[0::2] = -1.0 / widths
    values[1::2] = 1.0 / widths
    values[2 * (n - 1) + 1] *= np.exp(1j * bloch)
    return _sparse_1d(rows, cols, values, n, device)


def _make_1d_backward(widths, bloch, device):
    n = len(widths)
    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=np.int64)
    cols[0::2] = np.arange(n)
    cols[1::2] = (np.arange(n) - 1) % n
    values = np.empty(2 * n, dtype=np.complex128)
    values[0::2] = 1.0 / np.roll(widths, 1)
    values[1::2] = -1.0 / np.roll(widths, 1)
    values[1] *= np.exp(-1j * bloch)
    return _sparse_1d(rows, cols, values, n, device)


def _kron(axis_matrix, other_size, device, axis_first=True):
    other = torch.sparse_coo_tensor(
        torch.as_tensor(
            np.vstack((np.arange(other_size), np.arange(other_size))), device=device
        ),
        torch.ones(other_size, dtype=torch.complex128, device=device),
        (other_size, other_size),
        device=device,
    ).coalesce()
    left, right = (axis_matrix, other) if axis_first else (other, axis_matrix)
    left = left.coalesce()
    right = right.coalesce()
    left_indices = left.indices()
    right_indices = right.indices()
    left_nnz = left_indices.shape[1]
    right_nnz = right_indices.shape[1]
    rows = left_indices[0].repeat_interleave(right_nnz) * right.shape[
        0
    ] + right_indices[0].repeat(left_nnz)
    cols = left_indices[1].repeat_interleave(right_nnz) * right.shape[
        1
    ] + right_indices[1].repeat(left_nnz)
    values = left.values().repeat_interleave(right_nnz) * right.values().repeat(
        left_nnz
    )
    return torch.sparse_coo_tensor(
        torch.stack((rows, cols)),
        values,
        (left.shape[0] * right.shape[0], left.shape[1] * right.shape[1]),
        device=device,
    ).coalesce()


def make_Dxf(dxs, shape, bloch_x=0.0, device="cuda:0"):
    return _kron(_make_1d_forward(dxs, bloch_x, device), shape[1], device)


def make_Dxb(dxs, shape, bloch_x=0.0, device="cuda:0"):
    return _kron(_make_1d_backward(dxs, bloch_x, device), shape[1], device)


def make_Dyf(dys, shape, bloch_y=0.0, device="cuda:0"):
    return _kron(
        _make_1d_forward(dys, bloch_y, device), shape[0], device, axis_first=False
    )


def make_Dyb(dys, shape, bloch_y=0.0, device="cuda:0"):
    return _kron(
        _make_1d_backward(dys, bloch_y, device), shape[0], device, axis_first=False
    )


def createDws(
    component, direction, shape, dxs, dys, bloch_x=0.0, bloch_y=0.0, device="cuda:0"
):
    """Create one derivative matrix without PML."""

    nx, ny = shape
    if component == "x":
        if nx == 1:
            return torch.eye(nx * ny, dtype=torch.complex128, device=device).to_sparse()
        return (make_Dxf if direction == "f" else make_Dxb)(dxs, shape, bloch_x, device)
    if component == "y":
        if ny == 1:
            return torch.eye(nx * ny, dtype=torch.complex128, device=device).to_sparse()
        return (make_Dyf if direction == "f" else make_Dyb)(dys, shape, bloch_y, device)
    raise ValueError(f"Unsupported component/direction: {component}{direction}")


def _pml_factor(widths, npml, omega, direction, device):
    n = len(widths)
    if npml == 0:
        return torch.ones(n, dtype=torch.complex128, device=device)
    if not 0 <= npml * 2 <= n:
        raise ValueError(f"Invalid PML thickness {npml} for axis with {n} cells")
    total = float(np.sum(widths[:npml]))
    total = max(total, float(np.sum(widths[-npml:])))
    distances = np.zeros(n)
    distances[:npml] = np.cumsum(widths[:npml][::-1])[::-1] - widths[:npml] / 2
    distances[-npml:] = np.cumsum(widths[-npml:]) - widths[-npml:] / 2
    sigma = np.zeros(n)
    active = distances > 0
    sigma[active] = (
        -(3 + 1) * (-30) / (2 * ETA_0 * total) * (distances[active] / total) ** 3
    )
    # Forward/backward profiles are sampled at opposite sides of each cell.
    if direction == "b":
        distances = np.roll(distances, 1)
        sigma = np.zeros(n)
        active = distances > 0
        sigma[active] = (
            -(3 + 1) * (-30) / (2 * ETA_0 * total) * (distances[active] / total) ** 3
        )
    return torch.as_tensor(
        1 / (1 - 1j * sigma / (omega * EPSILON_0)),
        dtype=torch.complex128,
        device=device,
    )


def create_S_matrices(omega, shape, npml, dxs, dys, device="cuda:0"):
    """Return flattened inverse PML factors for x/y forward/backward rows."""

    sx_f = _pml_factor(dxs, npml[0], omega, "f", device)
    sx_b = _pml_factor(dxs, npml[0], omega, "b", device)
    sy_f = _pml_factor(dys, npml[1], omega, "f", device)
    sy_b = _pml_factor(dys, npml[1], omega, "b", device)
    return tuple(
        (
            tensor[:, None].expand(shape[0], shape[1]).reshape(-1)
            if axis == "x"
            else tensor[None, :].expand(shape[0], shape[1]).reshape(-1)
        )
        for tensor, axis in ((sx_f, "x"), (sx_b, "x"), (sy_f, "y"), (sy_b, "y"))
    )


def compute_derivative_matrices(
    omega,
    shape,
    npml,
    dxs=None,
    dys=None,
    *,
    boundaries=None,
    bloch_x=0.0,
    bloch_y=0.0,
    device="cuda:0",
):
    """Build rectilinear-grid FDFD derivatives, optionally with PML.

    ``boundaries`` may be the ``("x", "y", ...)`` boundary tuple stored in
    ``grid_info_dict["epsilon_map"]``; it is converted to ``dxs``/``dys``
    internally.
    """

    if len(shape) != 2:
        raise ValueError(f"Expected a 2-D shape, got {shape}")
    shape = tuple(int(v) for v in shape)
    dxs, dys = _spacing_from_args(shape, dxs, dys, boundaries)
    matrices = (
        createDws("x", "f", shape, dxs, dys, bloch_x, bloch_y, device),
        createDws("x", "b", shape, dxs, dys, bloch_x, bloch_y, device),
        createDws("y", "f", shape, dxs, dys, bloch_x, bloch_y, device),
        createDws("y", "b", shape, dxs, dys, bloch_x, bloch_y, device),
    )
    factors = create_S_matrices(omega, shape, npml, dxs, dys, device)
    return tuple(
        (factor[:, None] * matrix).coalesce()
        for factor, matrix in zip(factors, matrices)
    )
