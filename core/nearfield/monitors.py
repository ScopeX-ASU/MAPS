"""Adapt verified MAPS observation planes to SI surface samples."""

from collections.abc import Sequence

import torch
from torch import Tensor

from .projector import Surface


def from_maps_fields(
    fields: Tensor | Sequence[Tensor],
    *,
    frequencies_hz: Tensor | Sequence[float],
    coordinates_m: Sequence[Tensor | Sequence[float]],
    normal_axis: int,
    normal_sign: int,
    index: int,
    colocated: bool,
    temporally_aligned: bool,
    unfolded: bool,
    weights_m2: Tensor | None = None,
    boundaries_m: Sequence[Tensor | Sequence[float]] | None = None,
    sparse: bool = False,
    coverage: Tensor | None = None,
) -> Surface:
    """Extract one verified observation plane without changing its phasors.

    Args:
        fields: Six complex Torch tensors ``(Ex, Ey, Ez, Hx, Hy, Hz)`` each
            shaped ``(F, X, Y, Z)``, or one tensor ``(F, 6, X, Y, Z)``.
            All fields must share dtype and device.
        frequencies_hz: Positive frequencies in the field frequency order.
        coordinates_m: Three verified sample-coordinate vectors in meters,
            ordered x, y, z. These are not inferred from solver metadata.
        normal_axis: Plane normal axis, 0 for x, 1 for y, or 2 for z.
        normal_sign: Outward orientation, either -1 or +1.
        index: Nonnegative sample index along the normal axis.
        colocated: Explicit declaration that all six components share the
            supplied physical positions. Must be true.
        temporally_aligned: Declaration that electric and magnetic timing
            corrections have already been applied. Must be true.
        unfolded: Declaration that symmetry reconstruction is complete. Must
            be true; quadrature applies no symmetry multiplier.
        weights_m2: Positive plane areas in square meters, shaped as the two
            transverse axes or as their flattened product.
        boundaries_m: Three increasing physical cell-edge vectors in meters,
            used to compute transverse cell areas when weights are omitted.
            Each edge vector must have one more entry than its coordinate
            vector. Supply exactly one of this argument and ``weights_m2``.
        sparse: Whether fields may contain zero-filled unrecorded samples.
            Requires explicit ``coverage`` when true.
        coverage: Boolean recorded-sample mask, shaped ``(X,Y,Z)``,
            ``(6,X,Y,Z)``, or ``(F,6,X,Y,Z)``. Every tangential electric and
            magnetic sample on this plane must be covered.

    Returns:
        Surface samples in transverse x/y/z order, with fields ``(F,P,3)``.
        Torch extraction preserves the field dtype, device and autograd graph.

    Raises:
        ValueError: If layout, coordinates, representation, quadrature, or
            tangential coverage is invalid.

    Notes:
        No interpolation, half-step phase factor, source-spectrum division,
        impedance conversion or symmetry scaling is performed. Raw and
        source-normalized inputs retain their respective amplitude convention.
        Explicit coordinate verification is required because historical MAPS
        APIs describe both Ez Yee-point and cell-center observations.
    """
    if not (colocated and temporally_aligned and unfolded):
        raise ValueError("Fields must be colocated, temporally aligned, and unfolded")
    components = None
    if not isinstance(fields, Tensor):
        components = tuple(fields)
        if len(components) != 6 or any(not isinstance(v, Tensor) for v in components):
            raise ValueError("Expected six Torch field tensors")
        reference = components[0]
        if any(
            v.shape != reference.shape
            or v.dtype != reference.dtype
            or v.device != reference.device
            for v in components
        ):
            raise ValueError("Field components must share shape, dtype, and device")
        if reference.ndim != 4 or not reference.is_complex():
            raise ValueError("Expected complex field components shaped (F,X,Y,Z)")
        field_shape = (reference.shape[0], 6, *reference.shape[-3:])
    else:
        reference = fields
        field_shape = fields.shape
    if len(field_shape) != 5 or field_shape[1] != 6 or not reference.is_complex():
        raise ValueError("Expected complex fields shaped (F,6,X,Y,Z)")
    if normal_axis not in (0, 1, 2) or normal_sign not in (-1, 1):
        raise ValueError("Invalid plane normal axis or sign")
    shape = field_shape[-3:]
    if not isinstance(index, int) or not 0 <= index < shape[normal_axis]:
        raise ValueError("Plane index is outside the field grid")
    if len(coordinates_m) != 3:
        raise ValueError("coordinates_m must contain x, y, z vectors")
    axes = tuple(
        torch.as_tensor(a, device=reference.device, dtype=reference.real.dtype)
        for a in coordinates_m
    )
    for axis, size in zip(axes, shape):
        if (
            axis.ndim != 1
            or len(axis) != size
            or not torch.isfinite(axis).all()
            or (axis[1:] <= axis[:-1]).any()
        ):
            raise ValueError(
                "Coordinates must be finite, increasing and match the grid"
            )
    if (weights_m2 is None) == (boundaries_m is None):
        raise ValueError("Supply exactly one of weights_m2 and boundaries_m")
    transverse = tuple(a for a in range(3) if a != normal_axis)
    plane_shape = tuple(shape[a] for a in transverse)
    if boundaries_m is not None:
        if len(boundaries_m) != 3:
            raise ValueError("boundaries_m must contain three edge vectors")
        edges = tuple(
            torch.as_tensor(a, device=reference.device, dtype=reference.real.dtype)
            for a in boundaries_m
        )
        for edge, axis in zip(edges, axes):
            if (
                edge.ndim != 1
                or len(edge) != len(axis) + 1
                or not torch.isfinite(edge).all()
                or (edge[1:] <= edge[:-1]).any()
                or (axis < edge[:-1]).any()
                or (axis > edge[1:]).any()
            ):
                raise ValueError("Boundaries must increase and bracket each sample")
        widths = tuple(torch.diff(edges[a]) for a in transverse)
        weights = widths[0][:, None] * widths[1][None, :]
    else:
        weights = torch.as_tensor(
            weights_m2, device=reference.device, dtype=reference.real.dtype
        )
        if tuple(weights.shape) not in (
            plane_shape,
            (plane_shape[0] * plane_shape[1],),
        ):
            raise ValueError("weights_m2 must match the transverse plane shape")
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Plane areas must be finite and positive")
    selection = [slice(None)] * 5
    selection[normal_axis + 2] = index
    if sparse and coverage is None:
        raise ValueError("Sparse fields require explicit coverage")
    if coverage is not None:
        mask = torch.as_tensor(coverage, device=reference.device)
        if mask.dtype != torch.bool:
            raise ValueError("coverage must be boolean")
        if tuple(mask.shape) == tuple(shape):
            mask = mask[None, None].expand(field_shape)
        elif tuple(mask.shape) == (6, *shape):
            mask = mask[None].expand(field_shape)
        elif mask.shape != field_shape:
            raise ValueError(
                "coverage shape must match spatial or component field layout"
            )
        tangential = [a for a in range(6) if a % 3 != normal_axis]
        if not mask[tuple(selection)][:, tangential].all():
            raise ValueError(
                "Missing tangential field coverage on the observation plane"
            )
    if components is None:
        plane = fields[tuple(selection)].flatten(2).transpose(1, 2)
    else:
        component_selection = tuple(selection[:1] + selection[2:])
        plane = torch.stack(
            [v[component_selection].flatten(1) for v in components], dim=-1
        )
    mesh = torch.meshgrid(*(axes[a] for a in transverse), indexing="ij")
    position_components = [None] * 3
    for axis, samples in zip(transverse, mesh):
        position_components[axis] = samples.reshape(-1)
    position_components[normal_axis] = axes[normal_axis][index].expand(mesh[0].numel())
    positions = torch.stack(position_components, dim=-1)
    normals = torch.zeros_like(positions)
    normals[:, normal_axis] = normal_sign
    frequencies = torch.as_tensor(
        frequencies_hz, device=reference.device, dtype=reference.real.dtype
    )
    if frequencies.shape != (field_shape[0],):
        raise ValueError("frequencies_hz must match the leading field dimension")
    return Surface(
        positions_m=positions,
        normals=normals,
        weights_m2=weights.reshape(-1),
        electric_fields=plane[..., :3],
        magnetic_fields=plane[..., 3:],
        frequencies_hz=frequencies,
    )
