"""Conservative cell-overlap averaging for two-dimensional material maps."""

import torch


def average_uniform_cells(values, target_size):
    """Average uniform-grid material fractions over the same physical extent.

    Args:
        values: Two-dimensional arithmetic material fractions.
        target_size: Target ``(Nx, Ny)`` cell counts.

    Returns:
        Fractions averaged by cell overlap, including noninteger resize ratios.
    """
    if tuple(values.shape) == tuple(target_size):
        return values
    coords = tuple(
        torch.arange(s, device=values.device, dtype=torch.float64) + 0.5
        for s in values.shape
    )
    boundaries = tuple(
        torch.linspace(0, s, t + 1, device=values.device, dtype=torch.float64)
        for s, t in zip(values.shape, target_size)
    )
    bounds = tuple((0.0, float(s)) for s in values.shape)
    return average_region_cells(values, coords, boundaries, bounds)


def average_region_cells(values, source_coords, target_boundaries, region_bounds):
    """Average a sampled region onto physical, potentially nonuniform cells.

    Source samples represent piecewise-constant cells separated by coordinate
    midpoints; outer edges are the physical region bounds. Target edge cells
    are normalized by their overlap with the region, not their full width:
    region insertion applies the remaining exterior coverage exactly once.

    Args:
        values: Differentiable ``(Nx, Ny)`` cell values or material fractions.
        source_coords: Two strictly increasing source sample-coordinate vectors.
        target_boundaries: Two strictly increasing target cell-edge vectors.
        region_bounds: Two physical ``(lower, upper)`` region bounds.

    Returns:
        Target cell averages, preserving the values' dtype and device.

    Raises:
        ValueError: If coordinate sizes, ordering, or region overlaps are invalid.
    """
    if values.ndim != 2:
        raise ValueError("Region cell averaging requires a two-dimensional tensor")
    weights = []
    for axis in range(2):
        coords = torch.as_tensor(
            source_coords[axis], dtype=torch.float64, device=values.device
        )
        target = torch.as_tensor(
            target_boundaries[axis], dtype=torch.float64, device=values.device
        )
        lower, upper = region_bounds[axis]
        if coords.numel() != values.shape[axis] or target.numel() < 2:
            raise ValueError("Cell coordinates must match the material tensor shape")
        if (
            upper <= lower
            or not torch.all(coords[1:] > coords[:-1])
            or not torch.all(target[1:] > target[:-1])
        ):
            raise ValueError("Cell coordinates and region bounds must be increasing")
        inner = ((coords[1:] + coords[:-1]) / 2).clamp(lower, upper)
        edges = torch.cat(
            (coords.new_tensor([lower]), inner, coords.new_tensor([upper]))
        )
        overlap = (
            torch.minimum(target[1:, None], edges[None, 1:])
            - torch.maximum(target[:-1, None], edges[None, :-1])
        ).clamp_min(0)
        covered_width = overlap.sum(dim=1, keepdim=True)
        if torch.any(covered_width <= 0):
            raise ValueError(
                "Every target region cell must intersect the source region"
            )
        weights.append((overlap / covered_width).to(values.dtype))
    return weights[0] @ values @ weights[1].T
