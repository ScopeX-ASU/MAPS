"""Small, intentionally slow references for validating GPU rasterization."""

from __future__ import annotations

import numpy as np


def _inside_polygon(x, y, vertices):
    inside = False
    for index in range(len(vertices)):
        x0, y0 = vertices[index - 1]
        x1, y1 = vertices[index]
        crosses = (y0 > y) != (y1 > y)
        if crosses and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def supersampled_reference(structures, grid, background_eps=1.0, samples=32):
    """Return a CPU point-sampled volume average for test/validation use.

    This is not intended for production. It is independent of the CUDA
    clipping implementation and converges to the geometric average as
    ``samples`` increases.

    Args:
        structures: Tidy3D structures to rasterize in material-priority order.
        grid: Tidy3D grid whose cell boundaries define the output voxels.
        background_eps: Relative permittivity outside all structures.
        samples: Number of midpoint samples per cell and Cartesian axis.

    Returns:
        NumPy array of cell-averaged relative permittivity with shape
        ``(Nx, Ny, Nz)``.
    """
    bounds = [np.asarray(axis, dtype=float) for axis in grid.boundaries.to_list]
    output = np.full(
        tuple(axis.size - 1 for axis in bounds), background_eps, dtype=float
    )
    structures = list(structures)
    ordered = sorted(
        enumerate(structures),
        key=lambda item: (getattr(item[1], "priority", None) or 0, item[0]),
    )
    for i in range(output.shape[0]):
        xs = np.linspace(
            bounds[0][i], bounds[0][i + 1], samples, endpoint=False
        ) + np.diff(bounds[0])[i] / (2 * samples)
        for j in range(output.shape[1]):
            ys = np.linspace(
                bounds[1][j], bounds[1][j + 1], samples, endpoint=False
            ) + np.diff(bounds[1])[j] / (2 * samples)
            xx, yy = np.meshgrid(xs, ys, indexing="ij")
            for k in range(output.shape[2]):
                zs = np.linspace(
                    bounds[2][k], bounds[2][k + 1], samples, endpoint=False
                ) + np.diff(bounds[2])[k] / (2 * samples)
                values = np.full(
                    (samples, samples, samples), background_eps, dtype=float
                )
                for _, structure in ordered:
                    geometry = structure.geometry
                    material = float(structure.medium.permittivity)
                    if type(geometry).__name__ == "Box":
                        center = np.asarray(geometry.center)
                        size = np.asarray(geometry.size)
                        mask = (
                            (xx >= center[0] - size[0] / 2)
                            & (xx <= center[0] + size[0] / 2)
                            & (yy >= center[1] - size[1] / 2)
                            & (yy <= center[1] + size[1] / 2)
                        )
                        zmask = (zs >= center[2] - size[2] / 2) & (
                            zs <= center[2] + size[2] / 2
                        )
                    elif type(geometry).__name__ == "PolySlab":
                        mask = np.vectorize(_inside_polygon)(
                            xx, yy, np.asarray(geometry.vertices)
                        )
                        zmask = (zs >= geometry.slab_bounds[0]) & (
                            zs <= geometry.slab_bounds[1]
                        )
                    elif type(geometry).__name__ == "Cylinder":
                        center = np.asarray(geometry.center)
                        axis = int(geometry.axis)
                        transverse = [index for index in range(3) if index != axis]
                        coordinates = (
                            xx[..., None],
                            yy[..., None],
                            zs[None, None, :],
                        )
                        radial_squared = sum(
                            (coordinates[index] - center[index]) ** 2
                            for index in transverse
                        )
                        axial = np.abs(coordinates[axis] - center[axis]) <= (
                            geometry.length / 2
                        )
                        mask = radial_squared <= geometry.radius**2
                        values[mask & axial] = material
                        continue
                    else:
                        raise NotImplementedError(type(geometry).__name__)
                    mask = mask[..., None] & zmask[None, None, :]
                    values[mask] = material
                output[i, j, k] = values.mean()
    return output
