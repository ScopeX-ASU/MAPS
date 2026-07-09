from dataclasses import dataclass
from types import SimpleNamespace
from typing import Tuple

import numpy as np

_FACE_KEYS = {
    2: ("xmin", "xmax", "ymin", "ymax"),
    3: ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
}


@dataclass(frozen=True)
class StructuredMesh:
    shape: Tuple[int, ...]
    spacing: Tuple[float, ...]
    points: np.ndarray
    cells: np.ndarray
    ele_type: str
    dim: int
    domain_lengths: Tuple[float, ...]
    cell_centers: np.ndarray
    tol: float

    def as_problem_mesh(self) -> SimpleNamespace:
        return SimpleNamespace(
            points=self.points, cells=self.cells, ele_type=self.ele_type
        )

    @property
    def face_keys(self):
        return _FACE_KEYS[self.dim]


@dataclass(frozen=True)
class FixedMesh:
    points: np.ndarray
    cells: np.ndarray
    ele_type: str
    dim: int
    domain_lengths: Tuple[float, ...]
    cell_centers: np.ndarray
    tol: float
    shape: Tuple[int, ...] = ()

    def as_problem_mesh(self) -> SimpleNamespace:
        return SimpleNamespace(
            points=self.points, cells=self.cells, ele_type=self.ele_type
        )

    @property
    def face_keys(self):
        return _FACE_KEYS[self.dim]


def build_structured_mesh(shape, spacing):
    if len(shape) == 2:
        return _build_2d_mesh(shape, spacing)
    if len(shape) == 3:
        return _build_3d_mesh(shape, spacing)
    raise ValueError("Only 2D and 3D structured meshes are supported.")


def build_fixed_mesh(points, cells, ele_type=None):
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    if points.ndim != 2:
        raise ValueError("points must have shape [num_points, dim].")
    if cells.ndim != 2:
        raise ValueError("cells must have shape [num_cells, num_nodes_per_cell].")

    dim = points.shape[1]
    if dim not in (2, 3):
        raise ValueError("Only 2D and 3D fixed meshes are supported.")

    if ele_type is None:
        if dim == 2:
            if cells.shape[1] == 3:
                ele_type = "TRI3"
            elif cells.shape[1] == 4:
                ele_type = "QUAD4"
            elif cells.shape[1] == 6:
                ele_type = "TRI6"
        elif dim == 3:
            if cells.shape[1] == 4:
                ele_type = "TET4"
            elif cells.shape[1] == 8:
                ele_type = "HEX8"
            elif cells.shape[1] == 10:
                ele_type = "TET10"
        if ele_type is None:
            raise ValueError("Could not infer ele_type from cell connectivity.")

    cell_centers = np.mean(points[cells], axis=1)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    domain_lengths = tuple((maxs - mins).tolist())
    tol = max(domain_lengths) * 1e-6 + 1e-9
    return FixedMesh(
        points=points,
        cells=cells,
        ele_type=ele_type,
        dim=dim,
        domain_lengths=domain_lengths,
        cell_centers=cell_centers,
        tol=tol,
    )


def make_face_location_fn(face: str, mesh: StructuredMesh):
    axis = "xyz".index(face[0])
    target = 0.0 if face.endswith("min") else mesh.domain_lengths[axis]
    tol = mesh.tol

    def location_fn(point):
        import jax.numpy as jnp

        return jnp.isclose(point[axis], target, atol=tol)

    return location_fn


def _build_2d_mesh(shape, spacing):
    nx, ny = shape
    dx, dy = spacing
    x = np.linspace(0.0, nx * dx, nx + 1, dtype=np.float64)
    y = np.linspace(0.0, ny * dy, ny + 1, dtype=np.float64)
    xv, yv = np.meshgrid(x, y, indexing="ij")
    points = np.stack((xv, yv), axis=2).reshape(-1, 2)

    point_ids = np.arange(points.shape[0]).reshape(nx + 1, ny + 1)
    c0 = point_ids[:-1, :-1]
    c1 = point_ids[1:, :-1]
    c2 = point_ids[1:, 1:]
    c3 = point_ids[:-1, 1:]
    cells = np.stack((c0, c1, c2, c3), axis=2).reshape(-1, 4)

    xc = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    yc = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    xcv, ycv = np.meshgrid(xc, yc, indexing="ij")
    cell_centers = np.stack((xcv, ycv), axis=2).reshape(-1, 2)

    return StructuredMesh(
        shape=shape,
        spacing=spacing,
        points=points,
        cells=cells,
        ele_type="QUAD4",
        dim=2,
        domain_lengths=(nx * dx, ny * dy),
        cell_centers=cell_centers,
        tol=max(spacing) * 1e-6 + 1e-9,
    )


def _build_3d_mesh(shape, spacing):
    nx, ny, nz = shape
    dx, dy, dz = spacing
    x = np.linspace(0.0, nx * dx, nx + 1, dtype=np.float64)
    y = np.linspace(0.0, ny * dy, ny + 1, dtype=np.float64)
    z = np.linspace(0.0, nz * dz, nz + 1, dtype=np.float64)
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack((xv, yv, zv), axis=3).reshape(-1, 3)

    point_ids = np.arange(points.shape[0]).reshape(nx + 1, ny + 1, nz + 1)
    c0 = point_ids[:-1, :-1, :-1]
    c1 = point_ids[1:, :-1, :-1]
    c2 = point_ids[1:, 1:, :-1]
    c3 = point_ids[:-1, 1:, :-1]
    c4 = point_ids[:-1, :-1, 1:]
    c5 = point_ids[1:, :-1, 1:]
    c6 = point_ids[1:, 1:, 1:]
    c7 = point_ids[:-1, 1:, 1:]
    cells = np.stack((c0, c1, c2, c3, c4, c5, c6, c7), axis=3).reshape(-1, 8)

    xc = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    yc = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    zc = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    xcv, ycv, zcv = np.meshgrid(xc, yc, zc, indexing="ij")
    cell_centers = np.stack((xcv, ycv, zcv), axis=3).reshape(-1, 3)

    return StructuredMesh(
        shape=shape,
        spacing=spacing,
        points=points,
        cells=cells,
        ele_type="HEX8",
        dim=3,
        domain_lengths=(nx * dx, ny * dy, nz * dz),
        cell_centers=cell_centers,
        tol=max(spacing) * 1e-6 + 1e-9,
    )
