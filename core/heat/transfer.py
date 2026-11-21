import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def _axis_overlap_average_matrix(source_boundaries, target_boundaries):
    """Build intensive-value averaging weights between one-dimensional cells."""
    source = np.asarray(source_boundaries, dtype=np.float64)
    target = np.asarray(target_boundaries, dtype=np.float64)
    for name, boundaries in (("source", source), ("target", target)):
        if boundaries.ndim != 1 or boundaries.size < 2:
            raise ValueError(f"{name}_boundaries must be a non-empty 1D cell grid.")
        if np.any(np.diff(boundaries) <= 0.0):
            raise ValueError(f"{name}_boundaries must be strictly increasing.")
    scale = max(abs(source[0]), abs(source[-1]), abs(target[0]), abs(target[-1]), 1.0)
    if not np.allclose(
        (source[0], source[-1]),
        (target[0], target[-1]),
        rtol=1e-10,
        atol=1e-12 * scale,
    ):
        raise ValueError(
            "Conservative rectilinear remapping requires identical source and "
            "target physical extents."
        )
    overlap = np.maximum(
        0.0,
        np.minimum(target[1:, None], source[None, 1:])
        - np.maximum(target[:-1, None], source[None, :-1]),
    )
    return overlap / np.diff(target)[:, None]


def conservative_rectilinear_remap(values, source_boundaries, target_boundaries):
    """Remap cell-average intensive values using exact Cartesian overlaps.

    Args:
        values: Torch tensor whose final spatial dimensions correspond to the
            source cells.
        source_boundaries: Per-axis source cell-boundary coordinates.
        target_boundaries: Per-axis target cell-boundary coordinates. Each axis
            must have the same physical endpoints as its source counterpart.

    Returns:
        Tensor of target-cell averages. Its leading dimensions, device, and
        dtype match ``values``. For a volumetric density ``q``, the operation
        preserves ``sum(q * cell_volume)`` up to floating-point roundoff.

    Notes:
        The overlap matrices depend only on host-side grid geometry. Tensor
        contractions are native Torch operations and remain differentiable with
        respect to ``values`` on both CPU and CUDA.
    """
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(np.asarray(values))
    source_boundaries = tuple(source_boundaries)
    target_boundaries = tuple(target_boundaries)
    dim = len(source_boundaries)
    if dim == 0 or len(target_boundaries) != dim or values.ndim < dim:
        raise ValueError(
            "values and source/target boundaries must describe the same "
            "non-empty spatial dimension."
        )
    expected_shape = tuple(len(axis) - 1 for axis in source_boundaries)
    if tuple(values.shape[-dim:]) != expected_shape:
        raise ValueError(
            f"values spatial shape must be {expected_shape}, got "
            f"{tuple(values.shape[-dim:])}."
        )

    result = values
    first_spatial_axis = values.ndim - dim
    for offset, (source_axis, target_axis) in enumerate(
        zip(source_boundaries, target_boundaries)
    ):
        weights = torch.as_tensor(
            _axis_overlap_average_matrix(source_axis, target_axis),
            dtype=values.dtype,
            device=values.device,
        )
        axis = first_spatial_axis + offset
        moved = result.movedim(axis, -1)
        result = torch.matmul(moved, weights.transpose(0, 1)).movedim(-1, axis)
    return result


def _grid_centers(shape, spacing):
    axes = [
        (np.arange(n, dtype=np.float64) + 0.5) * float(d)
        for n, d in zip(shape, spacing)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=len(shape)).reshape(-1, len(shape))


def _validate_grid_points(grid_points, grid_shape):
    grid_points = np.asarray(grid_points, dtype=np.float64)
    expected = int(np.prod(grid_shape))
    if grid_points.shape != (expected, len(grid_shape)):
        raise ValueError(
            f"grid_points must have shape ({expected}, {len(grid_shape)}), got {grid_points.shape}."
        )
    return grid_points


def _flatten_index(indices, shape):
    flat = np.zeros(indices.shape[0], dtype=np.int64)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        flat += indices[:, axis] * stride
        stride *= shape[axis]
    return flat


def _build_grid_to_mesh_sparse(
    query_points, grid_shape, spacing=None, grid_points=None
):
    if grid_points is not None:
        return _build_mesh_to_grid_sparse(
            mesh_points=np.asarray(grid_points, dtype=np.float64),
            grid_points=np.asarray(query_points, dtype=np.float64),
            k_neighbors=1,
        )
    dim = len(grid_shape)
    n_rows = query_points.shape[0]
    rows = []
    cols = []
    vals = []

    centers_min = np.array([0.5 * s for s in spacing], dtype=np.float64)
    centers_max = np.array(
        [(n - 0.5) * s for n, s in zip(grid_shape, spacing)], dtype=np.float64
    )

    for row, point in enumerate(query_points):
        point = np.minimum(np.maximum(point, centers_min), centers_max)
        lower = []
        upper = []
        tvals = []
        for axis, (coord, n, step) in enumerate(zip(point, grid_shape, spacing)):
            u = coord / step - 0.5
            lo = int(np.floor(u))
            lo = max(0, min(lo, n - 1))
            hi = min(lo + 1, n - 1)
            if hi == lo:
                t = 0.0
            else:
                t = (coord - (lo + 0.5) * step) / step
            lower.append(lo)
            upper.append(hi)
            tvals.append(float(t))

        num_corners = 2**dim
        for mask in range(num_corners):
            corner = []
            weight = 1.0
            for axis in range(dim):
                use_upper = (mask >> axis) & 1
                if use_upper:
                    corner.append(upper[axis])
                    weight *= tvals[axis]
                else:
                    corner.append(lower[axis])
                    weight *= 1.0 - tvals[axis]
            if weight == 0.0:
                continue
            rows.append(row)
            cols.append(
                _flatten_index(np.asarray([corner], dtype=np.int64), grid_shape)[0]
            )
            vals.append(weight)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float64)
    return indices, values, (n_rows, int(np.prod(grid_shape)))


def _build_mesh_to_grid_sparse(mesh_points, grid_points, k_neighbors=8):
    if cKDTree is None:
        raise ImportError(
            "scipy.spatial.cKDTree is required for fixed-mesh grid projection."
        )

    tree = cKDTree(mesh_points)
    distances, neighbors = tree.query(grid_points, k=min(k_neighbors, len(mesh_points)))

    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
        distances = distances[:, None]

    rows = []
    cols = []
    vals = []
    for row in range(grid_points.shape[0]):
        d = distances[row]
        n = neighbors[row]
        if np.any(d == 0.0):
            cols_row = [int(n[np.argmin(d)])]
            vals_row = [1.0]
        else:
            inv = 1.0 / np.maximum(d, 1e-12)
            inv = inv / inv.sum()
            cols_row = [int(v) for v in n.tolist()]
            vals_row = inv.tolist()

        for col, val in zip(cols_row, vals_row):
            rows.append(row)
            cols.append(col)
            vals.append(float(val))

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float64)
    return indices, values, (grid_points.shape[0], mesh_points.shape[0])


class FixedMeshTransfer(nn.Module):
    def __init__(
        self,
        grid_shape,
        spacing,
        mesh_cell_centers,
        k_neighbors=8,
        grid_points=None,
    ):
        super().__init__()
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.spacing = tuple(float(v) for v in spacing)
        self.dim = len(self.grid_shape)
        self.mesh_cell_centers = np.asarray(mesh_cell_centers, dtype=np.float64)
        self.k_neighbors = int(k_neighbors)
        self.grid_points = (
            _grid_centers(self.grid_shape, self.spacing)
            if grid_points is None
            else _validate_grid_points(grid_points, self.grid_shape)
        )
        g2m_idx, g2m_val, g2m_shape = _build_grid_to_mesh_sparse(
            self.mesh_cell_centers,
            self.grid_shape,
            self.spacing,
            grid_points=self.grid_points,
        )
        m2g_idx, m2g_val, m2g_shape = _build_mesh_to_grid_sparse(
            self.mesh_cell_centers, self.grid_points, k_neighbors=self.k_neighbors
        )

        self.register_buffer(
            "grid_to_mesh_matrix",
            torch.sparse_coo_tensor(g2m_idx, g2m_val, size=g2m_shape).coalesce(),
        )
        self.register_buffer(
            "mesh_to_grid_matrix",
            torch.sparse_coo_tensor(m2g_idx, m2g_val, size=m2g_shape).coalesce(),
        )

    def grid_to_mesh(self, grid_tensor):
        vec = grid_tensor.reshape(-1, 1)
        out = torch.sparse.mm(
            self.grid_to_mesh_matrix.to(
                device=grid_tensor.device, dtype=grid_tensor.dtype
            ),
            vec,
        )
        return out.squeeze(1)

    def mesh_to_grid(self, mesh_tensor):
        vec = mesh_tensor.reshape(-1, 1)
        out = torch.sparse.mm(
            self.mesh_to_grid_matrix.to(
                device=mesh_tensor.device, dtype=mesh_tensor.dtype
            ),
            vec,
        ).squeeze(1)
        return out.reshape(self.grid_shape)
