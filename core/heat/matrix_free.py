"""Structured matrix-free operators for rectilinear Q1 heat elements.

This module contains only operator construction and application.  Solver dispatch
is kept in :mod:`core.heat.problem` so the existing sparse backends remain
unchanged.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - guarded by the heat dependency checks
    jax = None
    jnp = None


def _local_node_offsets(dim: int) -> np.ndarray:
    """Return local Q1 node coordinates in the repository's cell ordering."""
    if dim == 2:
        return np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.int8)
    if dim == 3:
        return np.asarray(
            (
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            ),
            dtype=np.int8,
        )
    raise ValueError(f"Matrix-free heat supports only 2D and 3D, got {dim}D.")


def _directional_local_matrices(dim: int) -> np.ndarray:
    """Build dimensionless directional Q1 stiffness matrices.

    The physical coefficient ``prod(h[b], b != axis) / (6**(dim-1) h[axis])``
    is applied separately for every element.
    """
    offsets = _local_node_offsets(dim)
    stiffness = np.asarray(((1.0, -1.0), (-1.0, 1.0)), dtype=np.float64)
    mass = np.asarray(((2.0, 1.0), (1.0, 2.0)), dtype=np.float64)
    matrices = np.empty((dim, offsets.shape[0], offsets.shape[0]), dtype=np.float64)
    for axis in range(dim):
        matrix = np.empty((offsets.shape[0], offsets.shape[0]), dtype=np.float64)
        for row, row_offset in enumerate(offsets):
            for col, col_offset in enumerate(offsets):
                value = stiffness[row_offset[axis], col_offset[axis]]
                for transverse_axis in range(dim):
                    if transverse_axis != axis:
                        value *= mass[
                            row_offset[transverse_axis], col_offset[transverse_axis]
                        ]
                matrix[row, col] = value
        matrices[axis] = matrix
    return matrices


@dataclass(frozen=True)
class RectilinearOperatorData:
    """Geometry and indexing for a reduced-DOF matrix-free heat operator.

    Attributes:
        axes: Strictly increasing nodal coordinate arrays, ordered ``x, y[, z]``.
        cell_shape: Number of Q1 elements along each coordinate axis.
        node_shape: Number of nodes along each coordinate axis.
        cells: Canonical cell-to-node connectivity with shape
            ``[num_cells, 2**dim]``.
        cell_widths: Per-axis element widths broadcast to shape
            ``[*cell_shape, dim]`` and flattened to ``[num_cells, dim]``.
        directional_matrices: Dimensionless local stiffness contributions with
            shape ``[dim, 2**dim, 2**dim]``.
        free_rows: Full nodal indices retained by ``reduced_free_dof``.
        constrained_rows: Full nodal indices prescribed by Dirichlet conditions.
        constrained_values: Prescribed values corresponding to
            ``constrained_rows``.
        cell_free_ids: Free-DOF IDs for every cell node, or ``-1`` for a
            constrained node.
        output_positions: Flattened element-local positions whose output node is
            free.
        output_free_ids: Free-DOF destinations corresponding to
            ``output_positions``.
        cell_unit_stiffness: Optional geometry-only JAX-FEM element stiffness
            matrices. These are independent of conductivity.
        cell_unit_load: Optional geometry-only unit-source element load vectors.
    """

    axes: Tuple[np.ndarray, ...]
    cell_shape: Tuple[int, ...]
    node_shape: Tuple[int, ...]
    cells: np.ndarray
    cell_widths: np.ndarray
    directional_matrices: np.ndarray
    free_rows: np.ndarray
    constrained_rows: np.ndarray
    constrained_values: np.ndarray
    cell_free_ids: np.ndarray
    output_positions: np.ndarray
    output_free_ids: np.ndarray
    cell_unit_stiffness: np.ndarray | None = None
    cell_unit_load: np.ndarray | None = None

    @property
    def dim(self) -> int:
        """Return the spatial dimension."""
        return len(self.axes)

    @property
    def num_nodes(self) -> int:
        """Return the number of full nodal degrees of freedom."""
        return int(np.prod(self.node_shape))


def build_rectilinear_operator_data(
    mesh,
    free_rows,
    constrained_rows=(),
    constrained_values=(),
    *,
    cell_unit_stiffness=None,
    cell_unit_load=None,
    coordinate_rtol: float = 1e-11,
    coordinate_atol: float = 1e-12,
) -> RectilinearOperatorData:
    """Validate a canonical rectilinear mesh and build static operator data.

    Args:
        mesh: Heat mesh containing nodal ``points`` and Q1 ``cells``.
        free_rows: Full nodal indices retained in the reduced system.
        constrained_rows: Full nodal Dirichlet indices.
        constrained_values: Temperatures prescribed on ``constrained_rows``.
        cell_unit_stiffness: Optional geometry-only element matrices with shape
            ``[num_cells, 2**dim, 2**dim]``. Supplying matrices computed from
            JAX-FEM geometry preserves its floating-point metric exactly.
        cell_unit_load: Optional unit volumetric-source element vectors with
            shape ``[num_cells, 2**dim]``.
        coordinate_rtol: Relative tolerance for Cartesian-product validation.
        coordinate_atol: Absolute tolerance for Cartesian-product validation.

    Returns:
        Immutable geometry and reduced-DOF indexing data.

    Raises:
        ValueError: If the mesh is not a canonical axis-aligned QUAD4/HEX8 grid
            or if the supplied DOF partition is inconsistent.

    Notes:
        Canonical validation is intentional. It prevents the experimental
        structured backend from silently changing the discretization of warped
        or unstructured fixed meshes.
    """
    points = np.asarray(mesh.points, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    dim = int(mesh.dim)
    expected_ele_type = {2: "QUAD4", 3: "HEX8"}.get(dim)
    if expected_ele_type is None or str(mesh.ele_type).upper() != expected_ele_type:
        raise ValueError(
            "jax_matrix_free_solver requires axis-aligned QUAD4 or HEX8 elements."
        )

    axes = tuple(np.unique(points[:, axis]) for axis in range(dim))
    node_shape = tuple(int(axis.size) for axis in axes)
    cell_shape = tuple(size - 1 for size in node_shape)
    if any(size < 1 for size in cell_shape):
        raise ValueError("Every rectilinear heat axis must contain at least one cell.")
    if int(np.prod(node_shape)) != points.shape[0]:
        raise ValueError("Heat mesh points do not form a complete Cartesian product.")
    if int(np.prod(cell_shape)) != cells.shape[0]:
        raise ValueError("Heat mesh cells do not fill the rectilinear Cartesian grid.")

    point_mesh = np.meshgrid(*axes, indexing="ij")
    expected_points = np.stack(point_mesh, axis=dim).reshape(-1, dim)
    if not np.allclose(
        points, expected_points, rtol=coordinate_rtol, atol=coordinate_atol
    ):
        raise ValueError(
            "jax_matrix_free_solver currently requires canonical C-order nodal "
            "ordering on a Cartesian-product grid."
        )

    node_ids = np.arange(points.shape[0], dtype=np.int64).reshape(node_shape)
    offsets = _local_node_offsets(dim)
    expected_cells = np.stack(
        [
            node_ids[
                tuple(
                    slice(int(offset[axis]), int(offset[axis]) + cell_shape[axis])
                    for axis in range(dim)
                )
            ]
            for offset in offsets
        ],
        axis=dim,
    ).reshape(-1, offsets.shape[0])
    if not np.array_equal(cells, expected_cells):
        raise ValueError(
            "jax_matrix_free_solver currently requires canonical rectilinear "
            "cell ordering and connectivity."
        )

    width_axes = tuple(np.diff(axis) for axis in axes)
    if any(np.any(widths <= 0.0) for widths in width_axes):
        raise ValueError("Rectilinear heat coordinates must be strictly increasing.")
    width_mesh = np.meshgrid(*width_axes, indexing="ij")
    cell_widths = np.stack(width_mesh, axis=dim).reshape(-1, dim)

    free_rows = np.asarray(free_rows, dtype=np.int64).reshape(-1)
    constrained_rows = np.asarray(constrained_rows, dtype=np.int64).reshape(-1)
    constrained_values = np.asarray(constrained_values, dtype=np.float64).reshape(-1)
    if constrained_rows.size != constrained_values.size:
        raise ValueError("Dirichlet rows and values must have the same length.")
    if np.intersect1d(free_rows, constrained_rows).size:
        raise ValueError("Free and constrained heat DOFs must be disjoint.")
    if free_rows.size + constrained_rows.size != points.shape[0]:
        raise ValueError("Free and constrained heat DOFs must partition all nodes.")

    free_id = np.full(points.shape[0], -1, dtype=np.int64)
    free_id[free_rows] = np.arange(free_rows.size, dtype=np.int64)
    cell_free_ids = free_id[cells]
    flat_cell_free_ids = cell_free_ids.reshape(-1)
    output_positions = np.nonzero(flat_cell_free_ids >= 0)[0].astype(
        np.int64, copy=False
    )
    output_free_ids = flat_cell_free_ids[output_positions]

    num_local_nodes = offsets.shape[0]
    if cell_unit_stiffness is not None:
        cell_unit_stiffness = np.asarray(cell_unit_stiffness, dtype=np.float64)
        expected_shape = (cells.shape[0], num_local_nodes, num_local_nodes)
        if cell_unit_stiffness.shape != expected_shape:
            raise ValueError(
                f"cell_unit_stiffness must have shape {expected_shape}, got "
                f"{cell_unit_stiffness.shape}."
            )
    if cell_unit_load is not None:
        cell_unit_load = np.asarray(cell_unit_load, dtype=np.float64)
        expected_shape = (cells.shape[0], num_local_nodes)
        if cell_unit_load.shape != expected_shape:
            raise ValueError(
                f"cell_unit_load must have shape {expected_shape}, got "
                f"{cell_unit_load.shape}."
            )

    return RectilinearOperatorData(
        axes=axes,
        cell_shape=cell_shape,
        node_shape=node_shape,
        cells=cells,
        cell_widths=cell_widths,
        directional_matrices=_directional_local_matrices(dim),
        free_rows=free_rows,
        constrained_rows=constrained_rows,
        constrained_values=constrained_values,
        cell_free_ids=cell_free_ids,
        output_positions=output_positions,
        output_free_ids=output_free_ids,
        cell_unit_stiffness=cell_unit_stiffness,
        cell_unit_load=cell_unit_load,
    )


def _cell_directional_coefficients(cell_widths):
    """Return physical coefficients for directional Q1 stiffness terms."""
    dim = cell_widths.shape[1]
    volume = jnp.prod(cell_widths, axis=1)
    return volume[:, None] / ((6.0 ** (dim - 1)) * jnp.square(cell_widths))


def apply_element_operator(
    cell_values,
    conductivity,
    cell_widths,
    local_matrices,
    cell_unit_stiffness=None,
):
    """Apply all local Q1 conductivity matrices without assembling them.

    Args:
        cell_values: Nodal values with shape ``[num_cells, 2**dim]``.
        conductivity: Positive cell-wise conductivity with shape ``[num_cells]``.
        cell_widths: Rectilinear cell widths with shape ``[num_cells, dim]``.
        local_matrices: Dimensionless directional matrices with shape
            ``[dim, 2**dim, 2**dim]``.
        cell_unit_stiffness: Optional JAX-FEM-compatible unit-conductivity
            element matrices with shape ``[num_cells, 2**dim, 2**dim]``.

    Returns:
        Element-local residual contributions with the same shape as
        ``cell_values``.
    """
    if cell_unit_stiffness is not None:
        return conductivity[:, None] * jnp.einsum(
            "eij,ej->ei", cell_unit_stiffness, cell_values, optimize=True
        )

    coefficients = _cell_directional_coefficients(cell_widths)
    directional_action = jnp.einsum(
        "aij,ej->eai", local_matrices, cell_values, optimize=True
    )
    return conductivity[:, None] * jnp.sum(
        coefficients[:, :, None] * directional_action, axis=1
    )


def matvec_full(x_full, conductivity, operator_data: RectilinearOperatorData):
    """Apply the structured FEM operator to all nodal DOFs.

    Args:
        x_full: Full nodal field with shape ``[num_nodes]``.
        conductivity: Cell-wise conductivity with shape ``[num_cells]``.
        operator_data: Static rectilinear geometry and indexing.

    Returns:
        Full nodal operator action with shape ``[num_nodes]``.
    """
    if operator_data.cell_unit_stiffness is not None:
        return matvec_full_compatible(
            x_full,
            conductivity,
            jnp.asarray(operator_data.cells),
            jnp.asarray(operator_data.cell_unit_stiffness, dtype=x_full.dtype),
            operator_data.num_nodes,
        )
    cells = jnp.asarray(operator_data.cells)
    cell_values = x_full[cells]
    cell_outputs = apply_element_operator(
        cell_values,
        conductivity,
        jnp.asarray(operator_data.cell_widths, dtype=x_full.dtype),
        jnp.asarray(operator_data.directional_matrices, dtype=x_full.dtype),
    )
    return (
        jnp.zeros((operator_data.num_nodes,), dtype=x_full.dtype)
        .at[cells.reshape(-1)]
        .add(cell_outputs.reshape(-1))
    )


def matvec_full_compatible(
    x_full, conductivity, cells, cell_unit_stiffness, num_nodes: int
):
    """Apply compatibility cell matrices supplied as dynamic JAX operands.

    Args:
        x_full: Full nodal field with shape ``[num_nodes]``.
        conductivity: Cell-wise conductivity with shape ``[num_cells]``.
        cells: Cell-to-node indices with shape ``[num_cells, num_local_nodes]``.
        cell_unit_stiffness: Geometry-only element matrices with shape
            ``[num_cells, num_local_nodes, num_local_nodes]``.
        num_nodes: Number of full nodal DOFs; static under JIT.

    Returns:
        Full nodal action with shape ``[num_nodes]`` and ``x_full.dtype``.

    Notes:
        Passing large arrays as operands avoids XLA embedding and constant-folding
        them into every compiled forward or adjoint executable.
    """
    cell_values = x_full[cells]
    cell_outputs = conductivity[:, None] * jnp.einsum(
        "eij,ej->ei", cell_unit_stiffness, cell_values, optimize=True
    )
    return (
        jnp.zeros((num_nodes,), dtype=x_full.dtype)
        .at[cells.reshape(-1)]
        .add(cell_outputs.reshape(-1))
    )


def matvec_free_full_grid_compatible(
    x_free,
    conductivity,
    cells,
    cell_unit_stiffness,
    free_rows,
    num_nodes: int,
):
    """Apply ``A_ff`` through a full field with dynamic JAX operands.

    Args:
        x_free: Free nodal values in reduced-system ordering.
        conductivity: Cell-wise conductivity with shape ``[num_cells]``.
        cells: Cell-to-full-node connectivity.
        cell_unit_stiffness: Geometry-only element stiffness matrices.
        free_rows: Full nodal indices of the reduced DOFs.
        num_nodes: Number of full nodal DOFs; static under JIT.

    Returns:
        Reduced action with the same shape and dtype as ``x_free``.
    """
    x_full = jnp.zeros((num_nodes,), dtype=x_free.dtype).at[free_rows].set(x_free)
    return matvec_full_compatible(
        x_full, conductivity, cells, cell_unit_stiffness, num_nodes
    )[free_rows]


def matvec_free_full_grid(x_free, conductivity, operator_data):
    """Apply ``A_ff`` by scattering through a structured full-grid field.

    Args:
        x_free: Free nodal values with shape ``[num_free_dofs]``.
        conductivity: Cell-wise conductivity with shape ``[num_cells]``.
        operator_data: Static geometry and reduced-DOF mappings.

    Returns:
        Reduced operator action with the same shape and dtype as ``x_free``.
    """
    free_rows = jnp.asarray(operator_data.free_rows)
    x_full = (
        jnp.zeros((operator_data.num_nodes,), dtype=x_free.dtype)
        .at[free_rows]
        .set(x_free)
    )
    return matvec_full(x_full, conductivity, operator_data)[free_rows]


def matvec_free_direct(x_free, conductivity, operator_data):
    """Apply ``A_ff`` directly with precomputed element-to-free mappings.

    Args:
        x_free: Free nodal values with shape ``[num_free_dofs]``.
        conductivity: Cell-wise conductivity with shape ``[num_cells]``.
        operator_data: Static geometry and direct free-DOF mappings.

    Returns:
        Reduced operator action with the same shape and dtype as ``x_free``.
    """
    cell_free_ids = jnp.asarray(operator_data.cell_free_ids)
    padded_values = jnp.concatenate((jnp.zeros((1,), dtype=x_free.dtype), x_free))
    cell_values = padded_values[cell_free_ids + 1]
    cell_outputs = apply_element_operator(
        cell_values,
        conductivity,
        jnp.asarray(operator_data.cell_widths, dtype=x_free.dtype),
        jnp.asarray(operator_data.directional_matrices, dtype=x_free.dtype),
        (
            None
            if operator_data.cell_unit_stiffness is None
            else jnp.asarray(operator_data.cell_unit_stiffness, dtype=x_free.dtype)
        ),
    ).reshape(-1)
    positions = jnp.asarray(operator_data.output_positions)
    destinations = jnp.asarray(operator_data.output_free_ids)
    return (
        jnp.zeros((x_free.size,), dtype=x_free.dtype)
        .at[destinations]
        .add(cell_outputs[positions])
    )


def matvec_free_direct_compatible(
    x_free,
    conductivity,
    cell_unit_stiffness,
    cell_free_ids,
    output_positions,
    output_free_ids,
):
    """Apply ``A_ff`` directly while passing all large arrays dynamically.

    Args:
        x_free: Free nodal values with shape ``[num_free_dofs]``.
        conductivity: Cell-wise conductivity with shape ``[num_cells]``.
        cell_unit_stiffness: Geometry-only element stiffness matrices.
        cell_free_ids: Element-to-free indices, using ``-1`` for constrained
            nodes.
        output_positions: Flattened element-local positions that contribute to
            a free output row.
        output_free_ids: Free-row destinations for ``output_positions``.

    Returns:
        Reduced action with the same shape and dtype as ``x_free``.
    """
    padded_values = jnp.concatenate((jnp.zeros((1,), dtype=x_free.dtype), x_free))
    cell_values = padded_values[cell_free_ids + 1]
    cell_outputs = conductivity[:, None] * jnp.einsum(
        "eij,ej->ei", cell_unit_stiffness, cell_values, optimize=True
    )
    return (
        jnp.zeros_like(x_free)
        .at[output_free_ids]
        .add(cell_outputs.reshape(-1)[output_positions])
    )


def diagonal_full(conductivity, operator_data, dtype=None):
    """Accumulate the exact diagonal of this matrix-free Q1 operator.

    Args:
        conductivity: Cell-wise conductivity with shape ``[num_cells]``.
        operator_data: Static rectilinear geometry and element operators.
        dtype: Optional arithmetic and output dtype.

    Returns:
        Exact full-DOF operator diagonal with shape ``[num_nodes]``.
    """
    dtype = conductivity.dtype if dtype is None else dtype
    widths = jnp.asarray(operator_data.cell_widths, dtype=dtype)
    matrices = jnp.asarray(operator_data.directional_matrices, dtype=dtype)
    if operator_data.cell_unit_stiffness is None:
        coefficients = _cell_directional_coefficients(widths)
        local_diagonal = conductivity[:, None] * jnp.einsum(
            "ea,aii->ei", coefficients, matrices, optimize=True
        )
    else:
        unit_stiffness = jnp.asarray(operator_data.cell_unit_stiffness, dtype=dtype)
        local_diagonal = conductivity[:, None] * jnp.diagonal(
            unit_stiffness, axis1=1, axis2=2
        )
    cells = jnp.asarray(operator_data.cells)
    return (
        jnp.zeros((operator_data.num_nodes,), dtype=dtype)
        .at[cells.reshape(-1)]
        .add(local_diagonal.reshape(-1))
    )


def diagonal_free(conductivity, operator_data, dtype=None):
    """Return ``diag(A_ff)`` in reduced-free-DOF ordering.

    Args:
        conductivity: Cell-wise conductivity with shape ``[num_cells]``.
        operator_data: Static rectilinear geometry and free-DOF mappings.
        dtype: Optional arithmetic and output dtype.

    Returns:
        Exact reduced diagonal with shape ``[num_free_dofs]``.
    """
    diagonal = diagonal_full(conductivity, operator_data, dtype=dtype)
    return diagonal[jnp.asarray(operator_data.free_rows)]


def select_line_axis(operator_data, line_axis="auto") -> int:
    """Select the rectilinear axis used by the block-line preconditioner.

    Args:
        operator_data: Static rectilinear geometry.
        line_axis: ``"auto"`` or an explicit ``"x"``, ``"y"``, or ``"z"``.

    Returns:
        Integer axis index.

    Notes:
        Automatic selection maximizes the integrated geometric diffusion scale
        ``sum(volume / h_axis**2)``. For scalar conductivity this identifies the
        direction with strongest mesh-induced coupling without hard-coding z.
    """
    normalized = str(line_axis).lower().strip()
    if normalized == "auto":
        widths = np.asarray(operator_data.cell_widths, dtype=np.float64)
        strengths = np.sum(np.prod(widths, axis=1)[:, None] / np.square(widths), axis=0)
        return int(np.argmax(strengths))
    if normalized not in "xyz" or len(normalized) != 1:
        raise ValueError("line_axis must be 'auto', 'x', 'y', or 'z'.")
    axis = "xyz".index(normalized)
    if axis >= operator_data.dim:
        raise ValueError(
            f"line_axis={line_axis!r} is unavailable for a {operator_data.dim}D mesh."
        )
    return axis


def line_offdiagonal_full(conductivity, operator_data, axis: int, dtype=None):
    """Accumulate same-line neighboring entries of the exact full operator.

    Args:
        conductivity: Cell-wise conductivity with shape ``[num_cells]``.
        operator_data: Static rectilinear operator data with compatibility cell
            matrices.
        axis: Selected line-axis index.
        dtype: Optional arithmetic dtype.

    Returns:
        Full nodal array. At each node except the last node of a line, the value
        is the matrix entry coupling it to its positive-axis neighbor.
    """
    if operator_data.cell_unit_stiffness is None:
        raise ValueError(
            "Line preconditioning requires geometry-compatible cell stiffness data."
        )
    dtype = conductivity.dtype if dtype is None else dtype
    offsets = _local_node_offsets(operator_data.dim)
    lower_local = []
    upper_local = []
    for local_index, offset in enumerate(offsets):
        if offset[axis] != 0:
            continue
        upper_offset = offset.copy()
        upper_offset[axis] = 1
        matches = np.nonzero(np.all(offsets == upper_offset[None, :], axis=1))[0]
        lower_local.append(local_index)
        upper_local.append(int(matches[0]))
    lower_local = np.asarray(lower_local, dtype=np.int64)
    upper_local = np.asarray(upper_local, dtype=np.int64)

    unit_stiffness = jnp.asarray(operator_data.cell_unit_stiffness, dtype=dtype)
    local_edges = conductivity[:, None] * unit_stiffness[:, lower_local, upper_local]
    destinations = jnp.asarray(operator_data.cells[:, lower_local]).reshape(-1)
    return (
        jnp.zeros((operator_data.num_nodes,), dtype=dtype)
        .at[destinations]
        .add(local_edges.reshape(-1))
    )


def apply_line_preconditioner(
    residual_free,
    diagonal_full_values,
    offdiagonal_full_values,
    operator_data,
    axis: int,
):
    """Apply independent exact tridiagonal line blocks to a reduced residual.

    Args:
        residual_free: Residual in free-DOF ordering.
        diagonal_full_values: Exact full operator diagonal.
        offdiagonal_full_values: Positive-axis neighboring entries returned by
            :func:`line_offdiagonal_full`.
        operator_data: Static geometry and free/constrained mappings.
        axis: Axis along which each tridiagonal block is formed.

    Returns:
        Preconditioned residual in free-DOF ordering.

    Notes:
        The blocks retain the principal same-line entries of ``A_ff`` and drop
        couplings between different lines. Because principal submatrices of an
        SPD matrix are SPD, this preconditioner is compatible with CG. Dirichlet
        nodes are represented by decoupled unit rows and are discarded afterward.
    """
    return apply_line_preconditioner_compatible(
        residual_free,
        diagonal_full_values,
        offdiagonal_full_values,
        jnp.asarray(operator_data.free_rows),
        jnp.asarray(operator_data.constrained_rows),
        operator_data.node_shape,
        axis,
        operator_data.num_nodes,
    )


def apply_line_preconditioner_compatible(
    residual_free,
    diagonal_full_values,
    offdiagonal_full_values,
    free_rows,
    constrained_rows,
    node_shape,
    axis: int,
    num_nodes: int,
):
    """Apply line blocks with index arrays supplied dynamically to JIT.

    Args:
        residual_free: Residual in reduced free-DOF ordering.
        diagonal_full_values: Exact full operator diagonal.
        offdiagonal_full_values: Positive-axis neighboring operator entries.
        free_rows: Full indices of free DOFs.
        constrained_rows: Full indices of prescribed DOFs.
        node_shape: Rectilinear full-node shape; static under JIT.
        axis: Line direction.
        num_nodes: Number of full nodal DOFs; static under JIT.

    Returns:
        Preconditioned residual in free-DOF ordering.
    """
    residual_full = (
        jnp.zeros((num_nodes,), dtype=residual_free.dtype)
        .at[free_rows]
        .set(residual_free)
    )
    free_mask = jnp.ones((num_nodes,), dtype=bool).at[constrained_rows].set(False)

    diagonal = jnp.where(free_mask, diagonal_full_values, 1.0)
    edge_mask = free_mask.reshape(node_shape)
    edge_mask = jnp.moveaxis(edge_mask, axis, -1)
    valid_edge = edge_mask[..., :-1] & edge_mask[..., 1:]

    diagonal_lines = jnp.moveaxis(diagonal.reshape(node_shape), axis, -1)
    edge_lines = jnp.moveaxis(offdiagonal_full_values.reshape(node_shape), axis, -1)
    edge_lines = jnp.where(valid_edge, edge_lines[..., :-1], 0.0)
    zeros = jnp.zeros_like(edge_lines[..., :1])
    lower = jnp.concatenate((zeros, edge_lines), axis=-1)
    upper = jnp.concatenate((edge_lines, zeros), axis=-1)
    rhs_lines = jnp.moveaxis(residual_full.reshape(node_shape), axis, -1)
    solution_lines = jax.lax.linalg.tridiagonal_solve(
        lower,
        diagonal_lines,
        upper,
        rhs_lines[..., None],
    )[..., 0]
    solution_full = jnp.moveaxis(solution_lines, -1, axis).reshape(-1)
    return solution_full[free_rows]


def source_load_full(source, operator_data, dtype=None):
    """Assemble the nodal load generated by cell-wise volumetric heat density.

    Args:
        source: Cell-wise volumetric heat density with shape ``[num_cells]``.
        operator_data: Static rectilinear operator data.
        dtype: Optional output dtype. Defaults to ``source.dtype``.

    Returns:
        Full nodal source load with shape ``[num_nodes]``.
    """
    dtype = source.dtype if dtype is None else dtype
    if operator_data.cell_unit_load is None:
        widths = jnp.asarray(operator_data.cell_widths, dtype=dtype)
        weight = jnp.prod(widths, axis=1) / (2.0**operator_data.dim)
        local_load = source[:, None] * weight[:, None]
        local_load = jnp.broadcast_to(
            local_load, (source.shape[0], 2**operator_data.dim)
        )
    else:
        unit_load = jnp.asarray(operator_data.cell_unit_load, dtype=dtype)
        local_load = source[:, None] * unit_load
    cells = jnp.asarray(operator_data.cells)
    return (
        jnp.zeros((operator_data.num_nodes,), dtype=dtype)
        .at[cells.reshape(-1)]
        .add(local_load.reshape(-1))
    )


def jax_fem_cell_data(problem):
    """Extract geometry-only cell operators from an initialized JAX-FEM problem.

    Args:
        problem: Scalar Q1 JAX-FEM heat problem whose geometric arrays have
            already been initialized.

    Returns:
        A pair ``(cell_unit_stiffness, cell_unit_load)`` with shapes
        ``[num_cells, 2**dim, 2**dim]`` and ``[num_cells, 2**dim]``.

    Notes:
        This calculation deliberately uses JAX-FEM's stored geometry. Those
        arrays are currently generated in float32, even for float64 solves.
        Reusing them makes the matrix-free action reproduce the existing sparse
        assembly rather than silently changing its geometric arithmetic.
    """
    fe = problem.fes[0]
    shape_grads = np.asarray(fe.shape_grads, dtype=np.float64)
    test_grads_jxw = np.asarray(fe.v_grads_JxW[..., 0, :], dtype=np.float64)
    cell_unit_stiffness = np.einsum(
        "eqjd,eqid->eij", shape_grads, test_grads_jxw, optimize=False
    )
    shape_values = np.asarray(fe.shape_vals, dtype=np.float64)
    jxw = np.asarray(fe.JxW, dtype=np.float64)
    cell_unit_load = np.einsum("qi,eq->ei", shape_values, jxw, optimize=False)
    return cell_unit_stiffness, cell_unit_load


def jax_fem_neumann_load(problem) -> np.ndarray:
    """Build the full nodal load for the problem's constant Neumann fluxes.

    Args:
        problem: Initialized scalar heat problem with JAX-FEM boundary
            quadrature data.

    Returns:
        NumPy array with shape ``[num_nodes]``. Its sign matches the right-hand
        side of ``A T = f`` for the convention ``n . (k grad T) = flux``.
    """
    load = np.zeros(problem.num_total_dofs_all_vars, dtype=np.float64)
    cells = np.asarray(problem.fes[0].cells, dtype=np.int64)
    for boundary_index, flux in enumerate(problem._neumann_values):
        boundary_cells = np.asarray(
            problem.boundary_inds_list[boundary_index][:, 0], dtype=np.int64
        )
        face_shape_values = np.asarray(
            problem.selected_face_shape_vals[boundary_index], dtype=np.float64
        )
        nanson_scale = np.asarray(
            problem.nanson_scale[boundary_index][:, 0, :], dtype=np.float64
        )
        local_load = float(flux) * np.sum(
            face_shape_values * nanson_scale[:, :, None], axis=1
        )
        np.add.at(load, cells[boundary_cells].reshape(-1), local_load.reshape(-1))
    return load
