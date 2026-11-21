# Structured matrix-free JAX heat solver

The opt-in `jax_matrix_free_solver` backend applies the existing rectilinear Q1
finite-element heat operator without constructing a global SciPy matrix or JAX
`BCOO` matrix during forward or adjoint CG solves. Existing `jax_solver`,
PARDISO, SciPy, PETSc, and AMJAX paths are unchanged.

This backend remains experimental and is not the default. Matrix-free Jacobi is
primarily a baseline: measured warm performance can be similar to BCOO because
both implementations are memory-bandwidth and scatter limited. Line
preconditioning is the intended optimization for strongly anisotropic grids.

## Discretization

The steady-state problem is

```text
-div(k grad(T)) = q
```

with prescribed temperatures on Dirichlet faces and the existing Neumann
convention `n . (k grad(T)) = flux`. Conductivity and volumetric source density
are piecewise constant per QUAD4/HEX8 element and are broadcast to the element's
quadrature points.

For an axis-aligned Q1 element, the stiffness is the sum of tensor products of a
one-dimensional stiffness matrix in one direction and one-dimensional mass
matrices in the transverse directions. The implementation retains the resulting
9-point 2D or 27-point 3D coupling; it does not substitute a finite-difference or
finite-volume stencil.

JAX-FEM currently creates geometric factors in float32 even for float64 solves.
The matrix-free backend therefore derives geometry-only element matrices from
those same factors. This preserves the action of the existing assembled FEM
operator, including its current geometric floating-point behavior. These static
matrices and reduced-DOF index mappings are reused by forward and adjoint solves;
quantities containing the changing `k_map` are recomputed.

The compatibility matrices require `num_local_nodes**2` geometry values per
cell (64 for HEX8). This is less indexing metadata than the assembled global
BCOO path but is not a zero-storage stencil. It is the deliberate cost of exact
agreement with the current JAX-FEM geometry arithmetic.

The backend accepts canonical axis-aligned QUAD4 and HEX8 Cartesian-product
meshes, including nonuniform rectilinear axes. It rejects warped or unstructured
fixed meshes instead of silently changing their discretization. It currently
requires `dirichlet_mode: reduced_free_dof` and `method: cg`. As with the
existing CG path, SPD behavior assumes positive conductivity and enough
Dirichlet constraints to remove the constant-temperature nullspace.

## Preconditioners

`jacobi`
: Uses the exact diagonal of `A_ff`, accumulated directly from element
  contributions without global sparse assembly.

`line`
: Retains the diagonal and same-line neighboring entries of `A_ff`, forming one
  independent tridiagonal principal block per grid line. Batched JAX
  tridiagonal solves apply the blocks. Since these are principal blocks of the
  SPD reduced operator, the preconditioner is SPD-compatible with CG.

  `line_axis: auto` selects the direction maximizing the integrated geometric
  diffusion scale `sum(cell_volume / cell_width_axis**2)`. This targets strong
  mesh anisotropy such as `dz << dx, dy`. Explicit `x`, `y`, and `z` selections
  are also supported.

The direct element-to-free-DOF path is the default because GPU profiling showed
it faster than full-grid scatter/apply/gather. Set
`free_dof_strategy: full_grid` to select the alternate implementation for
diagnostics.

## Configuration

Matrix-free Jacobi:

```json
{
  "dirichlet_mode": "reduced_free_dof",
  "jax_matrix_free_solver": {
    "method": "cg",
    "precond": "jacobi",
    "solve_dtype": "float64",
    "check_residual": true,
    "residual_rtol": 1e-6,
    "residual_atol": 1e-8,
    "maxiter": 1000
  }
}
```

Matrix-free line preconditioning:

```json
{
  "dirichlet_mode": "reduced_free_dof",
  "jax_matrix_free_solver": {
    "method": "cg",
    "precond": "line",
    "line_axis": "auto",
    "solve_dtype": "float64",
    "check_residual": true,
    "residual_rtol": 1e-6,
    "residual_atol": 1e-8,
    "maxiter": 1000
  }
}
```

## Benchmark

The standalone benchmark separates backend setup, first-call/JIT time, and
synchronized warm solve time. Its custom PCG loop is used only to report
iteration counts:

```bash
CUDA_VISIBLE_DEVICES=3 JAX_ENABLE_X64=1 \
  python \
  unitest/heat/benchmark_heat_matrix_free.py
```

Timing assertions are intentionally excluded from unit tests.

## Grid conversion and physical padding

Thermal material and source rasters may be generated on a nonuniform Tidy3D
grid even when the requested FEM mesh is uniform. Before a rectangular solve,
the device integration converts these cell-average fields using exact Cartesian
cell-overlap volumes:

```text
q_target[j] = sum_i(q_source[i] * overlap_volume[i, j]) / target_volume[j]
```

Consequently, `sum(q * cell_volume)` and therefore total generated heat are
independent of the chosen uniform or nonuniform solver grid. Conductivity is
converted by applying the same physical overlap weights to resistivity and then
inverting, preserving the existing resistance-aware coarsening convention.
Solved temperatures are interpolated back using the recorded physical cell
centers rather than array-index coordinates.

For nonuniform fixed meshes, distance-based graded padding now clips the final
padding cell so each requested physical padding distance is exact. Cell-count
padding retains its original behavior. A uniform mesh represents physical
padding to within its cell spacing when the requested distance is not an integer
multiple of that spacing.

Note that the TDM heat configuration's historical `resolution` field contains
cell spacing in microns. For example, 40 cells per micron corresponds to
`"resolution": [0.025, 0.025, 0.025]`; the top-level optical `resolution` does
not override the TDM thermal spacing.
