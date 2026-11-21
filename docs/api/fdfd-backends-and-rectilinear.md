# FDFD Sparse Backends and Rectilinear-Grid Contracts

## Rectilinear grids

`core.fdfd.derivatives_rectilinear` extends the uniform-grid derivative construction to cell-centered 2-D rectilinear grids.

The principal inputs are:

```text
shape == (Nx, Ny)
dxs.shape == (Nx,)
dys.shape == (Ny,)
```

`d_s` are physical cell widths in the same length unit used by the surrounding simulation metadata. They must be finite and strictly positive.

### Boundary metadata

`spacing_from_boundaries(boundaries)` converts coordinate boundaries into cell widths:

$$
\Delta x_i = x_{i+1}-x_i,\qquad
\Delta y_j = y_{j+1}-y_j.
$$

The first two boundary arrays must therefore have lengths `Nx + 1` and `Ny + 1`. Additional axes are ignored by this 2-D module.

### Derivative matrices

The returned sparse operators have shape:

```text
[Nx * Ny, Nx * Ny]
```

with complex128 values so Bloch phases and PML factors can be represented without changing the sparse matrix dtype.

The operator uses the same `(Nx, Ny)` flattening convention as the uniform-grid FDFD path. Material maps, sources, fields, and derivative matrices must use the same ordering.

Unlike the uniform-grid implementation, the rectilinear path divides each finite difference by the local cell width. This distinction is important when comparing results between `derivatives.py` and `derivatives_rectilinear.py`.

### Rectilinear PML

`create_S_matrices` returns flattened inverse PML factors for forward/backward x/y derivatives. The returned four vectors have length `Nx * Ny` and align with the sparse derivative row ordering.

`npml` remains a cell count per axis, not a physical distance. The physical PML thickness is derived from the local cell widths near the two boundaries.

## Pardiso backends

MAPS has two related Pardiso wrappers:

```text
core.fdfd.pardiso_solver.pardisoSolver
core.fdfd.pydiso_solver.MKLPardisoSolver
```

Both wrap Intel MKL Pardiso, but their APIs and ownership models differ.

### `pardisoSolver`

The older wrapper accepts a SciPy sparse matrix, converts it to CSR, configures an MKL matrix type, and exposes an explicit three-stage usage pattern:

```python
solver = pardisoSolver(A, mtype=13, double=True)
solver.factor()
x = solver.solve(b)
solver.clear()
```

Typical FDFD complex-system matrix types are:

```text
6  -> complex symmetric
13 -> complex nonsymmetric
```

The RHS can be either a vector or a 2-D matrix of multiple right-hand sides. Multiple RHS solutions are returned in matrix form using Fortran-style ordering.

### `MKLPardisoSolver`

The newer PyDISO wrapper validates that `A` is a square sparse matrix and that its values are single/double real or complex floating point. The `matrix_type` may be the integer code or a descriptive string such as:

```python
"complex_symmetric"
"complex_nonsymmetric"
"complex_hermitian_positive_definite"
```

The solver performs symbolic analysis and optional factorization during construction. `refactor(A)` reuses the symbolic structure and therefore requires the new matrix to have the same dimensions and nonzero pattern.

`solve(b)` accepts a 1-D or 2-D NumPy array. For multiple RHS, the array has shape `[N, nrhs]` and is processed in Fortran order. `transpose=True` solves the transposed linear system.

### Backend invariance

Pardiso, SciPy, cuDSS, CuPy-assisted, and other sparse backends are implementation choices. They must preserve the same physical linear system:

$$
A x = b.
$$

Backend substitution must not silently change field ordering, complex dtype semantics, PML/Bloch terms, normalization, or the meaning of multiple right-hand sides.

## Validation checklist

When changing a sparse backend, validate:

1. `A.shape == (N, N)` and `b.shape[0] == N`;
2. residual `||Ax-b||` under the intended tolerance;
3. agreement with a trusted reference backend;
4. agreement for complex symmetric and complex nonsymmetric systems where supported;
5. repeated-solve behavior after factorization;
6. multiple-RHS behavior and ordering;
7. compatibility with the Torch autograd bridge when the backend is used in inverse design.

## Source-docstring policy

The backend implementations already contain substantial NumPy-style docstrings and large low-level ctypes/PyDISO blocks. They are intentionally not mechanically rewritten through the remote connector in this pass. New MAPS-facing wrapper APIs should use Google-style docstrings with explicit matrix/RHS shapes, dtype, factorization lifecycle, units where relevant, and executable examples.
