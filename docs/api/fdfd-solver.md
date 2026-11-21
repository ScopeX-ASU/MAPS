# FDFD Sparse Solver API

This document describes the solver-layer interface used by `core/fdfd/solver.py`. It is intentionally more detailed than a conventional API reference because the solver sits between scientific FDFD tensors and sparse linear-algebra backends.

## Algebraic problem

The solver layer operates on the flattened linear system

```text
A x = b
```

where `A` is an `N x N` sparse system matrix and `N = Nx * Ny` for a 2D FDFD grid. The high-level `core.fdfd.fdfd` class converts between the 2D field grid and this flattened representation.

The right-hand side used by the `fdfd_ez` workflow is formed from the electric-current source as

```text
b = i * omega * Jz
```

and the solved electric field is reshaped back to `[Nx, Ny]` before magnetic-field components are computed.

## `solve_linear`

`solve_linear(A, b, ...)` is the central dispatcher for sparse linear solves.

### Inputs

| Argument | Expected form | Meaning |
|---|---|---|
| `A` | SciPy-compatible sparse matrix, shape `[N, N]` | Discretized FDFD system matrix. |
| `b` | 1D array/tensor, shape `[N]` | Flattened complex right-hand side. |
| `solver_type` | `"direct"`, `"iterative"`, or `"iterative_nn"` | Selects the numerical path. |
| `neural_solver` | Optional model/callable | Used by `iterative_nn` to generate an initial guess. |
| `eps` | Tensor with the spatial FDFD shape | Required when the neural initialization path reshapes `b` for a neural model. |
| `iterative_method` | Name from the registered SciPy methods | Examples: `"bicg"`, `"bicgstab"`, `"cg"`, `"gmres"`, `"lgmres"`. |
| `symmetry` | `bool` | Selects the complex-symmetric versus complex-nonsymmetric direct-solver matrix type. |
| `clear` | `bool` | Controls whether the reusable direct-solver factorization is cleared. |
| `double` | `bool` | Selects double-precision behavior in the direct backend where supported. |
| `use_pydiso` | `bool` | Prefer the Pydiso wrapper when available. |
| `**kwargs` | Backend-specific keywords | Forwarded to the iterative SciPy solver. |

### Solver modes

#### `solver_type="direct"`

The preferred direct path uses the available MKL/Pydiso backend. When no MKL backend is available, the implementation falls back to `scipy.sparse.linalg.spsolve`.

The direct path has an intentionally backend-dependent return convention:

```text
MKL/Pydiso, clear=True   -> (x, None)
MKL/Pydiso, clear=False  -> (x, reusable_solver)
SciPy fallback           -> x
```

`x` is a flattened solution with shape `[N]`.

#### `solver_type="iterative"`

Uses a SciPy sparse iterative method. MAPS passes the module-level absolute tolerance `ATOL` to the SciPy solver.

Returns:

```text
x: shape [N]
```

#### `solver_type="iterative_nn"`

Uses a neural prediction as an initial guess `x0`, then runs the selected SciPy iterative method. This is not a pure neural solve: numerical convergence is still determined by the iterative backend.

The neural model is called using a batch dimension added to the permittivity and reshaped source tensors. The implementation expects a final model layout that can be permuted into a real/imaginary channel representation and converted with `torch.view_as_complex`.

## Neural initial-guess tensor contract

The `iterative_nn` path is more restrictive than the generic linear solver interface.

Given `eps` with spatial shape `[Nx, Ny]` and flattened `b` of shape `[N]`, the implementation reshapes the source to `[Nx, Ny]`, adds a batch dimension, and calls the neural solver with tensors equivalent to:

```text
eps_input     : [1, Nx, Ny]
source_input  : [1, Nx, Ny]
```

The returned prediction is then permuted from a layout with the real/imaginary channel on the final dimension, converted with `torch.view_as_complex`, squeezed over the batch dimension, and flattened to `[N]`.

The exact channel semantics therefore need to be documented by the specific neural solver implementation; this generic solver should not invent them.

## `coo_torch2cupy`

Converts a PyTorch sparse COO tensor wrapped by an object exposing `.data` into a CuPy COO matrix.

Expected input structure:

```text
A.data
  -> torch sparse COO tensor
  -> coalesced
  -> values + indices
```

The conversion is intended for CUDA interoperability and does not densify the matrix.

## `compare_As`

Compares two CSR-like sparse matrices in two stages:

1. shape and sparse structure (`indptr` and `indices`);
2. stored values, using an absolute tolerance.

Returns:

```text
(structure_same, values_same)
```

Both values are `False` when the matrix shapes or sparse structure differ.

Use `atol=0.0` for exact value comparison.

## Direct-solver matrix semantics

When the `symmetry` flag is enabled, the direct backend uses the complex-symmetric matrix type. Otherwise it uses the complex-nonsymmetric type. This flag is therefore a numerical-property assertion, not merely a performance hint; callers should only enable it when the constructed system satisfies the required symmetry.

## Caching and factorization reuse

The FDFD layer can retain a factorized solver between calls. This is why `solve_linear` does not normalize all backend return values to one universal structure. Code that wants factorization reuse must handle the `(x, solver)` convention from the MKL/Pydiso direct path.

The higher-level `fdfd_ez` object exposes cache-control methods and stores the corresponding solver instance; see `docs/api/fdfd.md`.

## Usage example

A minimal CPU iterative example is:

```python
import numpy as np
import scipy.sparse as sp

A = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 4.0]]))
b = np.array([2.0, 8.0])

x = solve_linear(
    A,
    b,
    solver_type="iterative",
    iterative_method="cg",
)

# x.shape == (2,)
# x ~= [1.0, 2.0]
```

This example demonstrates only the algebraic solver interface. It does not represent a complete Maxwell/FDFD setup.

## Relationship to the FDFD field solver

The field-level API in `core/fdfd/fdfd.py` owns the physical grid, permittivity distribution, PML configuration, Bloch phases, and field reshaping. It supplies the sparse system and source vector to the solver layer. The solver layer should therefore be understood as an algebraic backend, not as a standalone electromagnetic solver.
