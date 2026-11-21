# FDFD Solver Contract

`core.fdfd.solver` is the sparse linear-system backend used by the FDFD electromagnetic formulations. It separates solver selection from Maxwell-operator construction and also contains the Torch autograd bridge used by inverse design.

## Linear-system contract

The FDFD solve reduces the discretized Maxwell equation to:

$$
A x = b,
$$

where:

```text
A : sparse complex matrix, shape [N, N]
x : complex field vector, shape [N]
b : complex source vector, shape [N]
N = Nx * Ny
```

The vector ordering must match the flattening convention used when constructing derivative matrices and material maps.

## `solve_linear`

`solve_linear` is the dispatch entry point. The supported `solver_type` values in the current implementation are:

```text
direct
iterative
iterative_nn
```

### Direct

The direct path prefers MKL/Pardiso when available and otherwise falls back to SciPy sparse solving. The `symmetry` flag selects the complex-symmetric Pardiso matrix type when the assembled system satisfies the corresponding symmetry contract.

Useful options include:

```text
clear       whether an external factorization object is released
 double      precision control for the MKL path
use_pydiso  whether the PyDISO backend may be selected
symmetry    complex-symmetric vs complex-nonsymmetric system
```

The direct result can be either a solution vector or, when `clear=False` on the MKL path, a solution plus a retained solver object for reuse.

### Iterative

The default iterative method is `bicg`. The implementation also dispatches to SciPy sparse iterative methods including:

```text
bicg, bicgstab, cg, cgs, gmres, lgmres, qmr, gcrotmk
```

The global absolute tolerance is currently `1e-8`. The returned vector has the same linear-system ordering as the input source.

### Neural-network initial guess

`iterative_nn` obtains an initial guess from a supplied `neural_solver` before invoking the selected SciPy iterative method.

The neural model receives the permittivity tensor with an added batch dimension and the source reshaped back to the spatial FDFD grid. Its predicted complex field is then flattened into the linear-system ordering and detached before being supplied to the iterative solver.

This path changes only the initial guess; it does not replace the final iterative linear-system solve.

## Sparse matrix conversion

`coo_torch2cupy` converts a coalesced Torch COO sparse tensor to a CuPy COO sparse matrix for GPU sparse operations. The input matrix must be coalesced before conversion.

## Autograd bridge

`SparseSolveTorchFunction` provides the custom autograd boundary between the sparse linear solver and PyTorch optimization.

The important distinction is:

```text
forward solve
    A(eps) x = b
        ↓
    complex field x
        ↓
custom backward
    gradient w.r.t. eps/design parameters
```

The forward solver may use non-PyTorch sparse backends, but the backward path must return a valid gradient to the differentiable permittivity/material input.

When documenting or modifying this layer, distinguish:

- sparse solver state (`A`, factorizations, preconditioners);
- differentiable material input (`eps_matrix`/equivalent);
- source vector `b`;
- field vector `x`;
- monitor/source metadata carried through the autograd context.

A backend that computes a correct forward field but breaks this dependency is not a valid inverse-design backend.

## Optional solver backends

The repository can integrate several sparse solver implementations:

```text
SciPy sparse
MKL / Pardiso
PyDISO
cuDSS
CuPy-assisted paths
```

These are optional acceleration layers. Scientific behavior should be backend-invariant up to normal floating-point solver tolerance.

## Validation expectations

For any solver backend or dispatch change, validate:

1. matrix dimensions and field ordering;
2. residual `||A x - b||` at the intended numerical tolerance;
3. agreement with a trusted reference solve on representative devices;
4. gradient agreement when the path is used by inverse design;
5. behavior with complex-valued operators, PML, and Bloch phase when applicable.
