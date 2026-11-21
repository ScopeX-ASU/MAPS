# MAPS-FDFD

MAPS-FDFD is the frequency-domain electromagnetic subsystem used by the inverse-design framework.

The FDFD documentation is organized around four contracts:

```text
field/grid representation
        |
derivative + PML operators
        |
sparse Maxwell system
        |
solver / autograd bridge
        |
objectives and near-to-far observables
```

## Start here

- [Scientific conventions](fdfd-scientific-conventions.md)
- [Derivative and PML operators](fdfd-derivatives.md)
- [Solver contract](fdfd-solver-contract.md)
- [Rectilinear grids and near-to-far](fdfd-grid-and-near2far.md)
- [Sparse solver backends](fdfd-backends.md)

## Representative problem contract

For a 2-D grid:

```text
Nx, Ny                         physical grid dimensions
A                              [Nx*Ny, Nx*Ny] sparse complex operator
b                              [Nx*Ny] complex source
x                              [Nx*Ny] complex field solution
eps_r                          [Nx, Ny] material map
```

The flattening order connecting `[Nx, Ny]` tensors to `[Nx*Ny]` vectors is a scientific API invariant.

## Inverse-design contract

The FDFD backend is not complete merely when it returns a numerically plausible field. It must also preserve the gradient path from material/design parameters to the scalar objective.

```text
rho / device parameters
        -> epsilon map
        -> sparse Maxwell operator
        -> complex field
        -> objective
        -> gradient w.r.t. epsilon/design parameters
```

When documenting new FDFD features, explicitly state tensor shapes, complex dtype, physical units, coordinate ordering, boundary/PML assumptions, and differentiability.
