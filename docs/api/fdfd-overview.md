# FDFD Overview

MAPS-FDFD is the frequency-domain electromagnetic subsystem used by the inverse-design stack.

Start with:

- [Scientific conventions](fdfd-scientific-conventions.md)
- [Derivative operators](fdfd-derivatives.md)
- [Solver contract](fdfd-solver-contract.md)
- [Rectilinear grids and near-to-far](fdfd-grid-and-near2far.md)
- [Sparse backends](fdfd-backends.md)

For a 2-D problem, the core numerical contract is:

```text
epsilon map : [Nx, Ny]
A           : [Nx*Ny, Nx*Ny] sparse complex matrix
b           : [Nx*Ny] complex source
x           : [Nx*Ny] complex field vector
```

The flattening convention and field ordering are scientific invariants shared by derivative matrices, material maps, sources, solvers, monitors, and inverse-design gradients.
