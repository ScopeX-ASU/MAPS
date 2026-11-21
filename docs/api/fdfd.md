# FDFD Field Solver API

The 2D FDFD field-level implementation is centered on `core/fdfd/fdfd.py`. The public solver works on a 2D spatial grid and delegates the flattened sparse linear system to the solver layer documented in `docs/api/fdfd-solver.md`.

## Grid contract

For a simulation grid with shape

```text
shape = [Nx, Ny]
```

the implementation stores:

```text
Nx = number of grid cells along x
Ny = number of grid cells along y
N  = Nx * Ny
```

A 2D field represented on the simulation grid therefore has shape `[Nx, Ny]`. The algebraic solver uses flattened vectors with shape `[N]` and reshapes the solved fields back to `[Nx, Ny]`.

The code also supports field tensors with additional leading dimensions in the `Ez -> Hx/Hy` conversion routines. The spatial interpretation remains the final/flattened FDFD degrees of freedom, so callers should state batch semantics explicitly when using these lower-level methods.

## `fdfd_ez`

`fdfd_ez` is the primary electric-field (`Ez`) polarization implementation exposed by `core/fdfd.__init__`.

### Constructor inputs

| Argument | Expected form | Meaning |
|---|---|---|
| `omega` | scalar float | Angular frequency in rad/s. |
| `dL` | scalar float | Uniform grid-cell size in meters when using the regular Cartesian grid path. |
| `eps_r` | NumPy array or PyTorch tensor, shape `[Nx, Ny]` | Relative permittivity distribution. Dimensionless. The `fdfd_ez` constructor converts NumPy input to a PyTorch tensor. |
| `npml` | sequence of two integers | Number of PML cells along x and y. Interpreted as `[Nx_pml, Ny_pml]`. |
| `power` | scalar float | Target power used by the adjoint normalization workflow. Default is `1e-8`. |
| `bloch_phases` | optional sequence | Bloch phase parameters used by the underlying derivative construction. The base implementation stores x/y/z entries with unspecified missing entries defaulting to zero. |
| `neural_solver` | optional object/configuration | Neural forward/adjoint solver used when the selected numerical backend supports it. |
| `numerical_solver` | string | Numerical backend selector. The release implementation uses names such as `"solve_direct"`, `"solve_iterative"`, and `"none"` in the lower solver/autograd path. |
| `use_autodiff` | bool | Selects the custom-solver autodiff behavior. |
| `sym_precond` | bool | When enabled, constructs the symmetric preconditioner used by the direct solver path. |
| `dxs`, `dys` | optional array-like | Nonuniform rectilinear cell widths along x/y. These are used by the rectilinear derivative and flux-weighting paths. |

### Example initialization

```python
import numpy as np
import torch

Nx, Ny = 128, 96
eps_r = np.ones((Nx, Ny), dtype=np.float32)

solver = fdfd_ez(
    omega=2 * np.pi * 2.3e14,
    dL=20e-9,
    eps_r=eps_r,
    npml=[10, 10],
    numerical_solver="solve_direct",
)
```

The numerical values above are only a representative construction pattern. Physical parameters must match the intended wavelength/material system.

## `eps_r` property

`solver.eps_r` is the relative-permittivity grid used to assemble the FDFD operator.

Expected shape:

```text
[Nx, Ny]
```

Expected semantics:

- dimensionless relative permittivity;
- spatial ordering follows the solver's x/y grid convention;
- changing the array shape also updates `Nx`, `Ny`, and `N` through the base property setter.

The implementation accepts NumPy input at construction but internally uses the configured tensor representation for the differentiable PyTorch path.

## `solve`

The field-level `solve(source_z, slice_name=None, mode=None, temp=None)` method converts a spatial source into a complex electromagnetic field solution.

### Input contract

`source_z` represents the 2D electric-current/source distribution corresponding to the `Ez` polarization.

Expected shape:

```text
[Nx, Ny]
```

The source is flattened internally to `[N]` before the sparse solve. It is multiplied by `i * omega` to form the linear-system right-hand side.

The optional `slice_name`, `mode`, and `temp` values are not arbitrary metadata: they participate in the solver/adjoint bookkeeping and gradient dictionaries. In particular, the release implementation uses the triple `("adj", "adj", "adj")` for a dedicated adjoint solve path.

### Return contract

The method returns a tuple corresponding to

```text
(Hx, Hy, Ez)
```

with each component represented on the simulation grid.

For the standard 2D path, each returned field has spatial shape:

```text
[Nx, Ny]
```

The fields are complex-valued electromagnetic quantities. The implementation computes `Hx` and `Hy` from the solved `Ez` using sparse derivative operators and then applies a Yee-to-colocated interpolation step before returning the final fields.

Conceptually:

```text
source_z [Nx, Ny]
    |
    v
flatten -> [N]
    |
    v
assemble A and b = i * omega * source
    |
    v
solve A Ez = b
    |
    +--> derive Hx
    |
    +--> derive Hy
    |
    v
reshape/interpolate
    |
    v
(Hx, Hy, Ez), each [Nx, Ny]
```

### Example usage

```python
source = torch.zeros((Nx, Ny), dtype=torch.complex64)
source[Nx // 2, Ny // 4] = 1.0 + 0.0j

Hx, Hy, Ez = solver.solve(source)

assert Hx.shape == (Nx, Ny)
assert Hy.shape == (Nx, Ny)
assert Ez.shape == (Nx, Ny)
```

The exact dtype and device should match the solver's tensor/backend configuration. Do not assume that a NumPy complex array and a CUDA PyTorch tensor are interchangeable across every backend.

## Field-component conversion

The base FDFD implementation documents the core Maxwell conversions as sparse derivative operations. For the `Ez` polarization, the implementation uses equations equivalent to

```text
Hx = -1 / (i * omega * mu_0) * D_y^- Ez
Hy =  1 / (i * omega * mu_0) * D_x^- Ez
```

where `D_x^-` and `D_y^-` are the stored backward derivative operators used by the implementation. The exact Yee-grid placement and subsequent colocating interpolation are part of the solver convention and should not be inferred from a generic finite-difference formula alone.

The base class also contains the inverse conversions between magnetic and electric components for complementary polarizations. These should be documented together with the polarization-specific implementation rather than treated as generic tensor transforms.

## Adjoint and gradient workflow

The `fdfd_ez` implementation stores adjoint-source and gradient information in the sparse-solver object. `read_gradients()` returns a dictionary keyed by

```text
(slice_name, mode, temp)
```

Each dictionary value is reshaped back to the simulation grid and therefore has shape `[Nx, Ny]`.

The key identifies the simulation context associated with the stored derivative. This means downstream optimization code must preserve the same key convention when requesting or accumulating gradients.

## `norm_adj_power`

`norm_adj_power()` normalizes stored adjoint fields using integrated boundary fluxes and the configured target `power`.

It returns four dictionaries:

```text
(ez_adj_dict, hx_adj_dict, hy_adj_dict, normalization_factor)
```

Each field dictionary is keyed by the solver's `(slice_name, mode, temp)` tuple. The field values are reshaped to `[Nx, Ny]` before being returned. The normalization-factor dictionary stores the scalar scaling factor used for the corresponding key.

The implementation evaluates flux on four boundary monitor slices and uses the cell-width information when the rectilinear-grid path is active. The current implementation should be treated as the authoritative definition of the normalization convention.

## Solver cache control

The release implementation exposes:

```python
solver.clear_solver_cache()
solver.set_cache_mode(True)
solver.set_cache_mode(False)
```

Cache state affects reuse of factorized sparse systems. This is especially important when running repeated solves for the same matrix structure but changing source vectors or material values.

The cache should be considered an implementation resource: callers should not assume that a cached factorization survives a shape change or an incompatible matrix-structure change.

## Numerical conventions that should accompany FDFD documentation

The following should always be stated or linked when documenting a new FDFD API:

- whether the frequency input is angular frequency `omega` or ordinary frequency `f`;
- length units for `dL`, `dxs`, and `dys`;
- whether `eps_r` is relative or absolute permittivity;
- which grid location stores each field component;
- x/y axis ordering and array indexing semantics;
- PML thickness convention;
- Bloch-phase convention;
- complex dtype and device requirements;
- whether a routine preserves autograd connectivity;
- whether output fields are Yee-staggered or colocated.

These details are more important than generic prose because two tensors with identical shapes can represent physically different quantities.

## Relationship to the algebraic solver

See [FDFD Sparse Solver API](fdfd-solver.md) for the `[N, N]` sparse matrix / `[N]` vector interface. The field solver should document the mapping from physical grids to that algebraic representation; the sparse solver should document only the algebraic contract.
