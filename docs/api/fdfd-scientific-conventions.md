# FDFD Scientific API and Conventions

MAPS-FDFD is the frequency-domain electromagnetic path used by the inverse-design stack. The implementation contains both sparse finite-difference operators and higher-level solver wrappers for 1-D/2-D problems, including PML, Bloch phase support, nonuniform/rectilinear grids, and optional sparse direct solvers.

## Field representation

The core 2-D FDFD formulations operate on a scalar polarization field plus the corresponding in-plane magnetic/electric components. The exact field dictionary depends on the TE/TM formulation, but common keys include:

```text
Ez / Hx / Hy     TM-like scalar formulation
Hz / Ex / Ey     TE-like scalar formulation
```

Spatial tensors conventionally use the physical simulation-grid ordering:

```text
[Nx, Ny]
```

Flattening into sparse linear systems produces:

```text
N = Nx * Ny
field_vector.shape == [N]
A.shape == [N, N]
source.shape == [N]
```

The ordering of the flattening operation is part of the solver contract and must remain consistent between derivative matrices, material maps, source vectors, and reconstructed fields.

## Units and wavelength

FDFD quantities use SI-compatible physical units internally. In particular:

```text
wavelength       : meters
frequency        : Hz
angular frequency: rad/s
cell spacing dL  : meters
permittivity     : relative permittivity where the API specifies eps_r
```

The conversion between wavelength and angular frequency is:

$$
\omega = \frac{2\pi c}{\lambda}.
$$

## Sparse derivative operators

`core.fdfd.derivatives` constructs forward/backward sparse derivative matrices for x/y directions.

The public construction flow is:

```text
compute_derivative_matrices
    -> createDws
        -> make_Dxf / make_Dxb
        -> make_Dyf / make_Dyb
    -> create_S_matrices
        -> PML scaling factors
```

For a grid with shape `[Nx, Ny]`, each derivative matrix has shape:

```text
[Nx * Ny, Nx * Ny]
```

and is complex-valued when Bloch phase or PML scaling is present.

The forward and backward operators use opposite one-cell difference conventions. Boundary wrap terms can include Bloch phase factors:

$$
E(x + L_x) = e^{i k_x L_x} E(x), \qquad
E(y + L_y) = e^{i k_y L_y} E(y).
$$

The API names `bloch_x` and `bloch_y` refer to the phase argument used in the sparse boundary entries; callers must not confuse these with a physical wavevector unless the surrounding code establishes that normalization.

## PML

Perfectly matched layer behavior is incorporated by diagonal scaling factors applied to derivative operators.

The derivative construction therefore conceptually becomes:

$$
D_x \rightarrow S_x D_x, \qquad D_y \rightarrow S_y D_y.
$$

The current implementation materializes flattened scaling vectors and multiplies them into sparse derivative matrices before returning coalesced COO tensors.

The PML width input is expressed as the number of grid cells per axis:

```python
npml = [Nx_pml, Ny_pml]
```

not a physical distance in meters.

## Symmetrization and preconditioning

`core.fdfd.preconditioner.create_symmetrizer` constructs diagonal matrices from the effective primal/dual cell widths:

$$
P_l = \operatorname{diag}(\sqrt{w_x w_y}),
\qquad
P_r = \operatorname{diag}\left(\frac{1}{\sqrt{w_x w_y}}\right).
$$

The resulting transformed system is:

$$
A_{sym} = P_l A P_r.
$$

This is useful when PML and/or nonuniform-grid scaling makes the raw operator nonsymmetric but a diagonally equivalent symmetric representation is desirable.

The effective widths are formulation-dependent. A docstring must state whether supplied widths are primal-grid or dual-grid widths; silently substituting uniform widths changes the mathematical system.

## Solvers

`core.fdfd.solver` contains the higher-level sparse-system solve path. Large solver implementations should be documented at the API level rather than copied into duplicate Markdown.

The principal solver contract is:

```text
epsilon/material map
    + grid metadata
    + source
    + wavelength/frequency
    + boundary/PML configuration
        ↓
 sparse Maxwell operator A
        ↓
 solve A x = b
        ↓
 reconstructed complex field(s)
```

Available direct/accelerated solver integrations include the MAPS sparse path and optional external backends such as Pardiso/PyDISO/cuDSS where installed.

Optional solver backends should not change the physical meaning, ordering, or normalization of the returned fields.

## Autograd and inverse design

The Torch FDFD path is used inside MAPS inverse design and therefore has an important contract beyond forward correctness:

```text
permittivity/design tensor
        ↓
 differentiable FDFD objective
        ↓
 scalar objective
        ↓
 torch.autograd.backward()
        ↓
 gradient w.r.t. design parameters
```

When adding or changing a solver path, verify both:

1. forward field/objective agreement with the reference solver;
2. gradient agreement using a finite-difference or complex-step-compatible validation where mathematically appropriate.

A function that returns numerically correct fields but detaches the material tensor from the autograd graph is not a valid inverse-design backend.

## Near-to-far transformation

`core.fdfd.near2far` converts near-field data on monitor slices into far-field quantities using Green-function-based propagation.

Its documentation must distinguish:

```text
near-field tensor
monitor geometry / coordinates
polarization/component selection
observation coordinates
frequency / wavelength
background material
```

from the returned far-field tensor. Monitor-slice coordinate metadata is as important as the raw field array because the Green-function integral depends on physical positions and cell spacing.

## Tensor/docstring standard for FDFD

For every public FDFD callable, document at minimum:

- input tensor shapes and axis meaning;
- sparse matrix shape and layout when applicable;
- dtype, including complex dtype expectations;
- physical units;
- grid ordering and flattening convention;
- PML/Bloch behavior;
- whether tensors remain connected to autograd;
- return container/schema and field keys;
- one representative usage example;
- mathematical equations where the operation implements a scientific operator.

Avoid vague descriptions such as "compute derivative". The useful contract is closer to:

```text
input:  x_widths [Nx], y_widths [Ny], complex128
output: Pl, Pr sparse matrices [Nx*Ny, Nx*Ny]
meaning: diagonal similarity transform for the discrete Maxwell operator
```

## Large-file policy

`fdfd.py`, `solver.py`, and `near2far.py` are substantial scientific implementations. As with the InvDes monoliths, they should receive source-level docstrings only through an exact editing workflow that preserves the implementation byte-for-byte apart from inserted documentation. Their cross-function scientific contracts belong in this reference documentation meanwhile.
