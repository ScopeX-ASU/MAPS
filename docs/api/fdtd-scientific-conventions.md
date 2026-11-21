# FDTD-X Scientific API and Conventions

MAPS FDTD-X is the 3-D time-domain electromagnetic path backed by `fdtdx`. The MAPS-facing layer provides a PyTorch/autodiff wrapper around a JAX/FDTD-X runtime and converts between native Yee-grid data and colocated Torch tensors.

## Simulation dimensionality and grid

The FDTD-X path is intrinsically three-dimensional. The physical simulation volume is described by:

```text
Nx, Ny, Nz                 number of Yee cells
x_edges.shape == Nx + 1
y_edges.shape == Ny + 1
z_edges.shape == Nz + 1
```

All boundary coordinates are physical positions in meters. Cell widths are:

$$
\\Delta x_i = x_{i+1}-x_i,\\qquad
\\Delta y_j = y_{j+1}-y_j,\\qquad
\\Delta z_k = z_{k+1}-z_k.
$$

The MAPS helper `build_rectilinear_grid_metadata` and related utilities preserve these physical coordinate arrays when translating between MAPS and FDTD-X representations.

## Yee-grid fields

Native FDTD-X electric and magnetic components live on a staggered Yee grid. MAPS commonly exposes colocated tensors with component-major organization:

```text
E: [batch_or_source, 3, Nx, Ny, Nz, frequency]
H: [batch_or_source, 3, Nx, Ny, Nz, frequency]
```

Exact leading/trailing axes depend on the detector and wrapper path. A docstring must state them explicitly rather than assuming every field tensor has the same rank.

Component ordering is:

```text
E: Ex, Ey, Ez
H: Hx, Hy, Hz
```

The solver also uses a combined six-component convention:

```text
(Ex, Ey, Ez, Hx, Hy, Hz)
```

The native FDTD-X detector may retain Yee staggering or exact interpolation behavior. MAPS colocation/interpolation helpers convert that representation for Torch-facing objectives.

## Permittivity and material map

`eps_r` is the relative-permittivity/material input to the FDTD-X solve. It is a 3-D simulation-grid quantity. When design data originate at a lower-resolution design grid, explicit resampling onto the FDTD-X grid is required.

Do not document `eps_r` merely as a generic tensor: specify whether it is:

```text
relative permittivity epsilon_r
conductivity sigma
normalized design density rho
or a derived material map
```

These are physically different quantities even when their tensors have the same shape.

## Frequency and phasors

The FDTD-X wrapper supports frequency-domain phasor extraction from time-domain simulation data. A frequency axis is therefore distinct from the time axis used internally by the FDTD solver.

A MAPS-facing combined phasor tensor uses:

```text
[Nfreq, 6, Nx, Ny, Nz]
```

with complex dtype and component order `(Ex, Ey, Ez, Hx, Hy, Hz)`. A
localized sparse detector internally records
`[Nfreq, Ncomponent_local, nx, ny, nz]`, but sparse detector states are unfolded
and scattered into the combined full-domain layout before reaching Torch
objectives. The frequency array corresponds to physical frequencies in Hz unless
a higher-level API explicitly states wavelength or normalized frequency.

## Source normalization

The current wrapper optionally normalizes phasor amplitudes by the source spectrum. This is controlled by the runtime feature flag `MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM`.

This normalization affects reported field amplitude and therefore objective values. Documentation for a source/objective workflow should state whether the phasor is:

```text
raw detector phasor
or
source-spectrum-normalized phasor
```

Source-spectrum normalization is temporal and frequency dependent. It is distinct from spatial source/mode normalization, which integrates the Poynting vector with physical cell-face areas in square meters.

MAPS loads the public pip-installed FDTD-X package through `thirdparty.fdtdx`. That compatibility layer installs MAPS' physical-power functions into the public metric, mode, source, and detector module bindings. Uniform-grid mode solves are routed through explicit physical transverse edge coordinates, and uniform Gaussian-plane sources use `resolution**2` as their area weight. Nonuniform paths use the actual per-cell face areas.

Do not implement MAPS-required numerical changes only in a local FDTD-X checkout. Such changes are absent when users install the pinned public dependency. Project-specific behavior must live under `thirdparty/fdtdx` and be protected by compatibility tests.

## Surface power and modal coefficients

Flux and modal overlap use one orientation-independent SI convention:

```text
x-normal: dA = dy * dz
y-normal: dA = dx * dz
z-normal: dA = dx * dy
```

Per-plane mean normalization is forbidden. It destroys the physical relationship between differently oriented monitors and can scale a reported transmission by the ratio of their mean cell areas.

Reference-mode power is computed under the same quadrature as the simulated-field overlap. Modal power is `|a|^2 / P_mode`; modal amplitude is `a / sqrt(P_mode)`.

For symmetry-reduced simulations, FDTD-X normalizes on the retained plane and MAPS preserves the corresponding inverse-square-root field scaling during full-domain reconstruction. The unfolded field is integrated once; no additional symmetry multiplier belongs in the surface weights.

## Autodiff contract

The solver bridges Torch parameters to a JAX/FDTD-X execution graph and returns Torch tensors that remain differentiable with respect to material/design inputs.

The conceptual chain is:

```text
Torch design parameters
        |
        v
material / epsilon tensor
        |
        v
FDTD-X JAX simulation
        |
        v
phasor fields / detector outputs
        |
        v
Torch objective
        |
        v
autograd / adjoint gradient
```

A backend is not inverse-design-compatible merely because it produces correct forward fields. The returned objective must remain connected to the optimization parameter graph, or an explicit adjoint bridge must supply the missing derivative.

## Gradient conventions

The FDTD-X solver groups monitor support spatially and by frequency for adjoint computation. Group entries conceptually contain:

```python
{
    "group_id": str,
    "monitor_slice": (x_slice, y_slice, z_slice),
    "freq_indices": Sequence[int],
}
```

The compact group representation uses:

```text
freq_indices     [N_group_freq]
spatial_indices  [N_active_points, 3]
```

Spatial indices refer to `(x, y, z)` simulation-grid coordinates. These groups are an optimization/runtime representation, not a replacement for the full field tensor.

## Units

FDTD-X geometry and time/frequency quantities use SI-compatible units:

```text
length            meters
frequency         Hz
angular frequency rad/s
time              seconds
epsilon_r         dimensionless relative permittivity
```

Examples and docstrings should state units explicitly whenever values such as `dL`, boundary coordinates, source widths, or observation coordinates are passed.

## Scientific documentation requirements

Every public FDTD-X callable should specify:

- 3-D tensor shape and axis semantics;
- Yee-grid versus colocated representation;
- component ordering;
- complex dtype expectations for phasors;
- units for geometry/frequency/time;
- source-spectrum normalization state;
- gradient/autograd connectivity;
- monitor slice schema;
- symmetry/unfolding behavior;
- whether arrays live on Torch, JAX, CPU, or GPU devices.
