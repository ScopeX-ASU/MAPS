# Cross-Backend Data Contract

This page defines the interfaces that connect MAPS geometry/utilities, FDFD, FDTD-X, and inverse design. The goal is to make tensor shape, coordinate, dtype, unit, and differentiability assumptions explicit at subsystem boundaries.

## 1. Canonical physical axes

Unless an API explicitly documents a different layout, physical axes are ordered:

```text
(x, y)          2-D
(x, y, z)       3-D
```

A tensor may contain leading batch/frequency/component axes, but the spatial-axis meaning must remain explicit in the API contract.

Do not infer physical-axis order from memory layout, flattening order, or an arbitrary tensor permutation performed by a backend adapter.

## 2. Coordinate centers versus boundaries

MAPS uses both coordinate centers and cell boundaries:

```text
centers[a]       : length N_a
boundaries[a]    : length N_a + 1
```

They have different meanings and must not be substituted for each other.

The local cell width is:

$$
\Delta_a[i] = b_a[i+1] - b_a[i].
$$

Integrals on nonuniform grids use these local widths rather than a scalar global spacing.

## 3. FDFD boundary

The frequency-domain solver consumes a material/grid representation and produces complex frequency-domain fields.

Conceptually:

```text
MAPS grid/material
      |
      v
sparse complex Maxwell system
      |
      v
E/H fields indexed by frequency
```

The utility layer supplies derivative matrices, PML factors, coordinate-aware resampling, flux/eigenmode integration, and near-to-far preprocessing.

Any adapter between geometry and FDFD must preserve:

- physical coordinate units;
- spatial axis order;
- complex dtype;
- frequency ordering;
- nonuniform cell widths;
- the intended 2-D polarization convention.

## 4. FDTD-X boundary

FDTD-X is a 3-D time-domain backend using a Yee-staggered grid internally. MAPS exposes Torch-facing colocated fields and phasors for objectives and inverse design.

Conceptually:

```text
MAPS epsilon_r / material grid
          |
          v
      FDTD-X JAX
          |
          v
Yee-staggered E/H samples
          |
          v
colocated / unfolded phasors
          |
          v
Torch objective
```

A utility that operates on FDTD-X fields must state whether its input is:

```text
Yee-staggered
or
colocated
```

This distinction is mandatory because modal overlap, flux, visualization, and many objective functions assume colocated E/H components.

## 5. Inverse-design boundary

Inverse design adds a differentiability requirement to the numerical contract.

A typical path is:

```text
parameter tensor
      |
      v
geometry / parametrization
      |
      v
material grid
      |
      +-----------------+
      |                 |
      v                 v
    FDFD             FDTD-X
      |                 |
      +--------+--------+
               v
        complex fields
               |
               v
          objective / FoM
               |
               v
           gradients
               |
               v
          parameter update
```

Every conversion along this path must preserve the gradient connection unless the operation is intentionally non-differentiable.

## 6. Backend conversion rules

Torch/JAX/NumPy conversions are not interchangeable.

### Differentiable conversion

The FDTD-X bridge uses DLPack-oriented Torch/JAX conversion to avoid unnecessary host copies.

Conceptually:

```text
Torch tensor
   |
   | DLPack
   v
JAX array
   |
   | DLPack
   v
Torch tensor
```

### Runtime-only conversion

A helper may intentionally detach data for runtime/configuration purposes. Such a helper must be treated as a graph boundary.

The dangerous pattern is:

```python
field = field.detach().cpu().numpy()
```

inside a function that is expected to participate in an inverse-design objective.

That can silently eliminate the gradient path.

## 7. Complex dtype rules

Scientific field data should remain explicitly complex when it represents phasors or frequency-domain fields.

Typical choices are:

```text
complex64  -> memory/performance oriented GPU workflows
complex128 -> higher-precision FDFD or validation workflows
```

A utility should not silently cast a complex field to real or silently downcast precision without documenting the consequence.

When a function accepts both NumPy and Torch arrays, document the returned backend and dtype explicitly.

## 8. Frequency axis rules

The frequency axis must never be inferred solely from tensor rank.

Typical MAPS representations include:

```text
[F, ...]
[batch, F, ...]
[..., F]
```

A public API must document which representation it consumes and produces.

FDTD-X phasors may be extracted at multiple requested frequencies from the time-domain simulation. FDFD solves are naturally organized frequency-by-frequency. An adapter that moves between the two must state the frequency axis explicitly.

## 9. Interpolation rules

Coordinate-aware resampling is different from index-space resizing.

For optical fields on rectilinear grids, the source and target physical coordinates determine interpolation locations.

For complex fields, MAPS supports at least two conceptual modes:

```text
cartesian: interpolate real/imaginary values directly
polar:     interpolate magnitude and wrapped phase
```

The choice can change physical results and should be part of the API contract.

For Yee fields, interpolation additionally means **staggered-to-colocated field reconstruction**, not ordinary image interpolation.

## 10. Integration weighting

Flux, modal overlap, far-field, and power calculations depend on physical integration weights.

For a 3-D plane normal to x:

$$
W(y,z)=\Delta y\,\Delta z.
$$

For a nonuniform grid, these weights vary spatially.

Do not replace physical area/volume weights with a constant scalar unless uniformity has been established by the grid metadata.

## 11. 2-D polarization contract

MAPS has scalar-polarization workflows in which the effective three-component field has different meanings.

For `Ez` polarization:

```text
(Fx, Fy, Fz) = (Hx, Hy, Ez)
```

For `Hz` polarization:

```text
(Fx, Fy, Fz) = (Ex, Ey, Hz)
```

This convention is important for flux and eigenmode routines and should be preserved when adding new objectives or field adapters.

## 12. Symmetry and unfolding

Symmetry-reduced simulations must document:

- which physical axes are reduced;
- whether the quantity is an E or H component;
- PEC/PMC parity;
- whether the tensor is spatially reduced or already volume-integrated.

For spatial fields, the common reconstruction is a flip/sign-aware concatenation. For reduced scalar quantities, a symmetry multiplicity/parity factor may be used instead.

These are not interchangeable operations.

## 13. Units

Current cross-backend conventions commonly use micrometers for optical geometry and coordinate metadata, while solver APIs may use SI quantities internally.

Every boundary object must therefore document:

```text
input units
output units
conversion location
```

A particularly dangerous class of bugs is mixing meter-valued coordinates with micrometer-valued wavelength or grid spacing in a near-to-far, material, or source calculation.

## 14. Artifact versus differentiable data

A useful documentation distinction is:

```text
Differentiable data:
    parameter tensors
    epsilon/material tensors
    complex fields used by objectives
    loss/FoM values

Metadata/artifacts:
    coordinate arrays
    slices
    grid descriptions
    source/monitor configuration
    exported NumPy/GDS/plot data
```

Metadata can influence a differentiable calculation without itself requiring gradients. The API should make the distinction clear.

## 15. Required documentation for new cross-backend APIs

Any new helper that connects two numerical subsystems should document:

1. exact input/output tensor shapes;
2. spatial-axis ordering;
3. component ordering;
4. dtype and complex/real semantics;
5. physical units;
6. coordinate center/boundary semantics;
7. frequency-axis position;
8. Yee versus colocated representation;
9. interpolation/integration convention;
10. backend conversion behavior;
11. autograd preservation or intentional graph break;
12. a minimal example showing realistic shapes.

This contract is deliberately stricter than ordinary software documentation because a scientifically plausible tensor with the wrong axis, unit, staggering, or normalization can produce numerically valid but physically incorrect results.
