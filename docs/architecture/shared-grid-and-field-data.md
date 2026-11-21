# Shared Grid and Field Data Flow

This page records the current cross-subsystem data-flow conventions. It is an architectural reference, not a proposed redesign.

## Canonical path

```text
physical device geometry
        |
        v
native/export rectilinear grids
        |
        +--------------------+
        |                    |
        v                    v
design/port masks       material maps
        |                    |
        v                    v
monitor/source slices   FDFD / FDTD solver
        |                    |
        +---------+----------+
                  |
                  v
          colocated fields
                  |
                  v
       eigenmode / flux / FoM
                  |
                  v
          inverse-design loss
```

## Native versus export grids

MAPS can maintain separate native and export representations.

The native grid is tied to the numerical solver. The export grid is used for visualization/downstream interfaces when the two grids differ.

Consequences:

```text
mask used for solver        -> native-grid mask
mask used for plotting      -> export-grid mask
FDTD adjoint grouping       -> native-grid support
field export               -> potentially resampled to export grid
```

A physical monitor is therefore not synonymous with one tensor index tuple. The same physical center/size/direction may need to be projected onto multiple grids.

## Physical coordinates are authoritative

The preferred conceptual flow is:

```text
physical center/size/direction
        |
        v
coordinate-aware index derivation
        |
        v
Slice / Slice3D / native indices / export indices
```

Avoid treating an integer slice from one grid as portable to another grid unless the coordinate arrays are identical.

## Field data contract

The solver-specific field representation differs:

### 2D FDFD

Polarization-specific reduced fields are commonly represented as:

```text
Ez polarization -> {Hx, Hy, Ez}
Hz polarization -> {Ex, Ey, Hz}
```

### 3D FDTD

The canonical six-component representation is:

```text
Ex, Ey, Ez, Hx, Hy, Hz
```

and broadband fdtdx fields introduce a frequency axis:

```text
[Nfreq, X, Y, Z]
```

for each returned component.

Before objective projection, fields must be colocated according to the FDTD Yee-grid convention.

### Sparse FDTD-X recording with a dense boundary

Sparse phasor mode changes the internal recording support, not the cross-system
field interface:

```text
localized forward detectors (design regions + objective ports)
        |
        v
source normalization + symmetry unfolding + scatter/zero fill
        |
        v
dense Torch field [Nfreq, 6, Nx, Ny, Nz]
        |
        v
dense objective cotangent [Nfreq, 6, Nx, Ny, Nz]
        |
        v
adjoint source + localized adjoint detectors (design regions only)
        |
        v
dense adjoint fields -> d objective / d epsilon_r
```

The localized detector slices use native-grid coordinates. If native/export
grids differ, full-domain assembly occurs on the native grid before the existing
coordinate-aware export resampling. Sparse detector storage must not leak a new
sparse container or reduced spatial shape into objective code.

## Reference modes

Mode profiles are stored separately from simulated fields and include E/H reference tensors. Their physical monitor grid must match the simulated field monitor grid for direct overlap evaluation.

The normalization workflow may construct reference modes on an auxiliary two-port device and then resample them to the primary device native coordinates before storing them in `port_sources_dict`.

## Grid weighting

Integration over a nonuniform rectilinear plane uses geometric cell-area weights. This affects:

```text
flux
mode overlap
S-parameter / eigenmode coefficients
power normalization
```

Replacing weighted integration with an unweighted sum is only valid when the grid cell areas are uniform and the same normalization convention is used.

All monitor orientations use the same physical-area convention in square meters. Per-plane normalization such as `area / mean(area)` is prohibited because source and output planes may have different normals and therefore different mean transverse cell areas. Symmetry-unfolded fields and modes already encode the full-domain multiplicity in their spatial samples and amplitude scaling, so grid weighting does not add a second symmetry factor.

## Backend boundaries

The major array-backend boundaries are:

```text
PyTorch <-> JAX/fdtdx
PyTorch <-> NumPy
```

DLPack conversion is used to minimize host copies where possible. Runtime backend conversion can intentionally detach gradients; differentiable solver paths must preserve the backend-specific autograd/adjoint mechanism instead of routing through NumPy.

## Architectural refactoring boundary

The current implementation has mixed responsibilities in both `device_base.py` and `core/utils.py`. This documentation deliberately freezes the **data contracts** before structural refactoring.

A future decomposition should preserve these boundaries:

```text
Geometry/configuration
Grid construction
Mask/index derivation
Source/mode construction
Solver field representation
Coordinate-aware resampling
Field colocation
Objective integration
```

Refactoring should change module ownership without silently changing tensor rank, physical-axis ordering, units, normalization, or grid weighting.
