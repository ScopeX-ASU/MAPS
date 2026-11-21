# Device Geometry and Simulation Contracts

`core/device_base.py` is the central MAPS device abstraction. It currently combines geometry construction, rasterization, ports, sources, monitors, slicing, material maps, design-region masks, and simulation-facing export. This page documents the behavioral contracts of that combined layer so future refactoring can preserve behavior.

## 1. Device lifecycle

A device follows the conceptual pipeline:

```text
constructor/configuration
        |
        v
geometry definition
        |
        v
physical grid / slice construction
        |
        v
material or epsilon_r rasterization
        |
        +--> ports / mode sources
        +--> monitors / slices
        +--> design-region masks
        |
        v
solver-facing tensors / metadata
```

The important invariant is that geometry, grid coordinates, slices, and solver-facing tensors refer to the same physical coordinate system.

A refactor must not allow a geometry object to silently change its physical extent, orientation, or grid alignment merely because the downstream rasterizer changes.

## 2. Physical coordinate conventions

Unless an API explicitly states otherwise:

```text
2-D: (x, y)
3-D: (x, y, z)
```

Lengths in device geometry and optical-coordinate metadata are commonly expressed in micrometers in the current MAPS workflow. Solver-specific wrappers may convert to SI internally.

Every geometry-related API should therefore identify:

- input unit;
- output unit;
- whether coordinates denote centers or boundaries;
- whether a value is a physical coordinate or an integer grid index.

## 3. Geometry versus rasterization

The device layer separates continuous geometry from solver-grid representation.

Conceptually:

```text
continuous/device geometry
        |
        | rasterization
        v
discrete occupancy/material map
```

A geometry parameter such as waveguide width, bend radius, or port location is not itself a tensor index. Rasterization maps the physical geometry onto a specific grid.

This distinction is essential for inverse design: changing a geometry parameter should update the differentiable material representation through the intended parametrization/rasterization path rather than through ad hoc integer rounding.

## 4. Rasterized material map

The solver-facing material representation is typically a tensor such as:

```text
[batch, Nx, Ny]
[batch, Nx, Ny, Nz]
```

or a component/spectral extension documented by the consuming solver.

The device layer should make clear whether the resulting tensor represents:

```text
binary occupancy
or
continuous material interpolation
or
a relative-permittivity / epsilon_r field
```

For inverse design, the gradient path from the design parameters to this tensor is part of the public contract.

A material tensor used by FDFD/FDTD should therefore not be detached or converted through an artifact-only path before optimization is complete.

## 5. Slices

Device slices connect physical geometry to array indexing.

A slice contract should specify:

```text
physical location or interval
axis order
integer grid range
whether stop is inclusive/exclusive
```

Python `slice(start, stop)` semantics are end-exclusive. A physical-coordinate slice description may instead refer to a closed spatial interval. The conversion between the two must be explicit.

A representative 3-D index representation is:

```python
(
    slice(ix0, ix1),
    slice(iy0, iy1),
    slice(iz0, iz1),
)
```

A refactor must not silently change whether the endpoint describes the final included cell or the first excluded cell.

## 6. Ports

Ports provide the interface between device geometry and modal excitation/measurement.

A port conceptually contains:

```text
name
location / center
orientation / propagation direction
cross-sectional extent
mode definition
sampling slice
```

The port orientation must agree with the physical geometry. For a direction such as:

```text
x+
x-
y+
y-
z+
z-
```

the first character identifies the normal axis and the sign identifies the propagation direction.

Port metadata should not be reconstructed from tensor shape alone.

## 7. Source and mode construction

A simulation source may be built from a port and a reference mode.

The conceptual flow is:

```text
port geometry
      |
      v
modal solver / reference mode
      |
      v
source profile + source placement
      |
      v
solver source object
```

A source docstring should identify:

- mode field component ordering;
- spatial tensor shape;
- frequency or wavelength axis;
- complex dtype;
- source plane/slice;
- propagation direction;
- normalization convention;
- whether the result is directly differentiable.

## 8. Mode-field tensor contract

When a mode is represented as a vector field, the component order must be explicit. A common full-field convention is:

```text
(Ex, Ey, Ez, Hx, Hy, Hz)
```

For reduced 2-D polarization workflows, MAPS uses the scalar-polarization conventions documented in the cross-backend data contract:

```text
Ez polarization -> (Hx, Hy, Ez)
Hz polarization -> (Ex, Ey, Hz)
```

A mode array that uses another component order must be converted explicitly rather than inferred.

## 9. Monitors

A monitor describes where and what the simulation records.

A useful conceptual monitor record contains:

```python
{
    "name": "monitor_name",
    "slice": (...),
    "center": (...),
    "size": (...),
    "direction": "x+",
}
```

The actual MAPS dictionaries may contain additional fields for solver-specific behavior, but the core invariant is that the monitor's physical location and array slice identify the same region.

Monitor outputs may be represented as:

```text
[batch, X, Y, F]
[batch, C, X, Y, Z, F]
```

or other solver-specific layouts. The exact axis ordering must be documented at the API boundary.

## 10. Near-field and far-field data

Near-field monitors commonly provide both tensor data and coordinate metadata. The metadata is required to interpret a physical near-to-far integral.

A far-field request may be:

```text
explicit target points: [N, 2] or [N, 3]

or

slice metadata with coordinate arrays
```

Do not convert a physical far-field target into array indices unless the API explicitly defines the target as grid-index based.

## 11. Design-region masks

A design region identifies the subset of the device whose material representation is controlled by optimization.

A mask conceptually has shape:

```text
[Nx, Ny]
[Nx, Ny, Nz]
```

and values corresponding to either:

```text
0 / 1 hard membership
or
continuous weighting / differentiable blend
```

The mask is part of the mapping:

```text
optimization parameters
        |
        v
design-region field
        |
        v
full-device material map
```

The implementation must distinguish a visualization mask from a differentiable optimization mask. A non-differentiable boolean conversion inside the optimization path can change gradient behavior.

## 12. Symmetry and orientation

Device geometry may be transformed by reflection, rotation, or symmetry reduction before rasterization.

Such transformations must preserve a documented convention for:

```text
physical coordinate transform
port orientation
mode polarization
field-component parity
monitor slice
```

For example, reflecting a geometry across x can change a directed port from `x+` to `x-` depending on how the transformed local frame is defined. This cannot be assumed to be invariant.

## 13. Build and export artifacts

The device layer may produce artifacts such as:

```text
epsilon/material tensors
field tensors
GDS geometry
plot arrays
port/monitor metadata
```

These artifacts are not interchangeable with differentiable tensors.

A documentation contract should state whether an output:

- preserves autograd history;
- is detached for export;
- is converted to NumPy;
- represents physical geometry or a discrete grid approximation.

## 14. Representative bending/crossing usage

The Bending and Crossing examples are representative because they exercise the full device path without requiring a large composition hierarchy.

A useful documentation example should make the following explicit:

```text
physical device size: (Lx, Ly)
waveguide width: w
port direction: x+ / x-
grid shape: (Nx, Ny)
material tensor shape: [B, Nx, Ny]
source field shape: [B, Nport, F] or solver-specific equivalent
monitor slice: explicit axis ranges
```

The exact example code should remain the executable source of truth for configuration names; the docstring/reference should explain tensor meaning and physical units around that code.

## 15. Refactoring boundary

`device_base.py` currently combines multiple responsibilities. A future refactor could reasonably split it into:

```text
Geometry / rasterization
Port / mode
Source construction
Monitor / slices
Design-region masks
Solver/export adapters
```

The present implementation should be treated as the compatibility source of truth until tests and documentation cover these interfaces independently.

The documentation goal of this stage is therefore not to make the existing class hierarchy look cleaner. It is to make the current behavior explicit enough that a later decomposition can preserve:

```text
physical geometry
-> grid mapping
-> material representation
-> source/mode placement
-> monitor sampling
-> optimization mask
-> solver-facing data
```

without changing scientific meaning.
