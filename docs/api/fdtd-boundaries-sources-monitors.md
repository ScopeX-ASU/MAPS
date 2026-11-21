# FDTD-X Boundaries, Sources, and Monitors

The MAPS FDTD-X layer translates user-facing geometry/source/monitor configuration into FDTD-X scene objects and physical coordinate constraints.

## Physical coordinate boundaries

The preferred placement representation uses physical coordinate arrays rather than implicit relative positions.

For a 3-D grid:

```text
x_edges.shape == [Nx + 1]
y_edges.shape == [Ny + 1]
z_edges.shape == [Nz + 1]
```

A cell-index bounds tuple:

```python
((ix0, ix1), (iy0, iy1), (iz0, iz1))
```

maps to physical coordinates using the corresponding edge arrays:

```python
(
    (x_edges[ix0], x_edges[ix1]),
    (y_edges[iy0], y_edges[iy1]),
    (z_edges[iz0], z_edges[iz1]),
)
```

The resulting coordinates are in meters.

## Boundary configuration

The current wrapper reads FDTD-X boundary configuration fields for each face, including thickness, type, and graded-profile parameters such as kappa, alpha, and sigma start/end/order values.

These are solver/material-absorber parameters, not generic geometric coordinates. A boundary docstring should distinguish:

```text
face kind        min_x, max_x, min_y, max_y, min_z, max_z
thickness        number/physical extent according to FDTD-X config
coordinate       physical boundary placement in meters
PML profile      kappa / alpha / sigma parameters
```

## Sources

MAPS passes FDTD-X source objects/specifications through the injected runtime. Source profiles can include single-frequency and Gaussian-pulse behavior, depending on the workflow.

The source contract should always specify:

- source object/profile type;
- spatial support and its coordinate convention;
- polarization/component;
- amplitude normalization;
- frequency or pulse bandwidth;
- whether the source is used for forward simulation, adjoint simulation, or both.

When source-spectrum normalization is enabled, a raw detector phasor should not be compared directly with a normalized objective/reference without applying the same convention.

Spatial power normalization uses physical monitor areas independently of source-spectrum normalization. Uniform sources use the physical `resolution**2` face area, while rectilinear sources use the local transverse face-area array. The weights are not divided by their mean.

Under `k` active transverse symmetry planes, the retained source fields are scaled so the unfolded full-domain source has the intended power. Monitor quadrature must not apply a second `2**k` factor after fields have been unfolded.

MAPS-specific source and mode normalization behavior is installed from `thirdparty.fdtdx`; modifying an external pip checkout is not a supported deployment mechanism.

### Grating-coupler Gaussian waist

`GratingCoupler(waist_radius=..., waist_distance=...)` accepts micrometers and
converts them to meters for `GaussianPlaneSourceTidy3d`. The radius is measured
at 1/e electric-field amplitude. At zero waist distance, the transverse field
envelope is `exp(-(x**2 + y**2) / waist_radius**2)`, evaluated at physical cell
centers on a rectilinear grid. Power normalization scales the amplitude while
preserving the transverse shape. The source plane stays the same size as the
design region, independently of the waist radius. A radius much larger than that
rectangle makes the prescribed incident profile approach a uniform field within
it. For a centered 4-by-4-micrometer rectangle at the waist, the continuous
corner-to-peak amplitude ratio is `exp(-8 / waist_radius**2)` with the radius in
micrometers: 92.31% at radius 10 and 99.92% at radius 100.

For all physically sized sources on nonuniform grids, MAPS resolves the requested
physical intervals against the native edges at the source location, then places
the explicit cell shape using physical edge constraints. This includes mode,
Gaussian, uniform, and hard plane sources, as well as physical source volumes.
It avoids converting a physical width to a cell count at the domain origin and
reusing that count where the cell widths differ. The intervals are covered to
native-cell precision; a Gaussian aperture does not expand when its waist
radius increases. This corrects a pre-existing placement error and intentionally
changes illumination on affected grids.

Size resolution is based on metadata rather than source class. An axis with
`partial_grid_shape` specified retains that cell count, even if a physical
extent is also stored. On mixed physical/grid-sized sources, those axes select
the cell interval closest to the requested physical center. Unspecified axes
span the native domain. Fully grid-sized sources, including distributed adjoint
currents and point dipoles, retain the previous placement path. Uniform-grid
placement also retains its existing behavior. Physical source volumes resolve
all three extents; a plane's explicit one-cell normal axis remains one cell.
The sparse input-source field recorder uses the same corrected bounds, so the
correct physical aperture is observed rather than filled with zeros outside an
older, smaller detector support.

The [placement audit](../design/gaussian-source-placement-audit.md) records an
independent coordinate counterexample, the original and patched grating-aperture
bounds, the discretization choice, and the limits of this validation.

The applied source's `_E` and `_H` arrays describe the injected incident profile;
recorded phasors describe the evolved electromagnetic field. These quantities
must not be interpreted interchangeably. In particular, exact Yee interpolation
of transverse electric fields averages samples on either side of the injection
boundary when a detector lies on that boundary. Such a detector does not measure
the prescribed incident profile alone. Away from the boundary, the finite
aperture can also produce edge modulation. The three-view plots show the evolved
field at the design-region planes, rather than a plot of the source profile.

Optimization initialization builds source profiles from the device parameters.
Changing a device attribute afterward does not rebuild the existing profiles;
construct a new device and optimization model for each waist comparison. Use
distinct experiment names when comparing the example's exported plots and logs,
because its default `exp_name` does not include the waist radius.

## Monitors and detector slices

Detector metadata uses grid slices for spatial selection. The solver converts slices into explicit `(x, y, z)` spatial indices when building compact adjoint groups.

A monitor slice therefore has semantic content beyond Python slicing syntax:

```text
slice on x -> simulation-grid x indices
slice on y -> simulation-grid y indices
slice on z -> simulation-grid z indices
```

For a monitor with `N` selected spatial points and `F` frequencies, a component phasor is conceptually:

```text
[batch, N, F]
```

before it is reshaped back into a monitor surface/volume layout.

## Exact versus colocated interpolation

FDTD-X's native detector can use exact interpolation at the Yee-grid locations. MAPS can also colocate/interpolate fields onto a common tensor grid for Torch objectives.

The two representations may differ at the level of individual samples while describing the same physical field. `MAPS_COMPARE_EXACT_INTERPOLATION` exists to compare them explicitly.

A scientific example should state which representation it is using.

## Monitor coordinate metadata

A monitor's physical coordinates and grid slices must remain synchronized. Near-field/far-field objectives, mode overlap, and adjoint grouping all depend on this metadata.

For any public monitor helper, document:

```text
slice schema
coordinate units
axis order
field component order
frequency order
interpolation mode
```

## Example contract

A representative surface monitor may be documented as:

```python
monitor_slice = (
    slice(10, 110),
    slice(20, 120),
    slice(30, 31),
)
```

This selects:

```text
100 x 100 x 1 = 10,000 grid locations
```

with the exact physical extent determined by the grid boundary arrays. A detector returning six field components over 16 frequencies therefore has an unflattened contract conceptually equivalent to:

```text
[batch, 6, 100, 100, 1, 16]
```

The implementation may use a different internal axis order, so the function docstring must state the actual returned order explicitly.

## Sparse phasor detector placement

On nonuniform grids, physical detector sizes are resolved against native edges
at the requested detector center, just like physical source sizes. Native port
registration and mode-overlap profile solves use this same rule. Explicit cell
counts take precedence over physical extents; an unspecified axis spans the
domain, while an unspecified center defaults to the domain midpoint. Plane
detectors retain their explicit one-cell normal axis, and volume detectors
resolve all physical extents.

Before placement, MAPS clones the detector with the resolved native shape,
clears its physical-center constraint, and fixes both lower and upper edges.
The original prototype is preserved. Clearing the center matters because
covering a physical interval with nonuniform cells can produce an asymmetric
snapped aperture. Stored recorder bounds, including variation-padded supports,
override physical metadata. Symmetry folding happens after this full-domain
placement. Uniform-grid port registration and mode solving retain their prior
placement paths.

This intentionally changes affected modal profiles, overlap integration, and
recorded coverage: the previous conversion could count cells at the domain
origin and reuse that count at a different location. It does not change the
detector interpolation or flux/overlap formulas.

`MAPS_SPARSE_PHASOR=1` places multiple localized phasor detectors instead of one
full-volume detector. Forward placement includes native-grid design regions and
objective ports. By default it also includes thin full-span `xy`, `xz`, and `yz`
slabs centered on each design region for `plot_eps_field_3d()`; coincident slabs
are deduplicated. With no design regions, a single three-plane set is centered
at physical origin using the native rectilinear coordinates. Set
`MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES=0` before import to disable this plotting
support. Adjoint placement includes design regions only; it must not add
visualization, objective-port, or full-volume detectors merely to constrain
source placement.

The full-domain adjoint source and the localized detectors are positioned
independently:

```text
adjoint source bounds     -> full native simulation domain
adjoint detector bounds  -> individual native design regions
```

For a renamed primary adjoint detector, placement reuses the bounds of the
corresponding forward design detector. Additional design detectors are selected
by their stored support names, while port detector names are excluded.

Under symmetry, FDTD-X places these objects on the reduced domain. MAPS unfolds
recorded phasors using PEC/PMC component parity and maps them back to their
full-domain slices before dense assembly. The placement bounds and unfolded
tensor coordinates must therefore be kept together; inferring a localized
detector as full-volume from its renamed object name is invalid.

Visualization slabs use three native cells along their normal axis to retain the
interpolation stencil needed when fields are resampled onto the plotting grid.
For a plane normal to a symmetric axis, placement uses the retained half and the
normal parity-unfolding path reconstructs the opposite half.
