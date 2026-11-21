# Tidy3D-compatible field projection interface

## Scope and inspected version

This note is the interface gate for MAPS field projection. It records the
public behavior inspected in installed Tidy3D 2.12.1 and defines the smallest
MAPS API that reproduces that workflow with PyTorch tensors. The implementation
may share private validation and numerical helpers, but it must not add a
second public hierarchy of generic projection configurations.

The installed source inspected for this audit is
`tidy3d/components/field_projection/projector.py` and the projection monitor
and data models imported by it. The stable rendered source is also available
in Tidy3D's `FieldProjector` documentation. Tidy3D is an optional validation
dependency; MAPS derives and implements the electromagnetic operators natively
in PyTorch.

## Tidy3D public interface

Tidy3D models use keyword construction through Pydantic, so their Python
signature is displayed as `(**data)`. The model fields below are the effective
public constructor interface.

### Near-field monitor and surface

`FieldMonitor` has these relevant public fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `center` | `(0, 0, 0)` | Monitor center in micrometers. |
| `size` | required | Monitor extent in micrometers. A projection surface has exactly one zero extent. |
| `freqs` | required | Frequencies in hertz. |
| `name` | required | Key used to find the corresponding monitor data. |
| `fields` | all six fields | Any subset of `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, `Hz`. |
| `interval_space` | `(1, 1, 1)` | Positive integer sampling stride used while recording monitor data. |
| `colocate` | `True` | Request colocated field output. |
| `use_colocated_integration` | `True` | Integrate on the colocated grid. |

`FieldProjectionSurface(monitor=..., normal_dir=...)` associates one planar
`FieldMonitor` with its outward normal. `normal_dir` is `"+"` or `"-"`; the
zero-size axis of the monitor supplies the normal axis. The monitor name is the
public association key between a projector surface and near-field data.

Only four components are required to form surface currents. The correct
component sets are:

| Plane | Normal axis | Required fields |
| --- | --- | --- |
| YZ | x | `Ey`, `Ez`, `Hy`, `Hz` |
| XZ | y | `Ex`, `Ez`, `Hx`, `Hz` |
| XY | z | `Ex`, `Ey`, `Hx`, `Hy` |

The sign of the surface normal does not change this selection. It changes the
signs of the equivalent currents through
`J_s = n cross H` and `M_s = -n cross E`.

### Projection monitors

All three projection monitor types share:

| Field | Default | Meaning |
| --- | --- | --- |
| `center` | `(0, 0, 0)` | Center of the planar or box source-monitor geometry. |
| `size` | required | One zero extent makes a plane; three nonzero extents make a closed box. |
| `freqs` | required | Projection frequencies in hertz. |
| `name` | required | Monitor name. |
| `interval_space` | `(1, 1, 1)` | Source-monitor recording stride, distinct from projector resampling. |
| `normal_dir` | `None` | Required orientation for a planar surface. |
| `exclude_surfaces` | `None` | Box faces to omit, named `x-`, `x+`, `y-`, `y+`, `z-`, `z+`. |
| `custom_origin` | `None` | Local coordinate origin; the monitor center is used when absent. |
| `far_field_approx` | `True` | Select asymptotic radiation fields or exact finite-distance fields. |
| `medium` | `None` | Homogeneous projection medium; otherwise inferred from the simulation. |

For a plane, `projection_surfaces` produces one surface with `normal_dir`. For
a box it produces up to six faces with outward normals and names derived from
the projection-monitor name. `exclude_surfaces` removes named faces.

`FieldProjectionAngleMonitor` adds required `theta` and `phi` coordinate
arrays in radians and `proj_distance=1e6` micrometers. Theta is polar angle
from +z and phi is azimuth about +z from +x. `proj_distance` is radial distance
from the local origin.

`FieldProjectionCartesianMonitor` adds required `proj_axis`, `x`, and `y`, and
`proj_distance=1e6` micrometers. `proj_axis` is 0, 1, or 2. The two local arrays
map to global `(y,z)`, `(x,z)`, or `(x,y)`, respectively. `proj_distance` is a
signed displacement along `proj_axis` from the local origin.

`FieldProjectionKSpaceMonitor` adds required `proj_axis`, `ux`, and `uy`, and
`proj_distance=1e6` micrometers. `ux` and `uy` are direction cosines normalized
by the wave number in the projection medium and must describe propagating
directions. `proj_distance` is radial distance from the local origin.

### Projector construction and result data

`FieldProjector` exposes:

- `sim_data`: simulation data containing the near-field results.
- `surfaces`: one or more `FieldProjectionSurface` objects.
- `pts_per_wavelength=10`: projector-side current-grid resampling density;
  `None` disables this optional resampling.
- `origin=None`: observation/phase origin used to translate source-current
  coordinates; if absent, Tidy3D uses the average source surface center.
- `from_near_field_monitors(sim_data, near_monitors, normal_dirs,
  pts_per_wavelength=10, origin=None)`.
- `from_near_field_data(...)` in installed 2.12.1 for externally supplied
  field data and a Cartesian projection monitor.
- `project_fields(proj_monitor, verbose=True, freq_chunk_size=8)`.

The constructor resolves each surface's `monitor.name` in `sim_data`, verifies
a common homogeneous exterior medium, and prepares equivalent currents.
Projection dispatch depends on the projection-monitor class.

The public results are `FieldProjectionAngleData`,
`FieldProjectionCartesianData`, and `FieldProjectionKSpaceData`. Each carries
its monitor, source surfaces, and complex `Er`, `Etheta`, `Ephi`, `Hr`,
`Htheta`, and `Hphi` arrays. Angle data is indexed by `(r, theta, phi, f)`,
Cartesian data by `(x, y, z, f)` with a singleton projected axis, and k-space
data by `(ux, uy, r, f)`. Cartesian field conversions and power-related
properties are exposed by the data classes. Directional power uses
`0.5 Re(Etheta conj(Hphi) - Ephi conj(Htheta))`; angular flux also includes
the radial-area and solid-angle factors.

### Public semantics versus implementation details

The classes, fields, constructors, `projection_surfaces`, projector factory,
`project_fields`, named result components, coordinate meanings, and units are
public behavior to mirror.

Tidy3D's xarray storage, Pydantic machinery, private cached properties,
internal current names, chunk implementation, interpolation implementation,
and exact-kernel decomposition are implementation details. MAPS will not copy
those structures. Tidy3D 2.12.1 uses linear interpolation during current-grid
resampling and trapezoidal integration. MAPS must match the numerical meaning
but can use differentiable Torch algorithms and explicit quadrature tensors.

## Smallest compatible MAPS interface

MAPS will export these public classes from `core.nearfield`:

```python
FieldMonitor(
    center, size, freqs, name,
    fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    interval_space=(1, 1, 1),
    colocate=True, use_colocated_integration=True,
)

FieldProjectionSurface(monitor, normal_dir)

FieldProjectionAngleMonitor(
    center, size, freqs, name, theta, phi,
    proj_distance=1e6, normal_dir=None, exclude_surfaces=None,
    custom_origin=None, far_field_approx=True, interval_space=(1, 1, 1),
)

FieldProjectionCartesianMonitor(
    center, size, freqs, name, proj_axis, x, y,
    proj_distance=1e6, normal_dir=None, exclude_surfaces=None,
    custom_origin=None, far_field_approx=True, interval_space=(1, 1, 1),
)

FieldProjectionKSpaceMonitor(
    center, size, freqs, name, proj_axis, ux, uy,
    proj_distance=1e6, normal_dir=None, exclude_surfaces=None,
    custom_origin=None, far_field_approx=True, interval_space=(1, 1, 1),
)

FieldProjector(
    sim_data, surfaces, pts_per_wavelength=10, origin=None,
    eps_r=1.0, mu_r=1.0, backend="auto",
)
```

Geometry arguments follow the existing MAPS/Tidy3D device convention and are
in micrometers; frequencies are in hertz and angles are in radians. Conversion
to SI occurs once at the projector numerical boundary. Coordinate arguments
may be Python sequences or real Torch tensors. Projection outputs stay on the
input field device, retain complex precision, and remain attached to the
autograd graph.

`sim_data` is a mapping from monitor name to MAPS phasor output. A private
adapter reads the recorded component tensors, their component-specific Yee
coordinates, frequency order, grid boundaries, and E/H temporal-alignment
metadata. This avoids a second public data-model hierarchy. Existing verified
`Surface` objects remain a low-level numerical interface.

`FieldProjector.from_near_field_monitors(...)` and `project_fields(...)` match
the Tidy3D workflow. `project_fields` returns a Torch-native
`FieldProjectionData` holding the same six named spherical components and
explicit coordinate tensors. It also exposes differentiable Cartesian
components, directional power, and integrated flux. The result layout follows
the corresponding Tidy3D coordinate order, with frequency last at the public
boundary; low-level kernels may use frequency-first layouts privately.

No public generic `ObservationGrid`, `ProjectionConfig`, or backend-specific
projector class is added. Existing `AngularObservation`,
`CartesianObservation`, and low-level projection functions remain supported as
numerical APIs and are used privately by the compatible facade.

For client-side projection, the projector's `origin` is authoritative: source
coordinates and observation coordinates are expressed relative to it. Tidy3D's
client projector does not independently apply the target monitor's
`custom_origin`; server-side monitor construction arranges for the corresponding
projector origin. MAPS therefore requires a supplied `custom_origin` to equal
the projector origin, preventing a silent phase or position offset.

## Target MAPS monitor integration

This section defines the intended raw-Yee end state. The implemented boundary
and its present limitations are recorded under **Current MAPS implementation
boundary** below.

`DeviceBase.insert_monitors_fdtd3d(type="near2far", ...)` will construct one
`PhasorDetector` per requested plane. The component tuple is selected from the
plane normal using the four-component table above. Six box faces therefore
carry four components each with explicit outward signs.

Dedicated projection monitors record native staggered Yee samples. The adapter
applies the established MAPS magnetic half-time correction for the
`exp(-i omega t)` convention, constructs `J_s` and `M_s`, and colocates current
components onto a common physical integration grid. It must not apply temporal
or spatial correction twice when detector metadata states that either step has
already occurred.

`interval_space` is monitor-side selection. The first and last samples on each
surface are retained, matching Tidy3D endpoint behavior. If the underlying
detector cannot allocate a strided non-contiguous subgrid, MAPS may initially
record the containing plane and apply the stride when materializing monitor
data; this preserves numerical semantics but does not claim detector-memory
savings.

## Colocation, resampling, and quadrature

Surface preparation has this fixed order:

1. Read the four tangential complex phasors with their native component Yee
   coordinates.
2. Apply the declared magnetic half-time phase/amplitude correction if needed.
3. Form signed equivalent electric and magnetic currents on their native
   component grids.
4. Linearly colocate all nonzero current components onto a common physical
   integration grid.
5. If `pts_per_wavelength` is not `None`, resample the common grid using the
   shortest wavelength among requested frequencies in the exterior medium.
6. Apply physical surface quadrature weights once and project.

`pts_per_wavelength=None` skips step 5 only. Mandatory Yee colocation remains.
Monitor-side `interval_space` is applied before projector-side wavelength
resampling.

When either operation reduces resolution, MAPS uses a differentiable separable
area/low-pass filter before interpolation to prevent avoidable aliasing. This
is an intentional MAPS quality improvement and is not represented as behavior
present in Tidy3D 2.12.1. Native-resolution identity paths perform no filtering.
Nonuniform grids use physical cell boundaries and conservative overlap weights;
uniform grids may use equivalent Torch pooling/convolution. Trapezoidal or
cell-area weights are derived from physical coordinates and are never inferred
from tensor indices.

## Projection behavior and backends

- Angular projection supports the far-field asymptotic operator and exact
  finite-distance points.
- Cartesian projection supports exact points and the far-field approximation.
- K-space projection supports propagating directions and the same far/exact
  monitor flag; the FFT angular-spectrum path is used only for compatible
  parallel regular planes.
- `direct_torch` is the correctness reference and always chunks source and
  observation axes rather than allocating a dense Green matrix.
- `separable_torch`, `fft_torch`, and `direct_triton` are selected only where
  their mathematical preconditions hold. `auto` falls back to `direct_torch`
  when they do not.
- Optimized backward operations must match native Torch complex gradients and
  implement the adjoint of the same quadrature-weighted forward operator.

## Verification gates

Implementation cannot progress to performance tuning until focused tests pass:

1. Public constructor defaults, validation, plane/box surface generation,
   outward normals, excluded faces, coordinate mapping, and four-component
   monitor selection.
2. Native Yee current colocation, E/H temporal phase alignment, endpoint
   retention, `pts_per_wavelength=None`, downsampled anti-alias behavior, and
   physical quadrature on uniform and nonuniform grids.
3. Identical-input comparisons with Tidy3D for complex E/H in angular,
   Cartesian, and k-space coordinates, separately for far and exact modes.
4. Analytical electric/magnetic dipoles, plane waves, apertures, and Gaussian
   beams, including phase, polarization, power, and weak-field-safe errors.
5. Directional-derivative and native-autograd comparisons for field matching,
   intensity, directional power, and polarization objectives.
6. MAPS full-wave comparisons between projected Huygens fields and direct DFT
   monitors under mesh, distance, surface-position, and sampling refinement.

Performance reports follow those gates and state dtype, hardware, batch sizes,
compile time, warm forward/backward time, peak GPU memory, and error against a
complex128 reference.

## Current MAPS implementation boundary

The public monitor/projector facade consumes verified `Surface` samples. The
current FDTD-X boundary does not provide component-specific Yee coordinates or
magnetic half-time metadata to this facade, so it does not claim to perform
native-Yee current colocation or temporal correction. `DeviceBase` records the
four tangential components with `exact_interpolation=True`, which leaves
colocation detector-side for the existing solver path. Raw phasor tensors are
rejected unless the caller supplies the verified low-level `Surface`
representation.

Projector-side wavelength resampling supports complete rectilinear planar
`Surface` grids using differentiable separable linear interpolation for equal
or finer grids and physical control-volume overlap averaging when reducing a
dimension. It rebuilds physical trapezoidal areas. Complete rectilinear grids
may be nonuniform; reduction uses their physical coordinates. `None` skips this
optional resampling and preserves the explicit input sample locations and
weights.
