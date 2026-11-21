# 3D Equivalent-Surface Near-to-Far Projection

`core.nearfield` provides a native PyTorch implementation of the homogeneous
medium equivalence principle for sampled three-dimensional surfaces. It
projects complex electric and magnetic phasors to finite observation points or
to radiation-zone directions, with Torch autograd through the surface fields
and observation geometry.

This API consumes already sampled fields. It does not solve Maxwell's
equations, collocate Yee-grid components, correct electric/magnetic time
staggering, divide out a source spectrum, reconstruct symmetry planes, or infer
monitor coordinates. Those steps must be completed and documented by the
caller before constructing a `Surface`.

## Conventions and tensor layout

The time convention is `exp(-i omega t)` and the outgoing Green function is
`exp(+i k R)/(4 pi R)`. Input field units are V/m for `E` and A/m for `H`;
coordinates and area weights are meters and square meters; frequency values are
Hz. The projection medium is homogeneous, lossless, and described by positive
scalar relative permittivity and permeability. For free space, use the
defaults `eps_r=1` and `mu_r=1`.

| Quantity | Shape | Meaning |
| --- | --- | --- |
| `positions_m` | `(P, 3)` | Source coordinates in Cartesian x/y/z order |
| `normals` | `(P, 3)` | Unit outward surface normals at each sample |
| `weights_m2` | `(P,)` | Positive quadrature areas |
| `electric_fields` | `(F, P, 3)` | Complex electric phasors |
| `magnetic_fields` | `(F, P, 3)` | Complex magnetic phasors |
| `frequencies_hz` | `(F,)` | Positive frequencies corresponding to the leading field axis |

All tensors must be on one device. Geometry and frequency tensors are real;
field tensors are complex and share a shape. Normals point out of the enclosed
equivalence region. The surface currents are `J_s = n × H` and `M_s = -n × E`.
Every included surface patch contributes to the integral exactly once. Patch
overlaps and quadrature choices are the caller's responsibility.

## Constructing a planar patch

`make_planar_surface` accepts an origin, two linearly independent direction
vectors, increasing coordinates along each direction, fields shaped
`(F, Nv, Nu, 3)` (or flattened `(F, Nv*Nu, 3)`), and frequencies. It forms
points in v-major/u-major order and uses tensor-product composite trapezoid
weights. The normal follows `u_axis × v_axis`; swap the axes when the opposite
orientation is required.

```python
import torch
from core.nearfield import make_planar_surface

dtype = torch.float64
cdtype = torch.complex128
x = torch.linspace(-1e-6, 1e-6, 9, dtype=dtype)
y = torch.linspace(-1e-6, 1e-6, 7, dtype=dtype)
freqs = torch.tensor([193.4e12], dtype=dtype)
e = torch.zeros((1, len(y), len(x), 3), dtype=cdtype)
h = torch.zeros_like(e)
surface = make_planar_surface(
    origin_m=torch.tensor([0.0, 0.0, 0.0], dtype=dtype),
    u_axis=torch.tensor([1.0, 0.0, 0.0], dtype=dtype),
    v_axis=torch.tensor([0.0, 1.0, 0.0], dtype=dtype),
    u_coordinates_m=x,
    v_coordinates_m=y,
    electric_fields=e,
    magnetic_fields=h,
    frequencies_hz=freqs,
)
```

For arbitrary quadrature or cell-centered samples, construct `Surface`
directly. `concatenate_surfaces` combines patches in input order and requires
identical frequency grids, device, and field dtypes. It does not remove shared
edges or corners.

## Finite-distance projection

`project_nearfield(surface, observation_points_m, eps_r=1.0, mu_r=1.0, *,
source_chunk_size=256, observation_chunk_size=256, checkpoint_chunks=True)`
returns `(electric, magnetic)`, each complex with shape `(F, M, 3)`. Each target
must be distinct from every source sample. Chunk sizes bound the pairwise
source/observation work arrays. When autograd is active, `checkpoint_chunks`
recomputes blocks during backward to reduce saved intermediate memory at the
cost of additional compute.

```python
from core.nearfield import project_nearfield

targets = torch.tensor([[0.0, 0.0, 5e-6]], dtype=dtype)
electric, magnetic = project_nearfield(
    surface, targets, source_chunk_size=128, observation_chunk_size=32,
)
assert electric.shape == magnetic.shape == (1, 1, 3)
```

The calculation includes reactive and radiating terms; it is the finite-point
homogeneous-medium Green-function projection. It does not model material
interfaces between the source surface and target.

## Radiation-zone projection

`project_farfield(surface, directions, eps_r=1.0, mu_r=1.0, *,
source_chunk_size=512, observation_chunk_size=256, checkpoint_chunks=True)`
accepts unit directions with shape `(M, 3)` and returns complex `E` and `H`,
each shaped `(F, M, 3)`. The electric result is a far-field coefficient with
units V and the magnetic result has units A: the radial factor
`exp(+i k r)/r` is omitted. Multiply both fields by that factor to compare
with a finite-radius field in V/m and A/m. Source/observation chunks bound the
Fourier phase work arrays; `checkpoint_chunks` trades recomputation for saved
memory during backward.
`direction_vectors_spherical(theta_rad, phi_rad)` creates directions with
polar angle from +z and azimuth from +x toward +y. `directional_power` computes
time-average radiation intensity `r² <S_r>` in W/sr from the electric
coefficient.

## Monitor adapters and observation helpers

The `FieldMonitor`, `FieldProjectionSurface`, three
`FieldProjection*Monitor` classes, and `FieldProjector` provide a compact
Tidy3D-style workflow from verified `Surface` samples. Monitor geometry and
projection coordinates are in micrometers; frequencies are in hertz and
angles are in radians. `FieldProjector.project_fields` returns
`FieldProjectionData` with spherical complex components and frequency last.
The facade dispatches far-field and exact finite-distance calculations to the
existing native Torch kernels and preserves autograd.

The current FDTD-X monitor boundary does not expose component-specific Yee
coordinates or magnetic half-step metadata to this facade. Therefore its
`sim_data` mapping must contain verified `Surface` objects keyed by monitor
name. It rejects raw phasor tensors rather than infer sample locations or
alignment. A non-`None` `pts_per_wavelength` resamples only surfaces whose
sample positions form a complete rectilinear Cartesian product on the
declared plane. Interpolation and physical-overlap averaging during
downsampling are differentiable, and rebuild trapezoidal area weights. The
resampler uses separable linear interpolation for equal or finer grids and
control-volume overlap averages when reducing a dimension. Use
`pts_per_wavelength=None` to preserve explicit positions and quadrature
unchanged. Monitor-side `interval_space` metadata is carried by
monitor descriptions but cannot be applied to an already materialized
`Surface`.

`from_maps_fields` in `core.nearfield` extracts a plane from six component
fields shaped `(F,6,X,Y,Z)` (or a six-tensor sequence) after the caller
provides verified x/y/z coordinates in meters, physical quadrature areas,
normal orientation, and explicit declarations that fields are colocated,
temporally aligned, and unfolded. It does not interpolate Yee components,
apply a half-step phase, divide by a source spectrum, or infer coordinates.
Sparse recorded planes require an explicit coverage mask. See the [MAPS
monitor audit](../design/nearfield-projection.md#maps-monitor-audit) for the
coordinate and Yee-registration constraints.

```python
from core.nearfield import from_maps_fields

# x_m, y_m, z_m and area_m2 are verified physical monitor data.
surface = from_maps_fields(
    fields, frequencies_hz=freqs_hz, coordinates_m=(x_m, y_m, z_m),
    normal_axis=2, normal_sign=1, index=monitor_k,
    colocated=True, temporally_aligned=True, unfolded=True,
    weights_m2=area_m2,
)
```

`direction_cosines` builds a hemisphere of unit directions from transverse
direction cosines; `cartesian_observation_plane` creates explicit target
coordinates in meters. `spherical_components` resolves Cartesian fields onto
the local spherical polarization basis, and `integrated_directional_power`
combines directional intensity with caller-supplied solid-angle weights.
Solid-angle quadrature is separate from surface area quadrature. See
[`core.nearfield.observations`](../design/nearfield-acceleration.md#observation-helpers)
for shape/order details. For example, `direction_cosines(u[:, None],
v[None, :])` creates an angular mesh whose vectors can be flattened to `(M,3)`
for `project_farfield`.

`projection_adjoint` applies a field-space VJP for fixed geometry and material
parameters in near or far mode, returning complex cotangents for sampled E/H.
It recomputes the forward operation without constructing a dense projection
matrix. It is useful when an explicit field cotangent is available; ordinary
Torch autograd remains the most direct option for scalar objectives.

```python
from core.nearfield import projection_adjoint

grad_e, grad_h = projection_adjoint(
    surface, targets, electric_cotangent, magnetic_cotangent,
    mode="near", source_chunk_size=128, observation_chunk_size=32,
)
```

## Inverse-design objective integration

`ShapeSimilarityNear2FarObjective` supports the legacy 2-D projector and the
native 3-D projector through the existing `intensity_shape_near2far` objective
type. For 3-D FDTD-X simulations it expands the broadband simulation key in the
same way as `SMatrixObjective`, extracts every registered monitor whose name
starts with `nearfield`, and projects all six colocated E/H components. Native
grid coordinates and cell boundaries supply the source positions and
nonuniform-grid area weights. MAPS's impedance-normalized magnetic field is
converted to A/m before projection.

The default `return_field=None` computes the configured scalar shape
similarity. Set `return_field="nearfield"` to return sampled surface fields and
their SI geometry, or `return_field="farfield"` to return projected Cartesian
E/H fields on the transverse observation grid. These raw results retain their
Torch autograd graph and are stored in `ObjectiveFunc.breakdown[name]["value"]`
under keys `(input_slice, wavelength_um, input_mode, control_state)`. A raw
field objective is excluded from scalar objective fusion by default; set
`include_in_fusion=True` only when the configured fusion function explicitly
handles its dictionary value.

```python
objective_cfg = {
    "weight": 0.0,
    "type": "intensity_shape_near2far",
    "in_slice_name": "in_slice_1",
    "out_slice_name": "observation_plane",
    "in_mode": "Ez1",
    "out_modes": ("Ez1",),
    "wl": (1.55,),
    "temp": (300,),
    "return_field": "farfield",
    "projection_backend": "direct_torch",
}
```

In 3-D, register each Huygens plane with `build_near2far_slice`. The device can
construct a localized four-component tangential `fdtdx.PhasorDetector` with
`insert_monitors_fdtd3d(..., type="near2far")`. Dense FDTD-X field recording
already covers these planes; sparse field recording creates localized port
detectors for all registered monitor slices. Each near-field plane must be one
native cell thick, lie in the homogeneous projection medium, and use the
outward sign in its direction string, such as `z+` or `x-`.

These detectors retain FDTD-X's `exact_interpolation=True` mode because the
current solver path has not been verified for native four-component staggered
output. The compatible facade consequently accepts only preverified
`Surface` samples and does not claim to perform native-Yee colocation or the
magnetic half-time correction.

## Differentiation and cost

Projection uses ordinary PyTorch operations, so gradients flow through complex
surface fields, real observation coordinates, normals, area weights, and
scalar material inputs when those tensors require gradients. Geometry is not
automatically detached. The current implementation evaluates pairwise
source/target blocks; runtime scales with frequency count, source count, and
target count. Select chunks based on available memory and measure the actual
workload. `core.nearfield.project(..., backend="auto")` can use caller-supplied,
exact-shape benchmark calibration records for direct Torch versus Triton
near-field selection; without a matching validated record it chooses Torch.
It does not extrapolate across workloads or benchmark candidates at runtime.
The selector does not apply to the separate planar far-field or FFT APIs. See
the [dispatch contract](../design/nearfield-dispatch.md). The `project` wrapper
also accepts explicit `direct_torch` and `direct_triton` requests:

```python
from core.nearfield import project

electric, magnetic = project(
    surface, targets, backend="auto", calibration_records=records,
    expected_calls=10, requires_backward=True,
)
```

The command below reports hardware, dtype, chunk sizes, first-call and warm
forward/backward times, CUDA peak allocated memory, and (for the small
float32 case) relative error against the same deterministic inputs evaluated
in complex128. Synthetic fields in that benchmark are for runtime and numeric
precision characterization only; they are not physical validation:

```bash
python -m benchmarks.nearfield.projector --case small --device cpu
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.nearfield.projector \
  --case sixbox --device cuda --dtype float32 \
  --source-chunk 256 --observation-chunk 128 --output /tmp/nearfield.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.nearfield.projector \
  --case small --device cuda --dtype float32 --backend triton
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.nearfield.projector \
  --case small --device cuda --dtype float32 --backend triton --cold-triton-cache
```

`--backend torch` selects the native reference implementation and `--backend
triton` selects the optional fused exact near-field CUDA kernel. Triton accepts
float32 geometry and complex64 fields; its first call may include kernel
compilation when the cache is cold. `--cold-triton-cache` creates an empty
private cache under the system temporary directory so that first-call timing
includes JIT compilation; it does not clear or reuse an existing cache. The
separable radiation-zone and FFT angular-spectrum workloads have separate
contracts and are benchmarked by
``benchmarks/nearfield/planar_backends.py``.

`project_farfield_separable` evaluates planar radiation-zone fields with
separable Fourier matrix products. It requires a tensor-product planar surface
and explicit axes, source coordinates and angular direction-cosine vectors;
it returns `(F,Kv,Ku,3)` V/A coefficients. `angular_spectrum_torch` propagates
a one-sided tangential E field on a uniform periodic plane and reconstructs
full E/H. It assumes the caller handles finite-aperture padding and sampling
adequacy. These specialized operations are not exact finite-distance
equivalent-current projections; their input and sampling assumptions are
documented in the [acceleration guide](../design/nearfield-acceleration.md).

## Examples and evidence status

The interactive notebooks in [`examples/nearfield/`](../../examples/nearfield/)
cover planar radiation, exact finite-distance fields, an analytical dipole on a
closed six-face surface, a Tidy3D comparison, a direct Yee FDTD comparison, and a
focal objective with spatial and directional finite-difference gradient checks.
They display E/H components, shared-scale intensity, complex-field and phase
errors, and measured forward/backward runtime and CUDA allocation where
applicable. See the [visual verification guide](../../examples/nearfield/README.md)
for interpreting these plots and measurements. A comparator or synthetic
demonstration alone is not evidence of independent physical agreement. The
optional MAPS-artifact comparison requires recorded, explicitly collocated,
temporally aligned, unfolded six-component fields with verified coordinates.

The `MAPS zone-plate notebook`
runs the production MAPS FDTD wrapper and projects its recorded near plane to
the observation grid in Flexcompute's zone-plate example. It includes a local
Tidy3D projector comparison using the same MAPS fields, without a cloud solve.
See the [case conventions](../design/zone-plate-projection.md) for the source
normalization, impedance conversion, Ez-point coordinates and projection origin.

`benchmarks/nearfield/verify_dipoles.py` independently compares exact and
radiation-zone projections against the retarded electric-dipole field and its
magnetic dual, with midpoint-box grid refinement. It is an analytic reference
check rather than FDTD evidence. On the recorded CUDA run, exact E/H relative
L2 errors against the analytic fields were 1.70%/1.70%, 0.41%/0.41%, and
0.102%/0.102% at 12, 24, and 48 samples per side; far-field E errors were
0.907%, 0.218%, and 0.0539%. The separate
``yee_reference.py`` implements a
small periodic Torch Yee leapfrog pulse run and returns time-DFT box fields and
direct-observation fields. The direct FDTD notebook compares those outputs at
multiple mesh resolutions. Box-projected versus direct Yee fields gave E/H
relative L2 errors of 1.95%/6.41%, 1.36%/3.05%, and 0.95%/1.89% at 10, 15, and
20 cells per wavelength. At 30 and 40 cells per wavelength, E/H errors were
0.80%/1.11% and 0.75%/0.84% at 1.0 wavelength, and 0.79%/1.28% and
0.57%/0.93% at 1.5 wavelengths. The 40-cell comparison is below 1% for both
fields at both distances. These are convergence results for that independent
periodic reference configuration, not a validation of the MAPS/FDTDX wrapper.
This independent validator does not invoke MAPS/FDTDX. The zone-plate example
now supplies a recorded production MAPS near plane and a same-input Tidy3D
projection comparison. Comparing that projection with production MAPS fields
recorded directly at its distant observation plane remains a separate pending
validation.
