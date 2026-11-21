# Near-field acceleration

The acceleration API is additive to the reference projectors in
`core.nearfield.projector`. The direct Torch paths remain the correctness
reference and support arbitrary sampled surfaces. The accelerated paths have
explicit data contracts. `core.nearfield.dispatch.project` offers a calibrated
`auto` choice for exact near-field Torch/Triton calls; it uses only applicable
caller-supplied benchmark records and falls back to Torch when none match.

| Strategy | Entry point | Inputs | Result |
| --- | --- | --- | --- |
| `direct_torch` | `project_nearfield`, `project_farfield` | Any valid `Surface` | Exact near field or radiation-zone E/H |
| `separable_torch` | `project_farfield_separable` in `core.nearfield.accelerated` | Planar tensor-product `Surface`, explicit axes, source coordinates, and direction-cosine vectors | Far-field E/H on a tensor-product angular grid |
| `fft_torch` | `angular_spectrum_torch` in `core.nearfield.accelerated` | Uniform planar tangential E samples, spacing, frequency, hemisphere, and distance | Full vector E/H on the propagated plane |
| `direct_triton` | `project_nearfield_triton` in `core.nearfield.triton_backend` | CUDA float32 geometry and complex64 fields | Exact arbitrary-point near-field E/H |
| `auto` | `project` in `core.nearfield.dispatch` | Exact near-field call plus validated calibration records | Selects direct Torch or Triton; defaults to direct Torch without a matching record |

All strategies use SI meters and Hz, `exp(-i omega t)`, the outgoing Green
function `exp(+ikR)/(4 pi R)`, and outward surface currents
`J=n cross H`, `M=-n cross E`. Relative material parameters are finite,
positive, real scalars. Lossy or anisotropic media are not supported.

## Separable planar far field

`project_farfield_separable` accepts orthonormal in-plane axes `u_axis` and
`v_axis`, coordinates in meters, an origin, and independent direction-cosine
vectors. The `Surface` must match
`origin + v_coordinate*v_axis + u_coordinate*u_axis` in v-major/u-minor order;
its outward normal must match `normal_sign * (u_axis cross v_axis)`. The
requested direction-cosine rectangle must be inside the propagating unit disk.
Actual `Surface.weights_m2` are applied once. The Fourier integral is evaluated
with two matrix products, and results have shape `(F,Kv,Ku,3)` with u varying
fastest. Arbitrary flattened surfaces cannot use this path because their grid
geometry is not inferable from sample count.

## FFT angular spectrum

`angular_spectrum_torch` takes a one-sided tangential E field shaped
`(Nv,Nu,2)` or `(F,Nv,Nu,2)`. `Nu` and its spacing refer to the first
transverse Cartesian axis; `Nv` and its spacing refer to the second. The final
component axis follows those same increasing Cartesian-axis indices. PyTorch's
unshifted FFT frequencies set the represented band, bounded by the two sample
Nyquist frequencies. The API does not infer sub-Nyquist sampling adequacy or
undo aliasing. The DFT also treats the sampled plane as periodic with periods
`Nu*spacing_u_m` and `Nv*spacing_v_m`; it does not add zero padding. A finite
aperture therefore wraps around under propagation unless the caller pads the
field explicitly and checks convergence. Padding changes the sampled domain and
must be reflected in the spacing and coordinate interpretation.

For each FFT mode, the longitudinal square root uses the branch with
nonnegative real and imaginary parts. The normal wave number is multiplied by
`normal_sign`; positive `distance_m` is measured along that selected direction.
The phase uses the unsigned distance and square-root branch, so propagating
modes acquire the expected signed longitudinal wave vector in E/H
reconstruction and evanescent modes decay for either hemisphere. The
`evanescent` option explicitly chooses decay or discard. Grazing modes cannot
recover the normal E component from tangential E alone; nonzero modes within
`grazing_tolerance` are rejected.

Electric fields are reconstructed using `k dot E=0`; magnetic fields follow
`H=k cross E/(omega mu)`. This is a one-sided input contract: the API does not
split a measured plane into forward and backward waves. Fields are propagated
through native PyTorch FFT operations and remain differentiable with respect to
the complex field and scalar frequency/material inputs.

## Triton exact near field

`project_nearfield_triton` supports CUDA float32 geometry and complex64 fields.
The forward kernel reduces source tiles within observation tiles, fuses the
homogeneous dyadic E/H evaluation, and avoids a dense observation-by-source
tensor. It processes every frequency in the same launch grid. Triton remains
optional and an unsupported device or dtype raises an error.

The custom first-order backward recomputes the reference Torch kernel in source
and observation chunks. This preserves the native complex autograd convention
and bounds saved-kernel memory, but backward is not Triton-accelerated.
Higher-order gradients are unsupported. Runtime
claims must report forward and backward separately, including compilation and
warm-up costs. The `auto` dispatcher accepts records that match point counts,
dtype, device/software identity, chunk sizes, call count, and backward
requirements. It ignores records without enough repeats or acceptable
complex-field error and selects Torch when the evidence does not match.

## Observation helpers

`core.nearfield.observations` keeps target-grid construction and angular
quadrature explicit. `direction_cosines(u, v, normal_axis, normal_sign)`
returns unit directions with broadcast shape `u/v shape + (3,)`; values outside
the propagating unit disk are rejected. `cartesian_observation_plane` returns
`(Nu*Nv,3)` meter coordinates in `meshgrid(indexing="ij")` order, with the
second transverse coordinate varying fastest. `spherical_components` maps
complex Cartesian vectors to `(radial, theta, phi)` components, with theta
polar from +z and phi azimuth from +x toward +y. `integrated_directional_power`
requires caller-supplied solid-angle weights in steradians and returns power
per frequency in watts. These angular weights are distinct from source-surface
areas.

```python
u = torch.linspace(-0.4, 0.4, 41, dtype=surface.positions_m.dtype,
                   device=surface.positions_m.device)
v = torch.linspace(-0.4, 0.4, 41, dtype=u.dtype, device=u.device)
directions = direction_cosines(u[:, None], v[None, :])
electric, magnetic = project_farfield(surface, directions.reshape(-1, 3))
intensity = directional_power(electric).reshape(1, -1)
# Supply solid-angle weights matching this angular grid before integrating.
```

The symbols above are public exports of [`core.nearfield`](../../core/nearfield/__init__.py).
The [API guide](../api/nearfield.md#monitor-adapters-and-observation-helpers)
documents field shapes, meter coordinates, and the verified monitor adapter.

## Validation and performance reporting

Correctness gates include direct-versus-separable complex E/H comparison,
complex finite-difference gradients, analytic propagating and evanescent FFT
modes in both hemispheres, a paraxial Gaussian beam at one Rayleigh range, and
Triton-versus-Torch complex E/H and adjoint comparisons. Analytical aperture and
dipole convergence are tracked separately from backend agreement; matching the
Torch reference alone does not validate the physical projection convention.

Measured cold/warm backend records and their validation limits are summarized
in the `near-field benchmark report`.
Benchmarks should record input shape, dtype, device, accuracy against the
reference, first-call compilation, warm forward and backward times, and peak
device memory. FFT throughput must be reported with its sample spacing and
Nyquist passband. The Triton backward currently runs Torch recomputation, so it
must not be presented as an end-to-end Triton speedup.
