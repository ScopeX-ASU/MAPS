# Three-dimensional electromagnetic projection

## Plan and correctness gates

The implementation is additive in `core/nearfield`, independent of external
solver packages. Existing two-dimensional FDFD projectors and monitor APIs keep
their behavior. Work proceeds through baseline Torch implementation, independent
complex-field/gradient verification, acceleration, then end-to-end validation.

1. Establish explicit near-field positions, outward normals, physical area weights,
   frequency order, medium, and common-time colocated complex E/H fields.
2. Implement source/observation-chunked exact and far-field integrals.
3. Gate acceleration on analytical electric/magnetic dipoles, complex E/H phase
   and normalization, interior cancellation, and directional-derivative checks.
4. Add planar separable Fourier and nonparaxial FFT propagation; compare forwards
   and gradients with independent baselines before GPU optimization.
5. Record fidelity and runtime separately, including first-call overhead, warm
   forward/backward, peak GPU memory and complex64/complex128 discrepancies.
6. Validate recorded MAPS near fields against independently recorded direct fields
   over mesh/distance/surface refinements. Passing analytical or projector
   cross-comparisons does not satisfy this full-wave gate.

## Mathematical convention

Use SI meters, Hz, V/m, A/m, absolute permittivity epsilon and permeability mu,
with real positive isotropic homogeneous background. Physical fields are
`Re(F exp(-i omega t))`; outgoing `G=exp(i k R)/(4 pi R)`, `k=omega sqrt(mu epsilon)`.
For oriented samples, `J=n cross H`, `M=-n cross E`. Area enters once.

Maxwell equations with equivalent currents are
`curl E=i omega mu H-M`, `curl H=J-i omega epsilon E`. They give

```
E = integral [i omega mu (G I + Hess(G)/k^2) J - grad(G) cross M] dA
H = integral [grad(G) cross J + i omega epsilon (G I + Hess(G)/k^2) M] dA.
```

Writing `q=kR`, `u=(r-r')/R`, the dyadic acting on a vector v is
`G[(1+i/q-1/q^2)v+(-1-3i/q+3/q^2)u(u dot v)]` and
`grad(G)=G(ik-1/R)u`. The source singularity is excluded; these are ordinary
exterior integrals, without a principal-value self term. A closed outward
surface reproduces fields outside and cancels fields inside when it encloses
all sources. A finite open plane is an aperture equivalence approximation;
truncation error must be converged rather than amplitude-normalized away.

For far-field directions s, integrate J and M with `exp(-ik s dot r')` to
obtain N and L, then
`E=ik exp(ikr)/(4 pi r)[eta(N-s(s dot N))-s cross L]`,
`H=s cross E/eta`, with `eta=sqrt(mu/epsilon)`. The common origin matters for
complex phase. Spherical theta is polar angle from +z, phi azimuth from +x
about +z. Radiation intensity is `r^2/2 Re(E cross conj(H)) dot s` in W/sr;
solid-angle quadrature is explicit and separate from source area quadrature.

## MAPS monitor audit

The existing MAPS-facing FDTD solve returns `(Ex,Ey,Ez,Hx,Hy,Hz)` with each
component `[frequency,x,y,z]`; combined shape is `[frequency,6,x,y,z]`.
`last_solve_frequencies_hz` and `last_solve_field_grid` record actual ordering
and native/export selection. Source demodulation in `core/fdtd/solver.py`
uses `exp(+i omega t)`. Source-spectrum normalization does not undo finite-time
spectral leakage.

Nonexact H output receives `exp(i omega dt/2) cos(omega dt/2)` before spatial
colocation. Exact and sparse output already includes its observation map.
The projection adapter must not repeat this timing correction. Native field
interpolation in `core/utils.py::interpolate_fields` explicitly targets the
Ez Yee point `(i,j,k+1/2)`, whereas some visualization documentation calls
outputs cell centers. Explicit independently verified coordinate arrays are
therefore required; shape matching is insufficient evidence of registration.

MAPS grid metadata uses micrometers and surface-power utilities convert widths
to meters before forming square-meter areas. Do not infer physical area from
singleton axes or normalize weights by their mean. Dense sparse-monitor output
contains zero-filled unrecorded fields: coverage must be supplied explicitly.
The surface adapter requires declared colocated, time-aligned, unfolded data;
raw Yee interpolation and E/H timing belong to the existing observation map.
Torch indexing preserves that map's existing autograd pullback.

A six-face box uses explicit outward signs on each face. An unsigned monitor
axis alone does not establish orientation. Shared edges have zero measure in
continuous integrals; each face's trapezoidal endpoint weights represent its
own area, and no extra corner volume weight or symmetry multiplier belongs
in the surface sum.

## Tidy3D inspection and attribution

Inspected the public `develop` field-projection package and installed Tidy3D
2.12.1. The inspected upstream develop revision was
`0573a453a33fc0da611f53b5c29550133d945aaa` (2026-10-09). The upstream implementation separates surface preparation, exact dyadic
projection, angular tensor grids, paired Cartesian/k-space observations, and
result assembly. It supports `FieldProjector.from_near_field_data` for custom
`FieldData`, permitting identical-input comparisons without a cloud run.

Useful techniques are explicit element-current area weighting, shared geometry
for tangential components, frequency/observation chunking, separable phases,
and recomputation in adjoints. Its optional staggered integration maintains
component-specific Yee coordinates/areas; colocated inputs cannot recover those
after recording. MAPS initially accepts a verified colocated representation.
Resampling must be disabled and quadrature matched in identical-input tests.
Tidy3D coordinates are micrometers; the Torch API uses meters. Match background
impedance and all six complex components, not only normalized intensity.

Sources: [projector](https://github.com/flexcompute/tidy3d/blob/develop/tidy3d/components/field_projection/projector.py),
[exact kernel](https://github.com/flexcompute/tidy3d/blob/develop/tidy3d/components/field_projection/exact.py),
[shared Fourier helpers](https://github.com/flexcompute/tidy3d/blob/develop/tidy3d/components/field_projection/common.py),
[API documentation](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.FieldProjector.html).
Upstream is LGPL-2.1; these equations are derived above from Maxwell equations.
The MAPS implementation is independently written in PyTorch, with no copied
upstream implementation. Tidy3D is an optional validation dependency only.
No FDTDX source is used as a projection reference.
