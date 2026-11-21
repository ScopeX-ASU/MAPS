# MAPS zone-plate projection example

This local workflow follows the physical settings in Flexcompute's
[ZonePlateFieldProjection example](https://www.flexcompute.com/tidy3d/examples/notebooks/ZonePlateFieldProjection/).
MAPS performs the time-domain simulation; `core.nearfield` performs the
finite-distance field integral. Optional Tidy3D comparisons run its local
projector on the **same MAPS near fields**. They do not submit cloud jobs and
are not independent Tidy3D FDTD simulations.

## Matched geometry and observations

| Quantity | Value |
| --- | --- |
| Free-space wavelength | 1 µm |
| Numerical aperture | 0.8 |
| Transverse simulation size | 20 × 20 µm |
| Nominal z extent | −1.85 to +1.85 µm |
| Silica substrate | n = 1.46, z = −0.35 to +0.15 µm |
| Titania zones | n = 2.40, z = +0.15 to +0.35 µm |
| Excitation | x-polarized plane wave toward +z, nominal source z = −1.1 µm |
| Near-field plane | Nominal z = +1.1 µm; actual recorded coordinate is exported |
| Projection plane | 40 × 40 points, x/y = −2 to +2 µm |
| Projection distance | 7.5 µm **relative to the near-plane origin** |

The ring boundaries satisfy
`r_n = sqrt(n * wavelength * focal_length + n² * wavelength² / 4)`.
The central first zone is air; succeeding zones alternate titania and air.
The overlapping-cylinder construction in the reference extends beyond the
square simulation window. There is no additional circular stop of radius 10 µm.

MAPS represents the substrate and every zone as native `geometry_cfgs` passed
to an `N_Ports` device. The cylinders are inserted from largest to smallest so
the inner zones override the outer material, exactly as in the reference. The
FDTD permittivity and grid metadata come from `device.epsilon_map` and
`device.fdtdx_field_grid_metadata`; there is no benchmark-owned NumPy geometry
rasterizer. The installed local Tidy3D license does not expose its local
subpixel evaluator, so MAPS uses its CuPy geometric volume-averaging fallback.
That backend supports cylinders along every Cartesian axis; this case retains
`subpixel=True` on the 50 nm uniform grid.

The fallback stores one scalar volume-averaged permittivity per FDTD cell. A
circle therefore still has a one-cell-wide, pixelated transition when the raw
solver array is plotted; subpixel averaging changes the values of intersected
cells rather than increasing the number of cells. The notebook overlays the
analytical MAPS cylinder boundaries and reports reflection/transposition
symmetry so a plotting stair step can be distinguished from a center offset.

The projection plane is at the recorded near-plane z plus 7.5 µm. This reproduces
the reference monitor's coordinate convention; measuring 7.5 µm from the lens
surface would produce a different plane. The high aperture makes the
radiation-zone approximation inappropriate here, so exact finite-distance E/H
are the primary result. The notebook also displays its discrepancy explicitly.

## Monitor transfer and numerical conventions

The near-field artifact records all six complex field components, physical
positions, normals, frequency and quadrature weights. MAPS' exact field
colocation targets the Ez Yee point `(i, j, k + 1/2)`: x and y use lower cell
edges, while z uses the cell center. These positions must not be replaced by
three cell-center coordinate arrays.

MAPS' native magnetic field is impedance normalized. For the SI surface
interface, its values are divided by the vacuum impedance. No second temporal
phase correction is applied to already-aligned exact monitor fields. The
phasors use `exp(-i omega t)`; the outgoing Green function uses `exp(+i k R)`.
The artifact metadata describes source-spectrum normalization and any source
amplitude calibration. Nodal trapezoidal weights are used for comparison with
the local Tidy3D projector.

Website images provide a qualitative visual reference. Their raw complex
simulation data are not used, so no quantitative cross-solver error is claimed.
Real-field panels depend on the source phasor reference. Squared field magnitude
and field magnitude are displayed separately, with their units, to make the
focusing pattern easy to compare.

## Reproducibility

The notebook is
``07_zone_plate_maps_projection.ipynb``.
Its simulation artifacts and rendered outputs remain local. The simulation
runner, geometry settings, projection outputs and JSON metadata are separate
so that an expensive time-domain simulation is not repeated merely to replot
fields or change observation points.

Use the requested Python environment and `CUDA_VISIBLE_DEVICES=0` for GPU
execution. Runtime reports distinguish simulation setup/compilation/execution
from projection costs. GPU allocation reports cover Torch allocations only
unless explicitly labeled otherwise; JAX/FDTD memory must not be inferred from
Torch's allocator counters.

## Recorded local result

The regenerated 50 nm run used GPU 0, an NVIDIA RTX PRO 6000 Blackwell Max-Q,
with PyTorch 2.9.1+cu128. It recorded 160,000 near-plane samples from a
400 × 400 × 74 MAPS grid. The maximum runtime was 1.3343 ps (13,996 steps);
the observed `solve()` call took 22.16 s including solver preparation,
compilation and time stepping; device materialization occurred before this
timer. The MAPS device reported the `local_cupy` geometry backend with
subpixel materialization enabled.
The source's spatial Ex profile was verified to have peak amplitude 1 and
energy normalization disabled. Source-spectrum normalization was enabled.

The actual Ez-point near plane was z = 1.075 µm, so the website-convention
7.5 µm projection distance placed the observation plane at z = 8.575 µm.
At the same 40 × 40 targets, complex128 Torch and the local CPU Tidy3D
projector agreed with relative complex-vector L2 errors of approximately
1.4137e-12 for E and 1.4142e-12 for H. No phase alignment or amplitude fitting
was applied. The notebook's refreshed projection calls took 1.53 s on GPU for
Torch and 73.34 s on CPU for Tidy3D; these are single observed calls on different
hardware, not a repeated warm benchmark.

At this plane the radial far-field approximation had relative complex E/H
errors of 1.536 and 1.519 against the exact integral. The notebook retains
these discrepancies. Its sampled transverse intensity-cut FWHM was about
0.755 µm. A separate 31-point axial scan found its largest sampled on-axis
intensity at an offset of 6.67 µm from the recorded plane; the website's
specified observation plane need not coincide with this maximum.

The regenerated FDTD process reported a Torch allocator peak of 1,129.17 MiB
and a JAX allocator peak of 2,421.94 MiB. These are separately measured allocator peaks,
not additive total-device memory. The notebook includes a distinct warm
forward benchmark for Torch and Triton with all 160,000 source samples and
nine targets, plus their errors against complex128 propagation.

These results establish a working production MAPS-to-projector path and
same-input projection agreement. The 50 nm uniform MAPS mesh and 0.5 µm
in-domain PML are explicit choices; they do not reproduce Tidy3D's automatic
material-dependent grid. No mesh-converged cross-solver simulation error or
quantitative error against the website images is claimed.

Local artifacts:

- `outputs/nearfield_zone_plate/maps_050nm.npz` and `.json`: unit-amplitude
  MAPS near fields and provenance.
- `outputs/nearfield_zone_plate/projected_050nm.npz`: complex native Torch
  exact and radial far-field outputs, with observation coordinates.
- `outputs/nearfield_zone_plate/backend_profile_050nm.json`: measured
  forward timing/memory and precision errors for the bounded benchmark.

The earlier `maps_smoke_100nm` and `maps_050nm_energy_normalized` artifacts
are development checks, not the final notebook inputs.
