# Physical source placement audit

This audit justifies the physical-placement changes in `core/fdtd/fdtd.py`.
It does not establish that a flatter field plot or a higher coupling efficiency
is more physically accurate. The contract being checked is the requested source
rectangle in physical coordinates on a rectilinear grid.

## Independent coordinate counterexample

Consider x edges, in micrometers:

```text
index:  0     1     2    3   4   5   6   7
edge:  -3   -2.5   -2   -1   0   1   2   3
```

A source centered at zero with width 4 must span `[-2, 2]`. Both endpoints
are grid edges, so there is no discretization ambiguity: its x slice is `2:6`.

The original `obtain_source_native_grid_slice()` returns `1:6`, spanning
`[-2.5, 2]`: width 4.5 and midpoint -0.25. The patched method returns `2:6`,
spanning exactly the independently specified physical rectangle.

The upstream physical-length conversion counts cells starting at the domain's
first edge, `edges[0] + length`. In this example, four micrometers starting
at -3 spans five cells. Reusing that five-cell count at the source location
cannot represent a four-micrometer rectangle there. On a uniform grid, a cell
count is independent of location; on a nonuniform grid it is not.

## Grating example coordinates

An initialization-only comparison used the same generated native grid and
source prototype for both methods. The original method was loaded from
`78f975364de166cedd97a3fe5961efb507dc545e:core/fdtd/fdtd.py`, without changing
the checkout. The current grating example supplied the geometry and auto-grid
configuration. The controlled waist was 10 micrometers; the waist does not
enter either method's aperture calculation. No optimization or FDTD time
integration was needed for this comparison.

The unreduced grid had shape `(272, 200, 122)`. The requested source size was
`(4, 4, 0)` micrometers, centered at approximately `(0, 0, 1.01)` micrometers.

| Quantity | Requested | Original placement | Patched placement |
| --- | --- | --- | --- |
| x bounds, micrometers | [-2, 2] | [-1.477625, 1.482500] | [-2, 2.024675] |
| y bounds, micrometers | [-2, 2] | [-1.728873, 1.728873] | [-2, 2] |
| x width, micrometers | 4 | 2.960125 | 4.024675 |
| y width, micrometers | 4 | 3.457746 | 4 |
| transverse area, square micrometers | 16 | 10.235362 | 16.098701 |
| x native slice | grid dependent | 55:174 | 34:196 |
| y native slice | grid dependent | 29:171 | 18:182 |
| z native slice | one cell at source plane | 85:86 | 85:86 |

The original aperture misses about 36.03% of the requested area. The patched
aperture covers the rectangle with about 0.617% excess area due to whole-cell
coverage at the x boundary. This is a documented discretization choice: the
patch includes boundary-overlapping cells rather than applying a fractional
source mask. It is not an exactly four-micrometer continuous hard aperture.
Floating-point edge alignment is handled by the helper's relative tolerance.

## Separate tolerance error

For edges `[-1, -0.995, 0, 1]` micrometers and requested lower bound
`-0.999` micrometers, the first cell overlaps the rectangle by four nanometers.
The original `np.isclose()` uses a default absolute tolerance of `1e-8` in
the supplied coordinate units. In meters that is ten nanometers, so it treats
the next edge as aligned and drops the overlapping cell. The patched relative
tolerance retains it.

The regression checks the same geometry at coordinate scales `1`, `1e-6`, and
`1e-9`. Unit conversion must not change the selected cell indices.

## Validation and scope

The initial six-case placement suite (now expanded in
`tests/fdtdx/test_source_placement.py`) were run against the committed
and patched helper implementations in separate processes:

| Implementation | Passed | Failed |
| --- | --- | --- |
| Original committed helpers | 3 | 3 |
| Patched helpers | 6 | 0 |

Both implementations pass the two aligned uniform-grid source controls and
the unit-scale overlap case. The original fails the nonuniform coordinate
counterexample and both small-coordinate overlap cases. The patched tests also
place the explicitly shaped source through FDTD-X and verify that placement
retains the specified native bounds without mutating the prototype.

The GPU integration regression
`test_nonuniform_gaussian_source_and_recorder_cover_physical_aperture` checks
actual placed source coverage, exact source/recorder slice agreement, the applied
Gaussian profile against its analytical expression, and finite parameter
gradients after a forward/adjoint solve. Together with the two existing variation
gradient regressions, it passed all three tests. Finite gradients are a runtime
check, not an independent proof of adjoint accuracy.

Use the configured Python with `CUDA_VISIBLE_DEVICES=0` and repository
`PYTHONPATH`. The focused placement command is:

```bash
python -m pytest -q tests/fdtdx/test_source_placement.py
```

Changing the aperture changes the Maxwell problem being simulated. Its
physical-area power normalization also changes the illumination amplitude.
Consequently, agreement with the original fields or FoM is not an appropriate
acceptance criterion for this placement correction. The evidence supports the
physical-coordinate contract; it does not prove absolute FDTD accuracy,
convergence of the coupling efficiency, uniform evolved fields, or reproduction
of an older figure produced with a different configuration/runtime.

## Upstream FDTD-X test coverage and API distinction

A follow-up checked the installed FDTD-X checkout at
`77b4bc523a8e98cb7dd388e99481a0000f71dd4d`. Its six nonuniform initialization
tests in `tests/unit/fdtd/test_initialization.py` pass. In particular,
`test_nonuniform_partial_real_shape_covers_metric_size` places the physically
sized object at the domain origin, where the length-to-cell conversion is
applicable. The physical-center test supplies an explicit `partial_grid_shape`;
it does not test physical-size conversion combined with center placement.

The nonuniform Gaussian profile tests in
`tests/unit/objects/sources/test_linear_polarization.py`, and the nonuniform mode
coordinate test in `tests/unit/objects/sources/test_mode.py`, likewise supply
explicit cell shapes and call `place_on_grid()` with known slices. The example
`examples/simulate_gaussian_source.py` uses `place_relative_to()` with physical
source dimensions, but its simulation grid is uniform. These tests/examples
therefore do not resolve the demonstrated combination of physical-size
conversion and placement away from the domain origin on a stretched grid.

A direct reproduction importing only upstream `fdtdx`, without MAPS overrides,
used the coordinate-counterexample grid above and a `ModePlaneSource` centered
with the same `place_relative_to(volume, axes=(0, 1, 2), own_positions=(0, 0, 0),
other_positions=(0, 0, 0))` constraint in both cases:

| Source definition | Resolved x slice | Physical x width |
| --- | --- | --- |
| `partial_real_shape=(4e-6, 4e-6, None)`, normal grid size 1 | 1:6 | 4.5 micrometers |
| Independently determined `partial_grid_shape=(4, 4, 1)` | 2:6 | 4 micrometers |

Both cases report no placement error. Thus `place_relative_to()` can still be
used once the source dimensions are resolved correctly. The defect is in the
location-independent physical-length-to-cell-count conversion, not a universal
failure of relative-position constraints. Resolving both physical endpoints
before placement is one valid correction; fixing the upstream size resolver
would be another. The initial MAPS Gaussian-only guard was a scoped workaround
and left physically sized mode-plane sources exposed to the same defect. The
generalized implementation below removes this limitation.

## SiN comparison: a substantially smaller aperture correction

The 4-by-4-micrometer aperture measurements above must not be extrapolated to
the 16-by-16-micrometer SiN example. A separate initialization-only audit of
`grating_coupler_SiN_3d.py`, with waist radius 5.2 micrometers, reconstructed
native grid shape `(723, 708, 105)` and requested source center approximately
`(0, 0, 1.9125)` micrometers. The same original and patched methods gave:

| Quantity | Original placement | Patched placement |
| --- | --- | --- |
| x native slice | 52:655 | 44:664 |
| y native slice | 51:656 | 44:665 |
| z native slice | 87:88 | 87:88 |
| x bounds, micrometers | [-7.793139, 7.798997] | [-8, 8.025833] |
| y bounds, micrometers | [-7.818966, 7.793103] | [-8, 8.025816] |
| x width, micrometers | 15.592136 | 16.025833 |
| y width, micrometers | 15.612069 | 16.025816 |

The iteration-zero PNGs in `figs/grating_coupler_SiN_1.55_16x16_override` and
`figs/grating_coupler_SiN_1.55_16x16_override2` show similar central beams.
Those folders contain figures rather than archived source/grid arrays or run
configurations, so the dimensions above describe the current configuration's
controlled reconstruction, not recovered metadata from each historical run.

For a centered, untilted waist-plane Gaussian with radius `w`, incident intensity
is proportional to `exp(-2*(x*x + y*y)/(w*w))`. Its analytical power fraction
inside a rectangle is the product, over x and y, of
`(erf(sqrt(2)*upper/w) - erf(sqrt(2)*lower/w))/2`. For the reconstructed bounds
and `w=5.2` micrometers, the original rectangle contains about 99.4614% of the
infinite-plane Gaussian power and the patched rectangle about 99.5890%.
The newly included tails therefore account for about 0.1281% of the patched
incident power. With equal incident power, this envelope model predicts only
about a 0.0641% decrease in peak amplitude from the aperture expansion.

This calculation explains why the central illumination can look nearly
unchanged despite a real placement correction. It is an analytical envelope
estimate, not a measured FDTD field difference or coupling-efficiency change.

## Generalized physical-source implementation

Placement now selects the physical-resolution path from the source shape
metadata rather than a Gaussian class check. Each axis with a physical size
and no explicit cell count is covered using its requested physical endpoints.
Grid-sized axes preserve their count and choose the interval nearest the
requested physical center. Unspecified axes inherit the full native domain.
This also supports volumetric sources, without forcing a plane-normal axis.

Explicit native counts take precedence when both grid and physical sizes are
stored. Fully native sources, including distributed adjoint currents, retain
the prior placement path, as do uniform grids. During nonuniform physical-source
preparation, both shape and edge constraints preserve the resolved bounds.
Sparse input recorders use those same bounds for every physically sized source.

The expanded plane regression covers GaussianPlaneSourceTidy3d, upstream
GaussianPlaneSource, UniformPlaneSource, ModePlaneSource, and
HardConstantAmplitudePlanceSource on each normal axis and both propagation
directions. Before this generalization, 24 of these cases failed; after the
change all 30 plane cases passed. Additional checks cover aligned uniform
sources, mixed physical/native sizes, a volumetric TFSF source, and a distributed
native adjoint source with redundant physical metadata.

`tests/integration/invdes/test_mode_source_placement.py` exercises the production
MAPS preparation path with a physically sized ModePlaneSource on a stretched
waveguide grid. It checks independently specified aperture endpoints, applied
mode-profile dimensions, matching recorder bounds, and unmodified prototypes,
then runs a forward simulation with the computed guided mode. This is a
placement and execution regression, not a convergence study.

The mode integration exposed roundoff at an aligned upper edge that could
include an unnecessary following cell. Both aperture endpoints now use the
same relative edge-alignment tolerance. The source recorder also clears its
old physical-center constraint after its native bounds have been fixed, avoiding
a competing center-based placement of an asymmetrically snapped aperture.

With this final alignment correction, the saved SiN grid above resolves x to
`44:663` and y to `44:664`, both spanning `[-8, 8]` micrometers to roundoff.
The earlier SiN table and power estimate describe the initial patch before
this upper-edge correction.

Final validation passed 69 focused checks (including the actual guided-mode
integration), and all three Gaussian/variation GPU regressions. The runs used
the requested Python environment and `CUDA_VISIBLE_DEVICES=0`; focused checks
used CPU JAX and the GPU regressions used CUDA. Formatting, test-file lint,
and whitespace checks passed.

## Detector follow-up

The size conversion is shared by FDTD-X simulation objects, so the source
counterexample also applies to detectors. On the independently specified x
edges `[-3, -2.5, -2, -1, 0, 1, 2, 3]` micrometers, a physical detector width
of 4 micrometers centered at zero requests `[-2, 2]`. Direct placement of both
PhasorDetector and ModeOverlapDetector instead produces x slice `1:6`, spanning
`[-2.5, 2]`, or 4.5 micrometers. This reproduction uses `apply_objects=False`;
it establishes a geometric error without relying on a field plot or mode solve.

MAPS native port registration and native mode-overlap profile construction
previously used that path. Detector placement now computes the native bounds
at the physical location, fixes the cell shape and both edge coordinates, and
clears the competing physical-center constraint on a clone. Explicit recorder
supports override physical metadata. This also fixes sparse detectors losing
multi-cell native axes or discarding all physical extents when one position
coordinate is unspecified. Dense auxiliary recorders use the same resolver.
Full-volume recording retains its full-domain support, and uniform-grid port
registration and mode solves retain their original paths.

Before the change, all four baseline cases in
`tests/fdtdx/test_detector_placement.py` failed. Coverage now includes five
detector families on all three plane orientations, native port registration
with and without symmetry, physical volume detectors, mixed native/physical
metadata, explicit support precedence, prototype preservation, and invalid
support rejection. `tests/integration/invdes/test_mode_detector_placement.py`
computes a guided mode on independently specified transverse endpoints and
checks physical and explicit auxiliary supports in both dense and sparse
production preparation.

This is a scientific behavior correction: affected modal cross-sections,
profiles, overlaps, and recorded field coverage can change. Interpolation,
flux integration, overlap formulas, and source injection are not modified by
this detector follow-up. The tests establish placement and execution contracts,
not a discretization convergence claim.

Detector follow-up validation passed 96 focused CPU checks, including the
source and sparse-scatter regressions, and all three CUDA forward/backward and
variation checks. These used the requested Python environment with
`CUDA_VISIBLE_DEVICES=0`. Black, test-file Ruff, and whitespace checks passed.
