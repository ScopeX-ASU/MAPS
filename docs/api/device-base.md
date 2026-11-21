# Device Base Architecture and API Contract

This document records the current design principles and public-facing contracts of `core/invdes/models/layers/device_base.py`.

This file is intentionally **descriptive rather than prescriptive**. `device_base.py` is a large orchestration module containing geometry construction, rasterization, optical and thermal grids, design-region masks, ports, monitors, source/mode generation, normalization runs, solver creation, and multiphysics material maps. The current implementation should be understood as the source of truth until a later refactor deliberately separates these responsibilities.

## 1. Mental model

The central object is `N_Ports`, a device description that connects physical geometry to simulation-ready arrays and port/source/monitor metadata.

The intended flow is:

```text
Physical configuration
    |
    +-- port_cfgs
    +-- geometry_cfgs
    +-- design_region_cfgs
    +-- active_region_cfgs
    +-- heat_source_cfgs
    +-- sim_cfg
    |
    v
Geometry backend representation
    |
    +-- Meep geometry
    +-- Tidy3D structures
    |
    v
Native optical raster grid
    |
    +-- epsilon map
    +-- native coordinates / boundaries
    +-- native design-region masks
    +-- native port/monitor slices
    |
    +--------------------+
    |                    |
    v                    v
Export optical grid     Thermal grid / maps
    |                    |
    |                    +-- conductivity
    |                    +-- heat capacity
    |                    +-- thermo-optic coefficient
    |                    +-- electrical conductivity
    |
    v
Port/monitor/source infrastructure
    |
    +-- source mode profiles
    +-- field monitors
    +-- flux monitors
    +-- near-to-far monitors
    |
    v
Simulation backend
    |
    +-- FDFD / Ceviche
    +-- FDTD-X
    |
    v
Objective / inverse design
```

The important principle is that **physical geometry and physical coordinates are authoritative; tensor indices are derived representations**.

## 2. `N_Ports` constructor

```python
N_Ports(
    eps_bg=...,
    k_bg=None,
    electrical_conductivity_bg=None,
    heat_capacity_bg=None,
    thermo_optic_coeff_bg=None,
    port_cfgs=...,
    geometry_cfgs=...,
    design_region_cfgs=...,
    heat_source_cfgs=...,
    active_region_cfgs=...,
    sim_cfg=...,
    grid=None,
    grid_info=None,
    device="cuda:0",
    verbose=True,
)
```

### Core configuration dictionaries

`port_cfgs`
: Mapping of named physical ports. A port configuration normally includes `type`, propagation `direction`, `center`, `size`, and optical material `eps`. The port geometry also determines the bounding region used to construct port masks and monitors.

`geometry_cfgs`
: Non-port geometry. Supported configurations include simple and compound geometry; some compound forms require Tidy3D. A configuration with `override_structure=True` is excluded from physical geometry and used as a mesh override structure instead. A false or absent flag preserves the ordinary-geometry behavior. This supports non-physical helper regions, such as a dummy low-index substrate used to control automatic meshing, without adding another top-level configuration mapping.

`design_region_cfgs`
: Named regions intended for inverse-design parameterization. These regions are rasterized differently from ordinary geometry: the code deliberately avoids adding them as ordinary geometry in order not to apply backend subpixel smoothing directly to their boundaries. Design-region information is instead carried through masks/override structures.

`active_region_cfgs`
: Subregions inside design regions used for active material modulation, such as thermal/electrical control.

`heat_source_cfgs`
: Configuration for thermal source maps used by the heat solver.

`sim_cfg`
: Simulation settings including `resolution`, `PML`, optional `cell_size`, wavelength settings, plot root, symmetry, backend/grid controls, and thermal solver configuration.

## 3. Dimensionality and units

`N_Ports` infers whether the device is 2D or 3D from configuration. The helper examines `cell_size`, geometry `center`/`size`, vertices, and finite `height` values.

The canonical physical coordinate convention is centered around the simulation cell. Geometry and monitor centers/sizes are expressed in the device's physical coordinate system, with optical workflows conventionally using microns.

The optical grid spacing defaults to:

```text
grid_step = 1 / resolution
```

with the optical coordinate convention used throughout MAPS. Conversion to SI occurs explicitly when creating FDTD-X objects through `MICRON_UNIT`.

Do not infer tensor axis order from generic image conventions. In MAPS, the axes represent physical x/y(/z) coordinates.

## 4. Geometry representation

### Supported conceptual geometry families

The implementation supports at least:

```text
box
prism
cylinder
sine_bend
compound/group
clip / boolean operations
```

Tidy3D-specific conversion supports additional compound operations such as geometry groups and clip/boolean combinations.

### Material values

Geometry configs use a value/background convention such as:

```text
eps   = structure permittivity
eps_bg = surrounding permittivity
```

The Tidy3D conversion path converts complex permittivity into an appropriate `Medium`, including complex refractive index when needed.

### Sidewall angle

Public sidewall angles are interpreted in degrees and converted to radians for Tidy3D geometric primitives.

### Roughness

When enabled, line-edge roughness can be applied to geometry configuration before rasterization. Geometry is first rebuilt without roughness to establish the stable cell bounds, then rebuilt with roughness so stochastic/perturbed boundaries do not alter the simulation-cell extent.

This ordering is an important architectural invariant.

## 5. Simulation-cell determination

If `sim_cfg["cell_size"]` is not explicitly specified, `get_geometry_box()` computes the bounding box of:

- design regions;
- ordinary geometry;
- port geometry;

and expands it using `border_width`.

The resulting values are:

```text
cell_size
cell_extend = ((xmin, xmax), (ymin, ymax), ...)
cell_center
```

If a native grid is supplied, its physical boundaries become authoritative for the cell extent and size.

The geometry is then recentered relative to the established cell bounds.

## 6. Native versus export optical grids

A central design principle is the distinction between:

### Native optical grid

The grid on which the primary optical material map is rasterized and on which the solver's physical discretization is based.

Stored metadata includes:

```text
coords
boundaries
shape
backend
grid
```

and is available through entries such as:

```python
self.grid_info_dict["epsilon_map"]
```

### Export optical grid

A potentially different, user-facing grid used for plotting, monitor presentation, or downstream field exchange.

The public convenience map is:

```python
self.epsilon_map
```

while the implementation also retains:

```python
self.native_epsilon_map
self.export_epsilon_map
```

When native and export coordinates differ, fields/maps are resampled between the two coordinate systems rather than simply reshaped.

### Why this distinction matters

Two tensors with the same shape may represent different physical sampling. Conversely, native and export tensors may have different shapes but represent the same physical region. Geometry/objective code must use coordinate metadata when moving between these representations.

## 7. Rasterization contract

`rasterize_geometry_map()` is the generic geometry-to-array bridge.

Conceptually:

```python
raster_map, grid_info = rasterize_geometry_map(
    cell_size=...,
    geometry=...,
    PML=...,
    resolution=...,
    background_value=...,
    ...,
)
```

Important parameters include:

`background_value`
: Value outside explicit geometry.

`spacing`
: Explicit per-axis spacing for nonuniform or prescribed grids.

`grid_spec_cfg`
: Backend-specific grid specification.

`map_name`
: Name under which grid metadata is recorded.

`subpixel`
: Enables backend/local subpixel volume averaging where supported.

### Optimized-region subpixel materialization

Fixed geometry and optimized regions use the same scalar mixing rule:
`epsilon_cell = sum(volume_fraction * epsilon_material)`. Optimization density
is not a geometric fraction when `denorm_mode` is nonlinear. The materialization
path therefore converts projected fine-grid density to physical permittivity
(equivalently, arithmetic epsilon fractions) **before** lateral resampling or
vertical partial-cell coverage. It encodes the result back into the configured
density coordinate only to retain existing interfaces. `inverse_eps` and
`linear_index` remain supported; they no longer reinterpret boundary coverage.

When physical grid edges are available, native and HR remapping averages
piecewise-constant source cells by their overlap with target cells, including
nonuniform grids. Source-cell interior edges are coordinate midpoints; outer
edges are the physical design-region bounds. Boundary-cell averages are
normalized over the region overlap: lateral coverage outside that overlap is
applied once during insertion. Uniform shape-only design materialization likewise
uses cell overlap for noninteger ratios. The standalone legacy
`convert_resolution_tensor` utility retains its established pooling contract;
design materialization now uses a separate conservative averaging routine.
Coordinate-only legacy metadata falls back to interpolation in arithmetic
epsilon space, not a guarantee of conservative volume integration.

Vertical coverage blends with the design region's background material, not the
already cell-averaged fixed port map. This prevents counting port boundary
coverage twice at their junction. Nonzero-sidewall halo handling retains the
existing fixed-context morphology and union; it is an approximate raster
construction, not exact polyhedral volume integration. Its existing z-averaged
gradient scaling is preserved.

This is an intentional scientific behavior correction for nonlinear
parametrizations and previously nonconservative remapping. Optimized trajectories
can change; this does not establish end-to-end WDM performance improvement.
Regression coverage includes an all-solid WDM port/design junction on the actual
GPU raster grid, fractional cells, nonuniform overlaps, and gradient checks in
`tests/gradient/test_subpixel_materialization.py`.

`override_structures`
: Structures used to override base geometry/material behavior for design-region rasterization.

### Backend behavior

The baseline can rasterize through Tidy3D when requested/required. If Tidy3D subpixel rasterization fails, the code can fall back to a local GPU/CuPy geometric volume-averaging implementation.

This fallback is intentional numerical behavior, not merely a performance optimization.

## 8. Grid metadata contract

Each recorded raster grid is represented by a dictionary with fields such as:

```python
{
    "name": str,
    "backend": str,
    "grid": backend_grid_object_or_None,
    "coords": tuple[np.ndarray, ...],
    "boundaries": tuple[np.ndarray, ...],
    "shape": tuple[int, ...],
}
```

`coords[axis]` contains cell-center coordinates.

`boundaries[axis]` contains cell-edge coordinates and therefore normally has length:

```text
len(boundaries[axis]) == len(coords[axis]) + 1
```

This relationship is used by physical-coverage and nonuniform-grid logic.

## 9. Design-region masks

After the epsilon map and optical grid are established, the device constructs:

```text
design_region_masks
```

Each named region maps to a slice/index object describing its location in the optical grid.

Additional coverage metadata may be present:

```text
design_region_mask_weights
design_region_axis_weights
```

These weights are used later for partial-cell/partial-axis blending.

### Active regions

`build_active_region_mask()` requires each active region to be contained entirely inside its corresponding design region. Out-of-bounds active regions raise an assertion error.

The active-region mask is therefore not an independent geometry universe; it is explicitly a subregion of an existing design region.

## 10. Port-region masks

`build_port_region()` creates a boolean mask of all port regions on the optical grid.

The port mask is used by workflows such as the radiation monitor to exclude port areas from radiation integration.

A port's normal axis has a width factor of 1, while transverse axes use the configured `rel_width` expansion when constructing this mask.

## 11. Monitor-slice contract

### `add_monitor_slice()`

```python
add_monitor_slice(
    slice_name,
    center,
    size,
    direction=None,
)
```

Input semantics:

`center`
: Physical center in the device coordinate system.

`size`
: Physical monitor size. Exactly one axis must have zero extent for a codimension-1 monitor surface.

`direction`
: Propagation/normal direction, such as `x+`, `x-`, `y+`, or `z-`. The axis is derived from the first character.

### Index representation

MAPS intentionally maintains multiple slice variants:

```text
port_monitor_slices
port_monitor_slices_native
port_monitor_slices_export
port_monitor_slices_native_symmetry
```

The distinctions are:

- `port_monitor_slices`: public/default monitor indexing, generally export-grid based;
- `port_monitor_slices_native`: native optical-grid indexing;
- `port_monitor_slices_export`: explicit export-grid indexing;
- `port_monitor_slices_native_symmetry`: reduced native slice before symmetry unfolding, used by FDTD-X adjoint grouping.

This is not redundant state. The simulation backend, visualization path, and adjoint path can require different index representations of the same physical monitor.

### Slice metadata

Each monitor also has:

```python
port_monitor_slices_info[slice_name] = {
    "center": [...],
    "size": [...],
    "direction": "x+",
    "xs": ...,
    "ys": ...,
    # and z information in 3D paths
}
```

`xs`/`ys`/`zs` represent physical coordinate samples associated with the monitor, not integer tensor indices.

## 12. Monitor construction helpers

### `build_port_monitor_slice()`

Derives a monitor plane relative to a configured port using:

```text
rel_loc
rel_width
rel_height (3D)
```

The resulting physical monitor is passed to `add_monitor_slice()`.

### Near-to-far

`build_near2far_slice()` constructs a monitor and warns when the extracted epsilon region is not homogeneous. The near-to-far method assumes a suitable surrounding medium for the Green-function transformation.

### Radiation monitor

`build_radiation_monitor()` builds four codimension-1 monitor slices around the PML region:

```text
xp
xm
yp
ym
```

It additionally removes cells intersecting the physical port regions so that source/port transmission is not double-counted as radiation.

### Far-field region

`build_farfield_region()` and `build_farfield_region_ext()` create rectangular monitor regions and record physical coordinate arrays in `port_monitor_slices_info`.

These are consumed by near-to-far and far-field objective code rather than functioning as ordinary source ports.

## 13. Port/source/mode model

The source infrastructure distinguishes:

```text
port geometry
    -> monitor slice
    -> source mode definition
    -> normalized source profile
    -> simulation source
```

A source profile is stored in:

```python
self.port_sources_dict[slice_name]
```

For the common optical FDFD path, an entry has the conceptual schema:

```text
(wavelength, mode) ->
    [source, H_mode, E_mode, power_scale, require_sim]
```

For the FDTD-X broadband path, the wavelength key becomes:

```text
((wl_cen, wl_width, n_wl), mode)
```

and the fifth element is used to indicate whether a real forward simulation must be run.

For example:

```python
port_sources_dict["in_slice_1"][(1.55, "Ez1")]
```

versus the broadband FDTD-X representation:

```python
port_sources_dict["in_slice_1"][((1.55, 0.02, 5), "Ez1")]
```

## 14. Mode naming convention

The current code interprets mode identifiers such as:

```text
Ez1
Ez2
Hz1
Hz2
```

The first two characters identify polarization:

```text
Ez -> TM-like 2D field representation
Hz -> TE-like 2D field representation
```

The numeric suffix is converted to a zero-based mode index for FDTD-X mode sources.

Do not assume that `Ez1` and `Hz1` are simply Cartesian field-component labels. Within the source API they are also mode/polarization selectors.

## 15. 2D mode insertion

`insert_modes()` performs a 2D eigenmode solve for each requested wavelength and mode.

The input physical data include:

```text
eps             -> 2D permittivity map
slice           -> source/monitor slice
wl_cen
wl_width
n_wl
source_modes
 direction
grid_step
dxs
dys
```

For each wavelength, it computes:

```text
omega = 2*pi*c / wavelength
```

with the implementation converting the wavelength to SI through `MICRON_UNIT`.

The mode solver returns electric/magnetic modal profiles which are retained in the source profile tuple.

For nonuniform optical grids, `dxs` and `dys` are derived from the epsilon-map boundary coordinates when they are not explicitly supplied.

## 16. FDTD-X mode insertion

`insert_modes_fdtd3d()` uses an FDTD-X `ModePlaneSource` with the center-frequency
profile by default. Select the low-order broadband profile approximation with
`sim_cfg["fdtdx_mode_source_method"] = "chebyshev_3"` or `"chebyshev_5"`.
The method suffix is the number of Chebyshev mode solves; no separate source
frequency-count setting is used. The existing `wl_cen`, `wl_width`, and `n_wl`
arguments continue to define the requested simulation wavelengths. The previous
exact-isolation `BroadbandModePlaneSource` is diagnostic-only and can be selected
with `sim_cfg["fdtdx_mode_source_method"] = "exact"`. The source construction is keyed by:

```text
(wl_cen, wl_width, n_wl)
```

is stored as one wavelength-group key, while the actual sampled wavelengths are generated internally by:

```text
linspace(wl_cen - wl_width/2,
         wl_cen + wl_width/2,
         n_wl)
```

For a mode such as `Ez1`, the FDTD-X source uses:

```text
mode_index = int("Ez1"[2:]) - 1
filter_pol = "tm"
```

A symmetry scaling factor of `sqrt(2)` per active symmetry axis is applied to the source amplitude.

The Chebyshev source solves the mode at first-kind Chebyshev nodes across
plus/minus 1.5 Gaussian spectral standard deviations. It phase-aligns the
sampled modes, transforms them into low-order Chebyshev coefficients, and
injects those coefficients with compact analytic Gaussian-derivative temporal
basis functions. The profile sampling is independent of the requested monitor
or objective wavelengths. It currently requires nondispersive,
nonconductive source media because the FDTD-X magnetic-source correction would
otherwise need to be applied independently to every temporal basis function.

For the experimental exact multi-frequency case, each sampled wavelength has its own eigenmode
solve. The source calibrates one real, zero-DC finite-duration waveform per
mode against the detector's discrete phasor convention. At a requested
frequency, the waveform for that frequency reproduces the original Gaussian
source spectrum while the other mode waveforms have zero response. All mode
components are injected in one forward FDTD run. The stored waveforms are
oversampled relative to the FDTD time step so fractional Yee time offsets do
not introduce a systematic amplitude error.

Temporal synthesis reports its condition number, stored-dtype reconstruction
error, waveform peak amplification, and the number of adjacent-frequency beat
cycles covered by the simulation. Large peak amplification or less than one
beat cycle warns that the configured duration is short for the requested
frequency spacing. A condition number above the configured safety limit fails
before FDTD, because an exact float64 synthesis residual does not guarantee a
reliable time-domain run. Diagnostics report an approximate longer duration
based on conditioning, cancellation scaling, and the adjacent-frequency beat
period.

Dispersive materials are supported: every wavelength component uses the native
frequency-corrected mode solve and dispersive H-side impedance filter.
Electrical conductivity is currently rejected because conductive eigenmodes
are complex while FDTD-X's sampled `CustomTimeSignalProfile` ignores the phase
shift used by the native source to inject the imaginary quadrature. Supporting
it requires a phase-aware sampled profile rather than bypassing the complex
modal phase.

### Chebyshev broadband profile model

`ChebyshevDistributedCurrentSource` is the reusable low-level production
abstraction for smoothly frequency-dependent distributed profiles. Its sample
frequencies are first-kind Chebyshev roots over the center frequency plus or
minus 1.5 Gaussian spectral widths. They therefore depend on the source pulse
and interpolation order, not on objective or monitor wavelength spacing.

Sampled complex profiles are phase-aligned from the center outward, checked for
low adjacent overlap as a simple mode-crossing diagnostic, and transformed with
a discrete cosine transform. Each Chebyshev frequency polynomial is realized
as a finite combination of analytic Gaussian-envelope derivatives, so the
source retains the compact duration of the original pulse and needs no
arbitrary-frequency isolation solve.

### Complex source normalization and Yee timing

Tidy3D frequency-domain monitor data are divided by a complex source spectrum,
not by its magnitude. Its public `SimulationData.source_spectrum()` evaluates
the source time dependence on the simulation time mesh and monitor data classes
divide complex field or mode amplitudes by that result; flux is divided by its
absolute square. Tidy3D mode profiles are power-normalized, with a mode source
specified as 1 W at its central frequency. See the public implementations of
[`SimulationData.source_spectrum()`](https://docs.flexcompute.com/projects/tidy3d/en/latest/_modules/tidy3d/components/data/sim_data.html),
[`SourceTime`](https://docs.flexcompute.com/projects/tidy3d/en/v2.11.2/_modules/tidy3d/components/source/time.html),
and [`ModeSource`](https://docs.flexcompute.com/projects/tidy3d/en/v2.11.2/_modules/tidy3d/components/source/field.html).

MAPS now follows the same discrete complex-spectrum principle through
`fdtdx.discrete_source_spectrum`. It evaluates the actual truncated real source
samples with the FDTD-X detector sign convention `exp(+i omega t)`, using the
placed detector's exact switch indices, DFT subsampling stride, and scaling
mode. Both dense and sparse forward/adjoint phasor normalization use this one
implementation. Deliberate static source amplitude remains unnormalized.

For `ChebyshevModePlaneSource`, a scalar temporal normalization alone is not
enough. A native mode source samples its electric and magnetic incident fields
at component-dependent Yee travel-time offsets. Using the center-frequency
offsets for every Chebyshev coefficient produced a smooth frequency-dependent
complex gain error even when the interpolated spatial profile was accurate.
The source therefore transforms the *effective injection profile* at each
Chebyshev node:

```text
E_eff(f, r) = E_mode(f, r) exp(-i 2 pi f dt offset_E(f, r))
H_eff(f, r) = H_mode(f, r) exp(-i 2 pi f dt offset_H(f, r))
```

Mode-branch phase tracking is computed from the raw eigenmodes and then applied
to these effective profiles, so it does not erase the physical delay phase.
The resulting coefficients are injected with zero additional travel-time
offset. This preserves both the relative E/H phase and the eigenmode's global
phase convention without a reference-derived correction table.

The optional broadband adjoint now factors the frequency-indexed Yee current
samples into phase-tracked unit profiles and arbitrary complex VJP weights.
Only the profiles are fitted in a low-order Chebyshev basis. Compact Gaussian
derivatives inject all spatial terms simultaneously; the complex source DTFT
normalizes the recorded adjoint fields, and the original complex weights are
applied before the existing material-gradient kernel. No reference-profile
projection or single-frequency simulation is used to construct this source.

Environment defaults, allowed values, and usage are centralized in
[`core/env_args.py`](../../core/env_args.py); run `python -m core.env_args`
for the complete catalog or see [environment arguments](env-args.md).

Enable with `MAPS_ENABLE_BROADBAND_ADJOINT=1`; the default remains independent
per-frequency adjoints. `MAPS_BROADBAND_ADJOINT_METHOD=auto` chooses the smallest
acceptable order from one through five (bounded by the available samples).
`MAPS_BROADBAND_ADJOINT_PROFILE_ORDER` selects an explicit order for validation.
`MAPS_BROADBAND_ADJOINT_METHOD=legacy` retains the original reference-profile
projection as a debug baseline. Unsafe fits, inadequate spectral support,
excessive waveform amplification, and missing or overlapping group coverage
fall back to independent frequency runs. Simulation duration is never extended
to isolate closely spaced objective frequencies.

See [broadband adjoint construction](../design/broadband-adjoint.md) for the
factorization, guards, numerical conventions, validation commands, and limits.

## 17. Gaussian beam source

`insert_gaussian_beam_fdtd3d()` constructs a broadband FDTD-X Gaussian-plane source.

For propagation along `z`, the current implementation supports:

```text
Ez1 -> x-polarized E field
Ez2 -> y-polarized E field
```

The API accepts:

```text
waist_radii
waist_distances
```

and maps them to the FDTD-X/Tidy3D-compatible Gaussian source representation.

The source is normalized by Poynting flux for unit-power launch semantics.

## 18. Mode monitors and field monitors

`insert_monitors_fdtd3d()` creates backend-specific detector objects.

Supported conceptual monitor types include:

```text
mode
flux
field
design_region_field
sparse_field
```

### Mode monitor

A FDTD-X mode-overlap detector provides modal field profiles. For symmetry-aware placement, fields can be unfolded and scaled before being converted into Torch tensors.

The resulting profile uses the same modal tuple concept:

```text
(source_object, H_mode, E_mode, power_scale)
```

### Flux monitor

A Poynting-flux detector stores a flux measurement object and does not need modal field profiles.

### Field monitor

A broadband phasor detector stores a tensor conceptually shaped:

```text
[Nfreq, Ncomponent, X, Y, Z]
```

where the component set depends on `MAPS_ENABLE_PHASOR_H`.

The detector is used through an auxiliary `*_field_monitor` entry in `port_sources_dict`.

### Sparse field monitors

When `MAPS_SPARSE_PHASOR=1`, FDTD-X source preparation selects
`sparse_field`. The monitor profile contains localized detectors for each native
design-region mask and objective port slice. Sparse entries use keys shaped as:

```text
((wl_cen, wl_width, n_wl), source_mode, support_name)
```

and store:

```text
(detector, None, None, power_scale, native_bounds)
```

where `native_bounds` is an `(x, y, z)` tuple of half-open grid-index bounds.
Design supports use the design-region name; port supports use a `port:<name>`
identifier and detectors named `field_detector_port_<name>_<mode>`.
With `MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES=1` (the default), forward support
also includes three-cell-thick full-span `xy`, `xz`, and `yz` slabs through each
design-region center. These entries use
`visualization:<region_name>:<plane>` support identifiers. Identical slabs are
deduplicated when regions share a cross-section. If the device has no design
regions, the profile instead contains one set with
`visualization:origin:<plane>` identifiers, centered using the native cells
nearest physical `(0, 0, 0)`.

All design, objective-port, and enabled visualization detectors participate in
the forward solve so the assembled dense field covers downstream numerical and
plotting consumers. The adjoint solve uses only design-region detectors because
the permittivity gradient is evaluated only there. The primary design detector
is renamed for the adjoint run but retains the original native bounds; it is not
duplicated under its forward name. Set
`MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES=0` before importing MAPS to omit plot
slabs and minimize forward detector work.

Environment-controlled monitor selection occurs when the module is imported and
source/monitor profiles are built. Start a new Python process after changing
`MAPS_SPARSE_PHASOR` or `MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES`.

## 19. Source normalization

`build_norm_sources()` is one of the most important orchestration routines in this module.

Its purpose is to construct a normalized source that produces a target input power.

Supported source types:

```text
"mode"
"plane_wave"
"gaussian_beam"
```

### Normalization principle

A source is first launched without a final power scaling. The resulting output flux is measured through the auxiliary two-port geometry. The code computes:

```text
input_scale = sqrt(target_power / measured_flux)
```

and multiplies the source and associated modal fields by that scale.

The normalized profile then stores the target power as its explicit `norm_power` entry.

### Why a two-port device is used

The normalization path constructs an auxiliary device containing only the relevant port geometry. This isolates the source normalization problem from the full device geometry.

The source fields and modal profiles are subsequently transferred back to the original device grid if the auxiliary and original native grids differ.

This is an important architectural detail: **normalization is a calibration stage, not the same simulation as the final device solve**.

## 20. Native-grid source transfer

When the auxiliary two-port native grid differs from the main device native grid, the source profile is resampled using physical coordinates.

Both are potentially resampled:

```text
source field -> full native target grid
H_mode/E_mode -> target monitor transverse coordinates
```

This operation is coordinate-aware and should not be replaced by a raw `resize()` or tensor reshape.

## 21. Simulation wrappers

The device layer provides lightweight simulation construction around the actual solver backends.

### FDFD/Ceviche

`create_simulation()` prepares a 2D FDFD simulation for the selected polarization. `solve_ceviche()` executes a single source and maps the result to the field dictionary:

```text
Ez polarization:
    {"Hx", "Hy", "Ez"}

Hz polarization:
    {"Ex", "Ey", "Hz"}
```

`solve()` loops over source-profile entries and returns:

```python
{
    (wl, mode): field_dict,
}
```

The baseline `solve()` does not support the `fdtdx` solver directly; FDTD-X uses the dedicated `solve_fdtdx()` path.

### FDTD-X

`create_simulation_fdtdx()` constructs the `fdtd3d` backend using:

```text
epsilon/permittivity map
electrical conductivity map
native/export grid metadata
port source dictionary
port monitor slices
PML
symmetry
```

`solve_fdtdx()` returns a six-component field dictionary:

```python
{
    "Ex": ...,
    "Ey": ...,
    "Ez": ...,
    "Hx": ...,
    "Hy": ...,
    "Hz": ...,
}
```

## 22. Thermal and electrical material maps

The device base is also a multiphysics material-map factory.

Supported properties include:

```text
epsilon
conductivity
heat_capacity
thermo_optic_coeff
electrical_conductivity
heat_source
```

Common aliases are normalized, for example:

```text
eps / permittivity -> epsilon
k / thermal_conductivity -> conductivity
sigma -> electrical_conductivity
dn_dT -> thermo_optic_coeff
q / heat_density -> heat_source
```

The module keeps optical and thermal coordinate systems distinct. Thermal maps may live on a different grid from the optical epsilon map.

## 23. Thermal solver

`solve_heat()` consumes:

```text
k_map   -> thermal conductivity map
q_map   -> heat source density map
```

and can use configurable boundary conditions, backend, solver options, and adjoint solver options.

The thermal solver may operate on:

```text
rectangular
fixed_distance_unstructured_tidy3d
fixed_rectilinear_nonuniform
```

mesh types.

The returned temperature map can optionally include metadata describing padding, thermal-grid coordinates/boundaries, and fixed-mesh information.

The device caches the heat solver and, depending on the configured solve interval, can cache temperature solutions between objective evaluations.

## 24. Active material modulation

`apply_active_modulation()` takes a denormalized permittivity map and modifies one or more active-region masks according to configured modulation functions.

The active-region transform is applied consistently to both background and active material endpoints, then interpolated within the active region using the normalized local permittivity fraction.

Conceptually:

```text
rho = (eps - eps_bg) / (eps_r - eps_bg)

eps_new = rho * (eps_r_new - eps_bg_new) + eps_bg_new
```

Only the active-region mask is modified; the rest of the device remains unchanged.

## 25. FDTD-X native-grid mapping

The FDTD-X path maintains explicit native/export grid metadata and native design-region masks.

If native and export grids differ, an export permittivity map is first resampled onto the native grid. The device then blends the result only inside FDTD-X native design-region masks, optionally using coverage weights.

This creates the FDTD-X simulation permittivity map while preserving the original background material outside design regions.

## 26. Current architectural coupling

For future refactoring, treat these as the major responsibility groups currently coupled inside `device_base.py`:

```text
A. geometry/configuration
B. backend geometry translation
C. optical rasterization
D. optical/native/export grid management
E. design-region and active-region masks
F. port geometry and port-region masks
G. monitor/slice indexing
H. source and eigenmode generation
I. source normalization
J. FDFD simulation wrappers
K. FDTD-X simulation wrappers
L. thermal/electrical material maps
M. thermal solver orchestration
N. plotting/export helpers
```

The module currently relies on shared mutable state across these groups. In particular, changing geometry can affect cell bounds, which affects raster grids, which affects monitor indices, which affects source profiles and normalization. A future refactor must preserve those dependency relationships explicitly.

## 27. Safe invariants for future refactoring

The following should be treated as compatibility invariants until intentionally redesigned:

- Physical coordinates are authoritative; grid indices are derived.
- Native and export grids are distinct concepts.
- Design-region masks are tied to a specific grid.
- Monitor slices have native/export/symmetry-specific representations.
- Port source profiles are keyed by wavelength/mode, or wavelength-group/mode for FDTD-X.
- `Ez*` and `Hz*` mode names encode both polarization family and mode index.
- Source normalization is based on measured flux and target-power scaling.
- Auxiliary two-port normalization can have a different raster grid and therefore requires coordinate-aware source/profile transfer.
- FDTD-X uses broadband wavelength groups and six-component fields.
- 2D Ceviche and 3D FDTD-X use different field dictionaries and solver-entry paths.
- Thermal grids are not assumed to match optical grids.
- Active regions must remain inside their parent design regions.
- Subpixel/fallback rasterization is part of numerical behavior.
- Symmetry changes field/mode amplitudes and monitor indexing; normalization and adjoint paths explicitly account for it.

## 28. Refactoring guidance for later work

When this module is eventually refactored, use the documentation in this page as the behavioral contract rather than mechanically splitting the file by function count.

A good future decomposition would likely separate:

```text
DeviceGeometry
GridManager
DesignRegionManager
PortManager
MonitorManager
SourceManager
OpticalSimulationManager
ThermalSimulationManager
MaterialMapManager
```

but that is a **future architectural direction, not a requirement of the current release**. Any refactor should first preserve the contracts above with targeted regression tests around grid shapes/coordinates, masks, monitor indices, source normalization, modal profiles, and optical/thermal map placement.
