# FDTD API and Scientific Data Contract

This page documents the current MAPS-facing 3D FDTD interface implemented in `core/fdtd/fdtd.py`; it does not redefine or normalize the implementation.

The FDTD layer is backed by **fdtdx** and bridges MAPS PyTorch tensors to a JAX/fdtdx time-domain simulation. The principal convenience class is `fdtd3d`.

## Coordinate and tensor conventions

### Spatial grid

`fdtd3d` requires a three-dimensional relative-permittivity grid:

```text
eps_r.shape == [Nx, Ny, Nz]
```

The implementation rejects non-3D `eps_r` inputs. It records:

```text
shape = (Nx, Ny, Nz)
N = Nx * Ny * Nz
```

`Nx`, `Ny`, and `Nz` refer to the native optical-grid cell counts. Do not reinterpret these dimensions as image `H/W` dimensions.

### Length units

`dL` is stored in **meters**. When optical-grid metadata is supplied by the MAPS device layer, the metadata coordinates are interpreted as microns and converted internally to SI meters for fdtdx. The fallback uniform-grid path already uses `dL` in meters.

### Field components

The user-facing field order is:

```text
Ex, Ey, Ez, Hx, Hy, Hz
```

Before splitting into individual components, the assembled field tensor has the documented implementation layout:

```text
[Nfreq, 6, X, Y, Z]
```

After splitting and removing the singleton component axis, each returned component has shape:

```text
[Nfreq, X, Y, Z]
```

where `Nfreq` is the number of frequencies represented by the phasor monitor. For the standard source configuration this corresponds to the requested wavelength/frequency sampling.

The returned phasor fields are complex PyTorch tensors. The solver marks the final fields as colocated (`fields_are_colocated = True`).

When `MAPS_SPARSE_PHASOR=1`, localization is an internal detector-storage
choice. The returned tensor and six component tensors keep the same dense shapes
shown above. Localized detector values are symmetry-unfolded and scattered into
the full `[Nx, Ny, Nz]` domain. By default, recorded support includes design
regions, objective ports, and design-centered cross-section slabs; cells outside
that union are zero. The slabs can be disabled with
`MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES=0`. See
[FDTD differentiable solver](fdtd-solver.md#sparse-phasor-mode) for the forward
and backward contracts.

## `fdtd3d`

### Constructor

```python
fdtd3d(
    wl_cen,
    wl_width,
    n_wl,
    dL,
    eps_r,
    port_sources_dict,
    port_monitor_slices,
    port_monitor_slices_info,
    npml,
    electrical_conductivity_map=None,
    optical_grid_metadata=None,
    max_time=1e-10,
    symmetry=(0, 0, 0),
    *,
    device="cpu",
    **runtime_kwargs,
)
```

### Constructor data contract

`wl_cen`
: Center wavelength for this simulation instance. Its unit must follow the source/profile conventions used by the surrounding MAPS workflow.

`wl_width`
: Wavelength span associated with the source group.

`n_wl`
: Number of wavelength/frequency samples in the source group.

`dL`
: Fallback uniform-grid spacing in meters.

`eps_r`
: Relative permittivity distribution. Accepts `numpy.ndarray` or `torch.Tensor` with shape `[Nx, Ny, Nz]`. NumPy inputs are converted to PyTorch tensors. The tensor is moved to `device`. Relative permittivity is dimensionless.

`port_sources_dict`
: Mapping from port/source names to source-profile objects. `prepare_forward_simulation()` indexes the selected source profile using:

```python
((wl_cen, wl_width, n_wl), mode)
```

The dictionary also contains a corresponding field-monitor profile under the key:

```text
<input_slice_name>_field_monitor
```

`port_monitor_slices`
: Mapping from port names to native monitor/source slice definitions.

`port_monitor_slices_info`
: Mapping from port names to physical/source metadata. The forward setup reads at least:

```text
"direction"
"center"
"size"
```

`npml`
: Boundary/PML configuration retained by the solver. The implementation comments describe per-axis min/max thickness organization.

`electrical_conductivity_map`
: Optional MAPS electrical-conductivity map in S/um. NumPy inputs are converted to PyTorch and stored as `torch.float32` on the solver device. During FDTD-X preparation it is converted to the dimensionless conductivity update coefficient used by FDTDX. When `eps_r` is complex, its imaginary part is additionally converted using the FDTDX-equivalent rule `sigma = omega * eps0 * imag(eps_r)` at `wl_cen`, then combined with this map. Broadband and multi-wavelength simulations are allowed, with a warning; the static-conductivity approximation is exact only at `wl_cen`.

`optical_grid_metadata`
: Optional dictionary describing the native and export rectilinear grids. The FDTD layer uses the following stable keys when present:

```text
"native"
"export"
"native_equals_export"
"export_policy"
```

The native/export entries contain grid coordinate/boundary metadata consumed by the MAPS rectilinear-grid utilities.

`max_time`
: Maximum fdtdx simulation time in seconds. Default: `1e-10`.

`symmetry`
: Three-entry tuple controlling fdtdx symmetry. The implementation documents the values as:

```text
0  -> no symmetry
-1 -> PEC symmetry
+1 -> PMC symmetry
```

`device`
: PyTorch device used for MAPS-side tensors.

## `solve()`

```python
solve(
    input_slice_name,
    wl_cen,
    wl_width,
    n_wl,
    mode,
    *,
    eps_r=None,
    adjoint_group_info=None,
    interpolate_to_export_grid=True,
    **runtime_kwargs,
)
```

Runs a differentiable fdtdx forward solve and returns all six electromagnetic field components.

### Inputs

`input_slice_name`
: Name of the port/input slice used to select the source and associated monitor metadata.

`wl_cen`, `wl_width`, `n_wl`
: Wavelength-group specification used to retrieve the source and monitor profiles.

`mode`
: Mode identifier used as part of the source-profile lookup. Existing code documents strings such as `"Ez1"` for a TM0-style mode. Treat supported mode names as defined by the source-profile construction code rather than as arbitrary strings.

`eps_r`
: Optional per-call relative-permittivity grid, shape `[Nx, Ny, Nz]`. When omitted, `self.eps_r` is used. Passing a value updates the solver's active shape and permittivity state.

`adjoint_group_info`
: Optional metadata forwarded to the differentiable FDTD bridge to group adjoint support by monitor/frequency information. See [FDTD differentiable solver](fdtd-solver.md).

`interpolate_to_export_grid`
: If `True` and native/export grids differ, resample the final colocated field tensor from native rectilinear coordinates to export-grid coordinates along the last three spatial axes.

`runtime_kwargs`
: Additional keyword options forwarded to the FDTD autograd/runtime bridge.

### Returns

```python
Ex, Ey, Ez, Hx, Hy, Hz = sim.solve(...)
```

Each value is a complex PyTorch tensor with shape:

```text
[Nfreq, X, Y, Z]
```

The spatial dimensions correspond to the export grid when export interpolation is enabled and a distinct export grid exists; otherwise they correspond to the native grid.

Sparse phasor mode does not alter this return type. Before optional export-grid
resampling, MAPS assembles the localized recordings into the native dense shape
`[Nfreq, 6, Nx, Ny, Nz]`; the returned components are split from that tensor.

### Yee-grid and temporal colocation

When native exact interpolation is not provided by the fdtdx phasor detector, MAPS performs two operations before returning fields:

1. The magnetic phasor is corrected for the half-time-step staggering using the implemented factor

```text
theta = omega * dt / 2
H <- H * exp(i * theta) * cos(theta)
```

where `dt` is the fdtdx time-step duration.

2. Electric and magnetic fields are spatially interpolated from their Yee-grid locations to colocated positions using `yee_to_colocate_interpolate()`.

The six components are then concatenated and unfolded into the full phasor representation.

## Source phase normalization

`normalize_fields_phase(E, H)` removes a global phase from source fields. It accepts either PyTorch tensors or JAX arrays and expects matching E/H shapes containing a three-component vector axis, typically conceptually:

```text
[..., 3, X, Y, Z]
```

The implementation squeezes singleton dimensions, finds the spatial location of maximum combined E/H magnitude, selects the strongest field component there, and rotates both E and H by

```text
exp(-i * phase)
```

so the selected reference component has zero global phase.

This operation changes global phase only; it does not normalize source power.

## Physical boxes and native-grid slices

`real_box_to_grid_slices()` converts a physical source/monitor box into three native-grid slices.

Inputs:

- `boundaries`: `(x_edges, y_edges, z_edges)`, each strictly increasing and of length `Naxis + 1`;
- `center`: length-3 physical center;
- `size`: length-3 physical box size;
- `direction`: propagation direction such as `"x+"`, `"x-"`, `"y+"`, or `"z-"`;
- `transverse_rule`: `"cover"` or `"center"`;
- `clip`: whether transverse slices are clipped to the domain.

The propagation axis is forced to one cell. For transverse axes:

- `"cover"` includes all cells whose intervals overlap the requested aperture;
- `"center"` includes cells whose centers lie inside the aperture and falls back to the nearest cell when the aperture is smaller than the local cell width.

Return value:

```text
(slice_x, slice_y, slice_z)
```

## Boundary construction

`boundary_objects_from_config_real_coordinates()` constructs fdtdx boundary objects and places them with real-coordinate constraints. The implementation supports:

```text
pml
periodic
pec
pmc
bloch
```

The function validates that each grid-boundary array is one-dimensional, strictly increasing, and has length `Naxis + 1` for `full_domain_grid_shape = (Nx, Ny, Nz)`.

## Native versus export grid

The FDTD layer distinguishes:

- **native grid**: the rectilinear grid on which fdtdx runs;
- **export grid**: the grid on which MAPS may expose fields to downstream code.

For a uniform native grid, a rectilinear uniform grid is constructed from spacing and origin. For nonuniform grids, fdtdx receives explicit x/y/z edge arrays.

When `native_equals_export` is false and `interpolate_to_export_grid=True`, fields are resampled after FDTD/Yee colocation.

## Usage pattern

A typical call uses a fully configured `fdtd3d` object whose port dictionaries were constructed by the higher-level MAPS device layer:

```python
Ex, Ey, Ez, Hx, Hy, Hz = sim.solve(
    input_slice_name="in_slice_1",
    wl_cen=1.55,
    wl_width=0.0,
    n_wl=1,
    mode="Ez1",
)

assert Ex.ndim == 4
assert Ex.shape == Ey.shape == Ez.shape
assert Ex.shape == Hx.shape == Hy.shape == Hz.shape
# Layout: [Nfreq, X, Y, Z]
```

The numerical wavelength units in an actual call must match the surrounding source-profile/device configuration; this example illustrates API structure and tensor rank rather than redefining that higher-level unit convention.

## Differentiability

The MAPS FDTD path is designed to be differentiable with respect to the active permittivity tensor through `FDTDXSolveTorch`. Forward scene construction is handled in `fdtd3d`, while the Torch/JAX autograd and adjoint machinery lives in `core/fdtd/solver.py`.

See [FDTD differentiable solver](fdtd-solver.md) for the adjoint/data-contract details.
