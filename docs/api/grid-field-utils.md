# Shared Grid, Field, and Coordinate Utilities

This page documents the cross-cutting numerical utilities in `core/utils.py` for the `v0.0.2_fdtdx` baseline. `core/utils.py` is intentionally broad and contains many unrelated helpers; this page focuses on the contracts that define data exchange between geometry, FDFD, FDTD, objective evaluation, and inverse design.

This is documentation only. No implementation behavior is changed.

## 1. Canonical rectilinear-grid representation

### `RectilinearGridMetadata`

The central grid object is:

```python
@dataclass
class RectilinearGridMetadata:
    shape: tuple[int, ...]
    coords: tuple[np.ndarray, ...]
    boundaries: tuple[np.ndarray, ...]
    spacing: tuple[float, ...]
    cell_weights: np.ndarray
    is_uniform: bool
    origin: tuple[float, ...]
    extent: tuple[float, ...]
    label: str = "grid"
```

The intended representation is **cell-center coordinates plus cell boundaries**, not simply a shape and one global spacing.

### Coordinate semantics

For each physical axis `a`:

```text
coords[a]       : cell-center coordinates, length N_a
boundaries[a]   : cell boundaries, length N_a + 1
```

For a uniform grid, boundary differences are constant. For a nonuniform grid, the local cell width is:

```text
Delta_a[i] = boundaries[a][i+1] - boundaries[a][i]
```

`shape` is derived from the number of coordinate samples:

```text
shape == tuple(len(coords[a]) for a in axes)
```

`extent` stores the full physical span from the first to the last boundary. `origin` stores the first boundary coordinate.

### Cell weights

`cell_weights` is the tensor-product cell-volume/area weighting constructed from boundary widths.

For 2D:

```text
W[i, j] = Delta_x[i] * Delta_y[j]
```

For 3D:

```text
W[i, j, k] = Delta_x[i] * Delta_y[j] * Delta_z[k]
```

These weights are essential when integrating fields on nonuniform grids. They should not be replaced by a scalar `dx**dim` unless the grid is known to be uniform.

## 2. Building grid metadata

`build_rectilinear_grid_metadata()` accepts either explicit `coords` or `shape` plus uniform `spacing`.

### Uniform construction

```python
metadata = build_rectilinear_grid_metadata(
    shape=(Nx, Ny, Nz),
    spacing=(dx, dy, dz),
)
```

The resulting center coordinates are symmetric around zero unless an explicit `extent` is provided.

### Explicit-coordinate construction

```python
metadata = build_rectilinear_grid_metadata(
    coords=(x, y, z),
    boundaries=(x_edges, y_edges, z_edges),
)
```

Explicit coordinates are authoritative for nonuniform optical/thermal grids.

### Boundary inference

When only cell centers are available, `centers_to_boundaries()` infers boundaries by midpoint construction. This should only be interpreted as a reconstruction rule; it does not recover unknown physical boundary locations beyond what the center samples imply.

For a one-cell axis, the helper uses a unit-width artificial interval centered on the single sample. Callers that care about physical absolute extent should provide explicit boundaries instead.

## 3. Coordinate-aware resampling

### `resample_rectilinear_tensor()`

```python
resample_rectilinear_tensor(
    values,
    src_coords,
    dst_coords,
    *,
    axes=None,
    mode="polar",
)
```

This is the primary coordinate-aware interpolation helper used by `device_base.py` for native/export and optical/thermal grid transfers.

### Input contract

`values`
: PyTorch tensor or array-like input. If not already a tensor, it is copied into a PyTorch tensor.

`src_coords`
: One strictly increasing 1D coordinate array per resampled physical axis.

`dst_coords`
: One strictly increasing 1D coordinate array per target physical axis.

`axes`
: Tensor axes corresponding to `src_coords`/`dst_coords`. If omitted, the final `len(src_coords)` tensor dimensions are interpreted as spatial axes.

`mode`
: Complex interpolation mode:

```text
"cartesian"
"polar"
```

### Shape contract

Suppose:

```text
values.shape = [B, C, Nx, Ny, Nz]
```

and the spatial axes are `(2, 3, 4)`. Then:

```text
src_coords = (x_src, y_src, z_src)
dst_coords = (x_dst, y_dst, z_dst)
```

produces:

```text
output.shape = [B, C, len(x_dst), len(y_dst), len(z_dst)]
```

Only the specified spatial axes are resampled; leading batch/channel axes are preserved.

### Complex interpolation

`"cartesian"` interpolates the complex value directly:

```text
z(t) = z0 + (z1 - z0)t
```

`"polar"` interpolates magnitude linearly and phase along the wrapped shortest angular difference:

```text
A(t)   = A0 + (A1 - A0)t
phi(t) = phi0 + wrap(phi1 - phi0)t
z(t)   = A(t) exp(i phi(t))
```

When either endpoint magnitude is too small for a stable phase interpretation, the helper falls back to Cartesian interpolation for that element.

This distinction is scientifically important for optical complex fields. Do not silently change the default interpolation mode when resampling phase-sensitive field data.

### Coordinate validity

Each source coordinate vector must be strictly increasing. Duplicate/decreasing source coordinates raise `ValueError`.

When source and target coordinates are identical, the axis is skipped without interpolation.

## 4. Plane integration weights

### `grid_plane_weights()`

This helper converts grid metadata into transverse area weights for a plane normal to a selected axis.

For example, for a plane normal to `x` in a 3D grid:

```text
axis = 0
W_plane[j, k] = Delta_y[j] * Delta_z[k]
```

The helper multiplies each boundary width by `MICRON_UNIT` before taking the outer product, so the returned area weights are explicitly in square meters.

These weights are used by the 3D eigenmode coefficient and flux routines to preserve area integration on nonuniform grids.

The weights retain their absolute physical scale. In particular, callers must not divide each monitor plane by its own mean area. Such a division changes cross-axis transmission whenever, for example, an `xy` input monitor and a `yz` output monitor have different mean cell areas.

## 5. Field integration and eigenmode conventions

### `get_eigenmode_coefficients_3d()`

This function projects a simulated six-component electromagnetic field onto a reference mode.

### Input field layout

The implementation expects the six field components as individual tensors, conceptually:

```text
Ex, Ey, Ez
Hx, Hy, Hz
```

The full field vectors are assembled as:

```text
E_sim.shape = [3, ...monitor-spatial-shape...]
H_sim.shape = [3, ...monitor-spatial-shape...]
```

The reference mode fields `et_m` and `ht_m` must have matching monitor-grid shape after squeezing their leading singleton dimensions. Shape mismatches raise `ValueError` rather than silently resampling.

### Important precondition

The docstring/source explicitly assumes that E/H fields are **already spatially colocated** before calling this function.

That is why the FDTD pipeline performs Yee-grid colocation first.

### Lorentz-overlap form

The forward/backward modal amplitudes are derived from:

```text
A+ = integral[(E_m x H_sim* + E_sim* x H_m)_n dA]
A- = integral[(E_m x H_sim* - E_sim* x H_m)_n dA]
```

where `n` is the monitor normal direction.

The implementation applies the same transverse area weighting used by `get_flux_3d()`.

The reference-mode power is evaluated using the same physical-area quadrature:

```text
P_mode = abs(0.5 * Re(integral[(E_mode x H_mode*)_n dA]))
```

When `energy=True`, the function returns:

```text
P+ = |A+|^2 / P_mode
P- = |A-|^2 / P_mode
```

Otherwise, it selects the requested propagation direction and returns a mode-power-normalized amplitude:

```text
direction[1] == "+" -> (A+ / sqrt(P_mode), 0)
direction[1] == "-" -> (0, A- / sqrt(P_mode))
```

Fields and reference modes must already be unfolded to the full domain. Symmetry multiplicity is not part of `dA` and must not be applied again during overlap integration.

### Direction contract

`direction` must be a two-character string such as:

```text
"x+"  "x-"
"y+"  "y-"
"z+"  "z-"
```

The first character selects the monitor-normal axis; the second selects the propagation sign.

## 6. Polarization-specific 2D field convention

The repository supports two common 2D scalar-polarization reductions:

### `Ez` polarization

The effective vector field passed to flux/eigenmode routines is assembled as:

```text
Fx = Hx
Fy = Hy
Fz = Ez
```

### `Hz` polarization

The effective vector field is:

```text
Fx = Ex
Fy = Ey
Fz = Hz
```

This convention is part of the FDFD/objective interface. It should not be replaced by a generic `E`/`H` vector assumption in 2D code.

## 7. Flux conventions

The utility layer contains separate 2D and 3D flux functions. The exact implementation should be treated as the source of truth for the sign and normalization convention.

At the API level, the important distinction is:

- a flux value may be **signed** according to propagation direction;
- objective functions may subsequently take an absolute value or normalize by source power;
- energy objectives may square a modal amplitude rather than use raw flux.

For 3D rectilinear grids, `get_flux_3d()` computes

```text
0.5 * Re(sum((E x H*)_normal * dA))
```

with `dA` in square meters. A fallback unit weight remains available only for legacy calls that supply neither grid metadata nor cell weights; scientific FDTD-X workflows should provide physical grid metadata.

Therefore a variable called `s`, `flux`, or `norm_power` should not be interpreted generically across all objective classes.

## 8. Yee-grid interpolation

### `interpolate_fields()` / `yee_to_colocate_interpolate()`

The FDTD utility stack converts staggered Yee-grid fields to a common reference location before objective evaluation and export.

The target reference location in the implemented 3D interpolation helper is the `E_z` Yee-grid location:

```text
(i, j, k + 1/2)
```

For a uniform grid, interpolation reduces to arithmetic averaging of adjacent samples.

For a nonuniform rectilinear grid, `_backward_edge_average()` performs a width-weighted interpolation based on local cell widths. This means:

```text
nonuniform grid != arithmetic mean in general
```

Changes to the interpolation rule can change both forward fields and gradients.

## 9. Boundary-aware field padding

### `pad_fields()`

Expected field shape:

```text
[..., component, Nx, Ny, Nz]
```

The function returns a field padded by one cell along every spatial axis:

```text
[..., component, Nx+2, Ny+2, Nz+2]
```

For each physical axis:

- periodic boundaries use circular padding;
- nonperiodic boundaries use replicate padding in this helper.

### `pad_fields_for_boundaries()`

This higher-level helper adds boundary-specific corrections after padding. In particular, Bloch-periodic boundaries can multiply ghost cells by a complex phase factor.

The phase is applied with conjugation on the negative-facing boundary and the forward phase on the positive-facing boundary, matching the source convention.

This function therefore has both geometric and complex-phase semantics and should not be reduced to a generic tensor-padding helper.

## 10. Symmetry unfolding

### `unfold_phasor()`

This function reconstructs a full-domain phasor from a symmetry-reduced FDTD detector output.

Supported phasor layouts:

```text
reduce_volume=False:
    [F, C, nx, ny, nz]

reduce_volume=True:
    [F, C]
```

where:

- `F` = frequency count;
- `C` = stored field-component count.

The canonical component order is:

```text
Ex, Ey, Ez, Hx, Hy, Hz
```

For spatial phasors, symmetry is implemented by `flip + sign-aware concatenate`. For reduced-volume phasors, the contribution factor is built from component parity.

A source or mode normalized on a retained domain with `k` mirrored transverse axes carries a field-amplitude factor `1 / sqrt(2**k)` when reconstructed as the equivalent full-domain field. Once reconstructed, ordinary physical surface quadrature is applied exactly once.

### PEC/PMC parity

For each symmetry plane:

```text
wall = -1 -> PEC
wall = +1 -> PMC
```

The electric/magnetic component parity depends on whether the component is normal or tangential to the symmetry plane. This parity must be preserved when adding new field components.

## 11. Coordinate conversions

The utility layer also contains Cartesian/spherical conversion helpers.

### `car_2_sph`

For 3D:

```text
r     = sqrt(x^2 + y^2 + z^2)
theta = arccos(z / r)
phi   = atan2(y, x)
```

For the 2D form (`z is None`), `theta` is omitted and the function returns the radial coordinate plus azimuthal angle.

### `sph_2_car`

For 3D:

```text
x = r sin(theta) cos(phi)
y = r sin(theta) sin(phi)
z = r cos(theta)
```

These conventions should be documented anywhere far-field or radiation calculations expose spherical angles.

## 12. Tensor/array backend interoperability

The utility layer supports Torch, JAX, and NumPy because FDFD/FDTD paths do not all execute in the same array backend.

### `_torch_to_jax()` / `_jax_to_torch()`

These helpers use DLPack when possible to avoid a host round-trip.

The typical scientific data flow is:

```text
Torch differentiable tensor
        |
        | DLPack
        v
JAX / fdtdx tensor
        |
        | DLPack
        v
Torch tensor
```

Callers must preserve device and dtype compatibility. Avoid inserting `.cpu().numpy()` into differentiable paths merely to make a utility accept an input.

### `_to_runtime_array()`

Supported runtime backends:

```text
"jax"
"torch"
"numpy"
```

A tensor passed to a runtime backend is detached before conversion. Therefore this helper is for runtime interface adaptation, not for preserving PyTorch autograd history across a backend boundary.

## 13. Scientific documentation rules for this module

Because `core/utils.py` is shared infrastructure, new APIs should document:

- exact tensor rank and component axis;
- spatial-axis ordering `(x, y, z)` when applicable;
- coordinate representation (center versus boundary coordinates);
- physical units;
- complex interpolation mode;
- area/volume weighting convention;
- whether fields must already be colocated;
- polarization convention in 2D;
- direction format such as `"x+"`;
- differentiability and backend conversions;
- whether a helper preserves leading batch/channel axes.

A docstring that only says "interpolate the field" is insufficient for this module.

## 14. Refactoring boundary

`core/utils.py` is another large mixed-responsibility module. The current documentation intentionally treats these utilities by contract rather than suggesting an immediate code split.

A future refactor should likely separate at least:

```text
grid metadata / coordinates
field interpolation / Yee-grid operations
eigenmode / flux integration
backend conversion
symmetry / boundary utilities
visualization / miscellaneous helpers
```

Such a split should happen only after existing callers and tensor contracts are covered by tests and documentation. The current module remains the compatibility source of truth for `v0.0.2_fdtdx`.
