# FDFD Rectilinear Grids and Near-to-Far Contracts

## Rectilinear FDFD grids

`core.fdfd.derivatives_rectilinear` is the nonuniform-grid counterpart of the uniform-grid derivative implementation.

The primary grid contract is:

```text
shape = (Nx, Ny)
dxs.shape = [Nx]
dys.shape = [Ny]
D*.shape = [Nx * Ny, Nx * Ny]
```

`dxs[i]` and `dys[j]` are physical cell widths. They are not boundary coordinates and must be strictly positive and finite.

### Boundary-to-spacing conversion

`spacing_from_boundaries(boundaries)` converts cell-boundary arrays to widths:

```python
dxs = boundaries[0][1:] - boundaries[0][:-1]
dys = boundaries[1][1:] - boundaries[1][:-1]
```

Thus the expected boundary lengths are `Nx + 1` and `Ny + 1`.

For native-grid metadata, the preferred contract is to derive spacing from the stored physical boundary arrays rather than reconstructing a uniform `dL`.

### Nonuniform finite differences

The forward x derivative for cell `i` follows the local cell width:

$$
(D_x^+ u)_i = \frac{u_{i+1} - u_i}{\Delta x_i}.
$$

The backward operator uses the preceding cell width:

$$
(D_x^- u)_i = \frac{u_i - u_{i-1}}{\Delta x_{i-1}}.
$$

The same construction applies in y. The 2-D matrices are assembled through sparse Kronecker products and preserve the `[Nx, Ny] -> [Nx*Ny]` ordering used by the rest of FDFD.

### Bloch and PML behavior

The rectilinear implementation supports the same wrapped/Bloch boundary convention as the uniform-grid path, but computes PML profiles from the actual physical widths. `npml=(nx_pml, ny_pml)` counts cells, while the PML profile's physical scale is determined from those cells' widths.

Do not treat `npml` as a distance in meters.

## Near-to-far transformation

`core.fdfd.near2far` contains Rayleigh-Sommerfeld and Green-function based transformations from monitor-slice near fields to far-field observations.

### Common tensor shapes

A wavelength-vectorized near-field monitor is represented as:

```text
fields : [batch, source_points, frequency]
```

For full-domain field arrays, a monitor slice selects the `source_points` axis before projection.

Far-field target locations are represented as:

```text
farfield_x : [Nfar, 2]
```

where the last dimension contains physical `(x, y)` coordinates in the current implementation's micrometer coordinate convention.

A far-field result for one component is typically reshaped to:

```text
[batch, farfield_shape..., Nfreq]
```

and returned inside a six-component dictionary:

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

Unused components are initialized as scalar zero values.

### Monitor metadata

The near-field array alone is insufficient to reproduce a far field. Each monitor must preserve its physical coordinate arrays and propagation direction. The current schema uses fields such as:

```python
{
    "xs": ...,          # physical monitor coordinates
    "ys": ...,          # physical monitor coordinates
    "direction": "x+"  # example orientation
}
```

A far-field slice can similarly be represented by `farfield_slice_info["xs"]` and `farfield_slice_info["ys"]` when explicit target coordinates are not passed.

### Rayleigh-Sommerfeld path

`RayleighSommerfieldPropagation` evaluates the scalar propagation kernel for a selected field component. Its near-field tensor has shape `[batch, source_points, frequency]`; `x` is `[Nfar, 2]`; `x0` is `[source_points, 2]`; and `z` is `[Nfar, source_points, 1]`.

The wavenumber is formed from the relative homogeneous-medium permittivity and spatial frequency:

$$
k = 2\pi\sqrt{\epsilon_r}\,f.
$$

The current implementation returns a factor-of-two-adjusted result to match the MAPS FDFD reference behavior. This correction is part of the current numerical contract and must not be removed as a mathematical simplification without validation.

### Green-function path

`get_farfields_GreenFunction` accepts multiple near-field monitor slices and sums their projected contributions. It supports the two scalar polarization conventions:

```text
component == "Ez" -> returns Ez, Hx, Hy
component == "Hz" -> returns Hz, Ex, Ey
```

The `decimation_factor` subsamples the near-field source points and scales `dL` accordingly so the discrete integration weight remains physically consistent.

### Batched observation points

`GreenFunctionProjection` processes far-field target points in batches controlled by `maximum_batch_size`. This is a memory-management parameter; it does not change the physical far-field definition.

### Units

The existing implementation uses micrometers for near/far coordinate arrays and `dL` in these near-to-far routines, while frequencies are represented as inverse wavelength. A new API should make this explicit rather than inheriting the SI convention used elsewhere in the FDFD solver.

## Documentation requirements

Near-to-far docstrings should always specify:

- near-field array shape;
- which field component is represented;
- monitor coordinate schema and units;
- monitor orientation/direction;
- far-field target coordinate shape and units;
- frequency/wavelength representation;
- homogeneous-medium `eps_r`/`mu` assumptions;
- decimation and batch-size effects;
- output component dictionary and shape.
