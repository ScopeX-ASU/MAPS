# FDFD Near-to-Far Contract

`core.fdfd.near2far` converts fields sampled on finite near-field monitor slices into fields at specified far-field observation coordinates.

## Near-field tensor schema

The common batch/frequency representation is:

```text
fields.shape == [batch, X, Y, frequency]
```

For a selected monitor slice, the implementation extracts:

```text
fz.shape == [batch, source_points, frequency]
```

The exact component dictionary may contain `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, and `Hz` depending on the source polarization and projection routine.

All field tensors are complex-valued. Coordinate tensors are real-valued physical positions, normally expressed in micrometers in this module's current API.

## Monitor metadata

Each near-field monitor description provides coordinate arrays and a propagation direction. The direction convention includes values such as:

```text
x+
x-
y+
y-
```

The coordinate arrays identify the physical locations of the sampled source points. The monitor metadata therefore cannot be separated from the field tensor when reproducing the near-to-far integral.

A typical conceptual monitor record is:

```python
{
    "slice": monitor_slice,
    "xs": x_coordinates,
    "ys": y_coordinates,
    "direction": "x+",
}
```

The actual stored dictionary may contain additional device-specific fields; callers should preserve the existing schema rather than infer missing geometry from tensor shape alone.

## Observation coordinates

`farfield_x` represents target observation locations in the same physical coordinate convention as the monitor locations.

For explicit point targets:

```text
farfield_x.shape == [N_target, 2]
```

where the final dimension is `(x, y)`.

A slice-based far-field request may instead provide `farfield_slice_info` containing `xs` and `ys`. A scalar x or y coordinate is expanded to form a vertical or horizontal observation line.

## Rayleigh-Sommerfeld projection

`RayleighSommerfieldPropagation` uses a Green-function-like Rayleigh-Sommerfeld expression for propagation from a selected near-field component.

The wave number is computed as:

$$
k = 2\pi\sqrt{\epsilon_r}\,f,
$$

where `f` is frequency and the surrounding homogeneous medium is characterized by relative permittivity `eps` and permeability `mu`.

The implementation returns a batch/target/frequency result:

```text
farfield.shape == [batch, N_target, N_frequency]
```

or the corresponding reshaped form when target coordinates are represented as a 2-D slice.

## Green-function projection

`GreenFunctionProjection` accepts tangential near-field components and reconstructs far-field electric/magnetic components according to the selected scalar polarization.

For an `Ez` source path, the returned tuple corresponds to the projected `Hx`, `Hy`, and `Ez` components; for an `Hz` path, it corresponds to `Ex`, `Ey`, and `Hz`.

The decimation option reduces near-field source sampling before evaluating the propagation integral:

```text
decimation_factor > 1
    -> subsample source points
    -> scale effective dL accordingly
```

This is a computational approximation, not merely an output-resolution parameter.

## Numerical and physical conventions

The near-to-far routines use:

```text
frequency      : inverse length when passed as 1 / wavelength
wavelength     : micrometers in current caller conventions
near-field dL  : micrometers
coordinates    : micrometers
fields         : complex amplitudes
```

Do not mix SI-meter coordinates with the current micrometer-based near-to-far inputs without converting `dL`, monitor coordinates, observation coordinates, and wavelength/frequency consistently.

## Inverse-design considerations

Near-to-far evaluation may sit inside an inverse-design objective. Therefore documentation and modifications must distinguish:

1. the near-field tensor that remains connected to autograd;
2. monitor coordinate metadata, which is usually non-differentiable geometry;
3. far-field observation locations;
4. scalar/material parameters such as background permittivity;
5. any source decimation or batching that changes numerical cost without changing the mathematical target.

A refactor that detaches the complex near-field tensor or converts it to NumPy before the objective is complete can silently break gradient-based inverse design.

## Reference equation

The physical interpretation is a Green-function/Rayleigh-Sommerfeld propagation integral over the sampled monitor surface:

$$
E_{far}(\mathbf r) \approx \int_{\Gamma}
G(\mathbf r,\mathbf r')\,F(\mathbf r')\,d\Gamma'.
$$

The discretized implementation uses the monitor-point coordinates, sampled field values, frequency-dependent propagation factor, and `dL` quadrature scale.

## Documentation requirements

Every future near-to-far API docstring should include:

- exact batch/target/frequency tensor shapes;
- field-component convention (`Ez`, `Hz`, etc.);
- coordinate units;
- monitor metadata schema;
- observation-coordinate shape;
- frequency/wavelength convention;
- decimation/batching behavior;
- whether the function preserves Torch autograd;
- a representative example showing input/output shapes.
