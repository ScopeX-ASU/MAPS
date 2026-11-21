# Units and Normalization

MAPS crosses device, electromagnetic-solver, optimization, and data-export boundaries where the same physical quantity may be represented in different numerical conventions. Units and normalization are therefore part of the scientific API contract.

## General rule

Do not infer units from tensor shape, variable name, or numerical magnitude. Document units at the boundary where a quantity is created, converted, or consumed.

## Geometry and coordinates

Device-layer geometry and grid coordinates are commonly represented in micrometers (um). Solver backends may convert physical lengths to SI units internally when required by their numerical formulation or external library.

A conversion at a backend boundary should be explicit and should not silently change the convention used by the device object.

## Wavelength and frequency

Wavelength configuration in MAPS photonic workflows is commonly expressed in micrometers, for example `1.55` for a nominal 1550 nm optical wavelength. Backend-specific frequency/angular-frequency quantities may use SI-derived units.

Document whether an API expects wavelength, frequency, or angular frequency rather than treating them as interchangeable scalars.

## Electromagnetic fields

Electromagnetic field tensors are generally complex-valued in frequency-domain and phasor representations. Field amplitude, magnitude, intensity, power/flux, and normalized modal coefficients are distinct quantities.

Do not describe a field-derived tensor as "normalized" without identifying the normalization operation and reference quantity.

## Source and mode normalization

MAPS workflows can apply several different normalizations, including source-profile normalization, eigenmode normalization, source-power calibration, and FDTD phasor/source-spectrum normalization. These operations serve different purposes and must not be combined or removed merely because they appear numerically redundant.

For three-dimensional FDTD-X surface observables, physical cell areas are expressed in square meters and are never rescaled independently for each monitor. Flux and eigenmode overlap must therefore retain the physical ratios among `dx * dy`, `dx * dz`, and `dy * dz` areas. Dividing each plane's weights by its own mean area is invalid because it makes cross-axis transmission depend on mesh anisotropy.

MAPS installs this convention over the pinned public FDTD-X package through `thirdparty.fdtdx`. Project code must import the compatibility layer before using FDTD-X numerical entry points; local changes to an external FDTD-X checkout are not part of the MAPS runtime contract.

Eigenmodes are normalized with the same surface quadrature used for simulated fields. A modal overlap returning power divides the squared overlap by the mode power evaluated under that quadrature.

Symmetry is a separate operation. A source or mode normalized on a retained domain with `k` transverse symmetry axes is scaled in field amplitude by `1 / sqrt(2**k)` before or during full-domain reconstruction. The unfolded full-domain fields are then integrated once with ordinary cell-area weights; surface quadrature must not apply another symmetry multiplicity.

Temporal source-spectrum normalization is also separate from spatial power normalization. Frequency-domain field amplitudes divide by the complex source spectrum at each frequency, while powers divide by its magnitude squared.

### Three-dimensional surface-power convention

For a monitor normal to axis `n`, MAPS evaluates time-averaged power as

$$
P_n = \frac{1}{2}\operatorname{Re}\sum_{i,j}
  (\mathbf{E}_{i,j}\times\mathbf{H}_{i,j}^{*})_n\,\Delta A_{i,j}.
$$

The area is selected from the two transverse grid widths:

```text
x-normal plane: Delta A = Delta y * Delta z
y-normal plane: Delta A = Delta x * Delta z
z-normal plane: Delta A = Delta x * Delta y
```

Every `Delta A` is expressed in square meters. This remains true on a uniform grid: the weight is the physical spacing squared, not an implicit unit weight and not `area / mean(area)`.

The modal overlap uses the same `Delta A` as the flux integral. If the reference mode has physical power `P_mode`, MAPS returns modal amplitude `a / sqrt(P_mode)` or modal power `|a|^2 / P_mode`. Consequently, reference modes do not have to rely on an assumed unit-power convention from another grid or solver.

### Reported transmission

An FDTD-X eigenmode objective is a modal power computed from source-spectrum-normalized phasors and physical surface quadrature. Device-level examples may additionally divide output modal power by measured incident power, for example `input_flux + refl_flux`, to remove residual launch calibration error. Total flux and modal transmission are different observables and must have different labels:

```text
modal transmission: selected output-mode power / incident power
total transmission:  output Poynting flux / incident power
```

Do not label total flux as `|s21|^2`. For a straight-through calibration, compare output modal power to modal power measured at an input reference plane after the source. Do not infer propagation loss in dB/cm from the raw output divided by a nominal source setting.

### Validation evidence

The physical-area change corrected a cross-axis grating-coupler bias caused by independently mean-normalizing an `xy` source monitor and a `yz` output monitor. In the SiN grating-coupler example, the mean-area ratio was approximately `2.08875`; a previously reported iteration-9 modal transmission near `0.323` therefore became approximately `0.675`, consistent with an independent native Tidy3D result near `0.7`. These values describe that run and are validation evidence, not universal golden values for the example.

A lossless straight-waveguide diagnostic measured source-plane and output modal powers of approximately `0.994731` and `0.994736`. Their ratio was `1.000005`, showing that the common absolute offset was launch/overlap calibration rather than physical propagation loss. Because the material model used real permittivity and no roughness, its physical propagation loss is zero; realistic dB/cm loss requires an absorptive, scattering, leakage, or empirical loss model and should be extracted from a multi-length fit.

## Objective normalization

Objective terms may represent transmission, reflection, flux, overlap, similarity, penalties, or other figures of merit. Their scale and normalization are defined by the objective implementation and configuration. Do not assume every transmission-like value is automatically a probability in `[0, 1]`.

## Design variables

A learnable design parameter is not necessarily a physical material value. Distinguish among:

```text
learnable parameter
normalized density / rho
projected or fabrication-transformed density
material property / relative permittivity
full simulation material map
```

For density-style parameterizations, document the expected range (commonly `[0, 1]`) where the implementation enforces or assumes it.

## Gradients

Gradient scaling and normalization are optimization conventions, not physical unit conversions. If a workflow applies gradient normalization, clipping, or rescaling, document the rule and where it occurs relative to objective differentiation and optimizer update.

## Documentation requirement

For scientific APIs, specify as applicable:

- physical unit;
- normalized versus dimensional representation;
- reference quantity used for normalization;
- conversion performed at backend boundaries;
- expected numerical range when meaningful;
- whether the returned value is amplitude-, intensity-, power-, or objective-normalized.

When the implementation does not establish a reliable unit or normalization convention, document that uncertainty rather than inventing one.
