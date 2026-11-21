# Design-Region Parametrization and Transform Contracts

This page documents the inverse-design parametrization stack under `core/invdes/models/layers/parametrization/` for the `v0.0.2_fdtdx` baseline, including the opt-in digital-box and circular-primitive material extensions.

## Role in the inverse-design pipeline

Each named design region is represented by an `nn.Module` parametrization. The parametrization maps learnable variables to a normalized design density and ultimately to material/permittivity values used by the simulation.

Conceptually:

```text
learnable parameters
       |
       v
build_density(weights, sharpness)
       |
       v
high-resolution normalized density
       |
       v
ordered density transforms
  symmetry / binarize / roughness /
  lithography / etch / blur / FFT / ...
       |
       v
resolution conversion / extrusion
       |
       v
permittivity/material map in design region
       |
       v
blend into full simulation map
```

For scientific reproducibility, the ordering of these transforms is part of the optimization definition.

## `parametrization_builder()`

```python
parametrization_builder(
    device,
    hr_device,
    sim_cfg,
    parametrization_cfgs,
    operation_device,
    **kwargs,
)
```

Builds an `nn.ModuleDict` keyed by design-region name.

### Configuration schema

`parametrization_cfgs` is a mapping:

```python
{
    "design_region_1": {
        "method": "levelset",
        ...
    },
    "design_region_2": {
        "method": "box",
        ...
    },
}
```

The baseline registry supports:

```text
"levelset" -> LevelSetParameterization
"box"      -> BoxParameterization
"circle"   -> CircleParameterization
"cylinder" -> CylinderParameterization (z-axis footprint)
"edgebox"  -> EdgeBoxParameterization
"polygon"  -> PolygonParameterization
```

Unknown methods raise `ValueError`.

Each parametrization receives the same `device`, `hr_device`, `sim_cfg`, region name, region-specific configuration, and operation device.

### Digital box and circle material-to-shape stages

The `box` parameterization can opt into per-primitive material variables by
setting `material_optimization=True` in `geometry_cfgs` (or at the region
configuration level). An optional `material_init` defaults to `0.5`. The box
module then exposes `set_optimization_stage("material" | "shape" | "joint" |
"frozen")`; centers remain fixed unless `optimize_centers=True` is supplied.
Set `material_background_value=1.0` when boxes represent lower-index etched
cells in a fixed high-index host; the default `0.0` retains the legacy
high-index-box-in-low-index-background convention.

`init_method="2d_array_full"` initializes every lattice box at its maximum
cell size. After material optimization, call `transfer_material_to_shape()`
with explicit host/inclusion endpoints when needed. The default linear-
permittivity transfer uses isotropic area scaling, `size = size_max * sqrt(f)`;
`x_only` and `y_only` mappings are also available. Circle material transfers
use the same linear-permittivity relation; circle area fractions scale the
current radius with `radius = radius_current * sqrt(f)`. A circle's learnable state is
`weights["circles"].centers`, `weights["circles"].radius`, and, when enabled,
`weights["circles"].material_rho`. The `cylinder` alias stores the equivalent
module under `weights["cylinders"]`. Circle/cylinder radius and center
gradients are aggregate coverage derivatives and use the configured uniform
`phi_resolution` grid. Neighboring primitives should be kept non-overlapping
unless the sequential visibility rule is explicitly intended.

`coverage_samples` controls the Torch area-fraction fallback (default 8 samples
per cell axis), and `gradient_step_um` controls the central finite-difference
step used for geometry gradients. Setting `gpu_subpixel=True` rasterizes
z-axis cylinders as regular `PolySlab` polygons with `gpu_circle_sides`
(default 128) sides through the CuPy/Tidy3D subpixel rasterizer. The GPU path
uses one unit-thickness z cell for the intrinsic 2-D footprint and applies the
same Cartesian y-axis orientation as the Torch path.

`max_radius_um`, when provided, is an upper bound on the effective radius. The
implementation also enforces the lattice-derived non-overlap bound, so the
actual upper limit is the smaller of those two values. The after-step radius
projection uses the same limit.

With `small_circle_policy="remove"`, radii below `min_radius_um` are omitted
from the forward geometry. With `small_circle_policy="clamp"`, they are
raised to `min_radius_um`. The optional
`small_circle_policy="snap_by_radius"` combines
both behaviors: radii below `min_radius_um / 2` are removed, while radii from
`min_radius_um / 2` up to `min_radius_um` are raised to `min_radius_um`.
The alternative `small_circle_policy="snap_by_area"` applies the same rule in
area space: areas below half the area of a circle with radius `min_radius_um`
are removed, which corresponds to a radius cutoff of
`min_radius_um / sqrt(2)`.
A removed circle would otherwise have zero ordinary radius derivative and could
never recover, so the cell containing its center receives
`zero_radius_gradient_scale` (default `1 / grid_step`) in the radius backward
pass for the `remove`, `snap_by_radius`, and `snap_by_area` policies. Set this
scale to zero to disable the recovery surrogate.

For the partially etched Y-branch example, where full silicon is the host and
the circle is the lower-index etched material, Stage 2 must call
`transfer_material_to_shape()` with `eps_host=eps_full_silicon` and
`eps_inclusion=eps_partial_etch`; this makes low Stage-1 `material_rho` produce
larger etched circles. The helper
`prepare_stage2_y_branch(invdes)` in
`core/invdes/examples/digital_y_branch_circle.py` performs this transition and
enables radius-only optimization.

The box parameterization accepts `phi_resolution` as a scalar or two-element
`(x, y)` value in pixels per micrometer. Boxes are rasterized on this uniform
internal grid and the physical permittivity is then sampled onto the device
grid using physical coordinates. The current rectangle rasterizer requires an
isotropic `phi_resolution`. Set `gpu_subpixel=True` to use the CuPy/Tidy3D
geometric area/volume-average rasterizer on this uniform grid. This mode requires
CUDA, CuPy, and Tidy3D, uses one z cell for 2D regions, and retains the existing
custom approximate backward pass.

Box sizes are independently optimized along x and y by default. Set
`share_size=True` to make every box square; the x size is the shared learnable
value, the y size is synchronized to it, and the backward pass aggregates both
axis contributions into that shared value. `min_size_um` and `max_size_um` may
be scalars or `(x, y)` pairs. The effective maximum is the smaller of
`max_size_um` and the lattice-derived per-axis maximum.

`small_box_policy="remove"` removes a box if either side is below its minimum,
while `"clamp"` independently raises each side to its minimum. The
`"snap_by_area"` policy removes boxes whose area is below half the minimum
legal box area, otherwise raising undersized sides to their minima. The
`"snap_by_min_side"` policy instead removes boxes whose smaller physical side
is below half the smaller configured minimum side, otherwise raising undersized
sides to their minima. Size constraints are applied in the forward rasterizer,
EMT material-to-size transfer, and after-step projection.

## `BaseParametrization`

The base class owns region metadata and defines the common transform contract.

Representative baseline configuration:

```python
{
    "method": "levelset",
    "rho_resolution": [50, 0],
    "transform": {},
    "init_method": "random",
    "denorm_mode": "linear_eps",
    "dims": (0, 1),
    "extrude_direction": "-",
    "extrude_angle": 0.0,
    "reference_plane": "middle",
}
```

### Region masks

The parametrization stores both simulation-resolution and high-resolution masks:

```text
design_region_mask    = device.design_region_masks[region_name]
hr_design_region_mask = hr_device.design_region_masks[region_name]
```

It may also consume per-axis coverage weights from the corresponding device.

### `dims`

`dims` identifies the physical axes described by the intrinsic parametrization. The default `(0, 1)` means the density is a 2D representation over axes 0 and 1. For a 3D simulation, the remaining physical axis is the extrusion axis.

Do not assume that a 2D density tensor is always an image-space `(H, W)` object; its dimensions represent the physical axes selected by `dims`.

### Sidewall extrusion support

For a nonzero `extrude_angle`, the design-region mask bounds the optimized
reference-plane density but does not clip the fabricated 3D geometry. The
parametrization derives a lateral halo from the sidewall-radius schedule,
composes the optimized density with fixed full-device geometry on the selected
`reference_plane`, and performs morphology over that expanded context. The
expanded result is merged back over a separate support-region mask so dilation
may extend outside the design region and fixed geometry in the halo is not
erased. A zero angle retains the design-region-only fast path.

## Learnable parameter storage

`build_parameters()` dispatches to a method-specific builder registered by the concrete parametrization.

The builder returns:

```text
weight_dict
param_dict
```

`weight_dict` must contain either:

- all `nn.Parameter` values, stored as `nn.ParameterDict`, or
- all `nn.Module` values, stored as `nn.ModuleDict`.

Mixing parameters and modules in the same returned dictionary is rejected.

`param_dict` is retained as non-weight parameterization state in `self.params`.

## Density and permittivity contract

`build_density(weights, sharpness, **kwargs)` is the core abstract operation. The source explicitly defines its intended result as a **normalized design density in `[0, 1]`**, typically at high resolution.

The base implementation of `build_permittivity()` delegates to `build_density()`. Concrete parameterizations or later processing may convert normalized density into physical permittivity according to their material/denormalization rules.

The distinction is important:

```text
density  != automatically physical epsilon
```

Code and documentation should identify which representation is being passed at each stage.

## Ordered density transforms

`density_transform()` applies the configured transform sequence to a high-resolution 2D normalized density. The baseline transform registry includes:

```text
mirror_symmetry
transpose_symmetry
rotation_symmetry
convert_resolution
litho
etching
blur
fft
```

In addition, `density_transform()` handles `binarize` and `add_roughness` as special transform types.

The transform list is processed in order. Reordering transformations can change both the forward geometry and its gradient and therefore constitutes a scientific behavior change.

## Binarization

For a transform with:

```python
{"type": "binarize", ...}
```

MAPS applies the parametrization's binary projection using the current optimization `sharpness` and `eta`.

Conceptually:

```text
rho_projected = binary_projection(rho, sharpness, eta)
```

The exact projection equation is defined by the configured projection layer; do not replace it with a generic sigmoid/Heaviside formula in documentation unless that layer has been inspected.

A transform may specify `preserve_boundaries`. Supported boundary labels are:

```text
"x"
"y"
```

After transformation, those outer rows/columns are restored from the pre-transform density.

## Etching transform

The etching helper applies binary projection to transformed density using explicit `sharpness` and `eta` tensors on the same device as the density.

The first element in the transform-list representation is retained unchanged, while subsequent density representations are projected.

This is an algorithmic convention of the current transform pipeline; do not collapse multiple density representations without tracing their downstream use.

## Symmetry transforms

### Mirror symmetry

`mirror_symmetry()` constructs mirrored density patterns along selected dimensions.

A custom autograd function, `DecayGradient`, leaves the forward value unchanged but scales its backward gradient:

```text
forward:  y = x
backward: dL/dx = decay_factor * dL/dy
```

Mirror-generated shared regions use a decay factor of `0.5` so duplicated symmetric contributions do not simply double-accumulate gradients into the shared free parameter region.

Both even and odd spatial sizes are handled; odd sizes retain a center row/column.

### Transpose symmetry

Transpose symmetry requires a square 2D density. It combines triangular portions of the density and its transpose, then applies a 90-degree rotation specified by `rot_k`.

Non-square inputs are rejected.

### Rotation symmetry

Rotation symmetry also requires a square matrix.

Supported modes:

```text
rot_k = 1 -> 90-degree rotational symmetry
rot_k = 2 -> 180-degree rotational symmetry
```

Odd-sized arrays preserve a center pixel/row according to the explicit construction in the implementation. Other rotation modes raise `ValueError`.

## Resolution conversion

`_convert_resolution()` operates on the last two dimensions of a density tensor.

If `target_size` is not explicitly supplied, the target dimensions are computed approximately as:

```text
target_n = round(source_n * target_resolution / source_resolution)
```

with a minimum of one cell in the internal conversion helper.

### Tensor rank

The helper explicitly handles:

```text
2D: [Nx, Ny]
3D: [C, Nx, Ny] or another leading-dimension interpretation
```

Only the last two dimensions are resized.

### Subpixel-smoothing downsampling

When both target dimensions are smaller and `subpixel_smoothing=True`, the normalized density is first converted to permittivity:

```text
epsilon = eps_bg + (eps_r - eps_bg) * rho
```

The permittivity is downsampled with adaptive average pooling, then converted back to normalized density:

```text
rho = (epsilon_avg - eps_bg) / (eps_r - eps_bg)
```

This is materially different from nearest-neighbor density resampling and is intended to preserve an averaged material representation during downsampling.

## Lithography transform

The lithography path embeds the design-region density into a full reference map, converts that map to resolution `310`, pads it to lithography tiles of size `620`, evaluates the differentiable lithography model, removes padding, resizes back to the original full-map shape, and extracts the design region.

The source comments identify the lithography model as a reimplementation associated with arXiv:2411.07311. Future formal documentation should add the full paper citation after verifying the intended publication metadata.

Important representation contract:

- input is expected to be a normalized density/pattern;
- the transform temporarily operates on the complete device map because lithographic effects near the design-region boundary depend on surrounding geometry;
- the final value returned to the parameterization pipeline is again the transformed design-region portion.

## Minimum-feature-size transforms

The baseline contains blur- and FFT-based transforms intended for minimum-feature-size (MFS) control.

`mfs` is expressed in physical units and combined with the configured resolution to determine the filtering scale. Because the exact filter/kernel behavior is implementation-specific, downstream code should not infer an MFS guarantee from the parameter name alone.

These transforms can depend on a full-device reference map and design-region mask, not only the isolated density tensor.

## Roughness model

The base class provides `_correlated_roughness_2d()` for a 2D correlated random field.

Inputs:

`density`
: 2D design density whose shape determines the roughness grid.

`A`
: Requested roughness standard-deviation scale. Must satisfy `A >= 0`.

`Lc`
: Correlation-length parameter. Must satisfy `Lc > 0`.

`seed`
: Optional deterministic random seed. If omitted, a random seed is generated once and retained on the parametrization instance for reuse.

The physical spacing is derived from the design-region physical size divided by `(Naxis - 1)`.

The implementation creates white Gaussian noise, Fourier transforms it, and applies the Gaussian spectral filter:

```text
G(fx, fy) = exp[-0.5 * (2*pi*Lc)^2 * (fx^2 + fy^2)]
```

The inverse-transformed real field is mean-centered, normalized by its standard deviation, and scaled by `A`.

Thus the returned roughness tensor has the same shape and dtype as the 2D density and approximately zero mean with standard deviation `A` up to numerical/sample effects.

`add_roughness` configuration accepts either:

```text
A
Lc
```

or aliases:

```text
amplitude -> A
correlation_length -> Lc
```

Both parameters are required.

## Transform reference maps

Transforms such as lithography, blur, and FFT may need context outside the isolated design region. `_uniform_transform_reference()` builds a uniform 2D reference representation from the high-resolution device/export map.

If the source map is 3D, the non-parametrized/extrusion dimension is reduced by selecting its middle plane.

The routine returns:

```text
reference_map
target_design_region_mask
effective_resolutions
```

This contract allows transforms to act with surrounding-device context while still returning a design-region density.

## 2D-to-3D semantics

The source states that the high-resolution input density to `density_transform()` is always a 2D normalized design density. For a 3D physical device, an additional extrusion operation is applied later in the parameterization pipeline.

Therefore an AI agent must not insert a singleton z dimension early merely because the simulation backend is 3D. The parameterization rank and simulation-map rank are intentionally distinct stages.

## Representative configuration

A representative design-region configuration can be structured as:

```python
design_region_param_cfgs = {
    "design_region_1": {
        "method": "levelset",
        "rho_resolution": [50, 0],
        "init_method": "random",
        "denorm_mode": "linear_eps",
        "dims": (0, 1),
        "reference_plane": "middle",
        "extrude_angle": sidewall_angle,
        "transform": [
            {"type": "mirror_symmetry", "dims": [0]},
            {"type": "binarize"},
        ],
    }
}
```

This illustrates the schema and ordered-transform concept. Exact transform keyword arguments must match the selected transform implementation.

## Scientific invariants for future changes

Preserve or explicitly validate:

- normalized-density versus physical-permittivity semantics;
- physical meaning of `dims`;
- design-region and high-resolution mask alignment;
- transform ordering;
- symmetry gradient scaling;
- sharpness/eta projection behavior;
- full-device context used by fabrication transforms;
- subpixel-smoothing averaging in permittivity space;
- 2D parameterization versus 3D extrusion stage;
- random roughness seed/reproducibility behavior;
- differentiability of all transforms intended for optimization.
