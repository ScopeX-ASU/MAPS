# Base Optimization and Material-Map Contracts

This page documents the shared optimization-model infrastructure in `core/invdes/models/base_optimization.py` for the `v0.0.2_fdtdx` baseline. It focuses on the interface between design-region parametrizations, material maps, simulation objectives, and multi-resolution/thermal grids.

No source behavior is changed by this documentation.

## `DefaultSimulationConfig`

The baseline simulation defaults are:

```text
solver: "ceviche"
binary_projection:
    fw_threshold: 100
    bw_threshold: 100
    mode: "regular"
border_width: [0, 0, 6, 6]
PML: [1, 1]
cell_size: None
resolution: 50
wl_cen: 1.55
wl_width: 0
n_wl: 1
plot_root: "./figs/bending"
```

These values are defaults, not universal physical conventions. Device/application code commonly overrides them.

## `DefaultOptimizationConfig`

The optimization configuration contains three major sections:

```text
design_region_param_cfgs
sim_cfg
obj_cfgs
```

`design_region_param_cfgs`
: Mapping from design-region names to parameterization configurations. See [Design-region parametrization](parametrization.md).

`sim_cfg`
: Simulation configuration used by the device/objective layers.

`obj_cfgs`
: Objective definitions plus optional `_fusion_func` controlling how individual objective terms are combined.

## Objective fusion

The default `_sum_objectives()` loops over the breakdown mapping and computes:

```text
total_obj = sum(weight_i * value_i)
```

for every entry whose name does **not** contain `"smatrix"`.

Each included breakdown entry therefore needs at least:

```text
"weight"
"value"
```

The default fusion function returns:

```text
(total_obj, extra_breakdown)
```

where `extra_breakdown` is currently an empty dictionary.

A custom fusion function is expected to consume the breakdown mapping and return the same two-value structure.

## `BaseOptimization`

```python
BaseOptimization(
    device,
    hr_device,
    design_region_param_cfgs={},
    sim_cfg={},
    obj_cfgs={},
    operation_device=torch.device("cuda:0"),
    verbose=True,
)
```

### Device roles

`device`
: Simulation-resolution physical device. Supplies `epsilon_map`, design-region masks/configurations, source/monitor definitions, grid metadata, and normalization runs.

`hr_device`
: Higher-resolution copy/representation used for parameterization and high-resolution geometry/material processing.

`operation_device`
: PyTorch device on which differentiable maps and design parameters are stored/evaluated.

### Permittivity maps

The constructor creates:

```text
self.epsilon_map = torch.from_numpy(device.epsilon_map).to(operation_device)
self.hr_eps_map  = torch.from_numpy(hr_device.epsilon_map).to(operation_device)
```

Their dimensionality follows the physical device: common paths may be 2D or 3D. Design-region parameterizations can remain intrinsically 2D even when the full device map is 3D.

## Initialization sequence

Construction performs several scientifically meaningful setup operations in order:

```text
1. build_parameters()
2. device.init_monitors()
3. device.norm_run()
4. save device.port_sources_dict as norm_run_profiles
5. build_objective(...)
```

This means creating a `BaseOptimization` subclass is not a passive data operation: source/monitor setup and normalization simulation work may occur during initialization.

## Design-region parameter modules

`build_parameters()` calls:

```python
parametrization_builder(
    device=device,
    hr_device=hr_device,
    sim_cfg=sim_cfg,
    parametrization_cfgs=design_region_param_cfgs,
    operation_device=operation_device,
)
```

and stores the result as:

```text
design_region_param_dict: nn.ModuleDict
```

with one module per named design region.

It also constructs:

```python
self.objective_layer = SimulatedFoM(self.cal_obj_grad, self.sim_cfg["solver"])
```

which links the optimization model to simulation-based objective/gradient evaluation.

## Objective construction inputs

`build_objective()` receives device-level simulation metadata including:

```text
port_profiles              = device.port_sources_dict
port_slices                = device.port_monitor_slices
port_slices_native         = device.port_monitor_slices_native
port_slices_native_symmetry= device.port_monitor_slices_native_symmetry
port_slices_info           = device.port_monitor_slices_info
grid_metadata              = device.fdtdx_native_grid_metadata
cell_weights               = device.fdtdx_native_cell_weights
epsilon_map                = device.epsilon_map
obj_cfgs                    = configured objective mapping
solver                      = sim_cfg["solver"]
```

These mappings are part of the contract between device geometry, simulation backends, and objective evaluation.

## Design-region blending

`_blend_design_region(eps_map, region_mask, region_eps, region_weight=None)` inserts a region tensor into the full map.

Without a weight:

```text
eps_map[region_mask] = region_eps
```

With a coverage weight `w`:

```text
eps_new = eps_base * (1 - w) + eps_region * w
```

The weight tensor must have exactly the same shape as `region_eps`; otherwise the operation raises `ValueError`.

This weighted blending is important for partial-cell/rectilinear coverage and should not be replaced by hard assignment without checking grid semantics.

## Coverage weights

`_coverage_weight_from_axes(axis_weights, exclude_axes=())` builds a multidimensional coverage tensor as the outer/product combination of one-dimensional axis weights.

For excluded axes, the weight is replaced by ones. This is used when a lower-dimensional design parametrization is embedded into a higher-dimensional physical region and coverage along the extrusion axis should not attenuate the design value.

## Region axes and extrusion

For a 3D design-region mask, `_region_exclude_axes()` reads the parameterization's configured `dims`, sorts them, and treats the remaining physical axes as excluded from parameterization coverage weighting.

For example:

```text
physical axes: (0, 1, 2)
parametrization dims: (0, 1)
exclude axes: (2,)
```

This connects the 2D design density to the 3D device map without confusing the parameterization plane with tensor image dimensions.

## Applying material/property maps

`_apply_region_property_map()` inserts one design-region property tensor into a full-device property map.

Inputs conceptually include:

```text
base_map: full-device Tensor or None
region_name: design-region key
region_values: Tensor or None
target_device: device defining target masks/grid
region_masks: optional alternate masks
resize_property_name: controls property-specific resampling
```

If either `base_map` or `region_values` is `None`, the base map is returned unchanged.

The region mask is clipped/fitted to the full map shape. If the region tensor shape differs from the target mask shape, the tensor is resampled according to the property semantics before blending.

## Property-specific resampling

The baseline intentionally uses different resampling rules for different physical quantities.

### Generic fields/properties

2D tensors use bilinear interpolation; 3D tensors use trilinear interpolation.

Input/output shapes:

```text
2D: [Nx, Ny] -> [Nx_target, Ny_target]
3D: [Nx, Ny, Nz] -> [Nx_target, Ny_target, Nz_target]
```

Temporary batch/channel dimensions are introduced internally for PyTorch interpolation and removed before return.

### Conductivity

For property name `"conductivity"`, resampling is performed in **resistivity/reciprocal space**:

```text
r = 1 / conductivity
```

The reciprocal field is averaged/interpolated, then converted back:

```text
conductivity_resampled = 1 / r_resampled
```

Values are clamped away from zero using the dtype's smallest positive representable value.

This is a physical averaging convention and must not be replaced with direct arithmetic interpolation of conductivity as a cosmetic simplification.

### Heat capacity and electrical conductivity

For:

```text
"heat_capacity"
"electrical_conductivity"
```

MAPS uses average-preserving resizing. When downsampling, adaptive average pooling is used; when upsampling, the generic interpolation path is used.

Note that the code distinguishes property names `"conductivity"` and `"electrical_conductivity"`; they follow different resampling paths. Preserve this distinction unless the physical model is intentionally redesigned.

## Coordinate-aware thermal resampling

When thermal design-region masks and thermal coordinates are available, material properties are resampled between **physical coordinate grids** rather than only by target tensor shape.

Source coordinates are extracted from the optical/device design-region mask; target coordinates are extracted from the thermal grid/mask.

Coordinates are normalized to `[-1, 1]` for `torch.nn.functional.grid_sample()`.

### 2D coordinate sampling

The implementation constructs a 2D normalized grid and internally permutes axes to satisfy PyTorch grid-sampling layout. The returned tensor is restored to MAPS physical-axis ordering.

### 3D coordinate sampling

The same process is extended to `(x, y, z)` coordinates with a 3D sampling grid.

Only 2D and 3D region tensors are supported by this path; other ranks raise `ValueError`.

## Runtime thermal material maps

For current-controlled devices, `_build_runtime_thermal_material_maps()` derives thermal material properties from the normalized design density.

For each region:

```text
density = design_region.normalize_permittivity(region_eps)
```

and the device converts density into properties such as:

```text
conductivity
thermo_optic_coeff
heat_capacity  (when enabled)
```

This establishes an important multiphysics contract:

```text
optimization parameters
 -> optical permittivity
 -> normalized material density
 -> thermal/electrical material properties
```

The same density therefore couples optical and thermal/electrical models.

## Thermal-state snapshots

Runtime objective evaluation may expose `objective.runtime_thermal_states`. `BaseOptimization` copies these states into `_thermal_states` and recursively calls `retain_grad()` on differentiable tensors.

If exactly one thermal state exists, it is also exposed through `_thermal_state` as a convenience alias.

This behavior allows plotting/inspection of temperature, epsilon, or other coupled fields without discarding gradients prematurely.

## Grid selection for plotting

The optimization layer distinguishes optical-native and thermal grids.

Optical-grid fields include:

```text
epsilon
electrical_conductivity
optical_temperature
thermo_optic_coeff
```

Thermal-grid fields include:

```text
temperature
conductivity
heat_capacity
```

Unknown field keys raise `ValueError` rather than silently choosing a grid.

For 3-D FDTD-X runs with both `MAPS_SPARSE_PHASOR=1` and visualization-plane
recording enabled, plotting resamples only the requested `xy`, `xz`, and `yz`
cross-sections. Each cross-section uses the two native samples required by the
existing rectilinear interpolation rule, and the resulting 2-D tensors are
moved to CPU before Matplotlib runs. This avoids full native-to-export volume
intermediates for vector fields, epsilon, epsilon gradients, and optional
thermal maps. Other solver and plotting modes retain full-volume resampling.

## Reset semantics

`reset_parameters()` delegates to every design-region parametrization.

`reset_levelset_sdf()` only acts on regions whose parametrization implements `reset_levelset_sdf()`. This supports level-set-specific reinitialization without requiring all parameterization families to expose that operation.

## Scientific invariants for future changes

Preserve or explicitly validate:

- objective breakdown weighting and fusion;
- initialization order, including normalization runs;
- simulation-resolution versus high-resolution device roles;
- physical-axis meaning of design-region masks;
- partial-cell coverage blending;
- property-specific resampling laws;
- coordinate-aware optical-to-thermal mapping;
- conductivity/resistivity averaging convention;
- coupling from optical density to thermal/electrical properties;
- differentiability and retained gradients across multiphysics states.
