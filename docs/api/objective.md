# Objective and Figure-of-Merit Contracts

This page documents the simulation-to-objective layer implemented in `core/invdes/models/layers/objective.py` for the `v0.0.2_fdtdx` baseline. It is documentation-only and does not modify objective behavior.

## High-level role

The objective layer consumes simulated electromagnetic fields and converts them into differentiable scalar/vector figures of merit (FoMs), S-parameters, fluxes, shape similarities, phase statistics, and fabrication/physics penalties.

The central flow is:

```text
permittivity / design variables
        |
        v
simulation(s)
        |
        v
fields keyed by (input slice, wavelength, mode, control state)
        |
        v
ObjectiveFunc.Js[*]["fn"]
        |
        v
per-objective value + weight
        |
        v
breakdown
        |
        v
_fusion_func(breakdown)
        |
        v
total objective
```

`BaseOptimization` uses this total objective as the quantity returned by its forward pass, and `InvDesign` subsequently applies its own optimization sign convention.

## Canonical field-key schema

The objective code expects solved fields to be indexed by:

```python
(in_slice_name, wl, in_mode, temp)
```

where:

`in_slice_name`
: String identifying the source/input port or slice.

`wl`
: Scalar wavelength after expansion of a broadband fdtdx frequency group. For 2D FDFD this is the scalar wavelength stored in the simulation key.

`in_mode`
: Mode identifier. Existing code uses strings such as `"Ez1"`; its first two characters are interpreted as the polarization/component family (`"Ez"` or `"Hz"`).

`temp`
: Control-state key. In legacy thermal mode this can represent temperature; in current-control/thermal-coupling workflows it may be a tuple describing currents.

### 2D field payload

For a 2D `Ez` source:

```python
fields[(slice_name, wl, mode, temp)] = {
    "Hx": Hx,
    "Hy": Hy,
    "Ez": Ez,
}
```

For a 2D `Hz` source:

```python
fields[(slice_name, wl, mode, temp)] = {
    "Ex": Ex,
    "Ey": Ey,
    "Hz": Hz,
}
```

### 3D fdtdx field payload

The FDTD path stores all six components:

```python
{
    "Ex": Ex,
    "Ey": Ey,
    "Ez": Ez,
    "Hx": Hx,
    "Hy": Hy,
    "Hz": Hz,
}
```

Each wavelength is split out of the broadband simulation and inserted under its own scalar-`wl` field key. See [FDTD API](fdtd.md) for the underlying tensor layout before wavelength splitting.

## Simulation-key schema

The objective code supports both:

### 2D-style simulation keys

```text
(wl, pol, temp)
```

### fdtdx 3D broadband keys

```text
((wl_cen, wl_width, n_wl), pol, temp)
```

For a broadband key, the implementation expands the wavelength group as:

```python
wls = np.linspace(
    wl_cen - wl_width / 2,
    wl_cen + wl_width / 2,
    n_wl,
)
```

Objective matching uses `math.isclose(..., abs_tol=1e-4)` against requested target wavelengths.

## Port-profile schema

Port profiles are dictionaries indexed by slice name and `(wl, mode)`:

```text
port_profiles[slice_name][(wl, mode)]
```

The objective code expects the stored value to unpack as:

```text
(source, ht_m, et_m, norm_power, require_sim)
```

where:

`source`
: Source/profile representation used to run the simulation.

`ht_m`, `et_m`
: Magnetic/electric mode-profile data required by eigenmode coefficient calculations.

`norm_power`
: Source normalization quantity used to normalize output power/amplitude for 2D objectives. FDTD-X 3D paths use `1` at the objective layer because phasor/source normalization is handled upstream.

`require_sim`
: Boolean indicating whether this profile must be included in the simulation queue.

Do not reduce this five-element structure to a generic tuple without preserving the element meanings.

## Objective configuration schema

`ObjectiveFunc.add_objective()` receives a configuration mapping containing objective entries plus a required fusion function:

```python
{
    "_fusion_func": fusion_function,
    "objective_name": {
        "type": "eigenmode",
        "in_slice_name": "in_slice_1",
        "out_slice_name": "out_slice_1",
        "in_mode": "Ez1",
        "out_modes": ("Ez1",),
        "direction": "x+",
        "wl": (1.55,),
        "temp": (...),
        # objective-specific fields...
    },
}
```

Common fields are:

`type`
: Objective implementation selector.

`in_slice_name`
: Source/input slice. Some objective classes accept a tuple/list of input slices.

`out_slice_name`
: Target/output slice. Some objectives accept multiple output slices.

`in_mode`
: Source mode identifier.

`out_modes`
: One or more output mode identifiers for eigenmode/shape objectives.

`direction`
: Propagation and side selection. For a single direction this is commonly a string such as `"x+"`; multi-output S-matrix objectives accept corresponding tuples.

`wl`
: Target wavelength collection.

`temp` or `control_keys`
: Target control-state collection. Current code accepts either a legacy `temp` field or an explicit `control_keys` field.

`weight`
: Outer optimization weight stored in the objective breakdown. Objective implementation itself generally computes the raw value; `BaseOptimization`'s fusion function combines `weight * value`.

`requires_grad`
: Defaults to `True`. If false, `ObjectiveFunc.obtain_objective()` detaches a tensor-valued objective value before putting it into the breakdown.

`requires_adjoint`
: Used when constructing fdtdx adjoint groups. Defaults to `True` when absent.

## Supported objective types

The baseline dispatch supports the following objective types in `add_objective()`:

```text
eigenmode
smatrix
flux
flux_minus_src
flux_near2far
response_record
intensity_shape
intensity_shape_near2far
interface_penalty
structure_simplify_penalty
```

Additional historical objective code may exist below the main dispatch or in helper modules; this page documents only types actively instantiated by the current `add_objective()` path.

---

## Eigenmode objective

### Purpose

`EigenmodeObjective` evaluates modal overlap/energy at one output slice across selected wavelengths, temperatures/control states, source modes, and output modes.

### Representative configuration

```python
{
    "type": "eigenmode",
    "weight": 1.0,
    "in_slice_name": "in_slice_1",
    "out_slice_name": "out_slice_1",
    "in_mode": "Ez1",
    "out_modes": ("Ez1", "Ez2"),
    "direction": "x+",
    "wl": (1.50, 1.55, 1.60),
    "temp": (300.0,),
    "energy": True,
}
```

### Direction

The implementation chooses either forward or backward coefficient according to the second character of `direction`:

```text
direction[0] -> propagation axis, e.g. x/y

direction[1] == "+" -> use s_p

direction[1] == "-" -> use s_m
```

Invalid signs raise `ValueError`.

### Output structure

The objective appends one result per matching `(wavelength, output_mode, control_state)` combination and returns a stacked tensor/array rather than reducing to a scalar internally.

For a tensor-valued result:

```text
shape = [N_selected_evaluations]
```

where the exact count depends on the selected wavelengths, control states, output modes, and source/polarization filtering.

With `obj_type="eigenmode"`, normalized `s_p` and `s_m` values are recorded in:

```python
s_params[(in_slice, out_slice, out_mode, wl, in_mode, temp)] = {
    "s_p": ...,
    "s_m": ...,
}
```

---

## S-matrix objective

`SMatrixObjective` generalizes eigenmode evaluation to multiple input and output slices.

### Input/output collections

The constructor expects:

```text
in_slice_names
out_slice_names
out_modes
directions
```

and requires:

```text
len(out_slice_names) == len(directions)
```

Each output slice is paired with its corresponding direction.

### Normalization

When `energy=False`:

```text
normalized_s = s / sqrt(norm_power)
```

When `energy=True`:

```text
normalized_s = s / norm_power
```

The S-matrix objective currently instantiates with `energy=False` from `add_objective()`.

### Recorded S-parameters

Entries are stored as:

```python
s_params[
    (in_slice_name, out_slice_name, out_mode, wl, in_mode, temp)
] = {
    "s_p": ...,
    "s_m": ...,
}
```

This key is the canonical lookup for downstream code that needs modal transmission/reflection information.

---

## Flux objective

`FluxObjective` evaluates power/flux through an output monitor.

For matched simulations, the output is derived from the selected E/H field components through `get_flux()` / `get_flux_3d()`.

### Polarization mapping in the 2D path

For `Ez` polarization:

```text
fx = Hx
fy = Hy
fz = Ez
```

For `Hz` polarization:

```text
fx = Ex
fy = Ey
fz = Hz
```

This mapping is a physical convention of the 2D solver and must not be replaced with a generic `(Ex, Ey, Ez)` assumption.

### `flux_minus_src`

The dispatch recognizes `"flux_minus_src"` and sets `minus_src=True`. This objective exists for comparisons against source flux and should be treated separately from plain output-flux objectives.

### Return behavior

The objective returns one value per matching wavelength/control-state simulation. For tensor execution the result is stacked rather than internally averaged.

---

## Near-to-far flux objective

`FluxNear2FarObjective` computes far-field flux using the Green-function near-to-far transform.

The near-field extraction follows the same 2D polarization mapping described above. The far-field result is then converted to the relevant E/H components and integrated with `get_flux()`.

### Special `total_farfield_region` mode

When `out_slice_name == "total_farfield_region"`, the objective does not return the computed far-field flux. Instead, it stores the full far-field fields in:

```python
total_farfield_region_solutions[(in_slice_name, wl, in_mode, temp)]
```

and returns:

```text
torch.tensor(0.0)
```

This is an important side-effect-only mode and should not be interpreted as an optimization objective by itself.

---

## Response recorder objective

`ResponseRecorderObjective` records the complex output field and phase statistics rather than directly producing a modal transmission objective.

For every matching wavelength/control state it stores:

```python
response[(in_slice_name, out_slice_name, wl, in_mode, temp)] = {
    "fz": fz,
    "mag": mag,
    "phase": phase,
    "phase_std": phase_std,
    "phase_mean": phase_mean,
}
```

### Tensor shapes

The monitored field is reshaped to:

```text
fz: [1, N_monitor_points]
mag: [1, N_monitor_points]
phase: [1, N_monitor_points]
```

The implementation requires the monitor slice to have at least one singleton spatial axis; arbitrary full 2D monitor planes are rejected by the explicit assertion.

The return value is the mean phase across selected evaluations.

---

## Shape-similarity objective

`ShapeSimilarityObjective` evaluates spatial output-pattern similarity.

Inputs include:

```text
shape_type
shape_cfg
intensity
similarity
```

The monitored field is extracted from the polarization component, then reshaped according to whether the monitor is effectively 1D or 2D.

For a 1D monitor:

```text
fz -> [1, N]
```

For a 2D monitor, the tensor may be transposed when the monitor direction is along y so that the spatial orientation matches the similarity utility's expected convention.

The final similarity is produced by `get_shape_similarity()` and the objective returns the mean over selected evaluations.

### Scientifically important point

`shape_type` and `similarity` are not merely plotting options. They define the mathematical comparison being optimized, so they belong in the experiment configuration and should be documented alongside objective weights.

---

## Near-to-far shape-similarity objective

`ShapeSimilarityNear2FarObjective` first computes a Green-function far field and then applies the same shape-similarity machinery to the far-field pattern.

The near-field input is collected from the corresponding source polarization, while the far-field frequency is constructed from:

```text
frequency = 1 / wavelength
```

with the simulation grid step and background permittivity passed to the Green-function transform.

---

## Interface penalty objective

`InterfacePenaltyObjective` penalizes interface fields using `interface_field_penalty()`.

For the 2D `Ez` path, the assembled field tensor is conceptually:

```text
[3, X, Y, Z]
```

with the last singleton dimension inserted for the helper's expected layout. The epsilon map is passed as:

```text
epsilon[..., None]
```

The objective uses configurable interface weights, defaulting to:

```text
(1.0, 1.0, 0.0)
```

This penalty is differentiable with respect to the field and, through the simulation graph, with respect to material/permittivity parameters.

---

## Structure-simplification penalty

`StructureSimplifyObjective` applies `structure_simplify_penalty()` inside every selected design-region mask.

Each region config supplies:

```text
eps_bg   default 1.0
 eps_r   default 12.0
```

and the objective uses:

```text
energy_threshold
soft
 temperature
```

For 2D the helper receives a scalar field component such as `fz` and a 2D epsilon map. For 3D it receives the vector E-field:

```text
[3, X, Y, Z]
```

with a 3D epsilon map and `is_3d=True`.

This objective therefore couples field behavior and material distribution inside the design region rather than evaluating only a port response.

---

# ObjectiveFunc orchestration

## Constructor

`ObjectiveFunc` is the central object joining simulations, port profiles, objective functions, and adjoint bookkeeping.

### Important constructor inputs

`simulations`
: Mapping keyed either by `(wl, pol, temp)` or by `((wl_cen, wl_width, n_wl), pol, temp)` for fdtdx broadband simulations.

`port_profiles`
: See the five-element port-profile contract above.

`port_slices`
: Export-grid monitor slices.

`port_slices_native`
: Native-grid monitor slices.

`port_slices_native_symmetry`
: Native-grid slices adjusted for symmetry/adjoint grouping.

`port_slices_info`
: Physical/geometric metadata for each port/slice. The objective and near-to-far paths read direction and coordinate arrays from this mapping.

`grid_step`
: Simulation grid spacing. Existing code documents this constructor argument in microns for the legacy 2D path; FDTD-X 3D paths may receive grid metadata instead. Follow the actual device/grid convention rather than assuming a universal unit.

`eps_bg`
: Background permittivity used by near-to-far and related utilities.

`control_states`
: Optional mapping from temperature/current control keys to structured state dictionaries.

`grid_metadata`
: Optional rectilinear-grid metadata.

`cell_weights`
: Optional cell-area/volume weighting tensor or array used by modal/flux/shape utilities.

## Native versus export objective grid

The objective keeps both export and native port slices. When native-grid metadata/cell weighting is active, it disables field interpolation to the export grid and uses native slices. The exact conditional is part of the baseline implementation and should be preserved until the grid policy is intentionally redesigned.

## Objective registration

`add_objective()` stores the supplied fusion function and each objective config in `self.obj_cfgs`, then instantiates the matching objective class into `self.Js`.

The objective object stores both:

```text
weight
fn
```

The `weight` is applied at the final breakdown/fusion stage rather than hidden inside each objective implementation.

## Objective evaluation

`obtain_objective(permittivity, custom_source=None)`:

1. Clears/reinitializes simulation/objective state.
2. Collects all target control states from objective configs.
3. Builds a simulation queue from port profiles with `require_sim=True`.
4. Clusters 2D solves by `(wl, polarization, control_state)` so solver factorizations can be reused.
5. Runs simulations and stores fields under the canonical field keys.
6. For fdtdx 3D, expands broadband wavelength groups and stores one field dictionary per scalar wavelength.
7. Evaluates every registered objective function.
8. Produces the `breakdown` mapping.
9. Calls `_fusion_func(breakdown)` to obtain the total objective.

### Breakdown schema

Each objective entry is stored as:

```python
breakdown[name] = {
    "weight": weight,
    "value": value,
}
```

If `requires_grad=False`, a tensor-valued `value` is detached before storage.

The fusion function may return either:

```text
scalar_total
```

or:

```text
(total, extra_breakdown)
```

When the latter is returned, `extra_breakdown` is merged into `self.breakdown`.

## Custom source evaluation

`obtain_objective(..., custom_source=...)` bypasses the normal port-profile simulation queue and expects a dictionary containing at least:

```text
"slice_name"
"source"
"mode"
"wl"
"direction"
```

The device constructs the plane-wave/source profile from these fields before running the simulation.

## Gradient path

`ObjectiveFunc` maintains explicit adjoint/gradient state:

```text
Js       field -> FoM callables
adj_Js   adjoint source-related state
D J      backward derivative representation
dJ       final derivative function/state
```

The exact backend differs by solver. The objective functions themselves are designed to remain differentiable with PyTorch tensors when the simulation/backend supports autograd.

`obtain_gradient(permittivity, eps_shape)` squeezes the backend gradient, reshapes it to `eps_shape`, and returns it.

The convenience call operator supports:

```python
obj(permittivity, mode="forward")
obj(permittivity, eps_shape=..., mode="backward")
```

The backward result is a gradient with the provided `eps_shape`.

## Adjoint grouping for fdtdx

For broadband 3D FDTD, `_build_fdtdx_adjoint_group_info()` groups objective contributions by output monitor and frequency indices.

Each group contains fields such as:

```text
group_id
monitor_name
monitor_slice
port_slice_info
freq_indices
objective_names
obj_types
directions
pol
input_slice_name
mode
```

`freq_indices` refer to indices within the `np.linspace()`-expanded broadband wavelength group, not arbitrary global frequency IDs.

These groups are forwarded into `fdtd3d.solve()` and ultimately used by the FDTD adjoint bridge.

## Thermal/control-state semantics

The objective supports both:

```text
legacy_temp
```

control keys and structured current-control keys.

For a tuple control key, the fallback state is represented as:

```python
{
    "mode": "currents",
    "control_key": control_key,
    "currents": currents_key_to_dict(control_key),
}
```

For a scalar legacy temperature key:

```python
{
    "mode": "legacy_temp",
    "control_key": control_key,
    "temp": control_key,
}
```

Runtime material maps and thermal states are cached per control key during objective evaluation.

## Scientific invariants

Do not change these conventions casually:

- field key schema `(slice, wavelength, mode, control)`;
- 2D `Ez`/`Hz` polarization-to-field mapping;
- broadband fdtdx wavelength expansion;
- `abs_tol=1e-4` wavelength matching;
- port-profile five-tuple structure;
- output-direction sign semantics;
- objective-value versus objective-weight separation;
- native-grid versus export-grid selection;
- cell-weight propagation into modal/flux utilities;
- fdtdx adjoint frequency-index grouping;
- runtime material/thermal state caching;
- gradient reshape to the requested epsilon shape.

## Recommended objective example

A minimal eigenmode configuration that matches the current factory contract is:

```python
obj_cfgs = {
    "_fusion_func": _sum_objectives,
    "through": {
        "type": "eigenmode",
        "weight": 1.0,
        "in_slice_name": "in_slice_1",
        "out_slice_name": "out_slice_1",
        "in_mode": "Ez1",
        "out_modes": ("Ez1",),
        "direction": "x+",
        "wl": (1.55,),
        "temp": (300.0,),
        "energy": True,
        "requires_grad": True,
        "requires_adjoint": True,
    },
}
```

The total objective returned by the default fusion function is then based on:

```text
weight * value
```

for each registered objective (with the separate `smatrix` handling used by the base fusion code documented in `invdes.md`).
