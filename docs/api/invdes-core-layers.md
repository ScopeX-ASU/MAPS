# InvDes Core Layers: Device, Objective, and Parametrization

## Device layer: `core/invdes/models/layers/device_base.py`

`BaseDevice` is the common physical-device abstraction used by the optimization models. It owns the geometry/configuration boundary between a device definition and simulation backends, and it exposes the metadata consumed by `BaseOptimization` and the objective layer.

The module supports multiple electromagnetic/simulation integrations, including the project's FDFD stack and optional FDTD/FDTD-X/Tidy3D-related integrations when those dependencies are installed. Capability flags are used for optional packages rather than making every backend a hard import requirement.

The device contract includes:

- geometry and design-region configuration;
- epsilon/material maps and design-region masks;
- port/slice/source/monitor metadata;
- optical-grid metadata and cell weights;
- thermal/electrical material-property maps when enabled;
- normalization runs and monitor initialization;
- copy/resolution conversion for high-resolution simulation/export devices.

Auxiliary port devices used for normalization and eigenmode solving retain the
box-based material cross section at the requested monitor plane. Intersecting
boxes are extruded along the port propagation axis, so buried layers and
substrates participate in the mode solve while finite structures elsewhere in
the device are excluded.

The `N_Ports` abstraction extends this physical-device contract to ported photonic structures. Higher-level optimization models should consume these device interfaces rather than duplicating geometry or monitor setup logic.

### MMI stacked layers

`core.invdes.models.layers.mmi.MMI` retains the legacy single-layer interface by
default. The preferred multilayer interface separates physical material roles:
`material_ports=(input, output)` controls the access waveguides,
`material_cladding` controls the simulation background, and `material_buried`
fills the 3-D region below the lowest port bottom plane. For the layer stack,
provide `region_material_r1_layers`, `region_material_r2_layers`,
`is_design_region`, `thickness_layers`, `etch_depth_layers`, `gap_layers`, and
`box_size_layers`. The r1/r2 entries become `eps`/`eps_bg` for design layers.
For a fixed layer, r1 is emitted as ordinary geometry and r2 is unused.

A single 2-D design plane may use one-entry r1/r2 material sequences and
`is_design_region` without physical stacking arguments. This preserves the
original `box_size` footprint and `mmi_region` name. A one-layer physical 3-D
stack also retains the `mmi_region` name for compatibility with existing
optimization configurations.

Each layer can contain a fixed lower slab and an etched/top portion;
`is_design_region=True` makes the top portion a design region, while `False`
makes it fixed geometry. Layer footprints are `(x, y)` in 2-D and
`(x, y[, z])` in 3-D; the separate thickness value determines the physical
vertical extent. A zero `thickness_layers` entry retains its layer index and
preceding gap but emits neither geometry nor a design region; its etch depth
must therefore also be zero. A scalar or one-entry layer argument is broadcast
across the stack. In 3-D, `port_height=(input_height, output_height)` controls
the two port heights independently of layer thicknesses; the output height also
defines the vertical slab bounds of output sine bends. Ports remain centered at
`z=0`.
`layer_stack_z_offset` translates all material layers relative to those ports,
while `gap_layers` remains a nonnegative physical separation between layers.
To bottom-align the first layer with a port, use
`layer_stack_z_offset=(first_layer_thickness - port_height) / 2`.

The older `material_r1`, `material_r2`, `material_bg`, `material_layers`, and
scalar `region_material_r1`/`region_material_r2` arguments remain available as
compatibility fallbacks for existing devices.

### Grating-coupler vertical materials

`core.invdes.models.layers.grating_coupler.GratingCoupler` uses
`material_cladding` for the simulation background, vertical input port, and
etched portions of each grating layer. `material_buried` optionally fills the
region below the output-port and layer-stack bottom plane. The existing
high-index `substrate` geometry remains separate: it is placed below
`substrate_gap` and overrides the buried material where the two overlap. This
represents cladding above the core, a buried dielectric below it, and an
optional substrate farther below.

The buried partition follows y in 2-D and z in 3-D and does not enlarge the
automatically inferred simulation cell. `material_bg` remains the compatibility
fallback for `material_cladding`; omitting `material_buried` preserves the
historical uniform-background behavior.

## Objective layer: `core/invdes/models/layers/objective.py`

The objective layer converts simulated fields into scalar/tensor figures of merit while retaining differentiability through the electromagnetic post-processing operations.

### `SMatrixObjective`

Evaluates normalized S-parameters from eigenmode coefficients. It supports multiple input/output slices, wavelength and temperature filtering, multiple output modes, direction selection, and both 2-D and 3-D field extraction paths.

For eigenmode objectives it also records normalized forward/backward coefficients in `s_params`, keyed by input/output slice, mode, wavelength, and temperature. This side-channel is used by downstream logging and analysis without changing the primary objective tensor.

### `FluxObjective`

Computes normalized power/flux through an output slice. The `minus_src`/flux-minus-source variants are used where the objective represents reflected or subtracted source power rather than ordinary transmitted flux.

### `FluxNear2FarObjective`

Transforms near-field monitor data into a far-field quantity using the Green-function near-to-far implementation. It supports a special total-farfield-region mode whose computed far fields are cached for reuse.

### Objective aggregation

`BaseOptimization` builds these objective objects from the configured `obj_cfgs`, evaluates them for each relevant wavelength/mode/temperature condition, and then fuses their results using the configured objective-fusion function. The default fusion is a weighted sum, excluding `smatrix` entries from the scalar loss aggregation because those entries are commonly recorded as diagnostic S-parameter data.

## Parametrization layer: `core/invdes/models/layers/parametrization`

The parametrization package maps a compact set of trainable design variables to physical material/geometry fields on the simulation grid. `parametrization_builder` selects the parametrization family from per-design-region configuration and returns an `nn.ModuleDict` consumed by `BaseOptimization`.

The package contains several families, including box/edge-box constructions, geometric and polygonal representations, extrusion, level-set representations, and morphological operators. The intended contract is that a parametrization owns its trainable parameters and exposes a differentiable field representation suitable for simulation.

### Resolution handling

`base_parametrization.py` provides the shared resolution-conversion and symmetry utilities. `cvt_res`/`_convert_resolution` resample a parameter field between design and simulation grids. Symmetry utilities include mirror, transpose, and 90/180-degree rotational constructions.

The mirror implementation explicitly compensates shared-parameter gradient accumulation: mirrored copies use a gradient decay factor so that duplicated geometry does not double the effective gradient. This is an optimization constraint, not merely a visualization transform.

### Level-set representation

The level-set family uses a lower-dimensional design representation to construct a signed-distance/implicit geometry field. The base optimization model exposes `reset_levelset_sdf()` so a parametrization can reinitialize its level-set representation when required during an optimization workflow.

## Data-flow contract

The complete differentiable path is:

`design parameters -> parametrization -> epsilon/material maps -> device simulation -> fields -> objective -> scalar objective -> closure/backward -> parameter gradients`

`BaseOptimization` owns the middle of this pipeline; `InvDesign` owns the outer optimization loop; device classes own physical geometry/simulation setup; objective classes own field-to-FoM transformations; parametrizations own the trainable design representation.
