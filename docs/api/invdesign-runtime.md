# InvDes Runtime and Optimization Architecture

This page documents the runtime contract of the orchestration layer without duplicating executable implementation details.

## `core/invdes/invdesign.py`

`InvDesign` is the optimization-loop wrapper. It does not construct a physical device or define the device parametrization itself; it receives an already-built `devOptimization` module and wires it to an optimizer, learning-rate scheduler, and sharpness/continuation scheduler.

### Configuration contract

The class-level `default_cfgs` groups runtime configuration into:

- `devOptimization`: the differentiable optimization model. Construction is normally handled by `core.invdes.builder`.
- `optimizer`: optimizer dispatch parameters such as `name`, `lr`, and `weight_decay`.
- `lr_scheduler`: learning-rate continuation, including `name` and `lr_min` for cosine scheduling.
- `sharp_scheduler`: physical continuation/sharpness control, with `mode`, `name`, `init_sharp`, and `final_sharp`.
- `run`: epoch range and loop behavior, including `norm_grad` and whether the closure performs backward propagation.
- `plot_cfgs`: objective/field plotting and optional gradient/thermal-map visualization controls.
- `debug_cfgs`: CUDA-memory logging, cache clearing, CUDA synchronization, and step-result lifetime.
- `checkpoint_cfgs`: model/GDS/epsilon-map persistence and checkpoint resumption.
- `run_management`: optional versioned run directory, provenance, metrics, and
  artifact routing. See [Optimization Run Management](optimization-run-management.md).
- `before_step_callbacks` and `after_step_callbacks`: user-extensible hooks around the optimizer step.

The implementation merges user configuration into the default `Config` object. The supplied `devOptimization` is mandatory.

### Gradient closure

The optimizer is driven by an internal closure. A closure invocation:

1. clears parameter gradients;
2. calls `devOptimization.forward(sharpness=...)`;
3. optionally backpropagates `-results["obj"]`;
4. optionally normalizes parameter gradients;
5. stores the complete forward result for logging/plotting;
6. returns `-results["obj"]` as the optimizer loss.

The sign is intentional: InvDes treats the reported objective as a quantity to maximize, while PyTorch optimizers minimize the closure loss.

### Step lifecycle

The public loop is organized into `before_step()`, `run_step()`, and `after_step()` followed by `after_epoch()`.

`before_step()` obtains the current sharpness from the sharpness scheduler and passes it through pre-step callbacks.

`run_step()` installs that sharpness into the closure, executes `optimizer.step(closure)`, runs post-step callbacks, and returns the closure's saved results.

`after_step()` advances both schedulers and optionally renders configured diagnostics. It also clears transient results and gradients and can release CUDA cache memory according to `debug_cfgs`.

`after_epoch()` handles persistent artifacts: model checkpoints,
high-resolution epsilon-map images, GDS, and optional self-contained Tidy3D
geometry. For Tidy3D export it rebuilds the design map from the final updated
parameters with the last closure sharpness, because the cached forward map
predates the optimizer update. Failures in these optional artifact paths are
logged rather than propagated as optimization-step failures.

### Resumable checkpoints

When `checkpoint_cfgs.save_model` is enabled, MAPS checkpoints use the
versioned `maps.invdesign.checkpoint/v1` schema. A full checkpoint includes the
device-optimization model, optimizer, learning-rate and sharpness schedulers,
completed/next step, latest FoM, Python/NumPy/PyTorch random-generator states,
the relevant resolved configuration sections, and variation-sampler state when
the model exposes it. Legacy files containing only `state_dict` remain
loadable, but cannot restore optimizer momentum or scheduler progress.

Set `checkpoint_cfgs.interval` to a positive number to atomically update
`<ckpt_name>_latest.pt` after that many completed steps. This deterministic
file is the preferred recovery point after interruption. `save_on_interrupt`
also writes `<ckpt_name>_interrupted.pt` when an exception occurs between
steps; an exception during a partially executed optimizer step never
overwrites the last known-consistent checkpoint.

With `checkpoint_cfgs.resume=True` and `load_ckpt` pointing to a full
checkpoint, `optimize()` begins at its recorded `next_step` and retains the
configured original end step `run.start_epoch + run.n_epochs`. Thus, resuming
a 100-step run interrupted after step 41 continues at step 42 and stops after
step 99. Increase `run.n_epochs` explicitly when intentionally extending a
completed run. Checkpoints use PyTorch serialization and must be loaded only
from trusted sources.

### Optimization loop

`optimize(verbose=True, dump_intermediate_result=False)` iterates from `run.start_epoch` for `run.n_epochs` steps. Each iteration executes the lifecycle above, optionally dumps intermediate artifacts, and logs objective breakdown values together with an aggregate gradient norm.

### Gradient normalization

`normalize_grad` supports three modes:

- `abs_mean`: divide by the mean absolute gradient magnitude;
- `rms`: divide by root-mean-square gradient magnitude;
- `p95`: divide by the 95th percentile of absolute gradients and clamp the normalized result to `[-1, 1]`.

A minimum scale is enforced with `eps` to avoid division by zero.

### Parameter clamping

The default `after_step_callbacks` includes `clamp_parameters`, which clamps all parameters exposed by `devOptimization` to `[-0.5, 0.5]` unless a caller replaces the callback list.

### High-resolution epsilon export

`_get_eps_dump_map` obtains the high-resolution epsilon map from `devOptimization.hr_eps_map`. The current implementation supports one design region for the optional 1-nm export path. When enabled, the design-region epsilon tensor is bilinearly upsampled to the target resolution and blended back into a copied device epsilon map.

`_dump_eps_image` converts the selected epsilon map to a binary grayscale image by thresholding at the midpoint of the design-region minimum and maximum permittivity.

## `core/invdes/models/base_optimization.py`

`BaseOptimization` is the device-level differentiable optimization model. It bridges a physical device representation, design-region parametrizations, the simulation backend, and objective evaluation.

### Configuration objects

`DefaultSimulationConfig` supplies baseline simulation settings such as solver selection, binary projection thresholds, PML and border widths, resolution, center wavelength, wavelength sweep width/count, and plotting root.

`DefaultOptimizationConfig` supplies three principal groups:

- `design_region_param_cfgs`: parametrization configuration per design region;
- `sim_cfg`: simulation configuration forwarded to the device/simulation stack;
- `obj_cfgs`: objective definitions plus an optional objective-fusion function.

The default objective fusion function is `_sum_objectives`, which ignores entries whose name contains `smatrix` and otherwise sums `weight * value`. Custom fusion functions are expected to consume the objective breakdown and return `(total_obj, extra_breakdown)`.

### Construction sequence

`BaseOptimization.__init__` performs the following high-level sequence:

1. stores the low-resolution device, high-resolution device, and operation device;
2. builds and exposes merged optimization configuration;
3. materializes low- and high-resolution epsilon maps;
4. records design-region masks and coverage weights;
5. builds the design-region parametrization modules;
6. initializes device monitors;
7. runs the device normalization simulation and stores port source profiles;
8. pre-builds the objective pipeline from device monitor/source metadata and objective configuration.

The resulting `design_region_param_dict` is an `nn.ModuleDict` whose entries are the trainable parametrization modules for each configured design region.

### Design-region property handling

The base class contains the common mechanics for applying parametrized material-property maps into a background map. The implementation supports:

- direct mask replacement;
- weighted blending through mask coverage weights;
- axis-factorized coverage weights;
- shape fitting and interpolation between parametrization and simulation grids;
- separate thermal design-region coordinates and maps.

This machinery is used for optical and thermal properties and is deliberately kept below device-specific optimization subclasses.

### Objective and gradient bridge

`build_parameters()` creates design-region parametrizations and instantiates the `SimulatedFoM` wrapper bound to `cal_obj_grad`.

`BaseOptimization` therefore defines the common differentiable bridge rather than a single device objective. Concrete optimization subclasses configure device-specific objective definitions and implement the simulation/adjoint behavior consumed by that layer.

### Thermal runtime state

The base class snapshots thermal control states produced by the objective pipeline and retains gradients inside nested state structures when the tensors require gradients. Plotting helpers resolve the correct optical/thermal map from these runtime control states.

### Reset operations

Two explicit reset operations are available:

- `reset_levelset_sdf()`: asks each parametrization that supports it to reset its level-set representation to a signed-distance-function form;
- `reset_parameters()`: resets the parameters of all design-region parametrizations.

These are state-reset mechanisms, not optimization steps.

## Relationship to `builder.py`

The builder constructs optimization models, while `BaseOptimization` provides the shared model contract and `InvDesign` executes the optimization loop. The resulting dependency direction is:

`device -> optimization model -> InvDesign loop`

with optimizer/scheduler factories attached at the `InvDesign` layer and parametrization/objective/simulation machinery attached at the optimization-model layer.
