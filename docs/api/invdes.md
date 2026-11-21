# Inverse-Design Workflow API

This page documents the MAPS inverse-design orchestration implemented by `core/invdes/invdesign.py` in the `v0.0.2_fdtdx` baseline. It is descriptive documentation only and does not modify optimization behavior.

## Role

`InvDesign` is the training-loop-style wrapper around a device optimization model. The optimization model is responsible for physics/device-specific forward evaluation; `InvDesign` manages optimizer/scheduler construction, backward propagation, gradient normalization, callbacks, logging, plotting, checkpoints, and artifact export.

Conceptually:

```text
design parameters
      |
      v
devOptimization.forward(sharpness)
      |
      v
results["obj"] + results["breakdown"]
      |
      v
backward on -results["obj"]
      |
      v
optional gradient normalization
      |
      v
optimizer.step(closure)
      |
      v
callbacks / schedulers / plotting / export
```

The sign convention is important: the closure returns `-results["obj"]` and, when backward is enabled, calls `(-results["obj"]).backward()`. Therefore the optimizer performs gradient descent on the negative objective, corresponding to maximization of `results["obj"]`.

## Required optimization-model contract

`InvDesign(devOptimization=...)` requires a non-`None` optimization model.

The wrapper assumes the object exposes standard PyTorch parameter iteration:

```python
devOptimization.parameters()
```

and a forward interface compatible with:

```python
results = devOptimization.forward(sharpness=sharpness)
```

The returned `results` mapping must contain at least:

```text
"obj"
"breakdown"
```

`results["obj"]`
: Scalar PyTorch objective tensor used for differentiation.

`results["breakdown"]`
: Mapping used for logging/plotting. Each logged entry is expected to contain a `"value"` field. Values may be scalar tensors, multi-element tensors, or scalar-like Python values.

Plotting/checkpoint features assume additional model attributes/methods such as `plot()`, `sim_cfg.plot_root`, `dump_gds_files()`, and high-resolution device/permittivity state when those features are enabled.

## Default configuration schema

`InvDesign.default_cfgs` defines the baseline configuration.

### `optimizer`

```text
name: "Adam"
lr: 1e-2
weight_decay: 0
```

Additional optimizer-specific fields may be consumed by `core.invdes.builder.make_optimizer()`.

### `lr_scheduler`

```text
name: "cosine"
lr_min: 2e-4
```

### `sharp_scheduler`

```text
mode: "cosine"
name: "sharpness"
init_sharp: 1
final_sharp: 256
```

Sharpness is read before each optimization step and passed to the device optimization model's forward call.

### `run`

```text
start_epoch: 0
n_epochs: 100
norm_grad: None
backward: True
```

`norm_grad`
: Optional gradient-normalization mode. Supported by `normalize_grad()`:

- `"abs_mean"`
- `"rms"`
- `"p95"`

`backward`
: If false, the closure skips `backward()`. This supports forward-only/normalization-style runs, but optimizer behavior should be considered carefully when gradients are absent.

### `plot_cfgs`

```text
plot: False
interval: 5
plot_name: None
objs: []
field_keys: []
in_slice_names: []
exclude_slice_names: []
thermal_map_names: []
field_component: None
eps_grad: False
param_grad: False
```

When plotting is enabled, the configuration validates that `plot_name`, `objs`, `field_keys`, and `in_slice_names` are populated.

A `field_key` is later indexed at position 2 and its first two characters are interpreted as a polarization/component identifier, e.g. a mode-like string beginning with `Ez`.

### `debug_cfgs`

```text
log_cuda_memory: False
empty_cache: False
clear_step_results: True
synchronize_cuda: False
```

CUDA memory statistics are reported in MiB using PyTorch allocated/reserved and peak counters.

### `checkpoint_cfgs`

```text
save_model: False
ckpt_name: None
interval: None
save_latest: True
latest_name: None
save_on_interrupt: True
dump_gds: False
gds_name: None
dump_eps: False
eps_name: None
upsample_eps_to_1nm: False
dump_tidy3d: False
tidy3d_name: None
load_ckpt: None
resume: False
```

If model saving is enabled, `ckpt_name` is required. If GDS dumping is enabled, `gds_name` is required.

### `run_management`

```text
enabled: False
root: ./runs
experiment: optimization
run_id: None
tag: None
```

When enabled, this creates the versioned run bundle documented in
[Optimization Run Management](optimization-run-management.md) and routes
runtime outputs into its typed subdirectories. When disabled, legacy
`sim_cfg.plot_root` behavior is unchanged. `checkpoint_cfgs.resume` is the
single resume switch; when managed runs are enabled, it also reopens the
existing run directory inferred from `load_ckpt`.

### callbacks

Default callbacks are:

```text
after_step_callbacks = [clamp_parameters]
before_step_callbacks = []
```

The default `clamp_parameters()` clamps every parameter of `invdes.devOptimization` in-place to `[-0.5, 0.5]` under `torch.no_grad()`.

## Gradient normalization

`normalize_grad(g, mode, eps=1e-12)` operates elementwise on a PyTorch gradient tensor while computing one scalar normalization scale.

### `abs_mean`

```text
scale = mean(abs(g))
g_norm = g / max(scale, eps)
```

### `rms`

```text
scale = sqrt(mean(g^2))
g_norm = g / max(scale, eps)
```

### `p95`

```text
scale = quantile(abs(g).flatten(), 0.95)
g_norm = clamp(g / max(scale, eps), -1, 1)
```

The `p95` mode both rescales and clips the normalized gradient.

The operation preserves the input tensor shape. It is applied independently to each trainable parameter tensor that has a gradient.

## Step lifecycle

### `before_step()`

1. Resets the internal log string.
2. Reads the current sharpness from the sharpness scheduler.
3. Creates:

```python
feed_dict = {"sharpness": sharpness}
```

4. Passes the dictionary through all `before_step_callbacks` in sequence.

Each before-step callback receives:

```python
callback(invdesign, feed_dict)
```

and must return the feed dictionary used by the next callback/step.

### `run_step(feed_dict)`

The method expects:

```text
feed_dict["sharpness"]
```

It sets the closure sharpness and calls:

```python
optimizer.step(closure)
```

The closure:

1. zeros gradients;
2. calls `devOptimization.forward(sharpness=...)`;
3. optionally differentiates `-results["obj"]`;
4. optionally normalizes each parameter gradient;
5. stores the complete results dictionary;
6. returns `-results["obj"]` to the optimizer.

After the optimizer step, each after-step callback is called as:

```python
callback(invdesign)
```

The method returns the model's results mapping.

### `after_step(output_dict)`

The learning-rate and sharpness schedulers are advanced once per step.

Depending on configuration, this phase also performs plotting and memory/result cleanup. With `clear_step_results=True`, references to closure results are cleared, `_eps_map.grad` is cleared when available, Python garbage collection is run, and CUDA cache clearing is optional.

### `optimize()`

```python
optimize(verbose=True, dump_intermediate_result=False)
```

Runs epochs from:

```text
start_epoch
```

to:

```text
start_epoch + n_epochs - 1
```

Each iteration executes:

```text
before_step -> run_step -> after_step
```

and optionally `dump_intermediate_artifacts()`.

After the loop, `after_epoch(results)` handles final checkpoint/artifact operations.

## Logging and gradient norm

After each step, the wrapper logs entries from `results["breakdown"]` and computes a global gradient norm by flattening and concatenating all available trainable parameter gradients:

```text
g = concat([param.grad.reshape(-1), ...])
grad_norm = ||g||_2
```

If no parameter has a gradient, the reported norm is `0.0`.

## Checkpoint loading and saving

### Loading

`load_model(path)` accepts both full resumable checkpoints and legacy mappings
containing:

```text
"state_dict"
```

which is loaded into `devOptimization`. A
`maps.invdesign.checkpoint/v1` checkpoint additionally restores optimizer and
scheduler states, the next unfinished step, the latest FoM, Python/NumPy/Torch
random states, and variation-sampling state when present. A legacy checkpoint
restores model parameters only.

### Saving

`save_model(fom, path)` delegates checkpoint ranking to `BestKModelSaver` using
the final configured epoch index, but stores the full resumable state. The
wrapper is configured with `k=1`, `descend=False`, and metric name `"err"`.

`save_checkpoint(path, fom)` writes the same full state atomically to an exact
path. When model saving, `save_latest`, and a positive `interval` are enabled,
the loop overwrites `<ckpt_name>_latest.pt` only after a complete step. Resume
by setting `checkpoint_cfgs.resume=True` and `load_ckpt` to that path. The
recorded `next_step` prevents completed steps from being repeated.

Set `dump_tidy3d=True` to have `after_epoch()` automatically save a
self-contained Tidy3D device-visualization file. It contains fixed and designed
geometry plus native visualization objects for physical port sources and modal
monitors; auxiliary design/full-domain phasor monitors and solved fields are not
included. Like `gds_name`, `tidy3d_name` is a stem; MAPS always writes it with
the fixed `.h5` extension. Existing `.h5` or `.hdf5` suffixes are normalized to
`.h5` rather than duplicated. When no name is supplied, MAPS derives one from
`ckpt_name` or `gds_name`. Before exporting, MAPS rebuilds the high-resolution
epsilon map from the final updated parameters using the last sharpness applied
by the optimizer closure. This avoids exporting the cached map from before the
final parameter update.

## High-resolution permittivity export

The epsilon-image export path expects:

```text
devOptimization.hr_eps_map
devOptimization.hr_device
devOptimization.hr_device.design_region_masks
```

Currently exactly one design region is supported by `_get_eps_dump_map()`.

If `upsample_eps_to_1nm=False`, the high-resolution epsilon map is exported at its existing resolution.

If `upsample_eps_to_1nm=True`, the code targets a resolution value of `1000`, copies the high-resolution device at that resolution, bilinearly interpolates the design-region epsilon tensor, and inserts it into the copied device's background epsilon map.

The interpolation input is formed as:

```text
[H, W] -> [1, 1, H, W]
```

and resized with PyTorch bilinear interpolation using `align_corners=False`.

The final PNG is binary. The threshold is the midpoint of the minimum and maximum epsilon values within the design region:

```text
threshold = 0.5 * (eps_min + eps_max)
```

Pixels with `eps >= threshold` become 255; others become 0.

## Representative usage

The baseline file contains a bending-device example following this pattern:

```python
sim_cfg = DefaultSimulationConfig()
sim_cfg.update({
    "solver": "ceviche_torch",
    "resolution": 100,
    "PML": [0.5, 0.5],
    "numerical_solver": "solve_direct",
    "use_autodiff": False,
})

device = Bending(sim_cfg=sim_cfg, ...)
hr_device = device.copy(resolution=310)

opt = BendingOptimization(
    device=device,
    hr_device=hr_device,
    sim_cfg=sim_cfg,
    operation_device=operation_device,
).to(operation_device)

invdesign = InvDesign(devOptimization=opt)
invdesign.optimize()
```

The omitted geometry arguments are device-specific. The important orchestration contract is that `InvDesign` receives a configured optimization model rather than constructing the physical device itself.

## Safety invariants for future changes

Preserve or explicitly validate:

- maximization sign convention through `-results["obj"]`;
- `results` dictionary schema;
- sharpness scheduling before forward evaluation;
- callback signatures and ordering;
- optimizer closure compatibility, especially for LBFGS-style methods;
- per-parameter gradient normalization semantics;
- high-resolution design-region shape mapping;
- checkpoint `state_dict` schema;
- cleanup behavior when differentiable simulation graphs are large.
