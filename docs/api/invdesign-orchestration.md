# InvDesign Orchestration Contracts

This page documents the runtime contract of `core/invdes/invdesign.py` while source-level docstrings are being migrated incrementally.

## Role

`InvDesign` is the optimization-loop wrapper for a physics-based inverse-design model (`devOptimization`). It connects:

```text
configuration
  -> optimizer
  -> LR scheduler
  -> physical continuation scheduler (for example sharpness)
  -> model forward
  -> objective
  -> backward on -objective
  -> optimizer step
  -> callbacks / plots / checkpoints
```

It is intentionally analogous to a training loop, but the "model" is a differentiable electromagnetic inverse-design model rather than a conventional neural network.

## Construction contract

Typical construction:

```python
invdesign = InvDesign(devOptimization=opt)
```

The required keyword is:

```text
`devOptimization`
```

It must be a configured inverse-design model exposing at least:

```text
parameters()
forward(sharpness=...)
sim_cfg.plot_root
```

and, for some checkpoint/export paths:

```text
gradients / _eps_map
hr_eps_map
hr_device
sim_cfg
plot(...)
dump_gds_files(...)
load_state_dict(...)
```

The wrapper constructs its optimizer and schedulers from the InvDes configuration tree.

## Default configuration

The current default configuration contains four major groups:

```text
optimizer
lr_scheduler
sharp_scheduler
run
plot_cfgs
debug_cfgs
checkpoint_cfgs
```

The current baseline defaults are:

```text
optimizer:
    name = Adam
    lr = 1e-2
    weight_decay = 0

lr_scheduler:
    name = cosine
    lr_min = 2e-4

sharp_scheduler:
    name = sharpness
    mode = cosine
    init_sharp = 1
    final_sharp = 256

run:
    start_epoch = 0
    n_epochs = 100
    norm_grad = None
    backward = True
```

These values are defaults, not universal recommendations for every device or solver.

## `load_cfgs(**cfgs)`

`load_cfgs` merges the supplied configuration values into the wrapper's configuration state and performs limited configuration validation.

Important validation rules currently implemented:

- plotting requires `plot_name`, at least one objective name, field key, and input-slice name;
- `exclude_slice_names` is expanded to one empty list per objective when omitted;
- `thermal_map_names` is expanded to one `None` per objective when omitted;
- checkpoint model export requires `ckpt_name`;
- GDS export requires `gds_name`.

The function expects a `pyutils.config.Config`-like object/structure, not a plain Python `dict` in every path.

## One optimization step

The step lifecycle is:

```python
feed_dict = invdesign.before_step()
results = invdesign.run_step(feed_dict)
invdesign.after_step(results)
```

### `before_step()`

Returns a dictionary containing the current continuation parameter:

```python
{"sharpness": sharpness}
```

The sharpness value is obtained from the configured sharpness scheduler and can be modified by `before_step_callbacks`.

### `run_step(feed_dict)`

The core optimization operation is conceptually:

```python
optimizer.zero_grad(set_to_none=False)
results = devOptimization.forward(sharpness=sharpness)
(-results["obj"]).backward()
optimizer.step(...)
```

The exact gradient path is owned by the underlying model and solver.

The wrapper therefore assumes the forward result contains at least:

```text
results["obj"]
results["breakdown"]
```

where `results["obj"]` is a differentiable scalar objective used for optimization, and `results["breakdown"]` is a mapping of named objective terms/metadata used for logging and plotting.

### Objective sign convention

The wrapper returns and optimizes:

```text
loss = -results["obj"]
```

because PyTorch optimizers are generally formulated as minimizers while MAPS inverse-design objectives are normally maximized.

This sign is part of the optimization contract. An optimizer/closure refactor must preserve it unless the objective convention is explicitly changed everywhere.

## Optional backward path

The configuration field:

```text
run.backward
```

controls whether the closure calls:

```python
(-results["obj"]).backward()
```

A forward-only run can therefore use the wrapper without constructing the backward graph for the objective.

This does not imply that the underlying model itself is non-differentiable; it means this wrapper iteration is configured not to execute the backward operation.

## Gradient normalization

When:

```text
run.norm_grad != None
```

the wrapper normalizes each available parameter gradient using `normalize_grad`.

Supported modes are:

```text
abs_mean
rms
p95
```

For `p95`, the scale is the 95th percentile of the absolute flattened gradient and the normalized gradient is clipped to `[-1, 1]`.

The normalization operation changes the optimizer input gradient, not the physical objective value.

## Parameter clamp callback

The default `after_step_callbacks` contains:

```python
clamp_parameters
```

with default bounds:

```text
[-0.5, 0.5]
```

The callback operates on the parameters returned by:

```python
devOptimization.parameters()
```

under `torch.no_grad()`.

This is a parameter-domain constraint applied after the optimizer step. It is distinct from material-map clipping or binary projection inside the design parameterization.

## `after_step(output_dict)`

`after_step` performs post-update work:

```text
LR scheduler step
sharpness scheduler step
optional plotting
optional result/gradient cleanup
optional CUDA cache clearing
```

The plotting path expects entries from:

```text
output_dict["breakdown"][objective_name]["value"]
```

and uses configured field keys and input-slice names to select the field representation for visualization.

## `after_epoch(output_dict)`

The epoch-end path handles optional checkpoint/GDS/epsilon artifacts.

Model checkpointing passes the objective scalar from:

```text
output_dict["obj"]
```

to the saver.

High-resolution epsilon export uses the model's `hr_eps_map` and `hr_device` state when available.

## `dump_intermediate_artifacts()`

This method emits intermediate high-resolution epsilon and/or GDS artifacts according to checkpoint configuration and interval.

The interval is taken from `checkpoint_cfgs.interval`, falling back to `plot_cfgs.interval`.

It skips the dump unless the current global step is on the configured interval or is the final optimization step.

## High-resolution epsilon export

`_get_eps_dump_map` starts from:

```text
`devOptimization.hr_eps_map`
```

and expects exactly one high-resolution design region in the current implementation.

When `upsample_eps_to_1nm=True`, the current implementation targets a spatial resolution of:

```text
1000 cells / unit length
```

and bilinearly resamples the design-region epsilon map into the target mask.

This export transform is an artifact-generation operation. It should not be confused with the optimization model's native parameterization resolution.

## `optimize()`

The high-level loop is:

```text
for i in range(start_epoch, start_epoch + n_epochs):
    global_step = i
    feed_dict = before_step()
    results = run_step(feed_dict)
    after_step(results)
    optional intermediate artifacts
    log objective breakdown + gradient norm

after_epoch(results)
```

The log includes:

```text
step
sharpness
objective breakdown values
grad_norm
```

The gradient norm is computed over all available gradients of learnable parameters and is a monitoring quantity, not itself the optimization objective.

## Results schema used by orchestration

The wrapper does not define a universal tensor shape for every result because the underlying objective model controls the detailed breakdown.

Stable orchestration expectations are:

```python
results["obj"]
```

and:

```python
results["breakdown"]
```

For a breakdown entry, typical usage expects:

```python
results["breakdown"][name]["value"]
```

The value can be a scalar tensor or a tensor with more than one element; the logging code handles both cases.

Do not document every objective value as a scalar probability or normalized transmission unless the corresponding objective implementation establishes that convention.

## CUDA memory diagnostics

When enabled through:

```text
debug_cfgs.log_cuda_memory = True
```

`_cuda_memory_stats()` reports values in MiB:

```text
allocated_mb
reserved_mb
max_allocated_mb
max_reserved_mb
```

If `synchronize_cuda=True`, CUDA synchronization is performed before collecting the statistics so asynchronous work is reflected more accurately.

These diagnostics are observability helpers and do not change optimization mathematics except insofar as synchronization can affect runtime behavior/timing.

## Example

The repository's explicit Bending workflow constructs the wrapper as:

```python
opt = BendingOptimization(
    device=device,
    hr_device=hr_device,
    sim_cfg=sim_cfg,
    operation_device=operation_device,
).to(operation_device)

invdesign = InvDesign(devOptimization=opt)
invdesign.optimize()
```

A configuration-driven workflow should normally provide the wrapper settings through the experiment configuration rather than hard-code them in the script.

See:

- [First inverse-design workflow](../getting-started/first-inverse-design.md)
- [First FDFD workflow](../getting-started/first-fdfd-workflow.md)
- [InvDesign API contract](invdes.md)
- [InvDesign builder](invdes-builder.md)

## Refactoring contract

Before future refactoring of `InvDesign`, preserve these boundaries:

```text
InvDesign wrapper
    owns optimization orchestration

BendingOptimization / other devOptimization
    owns physical design/model forward behavior

builder.py
    owns configuration -> optimizer/scheduler construction

objective/FoM modules
    own scientific objective definitions

solver modules
    own electromagnetic numerical behavior
```

A refactor should not move scientific objective semantics into the generic loop merely to shorten the class.
