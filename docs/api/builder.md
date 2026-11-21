# Top-Level Builder and Configuration API

This page documents the top-level factory layer in `core/builder.py` for the `v0.0.2_fdtdx` baseline. The file contains two related but distinct construction paths:

1. machine-learning/training factories (`make_dataloader`, `make_model`, `make_optimizer`, `make_scheduler`, `make_criterion`); and
2. inverse-design assembly helpers used by the broader configuration-driven workflow.

The implementation is strongly configuration-driven. This page records the currently supported dispatch behavior without changing or normalizing it.

## Configuration-driven construction principle

The builder uses the global `pyutils.config.configs` object for most training/model settings. In practice, the configuration provides names plus parameter groups that are consumed by the selected factory.

The high-level relationship is:

```text
configuration
   |
   +--> dataset config ------> make_dataloader()
   |
   +--> model config --------> make_model()
   |
   +--> optimizer config ----> make_optimizer()
   |
   +--> scheduler config ----> make_scheduler()
   |
   +--> criterion config ----> make_criterion()
```

This means a config key is an API dependency even when the selected Python class has the real constructor signature. When adding a new model/optimizer/scheduler/criterion, update the corresponding dispatch logic and document the required configuration keys.

## `make_dataloader()`

```python
make_dataloader(
    name=None,
    splits=["train", "valid", "test"],
    train_noise_cfg=None,
    out_frames=None,
)
```

The `name` defaults to `configs.dataset.name` and is lower-cased.

### Supported dataset dispatch

`name == "fdfd"`
: Builds `FDFDDataset` independently for the requested train/valid/test splits. Required dataset configuration includes the device type, root, split ratios, and processed-data directory.

`name == "ilt"`
: Builds `ILTDataset`. The builder consumes root/data directory, patch size, context, stride, split ratios, border size, transform mode, processed directory, and target resolution.

Other names
: Delegates to the external `pyutils.datasets.get_dataset()` factory with image height/width, root, and transform configuration.

### DataLoader contract

Three loaders are returned:

```python
train_loader, validation_loader, test_loader
```

The current implementation uses:

```text
batch_size        = configs.run.batch_size
num_workers       = configs.dataset.num_workers
pin_memory       = True
prefetch_factor   = 8
persistent_workers= True
shuffle           = bool(configs.dataset.shuffle)
drop_last         = True
```

The custom spatial-resolution collate functions are defined in this module but are currently commented out in the `DataLoader` construction. Do not assume they are active merely because the functions exist.

## Dataset tensor/schema conventions

The custom FDFD collate path documents the expected training sample keys:

```text
Ez
source
eps
padding_mask
mseWeight
src_mask
epsWeight
device_type
eps_bg
grid_step
```

The target mapping contains:

```text
Ez
std
mean
stdDacayRate
```

The spatial collate transforms each sample using its `scaling_factor`, then pads the batch to a common spatial shape. Fields/source/weights use zero padding, while `eps` uses the per-sample `eps_bg` for padded material values.

The collate path also forces even padded width for Fourier/inverse-Fourier compatibility.

These custom collate functions are research-pipeline utilities and should not be treated as the default runtime behavior unless explicitly wired into the DataLoader.

## `make_model()`

```python
make_model(device, random_state=None, **kwargs)
```

Dispatch is driven by `configs.model.name`.

The baseline includes dedicated constructor paths for model families including:

```text
repara_phc_1x1
repara_phc_1x1 + eff_vg
simplecnn
neurolight
fno3d
ffno2d
```

Unsupported names raise `NotImplementedError`.

### Important configuration principle

The builder does not use one uniform constructor signature. Each model family receives its own configuration subset. Examples include:

```text
device_cfg / sim_cfg / adjoint_mode
coupling_region_cfg / superlattice_cfg
in_channels / out_channels / kernel_list / mode_list
```

Therefore an AI agent should never infer that adding a new model name is equivalent to adding a single generic `**configs.model` call. The builder is the source of truth for which fields are passed to each model family in this release baseline.

## `make_optimizer()`

```python
make_optimizer(
    params,
    name=None,
    configs=None,
    no_decay_params=[],
)
```

The function separates parameters into two groups:

```text
weight_decay = configs.weight_decay
weight_decay = 0 for no_decay_params
```

The baseline supports:

```text
sgd
adam
adamw
dadaptadam
sam_sgd
sam_adam
lbfgs
```

Unsupported names raise `NotImplementedError`.

### Parameter-group contract

`params` must be an iterable of PyTorch trainable parameters. `no_decay_params` identifies parameters that should not receive weight decay.

For `lbfgs`, the implementation uses the full parameter list and does not use the same explicit decay/no-decay parameter groups as the other optimizer branches.

### Research-specific caution

SAM and LBFGS have closure semantics different from ordinary first-order optimizers. The training loop must call them using the optimizer API expected by the selected implementation.

## `make_scheduler()`

```python
make_scheduler(optimizer, name=None, config_file={})
```

The builder dispatches by scheduler name. The baseline supports custom schedules including:

```text
temperature
resolution
sharpness
constant
cosine
cosine_warmup
exp
```

The actual scheduler classes are a mixture of MAPS utilities and PyTorch/external scheduler classes.

The custom physics-oriented schedules represent different quantities:

- `temperature`: temperature/annealing state;
- `resolution`: simulation/design resolution;
- `sharpness`: projection/binarization sharpness.

The schedule step count is tied to `configs.run.n_epochs` in the baseline.

## `make_criterion()`

`make_criterion(name, cfgs)` dispatches objective/loss definitions used by the ML training pipeline. The exact available criterion names come from the implementation's explicit dispatch and imported loss classes.

Loss families in the module include:

```text
L2 / normalized-MSE variants
Lp losses
complex-valued L1
masked normalized losses
center NMSE
sliced normalized MSE
Maxwell residual
resolution/gradient losses
fabrication penalty losses
aspect-ratio and distance losses
```

When documenting or configuring a criterion, distinguish between:

```text
raw loss value
normalization/masking configuration
fabrication geometry penalty
physics residual
```

because these have different mathematical meanings even when all ultimately return a scalar tensor.

## Configuration-to-object lifecycle

For a typical ML training workflow:

```text
configs
  |
  +-- dataset -> make_dataloader()
  |
  +-- model   -> make_model()
  |
  +-- model.parameters()
  |      |
  |      +--> make_optimizer()
  |      +--> make_scheduler()
  |
  +-- criterion -> make_criterion()
```

For an inverse-design workflow, `InvDesign` and the inverse-design builder layer provide the more specific optimizer/scheduler contracts documented in [Inverse Design](invdes.md) and [Inverse-Design Optimizer Builder](invdes-builder.md).

Do not replace the inverse-design factories with the generic ML `make_optimizer()` semantics without checking closure, gradient normalization, and sharpness behavior.

## `collate_fn_keep_spatial_res()`

This research helper is worth documenting because its tensor contract is easy for an AI agent to misuse.

### Input

A batch of pairs:

```python
[(data_dict_0, target_dict_0), ...]
```

Each `data_dict` contains spatial tensors such as:

```text
Ez
source
eps
mseWeight
src_mask
epsWeight
```

and scalar metadata such as `scaling_factor`, `device_type`, `eps_bg`, and `grid_step`.

### Output

```python
raw_data, raw_targets
```

with tensors stacked along a leading batch dimension.

The spatial fields are resized before symmetric padding to a common `(H, W)`. `Ez`, source, masks, and material maps therefore do not retain their original per-sample spatial sizes in the returned tensors.

## AI-development invariants

When changing `core/builder.py`, preserve:

- configuration names used for dispatch;
- model-family-specific constructor arguments;
- dataset dictionary schemas;
- distinction between helper collate functions and active DataLoader configuration;
- optimizer closure requirements;
- scheduler physical meaning;
- inverse-design versus generic ML optimizer semantics;
- scalar-versus-tensor objective/loss expectations.

A builder change is an API change even if the scientific class itself is untouched, because experiment configuration files depend on the builder's dispatch names and argument mapping.
