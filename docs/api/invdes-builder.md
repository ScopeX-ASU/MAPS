# Inverse-Design Optimizer and Scheduler Builder

This page documents `core/invdes/builder.py` in the `v0.0.2_fdtdx` baseline. It records the configuration contract used by `InvDesign` without changing implementation behavior.

## `make_optimizer()`

```python
make_optimizer(params, total_config)
```

The function reads:

```text
total_config.optimizer
```

and dispatches on the lower-cased `optimizer.name`.

`params` must be an iterable compatible with the selected PyTorch/custom optimizer, normally `devOptimization.parameters()`.

## Supported optimizer names

### `sgd`

Uses `torch.optim.SGD`.

Required/used configuration:

```text
lr
momentum
weight_decay
```

Nesterov acceleration is enabled.

### `adam`

Uses MAPS' custom `core.invdes.optimizer.Adam`.

Configuration:

```text
lr
weight_decay
betas       default (0.9, 0.999)
eps         default 1e-8
init_v      default 0.0
```

### `adamw`

Uses `torch.optim.AdamW` with:

```text
lr
weight_decay
```

### `dadaptadam`

Uses `DAdaptAdam` with:

```text
lr
betas       default (0.9, 0.999)
weight_decay
```

### `sam_sgd`

Uses SAM with SGD as the base optimizer.

```text
rho         default 0.5
adaptive    default True
lr
weight_decay
```

### `sam_adam`

Uses SAM with Adam as the base optimizer.

```text
rho         default 0.001
adaptive    default True
lr
weight_decay
```

### `lbfgs`

Uses `torch.optim.LBFGS`.

```text
lr
line_search_fn     default None
max_iter           default 4
max_eval           default None
history_size       default 100
tolerance_grad     default 1e-7
tolerance_change   default 1e-9
```

`InvDesign` supplies its closure through `optimizer.step(self.closure)`, which is particularly important for LBFGS because the optimizer may reevaluate the closure multiple times per step.

### `nesterov`

Uses `NesterovAcceleratedGradientOptimizer`.

```text
lr
alg                     default "bb"
constraint_fn           default None
block_tile_size         default tile_size or None
block_blur_kernel_size  default 3
```

### `adahessian`

Uses `Adahessian`.

```text
lr
betas           default (0.9, 0.999)
eps             default 1e-4
weight_decay
hessian_power   default 0.5
```

Because this is a second-order/Hessian-aware optimizer, changes to closure/backward behavior can affect its correctness even if first-order optimizers continue to work.

### `muon`

Uses `Muon`.

```text
lr
weight_decay
momentum    default 0.9
nesterov    default True
```

### nonlinear conjugate-gradient variants

MAPS exposes several `BASIC_NCG` configurations:

```text
cg_fr
cg_prp
cg_hs
cg_cd
cg_ls
cg_dy
cg_hz
cg_hs-dy
```

These correspond to different nonlinear conjugate-gradient update formulas and line-search choices.

#### `cg_fr`

```text
method = "FR"
line_search = "Strong_Wolfe"
c1 default 1e-4
c2 default 0.5
lr default 0.2
max_ls default 5
```

#### `cg_prp`

```text
method = "PRP"
line_search = "Armijo"
c1 default 1e-4
c2 default 0.9
lr default 10
rho default 0.5
restart_interval default None
```

#### `cg_hs`

```text
method = "HS"
line_search = "Strong_Wolfe"
c1 default 1e-4
c2 default 0.4
lr default 0.2
max_ls default 5
```

#### `cg_cd`

```text
method = "CD"
line_search = "Armijo"
c1 default 1e-4
c2 default 0.9
lr default 1
rho default 0.5
```

#### `cg_ls`

```text
method = "LS"
line_search = "Armijo"
c1 default 1e-4
c2 default 0.9
lr default 1
rho default 0.5
```

#### `cg_dy`

```text
method = "DY"
line_search = "Strong_Wolfe"
c1 default 1e-4
c2 default 0.9
lr default 0.2
max_ls default 5
```

#### `cg_hz`

```text
method = "HZ"
line_search = "Strong_Wolfe"
c1 default 1e-4
c2 default 0.9
lr default 0.2
max_ls default 5
```

#### `cg_hs-dy`

```text
method = "HS-DY"
line_search = "Armijo"
c1 default 1e-4
c2 default 0.9
lr default 10
rho default 0.5
```

Unsupported optimizer names raise `NotImplementedError`.

## `make_scheduler()`

```python
make_scheduler(optimizer, scheduler_type, config_total)
```

`scheduler_type` is the name of a configuration section, such as:

```text
"lr_scheduler"
"sharp_scheduler"
```

The builder evaluates:

```python
getattr(config_total, scheduler_type).name.lower()
```

and constructs the corresponding scheduler.

## Supported scheduler names

### `temperature`

Uses `TemperatureScheduler` with values read from `temp_scheduler`:

```text
initial_T = temp_scheduler.lr
final_T = temp_scheduler.lr_min
total_steps = run.n_epochs
```

### `resolution`

Uses `ResolutionScheduler`:

```text
initial_res = res_scheduler.init_res
final_res = res_scheduler.final_res
total_steps = run.n_epochs
```

### `sharpness`

Uses `SharpnessScheduler`:

```text
initial_sharp = sharp_scheduler.init_sharp
final_sharp = sharp_scheduler.final_sharp
total_steps = run.n_epochs
mode = sharp_scheduler.mode, default "cosine"
```

`InvDesign.before_step()` reads `sharp_scheduler.get_sharpness()` and forwards the value to the optimization model.

### `constant`

Uses `torch.optim.lr_scheduler.LambdaLR` with a constant multiplier of 1.

### `cosine`

Uses `torch.optim.lr_scheduler.CosineAnnealingLR`:

```text
T_max = run.n_epochs
eta_min = lr_scheduler.lr_min
```

### `cosine_warmup`

Uses `CosineAnnealingWarmupRestarts`. The current implementation reads fields from both `optimizer` and `scheduler` configuration sections:

```text
first_cycle_steps = run.n_epochs
max_lr = optimizer.lr
min_lr = scheduler.lr_min
warmup_steps = scheduler.warmup_steps
```

This mixed-section dependency is part of the current configuration contract and should be checked carefully when using this scheduler.

### `exp`

Uses `torch.optim.lr_scheduler.ExponentialLR` with:

```text
gamma = scheduler.lr_gamma
```

Unsupported scheduler names raise `NotImplementedError`.

## Builder functions that are not active APIs

Two functions in the baseline should not be treated as implemented public factories:

`make_model()`
: Immediately raises `NotImplementedError` before the historical/dead dispatch code below it.

`make_criterion()`
: Immediately returns the `NotImplementedError` class object before the historical criterion-dispatch code below it.

Do not document the unreachable branches as supported behavior or build new code against them without first making an intentional implementation change and adding tests.

## Representative configuration

The default `InvDesign` configuration corresponds approximately to:

```python
optimizer = Config(
    name="Adam",
    lr=1e-2,
    weight_decay=0,
)

lr_scheduler = Config(
    name="cosine",
    lr_min=2e-4,
)

sharp_scheduler = Config(
    name="sharpness",
    mode="cosine",
    init_sharp=1,
    final_sharp=256,
)
```

The builder is then used internally as:

```python
optimizer = builder.make_optimizer(
    params=devOptimization.parameters(),
    total_config=cfg,
)

lr_scheduler = builder.make_scheduler(
    optimizer=optimizer,
    scheduler_type="lr_scheduler",
    config_total=cfg,
)

sharp_scheduler = builder.make_scheduler(
    optimizer=optimizer,
    scheduler_type="sharp_scheduler",
    config_total=cfg,
)
```

## Safety invariants

Optimizer/scheduler configuration is scientifically relevant because it changes the inverse-design trajectory. Future edits should preserve or explicitly validate:

- optimizer-name dispatch;
- closure requirements;
- line-search parameters;
- second-order optimizer behavior;
- gradient normalization interaction;
- scheduler step count relative to `run.n_epochs`;
- sharpness value passed into the physical parameterization;
- default values used when configuration fields are absent.
