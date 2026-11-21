# First Inverse-Design Run: Bending Device

This guide uses the `v0.0.2_fdtdx` Bending configuration as the first verified MAPS end-to-end inverse-design example. It documents the actual tagged configuration and `train.py` entry point rather than introducing a new example API.

## What this example demonstrates

The Bending example exercises the central MAPS inverse-design path:

```text
YAML configuration
      |
      v
builder.make_device()
      |
      v
builder.make_model()
      |
      v
BendingOptimization
      |
      +--> level-set design region
      +--> differentiable Ceviche Torch FDFD
      +--> forward-transmission objective
      |
      v
Adam + cosine/sharpness/resolution schedulers
      |
      v
field/objective plots + checkpoints
```

The example is intentionally small enough to understand while still exercising physical geometry, ports, design-region parameterization, a differentiable electromagnetic solver, an objective, and optimization.

## Baseline configuration

The primary configuration is:

```text
configs/bending/default.yml
```

The tagged configuration defines:

```yaml
model:
  name: "BendingOptimization"
  device_type: "Bending"
  device_cfg:
    box_size: [2, 2]
    port_len: [1.8, 1.8]
    port_width: [0.48, 0.48]
  hr_resolution: 310
```

The optical simulation configuration is:

```yaml
sim_cfg:
  solver: "ceviche_torch"
  resolution: 50
  wl_cen: 1.55
  wl_width: 0
  n_wl: 1
  PML: [0.5, 0.5]
  numerical_solver: solve_direct
  use_autodiff: False
```

The design region uses level-set parameterization:

```yaml
design_region_param_cfgs:
  design_region_1:
    method: levelset
    rho_resolution: [25, 25]
    transform:
      - type: blur
        mfs: 0.05
        resolutions: [310, 310]
        dim: xy
      - type: binarize
    init_method: random
    denorm_mode: linear_eps
    interpolation: bilinear
```

The primary objective in the plotting configuration is `fwd_trans`.

## Entry point

The repository entry point is:

```text
train.py
```

It expects the configuration file as the first positional argument:

```bash
python train.py configs/bending/default.yml
```

`train.py` loads the YAML recursively, applies command-line overrides, selects CPU/CUDA, constructs the device and optimization model through `core.builder`, creates the optimizer and schedulers, and runs the training loop. fileciteturn180file0L1-L2 fileciteturn181file0L1-L2

## Construction sequence

The actual entry point performs the following construction sequence:

```python
device = builder.make_device(device=operation_device)

model = builder.make_model(
    device=operation_device,
    random_state=...,
    optDevice=device,
)

optimizer = builder.make_optimizer(
    [p for p in model.parameters() if p.requires_grad],
    name=configs.optimizer.name,
    configs=configs.optimizer,
)

lr_scheduler = builder.make_scheduler(
    optimizer,
    config_file=configs.lr_scheduler,
)

sharp_scheduler = builder.make_scheduler(
    optimizer,
    name="sharpness",
    config_file=configs.sharp_scheduler,
)

res_scheduler = builder.make_scheduler(
    optimizer,
    name="resolution",
    config_file=configs.res_scheduler,
)
```

This distinction matters: the **device** and the **optimization model** are separate objects. The builder uses the device configuration to construct the physical problem and then passes the configured device into the model family. fileciteturn181file0L1-L2

## Design variables

The Bending configuration defines one design region named:

```text
design_region_1
```

The intrinsic parameterization grid is:

```text
[25, 25]
```

The parameterization is not the same as the simulation grid. The simulation uses a resolution of 50, while the high-resolution device path is configured at 310. The optimization stack can therefore contain multiple tensor representations of the same physical region.

The transform sequence is ordered:

```text
level-set variables
      |
      v
normalized density
      |
      v
blur / MFS transform
      |
      v
binary projection
      |
      v
material/permittivity map
```

The transform order is part of the optimization definition and must not be casually reordered.

## Electromagnetic simulation

The Bending example uses:

```text
solver        = "ceviche_torch"
wavelength    = 1.55
n_wl          = 1
```

The device/port layer constructs source profiles and monitor information. The field solver produces complex electromagnetic fields, with the 2D polarization convention determined by the source mode. For the example's `Ez1` field configuration, the forward solution is represented through the `Hx`, `Hy`, and `Ez` components used by the FDFD layer.

The detailed field contract is documented in [FDFD API](../api/fdfd.md).

## Objective

The configuration identifies:

```text
fwd_trans
```

as the forward-transmission quantity shown in plots. The objective layer produces a scalar objective and a `breakdown` mapping that records individual objective terms. The optimization loop then uses the negative objective as the minimization loss:

```text
loss = -objective
```

so that ordinary gradient descent increases the physical figure of merit. fileciteturn180file0L1-L2

## Training schedule

The baseline configuration specifies:

```text
optimizer: Adam
learning rate: 0.01
weight decay: 0
LR scheduler: cosine
minimum LR: 0
run.n_epochs: 100
random_state: 42
deterministic: 1
```

The entry point also constructs separate schedulers for learning rate, sharpness, and resolution. During each epoch, the training loop obtains the current sharpness and resolution, evaluates the model, backpropagates `-FoM`, steps the optimizer, then advances the learning-rate scheduler. fileciteturn181file0L1-L2

## Outputs

With plotting enabled in the Bending configuration, MAPS is configured to use:

```text
plot root: plot
plot directory: bending
plot name: bending_plot
interval: 5
objective: fwd_trans
field key:
    ("in_slice_1", 1.55, "Ez1", 300)
```

The training code calls the model plotting path using a source/field key centered at:

```text
("in_port_1", 1.55, 1)
```

and the `Ez` component for the forward-transmission visualization. fileciteturn180file0L1-L2

The checkpoint configuration records a checkpoint directory and allows optional GDS export. The default tagged configuration has `save_model: False`, so checkpoint retention is not automatically enabled unless the configuration is changed.

## Command-line overrides

`train.py` loads the YAML and then applies additional command-line options. Therefore an experiment can override configuration values without copying the entire YAML.

For example, the entry-point logic supports overrides of the form:

```bash
python train.py configs/bending/default.yml \
    --run.n_epochs=10 \
    --run.gpu_id=0 \
    --optimizer.lr=0.005
```

The exact nested-key behavior is provided by the repository's `pyutils.config` dependency; MAPS passes the remaining arguments directly to `configs.update()`. fileciteturn181file0L1-L2

## Important runtime prerequisites

The training entry point is not a pure standalone Python example. It initializes experiment logging through Weights & Biases and imports the repository's `pyutils` stack. It may therefore require:

- the MAPS environment from `environment.yml`;
- the external `pyutils` dependency used throughout the repository;
- CUDA when `run.use_cuda=1`;
- configured W&B authentication because `train.py` calls `wandb.login()` and then `wandb.init(...)`.

This page intentionally does not claim that the example can run in an offline environment without those services. fileciteturn181file0L1-L2

## What to inspect when modifying the example

The important scientific contracts are:

```text
Bending geometry
  -> port positions and directions
  -> monitor slices
  -> source mode "Ez1"
  -> FDFD field components
  -> fwd_trans objective
  -> gradient to level-set parameters
```

A change to any of these boundaries should update the corresponding API documentation and, ideally, add a focused regression test.

## Related documentation

- [System architecture](../architecture/overview.md)
- [Configuration and build flow](../architecture/configuration-and-build-flow.md)
- [Device API](../api/device-base.md)
- [FDFD API](../api/fdfd.md)
- [Parametrization](../api/parametrization.md)
- [Objective contracts](../api/objective.md)
- [Inverse-design workflow](../api/invdes.md)
