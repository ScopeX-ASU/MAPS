# Optimization Run Management

`OptimizationRunManager` provides a versioned, local-first storage contract for
MAPS inverse-design runs. It is optional so existing scripts retain their
legacy `sim_cfg.plot_root` behavior.

## Configuration

Pass a `run_management` section to `InvDesign`:

```python
run_management=Config(
    enabled=True,
    root="./runs",
    experiment="bending_3d",
    tag="bending_3d_uniform_15angle",
)
```

`root` is the user-selected work directory. `experiment` and `tag` must be
single path segments. `tag` is a human-readable label, while `run_id` is the
unique machine identity. When `run_id` is omitted, MAPS generates:

```text
<UTC timestamp>_<tag>_<8-character random suffix>
```

For example, `20261017T142530Z_bending_3d_uniform_15angle_7f12a9c8`.
An exact `run_id` may be supplied for controlled workflows, and MAPS refuses to
overwrite it unless `checkpoint_cfgs.resume=True`.

## Directory contract

The example above creates:

```text
runs/bending_3d/20261017T142530Z_bending_3d_uniform_15angle_7f12a9c8/
├── manifest.yaml
├── config/
│   └── resolved.yaml
├── metadata/
│   ├── environment.json
│   └── source.json
├── metrics/
│   ├── history.jsonl
│   └── summary.json
├── checkpoints/
├── design/
├── geometry/
├── layout/
├── plots/
└── logs/
    └── run.log
```

The manifest uses schema `maps.optimization-run/v1` and records lifecycle
status plus registered artifacts. Immutable final artifacts may include a
SHA-256 digest. The mutable `latest` checkpoint is registered without hashing
to avoid doubling checkpoint I/O on every save.

## Runtime behavior

- `config/resolved.yaml` contains the effective InvDesign configuration and
  portable optimization-model metadata.
- `metadata/environment.json` records Python, platform, Torch, CUDA visibility,
  and every allowlisted argument from `core.env_args`. Each MAPS argument
  records whether it came from the process environment or its registered
  default, its raw value, and its parsed backend/device values. Arbitrary
  environment variables are not copied, avoiding accidental secret capture.
- `metrics/history.jsonl` receives one append-only event per completed step,
  including FoM, breakdown, gradient norm, sharpness, and learning rates.
- full resumable checkpoints go under `checkpoints/`;
- high-resolution epsilon images go under `design/`;
- Tidy3D device visualizations (geometry and physical port planes) go under
  `geometry/`;
- GDS goes under `layout/`;
- runtime plots go under `plots/`;
- MAPS messages emitted through the project Python logger are mirrored to
  `logs/run.log` while continuing to appear in the terminal;
- `metrics/summary.json` and `manifest.yaml` record completion, failure, or
  interruption.

Direct `print()` output from third-party libraries is not redirected. Libraries
should use the project logger when their messages need to become part of the
managed run record.

### Initialization-time plots

Some device models, including `Bending`, run source normalization and create
diagnostic plots while the device or optimization model is being constructed.
Create the manager first in those workflows so initialization and optimization
share one run directory:

```python
from pyutils.general import logger

from core.invdes.run_management import OptimizationRunManager

run_manager = OptimizationRunManager(
    root="./runs",
    experiment="bending_3d",
    tag=exp_name,
    resume=resume_checkpoint is not None,
    resume_from=resume_checkpoint,
)
run_manager.attach_logger(logger)
run_manager.bind_simulation_config(sim_cfg)

# Construct the device and optimization model after assigning plot_root.
device = Bending(..., sim_cfg=sim_cfg)
optimization = BendingOptimization(..., device=device, sim_cfg=sim_cfg)
invdesign = InvDesign(
    devOptimization=optimization,
    run_manager=run_manager,
    ...,
)
```

`InvDesign` reuses the supplied manager and copies its identity into the
resolved `run_management` configuration. Do not also create a second manager
or duplicate its run ID. Workflows without initialization-time artifacts may
continue to use only the `run_management=Config(...)` form.

On managed resume, use the same `root` and `experiment`, set
`checkpoint_cfgs.resume=True`, and point `checkpoint_cfgs.load_ckpt` at the
checkpoint under that run. MAPS infers the original `run_id` from the
`checkpoints/` parent directory. The checkpoint flag both restores optimization
state and reopens the managed run, so there is no second run-management resume
flag or manual run-ID lookup. PyTorch checkpoints must only be loaded from
trusted sources.

## Compatibility

When `run_management.enabled=False`, no managed run directory is created and
all existing paths continue to use `devOptimization.sim_cfg.plot_root`.
