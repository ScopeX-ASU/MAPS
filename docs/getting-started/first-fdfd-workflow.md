# First FDFD Workflow: Bending Optimization and Data Generation

This guide traces the existing tagged script:

```text
data/fdfd/generate_bending.py
```

from `v0.0.2_fdtdx`. It complements the configuration-driven [first inverse-design workflow](first-inverse-design.md) by showing a lower-level, explicit Python construction path.

This is still an inverse-design script rather than a minimal `fdfd_ez` unit test, but it is a useful FDFD getting-started example because the complete device, solver, optimization model, optimizer, scheduler, forward solve, backward solve, plotting, and data export are visible in one file.

## Run the existing script

From the repository root:

```bash
python data/fdfd/generate_bending.py --random_seed 0 --gpu_id 0
```

Optional flags:

```text
--each_step 0|1
--include_perturb 0|1
```

The baseline script explicitly selects a CUDA device, so it is not a CPU-portable tutorial without modifying the script.

## Workflow overview

```text
random seed / GPU
      |
      v
DefaultSimulationConfig
      |
      v
Bending device
      |
      +--> simulation resolution = 50
      |
      +--> high-resolution copy = 1000
      |
      v
BendingOptimization
      |
      v
Adam + cosine LR + sharpness scheduler
      |
      v
opt.forward(sharpness)
      |
      v
results["obj"] + results["breakdown"]
      |
      v
(-results["obj"]).backward()
      |
      v
optimizer.step()
      |
      +--> plots
      +--> HDF5/YAML trajectory data
      +--> optional parameter perturbations
```

## Device geometry

The script targets a nominal raster extent of 256 cells at resolution 50 cells per unit length:

```python
target_img_size = 256
resolution = 50
target_cell_size = target_img_size / resolution
```

so:

```text
target_cell_size = 5.12
```

The input/output port widths are fixed at:

```text
0.48
```

while the port length is randomly selected from approximately `[1.6, 1.8]` and quantized to the simulation grid:

```python
port_len = round(random.uniform(1.6, 1.8) * resolution) / resolution
```

The bending-region size is then chosen so that:

```text
bending_region_size + 2 * port_len = target_cell_size
```

on both design-plane axes.

This is a useful example of a MAPS geometry principle: physical dimensions are selected first, then explicitly aligned to a raster resolution.

## FDFD simulation configuration

The script starts from `DefaultSimulationConfig()` and overrides:

```python
sim_cfg.update(
    dict(
        solver="ceviche_torch",
        border_width=[0, port_len, port_len, 0],
        resolution=50,
        PML=[0.5, 0.5],
        neural_solver=None,
        numerical_solver="solve_direct",
        use_autodiff=False,
    )
)
```

### Solver layers

These names refer to different layers of the simulation stack:

`solver="ceviche_torch"`
: Selects the MAPS/Ceviche Torch FDFD simulation path.

`numerical_solver="solve_direct"`
: Selects the sparse linear-system solution strategy used inside the FDFD backend.

`neural_solver=None`
: Disables a neural initial-guess/solver path.

`use_autodiff=False`
: Selects the non-autodiff simulation-gradient path configured by this workflow; this does **not** mean the overall optimization has no gradients. The optimization still calls `backward()` through MAPS' differentiable/adjoint objective layer.

## Construct the device

The baseline creates:

```python
device = Bending(
    sim_cfg=sim_cfg,
    bending_region_size=bending_region_size,
    port_len=(port_len, port_len),
    port_width=(0.48, 0.48),
    device=operation_device,
)
```

The `Bending` device is responsible for turning these physical dimensions into the geometry, epsilon map, ports, monitor slices, design-region masks, and source/mode metadata used by the simulation and objective layers.

See [Device API contracts](../api/device-base.md) for the general device-layer state model.

## Simulation grid versus parameterization grid

The workflow then creates:

```python
hr_device = device.copy(resolution=1000)
```

This distinction is fundamental:

```text
device      -> FDFD simulation-resolution representation
hr_device   -> high-resolution design/parameterization representation
```

Do not infer that learnable design variables have the same tensor shape as the FDFD epsilon map.

The optimization layer is responsible for transforming the design representation into a simulation-compatible material map.

## Construct the optimization model

```python
opt = BendingOptimization(
    device=device,
    hr_device=hr_device,
    sim_cfg=sim_cfg,
    operation_device=operation_device,
).to(operation_device)
```

Construction initializes the design parameterization and simulation/objective infrastructure described in [Base optimization](../api/base-optimization.md).

The optimization model's forward contract is:

```python
results = opt.forward(sharpness=sharpness)
```

with at least:

```text
results["obj"]
results["breakdown"]
```

`results["obj"]` is a scalar differentiable objective tensor.

`results["breakdown"]` is a mapping of named objective terms. This Bending workflow expects a forward-transmission entry:

```text
"fwd_trans"
```

## Optimizer and schedules

The script uses:

```python
n_epoch = 100
optimizer = torch.optim.Adam(opt.parameters(), lr=0.02)
```

with cosine learning-rate annealing:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.0002,
)
```

and a MAPS sharpness schedule:

```python
sharp_scheduler = SharpnessScheduler(
    initial_sharp=1,
    final_sharp=256,
    total_steps=100,
)
```

Sharpness controls the design projection/parameterization rather than the Maxwell solver itself.

## One optimization step

The essential loop is:

```python
optimizer.zero_grad()
sharpness = sharp_scheduler.get_sharpness()
results = opt.forward(sharpness=sharpness)
(-results["obj"]).backward()
optimizer.step()
scheduler.step()
sharp_scheduler.step()
```

### Objective sign

The negative sign is intentional:

```text
loss = -objective
```

PyTorch's optimizer minimizes the loss, so this performs gradient ascent on the MAPS FoM/objective.

This convention is also used by the higher-level `InvDesign` workflow.

## Inspect objective breakdown

The script prints every objective component:

```python
for k, obj in results["breakdown"].items():
    print(k, obj["value"])
```

The scientific meaning of each term is determined by the objective configuration/model. Do not assume all values are probabilities or normalized powers merely because they are named transmission/reflection terms.

See [Objective contracts](../api/objective.md).

## Design-region material state

After backward, the script obtains:

```python
current_design_region_dict = opt.get_design_region_eps_dict()
```

This is a mapping from named design regions to their current material/permittivity representations.

The script compares consecutive designs with flattened cosine similarity:

```text
similarity = cosine_similarity(
    previous_region.flatten(),
    current_region.flatten(),
)
```

and uses the similarity to decide when to dump another trajectory sample.

This comparison is a data-generation heuristic; it is not the physical optimization objective.

## Plotting fields

The script plots the forward case with:

```python
opt.plot(
    eps_map=opt._eps_map,
    obj=results["breakdown"]["fwd_trans"]["value"],
    field_key=("in_slice_1", 1.55, "Ez1", 300),
    field_component="Ez",
    in_slice_name="in_slice_1",
    exclude_slice_names=[],
)
```

This illustrates the field-key schema used by this workflow:

```text
(input slice, wavelength, mode, control/state value)
```

and requests the `Ez` component for visualization.

The final element `300` is part of the existing key used by this workflow; it should not be generalized into a universal temperature/control-state default without tracing the objective/device configuration.

## Data export

Trajectory samples are written as paired files:

```text
*.h5
*.yml
```

under a path rooted at:

```text
./data/fdfd/bending/raw_opt_traj_ptb/mfs_50
```

The script calls:

```python
opt.dump_data(
    filename_h5=filename_h5,
    filename_yml=filename_yml,
    step=step,
)
```

and for selected later samples can request:

```text
use_high_res_eps=True
binarize_eps=True
```

This makes the script both an optimization example and a dataset-generation workflow.

## Early stopping

The baseline uses:

```text
early_stop_threshold = 1e-3
patience = 3
```

It stores recent objective-breakdown values and stops when all compared changes remain below the threshold.

This is a heuristic convergence criterion based on objective terms, not a gradient norm or Maxwell residual tolerance.

## Optional perturbation dataset

With:

```bash
--include_perturb 1
```

selected parameter entries are sign-flipped with probabilities:

```text
0.1, 0.3, 0.5
```

The script then runs a separate forward/backward calculation and dumps the perturbed state before restoring the original model parameters and optimizer state.

The perturbation is performed under a deterministic context whose seed depends on the optimization step and perturbation index.

This branch is intended for trajectory/data augmentation and should not be confused with the main optimization update.

## Tensor and grid expectations

This script does not hard-code one final FDFD tensor shape because the physical `port_len` is randomized and the device layer derives the rasterized geometry.

The stable contracts are:

- FDFD physical epsilon/field maps are 2D grid quantities `[Nx, Ny]` at the solver layer;
- sparse solver vectors are flattened to `[N]`, `N = Nx * Ny`;
- high-resolution design/material tensors can have different `Nx, Ny` from the simulation map;
- electromagnetic fields are complex-valued;
- `Ez1` identifies the mode/polarization convention used by this workflow;
- field keys carry wavelength and control-state metadata in addition to spatial data.

See [FDFD API](../api/fdfd.md) and [FDFD solver](../api/fdfd-solver.md).

## What this example teaches

This workflow exposes the major MAPS FDFD data path in a compact form:

```text
physical geometry
 -> rasterized Bending device
 -> high-resolution design representation
 -> differentiable material map
 -> FDFD solve
 -> modal objective
 -> adjoint/autograd gradient
 -> parameter update
 -> trajectory export
```

It also demonstrates why MAPS documentation distinguishes physical geometry, parameterization resolution, simulation resolution, sparse-solver representation, and objective representation rather than describing all of them as generic tensors.

## Related documentation

- [First inverse-design workflow](first-inverse-design.md)
- [FDFD API](../api/fdfd.md)
- [FDFD sparse solver](../api/fdfd-solver.md)
- [Device API](../api/device-base.md)
- [Parametrization](../api/parametrization.md)
- [Objective contracts](../api/objective.md)
- [Base optimization](../api/base-optimization.md)
