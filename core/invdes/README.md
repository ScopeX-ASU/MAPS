# MAPS-InvDes

**MAPS-InvDes** is the differentiable photonic simulation and inverse-design subsystem of MAPS. It connects physical device geometry, multi-resolution material representations, electromagnetic solvers, modal/field objectives, differentiable parameterizations, fabrication-aware transforms, and numerical optimization into one research workflow.

This README is the subsystem entry point. Detailed scientific/API contracts live under [`docs/`](../../docs/index.md).

## What MAPS-InvDes provides

MAPS-InvDes supports the core workflow:

```text
physical device
    |
    v
grid + material rasterization
    |
    v
design-region parameterization
    |
    v
electromagnetic simulation
    |
    v
objective / figure of merit
    |
    v
adjoint / autograd gradient
    |
    v
optimizer update
```

Major capabilities in the current codebase include:

- differentiable 2D FDFD simulation through the MAPS/Ceviche Torch path;
- GPU-oriented 3D FDTD through `fdtdx`, including differentiable/adjoint support;
- native, export, and nonuniform rectilinear optical-grid representations;
- design-region parameterizations including level-set, box, circle/cylinder,
  edge-box, and polygon families;
- symmetry, binary projection, roughness, lithography, etching, blur/MFS, and related transforms;
- modal/S-parameter, flux, near-to-far, shape, phase/response, and structural objectives;
- multi-resolution design and simulation maps;
- optical/thermal/electrical material-map coupling in supported workflows;
- configurable optimizers, learning-rate schedulers, and sharpness schedulers;
- plotting, checkpointing, callbacks, GDS/epsilon export, and optimization trajectory generation.

## Architecture

A useful mental model is:

```text
Device
  geometry
  materials
  grids
  ports
  monitors
  sources/modes
  design-region masks
      |
      v
Parametrization
  learnable parameters
  normalized density
  fabrication transforms
  material/permittivity map
      |
      v
Simulation
  FDFD / FDTD-X
      |
      v
Fields
  complex E/H components
      |
      v
Objective / FoM
  transmission / reflection / flux / shape / ...
      |
      v
Gradient
  adjoint + PyTorch autograd
      |
      v
InvDesign
  optimizer / schedulers / callbacks
```

The current device layer is intentionally broad and owns geometry, rasterization, grids, ports, monitors, source construction, and several material-map responsibilities. This is a known future refactoring target; see [`docs/architecture/device-base.md`](../../docs/architecture/device-base.md) before modifying that architecture.

## Installation

The validated installation procedure is documented in:

- [`docs/getting-started/installation.md`](../../docs/getting-started/installation.md)
- [`README_install.md`](../../README_install.md)

For the default 2D FDFD/inverse-design stack:

```bash
bash install.sh
```

For the optional 3D FDTD-X/JAX stack:

```bash
INSTALL_OPTIONAL_3D=1 bash install.sh
```

Then verify the environment:

```bash
python verify_install.py
```

or, for the optional 3D stack:

```bash
python verify_install.py --require-optional-3d
```

## Quick start: Bending

The Bending device is the current canonical getting-started example because the tagged baseline contains both a configuration-driven optimization workflow and an explicit FDFD/data-generation workflow.

### Configuration-driven inverse design

The baseline configuration is:

```text
configs/bending/default.yml
```

and the root training entry point consumes configuration files as:

```bash
python train.py configs/bending/default.yml
```

The configuration selects, among other settings:

```yaml
model:
  name: BendingOptimization
  device_type: Bending
  hr_resolution: 310
  sim_cfg:
    solver: ceviche_torch
    resolution: 50
    wl_cen: 1.55
    wl_width: 0
    n_wl: 1
    numerical_solver: solve_direct
  design_region_param_cfgs:
    design_region_1:
      method: levelset
      rho_resolution: [25, 25]
```

See the verified walkthrough:

[`docs/getting-started/first-inverse-design.md`](../../docs/getting-started/first-inverse-design.md)

### Explicit FDFD workflow

For a lower-level example where device/model/optimizer/schedulers are visible in one script:

```bash
python data/fdfd/generate_bending.py --random_seed 0 --gpu_id 0
```

This workflow constructs:

```text
Bending
 -> high-resolution device copy
 -> BendingOptimization
 -> Adam
 -> cosine LR scheduler
 -> sharpness scheduler
 -> forward FDFD objective
 -> backward gradient
 -> trajectory/plot export
```

See:

[`docs/getting-started/first-fdfd-workflow.md`](../../docs/getting-started/first-fdfd-workflow.md)

## Device model

A MAPS device is more than an epsilon tensor. It establishes the physical and numerical state consumed by simulation and optimization:

```text
physical geometry
 -> optical grid(s)
 -> epsilon/material maps
 -> active/design-region masks
 -> ports
 -> monitor slices
 -> source/mode profiles
 -> normalization state
```

For a verified Bending construction example, see:

[`docs/getting-started/first-device.md`](../../docs/getting-started/first-device.md)

For the complete architecture contract, see:

- [`docs/api/device-base.md`](../../docs/api/device-base.md)
- [`docs/architecture/device-base.md`](../../docs/architecture/device-base.md)

## Grids and coordinates

MAPS-InvDes may maintain multiple representations of the same physical device:

```text
native optical grid
export optical grid
FDTD-X native/field grid
thermal grid
high-resolution design grid
```

Physical coordinates are authoritative. Tensor shape equality does not imply that two tensors occupy the same coordinates.

When adding or modifying code, always identify:

- physical axis meaning;
- grid ownership;
- coordinate arrays/boundaries;
- tensor shape and dtype;
- physical units;
- interpolation/resampling rule.

See:

- [`docs/architecture/shared-grid-and-field-data.md`](../../docs/architecture/shared-grid-and-field-data.md)
- [`docs/api/grid-field-utils.md`](../../docs/api/grid-field-utils.md)

## Design parameterization

Each named design region is represented by a parametrization module that maps learnable variables to a normalized design density and then to a material representation.

The important distinction is:

```text
learnable parameter
    != normalized density
    != physical permittivity
    != full simulation epsilon map
```

The baseline parameterization registry includes:

```text
levelset
box
edgebox
polygon
```

Transforms are ordered and can include operations such as symmetry, binarization, roughness, lithography, etching, blur/MFS filtering, and resolution conversion.

Transform order is part of the scientific definition of the optimization problem; do not reorder transforms as a code-cleanup operation.

See [`docs/api/parametrization.md`](../../docs/api/parametrization.md).

## FDFD

The 2D FDFD path operates on physical grid quantities such as:

```text
eps_r: [Nx, Ny]
field: [Nx, Ny]
```

while the sparse linear-system boundary uses:

```text
A: [N, N]
b: [N]
x: [N]
N = Nx * Ny
```

Complex electromagnetic fields, grid ordering, boundary conditions, modal normalization, and adjoint behavior are part of the API contract.

See:

- [`docs/api/fdfd.md`](../../docs/api/fdfd.md)
- [`docs/api/fdfd-solver.md`](../../docs/api/fdfd-solver.md)

## FDTD-X

The 3D time-domain path bridges MAPS/PyTorch state to JAX/`fdtdx`.

A typical physical permittivity grid has shape:

```text
[Nx, Ny, Nz]
```

and the field pipeline works with six electromagnetic components:

```text
Ex, Ey, Ez, Hx, Hy, Hz
```

The implementation handles native/export grids, broadband phasor detection, Yee-grid spatial/temporal colocation, and a differentiable adjoint bridge.

Broadband adjoint behavior is experimental in the current implementation and should be validated against single-frequency adjoint gradients for the target problem.

See:

- [`docs/api/fdtd.md`](../../docs/api/fdtd.md)
- [`docs/api/fdtd-solver.md`](../../docs/api/fdtd-solver.md)

## Objectives and FoM

The objective layer converts simulated fields and source/monitor profiles into optimization metrics.

Supported objective families include modal/eigenmode overlap, S-parameters, flux, near-to-far metrics, phase/response recording, shape similarity, and structural/interface penalties.

Optimization models return a structure containing at least:

```python
results["obj"]
results["breakdown"]
```

where `results["obj"]` is the fused scalar differentiable objective and `results["breakdown"]` records named objective terms and metadata.

See:

- [`docs/api/objective.md`](../../docs/api/objective.md)
- [`docs/api/fom-layer.md`](../../docs/api/fom-layer.md)

## Optimization loop

`InvDesign` provides the higher-level optimization orchestration:

```text
read sharpness
 -> model forward
 -> results["obj"]
 -> backward on -obj
 -> optional gradient normalization
 -> optimizer step
 -> callbacks
 -> scheduler updates
 -> plotting/checkpoint/export
```

The negative sign is intentional: standard PyTorch optimizers minimize, so minimizing `-obj` maximizes the MAPS figure of merit.

The subsystem supports multiple optimizer/scheduler families, including standard PyTorch optimizers and custom research optimizers.

See:

- [`docs/api/invdes.md`](../../docs/api/invdes.md)
- [`docs/api/invdes-builder.md`](../../docs/api/invdes-builder.md)

## Scientific API conventions

MAPS-InvDes documentation and docstrings should specify more than prose behavior. Important APIs should document, where applicable:

```text
shape
meaning of each dimension
dtype
real/complex representation
device
physical units
normalization
container/dictionary schema
mathematical equation
scientific assumptions
references
realistic usage example
```

For example, writing only "field tensor" is insufficient when the correct contract is a complex tensor with a specific wavelength/component/spatial layout.

See [`docs/development/documentation.md`](../../docs/development/documentation.md).

## Development and AI coding


The key rule is to preserve scientific contracts rather than merely Python-level behavior. In particular, changes must account for:

- physical coordinate conventions;
- native/export grid relationships;
- tensor dimension semantics;
- complex-field conventions;
- source/mode normalization;
- parameterization transform order;
- objective definitions;
- solver/adjoint sign and scaling conventions;
- differentiability across interpolation and custom autograd paths.

`device_base.py` is a known future refactoring target. Do not opportunistically decompose it while fixing unrelated behavior; use the architecture documentation as the specification for a dedicated refactor.

## Documentation map

### Getting started

- [Installation](../../docs/getting-started/installation.md)
- [First device: Bending](../../docs/getting-started/first-device.md)
- [First FDFD workflow](../../docs/getting-started/first-fdfd-workflow.md)
- [First inverse-design run](../../docs/getting-started/first-inverse-design.md)

### Architecture

- [MAPS system architecture](../../docs/architecture/overview.md)
- [Device architecture](../../docs/architecture/device-base.md)
- [Shared grid and field data](../../docs/architecture/shared-grid-and-field-data.md)

### API contracts

- [Device](../../docs/api/device-base.md)
- [Grid/field utilities](../../docs/api/grid-field-utils.md)
- [FDFD](../../docs/api/fdfd.md)
- [FDFD solver](../../docs/api/fdfd-solver.md)
- [FDTD](../../docs/api/fdtd.md)
- [FDTD solver](../../docs/api/fdtd-solver.md)
- [Parametrization](../../docs/api/parametrization.md)
- [Base optimization/material maps](../../docs/api/base-optimization.md)
- [Objective](../../docs/api/objective.md)
- [FoM layer](../../docs/api/fom-layer.md)
- [InvDesign](../../docs/api/invdes.md)
- [InvDesign builder](../../docs/api/invdes-builder.md)

## Relationship to MAPS

MAPS-InvDes is one of the three major MAPS components:

```text
MAPS-Data   -> simulation dataset generation
MAPS-Train  -> AI model training/evaluation
MAPS-InvDes -> physics-based differentiable inverse design
```

The root MAPS README should remain the balanced entry point for all three components. This README intentionally focuses only on the inverse-design subsystem.

## Citation

If MAPS contributes to published work, cite the MAPS paper:

```bibtex
@inproceedings{ma2025maps,
    title={{MAPS: Multi-Fidelity AI-Augmented Photonic Simulation and Inverse Design Infrastructure}},
    author={Pingchuan Ma and Zhengqi Gao and Meng Zhang and Haoyu Yang and Mark Ren and Rena Huang and Duane S. Boning and Jiaqi Gu},
    year={2025},
    booktitle={Design, Automation and Test in Europe Conference (DATE)},
    url={https://arxiv.org/abs/2503.01046},
}
```

For new solver, optimization, fabrication, or device-specific methods built on MAPS, also cite the relevant underlying method/software papers as appropriate.
