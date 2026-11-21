# Architecture Overview

MAPS is a multi-fidelity research software stack spanning differentiable 2D/3D electromagnetic simulation, thermo-optic heat simulation, adjoint photonic inverse design, near-to-far projection, fabrication-variation modeling, and neural surrogate training.

## Interactive Runtime Architecture (Archify)

Explore the code-backed, interactive runtime map generated with [Archify](https://github.com/tt-a1i/archify):

- **[Open Interactive Runtime Architecture (`maps-runtime.architecture.html`)](maps-runtime.architecture.html)** — standalone HTML viewer (download or open directly in any browser; includes **4 guided views**, upstream/downstream reach tracing, light/dark themes, source file/line evidence cards, and PNG/SVG/WebM export).
- **[Typed Archify Source (`maps-runtime.architecture.json`)](maps-runtime.architecture.json)** — schema-validated JSON IR backing the diagram.

Guided views available in the interactive viewer toolbar:

1. **MAPS-InvDes Loop**: `InvDesign` -> `BaseOptimization` -> `N_Ports` / `gpu_subpixel` -> `SimulatedFoM` / `ObjectiveFunc` -> physics solvers -> `OptimizationRunManager` (checkpoints, GDSII, 3D plots).
2. **2D/3D Multi-Physics Solvers**: PyTorch autograd VJP bridges to 2D Ceviche/Pardiso FDFD (`core/fdfd`), 3D JAX FDTD-X broadband Chebyshev & sparse-phasor solvers (`core/fdtd`, `thirdparty/fdtdx`), and JAX-FEM thermo-optic heat (`core/heat`).
3. **MAPS-Data & MAPS-Train**: Multi-GPU solver sweeps (`data/`) -> HDF5 Maxwell datasets -> FNO / NeurOLight training (`train_NN.py`, `core/train/`) -> neural solver surrogates.
4. **Fabrication & Variation**: Level-set & morphological parametrization, CUDA `gpu_subpixel` rasterization, `RobustForward` corner sampling (`core/invdes/variation/`), Stratton-Chu / angular-spectrum near-to-far projection (`core/nearfield/`), and `ErosionDilationPenalty`.

## High-level flow

```text
Configuration / Parameters
        |
        v
Geometry / Material Representation
        |
        +----------------------+
        |                      |
        v                      v
   FDFD / FDTD            Heat / Lithography
        |                      |
        +----------+-----------+
                   |
                   v
          Fields / Objectives
                   |
                   v
          Gradients / Sensitivities
                   |
                   v
       Optimization / Inverse Design
                   |
                   v
        Fabrication Constraints / DRC
                   |
                   v
             Results / Artifacts
```

The diagram is conceptual. Individual workflows may bypass stages or use additional third-party simulation backends.

## Major areas

### `core/`

Contains reusable scientific and algorithmic components. Important subpackages include:

- `core/fdfd/`: 2D uniform and rectilinear frequency-domain electromagnetic simulation and Pardiso/Torch VJP solvers.
- `core/fdtd/`: 3D JAX FDTD-X simulation wrapper, Chebyshev broadband sources/adjoints, and sparse phasor recording.
- `core/heat/`: JAX-FEM steady-state thermal simulation and thermo-optic coupling.
- `core/nearfield/`: 2D/3D Stratton-Chu and angular-spectrum near-to-far field projection.
- `core/invdes/`: inverse-design orchestration (`InvDesign`), `BaseOptimization`, `N_Ports` devices, parametrizations, objectives, run management, and stochastic variation (`core/invdes/variation/`).
- `core/inv_litho/`: differentiable inverse-lithography models.
- `core/train/`: neural operator models, datasets, PDE/field losses, and trainers.
- `core/visualization/`: Tidy3D 3D permittivity/field exports and cross-section plotting.
- `core/builder.py` and `core/utils.py`: shared builders, SI power/eigenmode quadrature, and morphological penalties; changes here require extra care because of their wide fan-out.

### `data/` and `scripts/`

Contain **MAPS-Data** multi-GPU simulation dataset generators (`data/fdfd/`, `data/fdtd/`) and experiment launch scripts (`scripts/`) for device inverse design and neural operator training.

### `gpu_subpixel/` and `thirdparty/`

- `gpu_subpixel/`: CuPy/CUDA exact geometric subpixel rasterization routines for rectilinear grids.
- `thirdparty/`: bundled `ceviche` primitives (`thirdparty/ceviche/`) and public `fdtdx` compatibility/adjoint extensions (`thirdparty/fdtdx/`).

## Architectural constraints

1. Core simulation code should remain reusable independently of UI code.
2. Physics and numerical conventions must be explicit in documentation and preserved across refactors.
3. Device placement and tensor shape conventions should be inferable from APIs and documented where nontrivial.
4. Optimization code should not silently alter objective definitions, normalization, or gradient semantics.
5. Experimental scripts may be less polished than reusable core modules, but they should not be used as the implicit specification for a public API without verification.

## Safe extension pattern

For a new feature:

1. Identify the lowest-level reusable abstraction that owns the behavior.
2. Add or update focused tests near that abstraction.
3. Expose the feature through higher-level workflows only after the lower-level interface is stable.
4. Document the public interface with a Google-style docstring.
5. Add a documentation example only after the API and expected behavior are stable.
