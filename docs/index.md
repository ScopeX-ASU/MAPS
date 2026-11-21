# MAPS Documentation

MAPS is a research framework for differentiable electromagnetic simulation, photonic device modeling, and inverse design. This documentation describes the scientific and software contracts of the `v0.0.2_fdtdx` baseline.

## Start here

New contributors should read these pages in order:

1. [System architecture](architecture/overview.md) — major layers and ownership boundaries.
2. [Configuration and build flow](architecture/configuration-and-build-flow.md) — how configuration becomes devices, simulations, objectives, and optimization runs.
3. [Shared grid and field data](architecture/shared-grid-and-field-data.md) — coordinates, native/export grids, tensor layouts, and cross-subsystem data movement.
4. [Device architecture](architecture/device-base.md) — geometry, rasterization, ports, monitors, sources, material maps, and the current `device_base.py` responsibility boundary.
5. [Scientific documentation standard](development/documentation.md) — required shapes, dtypes, units, equations, schemas, citations, and examples.

## Scientific concepts

### Coordinates, grids, and fields

MAPS treats physical coordinates as authoritative. Tensor indices are derived from the grid on which a quantity lives. A device can simultaneously own native optical, export optical, FDTD-X native, and thermal grids.

Read:

- [Shared grid and field data](architecture/shared-grid-and-field-data.md)
- [Grid and field utilities](api/grid-field-utils.md)
- [Numerical conventions](design/numerical-conventions.md)
- [Tensor shapes](design/tensor-shapes.md)
- [Units and normalization](design/units-and-normalization.md)

### Device representation

The device layer converts geometry/configuration into rasterized material maps, design-region masks, ports, monitor slices, source profiles, and backend-specific grid metadata.

Read:

- [Device architecture](architecture/device-base.md)
- [Device API contracts](api/device-base.md)
- [Base optimization and material maps](api/base-optimization.md)

### Simulation backends

MAPS currently exposes frequency-domain and time-domain electromagnetic paths.

**FDFD**

- [FDFD field API](api/fdfd.md)
- [FDFD sparse solver](api/fdfd-solver.md)

**FDTD / FDTD-X**

- [FDTD field API](api/fdtd.md)
- [FDTD differentiable solver and adjoint](api/fdtd-solver.md)
- [Differentiable 3D near-to-far projection](api/nearfield.md)
- [Projection conventions and monitor audit](design/nearfield-projection.md)
- [Independent projection verification](design/nearfield-verification.md)

### Inverse design

The inverse-design stack maps learnable variables to normalized density, material maps, simulated fields, objectives, and optimizer updates.

Read:

- [Design-region parametrization](api/parametrization.md)
- [Base optimization and material maps](api/base-optimization.md)
- [Objective contracts](api/objective.md)
- [FoM/autograd layer](api/fom-layer.md)
- [Inverse-design workflow](api/invdes.md)
- [Optimization run management](api/optimization-run-management.md)
- [Optimizer and scheduler builder](api/invdes-builder.md)

### Configuration and factories

- [Top-level builder](api/builder.md)
- [Configuration and build flow](architecture/configuration-and-build-flow.md)

The top-level training/ML factories and the physics inverse-design factories are related but distinct systems. Do not assume optimizer, scheduler, model, or closure semantics transfer between them.

## API reference map

| Area | Primary documentation |
| --- | --- |
| Device, geometry, ports, monitors, sources | [Device API](api/device-base.md) |
| Grid metadata, resampling, Yee fields | [Grid/field utilities](api/grid-field-utils.md) |
| FDFD simulation | [FDFD](api/fdfd.md) |
| FDFD linear solver | [FDFD solver](api/fdfd-solver.md) |
| FDTD-X simulation | [FDTD](api/fdtd.md) |
| FDTD-X differentiation/adjoint | [FDTD solver](api/fdtd-solver.md) |
| Design parameterization | [Parametrization](api/parametrization.md) |
| Material-map construction | [Base optimization](api/base-optimization.md) |
| Objectives | [Objective](api/objective.md) |
| Differentiable FoM | [FoM layer](api/fom-layer.md) |
| Optimization loop | [InvDesign](api/invdes.md) |
| Optimizer/scheduler factory | [InvDesign builder](api/invdes-builder.md) |
| Top-level ML/training factories | [Builder](api/builder.md) |

## Development documentation

- [Coding style](development/coding-style.md)
- [Scientific documentation and docstrings](development/documentation.md)
- [Docstring coverage](development/docstring-coverage.md)
- [Testing](development/testing.md)

The root `AGENTS.md` is the canonical instruction map for AI coding agents. `CLAUDE.md` is a compatibility entry point and should not become an independent source of repository policy.

## Documentation principles

The documentation follows several project-wide rules:

1. **Scientific behavior is part of the API.** Shapes, units, normalization, field ordering, coordinate systems, solver conventions, and transform order are not implementation trivia.
2. **Physical coordinates outrank array indices.** Any operation that crosses grids must carry coordinate metadata or make the mapping explicit.
3. **Do not guess contracts.** Documentation should be derived from implementation, callers, tests/examples, and verified scientific references.
4. **Examples must reflect real APIs.** A plausible-looking simulation example is harmful if its configuration keys, shapes, or units are wrong.
5. **Documentation precedes major refactoring.** In particular, the current `device_base.py` documentation is intended to serve as a behavioral specification for a later decomposition.

## Current documentation scope

This documentation prioritizes the scientific core and first-party interfaces used by differentiable simulation and inverse design. It does not yet attempt exhaustive coverage of every training model, dataset, DRC utility, lithography helper, or historical research experiment.

The next documentation stage is to add small, verified end-to-end examples built from the actual baseline APIs.
