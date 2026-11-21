# Documentation Navigation Map

Use [`index.md`](index.md) as the primary documentation home. This page records the intended information architecture so future documentation additions remain organized.

## Architecture

Architecture pages explain subsystem ownership, data flow, lifecycle, and future refactoring boundaries.

- [`architecture/overview.md`](architecture/overview.md) — overall MAPS stack.
- [`architecture/configuration-and-build-flow.md`](architecture/configuration-and-build-flow.md) — configuration-driven construction.
- [`architecture/shared-grid-and-field-data.md`](architecture/shared-grid-and-field-data.md) — grid/field movement across subsystems.
- [`architecture/device-base.md`](architecture/device-base.md) — responsibilities currently concentrated in `device_base.py`.

## API and scientific contracts

API pages document concrete inputs, outputs, tensor layouts, dictionary schemas, units, equations, backend behavior, and examples.

### Device and shared data

- [`api/device-base.md`](api/device-base.md)
- [`api/grid-field-utils.md`](api/grid-field-utils.md)
- [`api/builder.md`](api/builder.md)

### Electromagnetic simulation

- [`api/fdfd.md`](api/fdfd.md)
- [`api/fdfd-solver.md`](api/fdfd-solver.md)
- [`api/fdtd.md`](api/fdtd.md)
- [`api/fdtd-solver.md`](api/fdtd-solver.md)
- [`api/nearfield.md`](api/nearfield.md)

### Inverse design

- [`api/parametrization.md`](api/parametrization.md)
- [`api/base-optimization.md`](api/base-optimization.md)
- [`api/objective.md`](api/objective.md)
- [`api/fom-layer.md`](api/fom-layer.md)
- [`api/invdes.md`](api/invdes.md)
- [`api/invdes-builder.md`](api/invdes-builder.md)

## Design conventions

Design pages define conventions that apply across multiple modules.

- [`design/numerical-conventions.md`](design/numerical-conventions.md)
- [`design/tensor-shapes.md`](design/tensor-shapes.md)
- [`design/units-and-normalization.md`](design/units-and-normalization.md)
- [`design/nearfield-projection.md`](design/nearfield-projection.md)
- [`design/nearfield-verification.md`](design/nearfield-verification.md)
- [`design/nearfield-acceleration.md`](design/nearfield-acceleration.md)
- [`design/nearfield-dispatch.md`](design/nearfield-dispatch.md)
- [`design/nearfield-yee-validation.md`](design/nearfield-yee-validation.md)

## Development

- [`development/documentation.md`](development/documentation.md) — MAPS scientific docstring/documentation standard.
- [`development/docstring-coverage.md`](development/docstring-coverage.md) — coverage policy and checker.
- [`development/coding-style.md`](development/coding-style.md) — coding conventions.
- [`development/testing.md`](development/testing.md) — testing expectations.

## Releases

Release/changelog documents live under `releases/` and should not become substitutes for API or architecture documentation.

## Planned next section: getting started

The next documentation stage will introduce verified, end-to-end examples. Once those examples are validated against the baseline, the intended structure is:

```text
docs/getting-started/
├── installation.md
├── first-device.md
├── first-fdfd-simulation.md
└── first-inverse-design.md
```

Do not populate these pages with hypothetical code. Each example must be traced to an actual working baseline script/API and must document shapes, units, configuration schemas, and expected outputs.
