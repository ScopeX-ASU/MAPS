# Getting Started

These guides are built from verified examples in the `v0.0.2_fdtdx` baseline. They should remain executable/documented representations of real repository workflows, not synthetic tutorials that drift from the code.

## Current canonical example

### Bending inverse design

[First inverse-design run: Bending](first-inverse-design.md)

The Bending example is the current canonical end-to-end example because the tagged baseline contains a complete `configs/bending/default.yml` configuration and the root `train.py` entry point consumes that configuration to construct the device, optimization model, solvers, schedulers, and training loop.

## Planned examples

The next useful additions are:

```text
first-device.md
first-fdfd-simulation.md
first-crossing-inverse-design.md
```

These should only be added after the exact source/configuration path for each example is verified in the target release baseline.
