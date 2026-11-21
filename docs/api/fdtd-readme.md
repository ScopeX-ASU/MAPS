# MAPS FDTD-X Documentation

The MAPS FDTD-X backend provides a 3-D time-domain electromagnetic path using `fdtdx`, with a Torch-facing differentiable interface for inverse design.

## Architecture

```text
MAPS device / design parameters
        |
        v
3-D material / epsilon_r grid
        |
        v
core.fdtd.fdtd.fdtd3d
        |
        v
core.fdtd.solver FDTDXSolveTorch
        |
        +--> FDTD-X JAX simulation
        |
        +--> phasor detectors / monitors
        |
        v
Torch complex fields / objective
        |
        v
autograd / adjoint gradient
```

## Main documentation

- [Scientific conventions](fdtd-scientific-conventions.md) — grid, Yee fields, phasors, units, dtype and tensor-shape rules.
- [Solver and adjoint contract](fdtd-solver-contract.md) — Torch/JAX bridge, field components, source normalization, symmetry, and gradient grouping.
- [Boundaries, sources, and monitors](fdtd-boundaries-sources-monitors.md) — physical-coordinate placement, source schemas, detector slices, and interpolation.
- [Units and normalization](../design/units-and-normalization.md) — physical surface quadrature, modal coefficients, symmetry scaling, and transmission reporting.

## FDTD-X versus FDFD

FDTD-X is a time-domain 3-D solver. FDFD is a frequency-domain sparse linear-system solver. They should not be treated as interchangeable backends with identical tensor contracts.

```text
FDFD:
    A x = b
    sparse complex matrix
    explicit frequency-domain solve

FDTD-X:
    time-domain Maxwell update
    Yee grid
    phasor extraction at requested frequencies
```

The two backends may ultimately expose comparable electromagnetic objectives, but the internal grids, source semantics, field staggering, solver state, and gradient paths differ.

## Documentation rule

For FDTD-X APIs, examples must state the actual tensor rank/order, Yee versus colocated representation, physical units, complex dtype, frequency indexing, and whether phasors are normalized by the source spectrum.
