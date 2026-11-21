# Numerical Regression Policy

Numerical regression tests protect scientific behavior across implementation changes, backend changes, and refactors. They are distinct from ordinary unit tests because floating-point differences, discretization effects, solver tolerances, and hardware paths must be interpreted explicitly.

## Reference-case principles

Use a small, deterministic set of canonical cases that exercise the most important scientific workflows. Prefer cases that are:

- small enough to run reproducibly;
- physically representative;
- sensitive to the contract being protected;
- independent enough that a common implementation bug is unlikely to reproduce the same wrong result in every reference.

Reference cases should cover representative FDFD/FDTD-X workflows, inverse-design objectives, and adjoint/gradient behavior as the suite matures.

## What a reference should record

A reference case should identify, as applicable:

- solver/backend and precision;
- grid or mesh resolution;
- wavelength/frequency and source configuration;
- material parameters;
- boundary/PML configuration;
- objective/FoM definition;
- selected field or monitor statistics;
- S-parameters or modal quantities;
- gradient norm and direction diagnostics;
- software/version metadata needed to reproduce the baseline.

Do not hide scientifically meaningful parameters in an opaque fixture.

## Comparison policy

Use the comparison rule that matches the quantity:

- exact equality only for symbolic/discrete values that are truly deterministic;
- absolute tolerance when values are near zero or have a meaningful absolute scale;
- relative tolerance when scale varies materially;
- combined absolute/relative tolerance for general floating-point quantities;
- magnitude/phase-aware comparison for complex values when phase is scientifically relevant;
- shape/dtype/device checks separately from numerical value checks.

The comparison must be documented when the appropriate tolerance is not obvious.

## Gradient regression

Forward agreement alone is insufficient for differentiable scientific code. Gradient regressions should check shape/layout and a scale-aware value or direction metric.

For small cases, use finite differences or another independently implemented derivative check. For larger cases, use a reviewed gradient fingerprint and at least one independent sanity check.

A gradient tolerance must account for finite-difference step size, solver tolerances, discretization, floating-point precision, and backend differences. Do not choose a tolerance solely because it makes the current implementation pass.

## Golden-reference governance

Golden values are reviewed scientific baselines.

A reference update requires:

1. identification of the changed implementation or dependency;
2. evidence that the new result is expected;
3. comparison against an independent calculation, previous release, analytical result, or other trusted source when practical;
4. an explanation in the PR describing which values changed and why;
5. updates to the relevant documentation when the scientific contract changed.

Never regenerate references from a changed implementation and accept them without independent validation.

## Backend and hardware differences

CPU and CUDA paths may differ slightly while preserving the scientific contract. Backend-specific tolerances should be explicit rather than hidden behind a globally loose threshold.

When a solver supports multiple precision modes, preserve separate references or comparison policies when precision changes materially affect expected error.

A test marked `gpu` must report whether a GPU run actually occurred. Skipping because CUDA is unavailable is a skipped test, not evidence of GPU compatibility.

## Refactor policy

A pure refactor should preserve reference observables within existing tolerances. If a refactor changes a scientific observable beyond the approved tolerance, investigate whether the implementation changed behavior or the old reference was invalid.

Do not silently bless a changed baseline as a refactor result.

## Release baselines

Before a release, run the selected release regression set and record the reference revision/configuration used to produce it. A release baseline should be reproducible from the release branch/tag without relying on uncommitted local state.
