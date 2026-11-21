# MAPS Testing Strategy

MAPS is scientific research software. The test system must protect physical correctness, numerical stability, tensor/data contracts, and differentiability while still allowing internal refactoring.

## 1. Purpose and non-negotiable rule

Tests are part of the engineering contract, not only debugging utilities.

Before refactoring or materially modifying existing scientific code, establish regression coverage for the behavior being changed. If adequate coverage does not exist, the first change should normally be a test-only baseline PR. Refactoring should follow after that baseline is merged.

Tests should validate scientific and public behavior, not incidental implementation details. A refactor is successful when the tested contracts remain valid even if internal algorithms, class layout, helper functions, or data structures change.

Agents must not weaken, delete, skip, or rewrite a failing test solely to make a change pass. A changed tolerance, reference value, or expected behavior requires an explicit explanation of why the scientific or numerical behavior is intentionally different.

## 2. Test layers

| Layer | Marker | Purpose | Typical execution |
| --- | --- | --- | --- |
| Unit | `unit` | Fast deterministic tests of local behavior and contracts | Every PR |
| Integration | `integration` | Multi-component workflow wiring and serialization | Every PR when feasible |
| Gradient | `gradient` | Autograd/adjoint correctness, gradient shape and direction | Every PR touching differentiable code |
| Numerical | `numerical` | Solver/material/FoM scientific regressions | Selected PRs; full suite nightly/release |
| GPU | `gpu` | CUDA-specific behavior, kernels, and device/dtype contracts | GPU CI and release validation |
| Slow | `slow` | Expensive simulations, long optimization, and large meshes | Nightly/release |

A test may carry multiple markers. For example, a finite-difference adjoint check may be `gradient`, `numerical`, and `slow`.

The repository may retain subsystem-specific legacy test trees when those tests are authoritative or provide valuable coverage. New MAPS-wide tests should follow the repository-level organization described here rather than duplicating existing coverage without a reason.

## 3. Recommended repository structure

```text
tests/
    conftest.py
    unit/
        invdes/
        parametrization/
        objective/
        optimizer/
        fdfd/
        fdtd/
        utils/
    integration/
        invdes/
        fdfd/
        fdtd/
        workflows/
    gradient/
        invdes/
        fdfd/
        fdtd/
    numerical/
        fdfd/
        fdtd/
        reference/
    examples/
        bending/
        crossing/
    fixtures/
```

This is a target architecture. Existing tests should be migrated incrementally rather than moved mechanically in a single refactor.

## 4. Current pytest foundation

The repository-level pytest entry point is now `pytest` with configuration in `pytest.ini` and shared fixtures in `tests/conftest.py`. The fast CI gate installs the minimal test dependencies from `requirements-test.txt` and runs `python -m pytest -m unit` without requiring the full electromagnetic/GPU environment.

New MAPS-wide tests belong under `tests/`. The legacy `unitest/` directory and subsystem-specific suites such as `core/ceviche/tests/` remain valid during migration; do not move or delete them merely to satisfy the new structure. Promote stable checks into the repository-level suite when doing so improves reuse, determinism, and CI coverage.

The standard markers are:

```text
unit integration gradient numerical gpu slow
```

The optional `require_cuda` fixture skips cleanly when PyTorch or CUDA is unavailable. A skipped GPU test is not evidence of GPU compatibility; GPU compatibility requires an executed GPU regression gate.

## 5. What to test

### Public APIs and contracts

Test caller-visible contracts including tensor rank and shape; axis/coordinate ordering; dtype and device behavior; physical units; normalization and sign conventions; dictionary/tuple schemas; parameter defaults and validation errors; documented side effects; and checkpoint serialization.

### Parametrization

Test normalized density, permittivity/material generation, resolution conversion, symmetry, filtering/projection, roughness, extrusion, and parameter lifecycle. Include gradients for differentiable transformations.

### Objectives and FoM

Test objective aggregation, weighting/sign conventions, S-matrix/eigenmode schemas, wavelength/mode/control-state expansion, clipping/projection, and custom-autograd behavior.

### Optimization

Test optimizer initialization, state creation, closure semantics, scheduler behavior, update direction, checkpoint restore, and representative one-to-few-step optimization smoke cases.

### Electromagnetic solvers

Test field shapes, source/monitor schemas, boundary conditions, material maps, normalization, backend selection, and representative physical outputs. Full solver reference cases belong in the numerical layer rather than ordinary unit tests.

## 6. Gradient and adjoint tests

Differentiable code receives explicit gradient coverage. At minimum, tests should establish that the forward value is finite and meaningful, the gradient exists when expected, the gradient has the expected shape/dtype/device/layout, and the gradient direction is numerically consistent with a trusted reference.

For small problems, finite differences are the preferred independent check. For larger problems, use a reviewed gradient fingerprint/reference case or compare against a trusted backend.

Gradient-direction checks should usually be stronger than a single elementwise absolute tolerance. Cosine similarity and scale-aware norms are useful diagnostics, but the acceptance criterion must be chosen from the numerical behavior of the specific problem.

## 7. Numerical regression

Numerical tests should use small, deterministic canonical cases. Keep reference values in dedicated, reviewable files rather than scattering unexplained constants across tests.

A canonical regression case may record the solver/backend and precision, mesh/grid resolution, wavelength/frequency and material inputs, boundary/PML configuration, objective/FoM definition, selected field or monitor statistics, S-parameters or modal quantities, gradient fingerprints, and software/version metadata needed to reproduce the baseline.

Use explicit absolute and relative tolerances appropriate to the quantity. Complex quantities should be compared with a documented complex-number policy rather than silently taking real parts.

Reference values are scientific baselines. Updating them is not a routine maintenance operation.

## 8. Fixtures and determinism

Prefer reusable fixtures for canonical devices, grids, materials, wavelengths, and small solver configurations. Seed pseudo-random generators when randomness is part of the case.

Tests should avoid dependence on wall-clock time, local filesystem layout, machine-specific environment variables, or cached generated artifacts unless those are explicitly part of the contract.

GPU tests should state device/dtype assumptions and skip cleanly when the required accelerator is unavailable. A skipped GPU test is not equivalent to a passing GPU regression gate.

## 9. Test naming and scope

Use names that describe the contract rather than the implementation, for example:

```text
test_bending_fdfd_returns_expected_fwd_trans
test_smatrix_objective_preserves_mode_axis
test_projection_backward_matches_reference_gradient
test_checkpoint_roundtrip_restores_optimizer_state
```

Keep each test focused on one contract. Prefer several small assertions with clear diagnostics over one opaque end-to-end assertion when practical.

## 10. Local validation order

For a normal code change, agents should use this order:

```text
1. Focused unit test(s)
2. Focused integration/gradient test(s), when applicable
3. Relevant numerical regression(s)
4. Broader subsystem suite
5. Full required CI-equivalent suite when practical
```

When a focused test fails, diagnose the contract first. Do not jump directly to changing expectations.

## 11. Refactor baseline rule

A code area without adequate regression coverage is not a good candidate for a broad refactor.

```text
existing debug/validation script
        |
        v
small deterministic test case
        |
        v
explicit scientific assertions
        |
        v
CI regression gate
        |
        v
refactor / optimization / cleanup
```

Existing debug scripts are valuable sources of domain knowledge. Promote stable checks into pytest tests rather than deleting them solely because they are ad hoc.

## 12. Forbidden testing shortcuts

Agents must not delete a failing assertion to unblock a PR, widen numerical tolerances without scientific justification, regenerate golden values from the modified implementation and call them an independent baseline, mock away the numerical behavior that the test is meant to validate, mark a test skipped merely because a new implementation exposes a bug, convert a deterministic regression into a loose smoke test to avoid failures, or claim GPU/numerical coverage when the relevant test did not execute.

A legitimate change to expected numerical behavior should be documented as such and reviewed as a scientific change, not disguised as test maintenance.

## 13. Completion criterion

The test migration is complete when the high-value public and scientific contracts have deterministic coverage, critical numerical/gradient workflows have regression baselines, and CI reliably exercises the appropriate test tiers. Exhaustive line coverage is not the goal; trustworthy contract coverage is.
