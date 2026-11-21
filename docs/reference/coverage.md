# Generated API and Docstring Coverage

MAPS is migrating a large research codebase incrementally. Coverage is therefore a **reporting metric**, not a temporary reason to add generic boilerplate docstrings.

## Current policy

- Public callable docstrings are reported by subsystem.
- Generated API pages are enabled first for stable/high-value modules.
- Missing documentation is addressed in focused source-docstring PRs.
- Coverage should increase monotonically for mature subsystems.
- Scientific detail such as tensor semantics, units, equations, and references must be correct before coverage is treated as complete.

Run:

```bash
python tools/docstring_coverage_by_subsystem.py
```

The report intentionally uses the Python standard library and AST parsing, so it does not import MAPS or require CUDA/JAX/MKL.

## Completion criteria for a subsystem

A subsystem is considered sufficiently documented when important public APIs have:

- precise input/output contracts;
- shape and dimension semantics;
- dtype/device information where relevant;
- units and normalization conventions where relevant;
- dictionary/container schemas;
- equations and references for non-obvious scientific operations;
- realistic examples for important workflows;
- generated API coverage for the stable public surface.

The numerical percentage alone is not the acceptance criterion.
