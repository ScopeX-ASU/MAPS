# Continuous Documentation Maintenance

MAPS is an actively changing research codebase. Documentation must therefore be maintained as part of normal engineering work rather than treated as a one-time release artifact.

## Core principle

The implementation, tests, configuration, and verified examples remain the source of truth. Documentation explains and exposes those contracts; it does not silently redefine them.

The project uses **docs-as-code with selective automation**:

```text
source changes
    |
    +--> docstrings / API contracts
    +--> architecture/docs when behavior crosses boundaries
    +--> verified examples when user workflow changes
    |
    v
PR documentation checks
    |
    +--> broken internal links: blocking
    +--> docstring coverage: visible trend, initially non-blocking
    +--> example execution: targeted, not every PR
```

The objective is to catch mechanical drift automatically while keeping scientific meaning under human review.

## Ownership by change type

### Public function/class signature or behavior

Update in the same PR:

1. the Google-style source docstring;
2. the relevant API page under `docs/api/`;
3. a usage example when the interface or shape/schema is difficult to infer.

### Tensor shape, dtype, device, units, normalization, or coordinate convention

Update the source docstring and the corresponding design/API contract. These are scientific contracts, not optional prose.

### Algorithm/equation/solver/adjoint behavior

Update the docstring or API page with the implemented equation/convention and add or update a scientific reference when appropriate.

### Configuration schema

Update the consuming code and configuration/API documentation together. For large schemas, keep a concise stable contract in the docstring and detailed fields in `docs/`.

### Architecture or cross-subsystem data flow

Update the architecture page. `device_base.py` is intentionally documented as a behavioral specification before future refactoring; architectural changes should update the specification before or together with the refactor.

### Example workflow

If a command, import path, configuration key, or output contract changes, update the corresponding `docs/getting-started/` example in the same PR. Examples should be derived from real repository scripts/configurations.

### Dependency/install behavior

Update `docs/getting-started/installation.md` when install commands, environment profiles, supported runtime versions, or verification steps change. The executable installer remains authoritative.

## Single-source-of-truth rules

Source docstrings own concrete callable signatures and local data contracts. Hand-written API pages own cross-function schemas, equations, workflows, and scientific context. Generated API pages render source docstrings instead of duplicating them.

Architecture pages describe responsibility boundaries and data flow. Getting-started examples document verified workflows; they should not become independent synthetic implementations.

## Automation policy

1. **Internal Markdown links are blocking.**
2. **Docstring coverage is reported but initially non-blocking.** Coverage should improve by subsystem.
3. **Expensive examples are targeted.** GPU/FDFD/FDTD workflows should not run on every documentation PR.
4. **Generated API documentation should be built in an environment that can reliably import the selected modules.** Heavy optional backends may require a dedicated docs runner.

## Release synchronization

Before a release tag:

- verify version metadata and release notes;
- verify getting-started commands against the release branch/tag;
- run selected end-to-end smoke examples;
- confirm README/subsystem README navigation;
- record experimental and backend-specific caveats.

A release should not silently ship a documentation state describing a different API generation.
