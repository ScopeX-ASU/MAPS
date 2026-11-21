# Source Docstring Migration Backlog

This page records source files whose scientific contracts are documented but whose source-level docstrings were intentionally not rewritten during the `v0.0.2_fdtdx` documentation migration.

The purpose is to distinguish **documented externally** from **fully migrated at source level**.

## Why this backlog exists

Several MAPS modules are large mixed-responsibility scientific implementations. Reconstructing an entire file solely to insert docstrings creates unnecessary risk of implementation drift. During this migration, those modules were therefore documented through API/architecture contract pages first.

When an exact local editing workflow is available, migrate them incrementally with executable-code equivalence checks.

## Priority A — high-value public scientific APIs

### `core/invdes/invdesign.py`

Source-level targets include:

- `InvDesign` lifecycle;
- optimization-loop configuration;
- results dictionary schema;
- callbacks;
- scheduler interaction;
- checkpoint/export behavior;
- objective sign convention.

Companion contracts already exist under `docs/api/invdesign-*.md`.

### `core/invdes/models/base_optimization.py`

Source-level targets include:

- parameter/epsilon construction;
- normalization;
- material interpolation;
- objective assembly;
- gradient plumbing;
- solver selection.

### `core/invdes/models/objective.py`

Source-level targets include:

- S-parameter objectives;
- flux/power normalization;
- near-to-far objectives;
- tensor/dictionary schemas;
- direction and polarization semantics.

### `core/invdes/models/device_base.py`

This is the highest-risk monolith. Migrate only after establishing an exact source-edit/test workflow.

Docstrings should cover:

- geometry/rasterization;
- port representation;
- slice conversion;
- mode/source construction;
- monitor construction;
- design-region masks;
- optical/thermal grids;
- export/build lifecycle.

Do not combine this docstring migration with structural refactoring.

## Priority B — solver cores

### `core/fdfd/fdfd.py`

Document public solver-facing classes/functions with exact field shape, frequency, polarization, units, and material-grid conventions.

### `core/fdfd/solver.py`

Document sparse system inputs/outputs, direct/iterative backend behavior, complex dtype, batching, and autograd bridge semantics.

### `core/fdfd/near2far.py`

Document monitor dictionaries, source/target coordinate shapes, component-specific return tuples, units, decimation, and differentiability.

### `core/fdtd/fdtd.py`

Document `fdtd3d` and public scene/source/monitor helpers with 3-D grid shape, SI/micrometer conversion points, Yee/colocated representation, and returned phasor tensors.

### `core/fdtd/solver.py`

Document the Torch/JAX adjoint bridge, phasor detector shapes, symmetry unfolding, source-spectrum normalization, gradient grouping, and experimental controls.

## Priority C — shared utility monolith

### `core/utils.py`

Migrate by functional family rather than top-to-bottom:

1. grid metadata/coordinate helpers;
2. resampling/interpolation;
3. Yee colocation/padding;
4. flux/eigenmode integration;
5. symmetry/phasor unfolding;
6. backend conversion;
7. far-field coordinate utilities.

Each migrated function should include realistic tensor-shape examples. Avoid documenting miscellaneous unrelated helpers in the same PR.

## Migration acceptance criteria

For a source-docstring PR touching one of these files:

- no signature changes;
- no executable statement changes;
- no numerical constant changes;
- no import changes unless required solely by documentation tooling and separately justified;
- no comment deletion/rewording unless explicitly reviewed;
- AST/executable equivalence check when feasible;
- existing tests pass;
- documentation link/build checks pass;
- docstring coverage does not regress.

For scientific APIs, manually review:

- shape and axis order;
- dtype/device;
- units;
- dictionary schema;
- equations/normalization;
- gradient behavior;
- representative example correctness.

## Recommended PR granularity

Do not make one giant source migration PR. Use subsystem-focused PRs such as:

```text
docs/invdes-docstrings-runtime
docs/invdes-docstrings-objective
docs/device-docstrings
docs/fdfd-docstrings-core
docs/fdtd-docstrings-core
docs/shared-utils-docstrings-grid
```

The external contract pages should remain useful after source migration because they capture cross-function and cross-subsystem semantics that do not belong in a single callable's docstring.

## Completion definition

A subsystem is considered source-docstring migrated when its public scientific APIs have sufficient local documentation to understand inputs/outputs without reading implementation bodies, while cross-function architecture and workflow details remain linked to the corresponding hand-written documentation pages.
