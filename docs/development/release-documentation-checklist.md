# Documentation Release-Readiness Checklist

Use this checklist when preparing a MAPS release branch or tag. It complements the continuous PR checks in `documentation-maintenance.md` and focuses on scientific/API synchronization rather than packaging mechanics alone.

## 1. Version and branch identity

Confirm that the documentation being published corresponds to the intended release branch/tag.

Check:

- package/version metadata;
- release branch/tag name;
- README version statements and install commands;
- generated API source revision;
- backend-specific compatibility notes.

Do not copy version numbers into many independent pages. Prefer a small number of authoritative version surfaces and link to them.

## 2. Public API synchronization

For each subsystem changed since the previous release, review the public callable surface:

```text
InvDes
FDFD
FDTD-X
device/geometry
shared grid/field utilities
```

For changed public APIs, verify that documentation states:

- argument meaning and accepted types;
- tensor/array shape;
- dtype and complex/real semantics;
- device/backend constraints;
- units and coordinate convention;
- dictionary/configuration schema when applicable;
- exact return type/shape;
- important exceptions/preconditions;
- a representative usage example for nontrivial APIs.

## 3. Scientific-contract synchronization

Review changes that can produce physically plausible but incorrect results if documentation drifts:

- spatial-axis order;
- center versus boundary coordinates;
- uniform versus rectilinear grids;
- Yee-staggered versus colocated fields;
- component ordering;
- 2-D polarization convention;
- wavelength/frequency units;
- source normalization;
- flux/modal-overlap normalization;
- PML/Bloch/symmetry conventions;
- adjoint/autograd graph boundaries;
- design-region masks and material interpolation.

These require scientific review, not only link/build validation.

## 4. Solver-specific validation

### FDFD

Validate at least one representative workflow covering the documented sparse-system/field contract. When relevant, compare optional backend behavior against the reference path.

### FDTD-X

Validate at least one representative 3-D workflow covering source construction, phasor extraction, Yee-to-colocated conversion, objective evaluation, and gradient propagation.

Expensive simulations do not need to run on every documentation PR, but they should be part of release validation when the corresponding subsystem changed.

## 5. Inverse-design validation

For a representative device such as Bending or Crossing, confirm that the documented flow still matches the implementation:

```text
configuration
  -> device/model construction
  -> parametrization
  -> material grid
  -> solver
  -> objective/FoM
  -> backward/adjoint
  -> optimizer update
  -> results/checkpoint/export
```

Record any experimental flags or backend-specific limitations that materially change this flow.

## 6. Example synchronization

For each getting-started example:

- verify import paths;
- verify configuration keys;
- verify file paths;
- verify commands;
- verify expected output/artifact names;
- verify tensor shapes shown in prose;
- remove references to deleted or renamed APIs.

Specialized research examples under `core/invdes/examples/` may require GPU/solver resources and can be validated selectively.

## 7. Documentation automation

Required mechanical checks:

```bash
python tools/check_docs_links.py .
python tools/docstring_coverage.py
python tools/check_docs_drift.py
```

Use the repository's current script names/options as authoritative if they evolve.

The internal-link check is blocking. Coverage/drift reports should be inspected even when configured as non-blocking.

## 8. Generated API build

Build the MkDocs/mkdocstrings site in an environment that can import the selected MAPS modules.

Verify:

- no unresolved API references;
- Google-style docstrings render correctly;
- equations and code blocks render correctly;
- generated pages do not import optional heavy backends unexpectedly;
- navigation contains the current subsystem contract pages.

## 9. README hierarchy

Check the documentation hierarchy rather than duplicating content:

```text
repository README
    -> subsystem README
        -> getting-started workflow
        -> scientific API contracts
        -> generated API
        -> development/maintenance guidance
```

The repository README should remain an overview. Detailed InvDes/FDFD/FDTD-X contracts belong in their subsystem/API documentation.

## 10. Release notes

Release notes should call out user-visible changes such as:

- new/changed public APIs;
- changed tensor/config schemas;
- solver/backend changes;
- numerical convention changes;
- new experimental features;
- deprecated behavior;
- documentation/example additions that expose a previously undocumented workflow.

Do not describe internal refactoring as a user-facing feature unless it changes observable behavior or maintainability requirements relevant to developers.

## 11. Refactor readiness

Before refactoring a large compatibility module such as `device_base.py` or `core/utils.py`, ensure the current behavior is represented by:

1. scientific/API contract documentation;
2. representative tests/examples;
3. explicit tensor/schema conventions;
4. a compatibility plan for public call sites.

The documentation added for `v0.0.2_fdtdx` is intended to serve as the behavioral specification for such future refactors.

## 12. Release sign-off

A release documentation sign-off should answer four questions:

1. **Mechanical:** do links/build/generated references pass?
2. **API:** do public signatures, schemas, and examples match the code?
3. **Scientific:** do units, shapes, equations, normalization, and field conventions match the implementation?
4. **Workflow:** can a user follow the documented representative paths with the release branch?

Only the first question is fully automatable. The other three require code-aware and, for scientific conventions, domain-aware review.
