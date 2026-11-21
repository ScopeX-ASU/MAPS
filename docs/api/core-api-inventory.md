# Core API Inventory

The first API documentation pass targets reusable scientific interfaces, not every research script.

| Area | Priority | Documentation target |
| --- | --- | --- |
| `core/fdfd/` | P0 | solver classes, `solve` entry points, derivative/sparse-solver interfaces |
| `core/fdtd/` | P0 | simulation setup, execution, field/result interfaces |
| `core/invdes/` | P0 | inverse-design entry points, optimization models, reusable layers |
| `core/drc/` | P1 | fabrication-rule and penalty interfaces |
| `core/train/` | P1 | dataset/model/optimizer factories |
| `core/utils.py` | P1 | broadly reused numerical and device helpers |
| `core/builder.py` | P1 | high fan-out factory/build helpers |
| `app/` | P2 | user-facing service and application entry points |
| `unitest/` | P3 | tests: module docstrings where useful; no exhaustive API requirement |
| `core/ceviche/` | Exclude | vendored/third-party implementation; document MAPS wrappers instead |

## Definition of done for a Tier 1 API

- Google-style docstring is present and accurate.
- Parameter names match the function signature.
- Units and tensor/array shapes are documented when non-obvious.
- Return values and side effects are explicit.
- Interface-level exceptions are documented.
- Important numerical or physical conventions are called out in `Notes:`.
- At least one caller or example has been checked after documentation changes.

## First source pass

Start with `core/fdfd/`, then `core/fdtd/`, then `core/invdes`. Keep each subsystem in a separate changeset so scientific review remains tractable.
