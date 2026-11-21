# Generated API Reference

This section is rendered from Python source docstrings using MkDocs and `mkdocstrings`.

- **Generated reference**: signatures, public members, and source docstrings.
- **Scientific API contracts** under `docs/api/`: cross-module tensor schemas, units, equations, normalization, warnings, and workflows.

Do not duplicate every Python signature in hand-written Markdown. Update the source docstring, then let the generated reference render it.

## Initial scope

The first generated pages focus on stable/high-value modules while docstring quality is migrated incrementally:

- [`core.invdes.invdesign`](invdes.md)
- [`core.fdfd.fdfd`](fdfd.md)
- [`core.fdtd.fdtd`](fdtd.md)

See [coverage](coverage.md) for the incremental migration policy.

## Local build

```bash
python -m pip install -r docs/requirements-docs.txt
mkdocs build --strict
```

The full scientific runtime is not required for every documentation page, but selected generated modules must be importable in the build environment.
