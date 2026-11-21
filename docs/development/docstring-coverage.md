# Docstring Coverage

Run from the repository root:

```bash
python tools/check_docstrings.py
```

The default report covers public classes and functions in first-party Python code and excludes known vendored/external trees. Use `--all` to include private helpers.

For a strict local check:

```bash
python tools/check_docstrings.py --check
```

This is an incremental migration tool. CI enforcement should be enabled after the Tier 1 API surface has been documented and deliberate exceptions are identified.

The checker validates the presence of a docstring, not prose quality, Google-style section completeness, or scientific accuracy. Those remain review responsibilities for now.
