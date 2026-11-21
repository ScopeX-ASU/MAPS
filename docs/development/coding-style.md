# Coding Style

The project should evolve toward a small, predictable style surface that makes numerical code easy to review and safe for AI-assisted editing.

## General principles

- Prefer readable code over clever one-liners.
- Keep functions focused on one responsibility.
- Prefer explicit intermediate names for physical quantities and tensor transformations.
- Avoid unrelated formatting or refactoring in scientific changes.
- Match surrounding conventions when working in an older subsystem unless the task explicitly modernizes that subsystem.

## Naming

Use names that expose physical meaning or tensor role where possible. For example, prefer names such as `wavelength`, `eps_r`, `source`, `field`, `objective`, `device`, `batch_size`, or `grid_shape` over ambiguous abbreviations.

## NumPy / PyTorch / JAX code

When changing numerical code:

- preserve dtype semantics;
- preserve device semantics;
- make broadcasting assumptions clear;
- avoid accidental host-device transfers;
- avoid unnecessary materialization of large tensors;
- document non-obvious dimension order.

## Comments

Comments should explain why, numerical context, or a non-obvious invariant. Do not use comments to narrate obvious syntax.

## Compatibility

Do not make broad API or dependency changes as part of documentation work. Separate behavioral changes from cleanup so that reviewers can reason about regressions.
