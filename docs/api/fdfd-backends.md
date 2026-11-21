# FDFD Sparse Solver Backends

MAPS-FDFD can route the same sparse Maxwell system through multiple numerical backends. Backend selection is an implementation detail; the scientific contract is the same linear system and field ordering.

## Backend roles

```text
SciPy sparse        -> portable/reference fallback
MKL / Pardiso      -> CPU sparse direct solve
PyDISO             -> alternative CPU sparse direct integration
cuDSS              -> GPU sparse direct solve
CuPy                -> GPU sparse matrix interoperability
```

Optional integrations may not be installed in every environment. Documentation and examples should identify optional dependencies explicitly rather than assuming a backend is available.

## Backend invariants

For the same assembled operator and source, backend changes should preserve, within solver tolerance:

- matrix dimensions `[N, N]`;
- source ordering `[N]`;
- complex field ordering `[N]`;
- residual quality `||A @ x - b||`;
- physical source/field normalization;
- differentiability of the material-to-objective path when used by inverse design.

A backend optimization should therefore be validated against a trusted reference solve before it replaces a default path.

## Direct solver state

Some direct backends expose a reusable factorization/solver object. The `clear` option in the current solver interface controls whether that external solver state is released immediately.

When documenting functions that can return solver state, the docstring should distinguish:

```text
solution vector      : complex [N]
solver/factor state  : backend-specific object
```

and explain lifetime/ownership when `clear=False`.

## Precision

The `double` control on the MKL/Pardiso path selects the backend's numerical precision behavior. MAPS FDFD commonly constructs sparse operators with complex128 values; a backend-specific lower-precision path must state where precision is reduced and what validation tolerance is expected.

## GPU sparse interoperability

`coo_torch2cupy` accepts a coalesced Torch COO sparse matrix and converts values/indices into CuPy sparse storage. The conversion does not alter the logical matrix shape. Callers should not rely on implicit densification.

## Documentation requirement

For a backend-specific API, document:

- optional dependency;
- CPU/GPU execution location;
- supported matrix type (complex symmetric vs nonsymmetric, if applicable);
- precision behavior;
- whether factorization state can be reused;
- return type and shape;
- fallback behavior;
- expected numerical equivalence and validation strategy.
