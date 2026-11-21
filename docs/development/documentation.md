# Documentation and Docstring Standard

## MAPS scientific API documentation

MAPS uses Google-style Python docstrings as the structural convention, but scientific APIs require substantially more detail than a prose summary. A docstring should act as a compact interface specification that a developer or coding agent can use without reading the implementation.

For important scientific and numerical APIs, document four contracts when applicable:

1. **Interface contract**: what the callable does, its side effects, and externally visible behavior.
2. **Data contract**: dtype, shape, dimension semantics, device, units, valid ranges, container schema, and optionality.
3. **Scientific contract**: equations, coordinate conventions, normalization, boundary conditions, assumptions, differentiability, and references.
4. **Usage contract**: realistic examples showing representative inputs and outputs.

Do not invent information that cannot be established from implementation, callers, tests, examples, or scientific references. If a shape, unit, or convention is uncertain, investigate it before documenting it.

## Google-style sections

Use the standard Google-style sections as appropriate:

- `Args:` for parameters.
- `Returns:` for returned values.
- `Yields:` for generators.
- `Raises:` for interface-relevant exceptions.
- `Attributes:` for important public class state.
- `Notes:` for equations, scientific assumptions, numerical details, performance constraints, or compatibility information.
- `References:` when the implementation follows or adapts a published method.
- `Examples:` for representative executable or near-executable usage.

Not every callable requires every section. Prefer useful information over boilerplate.

## Tensor and array contracts

For tensors and arrays, document the following whenever meaningful:

- shape;
- meaning of every nontrivial dimension;
- dtype or accepted dtype family;
- real versus complex representation;
- CPU/CUDA/device expectations;
- physical units;
- expected value range or normalization;
- broadcasting or batching behavior;
- whether the input is modified in place;
- differentiability/gradient requirements when relevant.

Use symbolic dimension names consistently. Common notation includes:

- `B`: batch size;
- `C`: channel or field-component dimension;
- `H`, `W`: two spatial/grid dimensions when the API uses image-like layout;
- `Nx`, `Ny`, `Nz`: physical simulation-grid dimensions;
- `N`: flattened number of degrees of freedom or samples, defined locally;
- `Nwl`: number of wavelengths/frequencies;
- `P`: number of ports, points, or parameters, defined locally.

Never assume that `H/W` and `x/y` are interchangeable. State the mapping used by the implementation.

Example:

```python
Args:
    field: Complex electric field.
        Shape: ``[B, Nwl, Nx, Ny]``, where ``B`` is batch size,
        ``Nwl`` is the wavelength dimension, and ``Nx``/``Ny`` follow the
        solver grid ordering.
        Dtype: ``torch.complex64`` or ``torch.complex128``.
        Device: CPU or CUDA; must match the associated material tensor.
        Units: V/m unless the calling workflow uses an explicitly documented
        normalized field convention.
```

If different ranks have different semantics, document each accepted form explicitly rather than writing "array-like".

## Multiple input types

When a parameter accepts multiple types, explain what each type means and whether behavior differs.

```python
Args:
    eps_r: Relative permittivity representation.
        If ``torch.Tensor``: differentiable material grid with shape
        ``[Nx, Ny]``.
        If ``numpy.ndarray``: host-side grid converted according to the
        implementation's device/dtype rules.
        Values are dimensionless relative permittivity.
```

Do not imply support for a type merely because an internal operation might happen to accept it.

## Dictionary and configuration schemas

Do not describe a structured dictionary only as "configuration dictionary." Document keys that form part of the interface, including nested content when important.

```python
Args:
    sim_cfg: Simulation configuration with keys:
        ``"resolution"`` (int): Spatial resolution used by the backend.
        ``"PML"`` (Sequence[float]): PML thickness for each documented axis.
        ``"optical_grid"`` (dict): Grid-generation configuration containing
            ``"mode"`` and backend-specific grid options.
```

For large schemas, document the stable required keys in the docstring and link to a dedicated schema/configuration document for the complete format.

## Returns

For every important returned tensor, array, tuple, or dictionary, document its structure at the same level of detail as the input.

For tuples, name each element. For dictionaries, document meaningful keys. For scalar tensors, distinguish shape ``[]`` from Python scalars.

```python
Returns:
    A tuple ``(Hx, Hy, Ez)`` where each element is a complex tensor with
    shape ``[Nx, Ny]``. The tensors use the solver's field normalization and
    reside on the solver device.
```

## Scientific and mathematical operations

For scientific operations, a mathematical definition can be more precise than prose. Include equations when they define the API or remove ambiguity.

Example:

```text
Notes:
    The normalized field error is

        L = ||E_pred - E_ref||_2 / (||E_ref||_2 + epsilon).

    The norm is evaluated over the spatial dimensions described above.
```

When implementing a published algorithm, include a concise citation or stable reference. State material deviations from the cited formulation.

Equations must match the implementation. Do not add a canonical textbook equation when the code uses a modified normalization, discretization, sign convention, or approximation.

## Examples are part of the contract

Important public APIs should normally include at least one realistic example. Examples are especially important when:

- tensor shapes are non-obvious;
- multiple input types or ranks are accepted;
- dictionaries have required structure;
- complex tensors are represented in a special way;
- CPU/CUDA behavior matters;
- a solver requires several objects to be constructed together;
- the return type is nested;
- scientific conventions are difficult to infer from the signature.

Examples should show representative shapes and, where inexpensive, demonstrate output shape/type:

```python
Examples:
    >>> eps_r = torch.ones((128, 96), dtype=torch.float32, device="cuda")
    >>> source = torch.zeros((128, 96), dtype=torch.complex64, device="cuda")
    >>> source[16, 48] = 1.0 + 0.0j
    >>> Hx, Hy, Ez = solver.solve(source)
    >>> Ez.shape
    torch.Size([128, 96])
    >>> Ez.dtype
    torch.complex64
```

Only use an example after checking the actual constructor/signature and behavior. Do not create plausible-looking examples from assumptions.

## Classes

Class docstrings should document the scientific role and lifecycle, not just restate the class name. Include important constructor inputs and public attributes. When a class owns persistent tensors or simulation state, document their shape/device/unit conventions.

## Private scientific helpers

A private helper should still receive a detailed docstring when it encodes a non-obvious scientific operation, shape transformation, discretization, normalization, interpolation, coordinate conversion, or gradient rule. Public/private visibility does not reduce the need to preserve scientific knowledge.

## Citations and references

Use references when code implements a specific paper, numerical method, solver formulation, or externally defined convention. A useful reference includes enough information to locate the source: authors or project, title/method, venue/year when applicable, and DOI/URL when stable.

If an equation or algorithm is derived internally, explain the derivation in `Notes:` when short; otherwise link to a document under `docs/design/` or scientific notes in the repository.

## What not to do

Avoid:

- generic docstrings such as "Compute output";
- repeating the signature without documenting semantics;
- listing a tensor shape without defining dimensions;
- saying "dict" without documenting stable keys/content;
- saying "tensor" without documenting real/complex semantics when relevant;
- guessing units, dtype, device, shapes, equations, or citations;
- documenting implementation details that are not part of the contract;
- promising performance characteristics that are not measured;
- copying stale documentation from older APIs without checking callers and implementation.

## Documentation depth

Use judgment rather than forcing every function to have a long docstring.

- Trivial utility: concise behavior plus Args/Returns when needed.
- Data transformation: input/output shapes, dtype, semantics, and example.
- Numerical/scientific primitive: equation, conventions, shapes, units, and reference when applicable.
- Solver/model public API: full interface/data/scientific/usage contract.
- Large configuration API: stable schema plus link to detailed configuration documentation.

## API documentation strategy

The source-of-truth API description lives in Python docstrings. Generated API documentation should consume those docstrings rather than duplicate them. Hand-written pages under `docs/` should explain architecture, workflows, cross-cutting conventions, and scientific background.

## Documentation change checklist

Before opening a change that affects or documents an important API, verify:

- Parameter names match the signature.
- Accepted input types match actual behavior.
- Tensor/array shapes are stated and each dimension is defined.
- Dtype and real/complex representation are documented where relevant.
- CPU/CUDA/device behavior is documented where relevant.
- Physical units and normalization are documented where established.
- Dictionary/tuple/container structure is documented.
- Returned shapes/types/units are documented.
- Equations match the implemented computation.
- Scientific references are included when they materially explain the method.
- At least one realistic usage example is provided for important public APIs.
- Examples match actual imports, signatures, and behavior.
- Uncertain scientific facts are investigated rather than guessed.
