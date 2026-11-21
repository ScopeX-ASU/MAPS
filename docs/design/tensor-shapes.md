# Tensor and Array Shapes

Shape mistakes are among the highest-risk failure modes in scientific Python code. MAPS documentation should make dimension order explicit wherever a tensor or array has a nontrivial contract.

## Documentation rule

For public numerical APIs, document important shape conventions in the docstring. Prefer a concise description such as:

```text
field: Complex tensor with shape [batch, wavelength, height, width].
```

rather than forcing the reader to infer layout from indexing code.

## What to record

For each major subsystem, progressively document:

- batch dimensions;
- wavelength/frequency dimensions;
- spatial dimensions;
- field/component dimensions;
- channel or port dimensions;
- parameter/optimization dimensions;
- whether dimensions are optional or squeezed;
- device placement and dtype expectations.

## AI editing rule

Before changing tensor indexing, reshaping, concatenation, or broadcasting, trace the producing and consuming interfaces. Do not rely on variable names alone.

A reshape that preserves the element count can still be scientifically wrong if it changes dimension semantics.

## Examples

Subsystem-specific examples should be added after an implementation audit. Until then, examples must be labeled with their actual source and not presented as universal MAPS conventions.

## FDTD-X phasor contract

The differentiable FDTD-X boundary uses a complex combined field tensor:

```text
[Nfreq, 6, Nx, Ny, Nz]
```

The component axis is ordered `(Ex, Ey, Ez, Hx, Hy, Hz)`. `Nx`, `Ny`, and
`Nz` are the full native simulation-grid cell counts, including when FDTD-X ran
on a symmetry-reduced domain.

With `MAPS_SPARSE_PHASOR=1`, individual FDTD-X detector states are internal
localized tensors:

```text
[Nfreq, Ncomponent_local, nx, ny, nz]
```

MAPS unfolds and scatters those states into the same dense combined tensor before
returning from the custom autograd function. PyTorch backward receives a dense
cotangent with the same `[Nfreq, 6, Nx, Ny, Nz]` layout. Sparse recording must
therefore not introduce a detector axis, sparse tensor container, reduced
symmetry length, or local `(nx, ny, nz)` shape at the objective boundary.
