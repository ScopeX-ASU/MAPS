# FDFD Derivative Operators and PML Reference

`core.fdfd.derivatives` contains two closely related numerical layers: autograd-compatible finite-difference curls used by time-domain utilities and sparse derivative-matrix construction used by FDFD.

## Sparse Kronecker conventions

` sparse_kron` operates on coalesced Torch sparse tensors and constructs the Kronecker-product matrix without densifying the inputs.

If:

```text
A.shape = [m, n]
B.shape = [p, q]
```

the result has shape:

```text
[m * p, n * q]
```

The operation preserves the sparse COO representation and uses the device of the input tensor for index construction.

## `compute_derivative_matrices`

For a 2-D FDFD grid with `shape=(Nx, Ny)`, this helper returns four sparse matrices:

```python
Dxf, Dxb, Dyf, Dyb = compute_derivative_matrices(...)
```

with shape:

```text
[Nx * Ny, Nx * Ny]
```

The matrices represent forward/backward finite differences in the x/y directions and include the configured PML scaling. Optional `bloch_x` and `bloch_y` phase factors affect the periodic boundary entries before PML scaling is applied.

## Difference operators

The implementation constructs the 1-D operators first and then lifts them to the 2-D grid with a sparse Kronecker product:

$$
D_x = D_{x,1D} \otimes I_y,
\qquad
D_y = I_x \otimes D_{y,1D}.
$$

The physical derivative includes the inverse grid spacing:

$$
\partial_x \approx \frac{1}{\Delta x} D_x,
\qquad
\partial_y \approx \frac{1}{\Delta y} D_y.
$$

The current API assumes a scalar `dL` for the uniform-grid construction path.

## Bloch phase

`make_Dxf`, `make_Dxb`, `make_Dyf`, and `make_Dyb` insert complex boundary phasors at the wrapped edge.

The forward/backward pair uses conjugate boundary phases consistently with the discrete periodic/Bloch convention. A caller should provide the phase parameter in the normalization expected by the surrounding FDFD formulation rather than assuming the argument is a raw SI wavevector.

## PML scaling

`create_S_matrices` builds four one-dimensional PML scaling vectors and expands them to the 2-D simulation grid:

```text
Sx_f, Sx_b, Sy_f, Sy_b
```

The vectors are flattened in the same ordering as the derivative matrices. PML-scaled derivatives are formed by elementwise row scaling of the sparse operators.

`npml` is the number of PML cells per axis, not a physical width.

## Curl helpers

`curl_E` and `curl_H` use `torch.roll` and therefore remain compatible with PyTorch autograd. They consume three component tensors and a physical cell spacing `dL`.

The component tensors are expected to share the same simulation-grid alignment used by the FDTD utility path. They are not interchangeable with the sparse FDFD derivative matrices: the curl helpers operate directly on field tensors, while the derivative constructors operate on flattened linear-system operators.

## Documentation requirements

For derivative-related APIs, docstrings should state:

- whether the input is a field tensor or sparse operator;
- tensor/matrix shapes and flattening order;
- x/y axis meaning;
- physical units of `dL`;
- complex dtype requirements;
- Bloch/PML behavior;
- whether the operation preserves autograd;
- the corresponding finite-difference or Maxwell equation.
