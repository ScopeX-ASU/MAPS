# Differentiable Figure-of-Merit Layers

This page documents the small differentiable wrapper layers in `core/invdes/models/layers/fom_layer.py` for the `v0.0.2_fdtdx` baseline. These layers are used around simulation objectives and parameterization/projection operations. No Python source is modified by this documentation.

## `SimulatedFoM`

`SimulatedFoM` is an `nn.Module` wrapper around the project's objective/gradient callback.

```python
SimulatedFoM(cal_obj_and_grad_fn, adjoint_mode="fdtd")
```

### Forward contract

```python
forward(
    permittivity_list: list[Tensor],
    resolution: int | None = None,
    custom_source: Tensor | None = None,
) -> Tensor
```

`permittivity_list`
: List of differentiable permittivity tensors corresponding to the design variables/objective callback contract. The layer does not itself reshape or concatenate them.

`resolution`
: Optional simulation/parameterization resolution forwarded unchanged to the objective callback.

`custom_source`
: Optional custom source tensor forwarded to the objective callback when supported.

### Adjoint modes

`adjoint_mode="ceviche"`
: The current implementation explicitly raises `NotImplementedError`. Do not treat this mode as an active supported API.

`adjoint_mode="ceviche_torch"`
: Calls `cal_obj_and_grad_fn(..., need_item="need_value", ...)` directly. The path is intended to be fully differentiable with PyTorch.

`adjoint_mode="fdtdx"`
: Calls the same value-only objective callback convention with the fdtdx backend.

The module records the selected mode in `extra_repr()`.

## `ClipLayer`

```python
ClipLayer(mode)
```

Applies one of three custom autograd clipping operators:

```text
"lower_limit"
"upper_limit"
"both"
```

### Input/output

```text
x: Tensor -> Tensor
```

The shape is preserved.

For `lower_limit`:

```python
forward(x, lower_limit=value)
```

and `upper_limit` must be `None`.

For `upper_limit`:

```python
forward(x, upper_limit=value)
```

and `lower_limit` must be `None`.

For `both`:

```python
forward(x, upper_limit=upper, lower_limit=lower)
```

Both limits are required.

Invalid mode/argument combinations trigger assertions or `ValueError`.

## `BinaryProjectionLayer`

```python
BinaryProjectionLayer(threshold=0.05)
```

Applies the custom `BinaryProjection` autograd function.

```python
forward(x: Tensor, t_bny: float) -> Tensor
```

The tensor shape is preserved. `threshold` is stored as module configuration; the current `forward()` also receives a binary-transition parameter `t_bny` at runtime.

Because the implementation delegates to a custom autograd function, do not substitute a standard `torch.round()` or hard threshold if gradients must remain compatible with the current inverse-design method.

## `HeavisideProjection`

```python
HeavisideProjection(threshold=10)
```

```python
forward(x: Tensor, beta, eta) -> Tensor
```

Internally calls `HeavisideProjectionLayer.apply(x, beta, eta, threshold)` and is documented as using a straight-through-estimator-style custom layer.

The rank/shape of `x` is preserved.

`beta`
: Sharpness/steepness parameter. In the inverse-design stack it is typically linked to the current sharpness schedule.

`eta`
: Projection threshold/offset used by the underlying Heaviside implementation.

The exact mathematical projection function lives in the corresponding utility/autograd layer; this wrapper should not be documented as a generic sigmoid approximation.

## `InsensitivePeriod`

```python
InsensitivePeriod()
```

```python
forward(x: Tensor, i: int)
```

Delegates to `InsensitivePeriodLayer.apply(x, i)`. The tensor shape is preserved.

The integer `i` selects the insensitive-period behavior implemented by the custom autograd layer. This is not a generic indexing operation.

## `heightProjection`

The class name intentionally follows the existing implementation spelling.

```python
heightProjection(
    threshold=10,
    height_max=1.0,
)
```

### Forward contract

```python
forward(
    ridge_height,
    gratings: Tensor,
    sharpness,
    resolution,
) -> Tensor
```

The implementation constructs a 1D height coordinate mask:

```text
[0, height_max]
```

with a sample count based on:

```text
height_max * resolution + 1
```

and applies `heightProjectionLayer` to `ridge_height` and this coordinate mask.

The projected height mask is then broadcast over `gratings` by inserting a singleton dimension:

```text
gratings * height_mask.unsqueeze(1)
```

Therefore the returned tensor shape follows the broadcast-compatible layout of `gratings` and the generated height mask. Before changing this layer, inspect the consuming grating representation to verify which axis corresponds to height.

## Differentiability contract

These layers exist specifically because inverse-design variables must remain differentiable through non-smooth design operations such as clipping, binary projection, Heaviside projection, and fabrication-inspired transforms.

For future AI-generated changes:

- preserve custom autograd functions unless the mathematical gradient rule is intentionally redesigned;
- document whether a layer is exact, smoothed, or uses an STE/custom backward;
- never replace an STE/projection with an ordinary PyTorch operation solely to simplify code;
- preserve tensor rank and broadcasting semantics;
- document units or physical interpretation for scalar control parameters when established upstream.

## Relationship to parametrization

The FOM/projection layers sit downstream of the design-density parametrization. A typical conceptual sequence is:

```text
gradient-bearing design parameters
        |
        v
normalized design density
        |
        +--> symmetry/filter/fabrication transforms
        |
        v
binary/Heaviside/height projection
        |
        v
permittivity/material map
        |
        v
simulation objective
```

In practice, individual projects may apply these components in different orders. The actual configured transform sequence, not this schematic, determines the scientific experiment.
