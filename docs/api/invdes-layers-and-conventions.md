# InvDes Layer Contracts and Scientific Conventions

This page summarizes the smaller reusable layers beneath `BaseOptimization`. The large device and parametrization implementations are intentionally documented at the architecture level rather than duplicated here.

## `fom_layer.py`

The figure-of-merit layer is the bridge from a list of permittivity tensors to a differentiable scalar objective.

`SimulatedFoM.forward(permittivity_list, resolution=None, custom_source=None)` forwards the material tensors to the configured backend. In the supported Torch/JAX paths, the returned scalar remains connected to the parameter graph.

The supported runtime selectors are:

```text
ceviche         -> intentionally unsupported; raises NotImplementedError
ceviche_torch   -> fully differentiable Torch path
fdtdx           -> FDTD-X objective/adjoint bridge
```

### Projection and constraint layers

`BinaryProjectionLayer`, `HeavisideProjection`, `ClipLayer`, `InsensitivePeriod`, and `heightProjection` wrap custom autograd operations used to impose fabrication-oriented or numerical constraints while retaining gradient flow.

These layers should not be interpreted as interchangeable threshold functions. Their parameters have different meanings:

```text
binary projection threshold  -> BinaryProjection custom operator
(beta, eta)                  -> Heaviside projection
upper/lower limits           -> clipping constraints
period index                 -> insensitive-period transform
ridge_height / sharpness     -> grating-height projection
```

## `diff_bdry_layer.py`

`DiffBdry` wraps the custom `differentiable_boundary` autograd operator. Its `axis_tensor` is a physical-coordinate tensor; `w` and `temp` control the smooth boundary transition and are differentiated through the custom operator.

`SmoothRidge` evaluates a logistic ridge mask over three named regions: `left`, `middle`, and `right`. `center`, `width`, and `sharpness` use the same coordinate convention as `total_length`.

`multi_diff_bdry` applies the same idea to an ordered sequence of segments. Its `segment_widths` and `segment_types` must be aligned, with segment type `0` selecting the air-style expression and `1` selecting the media-style expression in the existing implementation.

## Device-layer auxiliary modules

Small device-specific layer implementations under `core/invdes/models/layers/` generally fall into four roles:

1. **device geometry wrappers** — construct named photonic geometries and their high-resolution/design-region variants;
2. **objective adapters** — convert simulated fields, port profiles, or far-field solutions into objective values;
3. **fabrication transforms** — apply blur, roughness, etch, projection, symmetry, or morphology operations;
4. **material/thermal helpers** — provide material-property maps and coupled thermal state handling.

When documenting one of these modules, the required contract is the physical meaning of its tensor axes, units, coordinate grid, and whether a tensor is native-grid, exported-grid, or design-grid data.

## Scientific conventions for layer APIs

A layer docstring should state, where applicable:

- tensor shape and the meaning of each dimension;
- dtype, especially for complex electromagnetic fields;
- physical coordinate system and units;
- whether inputs are normalized densities, refractive indices, permittivity, conductivity, temperature, or derived fields;
- whether the operation is differentiable and which custom autograd path is used;
- parameter ranges or required ordering;
- interpolation/resampling behavior when shapes do not match;
- an example showing a representative input/output contract.

For example, `eps_map` alone is not sufficient documentation. A useful contract distinguishes a 2-D simulation map `[Nx, Ny]` in relative permittivity from a 3-D FDTD-X map `[Nx, Ny, Nz]`, and from a normalized design-region density with a separate `rho_resolution`.

## Large-layer policy

`device_base.py`, `objective.py`, `base_optimization.py`, and the major parametrization implementations are intentionally handled by architecture/API documents first. These modules span many scientific contracts and are too large to reconstruct safely through the restricted remote editing interface.

Their source docstrings should be migrated only when an exact source-edit workflow is available. This is a safety constraint, not a documentation-completeness waiver.
