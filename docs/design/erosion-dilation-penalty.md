# Erosion-Dilation Manufacturing Penalty

## Overview and Scientific Intent

Photonic topology optimization frequently produces sub-micron geometric artifacts, including isolated sub-resolution islands, narrow gaps, and sharp re-entrant corners that cannot be reliably resolved by photolithography or dry etching. The `ErosionDilationPenalty` provides a differentiable surrogate for minimum feature size and minimum radius of curvature constraints.

By applying morphological operations (erosion and dilation) to the material occupancy field, the penalty measures geometric variance under structural shrinking and expanding:
1. **Closing** ($\mathcal{C} = \mathcal{E} \circ \mathcal{D}$): Dilates then erodes the occupancy, bridging gaps and filling voids smaller than the characteristic length scale.
2. **Opening** ($\mathcal{O} = \mathcal{D} \circ \mathcal{E}$): Erodes then dilates the occupancy, eliminating isolated islands and thin protrusions smaller than the characteristic length scale.

The magnitude of the difference between closed and opened geometries, $\mathcal{C}(\rho) - \mathcal{O}(\rho)$, serves as a penalty on all features narrower than the specified length scale $L$. Fully manufacturable layouts whose features exceed $L$ undergo negligible deformation under closing and opening, resulting in near-zero penalty.

---

## Mathematical Formulation

### 1. Conic Convolution Kernel

The spatial neighborhood is determined by a 2D normalized conic kernel matching Tidy3D's `ConicFilter`. Given physical length scale $L$ and grid step $\Delta l = (\Delta l_x, \Delta l_y)$, the kernel radius in pixels along each axis is:
$$R_x = \left\lceil \frac{L}{\Delta l_x} \right\rceil, \quad R_y = \left\lceil \frac{L}{\Delta l_y} \right\rceil$$

On a grid of size $(2 R_x + 1) \times (2 R_y + 1)$, normalized coordinates $g_x \in [-1, 1]$ and $g_y \in [-1, 1]$ define the radial distance:
$$d(x, y) = \sqrt{g_x^2 + g_y^2}$$
$$K_{\text{raw}}(x, y) = \max(0, 1 - d(x, y))$$
$$K(x, y) = \frac{K_{\text{raw}}(x, y)}{\sum_{u, v} K_{\text{raw}}(u, v)}$$

### 2. Differentiable Tanh Soft-Thresholding Projection

To map filtered continuous densities to near-binary values differentiably, MAPS reuses `tanh_projection` from `core.invdes.models.layers.parametrization.utils`, parameterized by steepness $\beta$ and midpoint $\eta$:
$$P_{\beta, \eta}(x) = \frac{\tanh(\beta \eta) + \tanh(\beta (x - \eta))}{\tanh(\beta \eta) + \tanh(\beta (1 - \eta))}$$

To ensure exact range preservation in $[0, 1]$ with an identity straight-through backward pass, `tanh_projection(..., clip=True)` applies `ApplyBothLimit(torch.autograd.Function)`.

### 3. Two-Stage Erosion and Dilation

For a continuous occupancy field $\rho \in [0, 1]$:
- Dilation threshold: $\eta_d = \delta_\eta$ (default $0.01$)
- Erosion threshold: $\eta_e = 1 - \delta_\eta$ (default $0.99$)

Morphological steps are computed via shared valid 2D convolutions:
1. **Stage 1 Filtering**:
   $$F_1 = K * \rho_{\text{halo}}$$
   $$\rho_{\text{eroded}} = P_{\beta, \eta_e}(F_1), \quad \rho_{\text{dilated}} = P_{\beta, \eta_d}(F_1)$$
2. **Stage 2 Filtering**:
   $$F_{2, \text{open}} = K * \rho_{\text{eroded}}, \quad F_{2, \text{close}} = K * \rho_{\text{dilated}}$$
   $$\rho_{\text{opened}} = P_{\beta, \eta_d}(F_{2, \text{open}}), \quad \rho_{\text{closed}} = P_{\beta, \eta_e}(F_{2, \text{close}})$$

### 4. Differentiable RMS Penalty Norm

On the target evaluation domain of size $N_{\text{eval}} = H_{\text{eval}} \times W_{\text{eval}}$, the difference field is:
$$\Delta(x, y) = \rho_{\text{closed}}(x, y) - \rho_{\text{opened}}(x, y)$$

The root-mean-square (RMS) penalty is defined with safe gradient stabilization to prevent $\text{NaN}$ gradients when $\Delta = 0$:
$$S = \sum_{x, y} |\Delta(x, y)|^2$$
$$\mathcal{P} = \begin{cases} \sqrt{\frac{\max(S, 10^{-30})}{N_{\text{eval}}}}, & S > 0 \\ 0, & S = 0 \end{cases}$$

---

## Multi-Plane Evaluation

In 3D photonic topology optimization, `BaseParametrization._extrude_density_tensor` extrudes the 2D design region across the vertical $Z$ extent $[z_0, z_1)$ (already incorporating any sidewall tapering in the 3D permittivity volume). To enforce minimum feature sizes on user-selected cross-sections, `ErosionDilationPenalty` applies the exact same 2D erosion-dilation penalty (`length_scale`) to each requested plane in `planes` (defaulting to `("middle",)`):
- **`"bottom"`**: $z = z_0$
- **`"middle"`**: $z = \lfloor (z_0 + z_1 - 1) / 2 \rfloor$
- **`"top"`**: $z = z_1 - 1$

### Multi-Plane Aggregation

Per-plane penalties $\mathcal{P}_p$ are aggregated using user-specified weights $w_p$:
- **`"mean"`**: $\mathcal{P}_{\text{total}} = \frac{\sum_p w_p \mathcal{P}_p}{\sum_p w_p}$
- **`"sum"`**: $\mathcal{P}_{\text{total}} = \sum_p w_p \mathcal{P}_p$
- **`"max"`**: $\mathcal{P}_{\text{total}} = \max_p (w_p \mathcal{P}_p)$
- **`"min"`**: $\mathcal{P}_{\text{total}} = \min_p (w_p \mathcal{P}_p)$

---

## Simulation-Context Boundary Conditions & Reflection Fallback

### 1. The Boundary Artifact Problem

In traditional topology optimization, filtering the design region with artificial boundary conditions (zero padding, replicate padding, or isolated reflection at the optimization boundary) creates artificial discontinuities at interfaces where waveguides enter the design region. This causes the optimizer to incorrectly taper, neck down, or erode the port interfaces.

### 2. Real Simulation Context Extraction

MAPS resolves this by embedding the design region in its actual physical context:
1. When `fixed_epsilon_map` is provided, surrounding fixed structures (waveguides, substrates, cladding) are extracted directly into the convolution halo.
2. A continuous waveguide entering the design region has identical geometry across the optimization boundary, eliminating artificial boundary penalties.

### 3. Asymmetric and Corner Reflection Fallback

When the required halo around the evaluation region extends past the outer physical boundaries of the simulation domain (e.g. design region touching or near a simulation domain edge), MAPS applies reflective padding *only to the missing exterior margin*.

For requested index range $[s, e)$ on a domain of size $N$:
- $p_{\text{left}} = \max(0, -s)$
- $p_{\text{right}} = \max(0, e - N)$
- If $p_{\text{left}} == 0$ and $p_{\text{right}} == 0$, the slice is extracted directly without reflection.
- For $N \ge 2$, indices are generated via `np.pad(np.arange(N), (p_left, p_right), mode="reflect")`.
- 2D corner reflections are strictly separable, matching full 2D `np.pad(..., mode="reflect")`.

### 4. Halo Sizing

To support two stages of valid 2D convolution without loss of spatial support, the required halo radius on each side of the evaluation domain is:
$$H_x = 2 R_x + H_{\text{extra}}$$
$$H_y = 2 R_y + H_{\text{extra}}$$
where:
- $2 R$: two valid convolution passes (each reducing support by $R$).
- $H_{\text{extra}}$: optional margin for halo independence verification.

### 5. Nonuniform Rectilinear Grids & Internal Design-Region Resolution

In MAPS (`BaseParametrization` and FDTDX autogrid workflows), two grid mismatches commonly arise:
1. **Internal Design-Region Resolution**: A design region `rho` (e.g., parameterized at `phi_resolution` or `rho_resolution`) has a uniform internal grid of shape $(N_x^{\text{dr}}, N_y^{\text{dr}})$ that differs from the native slice shape $(x_1 - x_0, y_1 - y_0)$ in `fixed_epsilon_map`.
2. **Nonuniform Rectilinear Simulation Grid**: The surrounding simulation domain (`fixed_epsilon_map` / `device.epsilon_map` with `device.optical_grid_metadata`) uses nonuniform rectilinear cell spacing (`is_uniform == False`).

Because spatial morphological filtering and conic convolution require a constant physical pixel spacing $(\Delta l_x, \Delta l_y)$ so that $(R_x, R_y)$ corresponds to a spatially invariant physical length scale $L$, the design region and surrounding context cannot be directly concatenated by array index when their grids differ. Instead, `erosion_dilation_penalty` and `ErosionDilationPenalty` establish a unified physical coordinate grid:
- The uniform evaluation grid spacing $(\Delta l_x, \Delta l_y)$ inside the physical design region $[x_{\min}^{\text{dr}}, x_{\max}^{\text{dr}}] \times [y_{\min}^{\text{dr}}, y_{\max}^{\text{dr}}]$ is determined from the internal `rho` shape, `design_region_param_cfg` (`phi_resolution` / `rho_resolution`), or `pixel_size`.
- This uniform coordinate grid is extended across the physical simulation domain $[x_{\min}^{\text{sim}}, x_{\max}^{\text{sim}}] \times [y_{\min}^{\text{sim}}, y_{\max}^{\text{sim}}]$.
- The fixed background context (`fixed_epsilon_map`) within the halo crop inside the physical simulation domain is resampled from its native rectilinear coordinates onto the uniform halo coordinates via `resample_rectilinear_tensor(..., mode="cartesian", interpolation="linear")` and cached on GPU.
- If `rho` is provided on its internal uniform grid, it is placed directly into the uniform design-region slice without loss of internal resolution; if `epsilon_map` is provided on the native nonuniform grid, its design-region window is differentiably resampled onto the uniform coordinates via `resample_rectilinear_tensor`.
- Reflective fallback (`_build_reflect_axis_indices`) is applied on the uniform domain grid strictly outside the physical simulation domain boundaries.

---

## Single-GPU Acceleration and Context Caching

### 1. Invariant Cached State

During iterative gradient descent (hundreds or thousands of iterations), the simulation domain geometry, design region placement, grid spacing, kernel radii, and reflection index maps remain invariant.

`ErosionDilationPenalty` caches the fixed context on GPU:
- Reflection index tensors `idx_x`, `idx_y` (`torch.long` on GPU).
- Simulation crop bounding slices `crop_x`, `crop_y`.
- Normalized fixed simulation background occupancy `fixed_crop_occ` (detached on GPU).
- Local design-region insertion slices within the cropped context.

### 2. Forward Execution Performance

On subsequent forward passes:
1. `comp_crop = fixed_crop_occ.clone()`
2. `comp_crop[loc_dr] = rho_dr` (differentiable in-place assignment)
3. `padded = comp_crop[idx_x, :][:, idx_y]` (differentiable index gather)
4. Sidewall morphology and 2D ED convolution.

This guarantees:
- **Zero CPU-to-GPU data transfers** per optimization iteration.
- **Zero background re-normalization** per optimization iteration.
- **Exact parity** between cached and uncached execution.

---

## API Reference and Examples

### 1. Standalone Functional Interface

```python
import torch
from core.utils import erosion_dilation_penalty

# Evaluate on a standalone 2D tensor
rho = torch.rand(64, 64, requires_grad=True)
penalty = erosion_dilation_penalty(
    rho=rho,
    length_scale=0.15,  # 150 nm minimum feature size
    pixel_size=0.02,     # 20 nm grid step
    beta=100.0,
    eta=0.5,
    delta_eta=0.01,
)
penalty.backward()
```

### 2. PyTorch Module with GPU Caching

```python
import torch
from core.invdes.models.layers.objective import ErosionDilationPenalty

penalty_module = ErosionDilationPenalty(
    length_scale=0.12,
    planes=("middle",),
    aggregation="mean",
    pixel_size=0.02,
    use_cache=True,
)

x = torch.rand(80, 80, device="cuda", requires_grad=True)
loss = penalty_module(x)
loss.backward()
print(f"Cache misses: {penalty_module.cache_misses}, hits: {penalty_module.cache_hits}")
```

### 3. Optimization Framework Integration (`ObjectiveFunc`)

```python
obj_cfgs = {
    "fwd_trans": {
        "type": "eigenmode",
        "weight": 1.0,
        "in_slice_name": "in_port",
        "out_slice_name": "out_port",
        "in_mode": "Ez0",
        "out_modes": ("Ez0",),
        "wl": 1.55,
    },
    "mfg_penalty": {
        "type": "erosion_dilation_penalty",
        "weight": -0.5,
        "temp": [300],
        "enable_adjoint": False,
        "length_scale": 0.15,
        "planes": ("middle",),
        "aggregation": "mean",
    },
    "_fusion_func": lambda bd: sum(v["weight"] * v["value"] for v in bd.values()),
}

objective_function.add_objective(obj_cfgs)
```
