# Visual near-field verification

Open these notebooks in order, or run any one independently. Each locates the
repository from its working directory and prints its device and precision.
The default cells render figures without requiring widget interaction.

| Notebook | Visual checks |
| --- | --- |
| [01: Planar far field](01_planar_farfield.ipynb) | Input E/H components, vector angular pattern, direct/separable complex fields and gradients, backend timing and CUDA memory |
| [02: Exact versus far field](02_exact_nearfield_vs_farfield.ipynb) | Observation-plane components, shared-scale intensity, phase and complex errors, distance dependence, exact/far-field and optional Triton costs |
| [03: Closed Huygens box](03_closed_sixbox.ipynb) | Surface geometry, dipole fields, projection versus independent analytical fields, quadrature convergence |
| [04: Tidy3D](04_tidy3d_comparison.ipynb) | Identical-input exact and far-field complex E/H comparisons, spatial errors, CPU/GPU timing with device labels |
| [06: Differentiable focus](06_focal_optimization_gradient.ipynb) | Phase optimization, before/after focal fields, spatial gradients, finite differences, backend field/gradient agreement and profiling |

The shared plotting/reporting functions live in `notebook_utils.py`.
Conversions to NumPy occur only for reporting; differentiable objectives use Torch.

Notebook 07 reproduces the physical zone-plate settings from Flexcompute's
[zone-plate projection example](https://www.flexcompute.com/tidy3d/examples/notebooks/ZonePlateFieldProjection/).
It loads a saved MAPS simulation by default. Its explicit recomputation switch
reruns the simulation locally; no Tidy3D cloud computation is used. See the
[case conventions](../../docs/design/zone-plate-projection.md) for grid, source,
and monitor-origin details.

## Reading the figures

- `|E|²` and `|H|²` are squared field magnitudes with their displayed units. They
  are not power flux. Radiation intensity, when shown, is labeled separately.
- Comparisons use common absolute scales. Relative complex L2 errors preserve
  phase and amplitude; intensity agreement alone does not establish fidelity.
- Weak-field phase is masked. Pointwise vector errors use the global reference
  peak, so a field null does not produce an artificial infinite relative error.
  Error tables distinguish strong samples (at least 1% of peak vector amplitude)
  and weak samples. An unavailable metric is displayed as `N/A`.
- A finite prescribed aperture is a model input. Agreement between two projection
  backends verifies implementation consistency; the dipole and direct FDTD cases
  supply independent physical checks.
- The FDTD reference records an observation **line**. Its projected plane is a
  prediction, without a separately recorded plane reference.

## Reading performance reports

Profiles synchronize CUDA calls and report measured first-observed and median
warm timings. First-observed time includes any lazy setup and may use existing
compiler caches; it is **not an isolated compilation measurement**. Backward
uses a fresh graph and the squared magnitude of all returned E/H fields.

CUDA memory reports include both total peak allocated memory and its increase
above allocations live before each method. They include forward/backward
execution where backward is requested. Allocator-reserved memory and CPU host
memory are not measured. Different notebook fixtures and CPU/GPU devices must
not be compared as equivalent benchmark workloads. These small visual examples

For this development environment use
`python` and GPU visibility
`CUDA_VISIBLE_DEVICES=0`. Saved outputs describe the local run; rerun the cells
for measurements on another machine. All development and outputs remain local.
