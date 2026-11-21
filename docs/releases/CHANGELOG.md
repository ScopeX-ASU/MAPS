# Changelog

This changelog records intentional project-level changes across public MAPS releases.

## v0.0.5 (2026-11-20)

### 3-D FDTD-X Backend & Broadband Adjoint Optimization

- Added the `thirdparty.fdtdx` compatibility and adjoint extension layer on top
  of the public `fdtdx` repository (pinned commit
  `77b4bc523a8e98cb7dd388e99481a0000f71dd4d`; note PyPI `fdtdx==0.6.2` is older
  and incompatible), removing the v0.0.4 dependency on a private internal
  `fdtdx-adjoint` fork.
- Added Chebyshev broadband 3-D eigenmode sources (`ChebyshevModePlaneSource`),
  exact frequency-sampled mode sources (`BroadbandModePlaneSource`), Tidy3D-compatible
  Gaussian-beam plane sources (`GaussianPlaneSourceTidy3d`), and factored Chebyshev
  broadband adjoint current sources (`FactoredChebyshevCurrentSource`).
- Restored unit 3-D optical launch power and physical SI surface-area normalization
  ($\text{m}^2$) across `get_flux_3d`, `get_eigenmode_coefficients_3d`, mode/Gaussian
  sources, cross-axis monitors, and mirror-symmetry unfolding.
- Added opt-in sparse phasor recording (`MAPS_SPARSE_PHASOR=1`) with localized
  design-region, objective-port, and `xy`/`xz`/`yz` visualization-slab detectors.

### Inverse Design, Manufacturing & Near-to-Far Projection

- Added metasurface inverse design (`Metasurface`) and GPU-accelerated 2-D/3-D
  angular-spectrum and Stratton-Chu near-to-far field projection (`core/nearfield/`).
- Added differentiable `ErosionDilationPenalty` (`erosion_dilation_penalty`) with
  multi-plane cross-section evaluation, nonuniform rectilinear resampling, and
  simulation-context / reflective boundary halo padding.
- Added multi-sample fabrication and source variation modeling (`core/invdes/variation/`).
- Added CuPy/CUDA `gpu_subpixel` exact polygon/box/circle subpixel rasterization
  and digital topology/shape examples (`digital_y_branch*`, `mrr_3d`, `metasurface_3d`,
  `waveguide_3d`).
- Added `OptimizationRunManager` (`core/invdes/run_management.py`) and Tidy3D
  field visualization export utilities (`core/visualization/`).

### Testing & Documentation

- Added repository-level `pytest` suite (`tests/`) with `unit`, `integration`,
  `gradient`, `numerical`, `fdtdx`, and `gpu` markers plus GitHub Actions CI workflows.
- Added MkDocs technical documentation (`docs/`) covering architecture, scientific
  conventions, units/normalization, and API contracts.
