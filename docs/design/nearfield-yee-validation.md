# Independent Yee reference for nearfield projection

`benchmarks/nearfield/yee_reference.py` contains a small native PyTorch time-domain Maxwell solver used as an independent validation reference for nearfield projection. It is not the MAPS production solver and does not call FDTDX or Tidy3D.

The reference uses a homogeneous, periodic cubic Yee grid, a soft x-directed electric-current pulse, and the staggered locations

| Component | Offset in cell widths |
| --- | --- |
| `Ex`, `Ey`, `Ez` | `(1/2,0,0)`, `(0,1/2,0)`, `(0,0,1/2)` |
| `Hx`, `Hy`, `Hz` | `(0,1/2,1/2)`, `(1/2,0,1/2)`, `(1/2,1/2,0)` |

It advances H with forward curls and E with backward curls. It extracts E at integer times and H at half times, samples staggered components onto common monitor locations by trilinear interpolation, then applies the same DFT phase to both fields at the requested carrier frequency. The default pulse is a Gaussian-modulated carrier centered at 1.5 cycles with a 0.4-cycle standard deviation. Phasors on the six-face box and at direct observation points are divided by the same source-current spectrum times cell volume.

The default domain is a periodic cube eight wavelengths wide. The box monitor has a half-width of 0.5 wavelength, and direct observations default to one wavelength along +y. A run stops at six cycles. The nearest periodic image path is therefore at least seven wavelengths for the default observation, and the runner rejects observation distances whose periodic image path would be shorter than the simulated interval. `wavelength_m`, `monitor_half_width_wavelengths`, and `observation_distances_wavelengths` can be changed explicitly.

## CUDA convergence sample

The following data were measured with CUDA, a unit free-space wavelength, an eight-wavelength periodic domain, a six-cycle run, a 0.5-wavelength box half-width, and direct observations at 1.0 and 1.5 wavelengths. “Direct vs analytic” compares the time-domain phasor at each point with the continuum electric-current dipole fields. “Projected vs direct” compares projection from the sampled box with the direct time-domain observation. Errors are complex vector relative norms.

| Cells per wavelength | Direct vs analytic E, 1.0/1.5 λ | Direct vs analytic H, 1.0/1.5 λ | Projected vs direct E, 1.0/1.5 λ | Projected vs direct H, 1.0/1.5 λ |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 5.44% / 9.91% | 5.76% / 9.92% | 1.95% / 6.41% | 6.41% / 8.93% |
| 15 | 2.32% / 4.32% | 2.50% / 4.25% | 1.36% / 4.59% | 3.05% / 3.34% |
| 20 | 1.33% / 2.38% | 1.39% / 2.43% | 0.95% / 1.59% | 1.89% / 2.40% |
| 30 | 0.61% / 1.06% | 0.61% / 1.08% | 0.80% / 0.79% | 1.11% / 1.28% |
| 40 | 0.39% / 0.59% | 0.35% / 0.60% | 0.75% / 0.57% | 0.84% / 0.93% |

The direct Yee fields converge toward the continuum dipole fields with refinement. The surface projection agrees with direct Yee observations to below 1% in E and H at both distances at 40 cells per wavelength. At 20 cells per wavelength, the comparison remains between roughly 0.95% and 2.40%, depending on field and distance. These figures describe this reference configuration; they do not claim a universal production-solver accuracy.

Run a configurable sweep from the repository root with:

```bash
python -m benchmarks.nearfield.yee_reference \
  --resolutions 10 15 20 30 40 \
  --observation-distances 1.0 1.5 \
  --monitor-half-width 0.5 \
  --domain-wavelengths 8 \
  --run-cycles 6 \
  --wavelength-m 1.0 \
  --device cuda
```

The output contains raw complex E/H phasors and grid/time-step metadata. It can be consumed by a notebook or another comparison script without importing any MAPS solver implementation.
