# Independent near-field projection verification

Audit date: 2026-10-20. Planning and independent verification used GPT-6.1 Sol;
implementation used GPT-6 Luna, with separate review before acceleration.

## Environment

Executed with `python`,
`CUDA_VISIBLE_DEVICES=0`, outside the filesystem sandbox for GPU access.
PyTorch 2.9.1+cu128; NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition;
Tidy3D 2.12.1. The baseline has no Tidy3D, Triton, JAX, or FDTDX import dependency.

## Physical formulation review

Reviewed outward equivalent-current signs, outgoing phase, exact longitudinal
terms, dimensional quadrature, impedance, both complex E/H fields and the
complex inner-product adjoint. Independent review corrected the draft test's
extra 4-pi factor and reversed interior/exterior expectation before accepting
its results. An outward box reproduces the exterior field and cancels the
interior field; it does not reconstruct an enclosed physical source.

MAPS affine-field regression independently confirms the existing nonexact
observation operator targets the Ez Yee location. The adapter deliberately
requires verified coordinates, complete tangential coverage, and explicit
common-time/colocated/unfolded declarations; no extra timing correction or
source-spectrum division is applied.

## Independent analytical convergence

`python -m benchmarks.nearfield.verify_dipoles --device cuda` evaluates a
separately written retarded electric-dipole formula and its electromagnetic
dual, with independently constructed midpoint box quadrature. Fixed
frequency is 800 MHz, box half-width 0.35 m, and exterior point
(0.9, 0.12, -0.1) m. Source E/H phase and amplitude are retained throughout.
The interior point (0.03, -0.02, 0.06) m is source-free. The far-field comparison
uses 100 approximately uniform directions and the exact dipole radiation
coefficient, including complex polarization. No values come from a modified
projector baseline.

| Nodes per face edge | Worst exact E/H relative error | Far E relative error | Worst interior residual / exterior field norm |
| ---: | ---: | ---: | ---: |
| 12 | 1.6977% | 0.9069% | 0.05684% |
| 24 | 0.41095% | 0.21785% | 0.01308% |
| 48 | 0.10193% | 0.05394% | 0.00321% |

Electric and magnetic-dual errors exchange E/H columns and otherwise agree.
The roughly fourfold reduction under doubled face resolution supports
second-order midpoint-quadrature convergence. Persistent numerical tests also
check electric and physical magnetic moments, 16-to-32 refinement, and total
radiated power within 1% of the analytic formula. A rectangular-aperture test
checks the full complex vector sinc diffraction pattern, source translation
phase, and impedance against its analytic integral within 0.01% relative L2.

## Cross-implementation and gradient gates

Eight identical-input Tidy3D tests compare exact/far Cartesian complex E and H,
positive/negative surface normals, and relative permittivities 1 and 2.25.
Resampling is disabled, the origin is fixed, source quadrature is matched, and
meter/micrometer conversion is explicit. Their relative vector L2 threshold is
2e-5. This is projector cross-validation, not an independent time-domain run.

Eight actual GPU tests compare complex64 fields and backward gradients against
CPU complex128 for focal intensity, directional emission, complex-field
matching and polarization objectives, across exact/far modes and two
frequencies. All satisfy relative L2 below 2e-5. Complex128 finite differences
and explicit complex inner-product adjoint identities provide separate
backward checks. CUDA monitor extraction checks gradient scatter into the
original dense fields.

## Validation status and limits

Production MAPS full-wave validation remains a separate gate. Its current
FDTD backend wraps an external solver, and historical coordinate descriptions
are inconsistent. Neither analytical fields nor agreement with Tidy3D's local
projector is reported as a MAPS full-wave result. The completed independent Yee
reference, accelerated backend results, and benchmark measurements are recorded
below. Complete executed-test totals follow the final combined regression.

## Broader repository gate

`python -m pytest -m unit -q` stopped with nine collection errors in existing
FDTD, inverse-design and FDFD tests (`pyutils.general.TimerCtx`,
`pyutils.torch_train`, `thirdparty.ceviche`, and optimization imports).
Repeating the command with every new near-field test directory excluded
reproduced the same nine errors. A clean-process `from core.fdtd import fdtd3d`
succeeds, and the affected installed modules exist, so these results should
not be described as proof of absent packages; existing collection-time module
stubbing/import interactions need separate investigation. No existing test,
dependency version, or numerical tolerance was changed to bypass this gate.

## Independent time-domain evidence

The separate PyTorch Yee pulse solver executes Maxwell curl updates on a
homogeneous periodic grid and records near-surface and direct observation
phasors using their actual staggered sampling times. Runs end before periodic
images can reach the observation points. It has no production-solver import.
See [method and refinement table](nearfield-yee-validation.md).

At 40 cells per wavelength, projected-versus-direct complex vector errors at
1.0/1.5 wavelengths are E: 0.75%/0.57%, H: 0.84%/0.93%. Direct Yee fields also
agree with the continuum current-dipole solution within 1%. Persistent tests
require improvement from 20 to 40 cells per wavelength, the 1% bound at both
points, and wavelength scaling. This supports the physical projection formula
independently of the two local projection implementations; it is not reported
as a production MAPS run.

## Measured GPU performance

One discarded warmup and three measured repeats; one frequency, float32
geometry and complex64 fields. Direct Torch uses source/target chunks 256/128
and checkpointed backward. Triton uses source/target tiles 16/32 with Torch
backward recomputation in 256/128 chunks. Peak memory is allocated GPU memory,
not reserved allocator capacity. Input fields in these timing cases are
synthetic; timings do not establish physical fidelity.

| Method | Source / target count | Warm forward | Warm forward + backward | Peak GPU allocation |
| --- | --- | ---: | ---: | ---: |
| Direct Torch exact | 1,024 / 100 | 2.002 ms | 8.450 ms | 7.384 MB |
| Triton exact | 1,024 / 100 | 0.599 ms | 9.423 ms | 7.390 MB |
| Direct Torch exact | 16,384 / 1,000 | 203.207 ms | 1026.245 ms | 12.400 MB |
| Triton exact | 16,384 / 1,000 | 2.894 ms | 1009.617 ms | 12.449 MB |
| Separable far field | 512² / 512² | 1.985 ms | 3.133 ms | 143.669 MB |
| FFT propagation | 512² / 512² | 1.034 ms | 1.652 ms | 100.941 MB |

First exact Torch calls were 45.44 ms and 249.12 ms; corresponding first Triton
calls in fresh private empty caches were 1.614 s and 1.590 s. First-call time also
includes ordinary setup/warmup and is not a measurement of isolated compiler
CPU time. Triton improves the measured warm forward calculation; its hybrid
backward does not demonstrate an end-to-end speedup. Separable far-field and
FFT propagation are specialized mathematical operations, not interchangeable
exact finite-distance backends. The 512² cases were not compared with a dense
68.7-billion-pair direct calculation.

For the 1,024/100 case, relative complex64-versus-identical-input-complex128
errors were E/H: 8.91e-7/8.87e-7 (Torch), 9.48e-7/9.58e-7 (Triton).
Machine-local JSON records are kept outside the repository; the benchmark CLIs
reproduce their workloads. No automatic choice should extrapolate these
measurements to unmeasured shapes or hardware.


## Calibrated dispatch and dependency isolation

`project(..., backend="auto", calibration_records=...)` compares supplied
validated exact-shape records, including measured first-call and warm costs.
It accounts for expected reuse count and backward work, rejects hardware,
software, precision, or configuration mismatches, and uses Torch when no
measurement applies. It does not profile or extrapolate automatically. See
[dispatch contract](nearfield-dispatch.md). Benchmarks and their precision
references support performance choices, not physical field validity.

A fresh-process CPU projection was also executed while imports of FDTDX, JAX,
Tidy3D, and Triton were deliberately blocked. The public native API imported
and returned finite fields. External solvers and optimized kernels therefore
remain optional for the native projection path.

## Final executed checks

The combined command below completed with **103 passed, zero skipped**, including
actual CUDA adapter/kernel/gradient checks, the optional Tidy3D comparisons,
and the slow 40-cell-per-wavelength Yee refinement test:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  -m pytest tests/unit/nearfield tests/gradient/nearfield \
  tests/numerical/nearfield tests/integration/nearfield -q
```

Black, Ruff's complete F rules, and `git diff --check` passed. Isort was not
installed in the specified environment. All six notebooks executed successfully;
the saved notebooks contain no execution outputs. The actual final-record AUTO
check selected Triton for the 16,384-source/1,000-target, ten-call forward
workload. Its E/H agreed with direct Torch at relative L2
1.14e-6/1.14e-6. Final cold-cache records correctly keep the small workload on
Torch at 1, 10, and 1,000 expected calls. This demonstrates why warm speed alone
is insufficient evidence for automatic selection.

Production MAPS full-wave recording, cross-solver Tidy3D time-domain simulation,
and scattering/total-field subtraction remain unverified. The native module
assumes a homogeneous lossless isotropic background. Triton currently accepts
complex64 CUDA inputs and first-order gradients; its backward uses Torch.
No higher-order Triton, lossy/layered Green function, or universal backend cost
model is claimed.
