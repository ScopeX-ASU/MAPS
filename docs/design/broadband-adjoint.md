# Broadband adjoint current construction

## Reference contract and existing data flow

The oracle is the sum of independent per-frequency MAPS adjoint gradients.
Each run uses that frequency's exact objective cotangent, native source
conversion, and the existing material-gradient kernel. This work changes
broadband source construction and grouping; it does not replace the gradient
formula or the single-frequency adjoint implementation.

The forward bridge returns complex source-normalized E/H phasors. Torch
evaluates arbitrary objectives and supplies their complex cotangents to the
custom backward callback. Before creating currents, MAPS applies the existing
symmetry fold, detector/colocation interpolation transpose, and magnetic
half-step pullback as applicable. The current conversion is
`J_e = -i conj(cotangent_E)` and `J_m = +i conj(cotangent_H)`. These are native
Yee-grid currents, including the monitor's quadrature and normalization through
the objective's VJP. They are not raw eigenmode fields to which the forward
mode source's Yee travel delay should be added again.

Previously the grouped path chose the highest-norm current sample as a single
spatial template, projected every sample onto it, injected a central Gaussian
pulse, divided the fields by its source spectrum, and multiplied each frequency
by its projection coefficient. The approximation discarded frequency-dependent
spatial current components perpendicular to that template.

## Tidy3D public implementation inspected

The installed Tidy3D 2.11.2 public implementation creates a reversed-direction
`ModeSource` for each nonzero modal VJP. Source amplitude and phase encode the
complex VJP (including the frequency-dependent source prefactor). Its
`SimulationData` groups sources by spatial identity after replacing source time
with a common time dependence, then chooses frequency or port grouping using
the smaller group count. A same-port broadband source carries a unit temporal
amplitude and the recorded frequency fields receive the original complex
source amplitudes through `post_norm`. Physical profile variation is the
responsibility of the underlying broadband source. The inspected modal source
factory does not explicitly increase `num_freqs` above the `ModeSource` default
of one; MAPS's new profile expansion is an additional implementation choice,
not evidence that Tidy3D automatically increases modal interpolation order.
The inspected path does
not implement PCA, a conditioned multitone temporal inversion, or a
timestep-based optimization of grouping. Field-monitor sources normally retain
per-frequency grouping. MAPS therefore implements its own source-fit guards.

Public sources:

- [SimulationData grouping, broadband source creation, and post_norm](https://docs.flexcompute.com/projects/tidy3d/en/latest/_modules/tidy3d/components/data/sim_data.html)
- [Modal source construction from monitor VJPs](https://github.com/flexcompute/tidy3d/blob/develop/tidy3d/components/autograd/source_factory.py)
- [Autograd support and field-monitor restrictions](https://github.com/flexcompute/tidy3d/blob/develop/tidy3d/plugins/autograd/README.md)

## Factored spatial family

For each spatial group, concatenate the complex E/M current components to form
`J[k, spatial_dof]`. Define `n[k] = ||J[k]||_2`, normalize the rows, and phase-track
adjacent profiles outwards from the central sample. If `q[k]` is the removed
unit phase, then

```text
phi[k]   = q[k] J[k] / n[k]
alpha[k] = n[k] / q[k]
J[k]     = alpha[k] phi[k]
```

This identity preserves arbitrary objective magnitudes, signs, and complex
phases. The norm is an algebraic factorization scale; it does not replace the
Poynting/modal normalization already embedded in the objective cotangent.

Fit `phi[k]` using a Chebyshev polynomial centered on the Gaussian carrier,
with frequency coordinate `(f[k] - f_center) / max(abs(f[k] - f_center))`.
An order of M means M spatial coefficients, capped by the number of available
profile samples. The small frequency matrix is inverted separately from the
spatial matrix, avoiding a large full-domain cuSOLVER QR workspace.

This first integration uses physical current samples already present at
objective frequencies. It does not add extra mode solves at Chebyshev roots:
the callback has already consumed the mode profiles through the objective and
its exact detector pullback. A future interface can supply additional physical
profile samples separately. The generic current-family factorization and
separable spatial/temporal source representation also permit later low-rank
extensions.

## Temporal realization and complex normalization

`FactoredChebyshevCurrentSource` realizes each frequency polynomial using the
same compact analytic Gaussian-derivative basis as the forward broadband
source. It injects `real(sum_m C[m, r] p[m, t]))` simultaneously in a single
FDTD run, with both electric and magnetic currents retaining the existing
distributed-source update conventions. There is no arbitrary-frequency
isolation inversion and no duration increase to resolve wavelength spacing.

At each target frequency, adjoint fields are divided by the common complex
Gaussian source DTFT through `discrete_source_spectrum`, under the same
detector conventions as the existing per-frequency adjoint. They are then
multiplied by `alpha[k]` before frequency-wise gradient accumulation. This
post-factor is the exact objective/current factor removed above; it is not a
projection onto a reference spatial profile. Both quadratures of the stored
complex64 temporal basis are checked against their expected discrete spectrum.
The weighted residual is bounded using the spatial coefficient norms.

Before injection, a common complex scale anchors the whole spatial family to
the strongest original current sample, with its inverse absorbed into the
post-weights. This preserves the original current's scale and global phase for
the decay detector's dynamic range. It is an exact common rescaling and does
not replace the frequency-dependent profiles with the anchor profile. The
stopping condition also consults the composite source's coefficient-weighted
envelope, so Gaussian derivative tails cannot be truncated at the carrier's
Gaussian-only cutoff. Sparse Torch crops use the existing compact DLPack bridge,
including lazy-conjugate resolution, before conversion to JAX currents.

## Group selection and fallbacks

The feature remains opt-in with `MAPS_ENABLE_BROADBAND_ADJOINT=1`. Environment
settings are read when `core.fdtd.solver` is imported:

| Setting | Default | Meaning |
| --- | --- | --- |
| `MAPS_BROADBAND_ADJOINT_METHOD` | `auto` | `auto`, `chebyshev`, or unguarded legacy projection for debugging |
| `MAPS_BROADBAND_ADJOINT_PROFILE_ORDER` | `0` | Zero selects the smallest safe order from 1 through 5; positive selects one candidate |
| `MAPS_BROADBAND_ADJOINT_PROFILE_RTOL` | `1e-3` | Maximum relative spatial fit residual at any target sample |
| `MAPS_BROADBAND_ADJOINT_MIN_OVERLAP` | `0.5` | Minimum adjacent normalized profile overlap |
| `MAPS_BROADBAND_ADJOINT_MAX_COEFFICIENT_NORM` | `100` | Maximum coefficient-weighted peak temporal norm bound |

Additional guards require a spatial fit condition number below `1e5`, a
coefficient-weighted temporal basis error below `1e-3`, and a minimum relative
source spectrum of `1e-8`. Zero-current samples fall back safely. Every active
Yee current must belong to exactly one group; overlapping or incomplete
metadata therefore cannot double-count or lose a contribution. Metadata support
includes the one-cell Yee interpolation halo. Grouping retains every nonzero
current sample, without a physical-unit-dependent absolute cutoff.

The fallback hierarchy is order one for nearly invariant profiles, a low-order
fit for smooth profile variation, then independent per-frequency runs. The
legacy reference-profile projection is available explicitly. General exact
multitone/PCA synthesis is not introduced. If one group fails, this first
implementation sends the batch to the existing per-frequency path rather than
mixing source groups and frequency runs with ambiguous coverage.

The first factored source stores dense coefficient grids, including when the
objective and design recordings use the sparse phasor path. Independent
reference/fallback adjoints retain localized `CurrentSourceBlock` injection.
The sparse compatibility regression validates their combined data flow; it
does not claim that factored coefficient storage is spatially sparse. Blockwise
coefficient storage and general PCA compression remain future extensions.

The run-count comparison uses the common configured time-step limit:
`work_budget = number_of_runs * configured_timesteps`. Broadband grouping is
considered only if fewer spatial groups than target frequencies exist, and no
group may extend the simulation duration. This is a budget model, not a wall
time prediction: independent runs can stop early at different steps. Benchmarks
therefore also record actual steps and wall time for every FDTD invocation.

## Validation

`tests/fdtdx/test_factored_broadband_adjoint.py` covers arbitrary complex weights,
unit-family reconstruction, Chebyshev approximation error, sampled electric and
magnetic source spectra, differing sample/basis counts during placement,
profile-invariant order selection, and unsafe-profile fallback.

The GPU benchmark uses the MAPS 3D silicon bend at 20 px/um, a 0.8 ps time limit,
Chebyshev-5 forward excitation, and physical-area modal overlaps. It records
per-frequency raw/aligned adjoint-field errors and complex gain, aggregate and
design-region gradient errors/cosines/pointwise maxima, source classes, actual
run counts and time steps, wall time, and PyTorch CUDA allocation peaks. The
memory metric excludes JAX-owned allocations. Initial/reference runs include
compilation, so their wall times should be interpreted alongside timestep work.
The reported GPU runs used CUDA device 0, an NVIDIA RTX PRO 6000 Blackwell Max-Q
Workstation Edition, with PyTorch `2.9.1+cu128` in the supplied Python 3.11
environment.

`--fixed-horizon` runs every adjoint to the common configured time limit. The
invariant-profile fixture uses this automatically: otherwise its independent
runs stopped at 3,001 steps while unit-scaled broadband injection reached 8,392
steps, conflating finite-time device response with source accuracy. Even an
exact source factorization cannot repair insufficiently converged field
recordings. The production decay criterion is unchanged for existing sources;
convergence of the device transient remains a separate requirement.

Example commands (use the configured Python environment and GPU 0):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python unitest/benchmark_fdtdx_broadband_adjoint_calibration.py --span-nm 50 --num-wavelengths 5 --orders 1 2 3 5 --phase-weights
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python unitest/benchmark_fdtdx_broadband_adjoint_calibration.py --span-nm 1 --num-wavelengths 3 --orders 1 3 --phase-weights
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python unitest/benchmark_fdtdx_broadband_adjoint_calibration.py --objective invariant --phase-weights
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python unitest/benchmark_fdtdx_broadband_adjoint_calibration.py --objective multiport --num-wavelengths 5 --sparse --phase-weights
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python unitest/benchmark_fdtdx_broadband_adjoint_calibration.py --nonuniform --symmetry --phase-weights
```

An explicitly requested order can be rejected and use per-frequency fallback.
Its error may then be zero because it ran the oracle path, not because that
Chebyshev order was accurate. Always inspect source classes and run counts.
For a diagnostic order sweep, `--debug-profile-orders` relaxes only the spatial
fit threshold of explicit Chebyshev orders to one, so a poor order-one fit can
be measured instead of replaced by the reference. The automatic method retains
the production `1e-3` spatial guard, and all temporal/conditioning guards remain
active. This flag is for benchmarking, not production use.

### Five-frequency 50 nm regression

The run used arbitrary complex objective weights `[1, -1, i, -i, 0.3+0.8i]` at
1.525, 1.5375, 1.550, 1.5625, and 1.575 micrometers. All methods had identical
forward outputs. The reference used five adjoint simulations, each with 8,392
steps, totaling 41,960 adjoint steps. Each broadband method used one simulation
with 8,392 steps. Explicit order one below used the diagnostic spatial-fit
override; production would reject this fit.

| Method | Gradient relative L2 | Cosine similarity | Maximum normalized gradient error | Maximum raw adjoint-field error | Warm forward/backward wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy reference projection | 2.2504e-3 | 0.999997556 | 2.5631e-3 | 7.9898e-3 | 7.13 s |
| Factored order 1 (debug) | 2.5070e-3 | 0.999997091 | 3.0829e-3 | 5.3203e-3 | 7.08 s |
| Factored order 2 / auto | 6.8938e-5 | 0.999999997720 | 9.2029e-5 | 7.0134e-5 | 7.81 s (auto) |
| Factored order 3 | 1.6250e-5 | 0.999999999927 | 1.6301e-5 | 1.2954e-5 | 7.29 s |
| Factored order 5 | 1.5610e-5 | 0.999999999929 | 1.7548e-5 | 1.2934e-5 | 7.30 s |

Order-five maximum shape-aligned field error was `3.85e-6`; its design-region
gradient relative L2 was `1.46e-5`. Reported PyTorch allocation peaks were
approximately 0.47–0.66 GB across methods and exclude JAX memory. The remaining
few-parts-per-million complex field error is not calibrated from the reference.

The reference forward/backward call took 24.47 s including initial compilation,
and the order-five call took 7.30 s with cached forward compilation. Summed
adjoint FDTD invocation wall times were 9.49 s for the reference and 2.05 s for
order five. Including the identical 5,001-step forward solve, total FDTD work
was 46,961 versus 13,393 steps.

Per-frequency order-five adjoint-field diagnostics (no fitted calibration):

| Wavelength (um) | Raw complex relative L2 | Shape-aligned relative L2 | Gain magnitude minus one | Gain phase (rad) |
| --- | ---: | ---: | ---: | ---: |
| 1.5250 | 1.2934e-5 | 3.8517e-6 | -2.0248e-7 | 1.2345e-5 |
| 1.5375 | 4.2098e-6 | 2.5349e-6 | 1.0446e-7 | 3.3594e-6 |
| 1.5500 | 2.7160e-6 | 1.9341e-6 | -1.2557e-7 | 1.9027e-6 |
| 1.5625 | 8.9595e-6 | 2.4126e-6 | 9.9666e-7 | 8.5708e-6 |
| 1.5750 | 7.6895e-6 | 2.8503e-6 | 4.4735e-7 | 7.1276e-6 |

### Additional compatibility regressions

| Case | Method | Gradient relative L2 | Cosine similarity | Max normalized gradient error | Max raw adjoint-field error | Adjoint runs (reference -> grouped) | Adjoint steps (reference -> grouped) |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Single frequency | Per-frequency fallback | 0 | 1 (roundoff) | 0 | 0 | 1 -> 1 | 8392 -> 8392 |
| Invariant spatial profile, 30 nm, 3 wavelengths | Order 1 / auto | 9.5674e-5 | 0.999999995442 | 1.4383e-4 | 2.1784e-5 | 3 -> 1 | 25176 -> 8392 |
| 1 nm, 3 wavelengths | Order 1 / auto | 6.5289e-5 | 0.999999997893 | 5.9599e-5 | 1.0321e-4 | 3 -> 1 | 25176 -> 8392 |
| 1 nm, 3 wavelengths | Order 3 | 1.0412e-5 | 0.999999999985 | 1.1204e-5 | 6.4854e-6 | 3 -> 1 | 25176 -> 8392 |
| 30 nm, nonuniform grid and z symmetry, 3 wavelengths | Order 3 | 1.2962e-5 | 0.999999999940 | 1.5271e-5 | 1.4314e-5 | 3 -> 1 | 29235 -> 9745 |
| 30 nm, sparse monitors/sources and two ports, 5 wavelengths | Order 3 | 3.1420e-5 | 0.999999999523 | 2.9269e-5 | 1.3116e-5 | 5 -> 2 | 41960 -> 16784 |

These cases used the same complex phase/sign weight sequence and matched
adjoint horizons. Their respective reference/grouped forward-backward wall
times were 18.91/7.43 s and 26.36/10.47 s, with initial compilation included in
the reference. The sparse source conversion regression also exercises cropped,
noncontiguous, conjugated Torch data. Symmetry and nonuniform geometry were
tested together on a reduced `(88, 88, 13)` simulation grid.

The invariant-profile legacy gradient error was `9.5680e-5`, agreeing with the
new order-one result at this converged horizon. In the 1 nm case the legacy
gradient error was `4.9444e-5`; the default automatic method chose order one,
while explicit order three improved the gradient without a longer source or
runtime limit. Narrow-band order-three forward/backward wall time was 6.91 s
versus 19.11 s for the reference (including its initial compilation). Its maximum
shape-aligned adjoint-field error was `1.97e-6`.

Focused validation passed 106 tests covering the source, sparse scatter,
placement, symmetry, compatibility imports, and complex-permittivity conversion.
Scoped Ruff checks, byte-compilation, and whitespace validation passed. A
separate existing configuration-contract test still fails because the untouched
canonical `waveguide_3d.py` example lacks its required `optical_grid` key; this
adjoint-source patch does not change that example or its test.
