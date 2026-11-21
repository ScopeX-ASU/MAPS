# Variation-aware inverse design

Variations are opt-in. Call `model.configure_variation(config)` before constructing
`InvDesign`. Omitting this call, or passing `{"enabled": false}`, executes the
existing nominal forward path, without sampling or extra simulations. Disabled
configuration leaves the nominal checkpoint schema unchanged. The original
`grating_coupler_3d.py` example is unchanged by this implementation.

```python
from core.invdes.examples.grating_coupler_3d_var import variation_config

model.configure_variation(variation_config(seed=42, num_samples=4))
# The existing optimizer contract is unchanged:
results = model.forward(sharpness=2)
(-results["obj"]).backward()
```

See [the bilayer configuration](../configs/variation/grating_coupler_3d.yml).
Load its `variation` dictionary with your existing configuration loader and pass
that dictionary to `configure_variation`; the module does not evaluate YAML
expressions, import arbitrary modules, or automatically modify device configs.

## Architecture and units

`core/invdes/variation/sampling.py` generates named scalar deviations with an
independent RNG. `transforms.py` interprets physical deviations on the existing
MAPS optical grid. `robust.py` evaluates sequential scenarios and aggregates a
maximization objective. There is one nominal parameterization evaluation per
forward call and one shared set of learnable parameters.

Geometry lengths, offsets, and device coordinates are in **micrometers**. Tensors
use **(x, y, z)** ordering. Source adapters explicitly convert micrometers to
meters at the native FDTDx boundary. Relative permittivity deviations are
unitless. Samples are constants; the design-to-material path stays in PyTorch.

## Sampling

The public sampler accepts an ordinary dictionary:

```python
from core.invdes.variation import VariationSampler

sampler = VariationSampler({
    "enabled": True, "seed": 42, "num_samples": 4,
    "resample_every": 10, "strategy": "iid",
    "distributions": [
        {"params": ["arbitrary_parameter"], "backend": "scipy",
         "type": "norm", "loc": 0.0, "scale": 0.01},
    ],
})
batch = sampler.get(step=100)  # {"scenarios": [...], "weights": [...]}
```

- An integer seed gives a deterministic sequence. `seed: null` generates a seed
  from system entropy and records it as `sampler.seed`.
- `get(step)` caches by `step // resample_every`. Repeated optimizer closures at
  the same logical step receive the same scenarios. `InvDesign.run_step` sets
  that logical step; manual optimization must set `model.variation.step`.
- `strategy: fixed`, or `freeze()`, retains the current/next batch indefinitely.
  `unfreeze()` resumes interval-based sampling; `resample()` explicitly replaces
  a cached batch. `set_seed(seed)` resets the stream/cache, and
  `randomize_seed()` returns a newly generated seed.
- `sample(n)` draws a fresh batch without replacing the logical-step cache.
- `scenarios: [{...}, {...}]` specifies explicit process corners. Optional
  `weights` are finite nonnegative probabilities normalized to sum to one.
  Unspecified weights are uniform.

SciPy YAML adapters support `norm`, `uniform`, `truncnorm`, `lognorm`, and
`multivariate_normal`. For uniform distributions, use intuitive `min` and `max`
bounds with either the SciPy or NumPy backend:

```python
{"params": ["grating_layer_1.etch_depth"], "type": "uniform",
 "min": -0.01, "max": 0.01}
```

Bounds must be finite with `max > min`; scalars apply to every named parameter,
or per-parameter vectors can specify distinct intervals. Internally these map
to the native libraries without changing sampling. Existing SciPy `loc`/`scale`
and NumPy `low`/`high` configurations remain supported, but cannot be mixed with
`min`/`max` on uniform entries. Gaussian entries retain `loc`/`scale` and
accept `min`/`max` for truncation in physical parameter units:

```python
{"params": ["grating_layer_1.dx"], "type": "norm",
 "loc": 0.0, "scale": 0.02, "min": -0.06, "max": 0.06}
```

This samples the conditional distribution within the interval; it never clips
samples onto endpoints. `loc`/`scale` describe the underlying Gaussian, so the
truncated distribution can have a different mean and standard deviation. NumPy
`normal` entries use the same conditional inverse-CDF implementation. Gaussian
scalar/vector bounds support IID, fixed, and scalar inverse-CDF strategies.
Joint distributions and other adapters accept min/max through IID rejection
sampling, retaining their conditional dependence. Rejection fails explicitly
after 1024 batches if the acceptance probability is too small. Existing native
SciPy `truncnorm` uses standardized `a`/`b`; min/max always use parameter units.
Configurations without min/max retain their previous draws.

Joint normal entries require
matching mean/covariance dimensions and symmetric covariance; SciPy validates
positive definiteness. NumPy adapters support `normal`, `uniform`, and
`multivariate_normal`. The sampler has no hardcoded physics parameter names.

A mixture entry uses `type: mixture`, `params`, `weights`, and `components`, each
component being a distribution entry without `params`. Mixtures support IID
sampling. Sobol, Halton, Latin hypercube (`latin_hypercube`), and antithetic
strategies use scalar inverse CDFs. Multivariate normals, mixtures, and
PyTorch/custom samplers reject these strategies rather than inventing a
unit-cube transform. Antithetic draws use inverse-CDF pairs `u` and `1-u`; odd
batch sizes have one unpaired draw. Sobol balance requires a power-of-two count.

Programmatic extensions require no registry or dynamic imports:

```python
from scipy import stats
import torch

# Additional native frozen scalar SciPy distributions:
entry = {"params": ["p"], "backend": "scipy",
         "distribution": stats.beta(2, 3)}
# Dedicated CPU-only Torch adapter, restoring the global CPU RNG afterward:
entry = {"params": ["p"], "backend": "torch",
         "distribution": torch.distributions.Normal(0.0, 0.01)}
# A custom distribution returns shape (n, number_of_params):
entry = {"params": ["p"], "backend": "custom",
         "distribution": lambda rng, n: rng.normal(size=(n, 1))}
# A custom strategy returns n named dictionaries:
config = {"enabled": True, "seed": 42,
          "strategy": lambda rng, n: [{"p": float(rng.uniform())} for _ in range(n)]}
```

## Geometry and overlay

The built-in adapter accepts 3D box regions with vertical, zero-angle extrusions
of xy patterns and `linear_eps` material blending. Named deviations are:

| Name | Meaning |
| --- | --- |
| `grating_layer_1.dx`, `.dy` | Translate layer 1 relative to fixed structures |
| `grating_layer_2.dx`, `.dy` | Independent layer 2 overlay |
| `<region>.etch_depth` | Additive etch-depth error, keeping the layer top fixed |
| `<region>.thickness` | Increase grating layer thickness, keeping the stack bottom fixed |
| `silicon_thickness_error` | Alias for `grating_layer_1.thickness` |
| `<region>.eps` | Change the patterned region's high relative permittivity |

Lateral offsets use differentiable, separable physical-coordinate interpolation
on uniform or nonuniform cell centers. The full-grid support moves, so features
can leave the original design box without clipping there. Fixed waveguides and
slabs remain fixed during overlay. Regions compose in their existing insertion
order, including overlaps. Shared/correlated layer deviations can be generated
by a joint distribution or explicit/custom scenarios.

Etch depth and layer thickness use exact fractional z-cell occupancy, not
rounded boundary indices. **The etch-depth distribution describes the delta,
not the absolute depth:** `depth = nominal_depth + sampled_delta`. For a nominal
depth of 0.16 um, the uniform example above produces depths between 0.15 and
0.17 um. A positive delta etches deeper; a negative delta etches shallower.
The grating's existing unetched slab safety margin
of one nominal grid step is preserved. Thickness shifts overlying layers and
changes the first layer's access-waveguide thickness; substrate and stack
bottom remain fixed. Etch depth must remain positive and no greater than the
layer thickness. The current adapter does not create/remove design regions at
zero etch depth. When a grating includes a surrounding `grating_layer_N_box`, it represents
the full-thickness material boundary defined by the second etch. Its lateral
padding is `design_region_border_size` per side, in micrometers. Changing the
first pattern etch depth changes the patterned region and remaining slab, but
**does not thin the protected border**. The boundary and pattern are separate
lithography/etch processes: the built-in pattern adapter never moves, resizes,
or changes the permittivity of separately named `grating_layer_N_box` geometry,
including during pattern thickness variation or upper-layer shifts. No process
coupling is inferred from the `_box` suffix. Deposited-layer thickness errors
that should affect both processes require an explicit coupled materializer.

The built-in adapter currently accepts design-region targets only; explicit
`grating_layer_N_box.*` variation is not yet supported by it. To model boundary
process errors independently, use a custom materializer with separately named
parameters and distributions (and `recording_bounds` for sparse recording).
Shared or correlated process errors must likewise be specified explicitly.
The second etch-depth error needs its own physical convention; it must not be
interpreted as the patterned region's first etch-depth delta. Static backgrounds use the existing rasterizer and a bounded
four-entry cache; simulation grids are reused.

Native MAPS extrusion already blends partial z cells with fixed geometry. The
adapter recovers the shared pattern by inverting one well-covered slice before
applying new coverage. It retains MAPS's established extrusion gradient-scaling
surrogate; it does not replace that convention with an unscaled derivative.
Zero-valued scenarios return the nominal tensor exactly.

Offsets along a symmetry-reduced axis are rejected. Disable the corresponding
symmetry before constructing the device. Out-of-domain regions, unknown names,
nonfinite samples, and impossible physical ranges fail explicitly. Unbounded
normal distributions can generate invalid scenarios; use min/max truncation
or explicit corners where fabrication bounds are required.

## Sparse recording under geometry variation

With `MAPS_SPARSE_PHASOR=1`, robust evaluation expands each design-region
phasor detector to a **fixed envelope of the configured support**, including
nominal geometry, lateral interpolation/resampling margins, etch-depth changes,
and cumulative stack thickness changes. Forward and adjoint solves use matching
envelopes through the immediate material backward. Port detectors and nominal
visualization slabs keep their existing locations. Original detector profiles
are restored after robust evaluation, including on failure.

Every geometry distribution must have finite support, supplied by min/max
truncation, bounded native distributions, bounded mixtures, or explicit corners.
The envelope comes from theoretical support, never observed sample extrema.
Unbounded geometry support fails explicitly in sparse mode; dense recording
continues to permit unbounded sampling subject to the existing physical checks.
Source-only variation does not enlarge design recorders. Custom materializers
must implement `recording_bounds(support)` in native half-open grid indices;
opaque custom sampling strategies must use explicit scenarios for sparse mode.

The physical envelope must lie inside the simulation domain; truncation does not
silently clip geometry or guarantee a valid etch-depth/thickness combination.
A larger envelope stores more phasor cells, but placement/shape stays independent
of each draw. This avoids variation-induced detector shape changes, without
claiming that all other JAX recompilation sources are eliminated. A region whose
support spans much of the domain can approach dense recording costs.

Plotting remains centered at the nominal region. A center slice can miss a
strongly shifted feature. The live field is the last evaluated scenario and the
exported material remains nominal. Where a runtime material state is available,
the plot uses that state's permittivity, matching the last scenario; otherwise
it uses the nominal map. These are diagnostic views, not aggregate fields.

## Gaussian sources

Use actual source slice names, for example `source.in_slice_1.dx` and `.dy`.
The built-in source adapter supports z-propagating native
`GaussianPlaneSourceTidy3d` beams and deviations in `waist_radius` and
`waist_distance`. Beam offsets move the profile within the fixed injection
plane. Native Poynting-flux normalization is recomputed at placement. The
nominal profile objects are restored even if evaluation fails, and zero beam
offsets preserve the existing native profile exactly.

The source extension adds only an optional `beam_center_offset` field, in
meters, to the existing Gaussian class. Wavelength, polarization, and incident
angle variation are not wired into this adapter. Source changes can trigger
native JAX compilation; avoiding all recompilation is not guaranteed.

## Robust objectives and mutable adjoint state

`aggregation` accepts `mode: mean`, `mean_std`, `minimum`, or `smooth_min`.
`mean_std` maximizes the weighted mean minus `penalty` times population standard
deviation. `smooth_min` is `-temperature * log(sum(w * exp(-J / temperature)))`.
Zero-weight scenarios are excluded from minimum/smooth-min aggregation.
Programmatic custom modes receive `(values, normalized_weights)` and return a
scalar tensor. All objectives use MAPS's **maximization** convention.

FDTDx adjoint preparation closes over mutable simulation objects. Robust mode
therefore computes each scenario's derivative with respect to an independent
material leaf immediately, before evaluating the next scenario. A first-order
surrogate reconnects that derivative to the shared design. The returned `obj`
supports the existing backward contract and weighted aggregation without
retaining all scenario solver graphs. Solver adjoints occur **inside robust
forward**; the outer backward propagates through materialization and nominal
design only. Higher solver derivatives/Hessians are unsupported. Variation
parameters themselves are not differentiated.

Results additionally contain `scenario_objectives`, `scenario_breakdowns`,
`scenarios` (including weights), and `variation_timings`. `eps_map` and design
region tensors remain nominal for regularization/export. Aggregate `breakdown`
values are weighted means for diagnostics, even when `obj` uses a nonlinear
aggregate. The objective's live field state belongs to the last scenario; these
fields must not be interpreted as an aggregate field or a nominal simulation.

### Permittivity-gradient plots

`eps_grad=True` uses the same xy/xz/yz slices and nominal plotting centers as
nominal optimization. Robust evaluation retains the last scenario's runtime
permittivity before its immediate adjoint. Detached material derivatives are
saved for diagnostics; outer backward supplies the actual loss sign and the
aggregation coefficient, then populates that runtime tensor's `.grad`. No
additional adjoint solve is needed and the shared design gradient is unchanged.
Before outer backward, the diagnostic `.grad` is absent rather than showing the
positive objective derivative from internal robust evaluation.

The plotted quantity is the **last scenario's contribution to the loss material
gradient on its physical optical grid**. For `loss = -obj` with mean aggregation,
this is `-weight_last * d(obj_last)/d(eps_last)` using normalized weights. Other
aggregation modes use their own scenario derivative; a last scenario that is
inactive under `minimum` has a zero contribution. These plots are not the
aggregate gradient pulled back to the shared nominal design. Parameter gradients
continue to include contributions from every scenario and regularizer.

A single zero-variation scenario must match nominal material gradients in all
three views, including sign, dtype, layout, and device. Nonzero shifts/etch or
source changes can physically change the gradient, while expanded sparse
recording covers more cells than nominal-only recording. Runtime fields and
material gradients belong to the last scenario; exported `eps_map` remains
nominal. Active/thermal robust integrations remain outside the built-in scope.

## Checkpoint continuation

Enabled models add `_variation_state` through optional state-dictionary hooks;
nominal/disabled models retain their existing keys. Configure the same variation
settings and adapters before loading an enabled model checkpoint. Sampler state
includes the actual seed, private RNG state, cached batch, interval bucket,
freeze state, and configuration signature. Controller state includes the logical
step. Executable custom callables are not stored in checkpoints.

```python
state = model.state_dict()  # Existing MAPS checkpoint machinery includes this.
restored.configure_variation(config)
restored.load_state_dict(state)
# Continue at the intended logical step, not the number of closure evaluations:
restored.variation.step += 1
```

For `InvDesign` continuation, set the existing `run.start_epoch` to the next
logical step. This change does not redesign optimizer/scheduler checkpointing.
Sampler-only continuation uses `state_dict()` and `load_state_dict()` directly.

## Optimization example, tests, and benchmark

```bash
export CUDA_VISIBLE_DEVICES=0
PYTHON=python
$PYTHON core/invdes/examples/grating_coupler_3d_var.py
$PYTHON -m pytest tests/unit/invdes tests/gradient/invdes tests/integration/invdes
# Optional profiling belongs to the test utility:
$PYTHON tests/integration/invdes/benchmark_grating_coupler_variation.py \
  --mode compare --samples 2 --steps 2 --output /tmp/grating-variation.json
```

The optimization example follows `grating_coupler_3d.py`: a 4 × 4 um bilayer
grating, configurable wavelengths, flux-normalized modal objectives, level-set design,
100 `InvDesign` epochs, Adam, cosine learning-rate/sharpness schedules, plots,
and GDS export. Edit `variation_cfgs` beside the optimization configuration to
set the seed, sample count, or disable variations. Its default uses an entropy
seed and one scenario per step. Symmetry reduction is disabled to support
lateral offsets. Permittivity-gradient plotting uses the last scenario runtime
material state, while regularization and exports use nominal maps.

Small validation devices and timing loops live exclusively in
`tests/integration/invdes/benchmark_grating_coupler_variation.py`. This test
utility offers nominal, fixed-seed (`--mode fixed`), and entropy-seeded
(`--mode random`) comparisons. Here `fixed` means a fixed
**seed**, with normal periodic resampling; sampler `strategy: fixed` freezes
scenarios. Compare mode restores the same nominal parameters before each run.
Initialization/normalization is excluded from step timings. Robust host timings
separate parameterization/setup, materialization, solver forward, and immediate
material backward. Outer backward and total steps are CUDA-synchronized in the
benchmark. Reported peak memory covers **Torch allocation only**, excluding JAX
and other CUDA allocations. Warm-up/compilation must be accounted for when
comparing timings.

Automated tests cover deterministic sampling/resume, global RNG isolation,
probability weights, integer/fractional/nonuniform translation, exact interval
coverage, analytical box reconstruction, finite-difference transform/aggregate
gradients, mutable backward-state safety, nominal checkpoint compatibility,
native Gaussian profiles, native grating reconstruction, and an actual robust
FDTDx GPU optimizer step. The recording regression compares fixed envelopes
with full-domain design detectors through the same sparse phasor/adjoint path
on uniform and nonuniform native grids, including a nominal scenario. This
isolates coverage from differences between legacy dense and sparse solver
paths. Its float32 objective/gradient tolerance is `rtol=1e-4, atol=1e-12`;
the small absolute floor is necessary for the short-pulse objectives near
1e-8 and prevents a vacuous near-zero comparison. GPU tests require executed CUDA runs to establish
GPU compatibility. Numerical tolerances remain quantity-specific.

### Scope and limitations

The generic sampler/aggregator can be used with FDFD or other backends through
`configure_variation(config, materializer=..., source_context=...)`; a custom
materializer receives `(nominal_build_tuple, geometry_deviations)` and returns a
PyTorch optical tensor. A custom source context receives `(model, scenario)`.
The built-in geometry/source integration is currently for lossless 3D grating
stacks with vertical extrusions. Sidewall halos, arbitrary 3D patterns,
nonlinear endpoint blending, complex material variations, thermal/control-state
coupling, and automatic FDFD source adapters need explicit custom integrations.
Nominal FDFD solvers are unchanged. No independent new FDFD robust EM validation
or full converged optical validation is claimed by the small FDTDx smoke case.

### Recorded validation (2026-10-08)

Using the requested Python 3.11 environment and `CUDA_VISIBLE_DEVICES=0`:
63 focused/related tests passed in one run, including the native GPU material-map
comparison, CUDA RNG-isolation check, and slow GPU FDTDx optimizer regression. Black
checks passed for all changed Python files; Ruff passed for the new package,
example, and tests; `git diff --check` passed.

The bounded-sampling/fixed-envelope update passed 80 focused tests with GPU
markers deselected. Executed GPU regressions also passed for uniform and native
nonuniform recording envelopes versus full-domain detectors, random robust
optimization, native etch reconstruction, and CUDA RNG isolation. The uniform
coverage comparison had zero objective and gradient differences. The native
etch adapter preserves separately named surrounding layer material boxes
without changing the nominal geometry path. The protected-border regression independently
integrates material thickness in a cross-section outside the patterned region
for deeper/shallower etches and combined thickness/etch changes. Native GPU
reconstruction also covers a 0.1 um border per side.

The eps_grad diagnostic correction passed 93 focused tests with GPU markers
excluded, plus two executed sparse FDTD GPU regressions. The single-zero-scenario
case compares actual runtime material gradients on the xy/xz/yz region-center
planes and shared parameter gradients with nominal optimization using the same
`rtol=1e-4, atol=1e-12` tolerance. Analytical tests verify loss sign and scenario
aggregation coefficients for mean, mean_std, minimum, and smooth_min.

An exploratory comparison against the legacy dense solver path showed
percent-level relative differences at very small short-pulse objectives; its
initial absolute tolerance was too permissive. The final coverage regression
instead uses full-domain detectors through the same sparse phasor/adjoint path
and the tighter scale-aware tolerance stated above. Equivalence of the two
solver paths is not established by this change.

A broader run also exposed three failures in unchanged grating material-role
and canonical waveguide configuration code. Their assertions were retained:
substrate-material expectations differ, the no-buried-material path calls a
missing material function, and the canonical waveguide example lacks the
expected `optical_grid` field. These are outside this variation change.

The benchmark used an RTX PRO 6000 Blackwell Max-Q, a **56 × 36 × 63** uniform
grid at 20 samples/um, one wavelength (1.31 um), two bilayer design patterns,
Adam, and a shortened **3e-14 s** FDTD duration. This validates execution and
profiling, not converged coupling efficiency. Source offsets, layer overlays,
and shallow-etch deviations were evaluated together.

| Mode | First measured step | Second measured step | Second-step peak Torch memory |
| --- | ---: | ---: | ---: |
| Nominal, one scenario | 16.79 s | 5.21 s | 50.15 MiB |
| Seed 42, two scenarios | 11.02 s | 10.59 s | 74.90 MiB |
| Entropy seed, two scenarios | 10.59 s | 11.01 s | 91.85 MiB |

All modes ran in the same process, nominal first; the nominal first step includes
compilation. The lower robust first-step time is therefore **not a speedup
claim**. For the second seed-42 step, setup/sampling took 1.66 ms, geometry
materialization 32.50 ms, solver forward 8.02 s, immediate material adjoints
2.53 s, and outer design backward 3.13 ms. Host component timings are diagnostic;
the total step measurement synchronizes CUDA. Peak Torch memory depends on prior
runs and excludes JAX allocations. Detailed measurements were written under
`/tmp/maps-variation-benchmark-final.json`, without adding generated artifacts
to the repository.
