# Exact near-field backend dispatch

`core.nearfield.dispatch.project` provides a small wrapper for the exact
arbitrary-point near-field projector. It accepts `direct_torch`,
`direct_triton`, or `auto`; both explicit backends preserve their existing
projector contracts. Planar separable far-field projection and FFT angular
spectrum propagation are different mathematical APIs and are not candidates
for this selector.

AUTO has no built-in cost model. A caller may pass JSON-like records emitted
by `benchmarks/nearfield/projector.py` through `calibration_records`. The
selector compares only records matching the exact source count `P`, observation
count `M`, frequency count `F`, real and field dtypes, current GPU name,
PyTorch version, CUDA version, and applicable chunk/block settings. It also
requires at least three measured warm repeats and both complex128 precision
reference errors at or below one percent. Invalid, stale, or incompatible
records are ignored with their rejection reason available from
`select_backend`.

There is no extrapolation to another shape, dtype, device, software version, or
chunk configuration. For each eligible record the cost estimate is its
measured first call plus `(expected_calls - 1)` times the measured warm median.
When backward is expected, the corresponding first forward-plus-backward and
warm forward-plus-backward measurements are used. If `requires_backward` is
omitted, it is inferred from differentiable input tensors. First-call timing
includes Triton compilation when the benchmark record says so, which matters
for one-shot calls; warm timings matter when the exact shape will be reused.
Because the benchmark's first gradient call follows an initial no-grad call,
the selector adds the measured first-forward premium over the warm-forward
median to the backward first-call cost. Torch source/observation chunks and
Triton forward source/observation blocks are matched independently; the
wrapper defaults to Triton blocks of 16 sources and 32 observations. For
backward selection, Triton records must also match the source and observation
chunks used by its Torch recomputation. AUTO checks whether Triton is
installed without importing it; the optional module is imported only if
Triton is selected. If a benchmark record does not contain a Triton version,
the caller should refresh it after changing Triton versions.

If no record applies, AUTO selects `direct_torch` and explains why through
`select_backend`. It does not run a benchmark or compile a candidate to gather
new evidence. Triton remains lazily imported only when the chosen backend is
`direct_triton`. The caller is responsible for providing the validated records
that reflect the intended workload and for deciding how to log the returned
decision.

The benchmark uses deterministic synthetic fields for performance and
complex128 agreement checks. Those records support backend cost comparison and
numerical agreement for the benchmark inputs; they do not validate the
physical accuracy of a particular sampled surface.
