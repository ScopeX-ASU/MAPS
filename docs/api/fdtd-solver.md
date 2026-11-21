# FDTD Differentiable Solver and Adjoint Contract

This page documents the current MAPS Torch/JAX differentiation bridge implemented in `core/fdtd/solver.py`. It is descriptive documentation only; it does not alter solver behavior.

## Role

`core/fdtd/solver.py` connects:

```text
PyTorch MAPS model / objective
        |
        v
Torch autograd bridge
        |
        v
JAX / fdtdx forward FDTD
        |
        v
phasor detector fields
        |
        v
adjoint-source construction + adjoint FDTD
        |
        v
gradient with respect to eps_r
```

The field-level API is documented in [FDTD API](fdtd.md).

## Component convention

The solver uses the fixed electromagnetic component order:

```text
("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
```

Electric-component indices are defined separately from magnetic-component indices:

```text
Ex -> 0    Hx -> 0
Ey -> 1    Hy -> 1
Ez -> 2    Hz -> 2
```

The phasor detector normally requests all six components when `MAPS_ENABLE_PHASOR_H=1`. If disabled, only electric phasors are requested and magnetic phasors are derived analytically where required.

## Environment-controlled behavior

Several environment variables alter solver behavior and should be treated as part of the runtime contract.

`MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM`
: Defaults to enabled (`"1"`). Controls source-spectrum normalization of phasor detector outputs.

`MAPS_ENABLE_PHASOR_H`
: Defaults to enabled. When disabled, the code warns that H phasors are analytically derived from E phasors, while source-plane H fields may still be recorded by an additional detector.

`MAPS_PHASOR_H_NATIVE_SLAB_CELLS`
: Integer number of native slab cells used by the H-field support path. Default: `3`.

`MAPS_SPARSE_PHASOR`
: Defaults to disabled (`"0"`). Set to `"1"` before importing the FDTD/device
  modules to record localized forward and adjoint phasors while preserving the
  dense Torch-facing field and autograd interfaces. See
  [Sparse phasor mode](#sparse-phasor-mode).

`MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES`
: Defaults to enabled (`"1"`) within sparse phasor mode. Forward simulation
  records thin full-span `xy`, `xz`, and `yz` slabs through each design-region
  center so `plot_eps_field_3d()` receives populated cross-sections. Set it to
  `"0"` before importing MAPS when plots are not needed and minimum forward
  detector cost is preferred. It does not change dense mode or sparse adjoint
  detector placement. When there are no design regions, one set of three slabs
  is centered at physical `(0, 0, 0)`.

`MAPS_SPARSE_ADJOINT_SOURCE`
: Defaults to enabled (`"1"`) independently of sparse phasor recording. It
  permits the per-frequency adjoint source builder to use spatial current
  blocks when grouping metadata is available. Set it to `"0"` for the
  validated sparse-phasor configuration described below, in which dense
  cotangents construct a dense adjoint source.

`MAPS_ENABLE_BROADBAND_ADJOINT`
: Defaults to disabled. The implementation explicitly marks this as experimental and warns that gradients may be incorrect unless the grouped frequencies share the required spatial profile. Validate against the equivalent per-frequency adjoint gradients before relying on this mode.

`MAPS_FDTDX_ADJOINT_DEBUG`
: Enables adjoint debug plotting/output.

`MAPS_FDTDX_ADJOINT_DEBUG_DIR`
: Directory for adjoint debug artifacts. Default: `figs/fdtdx_adjoint_debug`.

`MAPS_COMPARE_SPARSE_DENSE`
: Defaults to disabled. In sparse mode, enables diagnostics comparing each
  localized detector with the corresponding slice of the dense tensor assembled
  from those detectors. It does not add or run a full-volume reference detector.

`MAPS_WRITE_SPARSE_DENSE_PLOTS`
: Defaults to disabled. When both this flag and
  `MAPS_COMPARE_SPARSE_DENSE=1` are enabled, writes the configured sparse-scatter
  diagnostic plots.

Boolean environment variables are enabled only by the literal string `"1"` and
are read at module import time. Set them on the command that starts Python rather
than changing them after importing MAPS.

## Sparse phasor mode

Sparse phasor mode reduces FDTD-X detector work without changing the
optimization-facing data contract. It does not return a sparse PyTorch tensor.

### Forward contract

With `MAPS_SPARSE_PHASOR=1`, device setup creates localized phasor detectors for:

- every design region needed by the permittivity gradient; and
- every objective port needed by downstream field/overlap calculations; and
- when `MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES=1`, the three orthogonal
  cross-sections through each design-region center used by
  `plot_eps_field_3d()`.

Visualization supports are three native cells thick along the plane normal so
plot-grid interpolation has neighboring samples. They span both in-plane axes,
are folded onto the retained half of a symmetry-reduced domain, and are
parity-unfolded with the other forward detector states. Coincident planes shared
by multiple design regions are recorded only once. For a device with no design
regions, MAPS creates one `xy`/`xz`/`yz` set at the native cells nearest physical
origin, matching the fallback center used by `plot_eps_field_3d()`.

The sparse 3-D plotting path uses these slabs directly. For each export-grid
cross-section, it selects only the one or two native samples bracketing the
plane coordinate, performs the same separable rectilinear interpolation on the
thin slab, and transfers the resulting 2-D plane to CPU. It does not construct
full export-grid volumes for fields, epsilon, or epsilon gradients. Dense mode,
disabled visualization planes, and far-field-extended plots retain the legacy
full-volume plotting path.

Detector states have local shapes
`[Nfreq, Ncomponent_local, nx, ny, nz]`. After the time-domain solve, MAPS:

1. normalizes each detector by the source spectrum when normalization is enabled;
2. unfolds symmetry-reduced data with PEC/PMC component parity;
3. scatters each local detector into native full-domain coordinates; and
4. zero-fills cells and components not covered by a detector.

The custom autograd function returns a complex dense tensor:

```text
[Nfreq, 6, Nx, Ny, Nz]
```

in component order `(Ex, Ey, Ez, Hx, Hy, Hz)`. Consequently, existing Torch
objectives receive the same rank, axis order, full spatial extent, dtype/device
family, and dense `grad_output` interface as the dense-phasor path. Objectives
must be covered by a configured sparse port detector; reading an uncovered cell
returns the intentional zero fill.

Overlapping detector samples are not added. Identical values are retained once;
conflicting values raise an error instead of silently changing field amplitude.

### Backward contract

PyTorch supplies a dense cotangent with shape `[Nfreq, 6, Nx, Ny, Nz]`. Sparse
backward applies the transpose of the dense observation operations—spatial
colocation, magnetic half-time-step correction, and symmetry unfolding—to
construct the Yee-grid adjoint-source cotangent.

The adjoint FDTD simulation records phasors only in design regions. It does not
place a full-domain phasor detector and does not record objective-port phasors.
The localized adjoint states are source-normalized, symmetry-unfolded, and
scattered back into a dense six-component tensor. The permittivity derivative is
then computed from dense forward and dense adjoint electric fields.

This sequence preserves the dense autograd boundary while reducing detector
storage and per-step phasor accumulation inside FDTD-X.

### Symmetry

Sparse detector bounds are defined on the native full domain and resolved onto
the reduced FDTD-X grid during placement. For an active symmetric axis, source
profiles and detector states use the reduced length—for example, a full `Ny=162`
domain uses `81` cells under `(0, -1, 0)`—and are unfolded before exposure to
Torch. PEC and PMC parity are component-dependent; mirrored values must not be
copied with a single sign for all six fields.

### Validated invocation

The current dense-source sparse-phasor configuration is:

```bash
MAPS_SPARSE_ADJOINT_SOURCE=0 \
MAPS_SPARSE_PHASOR=1 \
MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES=1 \
MAPS_COMPARE_SPARSE_DENSE=0 \
MAPS_ENABLE_BROADBAND_ADJOINT=0 \
python core/invdes/examples/grating_coupler_3d.py
```

Set `MAPS_ENABLE_BROADBAND_ADJOINT=1` to select grouped broadband adjoints when
the number and spatial grouping of frequencies make that branch applicable.
Broadband grouping remains experimental and must be compared with the equivalent
per-frequency (`MAPS_ENABLE_BROADBAND_ADJOINT=0`) gradient.

For optimization runs that do not generate cross-section plots, use
`MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES=0` to retain only design-region and
objective-port forward support.

Optional adjoint diagnostics can be enabled without changing detector support:

```bash
MAPS_FDTDX_ADJOINT_DEBUG=1 \
MAPS_FDTDX_ADJOINT_DEBUG_DIR=./figs/fdtdx_adjoint_debug_sparse \
MAPS_SPARSE_ADJOINT_SOURCE=0 \
MAPS_SPARSE_PHASOR=1 \
python core/invdes/examples/grating_coupler_3d.py
```

## Symmetry unfolding

`unfold_array(arr, symmetry, spatial_axes=(0, 1, 2), signs=None)` reconstructs mirrored spatial domains.

### Inputs

`arr`
: PyTorch tensor. Spatial rank/layout is caller-defined through `spatial_axes`.

`symmetry`
: Three-entry tuple corresponding to the physical x/y/z axes. A zero entry means no unfolding along that axis; any nonzero entry causes the current tensor to be mirrored and concatenated.

`spatial_axes`
: Tensor-axis indices corresponding to physical x/y/z. Default `(0, 1, 2)`.

`signs`
: Optional mapping from physical axis index to a scalar/tensor sign applied to the mirrored half. This supports parity changes of field components under PEC/PMC symmetry.

### Operation

For each active physical axis `a`:

```text
mirror = flip(out, spatial_axes[a]) * sign[a]
out    = concat([mirror, out], spatial_axes[a])
```

The operation remains differentiable for PyTorch tensor inputs/signs.

## Adjoint grouping

MAPS can group adjoint support by frequency and spatial monitor region.

### `adjoint_group_info` schema

`build_adjoint_groups_from_monitor_info()` consumes a sequence of dictionaries. Each usable entry contains:

```text
"group_id"
"monitor_slice"
"freq_indices"
```

`group_id`
: Identifier converted to string for the output mapping.

`monitor_slice`
: Three-dimensional spatial monitor selection. Each axis may be a Python `slice`, scalar-like index, or array-like index set.

`freq_indices`
: Frequency indices associated with the group.

The routine evaluates the actual gradient support and removes spatial points that are inactive across the group's selected frequencies.

### Gradient tensor

The grouping code expects a gradient tensor whose first axis is frequency and second axis is electromagnetic component:

```text
[Nfreq, Ncomponent, X, Y, Z]
```

The implementation computes support using:

```text
amax(abs(grad), dim=1) > threshold
```

so the component axis is reduced when determining whether a spatial point is active.

### Group representation

Each retained group is represented as:

```python
{
    "freq_indices": LongTensor[Ngroup_freq],
    "spatial_indices": LongTensor[Npoint, 3],
}
```

where each row of `spatial_indices` is an `(x, y, z)` integer grid coordinate.

## Compact/group gradient extraction

`extract_group_grad_compact_torch(grad, group)` uses the group's frequency and spatial indices to retain only the selected support while returning a tensor with the same selected-frequency spatial layout as `grad[freq_indices]`.

The implementation initializes:

```python
out = torch.zeros_like(grad[freq_indices])
```

and copies only:

```text
[:, :, xs, ys, zs]
```

from the original gradient.

The existing function docstring describes the conceptual selected content as:

```text
[ngroup_freq, 6, N]
```

but the implementation currently preserves the spatial tensor layout in `out`. Consumers should follow the actual implementation rather than assuming a physically flattened return tensor.

## Monitor slices to spatial coordinates

The internal `_slice_to_spatial_indices()` converts a three-axis monitor selection into Cartesian-product integer coordinates.

For each axis:

- a Python `slice(start, stop)` becomes `arange(start, stop)`;
- a scalar becomes a length-1 integer tensor;
- an array-like selection is flattened, converted to int64, and uniqued.

The result is:

```text
LongTensor[Npoint, 3]
```

with rows `(x, y, z)`.

A slice must have an explicit `stop` value.

## Phasor normalization

By default MAPS normalizes phasor detector results by the source spectrum. This matters for broadband simulations because a detector phasor contains both the system response and the spectral amplitude of the excitation.

Do not compare raw phasor magnitudes from runs with different source spectra without accounting for this normalization setting.

## E/H phasor handling

When native H phasors are available, the field pipeline can consume all six components directly. Otherwise H is derived from E using the fdtdx state/configuration and monitor wave-character information.

At the field API boundary, `fdtd3d.solve()` returns six complex component tensors and performs temporal/spatial colocation before exposing them.

## Adjoint scaling

The module defines:

```text
SCALE = 5e-11
```

The source comments describe this as a deliberate gradient-scale choice for the FDTD adjoint path. This is an implementation-specific numerical scaling convention; it should not be silently removed or changed as cosmetic cleanup. Any future change requires gradient-regression validation.

## Broadband adjoint caution

Broadband adjoint mode is explicitly experimental in the current implementation. The warning states that each spatial group needs to share the same spatial profile from the same port and originate from a broadband objective.

Recommended validation when enabling it:

```text
1. Run the broadband grouped adjoint calculation.
2. Run the equivalent per-frequency adjoint calculation.
3. Compare gradients in magnitude, spatial support, and direction/sign.
4. Do not accept the broadband result solely because optimization converges.
```

## Typical data flow

Conceptually, a differentiable solve follows:

```python
# eps_r is a differentiable torch.Tensor [Nx, Ny, Nz]
Ex, Ey, Ez, Hx, Hy, Hz = sim.solve(
    input_slice_name="in_slice_1",
    wl_cen=1.55,
    wl_width=0.0,
    n_wl=1,
    mode="Ez1",
)

objective = some_objective(Ex, Ey, Ez, Hx, Hy, Hz)
objective.backward()

# Gradient propagates through the Torch/JAX fdtdx bridge to eps_r.
```

The exact objective and port configuration are application-specific. This example documents the differentiation relationship, not a complete standalone simulation setup.

## Scientific invariants for future changes

Changes to this subsystem should preserve or explicitly validate:

- electromagnetic component ordering;
- frequency-axis semantics;
- native versus colocated Yee-grid fields;
- E/H temporal staggering correction;
- symmetry parity during unfolding;
- source-spectrum normalization;
- adjoint spatial/frequency grouping;
- gradient scaling;
- Torch/JAX dtype and device conversion;
- native/export grid mapping.

Because these details directly affect gradients, a refactor that leaves forward field plots visually similar can still be scientifically incorrect.
