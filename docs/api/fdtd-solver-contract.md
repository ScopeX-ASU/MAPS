# FDTD-X Solver and Adjoint Contract

`core.fdtd.solver` is the Torch-facing runtime bridge around FDTD-X. It separates simulation execution, detector/phasor extraction, and optimization-facing gradient construction.

## Runtime boundary

The solver translates among three array ecosystems:

```text
PyTorch design/material tensors
        <->
JAX / FDTD-X simulation arrays
        <->
Torch objective / gradient tensors
```

Conversion helpers must preserve the intended shape, component order, dtype, device semantics, and complex phasor interpretation.

## FDTD-X solve

The physical simulation is a time-domain Maxwell solve on a 3-D Yee grid. The MAPS wrapper constructs the FDTD-X scene from material, source, boundary, monitor, and grid metadata, then extracts frequency-domain phasors from the requested detectors.

The forward result is not a generic time-history tensor. The optimization-facing path typically consumes frequency-domain detector data.

## Phasor detector components

The runtime supports six electromagnetic components:

```text
Ex, Ey, Ez, Hx, Hy, Hz
```

The environment flag `MAPS_ENABLE_PHASOR_H` controls whether native magnetic phasors are recorded as detector outputs. When disabled, the implementation can derive magnetic phasors analytically from electric phasors for supported workflows, while retaining additional source-plane data as needed for validation.

The helper `phasor_detector_components()` therefore returns either all six components or the electric subset.

## Native H-field slab

`native_h_slab_cells()` exposes the configured number of native H-field slab cells used when retaining magnetic data directly from FDTD-X. The current default is controlled by `MAPS_PHASOR_H_NATIVE_SLAB_CELLS`.

This is a memory/runtime configuration, not a physical boundary thickness.

## Symmetry unfolding

`unfold_array()` mirrors an array along each spatial axis selected by a symmetry tuple. A nonzero symmetry entry causes a flipped copy to be concatenated with the original along the corresponding spatial axis. Optional sign tensors implement odd/even field transformations.

Conceptually:

```text
symmetry == (0, 0, 0)  -> unchanged
symmetry[a] != 0        -> concat([flip(array_a) * sign_a, array_a], axis=a)
```

Because the operation uses Torch tensor operations, the unfolding remains differentiable.

## Adjoint group representation

The adjoint bridge can construct compact gradient-support groups from monitor metadata. Each group identifies:

```text
group_id
monitor_slice = (x, y, z)
freq_indices
```

The helper reduces full gradient support to the spatial points/frequency channels that are active above the configured threshold.

This representation is primarily an optimization for memory and adjoint scheduling. It must map back to the same physical simulation coordinates as the original monitor slices.

## Gradient scaling

The solver contains a numerical gradient scaling factor because FDTD-X material derivatives and optimization-variable gradients can otherwise differ greatly in magnitude. Documentation should distinguish:

```text
physical derivative d objective / d epsilon_r
optimization gradient d objective / d parameter
```

from an arbitrary implementation scaling constant.

When validating a gradient path, compare consistently normalized quantities and document any conversion factor.

## Broadband adjoint

`MAPS_ENABLE_BROADBAND_ADJOINT` enables an experimental grouped/broadband adjoint path. The implementation warns that this mode may produce incorrect gradients unless the spatial/profile assumptions of the grouped sources and broadband objective are satisfied.

This feature should therefore be documented as experimental, not as the default production adjoint path.

## Sparse phasor recording

`MAPS_SPARSE_PHASOR=1` changes where FDTD-X records phasors, but it does not
change the Torch-facing dense field/autograd contract. Forward simulations place
localized detectors at design regions and objective ports. Unless disabled with
`MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES=0`, they also place three orthogonal
design-centered slabs for 3-D cross-section plotting. Devices without a design
region use one three-plane set centered at physical origin. After source-spectrum
normalization and symmetry unfolding, their values are scattered into:

```text
[Nfreq, 6, Nx, Ny, Nz]
```

with component order `(Ex, Ey, Ez, Hx, Hy, Hz)` and zero fill outside recorded
support. PyTorch backward therefore still receives a dense cotangent of the same
shape.

Adjoint simulations place only localized design-region phasor detectors. Their
states are unfolded/scattered into dense adjoint fields before calculating the
permittivity gradient from forward and adjoint electric fields. No full-volume
placement/reference phasor detector is run in sparse mode.

`MAPS_SPARSE_ADJOINT_SOURCE` is independent of detector sparsity. Set it to `0`
to retain the validated dense-adjoint-source protocol while using sparse phasor
recording. See [FDTD differentiable solver](fdtd-solver.md#sparse-phasor-mode)
for the complete runtime and symmetry contract.

## Debug/validation controls

`MAPS_FDTDX_ADJOINT_DEBUG` enables adjoint debug output and uses `MAPS_FDTDX_ADJOINT_DEBUG_DIR` for diagnostic artifacts.

`MAPS_COMPARE_EXACT_INTERPOLATION` enables comparison between Torch colocation and native exact detector phasors.

These flags are validation/debug controls and should not be required for normal forward simulation.

## Interactive Tidy3D field visualization

`fdtd3d.solve()` continues to return its six phasor tensors in the order
`(Ex, Ey, Ez, Hx, Hy, Hz)`. To visualize a solved result, pass that tuple and an
existing Tidy3D `Simulation` containing the desired structures to
`fdtd3d.plot_3d(fields=..., simulation=..., field="Ey", wavelength=1.55e-6,
quantity="abs", plane=("z", 0.0))`. The solver instance also provides
`to_tidy3d_data()` when callers need the native Tidy3D `SimulationData` object
without rendering it.

The adapter expects output fields with shape `(frequency, x, y, z)`. It uses the
grid recorded by the latest solve or an explicitly selected native/export grid;
shape inference is used only when it is unambiguous. It preserves MAPS Tidy3D
geometry/data coordinates in micrometers; the fallback grid derived from `dL`
is converted from meters to micrometers. Public `plane`/`planes` positions and
`wavelength` values remain in meters; positions are converted to micrometers at
the plotting boundary.
Tensors returned by `solve()` have already received temporal correction, Yee
colocation, symmetry unfolding, and optional export resampling. The adapter does
not apply another field offset or interpolation. Tidy3D field data are stored in
`(x, y, z, frequency)` order and retain their complex values.

`to_tidy3d_data(grid="auto")` uses the grid recorded by the latest solver call.
For externally supplied fields, it infers a grid only when shape identifies a
single coordinate set (or native and export coordinates are identical). If
native and export grids have the same shape but different coordinates, callers
must pass `grid="native"` or `grid="export"` explicitly. By default, the
adapter uses exact frequencies from the latest solve; if unavailable it converts
the configured wavelength band from micrometers to meters before computing Hz.

Tidy3D 2.11.2 exposes `Simulation.plot_3d()` as a hosted geometry viewer; it does
not expose a Python field-overlay renderer. MAPS therefore returns a small
Matplotlib/ipywidgets notebook view that draws Tidy3D `Box` and straight,
z-extruded, zero-sidewall `PolySlab` structures with a movable field plane. It
supports component or `E`/`H` magnitude selection,
real/imaginary/absolute/squared-absolute/phase quantities, wavelength choice,
multiple planes, opacity, and geometry/field visibility. Interactive camera
rotate, pan, and zoom require the ipympl Jupyter backend (for example,
`%matplotlib widget`); the adapter selects its widget canvas automatically.
`CustomMedium` structures are omitted rather than shown as their misleading
bounding boxes. Other Tidy3D geometry, nonzero-sidewall slabs, and slabs with
bulged edges are also omitted by this fallback renderer. Install the optional
`notebook-visualization` dependencies (`ipympl` and `ipywidgets`) to use the
interactive field view. The generated `FieldMonitor` records metadata only;
Tidy3D flux and integration calculations on adapted MAPS data are not validated.

## Inverse-design geometry materialization

`BaseOptimization.to_tidy3d_geometry_simulation()` converts the current filled
high-resolution design map into visualization-only `PolySlab` structures and
combines them with the existing fixed Tidy3D geometry. It also converts the
initialized physical port planes into native `ModeSource` and `ModeMonitor`
objects. Auxiliary source-field detectors, design-region detectors, and
full-domain phasor detectors are intentionally excluded. `BaseOptimization.plot_3d()`
opens that simulation in Tidy3D's native geometry viewer. Both methods use the
latest cached `_hr_eps_map` when available; otherwise they evaluate the current
design parameters through `build_device()` without running FDTD.

The default threshold is the physical midpoint between the design region's
`eps_bg` and `eps`, not the observed minimum and maximum of an intermediate
optimization map. Exact mode unions thresholded cells using the stored MAPS
boundary arrays, preserving nonuniform cell extents. Smooth mode delegates
contour interpolation to Tidy3D's public `Simulation.to_gdstk()` path. The
resulting polygons use the design region's actual z bounds and the configured
`extrude_angle`, converted from MAPS degrees to Tidy3D radians. This conversion
does not alter the material map used by FDTD.

After optimization, `BaseOptimization.save_tidy3d_geometry(path)` writes the
materialized geometry and physical source/port-monitor visualization metadata as
a self-contained Tidy3D HDF5 simulation. Load it in a later process with
`core.visualization.load_device_visualization_simulation(path)` and call
`plot_3d()`; loading the artifact does not reconstruct the optimization model,
perform a normalization run, or execute FDTD. The source/monitor objects are
visualization metadata, not stored fields or a runnable replacement for the MAPS
simulation. The artifact preserves the chosen thresholded geometry but not the
original continuous optimization tensor, so changing the threshold later still
requires a checkpoint or saved epsilon map.

For local cross-section inspection,
`core.visualization.plot_tidy3d_slices(simulation)` provides independent x, y,
and z sliders for native Tidy3D YZ, XZ, and XY plots. It redraws only after a
slider is released and includes per-structure plus source/monitor visibility
controls shared by all three slices. These controls do not mutate the saved
simulation. The widget is separate from the hosted `plot_3d()` iframe, whose
visibility and clipping state are not exposed to Python.

See [the notebook example](../notebooks/tidy3d_field_visualization.ipynb) for a
compact end-to-end example.

## Correctness requirements

A FDTD-X solver change should be validated at three levels:

1. **Forward physics:** fields/phasors agree with trusted FDTD-X behavior.
2. **Representation:** Torch-facing arrays have the expected component order, shapes, coordinates, and normalization.
3. **Differentiation:** gradients agree with a trusted adjoint/autodiff reference on representative designs.
