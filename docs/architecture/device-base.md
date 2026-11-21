# Device Base Architectural Map

`core/invdes/models/layers/device_base.py` is currently the main integration layer between physical device descriptions and simulation/optimization infrastructure.

It should be read as an orchestration layer with several coupled state machines rather than as a collection of independent utility functions.

## State lifecycle

```text
constructor inputs
    |
    v
infer dimensionality
    |
    v
build backend geometry
    |
    v
establish cell extent
    |
    v
recenter geometry
    |
    v
rasterize optical epsilon
    |
    +--> native grid metadata
    +--> export grid metadata
    |
    v
build design/active/port masks
    |
    v
build monitor slices
    |
    v
build source/mode profiles
    |
    v
normalization run
    |
    v
simulation wrappers
    |
    +--> FDFD
    +--> FDTD-X
    +--> thermal/electrical
```

## Primary state groups

### Geometry state

```text
port_cfgs
geometry_cfgs
design_region_cfgs
active_region_cfgs
override_structures
geometry
cell_size
cell_extend
cell_center
```

### Optical-grid state

```text
grid_shape
Nx / Ny / Nz
coords
meshes
xs / ys
native_epsilon_map
epsilon_map
export_epsilon_map
grid_info_dict
optical_grid_metadata
export_grid_metadata
```

### Region state

```text
design_region_masks
design_region_mask_weights
design_region_axis_weights
active_region_masks
ports_regions
```

### Port/monitor state

```text
port_monitor_slices
port_monitor_slices_native
port_monitor_slices_export
port_monitor_slices_native_symmetry
port_monitor_slices_info
```

### Source state

```text
port_sources_dict
```

### Multiphysics state

```text
conductivity_map
heat_capacity_map
thermo_optic_coeff_map
electrical_conductivity_map
heat_source_map
thermal_coords
thermal_grid_shape
thermal_grid_spacing
```

## Coordinate graph

The most important relationship is:

```text
physical coordinate
       |
       +--> geometry
       +--> native optical coordinate
       +--> export optical coordinate
       +--> thermal coordinate
       +--> monitor physical metadata
       |
       v
integer grid indices
```

A grid index is meaningful only relative to its corresponding coordinate set. This is why the code maintains explicit grid metadata and uses coordinate-aware resampling.

## Port/source graph

```text
port_cfgs
   |
   v
port geometry
   |
   v
port monitor slice
   |
   +--> source plane location
   +--> modal profile location
   +--> field monitor location
   |
   v
auxiliary two-port normalization
   |
   v
normalized source profile
   |
   v
port_sources_dict
   |
   v
simulation
```

The same physical port can therefore have several representations: geometry, export monitor slice, native monitor slice, mode profile, and normalized source field.

## Objective coupling

The objective layer consumes the state produced here:

```text
port_sources_dict
port_monitor_slices
port_monitor_slices_native
port_monitor_slices_info
epsilon_map
fdtdx_native_grid_metadata
fdtdx_native_cell_weights
```

and the solver returns fields keyed by the source context.

This means that a seemingly local change to monitor indexing or source normalization can change objective results even when geometry and solver code are untouched.

## FDTD-X-specific split

FDTD-X introduces a second internal grid representation:

```text
export optical grid
       |
       | coordinate-aware resample
       v
FDTD-X native optical grid
       |
       +--> native design-region masks
       +--> native monitor slices
       +--> adjoint grouping
       +--> FDTD-X field grid metadata
```

This path should not be collapsed into the 2D Ceviche indexing model during future refactors.

## Thermal coupling

Thermal state follows a parallel but not identical path:

```text
optical design density
       |
       v
optical epsilon/material map
       |
       +--> thermo-optic coefficient
       +--> electrical conductivity
       +--> thermal conductivity
       +--> heat capacity
       |
       v
thermal grid
       |
       v
heat solver
       |
       v
temperature / runtime material state
       |
       v
optical permittivity update
```

The thermal grid can differ from the optical grid, so the coupling is a physical-coordinate mapping problem rather than a tensor-shape-copy problem.

## Future refactor boundary

A future decomposition should preserve these directional dependencies:

```text
Geometry -> Grid -> Masks -> Ports/Monitors -> Sources -> Solvers -> Objectives
                  \-> Material maps -> Thermal solver -> Runtime optical state
```

Avoid making lower-level components import higher-level objective/optimization logic. The existing file contains some reverse coupling; refactoring should remove that only with tests and a compatibility layer, not by silently changing behavior.
