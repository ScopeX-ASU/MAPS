# Configuration-Driven Build Flow

MAPS uses configuration objects as the connective tissue between geometry, simulation, optimization, machine-learning training, and post-processing. This document records the current `v0.0.2_fdtdx` organization so that future refactoring can preserve the experiment-level contract.

## Two build domains

The repository currently has two overlapping but distinct construction domains:

```text
ML / surrogate / ILT training
    configs
      -> dataset
      -> model
      -> optimizer
      -> scheduler
      -> criterion

Physics-based inverse design
    device + simulation config
      -> N_Ports / BaseDevice
      -> design-region parameterization
      -> simulation backend
      -> objective / FoM
      -> InvDesign optimizer loop
```

`core/builder.py` is primarily the first path. The inverse-design-specific builder lives under `core/invdes/`.

## Global configuration versus local configuration

The ML builder reads the global `pyutils.config.configs` object for most construction arguments. Inverse-design classes generally receive local `sim_cfg`, `obj_cfgs`, and design-region configuration mappings.

This means the same concept can appear in two forms:

```text
configs.model.sim_cfg
local sim_cfg passed to a device/model
```

When refactoring, do not assume these are interchangeable dictionaries. The existing code may rely on `Config` attribute access, nested update behavior, or a specific object identity/lifecycle.

## Device-first inverse-design flow

A typical physics-driven construction order is:

```text
Simulation configuration
        |
        v
N_Ports / device construction
        |
        +--> geometry configs
        +--> port configs
        +--> design-region configs
        +--> grid / epsilon map
        +--> design-region masks
        +--> port masks
        +--> monitor slices
        +--> source / mode profiles
        |
        v
BaseOptimization subclass
        |
        +--> parameterization modules
        +--> objective layer
        +--> material-property maps
        |
        v
InvDesign
        |
        +--> optimizer
        +--> scheduler
        +--> callbacks
        |
        v
forward simulation -> objective -> backward/adjoint -> parameter update
```

This sequence is deliberate. For example, source normalization depends on port/monitor definitions, while objective construction depends on the resulting source and monitor profile dictionaries.

## Canonical data handoffs

Several mappings function as de facto stable interfaces even though they are plain Python dictionaries.

### Device -> objective

Important device-level structures include:

```text
port_sources_dict
port_monitor_slices
port_monitor_slices_native
port_monitor_slices_native_symmetry
port_monitor_slices_info
fdtdx_native_grid_metadata
fdtdx_native_cell_weights
epsilon_map
```

### Parameterization -> optimization

Design-region modules expose:

```text
parameters / parameter modules
normalized density
physical permittivity
optional runtime material maps
```

### Objective -> InvDesign

The optimization model returns at least:

```text
results["obj"]
results["breakdown"]
```

and `InvDesign` applies the optimization sign convention, gradient processing, and callbacks.

## Resolution layers

MAPS commonly distinguishes:

```text
simulation/native optical grid
export optical grid
high-resolution design grid
thermal grid
fixed/unstructured thermal mesh
```

These are not merely different tensor resolutions. They can have different coordinates, cell boundaries, masks, physical averaging rules, and downstream consumers.

The shared-grid documentation should be consulted before adding interpolation or resizing code.

## Source and mode lifecycle

For a mode-based 2D optical workflow, the conceptual sequence is:

```text
port geometry
  -> monitor slice
  -> auxiliary two-port device
  -> eigenmode insertion
  -> normalization solve
  -> source field scaled to target power
  -> source profile dictionary
```

For 3D FDTD-X:

```text
port geometry
  -> native/export monitor slices
  -> fdtdx mode or Gaussian source object
  -> phasor detector profile
  -> broadband source/monitor dictionary
  -> differentiable FDTD simulation
```

The dictionaries are intentionally different because FDFD solves each wavelength separately while FDTD-X can represent a wavelength group as one broadband simulation.

## Experiment configuration as an API

For AI coding, treat experiment configuration files as API consumers.

A change to any of the following is potentially a breaking change even when Python function signatures stay unchanged:

- configuration section names;
- factory `name` dispatch strings;
- dictionary key names;
- tensor shape conventions;
- wavelength key formats;
- optimizer/scheduler parameter names;
- source/monitor naming conventions.

When refactoring, maintain a compatibility mapping or perform an intentional migration rather than silently changing the schema.

## Refactoring guidance

When `device_base.py` is eventually decomposed, preserve the current conceptual seams before changing ownership:

```text
Geometry description
  -> geometry backend
  -> raster/grid layer

Port description
  -> monitor slice construction
  -> mode/source construction
  -> normalization

Design region description
  -> masks / weights
  -> parameterization
  -> material-map insertion

Grid metadata
  -> native/export conversion
  -> objective sampling
  -> field integration

Thermal/material data
  -> thermal-grid mapping
  -> heat solver
  -> runtime multiphysics state
```

The code may be reorganized into separate classes/modules later, but these interfaces should remain explicit.

## AI-agent rule of thumb

Before changing a factory/configuration/data structure, trace both directions:

```text
producer -> consumer
consumer -> producer
```

A dictionary key that looks local may be read by a different subsystem several layers downstream. The safest documentation source is therefore the complete producer/consumer chain, not a single function signature.
