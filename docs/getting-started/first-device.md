# First Device: Constructing a Bending Device

This guide stops at the **device-construction boundary**. It uses the `Bending` device from the `v0.0.2_fdtdx` baseline and explains what is created before inverse-design optimization begins.

For the full optimization workflow, see [First inverse-design run: Bending](first-inverse-design.md). For the lower-level FDFD optimization/data-generation path, see [First FDFD workflow](first-fdfd-workflow.md).

## Minimal construction pattern

The tagged Bending workflow follows this pattern:

```python
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import Bending

sim_cfg = DefaultSimulationConfig()
sim_cfg.update(
    dict(
        solver="ceviche_torch",
        border_width=[0, 1.7, 1.7, 0],
        resolution=50,
        PML=[0.5, 0.5],
        neural_solver=None,
        numerical_solver="solve_direct",
        use_autodiff=False,
    )
)

device = Bending(
    sim_cfg=sim_cfg,
    bending_region_size=[1.72, 1.72],
    port_len=(1.7, 1.7),
    port_width=(0.48, 0.48),
    device="cpu",  # use the actual execution device in a real run
)
```

The numbers above illustrate the constructor contract. The canonical tagged script computes the bending-region size and port length together so that the physical geometry is aligned to the selected raster resolution; use the exact script/configuration for a reproducible experiment.

## 1. Simulation configuration

The device is constructed with a `DefaultSimulationConfig` object. The important fields for this FDFD example are:

```text
solver
resolution
PML
border_width
numerical_solver
neural_solver
use_autodiff
```

These fields describe how the device will later be simulated. They do not themselves define the geometry.

Keep the distinction clear:

```text
sim_cfg     -> simulation/grid/backend policy
Bending(...) -> physical device geometry and device state
```

## 2. Physical geometry

The Bending constructor receives the core geometry through:

```text
bending_region_size
port_len
port_width
```

The device layer interprets these in the physical coordinate system and derives cell boundaries/rasterized tensors from the configured resolution.

The canonical FDFD data-generation script first chooses:

```text
target image extent = 256 cells
resolution          = 50 cells per unit length
```

and then chooses a quantized port length. The design-region width/height is selected so that the ports plus bend exactly cover the intended optical extent.

This is an important MAPS design rule: **do not choose tensor dimensions independently of the physical geometry and grid resolution.**

## 3. The primary device state

After construction, the device is more than an epsilon image. The base device architecture maintains coordinated state for:

```text
physical geometry
material map / epsilon map
grid metadata
active-region masks
design-region masks
port geometry and masks
monitor slices
source/mode profiles
normalization data
native/export grid variants
backend-specific metadata
```

The exact attribute names vary by backend/device subclass; the behavioral contract is described in [Device API contracts](../api/device-base.md).

## 4. Epsilon / material map

The device creates a rasterized material representation on its configured simulation grid.

At the FDFD field boundary, the standard 2D representation is conceptually:

```text
eps_r.shape == [Nx, Ny]
```

with dimension meaning determined by the device's physical x/y grid.

`eps_r` is a **relative permittivity** representation, not an arbitrary image. Units are therefore dimensionless.

When high-resolution design is enabled later, a separate high-resolution device/map may have different `[Nx, Ny]` dimensions.

## 5. Grid metadata

The device owns grid information that maps physical coordinates to array indices. This may include:

```text
cell boundaries
cell centers
spacing
extent
cell weights
native/export grid metadata
```

Never infer physical coordinates from array indices without consulting the owning grid metadata.

This becomes especially important for nonuniform grids, partial-cell masks, thermal coupling, and FDTD-X native/export mappings.

See [Shared grid and field data](../architecture/shared-grid-and-field-data.md).

## 6. Active and design-region masks

A device can distinguish at least two conceptually different kinds of spatial masks:

### Active region

The region participating in device simulation/material modification according to the device geometry/configuration.

### Design region

The region where learnable inverse-design parameters are allowed to modify the material representation.

A design-region mask is therefore **not equivalent to the entire device mask**.

At the design layer, the mask may exist in both simulation and high-resolution representations:

```text
design_region_mask
hr_design_region_mask
```

The masks must refer to their respective grids.

## 7. Ports

A physical port is represented through several linked forms rather than a single tensor.

Conceptually:

```text
port geometry
   |
   +--> port mask
   +--> monitor slice
   +--> mode/source profile
   +--> normalization reference
```

The Bending example has two optical ports. In downstream objective/plotting paths the canonical names include:

```text
in_slice_1
```

for the forward input slice, with corresponding device-specific output/reflection representations.

## 8. Monitor slices

A port geometry is converted into a monitor selection on the simulation grid.

The conceptual form is a 3-axis selection even for a 2D optical device, because the general device API supports multiple backend dimensions and metadata structures.

A monitor/slice definition carries enough information to answer:

```text
Where is the monitor?
Which grid does it refer to?
What physical direction does the port represent?
Which transverse aperture is measured?
Which backend representation consumes it?
```

This is why monitor dictionaries should not be treated as anonymous `dict` objects during refactoring.

## 9. Mode and source profiles

The source is derived from a port/mode description rather than from a generic array alone.

A downstream FDFD objective/plot call can use a field key such as:

```text
("in_slice_1", 1.55, "Ez1", 300)
```

which identifies:

```text
input slice
wavelength
mode/polarization identifier
additional control/state metadata
```

The source/mode layer is also responsible for maintaining normalization information used by objective calculations.

See [Objective contracts](../api/objective.md) and [Device API contracts](../api/device-base.md).

## 10. Source normalization

MAPS' device layer can construct auxiliary normalization simulations/devices in order to determine source power and normalized modal quantities.

This means that source-profile construction is not purely a local operation on the final device geometry. It can depend on a calibrated reference problem.

As a result, future device refactoring must preserve the relationship between:

```text
source profile
normalization power
port mode
objective normalization
```

## 11. Native and export representations

A device may maintain multiple grid representations:

```text
native optical grid
export optical grid
FDTD-X native grid
thermal grid
```

For a simple uniform 2D FDFD example, native and export grids may coincide. More advanced workflows can intentionally separate them.

Do not use tensor shape equality as proof that two maps are physically aligned.

## 12. Inspecting the device before optimization

A useful inspection pattern is:

```python
print(device)
```

The canonical Bending script does exactly this immediately after construction.

For a more detailed debugging session, inspect the device according to the established contracts rather than reaching directly into private implementation state. Useful categories include:

```text
grid metadata
material/epsilon map
port definitions
monitor slices
design-region masks
source profiles
```

The exact debug attributes are implementation-specific and should be checked in the current device subclass before writing tooling against them.

## 13. What construction does not do

Constructing `Bending` does not by itself mean an inverse-design optimization has occurred.

The device-construction stage establishes the physical/simulation state that later layers consume:

```text
Bending device
      |
      +--> material map
      +--> grids
      +--> ports
      +--> monitors
      +--> source/mode data
      +--> design-region masks
           |
           v
BendingOptimization
           |
           v
FDFD forward/adjoint + objective
```

This separation is important for future refactoring: geometry/device state, simulation state, objective state, and optimization state should not be assumed to be interchangeable.

## Scientific invariants

When modifying or refactoring device construction, preserve:

- physical coordinate meaning of geometry parameters;
- rasterization/grid alignment;
- relative-permittivity semantics;
- design-region mask alignment with its owning grid;
- port names and monitor/source relationships;
- mode/polarization conventions;
- source normalization semantics;
- native/export grid coordinate mappings;
- backend-specific state needed by FDFD and FDTD-X;
- differentiability boundaries between device state and optimization parameters.

The current `device_base.py` implementation is intentionally broad. This guide documents its externally relevant behavior before any decomposition is attempted.

## Related documentation

- [Device API contracts](../api/device-base.md)
- [Device architecture](../architecture/device-base.md)
- [Shared grid and field data](../architecture/shared-grid-and-field-data.md)
- [First inverse-design run: Bending](first-inverse-design.md)
- [First FDFD workflow](first-fdfd-workflow.md)
