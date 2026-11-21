"""Physical detector apertures and native port registration contracts."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fdtdx")

from core.fdtd.fdtd import detector_native_placement, fdtd3d
from thirdparty.fdtdx import fdtdx

pytestmark = [pytest.mark.unit, pytest.mark.fdtdx]


@pytest.fixture
def detector_grid():
    """Provide independently specified edges with finer cells at the origin."""
    import jax.numpy as jnp

    edges = tuple(
        np.asarray(axis) * 1e-6
        for axis in (
            [-3, -2.5, -2, -1, 0, 1, 2, 3],
            [-3, -2, -1, 0, 1, 2, 3],
            [-1, -0.5, 0, 0.5, 1],
        )
    )
    grid = fdtdx.RectilinearGrid.custom(
        x_edges=jnp.asarray(edges[0]),
        y_edges=jnp.asarray(edges[1]),
        z_edges=jnp.asarray(edges[2]),
    )
    return edges, grid


def test_sparse_detector_preserves_explicit_cell_count(detector_grid):
    """An explicit multi-cell axis must not collapse to a single nearest cell."""
    edges, grid = detector_grid
    detector = fdtdx.PhasorDetector(
        name="mixed",
        partial_grid_shape=(3, None, 1),
        partial_real_shape=(None, 4e-6, None),
        partial_real_position=(0.5e-6, 0.0, 0.25e-6),
        wave_characters=(fdtdx.WaveCharacter(wavelength=1.55e-6),),
    )
    simulation = SimpleNamespace(
        shape=grid.shape,
        _native_boundaries=lambda: edges,
        _simulation_grid=lambda: grid,
    )
    simulation._monitor_real_bounds = MethodType(
        fdtd3d._monitor_real_bounds, simulation
    )
    assert fdtd3d._monitor_grid_bounds(simulation, detector) == ((3, 6), (1, 5), (2, 3))


def test_sparse_detector_partial_position_keeps_physical_axes(detector_grid):
    """An unspecified axis spans the domain without discarding other apertures."""
    edges, grid = detector_grid
    detector = fdtdx.PhasorDetector(
        name="partial",
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(4e-6, None, None),
        partial_real_position=(0.0, None, 0.25e-6),
        wave_characters=(fdtdx.WaveCharacter(wavelength=1.55e-6),),
    )
    simulation = SimpleNamespace(
        shape=grid.shape,
        _native_boundaries=lambda: edges,
        _simulation_grid=lambda: grid,
        _full_volume_bounds=lambda: tuple((0, n) for n in grid.shape),
    )
    simulation._monitor_real_bounds = MethodType(
        fdtd3d._monitor_real_bounds, simulation
    )
    assert fdtd3d._monitor_grid_bounds(simulation, detector) == ((2, 6), (0, 6), (2, 3))


@pytest.mark.parametrize("symmetry", [(0, 0, 0), (0, 1, 0)])
def test_native_port_registration_uses_location_dependent_extent(
    detector_grid, symmetry
):
    """A centered four-micrometer port covers [-2, 2], including with symmetry."""
    from core.invdes.models.layers.device_base import N_Ports

    edges, grid = detector_grid
    metadata = dict(
        shape=grid.shape,
        boundaries=tuple(axis * 1e6 for axis in edges),
        coords=tuple((axis[:-1] + axis[1:]) * 0.5e6 for axis in edges),
    )
    device = SimpleNamespace(
        dim=3,
        axes="xyz",
        epsilon_map=np.ones(grid.shape),
        grid_info_dict=dict(epsilon_map=metadata, export_epsilon_map=metadata),
        sim_cfg=dict(symmetry=symmetry),
        port_monitor_slices={},
        port_monitor_slices_info={},
        port_monitor_slices_native={},
        port_monitor_slices_native_symmetry={},
        port_monitor_slices_export={},
        _check_object_on_symmetry_axis=lambda **kwargs: True,
    )
    device._monitor_slice_on_coords = MethodType(
        N_Ports._monitor_slice_on_coords, device
    )
    N_Ports.add_monitor_slice(
        device, "output", center=(0, 0, 0.25), size=(4, 4, 0), direction="z-"
    )
    expected = (slice(2, 6), slice(1, 5), slice(2, 3))
    assert device.port_monitor_slices_native["output"] == expected
    assert device.port_monitor_slices_export["output"] == expected


@pytest.mark.parametrize("kind", ["phasor", "mode", "field", "energy", "flux"])
@pytest.mark.parametrize("normal_axis", [0, 1, 2])
def test_physical_detector_types_cover_aperture_at_actual_location(
    detector_grid, kind, normal_axis
):
    """Every detector family shares the same location-aware geometry contract."""
    import jax
    import jax.numpy as jnp

    original_edges, _ = detector_grid
    permutation = [(normal_axis + 1) % 3, (normal_axis + 2) % 3, normal_axis]
    edges = tuple(original_edges[permutation.index(axis)] for axis in range(3))
    grid = fdtdx.RectilinearGrid.custom(
        x_edges=jnp.asarray(edges[0]),
        y_edges=jnp.asarray(edges[1]),
        z_edges=jnp.asarray(edges[2]),
    )
    counts = tuple(1 if axis == normal_axis else None for axis in range(3))
    extents = tuple(None if axis == normal_axis else 4e-6 for axis in range(3))
    center = tuple(0.25e-6 if axis == normal_axis else 0 for axis in range(3))
    kwargs = dict(
        name="detector",
        partial_grid_shape=counts,
        partial_real_shape=extents,
        partial_real_position=center,
    )
    if kind in ("mode", "phasor"):
        kwargs["wave_characters"] = (fdtdx.WaveCharacter(wavelength=1.55e-6),)
    if kind in ("mode", "flux"):
        kwargs["direction"] = "-"
    cls = dict(
        phasor=fdtdx.PhasorDetector,
        mode=fdtdx.ModeOverlapDetector,
        field=fdtdx.FieldDetector,
        energy=fdtdx.EnergyDetector,
        flux=fdtdx.PoyntingFluxDetector,
    )[kind]
    prototype = cls(**kwargs)
    detector, constraints = detector_native_placement(prototype, edges, grid)
    volume = fdtdx.SimulationVolume(
        name="volume",
        partial_grid_shape=grid.shape,
        material=fdtdx.Material(permittivity=1),
    )
    objects, *_ = fdtdx.place_objects(
        object_list=[volume, detector],
        constraints=constraints,
        config=fdtdx.SimulationConfig(grid=grid, time=1e-14, dtype=jnp.float32),
        key=jax.random.PRNGKey(0),
        apply_objects=False,
    )
    bounds = objects[detector.name].grid_slice
    assert bounds[permutation[0]] == slice(2, 6)
    assert bounds[permutation[1]] == slice(1, 5)
    assert bounds[normal_axis] == slice(2, 3)
    assert prototype.partial_grid_shape == counts
    assert prototype.partial_real_position == center


def test_explicit_recorder_support_overrides_physical_metadata(detector_grid):
    """A recorder's padded support wins over its old physical center and extent."""
    edges, grid = detector_grid
    prototype = fdtdx.PhasorDetector(
        name="padded",
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        partial_real_position=(0.0, 0.0, 0.0),
        wave_characters=(fdtdx.WaveCharacter(wavelength=1.55e-6),),
    )
    detector, _ = detector_native_placement(
        prototype, edges, grid, bounds=((0, 7), (1, 5), (1, 4))
    )
    assert detector.partial_grid_shape == (7, 4, 3)
    assert detector.partial_real_position == (None, None, None)
    assert prototype.partial_real_position == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("cls", [fdtdx.FieldDetector, fdtdx.EnergyDetector])
def test_volume_detector_resolves_all_three_physical_extents(detector_grid, cls):
    """A volume detector has no implicit one-cell normal-axis restriction."""
    edges, grid = detector_grid
    prototype = cls(
        name="volume_detector",
        partial_real_shape=(4e-6, 4e-6, 1e-6),
        partial_real_position=(0.0, 0.0, 0.0),
    )
    detector, _ = detector_native_placement(prototype, edges, grid)
    assert detector.partial_grid_shape == (4, 4, 2)


@pytest.mark.parametrize("bounds", [((0, 8), (0, 6), (0, 4)), ((0, 0), (0, 6), (0, 4))])
def test_invalid_explicit_recorder_bounds_raise(detector_grid, bounds):
    """Empty and out-of-domain supports must fail before array initialization."""
    edges, grid = detector_grid
    prototype = fdtdx.EnergyDetector(name="invalid")
    with pytest.raises(ValueError, match="Invalid detector bounds"):
        detector_native_placement(prototype, edges, grid, bounds=bounds)
