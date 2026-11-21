"""Physical source-placement contracts on rectilinear simulation grids."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fdtdx")

from core.fdtd.fdtd import (
    _bounds_indices_to_real_bounds,
    _real_bounds_constraints,
    _source_has_physical_size,
    fdtd3d,
    real_box_to_grid_slices,
)
from thirdparty.fdtdx import fdtdx

pytestmark = [pytest.mark.unit, pytest.mark.fdtdx]


def _plane_source(kind, normal_axis, direction, real_shape, position):
    """Build a plane source with a physical aperture and one native normal cell.

    Args:
        kind: Gaussian, uniform, mode, or hard source selector.
        normal_axis: Axis with one normal cell.
        direction: Positive or negative propagation direction.
        real_shape: Three physical extents in meters, with a zero normal size.
        position: Source center in meters.

    Returns:
        Unplaced FDTD-X source prototype.
    """
    from fdtdx.objects.sources.source import HardConstantAmplitudePlanceSource

    shape = [None, None, None]
    shape[normal_axis] = 1
    physical = list(real_shape)
    physical[normal_axis] = None
    kwargs = dict(
        name="source",
        partial_grid_shape=tuple(shape),
        partial_real_shape=tuple(physical),
        partial_real_position=position,
        wave_character=fdtdx.WaveCharacter(wavelength=1.31e-6),
        direction=direction,
    )
    polarization = tuple(float(axis == (normal_axis + 1) % 3) for axis in range(3))
    if kind == "mode":
        return fdtdx.ModePlaneSource(mode_index=0, **kwargs)
    kwargs["fixed_E_polarization_vector"] = polarization
    if kind == "gaussian":
        return fdtdx.GaussianPlaneSourceTidy3d(waist_radius=10e-6, **kwargs)
    if kind == "native_gaussian":
        return fdtdx.GaussianPlaneSource(radius=10e-6, **kwargs)
    if kind == "hard":
        return HardConstantAmplitudePlanceSource(**kwargs)
    return fdtdx.UniformPlaneSource(**kwargs)


@pytest.mark.parametrize("length_scale", [1.0, 1e-6, 1e-9])
def test_meter_aperture_does_not_snap_across_an_overlapping_thin_cell(length_scale):
    """A nanometer-scale overlap must not disappear under an absolute tolerance."""
    edges = (
        np.array([-1.0, -0.995, 0.0, 1.0]) * length_scale,
        np.array([-1.0, 0.0, 1.0]) * length_scale,
        np.array([-1.0, 0.0, 1.0]) * length_scale,
    )
    bounds = real_box_to_grid_slices(
        edges,
        (0.0, 0.0, 0.5 * length_scale),
        (1.998 * length_scale, length_scale, 0.0),
        "z-",
    )
    assert bounds == (slice(0, 3), slice(0, 2), slice(1, 2))


@pytest.mark.parametrize("length_scale", [1.0, 1e-6, 1e-9])
def test_edge_roundoff_does_not_add_a_boundary_cell(length_scale):
    """Decimal edge roundoff must not enlarge an exactly aligned aperture."""
    edges = (
        np.array([-1.0, 0.0, 1.0]) * length_scale,
        np.linspace(-1.2, 1.2, 13) * length_scale,
        np.linspace(-1.2, 1.2, 13) * length_scale,
    )
    bounds = real_box_to_grid_slices(
        edges, (0.0, 0.0, 0.0), (0.0, 1.6 * length_scale, 1.6 * length_scale), "x+"
    )
    assert bounds[1:] == (slice(2, 10), slice(2, 10))


@pytest.mark.parametrize("width_um, expected", [(2.0, (4, 8)), (4.0, (2, 10))])
@pytest.mark.parametrize("kind", ["gaussian", "mode", "uniform"])
def test_uniform_source_aperture_retains_exact_physical_width(width_um, expected, kind):
    """Aligned uniform-grid apertures keep the legacy placement contract."""
    import torch

    shape = (12, 12, 4)
    grid = fdtdx.RectilinearGrid.uniform(shape=shape, spacing=0.5e-6)
    simulation = SimpleNamespace(
        eps_r=torch.ones(shape),
        _simulation_grid=lambda: grid,
        _uses_nonuniform_grid=lambda: False,
    )
    source = _plane_source(
        kind, 2, "-", (width_um * 1e-6, width_um * 1e-6, 0.0), (0.0, 0.0, 0.25e-6)
    )
    bounds = fdtd3d.obtain_source_native_grid_slice(
        simulation,
        source,
        (0.0, 0.0, 0.25e-6),
        (width_um * 1e-6, width_um * 1e-6, 0.0),
    )
    assert bounds[:2] == (slice(*expected), slice(*expected))
    for axis in (0, 1):
        edges = np.asarray(grid.edges(axis))
        assert edges[bounds[axis].stop] - edges[bounds[axis].start] == pytest.approx(
            width_um * 1e-6
        )


@pytest.mark.parametrize(
    "kind", ["gaussian", "native_gaussian", "uniform", "mode", "hard"]
)
@pytest.mark.parametrize("normal_axis", [0, 1, 2])
@pytest.mark.parametrize("direction", ["+", "-"])
def test_nonuniform_source_aperture_uses_edges_at_source_location(
    kind, normal_axis, direction
):
    """A four-micrometer aperture must cover its physical bounds, not origin cells."""
    import jax
    import jax.numpy as jnp
    import torch

    edges = (
        np.array([-3.0, -2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) * 1e-6,
        np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) * 1e-6,
        np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * 1e-6,
    )
    permutation = [(normal_axis + 1) % 3, (normal_axis + 2) % 3, normal_axis]
    edges = tuple(edges[permutation.index(axis)] for axis in range(3))
    center = tuple(0.25e-6 if axis == normal_axis else 0.0 for axis in range(3))
    size = tuple(0.0 if axis == normal_axis else 4e-6 for axis in range(3))
    expected = [None, None, None]
    expected[permutation[0]] = slice(2, 6)
    expected[permutation[1]] = slice(1, 5)
    expected[normal_axis] = slice(2, 3)
    shape = tuple(len(axis) - 1 for axis in edges)
    grid = fdtdx.RectilinearGrid.custom(
        x_edges=jnp.asarray(edges[0]),
        y_edges=jnp.asarray(edges[1]),
        z_edges=jnp.asarray(edges[2]),
    )
    simulation = SimpleNamespace(
        eps_r=torch.ones(shape),
        _simulation_grid=lambda: grid,
        _uses_nonuniform_grid=lambda: True,
        _native_boundaries=lambda: edges,
    )
    source = _plane_source(kind, normal_axis, direction, size, center)
    bounds = fdtd3d.obtain_source_native_grid_slice(simulation, source, center, size)
    assert bounds == tuple(expected)
    for axis in permutation[:2]:
        assert edges[axis][bounds[axis].stop] - edges[axis][
            bounds[axis].start
        ] == pytest.approx(4e-6)

    native_bounds = tuple((axis.start, axis.stop) for axis in bounds)
    placed_source = fdtd3d._object_with_native_shape(simulation, source, native_bounds)
    placed_source = placed_source.aset("partial_real_position", (None, None, None))
    volume = fdtdx.SimulationVolume(
        name="volume",
        partial_grid_shape=shape,
        material=fdtdx.Material(permittivity=1.44**2),
    )
    constraints = _real_bounds_constraints(
        placed_source, _bounds_indices_to_real_bounds(native_bounds, edges)
    )
    objects, *_ = fdtdx.place_objects(
        object_list=[volume, placed_source],
        config=fdtdx.SimulationConfig(grid=grid, time=1e-14, dtype=jnp.float32),
        constraints=constraints,
        key=jax.random.PRNGKey(0),
        apply_objects=False,
    )
    assert objects[source.name].grid_slice == bounds
    assert source.partial_grid_shape[normal_axis] == 1
    assert all(source.partial_grid_shape[axis] is None for axis in permutation[:2])
    assert source.partial_real_position == center


@pytest.fixture
def nonuniform_simulation():
    """Provide known native edges for physical, mixed, and grid-sized sources."""
    import jax.numpy as jnp
    import torch

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
    return SimpleNamespace(
        eps_r=torch.ones(grid.shape),
        _simulation_grid=lambda: grid,
        _uses_nonuniform_grid=lambda: True,
        _native_boundaries=lambda: edges,
    )


def test_nonuniform_volume_source_covers_all_physical_axes(nonuniform_simulation):
    """A physical TFSF box resolves every extent without a plane-axis assumption."""
    from fdtdx.objects.sources.tfsf_region import TFSFPlaneSourceRegion

    source = TFSFPlaneSourceRegion(
        name="region",
        partial_real_shape=(4e-6, 4e-6, 1e-6),
        partial_real_position=(0.0, 0.0, 0.0),
        propagation_axis=2,
        direction="+",
        fixed_E_polarization_vector=(1.0, 0.0, 0.0),
        wave_character=fdtdx.WaveCharacter(wavelength=1.31e-6),
    )
    bounds = fdtd3d.obtain_source_native_grid_slice(
        nonuniform_simulation, source, (0.0, 0.0, 0.0), (4e-6, 4e-6, 1e-6)
    )
    assert bounds == (slice(2, 6), slice(1, 5), slice(1, 3))


def test_nonuniform_mixed_source_preserves_explicit_cell_count(nonuniform_simulation):
    """Explicit cell counts take precedence on mixed physical/native sources."""
    source = _plane_source("mode", 2, "-", (4e-6, 4e-6, 0.0), (0.0, 0.0, 0.25e-6))
    source = source.aset("partial_grid_shape", (3, None, 1))
    assert _source_has_physical_size(source)
    bounds = fdtd3d.obtain_source_native_grid_slice(
        nonuniform_simulation, source, (0.0, 0.0, 0.25e-6), (4e-6, 4e-6, 0.0)
    )
    assert bounds == (slice(2, 5), slice(1, 5), slice(2, 3))


def test_grid_sized_adjoint_source_retains_native_domain(nonuniform_simulation):
    """Distributed native currents must not be resized from redundant physical extents."""
    import jax.numpy as jnp

    from thirdparty.fdtdx import DistributedCurrentSource

    shape = nonuniform_simulation._simulation_grid().shape
    wave = fdtdx.WaveCharacter(wavelength=1.31e-6)
    source = DistributedCurrentSource(
        name="adjoint",
        partial_grid_shape=shape,
        partial_real_shape=(4e-6, 4e-6, 1e-6),
        wave_character=wave,
        wave_characters=(wave,),
        electric_profiles=jnp.zeros((1, 3, *shape), dtype=jnp.complex64),
    )
    assert not _source_has_physical_size(source)
    bounds = fdtd3d.obtain_source_native_grid_slice(
        nonuniform_simulation, source, (0.0, 0.0, 0.0), (4e-6, 4e-6, 1e-6)
    )
    assert bounds == tuple(slice(0, n) for n in shape)
