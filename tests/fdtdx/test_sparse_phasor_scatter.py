"""Unit contracts for scattering localized FDTD-X phasors."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fdtdx")
torch = pytest.importorskip("torch")
jnp = pytest.importorskip("jax.numpy")

import core.fdtd.fdtd as fdtd_module
from core.fdtd.fdtd import fdtd3d
from core.fdtd.solver import (
    _build_separable_broadband_adjoint_source,
    _pullback_dense_cotangents_autodiff_reference,
    _pullback_dense_cotangents_like_reference,
    _sparse_phasors_to_dense,
    clone_phasor_detector_for_waves,
)
from core.utils import (
    _design_region_visualization_bounds,
    _origin_grid_cell_bounds,
    unfold_phasor,
    yee_to_colocate_interpolate,
)
from thirdparty.fdtdx.fdtd.initialization import resolve_object_constraints

pytestmark = [pytest.mark.fdtdx, pytest.mark.unit]


def _detector(name, grid_slice, components=("Ex",)):
    return SimpleNamespace(
        name=name,
        grid_slice=grid_slice,
        components=components,
    )


def _monitor():
    return SimpleNamespace(wave_characters=(object(),))


def test_uniform_sparse_monitor_uses_exact_half_open_grid_bounds() -> None:
    """Uniform sparse monitors retain exact integer bounds during placement."""
    volume = fdtd_module.fdtdx.SimulationVolume(
        name="volume",
        partial_grid_shape=(20, 30, 40),
        material=fdtd_module.fdtdx.Material(permittivity=1.0),
    )
    monitor = fdtd_module.fdtdx.UniformMaterialObject(
        name="monitor",
        partial_grid_shape=(3, 12, 5),
        material=fdtd_module.fdtdx.Material(permittivity=1.0),
    )
    expected_bounds = ((4, 7), (5, 17), (10, 15))
    simulation = SimpleNamespace(
        _sparse_monitor_bounds={monitor.name: expected_bounds},
        _uses_nonuniform_grid=lambda: False,
    )

    constraints = fdtd3d._monitor_constraints(simulation, monitor, volume)

    assert len(constraints) == 1
    assert isinstance(constraints[0], fdtd_module.fdtdx.GridCoordinateConstraint)
    grid = fdtd_module.fdtdx.RectilinearGrid.uniform(
        shape=volume.partial_grid_shape,
        spacing=50e-9,
    )
    config = fdtd_module.fdtdx.SimulationConfig(
        grid=grid,
        time=1e-12,
        dtype=jnp.float32,
        courant_factor=0.99,
    )
    resolved, errors = resolve_object_constraints(
        objects=[volume, monitor],
        constraints=constraints,
        config=config,
    )

    assert errors[monitor.name] is None
    assert resolved[monitor.name] == expected_bounds


def test_sparse_scatter_preserves_full_domain_contract_and_zero_fill() -> None:
    """Localized values map into a six-component full-domain tensor."""
    detector = _detector(
        "region",
        (slice(1, 3), slice(2, 4), slice(0, 1)),
        components=("Ex", "Hz"),
    )
    phasor = torch.tensor(
        [[[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]],
        dtype=torch.complex64,
    )
    dense = _sparse_phasors_to_dense(
        {detector.name: phasor},
        objects={detector.name: detector},
        monitor=_monitor(),
        spatial_shape=(4, 5, 2),
        config=SimpleNamespace(symmetry=(0, 0, 0)),
        device=torch.device("cpu"),
        dtype=torch.complex64,
    )

    assert dense.shape == (1, 6, 4, 5, 2)
    torch.testing.assert_close(dense[0, 0, 1:3, 2:4, 0:1], phasor[0, 0])
    torch.testing.assert_close(dense[0, 5, 1:3, 2:4, 0:1], phasor[0, 1])
    assert torch.count_nonzero(dense[:, 1:5]) == 0


def test_sparse_scatter_does_not_double_identical_overlaps() -> None:
    """Two identical detector samples retain their physical amplitude."""
    first = _detector("first", (slice(0, 1),) * 3)
    second = _detector("second", (slice(0, 1),) * 3)
    value = torch.tensor([[[[[2 + 3j]]]]], dtype=torch.complex64)
    dense = _sparse_phasors_to_dense(
        {first.name: value, second.name: value.clone()},
        objects={first.name: first, second.name: second},
        monitor=_monitor(),
        spatial_shape=(2, 2, 2),
        config=SimpleNamespace(symmetry=(0, 0, 0)),
        device=torch.device("cpu"),
        dtype=torch.complex64,
    )

    assert dense[0, 0, 0, 0, 0] == 2 + 3j


def test_sparse_scatter_rejects_conflicting_overlaps() -> None:
    """Conflicting samples cannot silently overwrite or add to one another."""
    first = _detector("first", (slice(0, 1),) * 3)
    second = _detector("second", (slice(0, 1),) * 3)
    with pytest.raises(ValueError, match="detectors disagree"):
        _sparse_phasors_to_dense(
            {
                first.name: torch.ones((1, 1, 1, 1, 1), dtype=torch.complex64),
                second.name: 2 * torch.ones((1, 1, 1, 1, 1), dtype=torch.complex64),
            },
            objects={first.name: first, second.name: second},
            monitor=_monitor(),
            spatial_shape=(2, 2, 2),
            config=SimpleNamespace(symmetry=(0, 0, 0)),
            device=torch.device("cpu"),
            dtype=torch.complex64,
        )


def test_sparse_adjoint_detector_selects_active_waves_in_order() -> None:
    """Per-frequency sparse adjoints do not retain every forward frequency."""
    waves = tuple(
        SimpleNamespace(get_frequency=lambda frequency=frequency: frequency)
        for frequency in (1.2, 1.3, 1.4)
    )
    detector = SimpleNamespace(
        name="region",
        partial_grid_shape=(2, 3, 4),
        partial_real_shape=None,
        partial_real_position=None,
        wave_characters=waves,
        components=("Ex", "Ey", "Ez"),
        scaling_mode="none",
        switch=True,
    )

    selected = clone_phasor_detector_for_waves(
        detector,
        wave_characters=(waves[2], waves[0]),
        components=("Ex",),
        exact_interpolation=True,
    )

    assert tuple(wave.get_frequency() for wave in selected.wave_characters) == (
        1.4,
        1.2,
    )
    assert selected.components == ("Ex",)
    assert selected.exact_interpolation


def test_broadband_source_accepts_symmetry_reduced_profile(monkeypatch) -> None:
    """A full placement accepts a profile reduced by the configured symmetry."""
    template = torch.ones((3, 4, 3, 8), dtype=torch.complex64)
    alpha = torch.ones(2, dtype=torch.complex64)
    monkeypatch.setattr(
        "core.fdtd.solver._fit_separable_current_profile_torch",
        lambda cotangents, components: (template, None, alpha, 0),
    )
    monkeypatch.setattr("core.fdtd.solver._torch_to_jax", lambda value: value)
    monkeypatch.setattr(
        "core.fdtd.solver.DistributedCurrentSource",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    center_wave = object()
    monitor = SimpleNamespace(
        name="adjoint",
        components=("Ex", "Ey", "Ez"),
        partial_grid_shape=(4, 6, 8),
    )
    source_template = SimpleNamespace(
        temporal_profile=SimpleNamespace(center_wave=center_wave)
    )

    source, actual_alpha = _build_separable_broadband_adjoint_source(
        adj_monitor=monitor,
        source_template=source_template,
        cotangents=torch.zeros((2, 3, 4, 3, 8), dtype=torch.complex64),
        symmetry=(0, -1, 0),
    )

    assert source.partial_grid_shape == (4, 6, 8)
    assert source.electric_profiles.shape == (1, 3, 4, 3, 8)
    assert source.wave_characters == (center_wave,)
    torch.testing.assert_close(actual_alpha, alpha)


def test_sparse_adjoint_extras_exclude_primary_and_port_monitors(
    monkeypatch,
) -> None:
    """Sparse adjoints register each design detector exactly once."""

    class FakePhasorDetector:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(fdtd_module.fdtdx, "PhasorDetector", FakePhasorDetector)
    primary = FakePhasorDetector("field_detector_layer_1_Hz2")
    adjoint_primary = FakePhasorDetector("adjoint_field_detector")
    secondary = FakePhasorDetector("field_detector_layer_2_Hz2")
    port = FakePhasorDetector("field_detector_port_out_Hz2")
    visualization = FakePhasorDetector("field_detector_visualization_layer_1_xy_Hz2")
    simulation = SimpleNamespace(
        port_sources_dict={
            "in_field_monitor": {
                ("waves", "Hz2", "layer_1"): (primary,),
                ("waves", "Hz2", "layer_2"): (secondary,),
                ("waves", "Hz2", "port:out"): (port,),
                ("waves", "Hz2", "visualization:layer_1:xy"): (visualization,),
            }
        }
    )
    simulation._profile_detectors = MethodType(fdtd3d._profile_detectors, simulation)

    extras = fdtd3d._sparse_monitor_extras(
        simulation,
        input_slice_name="in",
        mode="Hz2",
        monitor_name=(adjoint_primary, primary),
        include_port_monitors=False,
        include_visualization_monitors=False,
    )

    assert [detector.name for detector, _ in extras] == [secondary.name]


def test_visualization_slabs_span_planes_and_stay_on_retained_symmetry_half() -> None:
    """Plot slabs cover full planes and are placeable on a reduced domain."""
    bounds = _design_region_visualization_bounds(
        region_bounds=((4, 8), (6, 10), (10, 14)),
        full_shape=(20, 16, 24),
        symmetry=(0, -1, 0),
        slab_cells=3,
    )

    assert bounds["xy"] == ((0, 20), (0, 16), (11, 14))
    assert bounds["xz"] == ((0, 20), (8, 11), (0, 24))
    assert bounds["yz"] == ((5, 8), (0, 16), (0, 24))

    mirrored_bounds = _design_region_visualization_bounds(
        region_bounds=((4, 8), (2, 4), (10, 14)),
        full_shape=(20, 16, 24),
        symmetry=(0, -1, 0),
        slab_cells=3,
    )
    assert mirrored_bounds["xz"] == ((0, 20), (11, 14), (0, 24))


def test_origin_visualization_uses_nearest_nonuniform_grid_cells() -> None:
    """Origin plots use physical coordinates rather than shape midpoints."""
    bounds = _origin_grid_cell_bounds(
        (
            (-3.0, -0.4, 0.1, 1.5),
            (-2.0, -0.2, 0.3),
            (-4.0, -1.0, -0.05, 0.8, 2.0),
        )
    )

    assert bounds == ((2, 3), (1, 2), (2, 3))


def test_sparse_dense_cotangent_pullback_matches_reference_graph() -> None:
    """Sparse backward transposes colocation, H timing, and unfolding in order."""
    components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    wave = SimpleNamespace(get_frequency=lambda: 2.1)
    monitor = SimpleNamespace(
        components=components,
        wave_characters=(wave,),
        _grid_slice_tuple=((0, 2), (0, 2), (0, 2)),
        reduce_volume=False,
    )
    config = SimpleNamespace(
        has_symmetry=True,
        symmetry=(0, -1, 0),
        has_nonuniform_grid=False,
        time_step_duration=0.07,
    )
    objects = SimpleNamespace(
        volume=SimpleNamespace(grid_shape=(2, 2, 2)),
        boundary_objects=(),
    )
    cotangents = torch.complex(
        torch.arange(1 * 6 * 2 * 4 * 2, dtype=torch.float64).reshape(1, 6, 2, 4, 2),
        torch.ones((1, 6, 2, 4, 2), dtype=torch.float64),
    )

    actual = _pullback_dense_cotangents_like_reference(
        cotangents,
        monitor=monitor,
        objects=objects,
        config=config,
    )

    yee = torch.zeros((1, 6, 2, 2, 2), dtype=torch.complex128, requires_grad=True)
    electric = yee_to_colocate_interpolate(yee[:, :3], objects, config, is_E=True)
    magnetic = yee_to_colocate_interpolate(yee[:, 3:], objects, config, is_E=False)
    theta = torch.as_tensor(
        2 * torch.pi * wave.get_frequency() * config.time_step_duration / 2,
        dtype=magnetic.real.dtype,
    )
    magnetic = magnetic * (torch.exp(1j * theta) * torch.cos(theta))
    dense = unfold_phasor(
        torch.cat((electric, magnetic), dim=1),
        monitor=monitor,
        components=components,
        config=config,
    )
    (expected,) = torch.autograd.grad(dense, yee, cotangents)

    torch.testing.assert_close(actual, expected)


def _complex_cotangents(shape: tuple[int, ...]) -> torch.Tensor:
    """Create deterministic complex cotangents for VJP comparisons."""
    count = 1
    for size in shape:
        count *= size
    real = torch.arange(count, dtype=torch.float64).reshape(shape) / count
    return torch.complex(real, 0.3 * real.flip(-1) + 0.2)


@pytest.mark.parametrize(
    ("symmetry", "reduced_shape", "full_shape"),
    [
        ((0, 0, 0), (2, 3, 2), (2, 3, 2)),
        ((-1, 0, 1), (2, 3, 2), (4, 3, 4)),
    ],
)
def test_analytical_pullback_matches_autodiff_for_symmetry_axes(
    symmetry: tuple[int, int, int],
    reduced_shape: tuple[int, int, int],
    full_shape: tuple[int, int, int],
) -> None:
    """Analytical folding and interpolation match the frozen graph VJP."""
    waves = tuple(
        SimpleNamespace(get_frequency=lambda frequency=frequency: frequency)
        for frequency in (1.7, 2.3)
    )
    monitor = SimpleNamespace(
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        wave_characters=waves,
        _grid_slice_tuple=tuple((0, size) for size in reduced_shape),
        reduce_volume=False,
    )
    config = SimpleNamespace(
        has_symmetry=any(symmetry),
        symmetry=symmetry,
        has_nonuniform_grid=False,
        time_step_duration=0.07,
    )
    objects = SimpleNamespace(
        volume=SimpleNamespace(grid_shape=reduced_shape),
        boundary_objects=(),
    )
    cotangents = _complex_cotangents((len(waves), 6, *full_shape))

    analytical = _pullback_dense_cotangents_like_reference(
        cotangents, monitor=monitor, objects=objects, config=config
    )
    autodiff = _pullback_dense_cotangents_autodiff_reference(
        cotangents, monitor=monitor, objects=objects, config=config
    )

    torch.testing.assert_close(analytical, autodiff, rtol=1e-12, atol=1e-12)


def test_analytical_pullback_is_frequency_local() -> None:
    """Independent frequency pullbacks equal the batched analytical result."""
    reduced_shape = (2, 2, 3)
    waves = tuple(
        SimpleNamespace(get_frequency=lambda frequency=frequency: frequency)
        for frequency in (1.4, 1.8, 2.2)
    )
    monitor = SimpleNamespace(
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        wave_characters=waves,
        _grid_slice_tuple=tuple((0, size) for size in reduced_shape),
        reduce_volume=False,
    )
    config = SimpleNamespace(
        has_symmetry=True,
        symmetry=(0, -1, 0),
        has_nonuniform_grid=False,
        time_step_duration=0.08,
    )
    objects = SimpleNamespace(
        volume=SimpleNamespace(grid_shape=reduced_shape),
        boundary_objects=(),
    )
    cotangents = _complex_cotangents((len(waves), 6, 2, 4, 3))

    batched = _pullback_dense_cotangents_like_reference(
        cotangents, monitor=monitor, objects=objects, config=config
    )
    local = torch.cat(
        [
            _pullback_dense_cotangents_like_reference(
                cotangents[index : index + 1],
                monitor=monitor,
                objects=objects,
                config=config,
                wave_characters=(wave,),
            )
            for index, wave in enumerate(waves)
        ]
    )

    torch.testing.assert_close(local, batched, rtol=1e-12, atol=1e-12)


def test_analytical_pullback_matches_autodiff_on_nonuniform_grid() -> None:
    """Analytical interpolation transpose preserves rectilinear weights."""
    reduced_shape = (3, 2, 4)
    widths = (
        torch.tensor((0.2, 0.3, 0.5), dtype=torch.float64),
        torch.tensor((0.4, 0.6), dtype=torch.float64),
        torch.tensor((0.1, 0.2, 0.3, 0.4), dtype=torch.float64),
    )
    grid = SimpleNamespace(cell_widths=lambda axis: widths[axis])
    wave = SimpleNamespace(get_frequency=lambda: 1.9)
    monitor = SimpleNamespace(
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        wave_characters=(wave,),
        _grid_slice_tuple=tuple((0, size) for size in reduced_shape),
        reduce_volume=False,
    )
    config = SimpleNamespace(
        has_symmetry=False,
        symmetry=(0, 0, 0),
        has_nonuniform_grid=True,
        resolved_grid=grid,
        time_step_duration=0.11,
    )
    objects = SimpleNamespace(
        volume=SimpleNamespace(grid_shape=reduced_shape),
        boundary_objects=(),
    )
    cotangents = _complex_cotangents((1, 6, *reduced_shape))

    analytical = _pullback_dense_cotangents_like_reference(
        cotangents, monitor=monitor, objects=objects, config=config
    )
    autodiff = _pullback_dense_cotangents_autodiff_reference(
        cotangents, monitor=monitor, objects=objects, config=config
    )

    torch.testing.assert_close(analytical, autodiff, rtol=1e-12, atol=1e-12)


def test_analytical_pullback_matches_autodiff_for_bloch_padding() -> None:
    """Analytical padding transpose conjugates periodic Bloch ghost phases."""

    class Boundary:
        uses_wrap_padding = True
        needs_complex_fields = True

        def __init__(self, axis: int, direction: str, phase: complex) -> None:
            self.axis = axis
            self.direction = direction
            self.phase = phase

        def get_bloch_phase(self, volume_shape, spacing):
            assert volume_shape == (3, 2, 2)
            assert spacing == 0.25
            return self.phase

    boundaries = (
        Boundary(0, "-", np.exp(0.37j)),
        Boundary(0, "+", np.exp(0.37j)),
    )
    wave = SimpleNamespace(get_frequency=lambda: 2.2)
    monitor = SimpleNamespace(
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        wave_characters=(wave,),
        _grid_slice_tuple=((0, 3), (0, 2), (0, 2)),
        reduce_volume=False,
    )
    config = SimpleNamespace(
        has_symmetry=False,
        symmetry=(0, 0, 0),
        has_nonuniform_grid=False,
        time_step_duration=0.09,
        uniform_spacing=lambda: 0.25,
    )
    objects = SimpleNamespace(
        volume=SimpleNamespace(grid_shape=(3, 2, 2)),
        boundary_objects=boundaries,
    )
    cotangents = _complex_cotangents((1, 6, 3, 2, 2))

    analytical = _pullback_dense_cotangents_like_reference(
        cotangents, monitor=monitor, objects=objects, config=config
    )
    autodiff = _pullback_dense_cotangents_autodiff_reference(
        cotangents, monitor=monitor, objects=objects, config=config
    )

    torch.testing.assert_close(analytical, autodiff, rtol=1e-12, atol=1e-12)
