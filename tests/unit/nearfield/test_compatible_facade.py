"""Public Tidy3D-compatible near-field facade contracts."""

import pytest

torch = pytest.importorskip("torch")

from core.nearfield import FieldMonitor  # noqa: E402
from core.nearfield import (
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionData,
    FieldProjectionKSpaceMonitor,
    FieldProjectionSurface,
    FieldProjector,
    Surface,
    direction_vectors_spherical,
    project_farfield,
    project_nearfield,
)

pytestmark = pytest.mark.unit


def _sample(name="plane"):
    dtype = torch.float64
    positions = torch.tensor([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0]], dtype=dtype)
    return Surface(
        positions,
        torch.tensor([[0.0, 0.0, 1.0]] * 2, dtype=dtype),
        torch.tensor([0.5e-12, 0.5e-12], dtype=dtype),
        torch.ones((1, 2, 3), dtype=torch.complex128),
        torch.ones((1, 2, 3), dtype=torch.complex128),
        torch.tensor([2e14], dtype=dtype),
    )


def test_plane_and_box_projection_surface_generation_and_exclusions():
    plane = FieldProjectionAngleMonitor(
        center=(0, 0, 1),
        size=(2, 2, 0),
        freqs=(2e14,),
        name="plane",
        normal_dir="-",
        theta=(0.2,),
        phi=(0.3,),
        interval_space=(2, 3, 1),
    )
    assert len(plane.projection_surfaces) == 1
    assert plane.projection_surfaces[0].normal_dir == "-"
    assert plane.projection_surfaces[0].monitor.fields == ("Ex", "Ey", "Hx", "Hy")
    assert plane.projection_surfaces[0].monitor.interval_space == (2, 3, 1)
    box = FieldProjectionCartesianMonitor(
        center=(0, 0, 0),
        size=(2, 4, 6),
        freqs=(2e14,),
        name="box",
        x=(0,),
        y=(0,),
        exclude_surfaces=("z-",),
    )
    surfaces = box.projection_surfaces
    assert len(surfaces) == 5
    assert {item.monitor.name.rsplit("_", 1)[-1] for item in surfaces} == {
        "x-",
        "x+",
        "y-",
        "y+",
        "z+",
    }
    assert surfaces[-1].monitor.center == (0.0, 0.0, 3.0)


def test_projection_data_is_torch_native_frequency_last_and_differentiable():
    monitor = FieldMonitor(
        center=(0, 0, 0), size=(1, 1, 0), freqs=(2e14,), name="plane"
    )
    field = _sample()
    field = Surface(
        field.positions_m,
        field.normals,
        field.weights_m2,
        field.electric_fields.requires_grad_(),
        field.magnetic_fields,
        field.frequencies_hz,
    )
    projector = FieldProjector(
        {"plane": field},
        [FieldProjectionSurface(monitor, "+")],
        pts_per_wavelength=None,
    )
    assert projector.frequencies.tolist() == [2e14]
    assert projector.surfaces[0].axis == 2
    target = FieldProjectionAngleMonitor(
        center=(0, 0, 10),
        size=(1, 1, 0),
        freqs=(2e14,),
        name="target",
        normal_dir="+",
        theta=(0.2, 0.4),
        phi=(0.1, 0.2),
        proj_distance=10,
    )
    result = projector.project_fields(target)
    assert result.Etheta.shape == (1, 2, 2, 1)
    assert result.Etheta.is_complex()
    assert result.directional_power.shape == result.Etheta.shape
    assert result.integrated_flux.shape == (1,)
    assert result.coordinates["r"].tolist() == pytest.approx([10.0])
    assert result.cartesian_fields()["Ex"].shape == result.Etheta.shape
    assert result.projection_surfaces == result.surfaces
    torch.testing.assert_close(result.Ex, result.cartesian_fields()["Ex"])
    result.Etheta.abs().sum().backward()
    assert field.electric_fields.grad is not None


def test_cartesian_local_axes_map_around_projection_axis():
    monitor = FieldMonitor(size=(1, 1, 0), freqs=(2e14,), name="plane")
    projector = FieldProjector(
        {"plane": _sample()},
        [FieldProjectionSurface(monitor, "+")],
        pts_per_wavelength=None,
    )
    target = FieldProjectionCartesianMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=(2e14,),
        name="cart",
        proj_axis=1,
        x=(2,),
        y=(3,),
        proj_distance=5,
        normal_dir="+",
    )
    data = projector.project_fields(target)
    assert data.coordinates["x"].shape == (1, 1, 1)
    assert data.coordinates["y"].shape == (1, 1, 1)
    assert data.Er.shape == (1, 1, 1, 1)


def test_cartesian_exact_facade_matches_direct_projection():
    """Exact Cartesian facade points preserve local-to-global axis ordering."""
    source = _sample()
    monitor = FieldMonitor(size=(1, 1, 0), freqs=(2e14,), name="plane")
    projector = FieldProjector(
        {"plane": source},
        [FieldProjectionSurface(monitor, "+")],
        pts_per_wavelength=None,
        backend="direct_torch",
    )
    target = FieldProjectionCartesianMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=(2e14,),
        name="cart-exact",
        normal_dir="+",
        proj_axis=1,
        x=(2,),
        y=(3,),
        proj_distance=5,
        far_field_approx=False,
    )
    actual = projector.project_fields(target)
    point = torch.tensor([[2e-6, 5e-6, 3e-6]], dtype=torch.float64)
    expected_e, expected_h = project_nearfield(source, point)
    for index, name in enumerate(("Ex", "Ey", "Ez")):
        torch.testing.assert_close(
            getattr(actual, name).reshape(-1), expected_e[0, :, index]
        )
    for index, name in enumerate(("Hx", "Hy", "Hz")):
        torch.testing.assert_close(
            getattr(actual, name).reshape(-1), expected_h[0, :, index]
        )


def test_angular_far_facade_restores_radial_factor():
    """Angular facade equals the direct far coefficient times exp(ikr)/r."""
    source = _sample()
    monitor = FieldMonitor(size=(1, 1, 0), freqs=(2e14,), name="plane")
    projector = FieldProjector(
        {"plane": source},
        [FieldProjectionSurface(monitor, "+")],
        pts_per_wavelength=None,
        origin=(0, 0, 0),
    )
    target = FieldProjectionAngleMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=(2e14,),
        name="angle-far",
        normal_dir="+",
        theta=(0.3,),
        phi=(0.4,),
        proj_distance=20,
    )
    actual = projector.project_fields(target)
    direction = direction_vectors_spherical(
        torch.tensor([0.3], dtype=torch.float64),
        torch.tensor([0.4], dtype=torch.float64),
    )
    coefficient_e, coefficient_h = project_farfield(source, direction)
    radius = torch.tensor(20e-6, dtype=torch.float64)
    wave_number = 2 * torch.pi * source.frequencies_hz[0] / 299792458.0
    radial_factor = torch.exp(1j * wave_number * radius) / radius
    for index, name in enumerate(("Ex", "Ey", "Ez")):
        torch.testing.assert_close(
            getattr(actual, name).reshape(-1),
            coefficient_e[0, :, index] * radial_factor,
        )
    for index, name in enumerate(("Hx", "Hy", "Hz")):
        torch.testing.assert_close(
            getattr(actual, name).reshape(-1),
            coefficient_h[0, :, index] * radial_factor,
        )


def test_far_facade_rejects_exact_only_triton_backend():
    """An explicit exact Green-function backend is not silently ignored."""
    monitor = FieldMonitor(size=(1, 1, 0), freqs=(2e14,), name="plane")
    projector = FieldProjector(
        {"plane": _sample()},
        [FieldProjectionSurface(monitor, "+")],
        pts_per_wavelength=None,
        backend="direct_triton",
    )
    target = FieldProjectionAngleMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=(2e14,),
        name="far",
        normal_dir="+",
        theta=(0.2,),
        phi=(0.3,),
    )
    with pytest.raises(ValueError, match="exact finite-distance backend"):
        projector.project_fields(target)


@pytest.mark.parametrize("far_field_approx", [True, False])
def test_kspace_projection_dispatches_far_and_exact(far_field_approx):
    source_monitor = FieldMonitor(size=(1, 1, 0), freqs=(2e14,), name="plane")
    projector = FieldProjector(
        {"plane": _sample()},
        [FieldProjectionSurface(source_monitor, "+")],
        pts_per_wavelength=None,
    )
    target = FieldProjectionKSpaceMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=(2e14,),
        name="kspace",
        normal_dir="+",
        proj_axis=2,
        ux=(0.1,),
        uy=(0.2,),
        proj_distance=10,
        far_field_approx=far_field_approx,
    )
    data = projector.project_fields(target)
    assert data.Etheta.shape == (1, 1, 1, 1)


def test_kspace_monitor_rejects_nonpropagating_directions():
    with pytest.raises(ValueError, match="propagating"):
        FieldProjectionKSpaceMonitor(
            center=(0, 0, 0),
            size=(1, 1, 0),
            freqs=(2e14,),
            name="bad-kspace",
            normal_dir="+",
            proj_axis=2,
            ux=(0.8,),
            uy=(0.8,),
        )


def test_negative_cartesian_projection_distance_reverses_integrated_flux():
    target = FieldProjectionCartesianMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=(2e14,),
        name="negative-plane",
        normal_dir="-",
        proj_axis=2,
        x=(0, 1),
        y=(0, 1),
        proj_distance=-1,
    )
    shape = (2, 2, 1, 1)
    zero = torch.zeros(shape, dtype=torch.complex128)
    one = torch.ones(shape, dtype=torch.complex128)
    angle = torch.full((2, 2, 1), torch.pi / 2, dtype=torch.float64)
    azimuth = torch.zeros_like(angle)
    data = FieldProjectionData(
        target,
        (),
        {"theta": angle, "phi": azimuth, "r": torch.ones_like(angle)},
        one,
        zero,
        zero,
        zero,
        zero,
        one,
    )
    assert data.integrated_flux.shape == (1,)
    assert data.integrated_flux[0].item() == pytest.approx(
        -0.5e-12, rel=1e-12, abs=1e-24
    )


def test_monitor_requires_explicit_colocated_semantics():
    with pytest.raises(ValueError, match="colocate=True"):
        FieldMonitor(size=(1, 1, 0), freqs=(1e14,), name="bad", colocate=False)


def test_wavelength_resampling_refines_rectilinear_surface_and_preserves_gradients():
    x = torch.tensor([0.0, 1e-6, 2e-6], dtype=torch.float64)
    y = torch.tensor([0.0, 1e-6], dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    positions = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1).reshape(-1, 3)
    field = torch.stack((xx.reshape(-1), yy.reshape(-1), xx.reshape(-1) * 0), dim=-1)
    e = field.to(torch.complex128)[None].requires_grad_()
    h = torch.ones_like(e)
    source = Surface(
        positions,
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand(6, 3),
        torch.full((6,), 2e-6 / 6, dtype=torch.float64),
        e,
        h,
        torch.tensor([1e14], dtype=torch.float64),
    )
    monitor = FieldMonitor(size=(2, 1, 0), freqs=(1e14,), name="structured")
    projector = FieldProjector(
        {"structured": source},
        [FieldProjectionSurface(monitor, "+")],
        pts_per_wavelength=4,
    )
    assert projector._surface.positions_m.shape[0] > positions.shape[0]
    assert projector._surface.weights_m2.sum().item() == pytest.approx(2e-12)
    projector._surface.electric_fields.abs().sum().backward()
    assert e.grad is not None


def test_wavelength_resampling_none_preserves_explicit_sampling():
    monitor = FieldMonitor(size=(1, 1, 0), freqs=(2e14,), name="plane")
    source = _sample()
    projector = FieldProjector(
        {"plane": source},
        [FieldProjectionSurface(monitor, "+")],
        pts_per_wavelength=None,
    )
    torch.testing.assert_close(projector._surface.positions_m, source.positions_m)
    torch.testing.assert_close(projector._surface.weights_m2, source.weights_m2)


def test_wavelength_downsampling_averages_alternating_samples():
    x = torch.linspace(0, 1e-6, 11, dtype=torch.float64)
    y = torch.tensor([0.0], dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    positions = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1).reshape(-1, 3)
    electric = torch.zeros((1, 11, 3), dtype=torch.complex128)
    electric[0, :, 0] = torch.tensor([1, -1] * 5 + [1], dtype=torch.float64)
    magnetic = torch.ones_like(electric)
    source = Surface(
        positions,
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand(11, 3),
        torch.full((11,), 1e-7, dtype=torch.float64),
        electric,
        magnetic,
        torch.tensor([0.99e14], dtype=torch.float64),
    )
    monitor = FieldMonitor(size=(1, 1, 0), freqs=(0.99e14,), name="alternating")
    projector = FieldProjector(
        {"alternating": source},
        [FieldProjectionSurface(monitor, "+")],
        pts_per_wavelength=6,
    )
    assert projector._surface.positions_m.shape[0] == 3
    assert projector._surface.electric_fields[0, 1, 0].abs().item() == pytest.approx(
        0.2
    )
    assert projector._surface.electric_fields[0, 1, 0].abs() < 1
