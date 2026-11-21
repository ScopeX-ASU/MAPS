"""Integration contracts for native 3-D MAPS near-to-far projection."""

import numpy as np
import pytest
import torch

from core.invdes.models.layers.device_base import N_Ports
from core.invdes.models.layers.objective import ShapeSimilarityNear2FarObjective
from core.nearfield import project
from core.utils import Slice3D, build_rectilinear_grid_metadata

pytestmark = pytest.mark.integration


def _grid_metadata() -> dict:
    """Return a small nonuniform native grid expressed in micrometers."""
    boundaries = (
        np.asarray([-0.8, -0.2, 0.4, 1.0]),
        np.asarray([-0.9, -0.3, 0.2, 0.8, 1.3]),
        np.asarray([0.0, 0.25, 0.7]),
    )
    coordinates = tuple((axis[:-1] + axis[1:]) / 2 for axis in boundaries)
    return build_rectilinear_grid_metadata(
        coords=coordinates,
        boundaries=boundaries,
        label="near2far_test",
    ).to_dict()


def _case(return_field: str):
    """Build a prescribed six-component field and its 3-D objective."""
    grid = _grid_metadata()
    shape = tuple(grid["shape"])
    real = torch.float64
    values = torch.arange(np.prod(shape), dtype=real).reshape(shape) + 1
    ex = (values + 0.2j * values).to(torch.complex128).requires_grad_(True)
    fields = {
        ("source", 1.0, "Ez1", 300): {
            "Ex": ex,
            "Ey": 0.4j * ex,
            "Ez": 0.2 * ex,
            "Hx": -0.3j * ex,
            "Hy": 0.7 * ex,
            "Hz": -0.1 * ex,
        }
    }
    near_slice = Slice3D(x=slice(1, 3), y=slice(1, 4), z=0)
    far_slice = Slice3D(x=slice(0, 2), y=slice(0, 3), z=1)
    objective = ShapeSimilarityNear2FarObjective(
        sims={((1.0, 0.0, 1), "Ez", 300): object()},
        port_slices={"nearfield_top": near_slice, "farfield": far_slice},
        port_slices_info={
            "nearfield_top": {"direction": "z+"},
            "farfield": {
                "direction": "z+",
                "xs": np.asarray([-0.4, 0.4]),
                "ys": np.asarray([-0.5, 0.0, 0.5]),
                "zs": np.asarray(2.0),
            },
        },
        in_slice_name="source",
        out_slice_name="farfield",
        in_mode="Ez1",
        out_modes=("Ez1",),
        name="raw_3d",
        target_wls=(1.0,),
        target_temps=(300,),
        shape_type=None,
        shape_cfg=None,
        grid_step=0.1,
        eps_bg=1.0,
        grid_metadata=grid,
        return_field=return_field,
    )
    return objective, fields


def test_return_nearfield_exposes_si_surface_and_all_six_components():
    """The 3-D near-field result carries geometry, quadrature, and E/H fields."""
    objective, fields = _case("nearfield")
    key = ("source", 1.0, "Ez1", 300)
    result = objective(fields)[key]

    assert set(result) == {
        "positions_m",
        "normals",
        "weights_m2",
        "Ex",
        "Ey",
        "Ez",
        "Hx",
        "Hy",
        "Hz",
    }
    assert result["positions_m"].shape == (6, 3)
    assert result["weights_m2"].shape == (6,)
    assert torch.all(result["positions_m"][:, 0] >= -0.2e-6)
    assert torch.all(result["positions_m"][:, 1] >= -0.3e-6)
    assert torch.all(result["normals"][:, 2] == 1)
    result["Ex"].real.sum().backward()
    assert fields[key]["Ex"].grad is not None


def test_return_farfield_matches_direct_vector_projector():
    """The objective's raw 3-D result equals the native Torch projection."""
    objective, fields = _case("farfield")
    key = ("source", 1.0, "Ez1", 300)
    surface = objective._surface_3d(fields[key], 1.0)
    points, _, _ = objective._observation_points_3d(fields[key]["Ex"])
    expected_e, expected_h = project(surface, points, backend="direct_torch")

    result = objective(fields)[key]
    assert result["positions_m"].shape == (2, 3, 3)
    assert result["Ex"].shape == (2, 3)
    torch.testing.assert_close(result["positions_m"].reshape(-1, 3), points)
    for index, component in enumerate(("Ex", "Ey", "Ez")):
        torch.testing.assert_close(
            result[component].reshape(-1), expected_e[0, :, index]
        )
    for index, component in enumerate(("Hx", "Hy", "Hz")):
        torch.testing.assert_close(
            result[component].reshape(-1), expected_h[0, :, index]
        )


def test_default_3d_mode_reduces_projected_component_to_similarity(monkeypatch):
    """Unset ``return_field`` preserves the scalar shape-objective contract."""
    objective, fields = _case(None)
    captured = {}

    def _similarity(projected, **kwargs):
        captured["projected"] = projected
        captured["kwargs"] = kwargs
        return projected.abs().mean()

    monkeypatch.setattr(
        "core.invdes.models.layers.objective.get_shape_similarity", _similarity
    )
    result = objective(fields)

    assert result.ndim == 0
    assert captured["projected"].shape == (2, 3)
    assert captured["kwargs"]["intensity"] is True


def test_device_builds_planar_four_component_near2far_phasor_detector():
    """A registered native plane records the four tangential components."""
    pytest.importorskip("fdtdx")
    device = object.__new__(N_Ports)
    device.dim = 3
    device.epsilon_map = np.ones((4, 5, 6), dtype=np.float32)
    native_slice = Slice3D(x=slice(1, 4), y=slice(0, 4), z=2)
    device.port_monitor_slices_native = {"nearfield_top": native_slice}

    profiles = device.insert_monitors_fdtd3d(
        input_slice_name="nearfield_top",
        eps=device.epsilon_map,
        type="near2far",
        monitor_slice=native_slice,
        wl_cen=1.0,
        wl_width=0.0,
        n_wl=1,
        source_modes=("Ez1",),
        direction="z+",
    )
    key = ((1.0, 0.0, 1), "Ez1", "nearfield_top")
    detector, _, _, _, bounds = profiles[key]

    assert detector.partial_grid_shape == (3, 4, 1)
    assert tuple(detector.components) == ("Ex", "Ey", "Hx", "Hy")
    assert detector.exact_interpolation is True
    assert bounds == ((1, 4), (0, 4), (2, 3))
