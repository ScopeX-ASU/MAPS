import pytest
import torch

from core.invdes.models.layers.device_base import N_Ports
from core.invdes.models.layers.grating_coupler import GratingCoupler


def _capture_grating_config(monkeypatch, **overrides):
    """Build a grating coupler without rasterizing its material map."""

    def capture_init(self, **kwargs):
        self.eps_bg = kwargs["eps_bg"]
        self.port_cfgs = kwargs["port_cfgs"]
        self.geometry_cfgs = kwargs["geometry_cfgs"]
        self.design_region_cfgs = kwargs["design_region_cfgs"]

    monkeypatch.setattr(N_Ports, "__init__", capture_init)
    arguments = dict(
        material_r=12.0,
        material_layers=(10.0,),
        thickness_layers=(0.22,),
        etch_depth_layers=(0.22,),
        gap_layers=(0.0,),
        material_bg=7.0,
        material_buried=3.0,
        material_cladding=2.0,
        sim_cfg={
            "resolution": 100,
            "wl_cen": 1.55,
            "cell_size": None,
            "border_width": [0.0, 1.8, 0.6, 0.6, 0.6, 0.6],
            "PML": [0.2, 0.2, 0.2],
        },
        design_region_size=(3.0, 3.0),
        port_height=0.22,
        substrate_thickness=0.5,
        substrate_gap=2.0,
        is_3d=True,
        device=torch.device("cpu"),
    )
    arguments.update(overrides)
    return GratingCoupler(**arguments)


def test_grating_material_roles_are_applied_independently(monkeypatch):
    device = _capture_grating_config(monkeypatch)

    assert device.eps_bg == pytest.approx(2.0)
    assert device.port_cfgs["out_port_1"]["eps"] == pytest.approx(12.0)
    assert device.port_cfgs["in_port_1"]["eps"] == pytest.approx(2.0)
    assert device.design_region_cfgs["grating_layer_1"]["eps"] == pytest.approx(10.0)
    assert device.design_region_cfgs["grating_layer_1"]["eps_bg"] == pytest.approx(2.0)
    assert device.geometry_cfgs["substrate"]["eps"] == pytest.approx(12.0)


def test_grating_buried_region_fills_to_stack_bottom(monkeypatch):
    device = _capture_grating_config(monkeypatch)

    buried = device.geometry_cfgs["buried_region"]
    buried_bottom = buried["center"][2] - buried["size"][2] / 2
    buried_top = buried["center"][2] + buried["size"][2] / 2
    assert buried_bottom == pytest.approx(-3.21)
    assert buried_top == pytest.approx(-0.11)
    assert buried["eps"] == pytest.approx(3.0)
    assert buried["_exclude_from_bounds"] is True
    assert list(device.geometry_cfgs)[:2] == ["buried_region", "substrate"]


def test_grating_material_bg_remains_cladding_fallback(monkeypatch):
    device = _capture_grating_config(
        monkeypatch,
        material_bg=4.0,
        material_buried=None,
        material_cladding=None,
    )

    assert device.eps_bg == pytest.approx(4.0)
    assert device.port_cfgs["in_port_1"]["eps"] == pytest.approx(4.0)
    assert device.design_region_cfgs["grating_layer_1"]["eps_bg"] == pytest.approx(4.0)
    assert "buried_region" not in device.geometry_cfgs


def test_grating_2d_buried_geometry_uses_vertical_y_axis(monkeypatch):
    device = _capture_grating_config(
        monkeypatch,
        design_region_size=(3.0, 1.0),
        sim_cfg={
            "resolution": 100,
            "wl_cen": 1.55,
            "cell_size": None,
            "border_width": [0.0, 1.8, 0.6, 0.6],
            "PML": [0.2, 0.2],
        },
        is_3d=False,
    )

    assert len(device.geometry_cfgs["buried_region"]["center"]) == 2
    assert len(device.geometry_cfgs["buried_region"]["size"]) == 2
    assert len(device.geometry_cfgs["substrate"]["center"]) == 2
    assert len(device.geometry_cfgs["substrate"]["size"]) == 2
