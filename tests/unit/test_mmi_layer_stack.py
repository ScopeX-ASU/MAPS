import pytest
import torch

from core.invdes.models.layers.device_base import N_Ports
from core.invdes.models.layers.mmi import MMI


def _capture_device_config(monkeypatch, **overrides):
    """Build an MMI while replacing expensive device rasterization."""

    def capture_init(self, **kwargs):
        self.eps_bg = kwargs["eps_bg"]
        self.port_cfgs = kwargs["port_cfgs"]
        self.geometry_cfgs = kwargs["geometry_cfgs"]
        self.design_region_cfgs = kwargs["design_region_cfgs"]

    monkeypatch.setattr(N_Ports, "__init__", capture_init)
    arguments = dict(
        sim_cfg={
            "resolution": 100,
            "wl_cen": 1.55,
            "cell_size": None,
            "border_width": [0.0, 0.0, 0.0, 0.0, 1.2, 1.2],
        },
        box_size=(2.6, 2.6, 0.22),
        port_height=(0.22, 0.22),
        material_ports=(4.0, 9.0),
        material_buried=3.0,
        material_cladding=2.0,
        region_material_r1_layers=(12.0, 16.0),
        region_material_r2_layers=(2.0, 10.0),
        is_design_region=(False, True),
        thickness_layers=(0.08, 0.14),
        etch_depth_layers=(0.0, 0.14),
        gap_layers=(0.0, 0.0),
        box_size_layers=((2.6, 2.6, 0.08), (2.6, 2.6, 0.14)),
        layer_stack_z_offset=-0.07,
        is_3d=True,
        device=torch.device("cpu"),
    )
    arguments.update(overrides)
    return MMI(**arguments)


def test_mmi_role_based_materials_and_design_flags(monkeypatch):
    device = _capture_device_config(monkeypatch)

    assert device.eps_bg == pytest.approx(2.0)
    assert device.port_cfgs["in_port_1"]["eps"] == pytest.approx(4.0)
    assert device.port_cfgs["out_port_1"]["eps"] == pytest.approx(9.0)

    assert "mmi_region_layer_1" not in device.design_region_cfgs
    assert device.geometry_cfgs["fixed_slab_layer_1"]["eps"] == pytest.approx(12.0)
    pcm_region = device.design_region_cfgs["mmi_region_layer_2"]
    assert pcm_region["eps"] == pytest.approx(16.0)
    assert pcm_region["eps_bg"] == pytest.approx(10.0)
    assert pcm_region["center"][2] == pytest.approx(0.04)


def test_mmi_buried_region_ends_at_port_bottom(monkeypatch):
    device = _capture_device_config(monkeypatch)

    buried = device.geometry_cfgs["buried_region"]
    buried_bottom = buried["center"][2] - buried["size"][2] / 2
    buried_top = buried["center"][2] + buried["size"][2] / 2
    assert buried_bottom == pytest.approx(-1.31)
    assert buried_top == pytest.approx(-0.11)
    assert buried["eps"] == pytest.approx(3.0)
    assert buried["_exclude_from_bounds"] is True


def test_mmi_rejects_design_layer_without_design_thickness(monkeypatch):
    with pytest.raises(ValueError, match=r"etch_depth_layers\[0\] is zero"):
        _capture_device_config(
            monkeypatch,
            is_design_region=(True, True),
        )


def test_mmi_zero_thickness_layer_is_retained_but_emits_nothing(monkeypatch):
    device = _capture_device_config(
        monkeypatch,
        region_material_r1_layers=(12.0, 14.0, 16.0),
        region_material_r2_layers=(2.0, 3.0, 10.0),
        is_design_region=(False, True, True),
        thickness_layers=(0.08, 0.0, 0.14),
        etch_depth_layers=(0.0, 0.0, 0.14),
        gap_layers=(0.0, 0.03, 0.0),
        box_size_layers=(
            (2.6, 2.6, 0.08),
            (2.6, 2.6, 0.0),
            (2.6, 2.6, 0.14),
        ),
    )

    assert len(device.layer_bounds) == 3
    assert device.layer_bounds[1] == pytest.approx((0.0, 0.0, 0.0))
    assert "fixed_slab_layer_2" not in device.geometry_cfgs
    assert "fixed_top_layer_2" not in device.geometry_cfgs
    assert "mmi_region_layer_2" not in device.design_region_cfgs
    assert "mmi_region_layer_3" in device.design_region_cfgs


def test_mmi_single_2d_role_layer_preserves_legacy_footprint(monkeypatch):
    device = _capture_device_config(
        monkeypatch,
        box_size=(2.6, 2.6),
        material_buried=None,
        region_material_r1_layers=(12.0,),
        region_material_r2_layers=(2.0,),
        is_design_region=(True,),
        thickness_layers=None,
        etch_depth_layers=None,
        gap_layers=None,
        box_size_layers=None,
        layer_stack_z_offset=0.0,
        is_3d=False,
    )

    assert tuple(device.design_region_cfgs) == ("mmi_region",)
    assert device.design_region_cfgs["mmi_region"]["size"] == (2.6, 2.6)
    assert device.design_region_cfgs["mmi_region"]["eps"] == pytest.approx(12.0)
    assert device.design_region_cfgs["mmi_region"]["eps_bg"] == pytest.approx(2.0)


def test_mmi_single_physical_layer_preserves_legacy_region_name(monkeypatch):
    device = _capture_device_config(
        monkeypatch,
        material_buried=None,
        region_material_r1_layers=(12.0,),
        region_material_r2_layers=(2.0,),
        is_design_region=(True,),
        thickness_layers=(0.22,),
        etch_depth_layers=(0.22,),
        gap_layers=(0.0,),
        box_size_layers=((2.6, 2.6, 0.22),),
        layer_stack_z_offset=0.0,
    )

    assert tuple(device.design_region_cfgs) == ("mmi_region",)
    assert device.design_region_cfgs["mmi_region"]["center"] == [0.0, 0.0, 0.0]
    assert device.design_region_cfgs["mmi_region"]["size"] == [2.6, 2.6, 0.22]


def test_mmi_legacy_material_arguments_remain_compatible(monkeypatch):
    device = _capture_device_config(
        monkeypatch,
        material_r1=12.0,
        material_r2=2.0,
        material_bg=3.0,
        material_ports=None,
        material_buried=None,
        material_cladding=None,
        region_material_r1_layers=None,
        region_material_r2_layers=None,
        is_design_region=None,
        thickness_layers=None,
        etch_depth_layers=None,
        gap_layers=None,
        box_size_layers=None,
        layer_stack_z_offset=0.0,
    )

    assert device.eps_bg == pytest.approx(3.0)
    assert device.port_cfgs["in_port_1"]["eps"] == pytest.approx(12.0)
    assert device.port_cfgs["out_port_1"]["eps"] == pytest.approx(12.0)
    assert device.design_region_cfgs["mmi_region"]["eps"] == pytest.approx(12.0)
    assert device.design_region_cfgs["mmi_region"]["eps_bg"] == pytest.approx(2.0)
    assert "buried_region" not in device.geometry_cfgs
