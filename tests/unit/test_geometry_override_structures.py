from core.invdes.models.layers.device_base import N_Ports


def _device_with_captured_geometry_builds():
    device = object.__new__(N_Ports)
    device.geometry = {}
    device.override_structures = {}

    def build_geometry(cfg, **kwargs):
        return {"cfg": cfg, "kwargs": kwargs}

    device._build_geometry_from_cfg = build_geometry
    return device


def test_geometry_cfg_override_flag_routes_to_override_structures():
    device = _device_with_captured_geometry_builds()
    cfgs = {
        "physical": {"type": "box", "eps": 12.0},
        "explicit_physical": {
            "type": "box",
            "eps": 10.0,
            "override_structure": False,
        },
        "mesh_helper": {
            "type": "box",
            "eps": 2.0,
            "override_structure": True,
        },
    }

    device.add_geometries(cfgs, use_tidy3d=True, apply_roughness=False)

    assert set(device.geometry) == {"physical", "explicit_physical"}
    assert set(device.override_structures) == {"mesh_helper"}
    assert device.override_structures["mesh_helper"]["cfg"] is cfgs["mesh_helper"]
    assert device.override_structures["mesh_helper"]["kwargs"]["use_tidy3d"] is True


def test_design_region_cfgs_remain_unconditional_override_structures():
    device = _device_with_captured_geometry_builds()
    cfgs = {"design_region": {"type": "box", "eps": 12.0}}

    device.add_override_structures(cfgs, use_tidy3d=False)

    assert device.geometry == {}
    assert set(device.override_structures) == {"design_region"}


def test_mesh_overrides_are_excluded_from_physical_geometry_cfgs():
    device = _device_with_captured_geometry_builds()
    device.port_cfgs = {"port": {"type": "box"}}
    device.geometry_cfgs = {
        "physical": {"type": "box"},
        "mesh_helper": {"type": "box", "override_structure": True},
    }

    combined_cfgs = device._combined_geometry_cfgs()

    assert set(combined_cfgs) == {"port", "physical"}


def test_mesh_override_is_absent_from_physical_raster_geometry():
    device = _device_with_captured_geometry_builds()
    cfgs = {
        "substrate": {"type": "box", "eps": 12.0},
        "substrate_override": {
            "type": "box",
            "eps": 2.0,
            "override_structure": True,
        },
    }

    device.add_geometries(cfgs, use_tidy3d=True, apply_roughness=False)

    physical_structures = list(device.geometry.values())
    assert len(physical_structures) == 1
    assert physical_structures[0]["cfg"] is cfgs["substrate"]
    assert device.override_structures["substrate_override"] not in physical_structures
