from types import SimpleNamespace

from core.invdes.models.layers import device_base as device_base_module


class _CapturedPortDevice:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.geometry_cfgs = kwargs["geometry_cfgs"]
        self.port_cfgs = kwargs["port_cfgs"]
        self.grid_info_dict = {}
        self.monitor_calls = []

    def add_monitor_slice(self, name, center, size, direction):
        self.monitor_calls.append((name, center, size, direction))


def _source_device():
    return SimpleNamespace(
        dim=3,
        cell_size=(10.0, 4.0, 6.0),
        eps_bg=2.0,
        device="cpu",
        sim_cfg={"cell_size": None, "resolution": 20},
        grid_info_dict={"epsilon_map": {"grid": None}},
        port_cfgs={
            "out_port_1": {
                "type": "box",
                "direction": "x",
                "center": [2.0, 0.0, 0.0],
                "size": [1.0, 0.5, 0.22],
                "eps": 12.0,
            }
        },
        geometry_cfgs={
            "buried_region": {
                "type": "box",
                "center": [0.0, 0.0, -1.0],
                "size": [None, None, 2.0],
                "eps": 3.0,
                "_exclude_from_bounds": True,
            },
            "substrate": {
                "type": "box",
                "center": (0.0, 0.0, -2.5),
                "size": (None, None, 1.0),
                "eps": 12.0,
            },
            "central_box": {
                "type": "box",
                "center": [0.0, 0.0, 0.0],
                "size": [1.0, 1.0, 1.0],
                "eps": 8.0,
            },
            "plane_box": {
                "type": "box",
                "center": [1.8, 1.0, 0.0],
                "size": [0.2, 0.4, 0.4],
                "eps": 6.0,
                "roughness": {"amplitude": 0.01},
            },
            "curved_geometry": {
                "type": "cylinder",
                "center": [1.8, 0.0, 0.0],
                "radius": 0.2,
                "height": 0.4,
                "eps": 5.0,
            },
        },
        port_monitor_slices_info={
            "out_slice_1": {
                "center": [1.8, 0.0, 0.0],
                "size": [0.0, 1.0, 1.0],
                "direction": "x+",
            }
        },
    )


def test_two_ports_carries_intersecting_box_cross_section(monkeypatch):
    monkeypatch.setattr(device_base_module, "N_Ports", _CapturedPortDevice)

    port_device = device_base_module.get_two_ports(
        _source_device(),
        "out_port_1",
        "out_slice_1",
    )

    assert set(port_device.geometry_cfgs) == {
        "port_cross_section_buried_region",
        "port_cross_section_substrate",
        "port_cross_section_plane_box",
    }
    for cfg in port_device.geometry_cfgs.values():
        assert cfg["center"][0] == 0.0
        assert cfg["size"][0] == 10.0
    assert port_device.geometry_cfgs["port_cross_section_buried_region"]["eps"] == 3.0
    assert port_device.geometry_cfgs["port_cross_section_substrate"]["eps"] == 12.0
    assert "roughness" not in port_device.geometry_cfgs["port_cross_section_plane_box"]


def test_two_ports_uses_requested_monitor_plane(monkeypatch):
    monkeypatch.setattr(device_base_module, "N_Ports", _CapturedPortDevice)
    source = _source_device()
    source.port_monitor_slices_info["out_slice_1"]["center"][0] = 0.0

    port_device = device_base_module.get_two_ports(
        source,
        "out_port_1",
        "out_slice_1",
    )

    assert "port_cross_section_central_box" in port_device.geometry_cfgs
    assert "port_cross_section_plane_box" not in port_device.geometry_cfgs
