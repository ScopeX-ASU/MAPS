"""Contracts for deriving Tidy3D snapping points from prism configs."""

import pytest

from core.invdes.models.layers.device_base import N_Ports


def _snapping_device(geometry_cfgs):
    device = object.__new__(N_Ports)
    device.dim = 3
    device.cell_extend = ((-10.0, 10.0), (-10.0, 10.0), (-2.0, 2.0))
    device.port_cfgs = {}
    device.geometry_cfgs = geometry_cfgs
    device.design_region_cfgs = {}
    device.heat_source_cfgs = {}
    return device


@pytest.mark.unit
def test_prism_bounds_use_slab_bounds_for_z_snapping():
    # Matches the extruded polygon schema used by fdtd_pipeline geometry configs.
    prism = {
        "type": "prism",
        "center": [0.0, 0.0, 0.0],
        "vertices": [(-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)],
        "slab_bounds": (0.0, 0.22),
    }
    device = _snapping_device({"gds_1_0_0": prism})

    mins, maxs = device._cfg_bounds_3d(prism)
    assert mins == pytest.approx([-1.0, -0.5, 0.0])
    assert maxs == pytest.approx([1.0, 0.5, 0.22])
    assert device._derived_tidy3d_snapping_points() == [
        (-1.0, None, None),
        (1.0, None, None),
        (None, -0.5, None),
        (None, 0.5, None),
        (None, None, 0.0),
        (None, None, 0.22),
    ]


@pytest.mark.unit
def test_fdtd_pipeline_style_prisms_snap_both_layer_interfaces():
    # The benchmark emits separate silicon and slab prisms sharing z=0.
    base = {
        "type": "prism",
        "center": [0.0, 0.0, 0.0],
        "vertices": [(-2.0, -0.25), (2.0, -0.25), (2.0, 0.25), (-2.0, 0.25)],
    }
    geometry_cfgs = {
        "gds_1_0_0": {**base, "slab_bounds": (0.0, 0.22)},
        "gds_2_0_0": {**base, "slab_bounds": (0.0, 0.15)},
    }
    device = _snapping_device(geometry_cfgs)

    points = device._derived_tidy3d_snapping_points()
    assert (None, None, 0.0) in points
    assert (None, None, 0.15) in points
    assert (None, None, 0.22) in points
    assert (None, None, -0.11) not in points
    assert (None, None, 0.11) not in points


@pytest.mark.unit
def test_unbounded_prism_slab_does_not_add_finite_z_snapping_points():
    prism = {
        "type": "prism",
        "vertices": [(-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)],
        "slab_bounds": (-float("inf"), float("inf")),
    }
    device = _snapping_device({"infinite_prism": prism})

    assert device._derived_tidy3d_snapping_points() == [
        (-1.0, None, None),
        (1.0, None, None),
        (None, -0.5, None),
        (None, 0.5, None),
    ]


@pytest.mark.unit
def test_prism_axis_maps_vertices_and_slab_bounds_to_correct_snapping_axes():
    prism = {
        "type": "prism",
        "vertices": [(-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)],
        "axis": 1,
        "sidewall_angle": 0.0,
        "slab_bounds": (-0.2, 0.2),
    }
    device = _snapping_device({"y_extruded_prism": prism})

    mins, maxs = device._cfg_bounds_3d(prism)
    assert mins == pytest.approx([-1.0, -0.2, -0.5])
    assert maxs == pytest.approx([1.0, 0.2, 0.5])
    points = device._derived_tidy3d_snapping_points()
    assert (None, -0.2, None) in points
    assert (None, 0.2, None) in points
    assert (None, None, -0.5) in points
    assert (None, None, 0.5) in points


@pytest.mark.unit
def test_prism_bounds_reject_reversed_slab_bounds():
    prism = {
        "type": "prism",
        "vertices": [(-1.0, -0.5), (1.0, -0.5), (1.0, 0.5)],
        "slab_bounds": (0.22, 0.0),
    }
    device = _snapping_device({"invalid_prism": prism})

    with pytest.raises(ValueError, match="ordered pair"):
        device._cfg_bounds_3d(prism)
