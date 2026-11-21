"""Native modal-profile apertures computed by the inverse-design device API."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fdtdx")

from core.fdtd.fdtd import fdtd3d
from core.invdes.models.layers.device_base import N_Ports
from core.utils import Slice3D
from thirdparty.fdtdx import fdtdx

pytestmark = [pytest.mark.integration, pytest.mark.fdtdx, pytest.mark.numerical]


def test_native_mode_detector_solves_on_requested_cross_section():
    """The actual mode solve uses [-0.8, 0.8] um on both transverse axes."""
    edges_um = (
        np.linspace(-1.6, 1.6, 17),
        np.array(
            [
                -1.2,
                -1.1,
                -1.0,
                -0.8,
                -0.6,
                -0.4,
                -0.2,
                0.0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
                1.2,
            ]
        ),
        np.linspace(-1.2, 1.2, 13),
    )
    coords = tuple((axis[:-1] + axis[1:]) / 2 for axis in edges_um)
    shape = tuple(len(axis) for axis in coords)
    epsilon = np.full(shape, 1.44**2, dtype=np.float32)
    core = (np.abs(coords[1][:, None]) < 0.4) & (np.abs(coords[2][None, :]) < 0.2)
    epsilon[:, core] = 3.5**2
    device = SimpleNamespace(
        epsilon_map=epsilon,
        grid_info_dict={
            "epsilon_map": dict(boundaries=edges_um, coords=coords, shape=shape)
        },
        sim_cfg=dict(symmetry=(0, 0, 0)),
        port_monitor_slices_info={
            "output": dict(center=(-0.5, 0, 0), size=(0, 1.6, 1.6))
        },
        port_monitor_slices_native={},
        _check_object_symmetry=lambda *args, **kwargs: None,
        device="cpu",
    )
    profiles = N_Ports.insert_monitors_fdtd3d(
        device,
        "output",
        epsilon,
        "mode",
        Slice3D(5, slice(3, 11), slice(2, 10)),
        source_modes=("Hz1",),
        direction="x+",
        on_native_grid=True,
    )
    expected = (slice(5, 6), slice(3, 11), slice(2, 10))
    assert device.port_monitor_slices_native["output"] == expected
    prototype, magnetic, electric, _ = profiles[(1.55, "Hz1")]
    assert tuple(prototype.partial_grid_shape) == (1, None, None)
    for field in (electric, magnetic):
        assert tuple(field.shape[-3:]) == (1, 8, 8)
        assert field.isfinite().all()
        assert field.abs().max() > 0


@pytest.mark.parametrize("sparse", [False, True])
def test_production_recorders_honor_physical_and_explicit_bounds(monkeypatch, sparse):
    """Dense and sparse preparation place auxiliary apertures without conflicts."""
    import importlib

    import torch

    module = importlib.import_module("core.fdtd.fdtd")
    monkeypatch.setattr(module, "_MAPS_SPARSE_PHASOR", sparse)
    edges = tuple(
        np.asarray(axis)
        for axis in (
            [-3, -2.5, -2, -1, 0, 1, 2, 3],
            [-3, -2, -1, 0, 1, 2, 3],
            [-1, -0.5, 0, 0.5, 1],
        )
    )
    shape = tuple(len(axis) - 1 for axis in edges)
    native = dict(
        shape=shape,
        boundaries=edges,
        coords=tuple((axis[:-1] + axis[1:]) / 2 for axis in edges),
        is_uniform=False,
    )
    simulation = fdtd3d(
        wl_cen=1.55,
        wl_width=0.0,
        n_wl=1,
        dL=0.5e-6,
        eps_r=torch.ones(shape),
        port_sources_dict={},
        port_monitor_slices={},
        port_monitor_slices_info={},
        optical_grid_metadata=dict(
            native=native, export=native, native_equals_export=False
        ),
        npml=((0, 0), (0, 0), (0, 0)),
        max_time=1e-14,
        device="cpu",
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    aperture = dict(
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(4e-6, 4e-6, None),
        partial_real_position=(0.0, 0.0, 0.25e-6),
    )
    source = fdtdx.UniformPlaneSource(
        name="source",
        wave_character=wave,
        direction="-",
        fixed_E_polarization_vector=(1.0, 0.0, 0.0),
        **aperture,
    )
    primary = fdtdx.PhasorDetector(name="primary", wave_characters=(wave,), **aperture)
    padded = fdtdx.PhasorDetector(name="padded", wave_characters=(wave,), **aperture)
    physical = fdtdx.PhasorDetector(
        name="physical",
        wave_characters=(wave,),
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(2.2e-6, 2e-6, None),
        partial_real_position=(0.3e-6, 0.0, -0.25e-6),
    )
    explicit = ((1, 7), (0, 6), (1, 4))
    _, objects, *_ = simulation.prepare_arrays(
        source,
        ((2, 6), (1, 5), (2, 3)),
        (0.0, 0.0, 0.25e-6),
        (4e-6, 4e-6, 0.0),
        primary,
        extra_monitors=[(padded, explicit), (physical, None)],
        direction="z-",
    )
    expected_primary = ((2, 6), (1, 5), (2, 3)) if sparse else ((0, 7), (0, 6), (0, 4))
    assert objects[primary.name].grid_slice_tuple == expected_primary
    assert objects[padded.name].grid_slice_tuple == explicit
    assert objects[physical.name].grid_slice_tuple == ((3, 6), (2, 4), (1, 2))
    assert padded.partial_grid_shape == (None, None, 1)
    assert padded.partial_real_position == (0.0, 0.0, 0.25e-6)
