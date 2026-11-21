"""MAPS-level 3D straight waveguide forward regression for FDTD-X.

This test follows core/invdes/examples/waveguide_3d.py but removes the
optimization wrapper. It validates the forward-only MAPS device path:
StraightWaveguide3D -> N_Ports/device_base -> FDTD-X -> eigenmode power.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fdtdx")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from core.fdtd.fdtd import fdtd3d
from core.invdes.examples.waveguide_3d import StraightWaveguide3D
from core.utils import get_eigenmode_coefficients_3d

_WL = 1.31
_MODE = "Hz1"
_EXPECTED_FORWARD_MODAL_POWER = 0.991990447
_FORWARD_MODAL_POWER_ABS_TOL = 5e-3


pytestmark = [
    pytest.mark.fdtdx,
    pytest.mark.integration,
    pytest.mark.numerical,
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("MAPS_RUN_FDTDX_WAVEGUIDE3D", "0") != "1",
        reason="opt-in MAPS 3D FDTD-X waveguide regression",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="MAPS 3D FDTD-X waveguide regression requires CUDA",
    ),
]


def _build_waveguide():
    device = torch.device("cuda:0")
    sim_cfg = dict(
        solver="fdtdx",
        resolution=50,
        subpixel=False,
        wl_cen=_WL,
        wl_width=0.0,
        n_wl=1,
        max_time=3e-13,
        border_width=[0, 0, 1.5, 1.5, 1.3, 1.3],
        PML=[_WL / 2 + 0.2] * 3,
        cell_size=None,
        plot_root=".",
        optical_grid={
            "mode": "fdtdx_rectilinear_nonuniform",
            "wavelength": _WL,
            "grid_x": {"type": "auto", "min_steps_per_wvl": 25},
            "grid_y": {"type": "auto", "min_steps_per_wvl": 25},
            "grid_z": {"type": "auto", "min_steps_per_wvl": 25},
        },
    )

    return StraightWaveguide3D(
        material="Si",
        background_material="SiO2",
        thickness=0.22,
        waveguide_width=0.5,
        waveguide_length=1.0,
        sim_cfg=sim_cfg,
        mode=_MODE,
        device=device,
    )


def test_maps_waveguide3d_forward_fields_and_transmission():
    """Validate straight-waveguide fields and near-unity mode transmission."""
    device = _build_waveguide()
    device.init_monitors(verbose=False)
    device.norm_run(verbose=False)

    eps = torch.as_tensor(
        device.epsilon_map,
        dtype=torch.float32,
        device=device.device,
    )

    sim = fdtd3d(
        wl_cen=_WL,
        wl_width=0.0,
        n_wl=1,
        dL=device.grid_step * 1e-6,
        eps_r=eps,
        optical_grid_metadata=device._build_fdtdx_native_grid_metadata(),
        port_sources_dict=device.port_sources_dict,
        port_monitor_slices=device.port_monitor_slices,
        port_monitor_slices_info=device.port_monitor_slices_info,
        npml=device.NPML,
        max_time=device.sim_cfg["max_time"],
        device=device.device,
    )

    fields = sim.solve(
        input_slice_name="in_slice_1",
        wl_cen=_WL,
        wl_width=0.0,
        n_wl=1,
        mode=_MODE,
        eps_r=eps,
        interpolate_to_export_grid=False,
    )

    assert len(fields) == 6
    for field in fields:
        assert torch.is_complex(field)
        assert bool(torch.all(torch.isfinite(field)))

    assert "out_slice_1" in device.port_sources_dict
    assert "in_slice_1" in device.port_sources_dict

    _, ht_m, et_m, norm_power, _ = device.port_sources_dict["out_slice_1"][_WL, _MODE]
    assert np.isfinite(float(norm_power))
    assert float(norm_power) > 0.0

    Ex, Ey, Ez, Hx, Hy, Hz = (field[0] for field in fields)
    s_forward, s_backward = get_eigenmode_coefficients_3d(
        Ex,
        Ey,
        Ez,
        Hx,
        Hy,
        Hz,
        ht_m,
        et_m,
        monitor=device.port_monitor_slices_native["out_slice_1"],
        grid_step=1.0,
        energy=True,
        direction="x+",
        grid_metadata=device._build_fdtdx_native_grid_metadata()["native"],
        cell_weights=None,
    )

    transmission = float(s_forward.detach().cpu())
    reflection = float(s_backward.detach().cpu())
    assert np.isfinite(transmission)
    assert np.isfinite(reflection)
    assert transmission >= 0.0
    assert reflection >= 0.0
    assert transmission > reflection
    assert transmission == pytest.approx(
        _EXPECTED_FORWARD_MODAL_POWER,
        abs=_FORWARD_MODAL_POWER_ABS_TOL,
    )

    print(
        f"MAPS FDTD-X straight-waveguide modal power: forward={transmission:.10g}; "
        f"backward={reflection:.10g}; norm_power={float(norm_power):.10g}"
    )
