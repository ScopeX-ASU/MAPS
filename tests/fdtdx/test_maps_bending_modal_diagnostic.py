"""Diagnostic modal checks for the MAPS FDTD-X Bending regression.

This test is intentionally separate from the regression threshold test. It helps
validate whether a low bend transmission comes from the device physics or from
source/mode normalization.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("fdtdx")
torch = pytest.importorskip("torch")

_THIS_DIR = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "test_maps_bending_forward",
    _THIS_DIR / "test_maps_bending_forward.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_FORWARD_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FORWARD_MODULE)

_MODE = _FORWARD_MODULE._MODE
_WL = _FORWARD_MODULE._WL
_build_bending = _FORWARD_MODULE._build_bending

from core.fdtd.fdtd import fdtd3d
from core.utils import get_eigenmode_coefficients_3d


def _get_profile(port_sources_dict, slice_name):
    """Retrieve a mode profile using MAPS FDTD-X source-key conventions."""
    profiles = port_sources_dict[slice_name]
    candidates = [
        ((_WL, 0.0, 1), _MODE),
        (_WL, _MODE),
        (_WL, _MODE[:2], int(_MODE[2:])),
        (_WL, _MODE[:2]),
    ]
    for key in candidates:
        if key in profiles:
            return profiles[key]
    raise KeyError(
        f"Cannot find mode profile for {slice_name}; available keys={list(profiles.keys())}"
    )


@pytest.mark.fdtdx
@pytest.mark.integration
@pytest.mark.numerical
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("MAPS_RUN_FDTDX_BENDING", "0") != "1",
    reason="opt-in MAPS 3D FDTD-X bending regression",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MAPS 3D FDTD-X bending regression requires CUDA",
)
def test_maps_bending_input_modal_normalization_diagnostic() -> None:
    """Report input/output modal powers without freezing a numerical threshold."""
    bend, eps = _build_bending()
    grid_metadata = bend._build_fdtdx_native_grid_metadata()

    sim = fdtd3d(
        wl_cen=_WL,
        wl_width=0.0,
        n_wl=1,
        dL=bend.grid_step * 1e-6,
        eps_r=eps,
        optical_grid_metadata=grid_metadata,
        port_sources_dict=bend.port_sources_dict,
        port_monitor_slices=bend.port_monitor_slices,
        port_monitor_slices_info=bend.port_monitor_slices_info,
        npml=bend.NPML,
        max_time=bend.sim_cfg["max_time"],
        device=bend.device,
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

    Ex, Ey, Ez, Hx, Hy, Hz = (field[0] for field in fields)
    native_grid = grid_metadata["native"]

    results = {}
    for name, monitor_name, direction in (
        ("input", "in_slice_1", "x+"),
        ("output", "out_slice_1", "y+"),
        ("reflection", "refl_slice_1", "x-"),
    ):
        _, ht_m, et_m, *_ = _get_profile(bend.port_sources_dict, monitor_name)
        if ht_m is None or et_m is None:
            mode_profiles = bend.insert_monitors_fdtd3d(
                monitor_name,
                bend.epsilon_map,
                "mode",
                bend.port_monitor_slices[monitor_name],
                wl_cen=_WL,
                wl_width=0.0,
                n_wl=1,
                source_modes=(_MODE,),
                direction=direction,
                slice_info=bend.port_monitor_slices_info[monitor_name],
                on_native_grid=True,
            )
            _, ht_m, et_m, *_ = _get_profile({monitor_name: mode_profiles}, monitor_name)
        forward, backward = get_eigenmode_coefficients_3d(
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            ht_m,
            et_m,
            monitor=bend.port_monitor_slices_native[monitor_name],
            grid_step=1.0,
            energy=True,
            direction=direction,
            grid_metadata=native_grid,
        )
        results[name] = (float(forward.detach().cpu()), float(backward.detach().cpu()))

    print(f"MAPS FDTD-X Bending modal diagnostic: {results}")

    for values in results.values():
        assert all(value >= 0.0 for value in values)
        assert all(value == value for value in values)
