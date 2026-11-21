"""MAPS-level 3D Bending forward regression for the public FDTD-X backend.

This test intentionally exercises the real MAPS device path rather than a standalone
FDTD-X scene: Bending -> device monitor/mode/source construction -> fdtd3d -> fields.
It is opt-in because the full MAPS runtime and a CUDA-capable environment are expected.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fdtdx")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from core.fdtd.fdtd import fdtd3d
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers.bending import Bending
from core.utils import get_eigenmode_coefficients_3d

_WL = 1.55
_MODE = "Hz1"
_EPS_SI = 3.48**2


pytestmark = [
    pytest.mark.fdtdx,
    pytest.mark.integration,
    pytest.mark.numerical,
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("MAPS_RUN_FDTDX_BENDING", "0") != "1",
        reason="opt-in MAPS 3D FDTD-X bending regression",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="MAPS 3D FDTD-X bending regression requires CUDA",
    ),
]


def _build_bending() -> tuple[Bending, torch.Tensor]:
    """Construct a reduced version of the canonical ``bending_3d.py`` device."""
    device = torch.device("cuda:0")
    port_len = 1.0
    resolution = int(os.environ.get("MAPS_FDTDX_BENDING_RESOLUTION", "30"))

    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="fdtdx",
            border_width=[0, port_len, port_len, 0, 0.5, 0.5],
            max_time=float(os.environ.get("MAPS_FDTDX_BENDING_MAX_TIME", "1e-13")),
            subpixel=True,
            resolution=resolution,
            plot_root="./figs/fdtdx_bending_regression",
            PML=[0.2, 0.2, 0.2],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=_WL,
            wl_width=0.0,
            n_wl=1,
        )
    )

    bend = Bending(
        material_r1="Si",
        material_r2="SiO2",
        material_bg="SiO2",
        sim_cfg=sim_cfg,
        bending_region_size=(1.5, 1.5, 0.22),
        port_len=(port_len, port_len),
        port_width=(0.5, 0.5),
        port_height=(0.22, 0.22),
        LER_A=0.0,
        LER_Lc=0.0,
        sidewall_angle=0.0,
        is_3d=True,
        mode=_MODE,
        device=device,
        verbose=False,
    )

    # Use a deterministic all-Si design region. This keeps the regression focused
    # on forward device/FDTD integration rather than parametrization behavior.
    eps = bend.epsilon_map.copy()
    eps[bend.design_region_masks["bending_region"]] = _EPS_SI
    eps = np.asarray(eps.real if np.iscomplexobj(eps) else eps, dtype=np.float32)

    bend.init_monitors(verbose=False)
    bend.norm_run(verbose=False)
    return bend, torch.as_tensor(eps, dtype=torch.float32, device=device)


def test_maps_bending_forward_fields_are_finite_and_nonzero() -> None:
    """Run the canonical MAPS Bending path and validate fields plus modal transmission."""
    bend, eps = _build_bending()

    # Use MAPS's established structured native/export-grid metadata builder. The
    # lower-level raster-grid metadata is not passed directly to fdtd3d because
    # fdtd3d expects the solver-facing ``native``/``export`` contract.
    field_grid_metadata = bend._build_fdtdx_native_grid_metadata()

    sim = fdtd3d(
        wl_cen=_WL,
        wl_width=0.0,
        n_wl=1,
        dL=bend.grid_step * 1e-6,
        eps_r=eps,
        electrical_conductivity_map=getattr(bend, "electrical_conductivity_map", None),
        optical_grid_metadata=field_grid_metadata,
        port_sources_dict=bend.port_sources_dict,
        port_monitor_slices=bend.port_monitor_slices,
        port_monitor_slices_info=bend.port_monitor_slices_info,
        npml=bend.NPML,
        max_time=bend.sim_cfg["max_time"],
        device=bend.device,
    )

    # Keep the forward observable in the native physical grid because the MAPS
    # eigenmode-overlap convention consumes the native monitor and native weights.
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
    reference_shape = tuple(fields[0].shape)
    assert len(reference_shape) == 4
    assert reference_shape[0] == 1

    maxima = []
    for field in fields:
        assert tuple(field.shape) == reference_shape
        assert torch.is_complex(field)
        assert bool(torch.all(torch.isfinite(field)))
        maxima.append(float(field.abs().max().detach().cpu()))

    # The forward run must contain a propagated electromagnetic field, not merely
    # return correctly shaped zero buffers.
    assert max(maxima[:3]) > 0.0
    assert max(maxima[3:]) > 0.0

    # Bending specifically exercises orthogonal MAPS ports. Preserve that contract.
    assert bend.port_cfgs["in_port_1"]["direction"] == "x"
    assert bend.port_cfgs["out_port_1"]["direction"] == "y"
    assert {"in_slice_1", "refl_slice_1", "out_slice_1"}.issubset(
        bend.port_monitor_slices
    )

    # Reuse MAPS's existing 3D eigenmode-overlap convention. Bending.norm_run()
    # populates the output-port reference profile used by the objective layer.
    out_profile = bend.port_sources_dict["out_slice_1"][_WL, _MODE]
    _, ht_m, et_m, norm_power, _ = out_profile
    native_monitor = bend.port_monitor_slices_native["out_slice_1"]
    native_grid = field_grid_metadata["native"]

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
        monitor=native_monitor,
        grid_step=1.0,
        energy=True,
        direction="y+",
        grid_metadata=native_grid,
        cell_weights=None,
    )

    assert np.isfinite(float(norm_power))
    assert float(norm_power) > 0.0
    assert bool(torch.isfinite(s_forward))
    assert bool(torch.isfinite(s_backward))
    assert float(s_forward) >= 0.0
    assert float(s_backward) >= 0.0

    transmission = float(s_forward.detach().cpu())
    reflection = float(s_backward.detach().cpu())
    print(
        f"MAPS FDTD-X Bending normalized output-mode power: {transmission:.10g}; "
        f"backward-mode power: {reflection:.10g}"
    )
