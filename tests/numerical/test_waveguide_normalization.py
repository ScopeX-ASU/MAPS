"""End-to-end 2D FDFD and 3D FDTD-X waveguide and Gaussian-beam power normalization tests.

Validates that:
1. 2D FDFD straight waveguides (`Ez1` and `Hz1`, uniform and nonuniform rectilinear grids)
   yield unit normalized `FluxObjective`, `EigenmodeObjective`, and `SMatrixObjective`.
2. 3D FDTD-X straight waveguides (uniform, nonuniform rectilinear, mirror-symmetric, and
   broadband Chebyshev mode launch) yield near-unit normalized `FluxObjective`,
   `EigenmodeObjective`, and `SMatrixObjective` (preventing any 4x amplitude / 16x power
   mis-calibration between source injection and discrete source-spectrum normalization).
3. 3D FDTD-X `GratingCoupler` Gaussian-beam excitation in a homogeneous medium launches
   near-unit incident Poynting flux across the input monitor plane.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _ensure_real_maps_modules():
    """Reload real core.fdfd / core.invdes packages if another test installed lightweight stubs."""
    fdfd_mod = sys.modules.get("core.fdfd")
    if fdfd_mod is not None and not hasattr(fdfd_mod, "__file__"):
        for name in [
            key
            for key in list(sys.modules)
            if key == "core.fdfd"
            or key.startswith("core.fdfd.")
            or key == "core.invdes.models"
            or key.startswith("core.invdes.models.")
        ]:
            sys.modules.pop(name, None)
    importlib.import_module("core.fdfd")
    importlib.import_module("core.invdes.models")


class _StraightWaveguide2D:
    """Factory helper building a straight two-port 2D waveguide on `N_Ports`."""

    @staticmethod
    def build(sim_cfg: dict, mode: str, device: torch.device):
        _ensure_real_maps_modules()
        from core.invdes.models.layers.device_base import N_Ports
        from core.utils import material_fn_dict

        class StraightWaveguide2D(N_Ports):
            def __init__(self):
                self.mode = mode
                wl = sim_cfg["wl_cen"]
                eps_wg = material_fn_dict["Si"](wl)
                eps_bg = material_fn_dict["SiO2"](wl)
                total_len = 4.6
                width = 0.5
                port_cfgs = dict(
                    in_port_1=dict(
                        type="box",
                        direction="x",
                        center=[0.0, 0.0],
                        size=[total_len, width],
                        eps=eps_wg,
                    ),
                    out_port_1=dict(
                        type="box",
                        direction="x",
                        center=[0.0, 0.0],
                        size=[total_len, width],
                        eps=eps_wg,
                    ),
                )
                super().__init__(
                    eps_bg=eps_bg,
                    sim_cfg=sim_cfg,
                    port_cfgs=port_cfgs,
                    geometry_cfgs=dict(),
                    design_region_cfgs=dict(),
                    device=device,
                )

            def init_monitors(self, verbose: bool = False):
                port_len = self.port_cfgs["in_port_1"]["size"][0]
                offset = 0.5 + self.sim_cfg["PML"][0]
                rel_loc = offset / port_len
                bw_y = self.sim_cfg["border_width"][2]
                rel_w = (
                    self.port_cfgs["in_port_1"]["size"][1]
                    + 2 * bw_y
                    - 2 * self.sim_cfg["PML"][1]
                ) / self.port_cfgs["in_port_1"]["size"][1]
                s_in = self.build_port_monitor_slice(
                    port_name="in_port_1",
                    slice_name="in_slice_1",
                    rel_loc=rel_loc,
                    rel_width=rel_w,
                )
                s_out = self.build_port_monitor_slice(
                    port_name="out_port_1",
                    slice_name="out_slice_1",
                    rel_loc=1.0 - rel_loc,
                    rel_width=rel_w,
                )
                self.build_port_monitor_slice(
                    port_name="in_port_1",
                    slice_name="source_power_slice",
                    rel_loc=(offset + 0.4) / port_len,
                    rel_width=rel_w,
                )
                return s_in, s_out

            def norm_run(self, verbose: bool = False):
                for s_name, p_name, req in (
                    ("in_slice_1", "in_port_1", True),
                    ("out_slice_1", "out_port_1", False),
                    ("source_power_slice", "in_port_1", False),
                ):
                    self.build_norm_sources(
                        source_modes=(self.mode,),
                        input_port_name=p_name,
                        input_slice_name=s_name,
                        wl_cen=self.sim_cfg["wl_cen"],
                        wl_width=self.sim_cfg["wl_width"],
                        n_wl=self.sim_cfg["n_wl"],
                        solver=self.sim_cfg["solver"],
                        plot=False,
                        require_sim=req,
                    )

        return StraightWaveguide2D()


@pytest.mark.integration
@pytest.mark.numerical
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["Ez1", "Hz1"])
@pytest.mark.parametrize("nonuniform", [False, True], ids=["uniform", "nonuniform"])
def test_2d_fdfd_straight_waveguide_flux_and_eigenmode_normalization(
    require_cuda, tmp_path, mode: str, nonuniform: bool
) -> None:
    """2D FDFD straight waveguide must produce unit normalized flux and eigenmode transmission."""
    _ensure_real_maps_modules()
    from core.invdes.models.base_optimization import (
        BaseOptimization,
        DefaultSimulationConfig,
    )

    dev = torch.device("cuda:0")
    wl = 1.55
    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            resolution=50,
            wl_cen=wl,
            wl_width=0.0,
            n_wl=1,
            border_width=[0.0, 0.0, 1.5, 1.5],
            PML=[0.8, 0.8],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            plot_root=str(tmp_path),
        )
    )
    if nonuniform:
        sim_cfg["optical_grid"] = {
            "mode": "fdtdx_rectilinear_nonuniform",
            "wavelength": wl,
            "grid_x": {"type": "auto", "min_steps_per_wvl": 25},
            "grid_y": {"type": "auto", "min_steps_per_wvl": 25},
        }

    wg = _StraightWaveguide2D.build(sim_cfg=sim_cfg, mode=mode, device=dev)
    obj_cfgs = dict(
        source_power=dict(
            weight=1.0,
            in_slice_name="in_slice_1",
            out_slice_name="source_power_slice",
            in_mode=mode,
            wl=[wl],
            temp=[300],
            out_modes=(mode,),
            type="eigenmode",
            energy=True,
            direction="x+",
            requires_grad=False,
        ),
        output_power=dict(
            weight=0.0,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            in_mode=mode,
            wl=[wl],
            temp=[300],
            out_modes=(mode,),
            type="eigenmode",
            energy=True,
            direction="x+",
            requires_grad=False,
        ),
        output_amp=dict(
            weight=0.0,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            in_mode=mode,
            wl=[wl],
            temp=[300],
            out_modes=(mode,),
            type="eigenmode",
            energy=False,
            direction="x+",
            requires_grad=False,
        ),
        output_smatrix=dict(
            weight=0.0,
            in_slice_name=["in_slice_1"],
            out_slice_name=["out_slice_1"],
            in_mode=mode,
            wl=[wl],
            temp=[300],
            out_modes=(mode,),
            type="smatrix",
            energy=True,
            direction=["x+"],
            requires_grad=False,
        ),
        output_flux=dict(
            weight=0.0,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            in_mode=mode,
            wl=[wl],
            temp=[300],
            out_modes=(mode,),
            type="flux",
            direction="x",
            requires_grad=False,
        ),
        _fusion_func=lambda bd: (bd["source_power"]["value"], {}),
        override=True,
    )

    opt = BaseOptimization(
        device=wg,
        hr_device=wg,
        design_region_param_cfgs={},
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=dev,
        verbose=False,
    )
    eps_t = torch.as_tensor(wg.epsilon_map, dtype=torch.float32, device=dev)
    opt.objective(eps_t, mode="forward")
    bd = opt.objective.breakdown

    src_eig = float(bd["source_power"]["value"].detach().cpu())
    out_eig = float(bd["output_power"]["value"].detach().cpu())
    out_amp_sq = float(bd["output_amp"]["value"].detach().abs().square().cpu())
    out_smat = float(bd["output_smatrix"]["value"].detach().cpu())
    out_flux = float(bd["output_flux"]["value"].detach().cpu())

    assert out_flux == pytest.approx(1.0, abs=5e-3)
    assert src_eig == pytest.approx(1.0, abs=3e-2)
    assert out_eig == pytest.approx(1.0, abs=3e-2)
    assert out_amp_sq == pytest.approx(out_eig, rel=1e-5, abs=1e-6)
    assert out_smat == pytest.approx(out_eig, rel=1e-5, abs=1e-6)
    assert out_eig / src_eig == pytest.approx(1.0, abs=1.5e-2)


@pytest.mark.fdtdx
@pytest.mark.integration
@pytest.mark.numerical
@pytest.mark.gpu
@pytest.mark.parametrize(
    ("resolution", "nonuniform", "symmetry", "n_wl", "wl_width", "abs_tol"),
    [
        (30, False, (0, 0, 0), 1, 0.0, 3e-2),
        (30, True, (0, 0, 0), 3, 0.04, 3e-2),
        # Under y-mirror symmetry, fdtdx.core.physics.modes.compute_mode_symmetry_reduced
        # renormalizes over the kept half-plane where staggered Yee discretization has an
        # O(dy) half-plane split offset (~6-7% at 20 nm, ~10% at 33 nm) that vanishes with
        # refinement while preserving exact lossless ratio out_pow / src_pow == 1.
        (50, False, (0, -1, 0), 1, 0.0, 8e-2),
    ],
    ids=["uniform_single_wl", "nonuniform_broadband", "uniform_pec_symmetry"],
)
def test_3d_fdtdx_straight_waveguide_flux_and_eigenmode_normalization(
    require_cuda,
    tmp_path,
    resolution: int,
    nonuniform: bool,
    symmetry: tuple[int, int, int],
    n_wl: int,
    wl_width: float,
    abs_tol: float,
) -> None:
    """3D FDTD-X straight waveguide must launch unit power across flux, eigenmode, and S-matrix objectives."""
    pytest.importorskip("fdtdx")
    _ensure_real_maps_modules()
    from core.invdes.examples.waveguide_3d import StraightWaveguide3D
    from core.invdes.models.base_optimization import (
        BaseOptimization,
        DefaultSimulationConfig,
    )

    dev = torch.device("cuda:0")
    wl_cen = 1.31
    mode = "Hz1"
    wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl).tolist()

    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="fdtdx",
            resolution=resolution,
            subpixel=False,
            symmetry=symmetry,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
            max_time=2.5e-13,
            border_width=[0, 0, 1.2, 1.2, 1.0, 1.0],
            PML=[0.6, 0.6, 0.6],
            cell_size=None,
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            plot_root=str(tmp_path),
        )
    )
    if nonuniform:
        sim_cfg["optical_grid"] = {
            "mode": "fdtdx_rectilinear_nonuniform",
            "wavelength": wl_cen,
            "grid_x": {"type": "auto", "min_steps_per_wvl": 20},
            "grid_y": {"type": "auto", "min_steps_per_wvl": 20},
            "grid_z": {"type": "auto", "min_steps_per_wvl": 20},
        }

    wg = StraightWaveguide3D(
        material="Si",
        background_material="SiO2",
        thickness=0.22,
        waveguide_width=0.5,
        waveguide_length=0.8,
        port_length=1.1,
        sim_cfg=sim_cfg,
        mode=mode,
        device=dev,
    )
    obj_cfgs = dict(
        source_power=dict(
            weight=1.0,
            in_slice_name="in_slice_1",
            out_slice_name="source_power_slice",
            in_mode=mode,
            wl=wls,
            temp=[300],
            out_modes=(mode,),
            type="eigenmode",
            energy=True,
            direction="x+",
            requires_grad=False,
            requires_adjoint=False,
        ),
        output_power=dict(
            weight=0.0,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            in_mode=mode,
            wl=wls,
            temp=[300],
            out_modes=(mode,),
            type="eigenmode",
            energy=True,
            direction="x+",
            requires_grad=False,
            requires_adjoint=False,
        ),
        output_smatrix_amp=dict(
            weight=0.0,
            in_slice_name=["in_slice_1"],
            out_slice_name=["out_slice_1"],
            in_mode=mode,
            wl=wls,
            temp=[300],
            out_modes=(mode,),
            type="smatrix",
            energy=False,
            direction=["x+"],
            requires_grad=False,
            requires_adjoint=False,
        ),
        output_smatrix_pow=dict(
            weight=0.0,
            in_slice_name=["in_slice_1"],
            out_slice_name=["out_slice_1"],
            in_mode=mode,
            wl=wls,
            temp=[300],
            out_modes=(mode,),
            type="smatrix",
            energy=True,
            direction=["x+"],
            requires_grad=False,
            requires_adjoint=False,
        ),
        output_flux=dict(
            weight=0.0,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            in_mode=mode,
            wl=wls,
            temp=[300],
            out_modes=(mode,),
            type="flux",
            direction="x",
            requires_grad=False,
            requires_adjoint=False,
        ),
        _fusion_func=lambda bd: (bd["source_power"]["value"].mean(), {}),
        override=True,
    )

    opt = BaseOptimization(
        device=wg,
        hr_device=wg,
        design_region_param_cfgs={},
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=dev,
        verbose=False,
    )
    eps_t = torch.as_tensor(wg.epsilon_map, dtype=torch.float32, device=dev)
    opt.objective(eps_t, mode="forward")
    bd = opt.objective.breakdown

    src_pow = bd["source_power"]["value"].detach().cpu().numpy()
    out_pow = bd["output_power"]["value"].detach().cpu().numpy()
    smat_amp_sq = bd["output_smatrix_amp"]["value"].detach().abs().square().cpu().numpy()
    smat_pow = bd["output_smatrix_pow"]["value"].detach().cpu().numpy()
    out_flux = bd["output_flux"]["value"].detach().cpu().numpy()

    np.testing.assert_allclose(src_pow, 1.0, atol=abs_tol)
    np.testing.assert_allclose(out_pow, 1.0, atol=abs_tol)
    np.testing.assert_allclose(out_flux, 1.0, atol=abs_tol)
    np.testing.assert_allclose(smat_amp_sq, out_pow, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(smat_pow, out_pow, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out_pow / src_pow, 1.0, atol=1.5e-2)


@pytest.mark.fdtdx
@pytest.mark.integration
@pytest.mark.numerical
@pytest.mark.gpu
def test_3d_fdtdx_gaussian_beam_launches_unit_incident_flux(
    require_cuda, tmp_path
) -> None:
    """Homogeneous 3D GratingCoupler Gaussian-beam excitation must measure near-unit incident flux."""
    pytest.importorskip("fdtdx")
    _ensure_real_maps_modules()
    from core.invdes.models.base_optimization import (
        BaseOptimization,
        DefaultSimulationConfig,
    )
    from core.invdes.models.layers.grating_coupler import GratingCoupler

    dev = torch.device("cuda:0")
    wl_cen = 1.31
    eps_hom = 1.444**2
    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="fdtdx",
            subpixel=False,
            symmetry=(0, -1, 0),
            max_time=8e-13,
            border_width=[1.9, 0, 1.0, 1.0, 0.0, 0.0],
            resolution=25,
            PML=[0.6, 0.6, 0.6],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=0.0,
            n_wl=1,
            plot_root=str(tmp_path),
        )
    )
    gc = GratingCoupler(
        material_r=eps_hom,
        material_layers=[eps_hom],
        material_buried=eps_hom,
        material_cladding=eps_hom,
        thickness_layers=[0.22],
        etch_depth_layers=[0.22],
        gap_layers=(0.0,),
        sim_cfg=sim_cfg,
        design_region_size=(4.0, 4.0),
        port_len=1.9,
        port_width=0.5,
        port_height=0.22,
        top_port_len=2.0,
        substrate_thickness=0.5,
        substrate_gap=0.8,
        waist_radius=1.0,
        waist_distance=0.0,
        is_3d=True,
        in_mode="Hz2",
        out_mode="Hz1",
        device=dev,
    )
    obj_cfgs = dict(
        input_flux=dict(
            weight=1.0,
            in_slice_name="in_slice_1",
            out_slice_name="in_monitor_slice_1",
            in_mode="Hz2",
            wl=[wl_cen],
            temp=[300],
            out_modes=("Hz1",),
            type="flux",
            direction="z",
            requires_grad=False,
            requires_adjoint=False,
        ),
        refl_flux=dict(
            weight=0.0,
            in_slice_name="in_slice_1",
            out_slice_name="refl_slice_1",
            in_mode="Hz2",
            wl=[wl_cen],
            temp=[300],
            out_modes=("Hz1",),
            type="flux",
            direction="z",
            requires_grad=False,
            requires_adjoint=False,
        ),
        _fusion_func=lambda bd: (bd["input_flux"]["value"], {}),
        override=True,
    )
    opt = BaseOptimization(
        device=gc,
        hr_device=gc,
        design_region_param_cfgs={},
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=dev,
        verbose=False,
    )
    eps_t = torch.full(
        gc.epsilon_map.shape, eps_hom, dtype=torch.float32, device=dev
    )
    opt.objective(eps_t, mode="forward")
    bd = opt.objective.breakdown

    input_flux = float(bd["input_flux"]["value"].detach().cpu())
    refl_flux = float(bd["refl_flux"]["value"].detach().cpu())

    assert input_flux == pytest.approx(1.0, abs=4e-2)
    assert refl_flux < 5e-3
