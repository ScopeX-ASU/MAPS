from typing import Tuple

import torch
from .device_base import N_Ports

from core.utils import material_fn_dict
from pyutils.general import logger
import warnings

__all__ = ["Bending"]


class Bending(N_Ports):
    def __init__(
        self,
        material_r: str = "Si",  # waveguide material
        material_bg: str = "SiO2",  # background material
        sim_cfg: dict = {
            "border_width": [
                0,
                1.8,
                1.8,
                0,
            ],  # left, right, lower, upper, containing PML
            "PML": [0.5, 0.5],  # left/right, lower/upper
            "cell_size": None,
            "resolution": 50,
            "wl_cen": 1.55,
            "wl_width": 0,
            "n_wl": 1,
        },
        bending_region_size: Tuple[float] = (1.5, 1.5),
        port_len: Tuple[float] = (1.8, 1.8),
        port_width: Tuple[float] = (0.48, 0.48),
        device: torch.device = torch.device("cuda:0"),
    ):
        # ----------------------------------
        # |                                |
        # |                                |
        # |                                |
        # |[1]                             |
        # |                                |
        # |                                |
        # |              [0]               |
        # ----------------------------------
        if bending_region_size[0] != bending_region_size[1]:
            warnings.warn(
                "Bending region width and length are not equal, this is not a square bending region."
            )
        wl_cen = sim_cfg["wl_cen"]
        eps_r_fn = material_fn_dict[material_r]
        eps_bg_fn = material_fn_dict[material_bg]
        box_size = list(bending_region_size)
        port_cfgs = dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-(port_len[0] + box_size[0] / 2) / 2, 0],
                size=[port_len[0] + box_size[0] / 2, port_width[0]],
                eps=eps_r_fn(wl_cen),
            ),
            out_port_1=dict(
                type="box",
                direction="y",
                center=[0, (port_len[1] + box_size[1] / 2) / 2],
                size=[port_width[1], port_len[1] + box_size[1] / 2],
                eps=eps_r_fn(wl_cen),
            ),
        )

        geometry_cfgs = dict()

        design_region_cfgs = dict()
        design_region_cfgs["bending_region"] = dict(
            type="box",
            center=[0, 0],
            size=box_size,
            eps=eps_r_fn(wl_cen),
            eps_bg=eps_bg_fn(wl_cen),
        )

        super().__init__(
            eps_bg=eps_bg_fn(wl_cen),
            sim_cfg=sim_cfg,
            port_cfgs=port_cfgs,
            geometry_cfgs=geometry_cfgs,
            design_region_cfgs=design_region_cfgs,
            device=device,
        )

    def init_monitors(self, verbose: bool = True):
        rel_width = 4
        if verbose:
            logger.info("Start generating sources and monitors ...")
        src_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_slice_1",
            rel_loc=0.4,
            rel_width=rel_width,
        )
        refl_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="refl_slice_1",
            rel_loc=0.41,
            rel_width=rel_width,
        )
        out_slice = self.build_port_monitor_slice(
            port_name="out_port_1",
            slice_name="out_slice_1",
            rel_loc=0.6,
            rel_width=rel_width,
        )
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        radiation_monitor = self.build_radiation_monitor(monitor_name="rad_slice")
        return src_slice, out_slice, refl_slice, radiation_monitor

    def norm_run(self, verbose: bool = True):
        if verbose:
            logger.info("Start normalization run ...")
        # norm_run_sim_cfg = copy.deepcopy(self.sim_cfg)
        # norm_run_sim_cfg["numerical_solver"] = "solve_direct"
        norm_source_profiles = self.build_norm_sources(
            source_modes=("Hz1",),
            input_port_name="in_port_1",
            input_slice_name="in_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=True,
        )

        norm_refl_profiles = self.build_norm_sources(
            source_modes=("Hz1",),
            input_port_name="in_port_1",
            input_slice_name="refl_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )
        norm_monitor_profiles = self.build_norm_sources(
            source_modes=("Hz1",),
            input_port_name="out_port_1",
            input_slice_name="out_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )
        return norm_source_profiles, norm_refl_profiles, norm_monitor_profiles
