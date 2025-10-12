import copy
import warnings
from typing import Tuple

import torch
from pyutils.general import logger

from core.utils import material_fn_dict

from .device_base import N_Ports

__all__ = ["Crossing"]


class Crossing(N_Ports):
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
        box_size: Tuple[float] = (1.5, 1.5),
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
        if box_size[0] != box_size[1]:
            warnings.warn(
                "Crossing region width and length are not equal, this is not a square crossing region."
            )
        wl_cen = sim_cfg["wl_cen"]
        eps_r_fn = material_fn_dict[material_r]
        # eps_r_fn = lambda wl: 3.48**2 # effective index of Si waveguide
        # eps_r_fn = lambda wl: 2.848152 ** 2  # effective index of Si waveguide
        eps_bg_fn = material_fn_dict[material_bg]
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
                direction="x",
                center=[(port_len[0] + box_size[0] / 2) / 2, 0],
                size=[port_len[0] + box_size[0] / 2, port_width[0]],
                eps=eps_r_fn(wl_cen),
            ),
            top_port=dict(
                type="box",
                direction="y",
                center=[0, (port_len[1] + box_size[1] / 2) / 2],
                size=[port_width[1], port_len[1] + box_size[1] / 2],
                eps=eps_r_fn(wl_cen),
            ),
            bot_port=dict(
                type="box",
                direction="y",
                center=[0, -(port_len[1] + box_size[1] / 2) / 2],
                size=[port_width[1], port_len[1] + box_size[1] / 2],
                eps=eps_r_fn(wl_cen),
            ),
        )

        geometry_cfgs = dict()

        design_region_cfgs = dict()
        design_region_cfgs["crossing_region"] = dict(
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
        rel_width = 3
        pml_x, pml_y = self.sim_cfg["PML"]
        offset_x, offset_y = 0.2 + pml_x, 0.2 + pml_y
        port_len_x = self.port_cfgs["in_port_1"]["size"][0]
        port_len_y = self.port_cfgs["top_port"]["size"][1]
        if verbose:
            logger.info("Start generating sources and monitors ...")
        src_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_slice_1",
            rel_loc=offset_x / port_len_x,
            rel_width=rel_width,
        )
        refl_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="refl_slice_1",
            rel_loc=(offset_x + 0.05) / port_len_x,
            rel_width=rel_width,
        )
        out_slice = self.build_port_monitor_slice(
            port_name="out_port_1",
            slice_name="out_slice_1",
            rel_loc=1 - offset_x / port_len_x,
            rel_width=rel_width,
        )
        top_slice = self.build_port_monitor_slice(
            port_name="top_port",
            slice_name="top_slice",
            rel_loc=1 - offset_y / port_len_y,
            rel_width=rel_width,
        )
        bot_slice = self.build_port_monitor_slice(
            port_name="bot_port",
            slice_name="bot_slice",
            rel_loc=offset_y / port_len_y,
            rel_width=rel_width,
        )
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        radiation_monitor = self.build_radiation_monitor(monitor_name="rad_slice")
        return src_slice, out_slice, refl_slice, top_slice, bot_slice, radiation_monitor

    def norm_run(self, verbose: bool = True):
        if verbose:
            logger.info("Start normalization run ...")
        # norm_run_sim_cfg = copy.deepcopy(self.sim_cfg)
        # norm_run_sim_cfg["numerical_solver"] = "solve_direct"
        norm_source_profiles = self.build_norm_sources(
            source_modes=("Ez1",),
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
            source_modes=("Ez1",),
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
            source_modes=("Ez1",),
            input_port_name="out_port_1",
            input_slice_name="out_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )
        norm_top_profiles = self.build_norm_sources(
            source_modes=("Ez1",),
            input_port_name="top_port",
            input_slice_name="top_slice",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )
        norm_bot_profiles = self.build_norm_sources(
            source_modes=("Ez1",),
            input_port_name="bot_port",
            input_slice_name="bot_slice",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )
        return (
            norm_source_profiles,
            norm_refl_profiles,
            norm_monitor_profiles,
            norm_top_profiles,
            norm_bot_profiles,
        )
