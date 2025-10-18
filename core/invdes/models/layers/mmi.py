import copy
import warnings
from functools import partial
from typing import Tuple

import numpy as np
import torch
from pyutils.general import logger

from core.utils import material_fn_dict

from .device_base import N_Ports

__all__ = ["MMI"]


class MMI(N_Ports):
    def __init__(
        self,
        material_r1: str = "Si_eff",  # waveguide material
        material_r2: str = "SiO2",  # waveguide material
        thickness_r1: float = 0.22,  # waveguide thickness
        thickness_r2: float = 0.0,  # waveguide thickness
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
        num_inports: int = 1,
        num_outports: int = 2,
        port_box_margin: float = 0.2,
        device: torch.device = torch.device("cuda:0"),
    ):
        #   ----------------------------------
        #   |                                |
        # --|                                |--
        #   |                                |
        # --|[1]                             |--
        #   |                                |
        # --|                                |--
        #   |              [0]               |
        #   ----------------------------------
        assert (
            box_size[1] - 2 * port_box_margin >= num_inports * port_width[0]
        ), "box_size[1] should be larger than num_inports * port_width[0]"

        assert (
            box_size[1] - 2 * port_box_margin >= num_outports * port_width[1]
        ), "box_size[1] should be larger than num_outports * port_width[1]"
        self.num_inports = num_inports
        self.num_outports = num_outports
        wl_cen = sim_cfg["wl_cen"]
        if isinstance(material_r1, str):
            eps_r1_fn = material_fn_dict[material_r1]
            if "_eff" in material_r1:
                eps_r1_fn = partial(eps_r1_fn, thickness=thickness_r1)
        else:
            eps_r1_fn = lambda wl: material_r1

        if isinstance(material_r2, str):
            eps_r2_fn = material_fn_dict[material_r2]
            if "_eff" in material_r2:
                eps_r2_fn = partial(eps_r2_fn, thickness=thickness_r2)
        else:
            eps_r2_fn = lambda wl: material_r2

        eps_bg_fn = material_fn_dict[material_bg]

        port_cfgs = dict()

        inport_y_coords = np.linspace(
            -box_size[1] / 2 + port_width[0] / 2 + port_box_margin,
            box_size[1] / 2 - port_width[0] / 2 - port_box_margin,
            num_inports,
        )
        for i in range(1, num_inports + 1):
            port_cfgs[f"in_port_{i}"] = dict(
                type="box",
                direction="x",
                center=[-(port_len[0] + box_size[0] / 2) / 2, inport_y_coords[i - 1]],
                size=[port_len[0] + box_size[0] / 2, port_width[0]],
                eps=eps_r1_fn(wl_cen),
            )

        outport_y_coords = np.linspace(
            -box_size[1] / 2 + port_width[1] / 2 + port_box_margin,
            box_size[1] / 2 - port_width[1] / 2 - port_box_margin,
            num_outports,
        )
        for i in range(1, num_outports + 1):
            port_cfgs[f"out_port_{i}"] = dict(
                type="box",
                direction="x",
                center=[(port_len[0] + box_size[0] / 2) / 2, outport_y_coords[i - 1]],
                size=[port_len[0] + box_size[0] / 2, port_width[0]],
                eps=eps_r1_fn(wl_cen),
            )

        geometry_cfgs = dict()

        design_region_cfgs = dict()
        design_region_cfgs["mmi_region"] = dict(
            type="box",
            center=[0, 0],
            size=box_size,
            eps=eps_r1_fn(wl_cen),
            eps_bg=eps_r2_fn(wl_cen),
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
        rel_width = 2
        if verbose:
            logger.info("Start generating sources and monitors ...")
        pml = self.sim_cfg["PML"][0]
        port_len = self.port_cfgs["in_port_1"]["size"][0]
        offset = 0.2 + pml

        src_slices = [
            self.build_port_monitor_slice(
                port_name=f"in_port_{i}",
                slice_name=f"in_slice_{i}",
                rel_loc=offset / port_len,
                rel_width=rel_width,
            )
            for i in range(1, self.num_inports + 1)
        ]
        refl_slices = [
            self.build_port_monitor_slice(
                port_name=f"in_port_{i}",
                slice_name=f"refl_slice_{i}",
                rel_loc=(offset + 0.1) / port_len,
                rel_width=rel_width,
            )
            for i in range(1, self.num_inports + 1)
        ]

        out_slices = [
            self.build_port_monitor_slice(
                port_name=f"out_port_{i}",
                slice_name=f"out_slice_{i}",
                rel_loc=1 - offset / port_len,
                rel_width=rel_width,
            )
            for i in range(1, self.num_outports + 1)
        ]
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        radiation_monitor = self.build_radiation_monitor(monitor_name="rad_monitor")
        return src_slices, out_slices, refl_slices, radiation_monitor

    def norm_run(self, verbose: bool = True):
        if verbose:
            logger.info("Start normalization run ...")
        # norm_run_sim_cfg = copy.deepcopy(self.sim_cfg)
        # norm_run_sim_cfg["numerical_solver"] = "solve_direct"
        norm_source_profiles = [
            self.build_norm_sources(
                source_modes=("Ez1",),
                input_port_name=f"in_port_{i}",
                input_slice_name=f"in_slice_{i}",
                wl_cen=self.sim_cfg["wl_cen"],
                wl_width=self.sim_cfg["wl_width"],
                n_wl=self.sim_cfg["n_wl"],
                # solver=self.sim_cfg["solver"],
                solver="ceviche",
                plot=True,
                require_sim=True,
            )
            for i in range(1, self.num_inports + 1)
        ]

        norm_refl_profiles = [
            self.build_norm_sources(
                source_modes=("Ez1",),
                input_port_name=f"in_port_{i}",
                input_slice_name=f"refl_slice_{i}",
                wl_cen=self.sim_cfg["wl_cen"],
                wl_width=self.sim_cfg["wl_width"],
                n_wl=self.sim_cfg["n_wl"],
                # solver=self.sim_cfg["solver"],
                solver="ceviche",
                plot=True,
                require_sim=False,
            )
            for i in range(1, self.num_inports + 1)
        ]
        norm_monitor_profiles = [
            self.build_norm_sources(
                source_modes=("Ez1",),
                input_port_name=f"out_port_{i}",
                input_slice_name=f"out_slice_{i}",
                wl_cen=self.sim_cfg["wl_cen"],
                wl_width=self.sim_cfg["wl_width"],
                n_wl=self.sim_cfg["n_wl"],
                # solver=self.sim_cfg["solver"],
                solver="ceviche",
                plot=True,
                require_sim=False,
            )
            for i in range(1, self.num_outports + 1)
        ]
        return norm_source_profiles, norm_refl_profiles, norm_monitor_profiles
