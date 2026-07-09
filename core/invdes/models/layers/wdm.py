from functools import partial
from typing import Tuple

import numpy as np
import torch
from pyutils.general import logger

from core.utils import material_fn_dict

from .device_base import N_Ports

__all__ = ["WDM"]


class WDM(N_Ports):
    def __init__(
        self,
        material_r1: str = "Si_eff",  # waveguide material
        material_r2: str = "SiO2",  # waveguide material
        thickness_r1: float = 0.22,
        thickness_r2: float = 0,
        material_bg: str = "SiO2",  # background material
        sim_cfg: dict = {
            "border_width": [
                0,
                0,
                1.5,
                1.5,
            ],  # left, right, lower, upper, containing PML
            "PML": [0.5, 0.5],  # left/right, lower/upper
            "cell_size": None,
            "resolution": 50,
            "wl_cen": 1.55,
            "wl_width": 0,
            "n_wl": 1,
        },
        box_size: Tuple[float] = (2.6, 2.6),
        port_len: Tuple[float] = (5, 5),
        port_width: Tuple[float] = (0.48, 0.8),
        port_height: Tuple[float] = (0.22, 0.22),
        num_outports: int = 2,
        port_box_margin: float = 1,
        is_3d: bool = False,
        mode: str = "Ez1",
        device: torch.device = torch.device("cuda:0"),
    ):
        self.is_3d = is_3d
        self.mode = mode
        wl_cen = sim_cfg["wl_cen"]
        self.material_r1 = material_r1
        self.num_outports = num_outports
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
        grid_step = 1 / sim_cfg["resolution"]
        port_cfgs = dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-(port_len[0] + box_size[0]) / 2 + grid_step / 2, 0]
                + ([0] if is_3d else []),
                size=[port_len[0] + grid_step, port_width[0]]
                + ([port_height[0]] if is_3d else []),
                eps=eps_r1_fn(wl_cen),
            ),
        )
        ###
        if num_outports == 2:  # keep the same as before
            outport_y_coords = [box_size[1] / 6, -box_size[1] / 6]
        else:
            outport_y_coords = np.linspace(
                -box_size[1] / 2 + port_width[1] / 2 + port_box_margin,
                box_size[1] / 2 - port_width[1] / 2 - port_box_margin,
                num_outports,
            )
        for i in range(1, num_outports + 1):
            port_cfgs[f"out_port_{i}"] = dict(
                type="box",
                direction="x",
                center=[
                    (port_len[1] + box_size[0]) / 2 - grid_step / 2,
                    outport_y_coords[i - 1],
                ]
                + ([0] if is_3d else []),
                size=[port_len[1] + grid_step, port_width[1]]
                + ([port_height[1]] if is_3d else []),
                eps=eps_r1_fn(wl_cen),
            )

        geometry_cfgs = dict()
        design_region_cfgs = dict(
            design_region_1=dict(
                type="box",
                center=[0, 0] + ([0] if is_3d else []),
                size=box_size,
                eps=eps_r1_fn(wl_cen),
                eps_bg=eps_r2_fn(wl_cen),
            )
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
        if self.is_3d:
            rel_width = 3
            rel_height = 4.5
        else:
            rel_width = 4
            rel_height = None
        pml = self.sim_cfg["PML"][0]
        offset = 0.2 + pml
        port_len = self.port_cfgs["in_port_1"]["size"][0]
        if verbose:
            logger.info("Start generating sources and monitors ...")
        src_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_slice_1",
            rel_loc=offset / port_len,
            rel_width=rel_width,
            rel_height=rel_height,
        )
        refl_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="refl_slice_1",
            rel_loc=(offset + 0.05) / port_len,
            rel_width=rel_width,
            rel_height=rel_height,
        )

        out_slices = [
            self.build_port_monitor_slice(
                port_name=f"out_port_{i}",
                slice_name=f"out_slice_{i}",
                rel_loc=1 - offset / port_len,
                rel_width=rel_width,
                rel_height=rel_height,
            )
            for i in range(1, self.num_outports + 1)
        ]
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        if not self.is_3d:
            radiation_monitor = self.build_radiation_monitor(monitor_name="rad_slice")
            return src_slice, refl_slice, *out_slices, radiation_monitor
        return src_slice, refl_slice, *out_slices

    def norm_run(self, verbose: bool = True):
        if verbose:
            logger.info("Start normalization run ...")
        norm_source_profiles = self.build_norm_sources(
            source_modes=(self.mode,),
            input_port_name="in_port_1",
            input_slice_name="in_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=True,
        )

        norm_refl_profiles_1 = self.build_norm_sources(
            source_modes=(self.mode,),
            input_port_name="in_port_1",
            input_slice_name="refl_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )

        norm_monitor_profiles = [
            self.build_norm_sources(
                source_modes=(self.mode,),
                input_port_name=f"out_port_{i}",
                input_slice_name=f"out_slice_{i}",
                wl_cen=self.sim_cfg["wl_cen"],
                wl_width=self.sim_cfg["wl_width"],
                n_wl=self.sim_cfg["n_wl"],
                solver=self.sim_cfg["solver"],
                plot=True,
                require_sim=False,
            )
            for i in range(1, self.num_outports + 1)
        ]

        return (
            norm_source_profiles,
            norm_refl_profiles_1,
            *norm_monitor_profiles,
        )
