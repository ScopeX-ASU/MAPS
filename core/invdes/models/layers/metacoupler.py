from typing import Tuple

import torch
from pyutils.general import logger

from core.utils import material_fn_dict

from .device_base import N_Ports

__all__ = ["MetaCoupler"]


class MetaCoupler(N_Ports):
    def __init__(
        self,
        material_r_1: str = "Si",  # waveguide material
        material_r_2: str = "SiN",  # waveguide material
        material_bg: str = "SiO2",  # background material
        sim_cfg: dict = {
            "border_width": [
                0,
                0,
                1.5,
                1.5,
            ],  # left, right, lower, upper, containing PML
            "PML": [1, 1],  # left/right, lower/upper
            "cell_size": None,
            "resolution": 50,
            "wl_cen": 1.55,
            "wl_width": 0,
            "n_wl": 1,
        },
        aperture: float = 6,
        n_layers: int = 6,
        ridge_height_max: float = 1,
        port_len: Tuple[float] = (3, 3),
        port_width: Tuple[float] = (6, 3),
        device: torch.device = torch.device("cuda:0"),
    ):
        wl_cen = sim_cfg["wl_cen"]
        eps_r_fn_1 = material_fn_dict[material_r_1]
        eps_r_fn_2 = material_fn_dict[material_r_2]
        eps_bg_fn = material_fn_dict[material_bg]
        box_size = [n_layers * ridge_height_max, aperture]
        port_cfgs = dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-(port_len[0] + box_size[0] / 2) / 2, 0],
                size=[port_len[0] + box_size[0] / 2, port_width[0]],
                eps=eps_r_fn_1(wl_cen),
            ),
            out_port_1=dict(
                type="box",
                direction="x",
                center=[(port_len[1] + box_size[0] / 2) / 2, 0],
                size=[port_len[1] + box_size[0] / 2, port_width[1]],
                eps=eps_r_fn_2(wl_cen),
            ),
        )

        geometry_cfgs = dict()

        design_region_cfgs = dict()
        for i in range(n_layers):
            design_region_cfgs[f"design_region_{i}"] = dict(
                type="box",
                center=[
                    -(box_size[0] / 2) + ridge_height_max / 2 + i * ridge_height_max,
                    0,
                ],
                size=[ridge_height_max, aperture],
                eps=eps_r_fn_1(wl_cen),
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
        rel_width = 1.5
        if verbose:
            logger.info("Start generating sources and monitors ...")
        src_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_port_1",
            rel_loc=0.4,
            rel_width=rel_width,
        )
        refl_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="refl_port_1",
            rel_loc=0.41,
            rel_width=rel_width,
        )
        out_slice = self.build_port_monitor_slice(
            port_name="out_port_1",
            slice_name="out_port_1",
            rel_loc=0.6,
            rel_width=rel_width,
        )
        out_refl_slice = self.build_port_monitor_slice(
            port_name="out_port_1",
            slice_name="refl_port_2",
            rel_loc=0.59,
            rel_width=rel_width,
        )
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        radiation_monitor = self.build_radiation_monitor(monitor_name="rad_monitor")
        return src_slice, out_slice, refl_slice, out_refl_slice, radiation_monitor

    def norm_run(self, verbose: bool = True):
        if verbose:
            logger.info("Start normalization run ...")
        norm_source_profiles = self.build_norm_sources(
            source_modes=(1,),
            input_port_name="in_port_1",
            input_slice_name="in_port_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=False,
        )

        norm_refl_profiles_1 = self.build_norm_sources(
            source_modes=(1,),
            input_port_name="in_port_1",
            input_slice_name="refl_port_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=False,
        )
        norm_refl_profiles_2 = self.build_norm_sources(
            source_modes=(1,),
            input_port_name="out_port_1",
            input_slice_name="refl_port_2",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=False,
        )

        norm_monitor_profiles = self.build_norm_sources(
            source_modes=(1,),
            input_port_name="out_port_1",
            input_slice_name="out_port_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=False,
        )
        return (
            norm_source_profiles,
            norm_refl_profiles_1,
            norm_refl_profiles_2,
            norm_monitor_profiles,
        )
