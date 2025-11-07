from typing import Tuple

import torch
from pyutils.general import logger

from core.utils import material_fn_dict

from .device_base import N_Ports

__all__ = ["VerticalCoupler"]


class VerticalCoupler(N_Ports):
    def __init__(
        self,
        material_r: str = "Si",  # waveguide material
        material_bg: str = "SiO2",  # background material
        sim_cfg: dict = {
            "border_width": [0, 0, 2, 3],  # left, right, lower, upper, containing PML
            "PML": [1, 1],  # left/right, lower/upper
            "cell_size": None,
            "resolution": 50,
            "wl_cen": 1.55,
            "wl_width": 0,
            "n_wl": 1,
            "plot_root": "./figs/vertical_coupler",
        },
        box_size: Tuple[float] = (10, 0.06),
        farfield_dist: float = 2.0,
        farfield_spot_size: float = 10.8, # spot_size
        port_len: Tuple[float] = (1.8, 1.8),
        port_width: Tuple[float] = (0.26, 0.26),
        device: torch.device = torch.device("cuda:0"),
    ):
        wl_cen = sim_cfg["wl_cen"]
        self.box_size = box_size
        self.farfield_spot_size = farfield_spot_size
        monitor_size = farfield_spot_size * 1.2 # this source plane should be 1.2x the gaussian spot size to avoid truncation
        if isinstance(material_r, float):
            eps_r_fn = lambda wl: material_r
        else:
            eps_r_fn = material_fn_dict[material_r]
        eps_bg_fn = material_fn_dict[material_bg]
        wl_cen = sim_cfg["wl_cen"]
        port_cfgs = dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-(port_len[0] + self.box_size[0] / 2) / 2, 0],
                size=[port_len[0] + self.box_size[0] / 2, port_width[0]],
                eps=eps_r_fn(wl_cen),
            ),
            out_port_1=dict(
                type="box",
                direction="x",
                center=[(port_len[1] + self.box_size[0] / 2) / 2, 0],
                size=[port_len[1] + self.box_size[0] / 2, port_width[1]],
                eps=eps_r_fn(wl_cen),
            ),
            out_port_2=dict(
                type="box",
                direction="y",
                center=[-box_size[0]/2 + monitor_size / 2, farfield_dist / 2],
                size=[monitor_size, farfield_dist],
                eps=eps_bg_fn(wl_cen),
            ),
        )

        geometry_cfgs = dict()

        design_region_cfgs = dict(
            design_region_1=dict(
                type="box",
                center=[
                    0,
                    port_width[0] / 2
                    - self.box_size[1] / 2
                    - 1 / sim_cfg["resolution"],
                ],
                size=self.box_size,
                eps=eps_r_fn(wl_cen),
                eps_bg=eps_bg_fn(wl_cen),
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
        rel_width = 10
        pml = self.sim_cfg["PML"][0]
        port_len = self.port_cfgs["in_port_1"]["size"][0]
        offset = 0.2 + pml

        if verbose:
            logger.info("Start generating sources and monitors ...")
        src_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_slice_1",
            rel_loc=offset / port_len,
            rel_width=rel_width,
            direction="x+"
        )
        refl_slice = self.build_port_monitor_slice(
            port_name="out_port_2",
            slice_name="refl_slice_1",
            rel_loc=0.95,
            rel_width=1,
            direction="y+"
        )
        out_slice_1 = self.build_port_monitor_slice(
            port_name="out_port_1",
            slice_name="out_slice_1",
            rel_loc=(1 - offset / port_len),
            rel_width=rel_width,
            direction="x+"
        )
        out_slice_2 = self.build_port_monitor_slice(
            port_name="out_port_2", slice_name="out_slice_2", rel_loc=0.9, rel_width=1, direction="y-"
        )
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        radiation_monitor = self.build_radiation_monitor(monitor_name="rad_monitor")
        return src_slice, out_slice_1, out_slice_2, refl_slice, radiation_monitor

    def norm_run(self, verbose: bool = True):
        if verbose:
            logger.info("Start normalization run ...")
        norm_output_profiles = self.build_norm_sources(
            source_modes=("Hz1",),
            input_port_name="in_port_1",
            input_slice_name="in_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
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
            plot=True,
            require_sim=False,
        )
        norm_source_profiles = self.build_norm_sources(
            source_modes=("Hz1",),
            input_port_name="out_port_2",
            input_slice_name="out_slice_2",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            plot=True,
            require_sim=True,
            source_type="gaussian_beam",
            spot_size=self.farfield_spot_size,
        )

        norm_refl_profiles = self.build_norm_sources(
            source_modes=("Hz1",),
            input_port_name="out_port_2",
            input_slice_name="refl_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            plot=True,
            require_sim=True,
            source_type="gaussian_beam",
            spot_size=self.farfield_spot_size,
        )


        return norm_source_profiles, norm_refl_profiles, norm_monitor_profiles
