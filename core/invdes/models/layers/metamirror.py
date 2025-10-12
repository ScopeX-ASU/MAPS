"""
Date: 2024-10-04 15:10:35
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-16 16:02:02
FilePath: /MAPS/core/invdes/models/layers/metamirror.py
"""

from typing import Tuple

import torch
from pyutils.general import logger

from core.utils import material_fn_dict

from .device_base import N_Ports

__all__ = ["MetaMirror"]


class MetaMirror(N_Ports):
    def __init__(
        self,
        material_r: str = "Si",  # waveguide material
        material_bg: str = "SiO2",  # background material
        sim_cfg: dict = {
            "border_width": [
                0,
                1.8,
                0,
                1.8,
            ],  # left, right, lower, upper, containing PML
            "PML": [1, 1],  # left/right, lower/upper
            "cell_size": None,
            "resolution": 50,
            "wl_cen": 1.55,
            "wl_width": 0,
            "n_wl": 1,
            "plot_root": "./figs/metamirror",
        },
        aperture: float = 1.5,
        ridge_height_max: float = 0.06,
        port_len: Tuple[float] = (5, 4),
        port_width: Tuple[float] = (0.22, 0.22),
        mirror_size: Tuple[float] = (0.32, 0.32),  # d1, d2
        device: torch.device = torch.device("cuda:0"),
    ):
        wl_cen = sim_cfg["wl_cen"]
        eps_r_fn = material_fn_dict[material_r]
        eps_bg_fn = material_fn_dict[material_bg]
        port_cfgs = dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-port_len[0] / 2 + aperture / 2 + mirror_size[0], 0],
                size=[port_len[0], port_width[0]],
                eps=eps_r_fn(wl_cen),
            ),
            out_port_1=dict(
                type="box",
                direction="y",
                center=[
                    aperture / 2 + mirror_size[0] - port_width[1] / 2,
                    -port_len[1] / 2 + port_width[0] / 2,
                ],
                size=[port_width[1], port_len[1]],
                eps=eps_r_fn(wl_cen),
            ),
        )

        geometry_cfgs = dict(
            mirror_45=dict(
                type="prism",
                vertices=[
                    (aperture / 2, port_width[0] / 2),
                    (aperture / 2 + mirror_size[0], port_width[0] / 2),
                    (aperture / 2 + mirror_size[0], port_width[0] / 2 - mirror_size[1]),
                ],
                # size=[port_len[1] + port_width[0] / 2, 0.48],
                eps=eps_bg_fn(wl_cen),
            )
        )

        design_region_cfgs = dict(
            design_region_1=dict(
                type="box",
                center=[
                    0,
                    port_width[0] / 2
                    - ridge_height_max / 2
                    - 1 / sim_cfg["resolution"],
                ],
                size=[aperture, ridge_height_max],
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
        rel_width = 5
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
        radiation_monitor = self.build_radiation_monitor(monitor_name="rad_monitor")
        return src_slice, out_slice, refl_slice, radiation_monitor

    def norm_run(self, verbose: bool = True):
        if verbose:
            logger.info("Start normalization run ...")
        norm_source_profiles = self.build_norm_sources(
            source_modes=("Ez1",),
            input_port_name="in_port_1",
            input_slice_name="in_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
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
            plot=True,
            require_sim=False,
        )
        return norm_source_profiles, norm_refl_profiles, norm_monitor_profiles
