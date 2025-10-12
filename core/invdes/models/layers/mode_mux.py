from functools import partial
from typing import Tuple

import torch
from pyutils.general import logger

from core.utils import material_fn_dict

from .device_base import N_Ports

__all__ = ["ModeCvtMux"]


class ModeCvtMux(N_Ports):
    ### this is mode converter and mode mux fused together.
    ### Ez1 from in_port_1 goes to Ez1 of out_port_1
    ### Ez1 from in_port_2 goes to Ez2 of out_port_1
    def __init__(
        self,
        material_r1: str = "Si_eff",  # waveguide material
        material_r2: str = "SiO2",  # waveguide material
        thickness_r1: float = 0.22,  # waveguide thickness
        thickness_r2: float = 0.0,  # waveguide thickness
        material_bg: str = "SiO2",  # background material
        etch_thickness: float = 0.15,
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
        device: torch.device = torch.device("cuda:0"),
    ):
        wl_cen = sim_cfg["wl_cen"]
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
        port_cfgs = dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-(port_len[0] + box_size[0] / 2) / 2, box_size[1] / 6],
                size=[port_len[0] + box_size[0] / 2, port_width[0]],
                eps=eps_r1_fn(wl_cen),  # neff from Lumerical
            ),
            in_port_2=dict(
                type="box",
                direction="x",
                center=[-(port_len[0] + box_size[0] / 2) / 2, -box_size[1] / 6],
                size=[port_len[0] + box_size[0] / 2, port_width[0]],
                eps=eps_r1_fn(wl_cen),  # neff from Lumerical
            ),
            out_port_1=dict(
                type="box",
                direction="x",
                center=[(port_len[1] + box_size[0] / 2) / 2, 0],
                size=[port_len[1] + box_size[0] / 2, port_width[1]],
                eps=eps_r1_fn(wl_cen),  # neff from Lumerical
            ),
        )

        geometry_cfgs = dict()
        design_region_cfgs = dict(
            design_region_1=dict(
                type="box",
                center=[
                    0,
                    0,
                ],
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
        rel_width = 2
        pml = self.sim_cfg["PML"][0]
        offset = 0.2 + pml
        port_len = self.port_cfgs["in_port_1"]["size"][0]
        if verbose:
            logger.info("Start generating sources and monitors ...")
        src_slice_1 = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_slice_1",
            rel_loc=offset / port_len,
            rel_width=rel_width,
        )
        src_slice_2 = self.build_port_monitor_slice(
            port_name="in_port_2",
            slice_name="in_slice_2",
            rel_loc=offset / port_len,
            rel_width=rel_width,
        )
        refl_slice_1 = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="refl_slice_1",
            rel_loc=(offset + 0.05) / port_len,
            rel_width=rel_width,
        )
        refl_slice_2 = self.build_port_monitor_slice(
            port_name="in_port_2",
            slice_name="refl_slice_2",
            rel_loc=(offset + 0.05) / port_len,
            rel_width=rel_width,
        )
        out_slice = self.build_port_monitor_slice(
            port_name="out_port_1",
            slice_name="out_slice_1",
            rel_loc=1 - offset / port_len,
            rel_width=rel_width,
        )
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        radiation_monitor = self.build_radiation_monitor(monitor_name="rad_slice")
        return (
            src_slice_1,
            src_slice_2,
            refl_slice_1,
            refl_slice_2,
            out_slice,
            radiation_monitor,
        )

    def norm_run(self, verbose: bool = True):
        if verbose:
            logger.info("Start normalization run ...")
        norm_source_profiles_mode1 = self.build_norm_sources(
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

        norm_source_profiles_mode2 = self.build_norm_sources(
            source_modes=("Ez1",),
            input_port_name="in_port_2",
            input_slice_name="in_slice_2",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=True,
        )

        norm_refl_profiles_1 = self.build_norm_sources(
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

        norm_refl_profiles_2 = self.build_norm_sources(
            source_modes=("Ez1",),
            input_port_name="in_port_2",
            input_slice_name="refl_slice_2",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )

        norm_monitor_profiles = self.build_norm_sources(
            source_modes=("Ez1", "Ez2"),
            input_port_name="out_port_1",
            input_slice_name="out_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )

        return (
            norm_source_profiles_mode1,
            norm_source_profiles_mode2,
            norm_refl_profiles_1,
            norm_refl_profiles_2,
            norm_monitor_profiles,
        )
