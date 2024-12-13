from typing import Tuple

import torch
from .device_base import N_Ports

from core.utils import material_fn_dict
from pyutils.general import logger

__all__ = ["MetaLens"]


class MetaLens(N_Ports):
    def __init__(
        self,
        material_r: str = "Si",  # waveguide material
        material_bg: str = "Air",  # background material
        sim_cfg: dict = {
            "border_width": [
                0,
                0,
                1.5,
                1.5,
            ],  # left, right, lower, upper, containing PML
            "PML": [0.8, 0.8],  # left/right, lower/upper
            "cell_size": None,
            "resolution": 50,
            "wl_cen": 0.832,
            "wl_width": 0,
            "n_wl": 1,
        },
        aperture: float = 3,
        n_layers: int = 1,
        ridge_height_max: float = 0.75,
        substrate_depth: float = 0,
        port_len: Tuple[float] = (1, 1),
        nearfield_dx: float = 0.5,  # distance from metalens surface to nearfield monitor, e.g., 500 nm
        nearfield_size: float = 4,  # monitor size of nearfield monitor, e.g., 1um
        farfield_dxs: Tuple[float] = (
            2,
        ),  # distance from metalens surface to multiple farfield monitors, e.g., (2 um)
        farfield_sizes: Tuple[float] = (
            1,
        ),  # monitor size of multiple farfield monitors, e.g., (1um)
        device: torch.device = torch.device("cuda:0"),
    ):
        wl_cen = sim_cfg["wl_cen"]
        eps_r_fn = material_fn_dict[material_r]
        eps_bg_fn = material_fn_dict[material_bg]
        box_size = [n_layers * ridge_height_max, aperture]
        size_x = box_size[0] + port_len[0] + port_len[1] + substrate_depth
        size_y = aperture
        self.nearfield_dx = nearfield_dx
        self.nearfield_size = nearfield_size
        self.farfield_dxs = farfield_dxs
        self.farfield_sizes = farfield_sizes
        self.box_size = box_size
        self.aperture = aperture
        self.substrate_depth = substrate_depth
        self.eps_bg = eps_bg_fn(wl_cen)

        port_cfgs = dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-size_x / 2 + port_len[0] / 2, 0],
                size=[port_len[0], aperture + 0.3],
                eps=eps_r_fn(wl_cen),
            ),
            out_port_1=dict(
                type="box",
                direction="x",
                center=[size_x / 2 - port_len[1] / 2, 0],
                # size=[port_len[1], aperture],
                size=[port_len[1], 1],
                eps=eps_bg_fn(wl_cen),
            ),
        )

        geometry_cfgs = dict(
            substrate=dict(
                type="box",
                center=[-size_x / 2 + port_len[0] + substrate_depth / 2, 0],
                size=[substrate_depth, aperture + 0.3],  # some margin
                eps=eps_r_fn(wl_cen),
            )
        )

        design_region_cfgs = dict()
        for i in range(n_layers):
            design_region_cfgs[f"design_region_{i}"] = dict(
                type="box",
                center=[
                    -size_x / 2
                    + port_len[0]
                    + substrate_depth
                    + ridge_height_max / 2
                    + i * ridge_height_max,
                    0,
                ],
                size=[ridge_height_max, aperture],
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
        rel_width = 1.5
        if verbose:
            logger.info("Start generating sources and monitors ...")
        src_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_port_1",
            rel_loc=0.6,
            rel_width=rel_width,
        )
        refl_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="refl_port_1",
            rel_loc=0.61,
            rel_width=rel_width,
        )
        # near field monitor
        nearfield_slice_1 = self.build_near2far_slice(
            slice_name="nearfield_1",
            center=(self.nearfield_dx + self.port_cfgs["out_port_1"]["center"][0] - self.port_cfgs["out_port_1"]["size"][0]/2, 0),
            size=(0, self.nearfield_size),
            direction="x+",
        )

        nf1_center = self.port_monitor_slices_info["nearfield_1"]["center"]
        nf1_size = self.port_monitor_slices_info["nearfield_1"]["size"]
        nf2_width = self.box_size[0] + self.nearfield_dx + self.substrate_depth + self.port_cfgs["in_port_1"]["size"][0]/3

        nearfield_slice_2 = self.build_near2far_slice(
            slice_name="nearfield_2",
            center=(nf1_center[0] - nf2_width/2, nf1_size[1]/2),
            size=(nf2_width, 0),
            direction="y+",
        )

        nearfield_slice_3 = self.build_near2far_slice(
            slice_name="nearfield_3",
            center=(nf1_center[0] - nf2_width/2, -nf1_size[1]/2),
            size=(nf2_width, 0),
            direction="y-",
        )

        # nearfield_slice_4 = self.build_near2far_slice(
        #     slice_name="nearfield_4",
        #     center=(nf1_center[0] - nf2_width, 0),
        #     size=(0, 2*self.aperture),
        #     direction="x-",
        # )

        farfield_slices = [
            self.build_port_monitor_slice(
                port_name="out_port_1",
                slice_name=f"farfield_{i}",
                rel_loc=farfield_dx / self.port_cfgs["out_port_1"]["size"][0],
                rel_width=farfield_size / self.port_cfgs["out_port_1"]["size"][1],
            )
            for i, (farfield_dx, farfield_size) in enumerate(
                zip(self.farfield_dxs, self.farfield_sizes), 1
            )
        ]

        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=1)
        radiation_monitor = self.build_radiation_monitor(monitor_name="rad_monitor")
        # farfield_radiation_monitor = self.build_farfield_radiation_monitor(monitor_name="farfield_rad_monitor")

        farfield_region = self.build_farfield_region(
            region_name="farfield_region",
            direction="x",
            extension_range=(nf1_center[0]+0.2, 6),
        )

        return (
            src_slice,
            nearfield_slice_1,
            nearfield_slice_2,
            nearfield_slice_3,
            refl_slice,
            farfield_slices,
            radiation_monitor,
            # farfield_radiation_monitor,
            farfield_region,
        )

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
            plot=True,
        )

        norm_refl_profiles_1 = self.build_norm_sources(
            source_modes=(1,),
            input_port_name="in_port_1",
            input_slice_name="refl_port_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
        )

        return norm_source_profiles, norm_refl_profiles_1
