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
        material_sub: str = "SiO2",  # substrate material
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
        port_width: Tuple[float] = (3, 2),
        nearfield_dx: float = 0.5,  # distance from metalens surface to nearfield monitor, e.g., 500 nm
        nearfield_size: float = 4,  # monitor size of nearfield monitor, e.g., 1um
        farfield_dxs: Tuple[float] = (
            (10, 30),
        ),  # distance from metalens surface to multiple farfield monitors, e.g., (2 um) ((min, max), ...)
        farfield_sizes: Tuple[float] = (
            2,
        ),  # monitor size of multiple farfield monitors, e.g., (1um) (dim-x, dim-y)
        device: torch.device = torch.device("cuda:0"),
    ):
        wl_cen = sim_cfg["wl_cen"]
        eps_r_fn = material_fn_dict[material_r]
        eps_bg_fn = material_fn_dict[material_bg]
        eps_sub_fn = material_fn_dict[material_sub]
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
        self.resolution = sim_cfg["resolution"]
        self.max_port_width = max(port_width)

        port_cfgs = dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-size_x / 2 + port_len[0] / 2 + 0.5 / self.resolution, 0],
                size=[port_len[0] + 1 / self.resolution, port_width[0]],
                eps=eps_sub_fn(wl_cen),
            ),
            out_port_1=dict(
                type="box",
                direction="x",
                center=[size_x / 2 - port_len[1] / 2, 0],
                size=[port_len[1], port_width[1]],
                eps=eps_bg_fn(wl_cen),
            ),
        )
        if substrate_depth > 0:
            geometry_cfgs = dict(
                substrate=dict(
                    type="box",
                    center=[-size_x / 2 + port_len[0] + substrate_depth / 2, 0],
                    size=[substrate_depth, aperture],  # some margin
                    eps=eps_sub_fn(wl_cen),
                )
            )
        else:
            geometry_cfgs = dict()

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
        rel_width = 1.2
        if verbose:
            logger.info("Start generating sources and monitors ...")
        src_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_port_1",
            rel_loc=0.85,
            rel_width=float("inf"),
            direction="x+",
        )
        refl_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="refl_port_1",
            rel_loc=0.9,
            rel_width=float("inf"),
            direction="x+",
        )
        # near field monitor
        nearfield_slice_1 = self.build_near2far_slice(
            slice_name="nearfield_1",
            center=(
                self.nearfield_dx
                + self.port_cfgs["out_port_1"]["center"][0]
                - self.port_cfgs["out_port_1"]["size"][0] / 2,
                0,
            ),
            size=(0, self.nearfield_size),
            direction="x+",
        )

        nf1_center = self.port_monitor_slices_info["nearfield_1"]["center"]
        nf1_size = self.port_monitor_slices_info["nearfield_1"]["size"]
        nf2_width = (
            self.box_size[0]
            + self.nearfield_dx
            + self.substrate_depth
            - 0.5 / self.resolution
        )

        # nearfield_slice_2 = self.build_near2far_slice(
        #     slice_name="nearfield_2",
        #     center=(nf1_center[0] - nf2_width / 2, nf1_size[1] / 2),
        #     size=(nf2_width, 0),
        #     direction="y+",
        # )

        # nearfield_slice_3 = self.build_near2far_slice(
        #     slice_name="nearfield_3",
        #     center=(nf1_center[0] - nf2_width / 2, -nf1_size[1] / 2),
        #     size=(nf2_width, 0),
        #     direction="y-",
        # )

        # nearfield_slice_4 = self.build_near2far_slice(
        #     slice_name="nearfield_4",
        #     center=(nf1_center[0] - nf2_width, 0),
        #     size=(0, 2*self.aperture),
        #     direction="x-",
        # )

        # farfield_slices = [
        #     self.build_port_monitor_slice(
        #         port_name="out_port_1",
        #         slice_name=f"farfield_{i}",
        #         rel_loc=farfield_dx / self.port_cfgs["out_port_1"]["size"][0],
        #         rel_width=farfield_size / self.port_cfgs["out_port_1"]["size"][1],
        #     )
        #     for i, (farfield_dx, farfield_size) in enumerate(
        #         zip(self.farfield_dxs, self.farfield_sizes), 1
        #     )
        # ]

        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=1)
        radiation_monitor = self.build_radiation_monitor(monitor_name="rad_monitor")
        # farfield_radiation_monitor = self.build_farfield_radiation_monitor(monitor_name="farfield_rad_monitor")

        surface_x = (
            self.port_cfgs["out_port_1"]["center"][0]
            - self.port_cfgs["out_port_1"]["size"][0] / 2
        )  # this is the x coordinate of the right edge of the metalens
        out_port_end = (
            self.port_cfgs["out_port_1"]["center"][0]
            + self.port_cfgs["out_port_1"]["size"][0] / 2
        )
        farfield_regions = [
            self.build_farfield_region(
                region_name=f"farfield_{i}",
                direction="x+",
                center=(
                    surface_x + (farfield_dx[0] + farfield_dx[1]) / 2,
                    0,
                ),
                size=(farfield_dx[1] - farfield_dx[0], farfield_size),
            )
            for i, (farfield_dx, farfield_size) in enumerate(
                zip(self.farfield_dxs, self.farfield_sizes), 1
            )
        ]
        maximum_x_coord = [
            (surface_x + farfield_dx[1]) for farfield_dx in self.farfield_dxs
        ]
        maximum_x_coord = (
            max(maximum_x_coord) + 1
        )  # add additional 1 to leave some margin
        total_farfield_region = self.build_farfield_region(
            region_name="total_farfield_region",
            direction="x+",
            center=(
                (out_port_end + maximum_x_coord) / 2,
                0,
            ),
            size=(
                maximum_x_coord - out_port_end,
                max(
                    self.box_size[1]
                    + self.sim_cfg["border_width"][2]
                    + self.sim_cfg["border_width"][3],
                    self.max_port_width,
                ),
            ),
        )

        farfield_rad_trans_yp_region = self.build_farfield_region(
            region_name="rad_trans_farfield_yp",
            direction="y",
            center=(
                (out_port_end + maximum_x_coord) / 2,
                max(
                    self.box_size[1]
                    + self.sim_cfg["border_width"][2]
                    + self.sim_cfg["border_width"][3],
                    self.max_port_width,
                )
                / 2
                - self.sim_cfg["PML"][1]
                - 0.1,
            ),
            size=(
                maximum_x_coord - out_port_end - 0.5,
                2 * self.grid_step,
            ),
        )

        farfield_rad_trans_ym_region = self.build_farfield_region(
            region_name="rad_trans_farfield_ym",
            direction="y",
            center=(
                (out_port_end + maximum_x_coord) / 2,
                -max(
                    self.box_size[1]
                    + self.sim_cfg["border_width"][2]
                    + self.sim_cfg["border_width"][3],
                    self.max_port_width,
                )
                / 2
                + self.sim_cfg["PML"][1]
                + 0.1,
            ),
            size=(
                maximum_x_coord - out_port_end - 0.5,
                2 * self.grid_step,
            ),
        )
        return (
            src_slice,
            nearfield_slice_1,
            # nearfield_slice_2,
            # nearfield_slice_3,
            refl_slice,
            # farfield_slices,
            radiation_monitor,
            # farfield_radiation_monitor,
            farfield_regions,
            total_farfield_region,
            farfield_rad_trans_yp_region,
            farfield_rad_trans_ym_region,
        )

    def norm_run(self, verbose: bool = True):
        if verbose:
            logger.info("Start normalization run ...")

        norm_source_profiles = self.build_norm_sources(
            source_modes=("Hz1",),
            input_port_name="in_port_1",
            input_slice_name="in_port_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            source_type="plane_wave",
            plot=True,
        )

        norm_refl_profiles_1 = self.build_norm_sources(
            source_modes=("Hz1",),
            input_port_name="in_port_1",
            input_slice_name="refl_port_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            source_type="plane_wave",
            plot=True,
        )

        return norm_source_profiles, norm_refl_profiles_1
