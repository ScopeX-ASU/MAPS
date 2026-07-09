from functools import partial
from typing import Tuple

import numpy as np
import torch
from pyutils.general import logger as lg

from core.utils import material_fn_dict

from .device_base import N_Ports

__all__ = ["PhC"]


class PhC(N_Ports):
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
        box_size: Tuple[float, float] = (4.0, 4.0),  # design box (Lx, Ly)
        port_len: Tuple[float, float] = (
            1.8,
            1.8,
        ),  # (x-directed length, y-directed length)
        port_width: Tuple[float, float] = (
            0.48,
            0.48,
        ),  # (x-directed width, y-directed width) — usually equal
        port_box_margin: float = 0.2,
        a=0.4,
        r0=0.09,
        N_rows=15,
        N_cols=19,
        device: torch.device = torch.device("cuda:0"),
    ):
        # print(box_size)
        # print(port_width)
        # print(port_box_margin)
        # assert box_size[1] - 2 * port_box_margin >= port_width[0], (
        #     "box_size[1] too small to place the West/East ports"
        # )
        # assert box_size[0] - 2 * port_box_margin >= port_width[1], (
        #     "box_size[0] too small to place the North/South ports"
        # )

        wl_cen = sim_cfg["wl_cen"]
        Lx_phc = N_cols * a + a / 2
        sqrt3_div2 = np.sqrt(3) / 2.0
        Ly_phc = N_rows * sqrt3_div2 * a + a / 2
        box_size = (Lx_phc, Ly_phc)

        # Materials
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
                center=[-(port_len[0] + box_size[0]) / 2, 0.0],
                size=[port_len[0], port_width[0]],
                eps=eps_r1_fn(wl_cen),
            ),
            out_port_1=dict(
                type="box",
                direction="x",
                center=[(port_len[0] + box_size[0]) / 2, 0.0],
                size=[port_len[0], port_width[0]],
                eps=eps_r1_fn(wl_cen),
            ),
        )

        geometry_cfgs = dict(
            region_1=dict(
                type="box",
                center=[0.0, 0.0],
                size=box_size,
                eps=eps_r1_fn(wl_cen),
            )
        )

        hole_spacing_x = a

        hole_spacing_y = a * sqrt3_div2

        x_slab_length, y_slab_length = (
            hole_spacing_x * (N_cols + 0.5),
            hole_spacing_y * N_rows,
        )
        start_x, start_y = (
            -x_slab_length / 2 + hole_spacing_x / 2,
            -y_slab_length / 2 + hole_spacing_y / 2,
        )

        rmid = r0

        for i in range(0, N_cols):
            for j in range(0, N_rows):
                # depending on distance from central column, hole is either static
                i_dist = abs(i - N_cols // 2)
                radius = rmid
                dx = dy = 0

                x0 = dx + start_x + (i + (j % 2) * 0.5) * hole_spacing_x
                y0 = dy + start_y + j * hole_spacing_y
                if j != N_rows // 2:
                    geometry_cfgs[f"hole_{i}_{j}"] = dict(
                        type="cylinder",
                        center=(x0, y0),
                        radius=radius,
                        eps=eps_bg_fn(wl_cen),
                    )

        design_region_cfgs = dict(
            fourport_region=dict(
                type="box",
                center=[0.0, 0.0],
                size=[0.1, 0.1],
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
        if verbose:
            lg.info("Start generating sources and monitors...")
        pml = self.sim_cfg["PML"][0]
        port_len = self.port_cfgs["in_port_1"]["size"][0]
        offset = 0.2 + pml

        src_slices = [
            self.build_port_monitor_slice(
                port_name="in_port_1",
                slice_name="in_slice_1",
                rel_loc=offset / port_len,
                rel_width=rel_width,
            )
        ]
        refl_slices = [
            self.build_port_monitor_slice(
                port_name="in_port_1",
                slice_name="refl_slice_1",
                rel_loc=(offset + 0.1) / port_len,
                rel_width=rel_width,
            )
        ]

        out_slices = [
            self.build_port_monitor_slice(
                port_name="out_port_1",
                slice_name="out_slice_1",
                rel_loc=1 - offset / port_len,
                rel_width=rel_width,
            ),
        ]

        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        radiation_monitor = self.build_radiation_monitor(monitor_name="rad_monitor")
        return src_slices, out_slices, refl_slices, radiation_monitor

    def norm_run(self, verbose: bool = True):
        if verbose:
            lg.info("Start normalization run...")

        norm_source_profiles = [
            self.build_norm_sources(
                source_modes=("Hz1",),
                input_port_name="in_port_1",
                input_slice_name="in_slice_1",
                wl_cen=self.sim_cfg["wl_cen"],
                wl_width=self.sim_cfg["wl_width"],
                n_wl=self.sim_cfg["n_wl"],
                solver="ceviche",
                plot=True,
                require_sim=True,
            )
        ]

        norm_refl_profiles = [
            self.build_norm_sources(
                source_modes=("Hz1",),
                input_port_name="in_port_1",
                input_slice_name="refl_slice_1",
                wl_cen=self.sim_cfg["wl_cen"],
                wl_width=self.sim_cfg["wl_width"],
                n_wl=self.sim_cfg["n_wl"],
                solver="ceviche",
                plot=True,
                require_sim=False,
            )
        ]

        norm_monitor_profiles = [
            self.build_norm_sources(
                source_modes=("Hz1",),
                input_port_name="out_port_1",
                input_slice_name="out_slice_1",
                wl_cen=self.sim_cfg["wl_cen"],
                wl_width=self.sim_cfg["wl_width"],
                n_wl=self.sim_cfg["n_wl"],
                # solver=self.sim_cfg["solver"],
                solver="ceviche",
                plot=True,
                require_sim=False,
            )
        ]

        return norm_source_profiles, norm_refl_profiles, norm_monitor_profiles
