import copy
import warnings
from functools import partial
from typing import Tuple

import numpy as np
import tidy3d as td
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
        rel_port_width: float = 3,
        output_sine_bend_length: float = 0.0,  # if >0, this is the length of the sine_bend
        output_sine_bend_expand: float = 1.0,  # extra expansion from the starting points
        is_3d: bool = False,
        pol: str = "Ez",
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
        self.rel_width = rel_port_width
        self.pol = pol
        self.is_3d = is_3d
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
        grid_step = 1 / sim_cfg["resolution"]

        if num_inports == 1:
            inport_y_coords = np.array([0.0])
        else:
            inport_y_coords = np.linspace(
                -box_size[1] / 2 + port_width[0] / 2 + port_box_margin,
                box_size[1] / 2 - port_width[0] / 2 - port_box_margin,
                num_inports,
            )

        for i in range(1, num_inports + 1):
            port_cfgs[f"in_port_{i}"] = dict(
                type="box",
                direction="x",
                center=[
                    -(port_len[0] + box_size[0]) / 2 + grid_step / 2,
                    inport_y_coords[i - 1],
                ]
                + ([0] if is_3d else []),
                size=[port_len[0] + grid_step, port_width[0]]
                + ([thickness_r1] if is_3d else []),
                eps=eps_r1_fn(wl_cen),
            )
        geometry_cfgs = {}
        if output_sine_bend_length > 0:
            outport_y_start_coords = np.linspace(
                -box_size[1] / 2 + port_width[1] / 2 + port_box_margin,
                box_size[1] / 2 - port_width[1] / 2 - port_box_margin,
                num_outports,
            )
            outport_y_end_coords = np.linspace(
                -box_size[1] / 2
                + port_width[1] / 2
                + port_box_margin
                - output_sine_bend_expand,
                box_size[1] / 2
                - port_width[1] / 2
                - port_box_margin
                + output_sine_bend_expand,
                num_outports,
            )

            for i in range(1, num_outports + 1):
                geometry_cfgs[f"out_sine_bend_{i}"] = dict(
                    type="sine_bend",
                    direction="x",
                    start=[
                        box_size[0] / 2 - grid_step / 2,
                        outport_y_start_coords[i - 1],
                    ],
                    end=[
                        box_size[0] / 2 - grid_step / 2 + output_sine_bend_length,
                        outport_y_end_coords[i - 1],
                    ],
                    axis=2,
                    slab_bounds=(
                        (-thickness_r1 / 2, thickness_r1 / 2)
                        if is_3d
                        else (-td.inf, td.inf)
                    ),
                    width=port_width[1],
                    num_samples=100,
                    eps=eps_r1_fn(wl_cen),
                )
        else:
            outport_y_end_coords = np.linspace(
                -box_size[1] / 2 + port_width[1] / 2 + port_box_margin,
                box_size[1] / 2 - port_width[1] / 2 - port_box_margin,
                num_outports,
            )
        for i in range(1, num_outports + 1):
            port_cfgs[f"out_port_{i}"] = dict(
                type="box",
                direction="x",
                center=[
                    box_size[0] / 2
                    + output_sine_bend_length
                    + port_len[1] / 2
                    - grid_step / 2,
                    outport_y_end_coords[i - 1],
                ]
                + ([0] if is_3d else []),
                size=[port_len[1] + grid_step, port_width[1]]
                + ([thickness_r1] if is_3d else []),
                eps=eps_r1_fn(wl_cen),
            )

        design_region_cfgs = dict()
        design_region_cfgs["mmi_region"] = dict(
            type="box",
            center=[0, 0] + ([0] if is_3d else []),
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
        rel_width = self.rel_width
        if verbose:
            logger.info("Start generating sources and monitors ...")
        pml = self.sim_cfg["PML"][0]
        port_len = self.port_cfgs["in_port_1"]["size"][0]
        offset = 0.2 + pml

        if self.is_3d:
            rel_height = 4.5
        else:
            rel_height = None

        src_slices = [
            self.build_port_monitor_slice(
                port_name=f"in_port_{i}",
                slice_name=f"in_slice_{i}",
                rel_loc=offset / port_len,
                rel_width=rel_width,
                rel_height=rel_height,
                direction="x+",
            )
            for i in range(1, self.num_inports + 1)
        ]
        refl_slices = [
            self.build_port_monitor_slice(
                port_name=f"in_port_{i}",
                slice_name=f"refl_slice_{i}",
                rel_loc=(offset + 0.1) / port_len,
                rel_width=rel_width,
                rel_height=rel_height,
                direction="x+",
            )
            for i in range(1, self.num_inports + 1)
        ]

        out_slices = []
        for i in range(1, self.num_outports + 1):
            center = self.port_cfgs[f"out_port_{i}"]["center"]
            if (
                self.is_3d
                and "symmetry" in self.sim_cfg
                and self.sim_cfg["symmetry"] is not None
                and self.sim_cfg["symmetry"][1] != 0
                and center[1] < 0
            ):
                ## this will not appear in the reduced simulation space with symmetry
                logger.warning(
                    f"out_port_{i} is not in the simulation space with symmetry, skip building the monitor slice"
                )
            else:
                out_slices.append(
                    self.build_port_monitor_slice(
                        port_name=f"out_port_{i}",
                        slice_name=f"out_slice_{i}",
                        rel_loc=1 - offset / port_len,
                        rel_width=rel_width,
                        rel_height=rel_height,
                        direction="x+",
                    )
                )

        # out_slices = [
        #     self.build_port_monitor_slice(
        #         port_name=f"out_port_{i}",
        #         slice_name=f"out_slice_{i}",
        #         rel_loc=1 - offset / port_len,
        #         rel_width=rel_width,
        #         rel_height=rel_height,
        #         direction="x+",
        #     )
        #     for i in range(1, self.num_outports + 1)
        # ]
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)

        if not self.is_3d:
            radiation_monitor = self.build_radiation_monitor(monitor_name="rad_slice")
        else:
            radiation_monitor = None

        return src_slices, out_slices, refl_slices, radiation_monitor

    def norm_run(self, verbose: bool = True):
        if verbose:
            logger.info("Start normalization run ...")
        # norm_run_sim_cfg = copy.deepcopy(self.sim_cfg)
        # norm_run_sim_cfg["numerical_solver"] = "solve_direct"
        norm_source_profiles = [
            self.build_norm_sources(
                source_modes=(f"{self.pol}1",),
                input_port_name=f"in_port_{i}",
                input_slice_name=f"in_slice_{i}",
                wl_cen=self.sim_cfg["wl_cen"],
                wl_width=self.sim_cfg["wl_width"],
                n_wl=self.sim_cfg["n_wl"],
                solver=self.sim_cfg["solver"],
                plot=True,
                require_sim=True,
            )
            for i in range(1, self.num_inports + 1)
        ]

        norm_refl_profiles = [
            self.build_norm_sources(
                source_modes=(f"{self.pol}1",),
                input_port_name=f"in_port_{i}",
                input_slice_name=f"refl_slice_{i}",
                wl_cen=self.sim_cfg["wl_cen"],
                wl_width=self.sim_cfg["wl_width"],
                n_wl=self.sim_cfg["n_wl"],
                solver=self.sim_cfg["solver"],
                plot=True,
                require_sim=False,
            )
            for i in range(1, self.num_inports + 1)
        ]
        # norm_monitor_profiles = [
        #     self.build_norm_sources(
        #         source_modes=(f"{self.pol}1",),
        #         input_port_name=f"out_port_{i}",
        #         input_slice_name=f"out_slice_{i}",
        #         wl_cen=self.sim_cfg["wl_cen"],
        #         wl_width=self.sim_cfg["wl_width"],
        #         n_wl=self.sim_cfg["n_wl"],
        #         solver=self.sim_cfg["solver"],
        #         plot=True,
        #         require_sim=False,
        #     )
        #     for i in range(1, self.num_outports + 1)
        # ]

        norm_monitor_profiles = []
        for i in range(1, self.num_outports + 1):
            center = self.port_cfgs[f"out_port_{i}"]["center"]
            if (
                self.is_3d
                and "symmetry" in self.sim_cfg
                and self.sim_cfg["symmetry"] is not None
                and self.sim_cfg["symmetry"][1] != 0
                and center[1] < 0
            ):
                ## this will not appear in the reduced simulation space with symmetry
                logger.warning(
                    f"out_port_{i} is not in the simulation space with symmetry, skip norm run"
                )
            else:
                norm_monitor_profiles.append(
                    self.build_norm_sources(
                        source_modes=(f"{self.pol}1",),
                        input_port_name=f"out_port_{i}",
                        input_slice_name=f"out_slice_{i}",
                        wl_cen=self.sim_cfg["wl_cen"],
                        wl_width=self.sim_cfg["wl_width"],
                        n_wl=self.sim_cfg["n_wl"],
                        solver=self.sim_cfg["solver"],
                        plot=True,
                        require_sim=False,
                    )
                )

        return norm_source_profiles, norm_refl_profiles, norm_monitor_profiles
