import warnings
from functools import partial
from typing import Tuple

import torch
from pyutils.general import logger

from core.utils import material_fn_dict

from .device_base import N_Ports

__all__ = ["GratingCoupler"]


class GratingCoupler(N_Ports):
    def __init__(
        self,
        material_r: str = "Si",
        material_layers: Tuple[str, ...] = ("Si",),
        thickness_layers: Tuple[float, ...] = (0.22,),
        etch_depth_layers: Tuple[float, ...] = None,
        gap_layers: Tuple[float, ...] = (0.0,),
        material_bg: str = "SiO2",
        sim_cfg: dict = {
            "border_width": [0, 1.8, 0.6, 0.6, 0.6, 0.6],
            "PML": [0.2, 0.2, 0.2],
            "cell_size": None,
            "resolution": 25,
            "wl_cen": 1.55,
            "wl_width": 0,
            "n_wl": 1,
        },
        design_region_size: Tuple[float, float] = (3.0, 3.0),
        port_len: float = 1.5,
        port_width: float = 0.48,
        port_height: float = 0.22,
        top_port_len: float = 1.6,
        substrate_thickness: float = 0.5,
        substrate_gap: float = 2.0,
        waist_radius: float = 5.6568,  # um
        waist_distance: float = 0.0,  # um
        device: torch.device = torch.device("cuda:0"),
        is_3d: bool = False,
        in_mode: str = "Hz2",
        out_mode: str = "Hz1",
        verbose: bool = True,
    ):
        self.is_3d = is_3d
        self.in_mode = in_mode
        self.out_mode = out_mode
        self.top_port_len = top_port_len
        top_port_height = top_port_len  # sim_cfg["border_width"][-1]
        self.waist_radius = waist_radius
        self.waist_distance = waist_distance

        self.design_region_size = design_region_size

        if len(material_layers) != len(thickness_layers):
            raise ValueError(
                "material_layers and thickness_layers must have the same length."
            )

        if len(gap_layers) == 1 and len(thickness_layers) > 1:
            gap_layers = tuple(gap_layers[0] for _ in thickness_layers)

        if len(gap_layers) != len(thickness_layers):
            raise ValueError(
                "gap_layers must either have length 1 or match thickness_layers."
            )

        wl_cen = sim_cfg["wl_cen"]
        grid_step = 1 / sim_cfg["resolution"]

        def make_eps_fn(material, thickness=None):
            if isinstance(material, str):
                eps_fn = material_fn_dict[material]
                if "_eff" in material and thickness is not None:
                    eps_fn = partial(eps_fn, thickness=thickness)
                return eps_fn
            return lambda wl: material

        # eps_bg_fn = material_fn_dict[material_bg]
        eps_bg_fn = make_eps_fn(material_bg, port_height)
        eps_r_fn = make_eps_fn(material_r, port_height)

        eps_layer_fns = [
            make_eps_fn(mat, th) for mat, th in zip(material_layers, thickness_layers)
        ]

        if etch_depth_layers is None:
            etch_depth_layers = thickness_layers

        if len(etch_depth_layers) == 1 and len(thickness_layers) > 1:
            etch_depth_layers = tuple(etch_depth_layers[0] for _ in thickness_layers)

        if len(etch_depth_layers) != len(thickness_layers):
            raise ValueError(
                "etch_depth_layers must either have length 1 or match thickness_layers."
            )

        for i, (etch_depth, th) in enumerate(zip(etch_depth_layers, thickness_layers)):
            if etch_depth < 0 or etch_depth > th:
                raise ValueError(
                    f"etch_depth_layers[{i}]={etch_depth} must be between 0 and "
                    f"thickness_layers[{i}]={th}."
                )

        # Stack is centered around vertical coordinate 0.
        # 2D vertical axis: y
        # 3D vertical axis: z
        total_stack_height = sum(thickness_layers) + sum(gap_layers)
        stack_bottom = -thickness_layers[0] / 2
        # stack_bottom = -total_stack_height / 2

        layer_bounds = []
        current_bottom = stack_bottom

        for th, gap in zip(thickness_layers, gap_layers):
            layer_bottom = current_bottom + gap
            layer_top = layer_bottom + th
            layer_center = 0.5 * (layer_bottom + layer_top)
            layer_bounds.append((layer_bottom, layer_top, layer_center))
            current_bottom = layer_top

        top_layer_top = layer_bounds[-1][1]

        port_cfgs = {}

        if not is_3d:
            # x-y simulation. Layers stack in y.
            stack_height = total_stack_height

            port_cfgs["out_port_1"] = dict(
                type="box",
                direction="x",
                center=[
                    (port_len + design_region_size[0]) / 2 - grid_step / 2,
                    0.0,
                ],
                size=[
                    port_len + grid_step,
                    stack_height,
                ],
                eps=eps_r_fn(wl_cen),
            )

            # Virtual vertical port abutting the top surface.
            port_cfgs["in_port_1"] = dict(
                type="box",
                direction="y",
                center=[
                    0.0,
                    top_layer_top + top_port_height / 2,
                ],
                size=[
                    design_region_size[0],
                    top_port_height,
                ],
                eps=eps_bg_fn(wl_cen),
            )

        else:
            # x-y-z simulation. Layers stack in z.
            port_cfgs["out_port_1"] = dict(
                type="box",
                direction="x",
                center=[
                    (port_len + design_region_size[0]) / 2 - grid_step / 2,
                    0.0,
                    stack_bottom + port_height / 2,
                ],
                size=[
                    port_len + grid_step,
                    port_width,
                    port_height,
                ],
                eps=eps_r_fn(wl_cen),
            )

            # Virtual vertical port abutting the top z surface.
            port_cfgs["in_port_1"] = dict(
                type="box",
                direction="z",
                center=[
                    0.0,
                    0.0,
                    top_layer_top + top_port_height / 2,
                ],
                size=[
                    design_region_size[0],
                    design_region_size[1],
                    top_port_height,
                ],
                eps=eps_bg_fn(wl_cen),
            )

        geometry_cfgs = dict(
            substrate=dict(
                type="box",
                center=[
                    0.0,
                    0.0,
                    stack_bottom - substrate_gap - substrate_thickness / 2,
                ],
                size=[None, None, substrate_thickness],
                eps=eps_r_fn(wl_cen),
            ),
        )
        design_region_cfgs = {}

        for i, ((layer_bottom, layer_top, _), th, etch_depth, eps_fn) in enumerate(
            zip(layer_bounds, thickness_layers, etch_depth_layers, eps_layer_fns)
        ):
            fixed_thickness = th - etch_depth

            # Bottom unetched fixed slab.
            if fixed_thickness > 0:
                fixed_bottom = layer_bottom
                ## add a small extra margin.
                fixed_thickness += grid_step
                fixed_top = layer_bottom + fixed_thickness
                fixed_center = 0.5 * (fixed_bottom + fixed_top)

                if not is_3d:
                    fixed_center_xyz = [0.0, fixed_center]
                    fixed_size = [design_region_size[0], fixed_thickness]
                else:
                    fixed_center_xyz = [0.0, 0.0, fixed_center]
                    fixed_size = [
                        design_region_size[0],
                        design_region_size[1],
                        fixed_thickness,
                    ]

                geometry_cfgs[f"fixed_slab_layer_{i + 1}"] = dict(
                    type="box",
                    center=fixed_center_xyz,
                    size=fixed_size,
                    eps=eps_fn(wl_cen),
                )

            # Top etched/design region.
            if etch_depth > 0:
                design_bottom = layer_top - etch_depth
                design_top = layer_top
                design_center = 0.5 * (design_bottom + design_top)

                if not is_3d:
                    design_center_xyz = [0.0, design_center]
                    design_size = [design_region_size[0], etch_depth]
                else:
                    design_center_xyz = [0.0, 0.0, design_center]
                    design_size = [
                        design_region_size[0],
                        design_region_size[1],
                        etch_depth,
                    ]

                design_region_cfgs[f"grating_layer_{i + 1}"] = dict(
                    type="box",
                    center=design_center_xyz,
                    size=design_size,
                    eps=eps_fn(wl_cen),
                    eps_bg=eps_bg_fn(wl_cen),
                )

        super().__init__(
            eps_bg=eps_bg_fn(wl_cen),
            sim_cfg=sim_cfg,
            port_cfgs=port_cfgs,
            geometry_cfgs=geometry_cfgs,
            design_region_cfgs=design_region_cfgs,
            device=device,
            verbose=verbose,
        )

    def init_monitors(self, verbose: bool = True):
        rel_width = 6
        pml = self.sim_cfg["PML"][0]
        offset = 0.2 + pml
        in_port_len = self.top_port_len
        out_port_len = self.port_cfgs["out_port_1"]["size"][0]
        in_port_width = self.port_cfgs["in_port_1"]["size"][0]
        cell_size = min(self.cell_size[0:2]) - self.grid_step
        wl_cen = self.sim_cfg["wl_cen"]

        rel_height = None
        if self.is_3d:
            rel_height = 4.5
        if verbose:
            logger.info("Start generating sources and monitors ...")
        src_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_slice_1",
            rel_loc=wl_cen / 2 / in_port_len,
            rel_width=1,
            rel_height=1,
        )
        refl_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="refl_slice_1",
            rel_loc=(wl_cen / 2 + 0.1) / in_port_len,
            rel_width=cell_size / in_port_width,
            rel_height=cell_size / in_port_width,
        )
        in_monitor_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_monitor_slice_1",
            rel_loc=(wl_cen / 2 - 0.1) / in_port_len,
            rel_width=cell_size / in_port_width,
            rel_height=cell_size / in_port_width,
        )
        out_slice = self.build_port_monitor_slice(
            port_name="out_port_1",
            slice_name="out_slice_1",
            # rel_loc=offset / port_len,
            rel_loc=1 - offset / out_port_len,
            rel_width=rel_width,
            rel_height=rel_height,
        )
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        if not self.is_3d:
            radiation_monitor = self.build_radiation_monitor(monitor_name="rad_slice")
            return src_slice, out_slice, refl_slice, radiation_monitor
        return src_slice, out_slice, refl_slice, in_monitor_slice

    def norm_run(self, verbose: bool = True):
        if verbose:
            logger.info("Start normalization run ...")
        # norm_run_sim_cfg = copy.deepcopy(self.sim_cfg)
        # norm_run_sim_cfg["numerical_solver"] = "solve_direct"

        """https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd16BilayerCoupler/"""
        ## for original fdtdx.GaussianPlaneSource
        ## in tidy3d, waist_radius = √2 × σ
        ## in fdtdx, radius = 3 × σ
        ## so to have the same gaussian source, we need to set gaussian_radius = 3/√2 * waist_radius
        ## in tidy3d, waist_radius = design_region_size * sqrt(2) / 2
        ## so in fdtdx, gaussian_radius = design_region_size * 3 / sqrt(2) * sqrt(2) / 2 = design_region_size * 1.5
        ## gaussian_radius = gaussian_waist / 2 * 3 / sqrt(2)

        ## [06/25/2026] We support GaussianPlaneSourceTidy3d with same waist_radius and waist_distance as Tidy3d
        norm_source_profiles = self.build_norm_sources(
            source_modes=(self.in_mode,),
            input_port_name="in_port_1",
            input_slice_name="in_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            source_type="gaussian_beam",
            # gaussian_radii=(self.design_region_size[0] * 1.5 * 1e-6,),
            waist_radii=(self.waist_radius * 1e-6,),
            waist_distances=(self.waist_distance * 1e-6,),
            plot=True,
            require_sim=True,
        )
        norm_source_profiles = self.build_norm_sources(
            source_modes=(self.out_mode,),
            input_port_name="out_port_1",
            input_slice_name="out_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )

        return norm_source_profiles
