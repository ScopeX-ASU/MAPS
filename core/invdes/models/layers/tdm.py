from functools import partial
from typing import Sequence, Tuple

import torch
from pyutils.general import logger

from core.utils import get_material_heat_optic_spec, material_fn_dict

from .device_base import N_Ports

__all__ = ["TDM"]


class TDM(N_Ports):
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
        heater_material: str = "TiN",
        heater_gap: float = 0.3,
        heater_length: float | None = None,
        heater_width: float | None = None,
        heater_height: float = 0.14,
        heater_center: Sequence[float] | None = None,
        heater_on_current: float = 7.5e-3,
        input_port_offset: float = 0.0,
        is_3d: bool = False,
        mode: str = "Ez1",
        port_box_margin: float = 0.5,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.is_3d = is_3d
        self.mode = mode
        wl_cen = sim_cfg["wl_cen"]
        sim_cfg = dict(sim_cfg)
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
        heater_eps_fn = material_fn_dict.get(heater_material, eps_bg_fn)
        box_size = tuple(float(v) for v in box_size)
        if self.is_3d:
            if len(box_size) != 3:
                raise ValueError(f"3D TDM requires box_size=(x, y, z), got {box_size}")
        elif len(box_size) != 2:
            raise ValueError(f"2D TDM requires box_size=(x, y), got {box_size}")

        heater_length = float(box_size[0] if heater_length is None else heater_length)
        heater_width = float(box_size[1] if heater_width is None else heater_width)
        grid_step = 1 / sim_cfg["resolution"]

        input_port_offset = float(input_port_offset)

        if heater_center is not None and len(tuple(heater_center)) != 2:
            raise ValueError(
                f"heater_center must have length 2 for in-plane (x, y) placement, got {heater_center}."
            )

        if self.is_3d:
            default_heater_center = [
                0,
                0,
                box_size[2] / 2 + heater_gap + heater_height / 2,
            ]

            heater_size = [heater_length, heater_width, heater_height]
            heater_cross_section_area = max(heater_width * heater_height, 1e-6)
        else:
            default_heater_center = [
                0,
                box_size[1] / 2 + heater_gap + heater_height / 2,
            ]
            heater_size = [heater_length, heater_height]
            heater_cross_section_area = max(heater_height * thickness_r1, 1e-6)
        if heater_center is not None:
            heater_center = [float(v) for v in heater_center]
            if self.is_3d:
                heater_center = [
                    heater_center[0],
                    heater_center[1],
                    default_heater_center[2],
                ]
            else:
                heater_center = [heater_center[0], heater_center[1]]
        else:
            heater_center = default_heater_center
        self.z_heater = heater_center[-1]

        bg_spec = get_material_heat_optic_spec(material_bg, wl_cen)
        region_spec = get_material_heat_optic_spec(material_r1, wl_cen)
        region_bg_spec = get_material_heat_optic_spec(material_r2, wl_cen)

        if self.is_3d:
            heat_cfg = {
                "backend": "jax",
                "mesh_type": "rectangular",
                "resolution": (0.05, 0.05, 0.05),
                "dirichlet_bc": {"zmin": 300.0},
                "neumann_bc": {
                    "xmin": 0.0,
                    "xmax": 0.0,
                    "ymin": 0.0,
                    "ymax": 0.0,
                    "zmax": 0.0,
                },
                "build_source_maps": True,
                "combine_sources": True,
                "use_tidy3d": False,
            }
        else:
            heat_cfg = {
                "backend": "jax",
                "mesh_type": "rectangular",
                "resolution": (0.05, 0.05),
                "dirichlet_bc": {"ymin": 300.0},
                "neumann_bc": {"xmin": 0.0, "xmax": 0.0, "ymax": 0.0},
                "build_source_maps": True,
                "combine_sources": True,
                "use_tidy3d": False,
            }
        heat_cfg.update(sim_cfg.get("heat", {}))
        sim_cfg["heat"] = heat_cfg

        port_cfgs = dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=(
                    [-(port_len[0] + box_size[0]) / 2 + grid_step / 2]
                    + [input_port_offset]
                    + ([0.0] if self.is_3d else [])
                ),
                size=(
                    [port_len[0] + grid_step, port_width[0]]
                    + ([port_height[0]] if self.is_3d else [])
                ),
                eps=eps_r1_fn(wl_cen),
                material=material_r1,
            ),
            out_port_1=dict(
                type="box",
                direction="x",
                center=(
                    [
                        (port_len[1] + box_size[0]) / 2 - grid_step / 2,
                        box_size[1] / 2 - port_box_margin - port_width[1] / 2,
                    ]
                    + ([0] if self.is_3d else [])
                ),
                size=(
                    [port_len[1] + grid_step, port_width[1]]
                    + ([port_height[1]] if self.is_3d else [])
                ),
                eps=eps_r1_fn(wl_cen),
                material=material_r1,
            ),
            out_port_2=dict(
                type="box",
                direction="x",
                center=(
                    [
                        (port_len[1] + box_size[0]) / 2 - grid_step / 2,
                        -box_size[1] / 2 + port_box_margin + port_width[1] / 2,
                    ]
                    + ([0] if self.is_3d else [])
                ),
                size=(
                    [port_len[1] + grid_step, port_width[1]]
                    + ([port_height[1]] if self.is_3d else [])
                ),
                eps=eps_r1_fn(wl_cen),
                material=material_r1,
            ),
        )

        geometry_cfgs = dict(
            heater_1=dict(
                type="box",
                center=heater_center,
                size=heater_size,
                eps=heater_eps_fn(wl_cen),
                material=heater_material,
            )
        )
        design_region_cfgs = dict(
            design_region_1=dict(
                type="box",
                center=[0, 0] + ([0] if self.is_3d else []),
                size=box_size,
                eps=eps_r1_fn(wl_cen),
                eps_bg=eps_r2_fn(wl_cen),
                material=material_r1,
                material_bg=material_r2,
                thermal_conductivity=region_spec.thermal_conductivity,
                thermal_conductivity_bg=region_bg_spec.thermal_conductivity,
                heat_capacity=region_spec.heat_capacity,
                heat_capacity_bg=region_bg_spec.heat_capacity,
                thermo_optic_coeff=region_spec.thermo_optic_coeff,
                thermo_optic_coeff_bg=region_bg_spec.thermo_optic_coeff,
            )
        )

        heat_source_cfgs = dict(
            heater_1=dict(
                type="box",
                center=heater_center,
                size=heater_size,
                current=heater_on_current,
                material=heater_material,
                current_direction="x",
                cross_section_area=heater_cross_section_area,
                mesh_override_dl=(0.1, 0.1, 0.02),
            )
        )
        super().__init__(
            eps_bg=eps_bg_fn(wl_cen),
            k_bg=bg_spec.thermal_conductivity,
            heat_capacity_bg=bg_spec.heat_capacity,
            thermo_optic_coeff_bg=bg_spec.thermo_optic_coeff,
            sim_cfg=sim_cfg,
            port_cfgs=port_cfgs,
            geometry_cfgs=geometry_cfgs,
            design_region_cfgs=design_region_cfgs,
            heat_source_cfgs=heat_source_cfgs,
            device=device,
        )

    def init_monitors(self, verbose: bool = True):
        if self.is_3d:
            rel_width = 3
            rel_height = 4.5
        else:
            rel_width = 2
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
        temp1_out_slice = self.build_port_monitor_slice(
            port_name="out_port_1",
            slice_name="out_slice_1",
            rel_loc=1 - offset / port_len,
            rel_width=rel_width,
            rel_height=rel_height,
        )
        temp2_out_slice = self.build_port_monitor_slice(
            port_name="out_port_2",
            slice_name="out_slice_2",
            rel_loc=1 - offset / port_len,
            rel_width=rel_width,
            rel_height=rel_height,
        )
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        if not self.is_3d:
            radiation_monitor = self.build_radiation_monitor(monitor_name="rad_slice")
            return (
                src_slice,
                refl_slice,
                temp1_out_slice,
                temp2_out_slice,
                radiation_monitor,
            )
        return (
            src_slice,
            refl_slice,
            temp1_out_slice,
            temp2_out_slice,
        )

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

        temp1_norm_monitor_profiles = self.build_norm_sources(
            source_modes=(self.mode,),
            input_port_name="out_port_1",
            input_slice_name="out_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )

        temp2_norm_monitor_profiles = self.build_norm_sources(
            source_modes=(self.mode,),
            input_port_name="out_port_2",
            input_slice_name="out_slice_2",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )

        return (
            norm_source_profiles,
            norm_refl_profiles_1,
            temp1_norm_monitor_profiles,
            temp2_norm_monitor_profiles,
        )
