from __future__ import annotations

import os
import sys
from collections import OrderedDict
from functools import partial
from typing import Tuple

import numpy as np
import torch

from core.utils import get_material_heat_optic_spec, material_fn_dict

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from core.invdes.models.layers.device_base import N_Ports
from core.utils import Air_heat_capacity

__all__ = ["ThermoOpticMRR"]


class ThermoOpticMRR(N_Ports):
    def __init__(
        self,
        material_r1: str = "Si_eff",  # waveguide bus material
        material_r2: str = "Si_eff",  # ring material
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
        ring_radius: float = 5.0,
        ring_width: float = 0.5,
        ring_height: float = 0.22,
        couple_gap: float = 0.1,
        h_clad: float = 2.8,
        h_box: float = 2.0,
        h_wafer: float = 0.5,
        port_len: Tuple[float] = (12.0, 12.0),
        port_width: Tuple[float] = (0.5, 0.5),
        port_height: Tuple[float] = (0.22, 0.22),
        heater_material: str = "TiN",
        heater_gap: float = 2.0,
        heater_height: float = 0.14,
        heater_width: float = 2.0,
        mode: str = "Hz1",
        is_3d: bool = False,
        heater_on_current: float = 15e-3,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.mode = mode
        self.ring_radius = float(ring_radius)
        self.ring_width = float(ring_width)
        self.ring_height = float(ring_height)
        self.couple_gap = float(couple_gap)
        self.heater_height = float(heater_height)
        self.heater_width = float(heater_width)
        self.heater_gap = float(heater_gap)
        self.h_clad = float(h_clad)
        self.h_box = float(h_box)
        self.h_wafer = float(h_wafer)
        self.heater_on_current = float(heater_on_current)
        self.logical_heat_source_name = "heater_1"
        self.is_3d = bool(is_3d)
        self.port_len = tuple(float(v) for v in port_len)
        self.port_width = tuple(float(v) for v in port_width)
        self.port_height = tuple(float(v) for v in port_height)

        wl_cen = float(sim_cfg["wl_cen"])

        self.mode = mode
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

        bg_spec = get_material_heat_optic_spec(material_bg, wl_cen)
        region_spec = get_material_heat_optic_spec(material_r1, wl_cen)
        region_bg_spec = get_material_heat_optic_spec(material_r2, wl_cen)
        heater_spec = get_material_heat_optic_spec(heater_material, wl_cen)

        grid_step = 1.0 / float(sim_cfg["resolution"])

        if self.is_3d:
            heat_cfg = {
                "backend": "jax",
                "mesh_type": "rectangular",
                "resolution": (0.2, 0.2, 0.05),
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
                "use_tidy3d": True,
            }
        else:
            heat_cfg = {
                "backend": "jax",
                "mesh_type": "rectangular",
                "resolution": (0.2, 0.2),
                "dirichlet_bc": {"ymin": 300.0},
                "neumann_bc": {"xmin": 0.0, "xmax": 0.0, "ymax": 0.0},
                "build_source_maps": True,
                "combine_sources": True,
                "use_tidy3d": True,
            }
        heat_cfg.update(sim_cfg.get("heat", {}))
        sim_cfg["heat"] = heat_cfg

        self.sim_cfg = sim_cfg

        self.y_wg_1 = (
            self.ring_radius
            + self.ring_width / 2
            + self.couple_gap
            + self.port_width[1] / 2.0
        )
        self.y_wg_2 = (
            -self.ring_radius
            - self.ring_width / 2
            - self.couple_gap
            - self.port_width[0] / 2.0
        )

        self.z_sim = (self.h_clad - self.h_box - self.h_wafer) / 2.0
        self.z_wg = self.ring_height / 2.0 - self.z_sim
        self.z_heater = (
            self.port_height[0] / 2 + self.heater_gap + self.heater_height / 2.0
        )
        self.z_wafer = -self.port_height[0] / 2 - self.h_box - self.h_wafer / 2.0

        self.h_optic_sim = 3.0

        port_cfgs = OrderedDict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[0.0, self.y_wg_1] + ([0] if self.is_3d else []),
                size=[self.port_len[0], self.port_width[0]]
                + ([self.port_height[0]] if self.is_3d else []),
                eps=eps_r1_fn(wl_cen),
                material=material_r1,
            ),
            drop_port=dict(
                type="box",
                direction="x",
                center=[0.0, self.y_wg_2] + ([0] if self.is_3d else []),
                size=[self.port_len[1], self.port_width[1]]
                + ([self.port_height[1]] if self.is_3d else []),
                eps=eps_r1_fn(wl_cen),
                material=material_r1,
            ),
        )

        cell_length = (
            self.port_len[0]
            + self.sim_cfg["border_width"][0]
            + self.sim_cfg["border_width"][1]
        )
        cell_width = (
            self.ring_radius * 2
            + self.heater_width
            + self.sim_cfg["border_width"][2]
            + self.sim_cfg["border_width"][3]
        )
        cell_height = abs(self.z_wafer - self.h_wafer / 2) * 2
        self.sim_cfg["cell_size"] = [cell_length, cell_width] + (
            [cell_height] if self.is_3d else []
        )

        geometry_cfgs = OrderedDict(
            cladding=dict(
                type="box",
                center=[
                    0.0,
                    0.0,
                    (self.h_clad + (self.z_wafer - self.h_wafer / 2)) / 2,
                ],
                size=[self.port_len[0], cell_width, self.h_clad],
                eps=eps_bg_fn(wl_cen),
                material=material_bg,
            ),
            ## need to put ports here, otherwise it will be overridden by cladding.
            port_1=dict(
                type="box",
                center=[0.0, self.y_wg_1] + ([0] if self.is_3d else []),
                size=[self.port_len[0], self.port_width[0]]
                + ([self.port_height[0]] if self.is_3d else []),
                eps=eps_r1_fn(wl_cen),
                material=material_r1,
            ),
            port_2=dict(
                type="box",
                center=[0.0, self.y_wg_2] + ([0] if self.is_3d else []),
                size=[self.port_len[1], self.port_width[1]]
                + ([self.port_height[1]] if self.is_3d else []),
                eps=eps_r1_fn(wl_cen),
                material=material_r1,
            ),
            wafer=dict(
                type="box",
                center=[0.0, 0.0, self.z_wafer],
                size=[self.port_len[0], cell_width, self.h_wafer],
                eps=eps_r1_fn(wl_cen),
                material=material_r1,
            ),
            ring=dict(
                type="clip_operation",
                operation="difference",
                geometry_a=dict(
                    type="cylinder",
                    center=[0.0, 0.0] + ([0.0] if self.is_3d else []),
                    radius=self.ring_radius + self.ring_width / 2.0,
                    height=self.ring_height if self.is_3d else 0,
                ),
                geometry_b=dict(
                    type="cylinder",
                    center=[0.0, 0.0] + ([0.0] if self.is_3d else []),
                    radius=max(self.ring_radius - self.ring_width / 2.0, 0.0),
                    height=self.ring_height if self.is_3d else 0,
                ),
                eps=eps_r2_fn(wl_cen),
                material=material_r2,
            ),
            heater_1=dict(
                type="clip_operation",
                operation="difference",
                geometry_a=dict(
                    type="cylinder",
                    center=[0.0, 0.0, self.z_heater],
                    radius=self.ring_radius + self.heater_width / 2.0,
                    height=self.heater_height,
                ),
                geometry_b=dict(
                    type="geometry_group",
                    geometries=[
                        dict(
                            type="cylinder",
                            center=[0.0, 0.0, self.z_heater],
                            radius=max(self.ring_radius - self.heater_width / 2.0, 0.0),
                            height=self.heater_height * 1.5,
                        ),
                        dict(
                            type="box",
                            center=[0.0, -self.ring_radius]
                            + ([self.z_heater] if self.is_3d else []),
                            size=[0.5 * self.heater_width, 1.5 * self.heater_width]
                            + ([1.5 * self.heater_height] if self.is_3d else []),
                        ),
                    ],
                ),
                eps=heater_eps_fn(wl_cen),
                material=heater_material,
                sigma=heater_spec.electrical_conductivity,
            ),
        )

        heater_cross_section_area = max(heater_width * heater_height, 1e-6)
        heat_source_cfgs = {
            self.logical_heat_source_name: dict(
                type="clip_operation",
                operation="difference",
                geometry_a=dict(
                    type="cylinder",
                    center=[0.0, 0.0, self.z_heater],
                    radius=self.ring_radius + self.heater_width / 2.0,
                    height=self.heater_height,
                ),
                geometry_b=dict(
                    type="geometry_group",
                    geometries=[
                        dict(
                            type="cylinder",
                            center=[0.0, 0.0, self.z_heater],
                            radius=max(self.ring_radius - self.heater_width / 2.0, 0.0),
                            height=self.heater_height * 1.5,
                        ),
                        dict(
                            type="box",
                            center=[0.0, -self.ring_radius]
                            + ([self.z_heater] if self.is_3d else []),
                            size=[0.5 * self.heater_width, 1.5 * self.heater_width]
                            + ([1.5 * self.heater_height] if self.is_3d else []),
                        ),
                    ],
                ),
                current=heater_on_current,
                material=heater_material,
                current_direction="x",
                cross_section_area=heater_cross_section_area,
                mesh_override_dl=(None, None, 0.02),
            ),
        }

        super().__init__(
            eps_bg=eps_bg_fn(wl_cen),
            k_bg=bg_spec.thermal_conductivity,
            heat_capacity_bg=bg_spec.heat_capacity,
            thermo_optic_coeff_bg=bg_spec.thermo_optic_coeff,
            port_cfgs=port_cfgs,
            geometry_cfgs=geometry_cfgs,
            design_region_cfgs={},
            heat_source_cfgs=heat_source_cfgs,
            sim_cfg=sim_cfg,
            device=device,
            verbose=True,
        )

    # def _thermal_combined_geometry_cfgs(self):
    #     combined_cfgs = OrderedDict()
    #     combined_cfgs.update(self.port_cfgs)
    #     combined_cfgs.update(self.geometry_cfgs)
    #     combined_cfgs.update(self._thermal_geometry_cfgs)
    #     return combined_cfgs

    # def build_thermal_property_maps(self, use_tidy3d: bool | None = None):
    #     if use_tidy3d is None:
    #         use_tidy3d = self._heat_use_tidy3d_default()

    #     spacing = self._heat_mesh_spacing()
    #     cell_size = self._heat_mesh_cell_size()
    #     resolution = self._heat_mesh_resolution_for_raster() or self.resolution
    #     pml = self._heat_mesh_pml()
    #     cfgs = self._thermal_combined_geometry_cfgs()

    #     self.conductivity_map = self._build_scalar_property_map(
    #         cfgs=cfgs,
    #         map_name="thermal conductivity map",
    #         value_keys=("thermal_conductivity", "k"),
    #         bg_value=self.k_bg,
    #         value_from_cfg=self._heat_value_from_cfg,
    #         bg_from_cfg=self._heat_bg_from_cfg,
    #         cell_size=cell_size,
    #         PML=pml,
    #         resolution=resolution,
    #         use_tidy3d=use_tidy3d,
    #         spacing=spacing,
    #     )
    #     self.heat_capacity_map = self._build_scalar_property_map(
    #         cfgs=cfgs,
    #         map_name="heat capacity map",
    #         value_keys=("heat_capacity", "capacity"),
    #         bg_value=Air_heat_capacity(self.sim_cfg["wl_cen"]),
    #         value_from_cfg=self._heat_capacity_value_from_cfg,
    #         bg_from_cfg=self._heat_capacity_bg_from_cfg,
    #         cell_size=cell_size,
    #         PML=pml,
    #         resolution=resolution,
    #         use_tidy3d=use_tidy3d,
    #         spacing=spacing,
    #     )
    #     self.thermal_capacity_map = self.heat_capacity_map
    #     self.thermo_optic_coeff_map = self._build_scalar_property_map(
    #         cfgs=cfgs,
    #         map_name="thermo-optic coefficient map",
    #         value_keys=("thermo_optic_coeff", "dn_dT"),
    #         bg_value=self.thermo_optic_coeff_bg,
    #         value_from_cfg=self._thermo_optic_value_from_cfg,
    #         bg_from_cfg=self._thermo_optic_bg_from_cfg,
    #         cell_size=cell_size,
    #         PML=pml,
    #         resolution=resolution,
    #         use_tidy3d=use_tidy3d,
    #         spacing=spacing,
    #     )
    #     self.thermal_coefficient_map = self.thermo_optic_coeff_map
    #     self.thermal_grid_spacing = tuple(float(v) for v in spacing)
    #     self.thermal_grid_shape = tuple(int(v) for v in self.conductivity_map.shape)
    #     self.thermal_coords = self._heat_coords_from_shape_spacing(
    #         self.thermal_grid_shape,
    #         self.thermal_grid_spacing,
    #     )
    #     self.thermal_design_region_masks = {}
    #     self.thermal_design_region_mask_weights = {}
    #     self.thermal_design_region_axis_weights = {}
    #     self.material_property_maps.update(
    #         {
    #             "conductivity": self.conductivity_map,
    #             "heat_capacity": self.heat_capacity_map,
    #             "thermal_capacity": self.thermal_capacity_map,
    #             "thermo_optic_coeff": self.thermo_optic_coeff_map,
    #             "thermal_coefficient": self.thermal_coefficient_map,
    #         }
    #     )
    #     self.material_property_backend = "tidy3d" if use_tidy3d else "meep"
    #     return {
    #         "conductivity": self.conductivity_map,
    #         "heat_capacity": self.heat_capacity_map,
    #         "thermal_capacity": self.thermal_capacity_map,
    #         "thermo_optic_coeff": self.thermo_optic_coeff_map,
    #         "thermal_coefficient": self.thermal_coefficient_map,
    #     }

    # def build_heat_source_maps(
    #     self,
    #     heat_source_cfgs: dict | None = None,
    #     *,
    #     use_tidy3d: bool | None = None,
    #     combine: bool | None = None,
    # ):
    #     if use_tidy3d is None:
    #         use_tidy3d = self._heat_use_tidy3d_default()
    #     if combine is None:
    #         combine = self._heat_combine_sources_default()

    #     cfgs = (
    #         self._thermal_heat_source_cfgs_template
    #         if heat_source_cfgs is None
    #         else heat_source_cfgs
    #     )
    #     spacing = self._heat_mesh_spacing()
    #     cell_size = self._heat_mesh_cell_size()
    #     resolution = self._heat_mesh_resolution_for_raster() or self.resolution
    #     pml = self._heat_mesh_pml()

    #     built_sources = {}
    #     for name, cfg in cfgs.items():
    #         source_map = self._build_scalar_property_map(
    #             cfgs={name: cfg},
    #             map_name=f"heat source map '{name}'",
    #             value_keys=(
    #                 "heat_density",
    #                 "q",
    #                 "q_density",
    #                 "source",
    #                 "total_power",
    #                 "current",
    #             ),
    #             bg_value=0.0,
    #             value_from_cfg=self._heat_source_value_from_cfg,
    #             bg_from_cfg=self._heat_source_bg_from_cfg,
    #             cell_size=cell_size,
    #             PML=pml,
    #             resolution=resolution,
    #             use_tidy3d=use_tidy3d,
    #             spacing=spacing,
    #         )
    #         if source_map is not None:
    #             built_sources[name] = source_map

    #     self.heat_sources_dict = built_sources
    #     if combine and built_sources:
    #         combined = np.zeros_like(next(iter(built_sources.values())))
    #         for source_map in built_sources.values():
    #             combined = combined + source_map
    #         self.heat_source_map = combined
    #     elif combine:
    #         self.heat_source_map = None
    #     return self.heat_sources_dict

    # def build_runtime_heat_source_map(self, currents: dict[str, float] | None):
    #     normalized_currents = self._normalized_runtime_currents(currents)
    #     cached_q_map = self._get_cached_runtime_heat_source_map(normalized_currents)
    #     if cached_q_map is not None:
    #         return cached_q_map

    #     current = float(normalized_currents.get(self.logical_heat_source_name, 0.0))
    #     runtime_cfgs = OrderedDict()
    #     for name, cfg in self._thermal_heat_source_cfgs_template.items():
    #         cfg_copy = dict(cfg)
    #         if name == "heater_outer":
    #             cfg_copy["current"] = current
    #         runtime_cfgs[name] = cfg_copy

    #     built_sources = self.build_heat_source_maps(
    #         heat_source_cfgs=runtime_cfgs, combine=True
    #     )
    #     if not built_sources:
    #         if self.conductivity_map is None:
    #             self.build_thermal_property_maps()
    #         q_map = torch.zeros(
    #             tuple(int(v) for v in self.thermal_grid_shape),
    #             dtype=torch.float32,
    #             device=self.device,
    #         )
    #         return self._store_cached_runtime_heat_source_map(
    #             normalized_currents, q_map
    #         )

    #     q_map = torch.as_tensor(
    #         self.heat_source_map, dtype=torch.float32, device=self.device
    #     )
    #     return self._store_cached_runtime_heat_source_map(normalized_currents, q_map)

    def init_monitors(self, verbose: bool = True):
        if verbose:
            print("Start generating sources and monitors ...", flush=True)
        rel_width = 4.0
        rel_height = 4.5
        pml = float(self.sim_cfg["PML"][0])
        offset = 0.2 + pml
        port_len = float(self.port_cfgs["in_port_1"]["size"][0])

        src_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_slice_1",
            rel_loc=offset / port_len,
            rel_width=rel_width,
            rel_height=rel_height,
            direction="x+",
        )
        refl_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="refl_slice_1",
            rel_loc=(offset + 0.15) / port_len,
            rel_width=rel_width,
            rel_height=rel_height,
            direction="x-",
        )
        through_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="through_slice",
            rel_loc=1.0 - offset / port_len,
            rel_width=rel_width,
            rel_height=rel_height,
            direction="x+",
        )
        drop_slice = self.build_port_monitor_slice(
            port_name="drop_port",
            slice_name="drop_slice",
            rel_loc=offset / port_len,
            rel_width=rel_width,
            rel_height=rel_height,
            direction="x-",
        )
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        return src_slice, refl_slice, through_slice, drop_slice

    def norm_run(self, verbose: bool = True):
        if verbose:
            print("Start normalization run ...", flush=True)
        common = dict(
            source_modes=(self.mode,),
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=False,
        )
        self.build_norm_sources(
            input_port_name="in_port_1",
            input_slice_name="in_slice_1",
            require_sim=True,
            **common,
        )
        self.build_norm_sources(
            input_port_name="in_port_1",
            input_slice_name="refl_slice_1",
            require_sim=False,
            **common,
        )
        self.build_norm_sources(
            input_port_name="in_port_1",
            input_slice_name="through_slice",
            require_sim=False,
            **common,
        )
        self.build_norm_sources(
            input_port_name="drop_port",
            input_slice_name="drop_slice",
            require_sim=False,
            **common,
        )
        return self.port_sources_dict


if __name__ == "__main__":
    """https://www.flexcompute.com/tidy3d/examples/notebooks/ThermallyTunedRingResonator/"""
    device = torch.device("cuda:0")
    wl_cen = 1.55
    resolution = 30
    exp_name = "mrr_test"
    heat_padding = [5, 5, 5, 5, 0, 0]
    from core.invdes.models.base_optimization import DefaultSimulationConfig

    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="fdtdx",
            border_width=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            PML=[0.5, 0.5, 0.5],
            resolution=resolution,
            optical_grid={
                "mode": "fdtdx_rectilinear_nonuniform",
                "wavelength": wl_cen,
                "grid_x": {"type": "auto", "min_steps_per_wvl": 15},
                "grid_y": {"type": "auto", "min_steps_per_wvl": 15},
                "grid_z": {"type": "auto", "min_steps_per_wvl": 15},
            },
            plot_root=f"./figs/{exp_name}",
            wl_cen=1.55,
            wl_width=0,
            n_wl=1,
            numerical_solver="solve_direct",
            use_autodiff=False,
            heat={
                "backend": "jax",
                "mesh_type": "fixed_rectilinear_nonuniform",
                "use_tidy3d": True,
                "grid_spec": {
                    "wavelength": 1.55,
                    "grid_x": {"type": "auto", "min_steps_per_wvl": 12},
                    "grid_y": {"type": "auto", "min_steps_per_wvl": 12},
                    "grid_z": {"type": "auto", "min_steps_per_wvl": 25},
                },
                "padding": {
                    "distance": [3.9, 3.9, 3.0, 3.0, 0.0, 0.0],
                    "mode": "replicate",
                    "source_mode": "constant",
                    "scheme": "graded",
                    "growth_rate": 1.07,
                    "max_scale": 3.0,
                },
                "build_source_maps": True,
                "combine_sources": True,
                "requires_temp_grad": False,
                "solver_options": {
                    "dirichlet_mode": "reduced_free_dof",
                    "jax_solver": {
                        "method": "cg",
                        "precond": True,
                        "solve_dtype": "float64",
                        "check_residual": True,
                        "residual_rtol": 1e-6,
                        "residual_atol": 1e-8,
                    },
                },
                "adjoint_solver_options": {
                    "dirichlet_mode": "reduced_free_dof",
                    "jax_solver": {
                        "method": "cg",
                        "precond": True,
                        "solve_dtype": "float64",
                        "check_residual": True,
                        "residual_rtol": 1e-6,
                        "residual_atol": 1e-8,
                    },
                },
            },
        )
    )
    heater_current = 15e-3

    mrr = ThermoOpticMRR(
        material_r1="Si",
        material_r2="Si",
        material_bg="SiO2",
        sim_cfg=sim_cfg,
        ring_radius=5.0,
        ring_width=0.5,
        ring_height=0.22,
        couple_gap=0.1,
        h_clad=2.8,
        h_box=2.0,
        h_wafer=0.5,
        port_len=(12.2, 12.2),
        port_width=(0.5, 0.5),
        port_height=(0.22, 0.22),
        heater_material="TiN",
        heater_gap=2.0,
        heater_height=0.14,
        heater_width=2.0,
        mode="Hz1",
        is_3d=True,
        heater_on_current=heater_current,
        device=device,
    )
    print(mrr)
    mrr.build_electrical_conductivity_map(use_tidy3d=True)
    eps_map = mrr.epsilon_map
    mrr.init_monitors()
    mrr.norm_run()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    mrr.plot_eps(
        z=0,
        cmap="Greys",
        value_mode="real",
        ax=axes[0, 0],
        overlay_monitors=True,
        overlay_pml=True,
        overlay_heat_sources=True,
        title="epsilon slice at z=0",
    )
    mrr.plot_eps(
        y=0,
        cmap="Greys",
        value_mode="real",
        ax=axes[0, 1],
        overlay_monitors=True,
        overlay_pml=True,
        overlay_heat_sources=True,
        title="epsilon slice at y=0",
    )
    mrr.plot_eps(
        x=0,
        cmap="Greys",
        value_mode="real",
        ax=axes[0, 2],
        overlay_monitors=True,
        overlay_pml=True,
        overlay_heat_sources=True,
        title="epsilon slice at x=0",
    )
    mrr.plot_eps(
        z=mrr.z_heater,
        cmap="Greys",
        ax=axes[1, 0],
        overlay_monitors=True,
        overlay_pml=True,
        overlay_heat_sources=True,
        title=f"epsilon slice at heater z={mrr.z_heater:.3f}",
    )
    mrr.plot_eps(
        y=0,
        cmap="Greys",
        ax=axes[1, 1],
        overlay_monitors=True,
        overlay_pml=True,
        overlay_heat_sources=True,
        title=f"epsilon slice at y={0:.3f}",
    )
    mrr.plot_eps(
        x=0,
        cmap="Greys",
        ax=axes[1, 2],
        overlay_monitors=True,
        overlay_pml=True,
        overlay_heat_sources=True,
        title=f"epsilon slice at x={0:.3f}",
    )

    plt.tight_layout()
    os.makedirs(f"./figs/{exp_name}", exist_ok=True)
    plt.savefig(f"./figs/{exp_name}/mrr_eps_map_cross_sections.png", dpi=300)

    thermal_maps = mrr.build_thermal_property_maps(use_tidy3d=True)
    currents = {"heater_1": heater_current}
    q_map = mrr.build_runtime_heat_source_map(currents).to(device)
    conductivity_map = torch.as_tensor(
        thermal_maps["conductivity"], dtype=torch.float32, device=device
    )
    print(
        "[MRR HEAT] conductivity shape:",
        tuple(int(v) for v in conductivity_map.shape),
        "range:",
        (
            float(conductivity_map.min().item()),
            float(conductivity_map.max().item()),
        ),
    )
    print(
        "[MRR HEAT] q_map shape:",
        tuple(int(v) for v in q_map.shape),
        "range:",
        (float(q_map.min().item()), float(q_map.max().item())),
        "sum:",
        float(q_map.sum().item()),
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    mrr.plot_property(
        "conductivity",
        property_map=conductivity_map,
        z=mrr.z_heater,
        cmap="magma",
        ax=axes[0, 0],
        title=f"conductivity at heater z={mrr.z_heater:.3f}",
        overlay_monitors=False,
    )
    mrr.plot_property(
        "conductivity",
        property_map=conductivity_map,
        x=0,
        cmap="magma",
        ax=axes[0, 1],
        title="conductivity at x=0",
        overlay_monitors=False,
    )
    mrr.plot_property(
        "conductivity",
        property_map=conductivity_map,
        y=0,
        cmap="magma",
        ax=axes[0, 2],
        title="conductivity at y=0",
        overlay_monitors=False,
    )
    mrr.plot_property(
        "heat_source",
        property_map=q_map,
        z=mrr.z_heater,
        cmap="inferno",
        ax=axes[1, 0],
        title=f"heat source at heater z={mrr.z_heater:.3f}",
        overlay_monitors=False,
    )
    mrr.plot_property(
        "heat_source",
        property_map=q_map,
        x=0,
        cmap="inferno",
        ax=axes[1, 1],
        title="heat source at x=0",
        overlay_monitors=False,
    )
    mrr.plot_property(
        "heat_source",
        property_map=q_map,
        y=0,
        cmap="inferno",
        ax=axes[1, 2],
        title="heat source at y=0",
        overlay_monitors=False,
    )
    plt.tight_layout()
    fig.savefig(f"./figs/{exp_name}/mrr_heat_inputs.png", dpi=300)

    temperature_map = mrr.solve_heat(
        k_map=thermal_maps["conductivity"],
        q_map=q_map,
    )
    print(torch.max(temperature_map), torch.min(temperature_map))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mrr.plot_property(
        property_map=temperature_map,
        z=0,
        cmap="coolwarm",
        ax=axes[0],
        overlay_monitors=False,
    )
    mrr.plot_property(
        property_map=temperature_map,
        x=0,
        cmap="coolwarm",
        ax=axes[1],
        overlay_monitors=False,
    )
    mrr.plot_property(
        property_map=temperature_map,
        y=0,
        cmap="coolwarm",
        ax=axes[2],
        overlay_monitors=False,
    )
    fig.savefig(f"./figs/{exp_name}/mrr_temperature_map.png", dpi=300)

    exit(0)
    fields = mrr.solve_fdtdx(
        eps_map,
        input_slice_name="in_slice_1",  # every simulation, only one source
        wl_cen=1.55,
        wl_width=0,
        n_wl=1,
        mode="Hz1",
    )

    Ey = fields["Ey"].cpu().numpy()[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im = axes[0].imshow(Ey[:, :, Ey.shape[2] // 2].T.real, cmap="RdBu")
    plt.colorbar(im, ax=axes[0])
    im = axes[1].imshow(Ey[:, Ey.shape[1] // 2, :].T.real, cmap="RdBu")
    plt.colorbar(im, ax=axes[1])
    im = axes[2].imshow(Ey[Ey.shape[0] // 2, :, :].T.real, cmap="RdBu")
    plt.colorbar(im, ax=axes[2])
    plt.savefig(f"./figs/{exp_name}/mrr_eps_map_cross_sections_Ey.png", dpi=300)
