"""
Date: 2024-10-02 20:59:04
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-09 02:57:17
FilePath: /MAPS/core/invdes/models/layers/device_base.py
"""

import copy
import os
from functools import lru_cache
from typing import Tuple

import meep as mp
import numpy as np
import torch
from pyutils.config import Config
from pyutils.general import ensure_dir

from core.fdfd import fdfd_ez as fdfd_ez_torch
from core.fdfd import fdfd_hz as fdfd_hz_torch
from core.invdes.models.layers.utils import modulation_fn_dict
from core.utils import (
    Si_eps,
    SiO2_eps,
    Slice,
    get_eigenmode_coefficients,
    get_flux,
)
from thirdparty.ceviche import fdfd_ez, fdfd_hz
from thirdparty.ceviche.constants import C_0, MICRON_UNIT

from .utils import (
    apply_regions_gpu,
    get_grid,
    insert_mode,
    insert_mode_spins,
    plot_eps_field,
)
import matplotlib.pyplot as plt
__all__ = ["BaseDevice", "N_Ports"]


class SimulationConfig(Config):
    def __init__(self):
        super().__init__()
        self.update(
            dict(
                device=dict(
                    type="",
                    cfg=dict(),
                ),
                sources=[],
                simulation=dict(),
            )
        )


class BaseDevice(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = SimulationConfig()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.sources = []
        self.geometry = {}
        self.sim = None

    def build_ports(self):
        ### build geometry for input/output ports
        pass

    def update_device_config(self, device_type, device_cfg):
        self.config.device.type = device_type
        self.config.device.update(dict(cfg=device_cfg))

    def reset_device_config(self):
        self.config.device.type = ""
        self.config.device.update(dict(cfg=dict()))

    def add_source_config(self, source_config):
        self.config.sources.append(source_config)

    def reset_source_config(self):
        self.config.sources = []

    def update_simulation_config(self, simulation_config):
        self.config.update(dict(simulation=simulation_config))

    def reset_simulation_config(self):
        self.config.update(dict(simulation=dict()))

    def dump_config(self, filepath, verbose=False):
        ensure_dir(os.path.dirname(filepath))
        self.config.dump_to_yml(filepath)
        if verbose:
            print(f"Dumped device config to {filepath}")

    def trim_pml(self, resolution, PML, x):
        PML = [int(round(i * resolution)) for i in PML]
        return x[..., PML[0] : -PML[0], PML[1] : -PML[1]]


def get_two_ports(device, port_name):
    port = device.port_cfgs[port_name]
    center = port["center"]
    size = port["size"]
    direction = port["direction"]
    eps = port["eps"]
    cell_size = device.cell_size
    if direction == "x":
        center = [0, center[1]]
        size = [cell_size[0], size[1]]
    elif direction == "y":
        center = [center[0], 0]
        size = [size[0], cell_size[1]]
    else:
        raise ValueError(f"Direction {direction} not supported")
    sim_cfg = copy.deepcopy(device.sim_cfg)
    sim_cfg["cell_size"] = device.cell_size
    two_ports = N_Ports(
        eps_bg=device.eps_bg,
        port_cfgs={
            port_name: dict(
                type="box",
                direction=direction,
                center=center,
                size=size,
                eps=eps,
            ),
        },
        design_region_cfgs=dict(),
        sim_cfg=sim_cfg,
        device=device.device,
    )
    return two_ports


class N_Ports(BaseDevice):
    def __init__(
        self,
        eps_bg: float = SiO2_eps(1.55),
        port_cfgs=dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-1.5, 0],
                size=[3, 0.48],
                eps=Si_eps(1.55),
            ),
            out_port_1=dict(
                type="box",
                direction="x",
                center=[1.5, 0],
                size=[3, 0.48],
                eps=Si_eps(1.55),
            ),
        ),
        geometry_cfgs=dict(),
        design_region_cfgs=dict(
            region_1=dict(
                type="box",
                center=[0, 0],
                size=[1, 1],
                eps_bg=SiO2_eps(1.55),
                eps=Si_eps(1.55),
            )
        ),
        active_region_cfgs=dict(),
        sim_cfg: dict = {
            "border_width": [
                0,
                0,
                1.5,
                1.5,
            ],  # left, right, lower, upper, containing PML
            "PML": [1, 1],  # left/right, lower/upper
            "cell_size": None,
            "resolution": 50,
            "wl_cen": 1.55,
            "wl_width": 0,
            "n_wl": 1,
            "plot_root": "./figs",
        },
        device="cuda:0",
    ):
        super().__init__()
        self.eps_bg = eps_bg
        self.port_cfgs = port_cfgs
        self.geometry_cfgs = geometry_cfgs

        self.design_region_cfgs = design_region_cfgs
        self.active_region_cfgs = active_region_cfgs

        self.resolution = sim_cfg["resolution"]
        self.grid_step = 1 / self.resolution

        device_cfg = dict(
            port_cfgs=port_cfgs,
            geometry_cfgs=geometry_cfgs,
            eps_bg=eps_bg,
            resolution=self.resolution,
            grid_step=self.grid_step,
        )
        self.device = device
        super().__init__(**device_cfg)
        self.update_device_config(self.__class__.__name__, device_cfg)
        self.update_simulation_config(sim_cfg)
        self.sim_cfg = sim_cfg
        self.add_geometries(port_cfgs)
        self.add_geometries(geometry_cfgs)
        ## do not add design region to geometry, otherwise meep will have subpixel smoothing on the border
        ## but need to consider this in bounding box
        # self.add_geometries(design_region_cfgs)

        if self.sim_cfg["cell_size"] is None or self.sim_cfg["cell_size"] == "None":
            self.cell_size = self.get_geometry_box(
                border_width=sim_cfg["border_width"], PML=sim_cfg["PML"]
            )
        else:
            self.cell_size = sim_cfg["cell_size"]
        ### here we use ceil to match meep

        self.epsilon_map = self.get_epsilon_map(
            self.cell_size,
            self.geometry,
            sim_cfg["PML"],
            self.resolution,
            self.eps_bg,
        )

        self.Nx, self.Ny, self.Nz = [
            int(round(i * self.resolution)) for i in self.cell_size
        ]  # change math.ceil to round since sometimes we will have like 10.200000000000001 in the cell_size which will cause a size mismatch
        self.Nx, self.Ny = self.epsilon_map.shape
        self.Nz = int(round(self.cell_size[-1] * self.resolution))
        self.NPML = [int(round(i * self.resolution)) for i in sim_cfg["PML"]]
        self.xs, self.ys = get_grid((self.Nx, self.Ny), self.grid_step)

        self.design_region_masks = self.build_design_region_mask(design_region_cfgs)
        ## active region must within the design region
        self.active_region_masks = self.build_active_region_mask(active_region_cfgs)
        self.ports_regions = self.build_port_region(port_cfgs)

        self.port_monitor_slices = {}  # {port_name: Slice or mask}
        self.port_monitor_slices_info = {}  # {port_name: dict of slice info}
        self.port_sources_dict = {}  # {slice_name: {(wl, mode): (profile, ht_m, et_m, norm_power)}}

    def add_geometries(self, cfgs):
        for name, cfg in cfgs.items():
            self.add_geometry(name, cfg)

    def add_geometry(self, name, cfg):
        geo_type = cfg["type"]
        eps_r = cfg["eps"]
        eps_bg = cfg.get("eps_bg", eps_r)
        eps_r = (eps_r + eps_bg) / 2

        match geo_type:
            case "box":
                geometry = mp.Block(
                    mp.Vector3(*cfg["size"]),
                    center=mp.Vector3(*cfg["center"]),
                    material=mp.Medium(epsilon=eps_r),
                )
            case "prism":
                geometry = mp.Prism(
                    [mp.Vector3(*v) for v in cfg["vertices"]],
                    height=cfg.get("height", mp.inf),
                    material=mp.Medium(epsilon=eps_r),
                )
            case _:
                raise ValueError(f"Geometry type {geo_type} not supported")

        self.geometry[name] = geometry

    def get_geometry_box(self, border_width=[0, 0], PML=[0, 0]):
        left, lower = float("inf"), float("inf")
        right, upper = float("-inf"), float("-inf")
        for design_region in self.design_region_cfgs.values():
            left = min(left, design_region["center"][0] - design_region["size"][0] / 2)
            right = max(
                right, design_region["center"][0] + design_region["size"][0] / 2
            )
            lower = min(
                lower, design_region["center"][1] - design_region["size"][1] / 2
            )
            upper = max(
                upper, design_region["center"][1] + design_region["size"][1] / 2
            )

        for geometry in self.geometry.values():
            if isinstance(geometry, mp.Block):
                left = min(left, geometry.center.x - geometry.size.x / 2)
                right = max(right, geometry.center.x + geometry.size.x / 2)
                lower = min(lower, geometry.center.y - geometry.size.y / 2)
                upper = max(upper, geometry.center.y + geometry.size.y / 2)
            elif isinstance(geometry, mp.Prism):
                for vertex in geometry.vertices:
                    left = min(left, vertex.x)
                    right = max(right, vertex.x)
                    lower = min(lower, vertex.y)
                    upper = max(upper, vertex.y)
            else:
                raise ValueError(f"Geometry type {type(geometry)} not supported")
        sx = (
            right - left + border_width[0] + border_width[1]
        )  # PML is already contained in border
        sy = (
            upper - lower + border_width[2] + border_width[3]
        )  # PML is already contained in border
        return (sx, sy, 0)

    def get_epsilon_map(self, cell_size, geometry, PML, resolution, eps_bg):
        boundary = [
            mp.PML(PML[0], direction=mp.X),
            mp.PML(PML[1], direction=mp.Y),
        ]
        sim = mp.Simulation(
            resolution=resolution,
            cell_size=mp.Vector3(*cell_size),
            boundary_layers=boundary,
            geometry=list(geometry.values()),
            sources=None,
            default_material=mp.Medium(epsilon=eps_bg),
            eps_averaging=False,
        )
        sim.run(until=0)
        epsilon_map = sim.get_epsilon().astype(np.float32)
        return epsilon_map

    def build_design_region_mask(self, design_region_cfgs):
        design_region_masks = {}
        for name, cfg in design_region_cfgs.items():
            center = cfg["center"]
            size = cfg["size"]
            left = center[0] - size[0] / 2 + self.cell_size[0] / 2
            right = left + size[0]
            lower = center[1] - size[1] / 2 + self.cell_size[1] / 2
            upper = lower + size[1]
            left = max(0, int(np.round(left / self.grid_step)))
            right = min(
                self.Nx, int(np.round(right / self.grid_step)) + 1
            )  # +1 to include the right boundary
            lower = max(0, int(np.round(lower / self.grid_step)))
            upper = min(
                self.Ny, int(np.round(upper / self.grid_step)) + 1
            )  # +1 to include the right boundary
            region = Slice(
                x=slice(left, right), y=slice(lower, upper)
            )  # a rectangular region
            design_region_masks[name] = region

        return design_region_masks

    def build_active_region_mask(self, active_region_cfgs):
        active_region_masks = {}
        for name, cfg in active_region_cfgs.items():
            assert (
                name in self.design_region_masks
            ), f"Active region {name} not found in design region"
            design_region_cfg = self.design_region_cfgs[name]
            center = cfg["center"]
            size = cfg["size"]
            design_center, design_size = (
                design_region_cfg["center"],
                design_region_cfg["size"],
            )

            left = center[0] - size[0] / 2 + self.cell_size[0] / 2
            right = left + size[0]
            lower = center[1] - size[1] / 2 + self.cell_size[1] / 2
            upper = lower + size[1]

            ## active region must be contained in design region
            assert (
                center[0] - size[0] / 2 >= design_center[0] - design_size[0] / 2
            ), f"Active region {name} left boundary ({center[0] - size[0] / 2}) out of design region ({design_center[0] - design_size[0] / 2})"
            assert (
                center[0] + size[0] / 2 <= design_center[0] + design_size[0] / 2
            ), f"Active region {name} right boundary ({center[0] + size[0] / 2}) out of design region ({design_center[0] + design_size[0] / 2})"
            assert (
                center[1] - size[1] / 2 >= design_center[1] - design_size[1] / 2
            ), f"Active region {name} lower boundary ({center[1] - size[1] / 2}) out of design region ({design_center[1] - design_size[1] / 2})"
            assert (
                center[1] + size[1] / 2 <= design_center[1] + design_size[1] / 2
            ), f"Active region {name} upper boundary ({center[1] + size[1] / 2}) out of design region ({design_center[1] + design_size[1] / 2})"
            left = int(np.round(left / self.grid_step))
            right = int(np.round(right / self.grid_step))
            lower = int(np.round(lower / self.grid_step))
            upper = int(np.round(upper / self.grid_step))
            region = Slice(
                x=slice(left, right + 1), y=slice(lower, upper + 1)
            )  # a rectangular region
            active_region_masks[name] = region

        return active_region_masks

    def apply_active_modulation(self, eps, control_cfgs):
        ## eps_r: permittivity tensor, denormalized
        ## control_cfgs, include control signals for (multiple) active region(s).
        if isinstance(eps, torch.Tensor):
            eps_copy = eps.clone()
        else:
            eps_copy = eps.copy()
        for name, control_cfg in control_cfgs.items():
            design_region_cfg = self.design_region_cfgs[name]
            eps_bg, eps_r = design_region_cfg["eps_bg"], design_region_cfg["eps"]
            active_region_cfg = self.active_region_cfgs[name]
            method = active_region_cfg["method"]
            eps_r_cfg = active_region_cfg["eps_r"]
            eps_bg_cfg = active_region_cfg["eps_bg"]
            mod_fn = modulation_fn_dict[method]

            eps_r_new = mod_fn(eps_r, **eps_r_cfg, **control_cfg)
            eps_bg_new = mod_fn(eps_bg, **eps_bg_cfg, **control_cfg)

            active_region = self.active_region_masks[name]
            eps_region = (eps[active_region] - eps_bg) / (eps_r - eps_bg) * (
                eps_r_new - eps_bg_new
            ) + eps_bg_new
            eps_copy[active_region] = eps_region
        return eps_copy

    def build_port_region(self, port_cfgs, rel_width=2):
        ports_regions = []
        for name, cfg in port_cfgs.items():
            center = cfg["center"]
            size = cfg["size"]
            direction = cfg["direction"]
            if direction == "x":
                region = lambda x, y, center=center, size=size: (
                    torch.abs(x - center[0]) < size[0] / 2
                ) * (torch.abs(y - center[1]) < size[1] / 2 * rel_width)
            elif direction == "y":
                region = lambda x, y, center=center, size=size: (
                    torch.abs(x - center[0]) < size[0] / 2 * rel_width
                ) * (torch.abs(y - center[1]) < size[1] / 2)
            ports_regions.append(region)
        ports_regions = apply_regions_gpu(
            ports_regions, self.xs, self.ys, eps_r_list=1, eps_bg=0, device=self.device
        )
        return ports_regions.astype(np.bool_)

    def add_monitor_slice(
        self,
        slice_name: str,
        center: Tuple[int, int],
        size: Tuple[int, int],
        direction: str | None = None,
    ):
        '''
        the center is the center of the slice in um within the coordinate system where the center is (0, 0)
        the size is in the unit of um
        '''
        assert size[0] == 0 or size[1] == 0, "Only 1D slice is supported"
        if direction is None:
            direction = "x" if size[0] == 0 else "y"

        if direction[0] == "x":
            monitor_center = [
                int(round((c + offset / 2) / self.grid_step))
                for c, offset in zip(center, self.cell_size)
            ]
            monitor_half_width = int(round(size[1] / 2 / self.grid_step))
            monitor_slice = Slice(
                x=np.array(monitor_center[0]),
                y=np.arange(
                    max(0, monitor_center[1] - monitor_half_width),
                    min(self.Ny, monitor_center[1] + monitor_half_width),
                ),
            )
        elif direction[0] == "y":
            monitor_center = [
                int(round((c + offset / 2) / self.grid_step))
                for c, offset in zip(center, self.cell_size)
            ]
            monitor_half_width = int(round(size[0] / 2 / self.grid_step))
            monitor_slice = Slice(
                x=np.arange(
                    max(0, monitor_center[0] - monitor_half_width),
                    min(self.Nx, monitor_center[0] + monitor_half_width),
                ),
                y=np.array(monitor_center[1]),
            )
        else:
            raise ValueError(f"Direction {direction} not supported")
        # center of pixel's physical locations (um)
        xs = (-(self.Nx - 1) / 2 + monitor_slice.x) * self.grid_step
        ys = (-(self.Ny - 1) / 2 + monitor_slice.y) * self.grid_step
        self.port_monitor_slices[slice_name] = monitor_slice
        self.port_monitor_slices_info[slice_name] = dict( # please note that the radiation monitor info can only use the direction
            center=center,
            size=size,
            xs=xs,
            ys=ys,
            direction=direction,
        )

        return monitor_slice

    def build_port_monitor_slice(
        self,
        port_name: str = "in_port_1",
        slice_name: str = "in_port_1",
        rel_loc=0.2,
        rel_width=2,
        direction: str = None,
    ):
        port_cfg = self.port_cfgs[port_name]
        direction = port_cfg["direction"] if direction is None else direction
        center = port_cfg["center"]
        size = port_cfg["size"]

        if rel_width == float("inf"):
            if direction[0] == "x":
                rel_width = self.cell_size[1] / size[1]
            elif direction[0] == "y":
                rel_width = self.cell_size[0] / size[0]

        if direction[0] == "x":
            monitor_center = [
                center[0] - size[0] / 2 + rel_loc * size[0],
                center[1],
            ]
            monitor_size = [0, size[1] * rel_width]
        elif direction[0] == "y":
            monitor_center = [
                center[0],
                center[1] - size[1] / 2 + rel_loc * size[1],
            ]
            monitor_size = [size[0] * rel_width, 0]
        else:
            raise ValueError(f"Direction {direction} not supported")
        return self.add_monitor_slice(
            slice_name, monitor_center, monitor_size, direction
        )

    def build_farfield_region(
        self,
        region_name: str = "farfield",
        center: Tuple[float, float] = (3, 0),
        size: Tuple[float, float] = (1, 1),
        direction: str = "x+",
    ):
        ## extend the farfield from range[0] to range[1] um along the direction

        region_center = [
            int(round((c + offset / 2) / self.grid_step))
            for c, offset in zip(center, self.cell_size)
        ]
        half_width_x = int(round(size[0] / 2 / self.grid_step))
        half_width_y = int(round(size[1] / 2 / self.grid_step))
        xs = np.arange(region_center[0] - half_width_x, region_center[0] + half_width_x)
        ys = np.arange(
            region_center[1] - half_width_y,
            region_center[1] + half_width_y,
        )

        region = Slice(
            x=xs[:, None],
            y=ys[None, :],
        )

        # center of pixel's physical locations (um)
        xs = (-(self.Nx - 1) / 2 + region.x) * self.grid_step
        ys = (-(self.Ny - 1) / 2 + region.y) * self.grid_step
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        self.port_monitor_slices[region_name] = region
        self.port_monitor_slices_info[region_name] = dict(
            center=center,
            size=size,
            xs=xs,
            ys=ys,
            direction=direction,
        )

        return region

    def build_farfield_region_ext(
        self,
        region_name: str = "farfield",
        direction: str = "x+",
        extension_range: Tuple[float, float] = (3, 6),
    ):
        ## extend the farfield from range[0] to range[1] um along the direction
        if direction == "x":
            center = (sum(extension_range) / 2, 0)
            size = (
                extension_range[1] - extension_range[0],
                (self.Ny - 0.5) * self.grid_step,
            )
            region_center = [
                int(round((c + offset / 2) / self.grid_step))
                for c, offset in zip(center, self.cell_size)
            ]
            half_width_x = int(round(size[0] / 2 / self.grid_step))
            half_width_y = int(round(size[1] / 2 / self.grid_step))
            xs = np.arange(
                region_center[0] - half_width_x, region_center[0] + half_width_x
            )
            ys = np.arange(self.Ny)

        elif direction == "y":
            center = (0, sum(extension_range) / 2)
            size = (
                (self.Nx - 0.5) * self.grid_step,
                extension_range[1] - extension_range[0],
            )
            region_center = [
                int(round((c + offset / 2) / self.grid_step))
                for c, offset in zip(center, self.cell_size)
            ]

            half_width_x = int(round(size[0] / 2 / self.grid_step))
            half_width_y = int(round(size[1] / 2 / self.grid_step))
            xs = np.arange(self.Nx)
            ys = np.arange(
                region_center[1] - half_width_y,
                region_center[1] + half_width_y,
            )
        else:
            raise ValueError(f"Direction {direction} not supported")

        region = Slice(
            x=xs[:, None],
            y=ys[None, :],
        )

        # center of pixel's physical locations (um)
        xs = (-(self.Nx - 1) / 2 + region.x) * self.grid_step
        ys = (-(self.Ny - 1) / 2 + region.y) * self.grid_step
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        self.port_monitor_slices[region_name] = region
        self.port_monitor_slices_info[region_name] = dict(
            center=center,
            size=size,
            xs=xs,
            ys=ys,
            direction=direction,
        )

        return region

    def build_near2far_slice(
        self,
        slice_name: str = "nearfield_1",
        center: Tuple[float, float] = (0, 0),
        size: Tuple[float, float] = (0, 1),
        direction="x+",
    ):
        monitor_slice = self.add_monitor_slice(slice_name, center, size, direction)
        ## need to check the slice of eps is homogeneous medium
        eps_slice = self.epsilon_map[monitor_slice.x, monitor_slice.y]
        if not (np.unique(eps_slice).size == 1):
            print(
                f"Near2far slice {slice_name} is not in a homogeneous medium",
                flush=True,
            )
        return monitor_slice

    def build_radiation_monitor(
        self, monitor_name: str = "rad_slice", distance_to_PML=[0.2, 0.2]
    ):  
        '''
        Currently, the way to build the radiation monitor is through
        1. build a zeros_like epsilon map
        2. set the surrounding region of the epsilon map to 1
        3. set the ports region to 0 so that the monitor will not include the ports and the transmission will not be calculated as radiation
        so the radiation monitor is a 2D boolean array, not like other monitors which are the Slice object

        we need to make the monitor uniform, the radiation monitor should be a Slice object too
        '''
        xp_slice_name = monitor_name + "_xp"
        xp_center = (self.cell_size[0] / 2 - self.sim_cfg["PML"][0] - distance_to_PML[0], 0)
        monitor_size_x = [0, self.cell_size[1] - 2 * distance_to_PML[1] - 2 * self.sim_cfg["PML"][1]]
        radiation_monitor_xp = self.add_monitor_slice(
            xp_slice_name, xp_center, monitor_size_x, "x", 
        )
        xm_slice_name = monitor_name + "_xm"
        xm_center = (-self.cell_size[0] / 2 + self.sim_cfg["PML"][0] + distance_to_PML[0], 0)
        radiation_monitor_xm = self.add_monitor_slice(
            xm_slice_name, xm_center, monitor_size_x, "x",
        )
        yp_slice_name = monitor_name + "_yp"
        yp_center = (0, self.cell_size[1] / 2 - self.sim_cfg["PML"][1] - distance_to_PML[1])
        monitor_size_y = [self.cell_size[0] - 2 * distance_to_PML[0] - 2 * self.sim_cfg["PML"][0], 0]
        radiation_monitor_yp = self.add_monitor_slice(
            yp_slice_name, yp_center, monitor_size_y, "y",
        )
        ym_slice_name = monitor_name + "_ym"
        ym_center = (0, -self.cell_size[1] / 2 + self.sim_cfg["PML"][1] + distance_to_PML[1])
        radiation_monitor_ym = self.add_monitor_slice(
            ym_slice_name, ym_center, monitor_size_y, "y",
        )
        # quit()
        def exclude_ports(slice_obj):
            if slice_obj.x.size > 1:  # x is a range, y is a single value
                y_coord = int(slice_obj.y)  # Fixed y coordinate
                x_filtered = np.array([x for x in slice_obj.x if not self.ports_regions[x, y_coord]])
                y_filtered = slice_obj.y  # y remains unchanged
            elif slice_obj.y.size > 1:  # y is a range, x is a single value
                x_coord = int(slice_obj.x)  # Fixed x coordinate
                y_filtered = np.array([y for y in slice_obj.y if not self.ports_regions[x_coord, y]])
                x_filtered = slice_obj.x  # x remains unchanged
            else:
                raise ValueError("Both x and y are single values")

            return Slice(x=x_filtered, y=y_filtered)

        self.port_monitor_slices[xp_slice_name] = exclude_ports(self.port_monitor_slices[xp_slice_name])
        self.port_monitor_slices[xm_slice_name] = exclude_ports(self.port_monitor_slices[xm_slice_name])
        self.port_monitor_slices[yp_slice_name] = exclude_ports(self.port_monitor_slices[yp_slice_name])
        self.port_monitor_slices[ym_slice_name] = exclude_ports(self.port_monitor_slices[ym_slice_name])
        return (
            self.port_monitor_slices[xp_slice_name],
            self.port_monitor_slices[xm_slice_name],
            self.port_monitor_slices[yp_slice_name],
            self.port_monitor_slices[ym_slice_name],
        )

    def build_farfield_radiation_monitor(
        self, monitor_name: str = "farfield_rad_monitor"
    ):
        """
        for now, only xp_plus, xp_minus, yp and ym will be initialized
        """
        # self.port_monitor_slices[slice_name] = monitor_slice # index to refer in the np array or torch tensor
        # self.port_monitor_slices_info[slice_name] = dict(
        #     center=center, # coordinates of the center of the slice (um)
        #     size=size,     # size of the slice (um)
        #     xs=xs,         # x coordinates of the slice (um)
        #     ys=ys,         # y coordinates of the slice (um)
        #     direction=direction, # direction of the slice
        # )
        nearfield_vertices_x_coords = []
        nearfield_vertices_y_coords = []
        farfield_vertices_x_coords = []
        farfield_vertices_y_coords = []
        yp_info = {}
        ym_info = {}
        xp_plus_info = {}
        xp_minus_info = {}
        for key in list(self.port_monitor_slices_info.keys()):
            if "nearfield" in key:
                nearfield_vertices_x_coords.append(
                    self.port_monitor_slices_info[key]["xs"]
                )
                nearfield_vertices_y_coords.append(
                    self.port_monitor_slices_info[key]["ys"]
                )
            elif "farfield" in key:
                print(self.port_monitor_slices_info[key])
                farfield_vertices_x_coords.append(
                    self.port_monitor_slices_info[key]["xs"]
                )
                farfield_vertices_y_coords.append(
                    self.port_monitor_slices_info[key]["ys"]
                )

        def find_abs_max(input_list):
            max_abs_value = float("-inf")
            # Traverse the list
            for item in input_list:
                if isinstance(item, (int, float)):  # If it's a float or int
                    max_abs_value = max(max_abs_value, abs(item))
                elif isinstance(item, np.ndarray):  # If it's an array
                    max_abs_value = max(max_abs_value, np.max(np.abs(item)))
            return max_abs_value

        def find_max(input_list):
            max_value = float("-inf")
            # Traverse the list
            for item in input_list:
                if isinstance(item, (int, float)):
                    max_value = max(max_value, item)
                elif isinstance(item, np.ndarray):
                    max_value = max(max_value, np.max(item))
            return max_value

        nearfield_x_max = find_max(nearfield_vertices_x_coords) + 1
        nearfield_y_max_abs = find_abs_max(nearfield_vertices_y_coords)
        farfield_x_max = find_max(farfield_vertices_x_coords)
        farfield_y_max_abs = find_abs_max(farfield_vertices_y_coords)
        yp_info["center"] = [
            (nearfield_x_max + farfield_x_max) / 2,
            max(nearfield_y_max_abs, farfield_y_max_abs) + 1,
        ]
        yp_info["size"] = [farfield_x_max - nearfield_x_max, 0]
        yp_info["direction"] = "y"
        ym_info["center"] = [
            (nearfield_x_max + farfield_x_max) / 2,
            -max(nearfield_y_max_abs, farfield_y_max_abs) - 1,
        ]
        ym_info["size"] = [farfield_x_max - nearfield_x_max, 0]
        ym_info["direction"] = "y"
        yp_info["xs"] = (
            np.arange(
                int(
                    round(
                        (yp_info["center"][0] - yp_info["size"][0] / 2) / self.grid_step
                    )
                ),
                int(
                    round(
                        (yp_info["center"][0] + yp_info["size"][0] / 2) / self.grid_step
                    )
                ),
            )
            * self.grid_step
        )
        yp_info["ys"] = np.float32(yp_info["center"][1])
        ym_info["xs"] = (
            np.arange(
                int(
                    round(
                        (ym_info["center"][0] - ym_info["size"][0] / 2) / self.grid_step
                    )
                ),
                int(
                    round(
                        (ym_info["center"][0] + ym_info["size"][0] / 2) / self.grid_step
                    )
                ),
            )
            * self.grid_step
        )
        ym_info["ys"] = np.float32(ym_info["center"][1])

        xp_plus_info["center"] = [
            farfield_x_max,
            (farfield_y_max_abs + yp_info["center"][1]) / 2,
        ]
        xp_plus_info["size"] = [0, yp_info["center"][1] - farfield_y_max_abs]
        xp_plus_info["direction"] = "x"
        xp_minus_info["center"] = [
            nearfield_x_max,
            -(farfield_y_max_abs + yp_info["center"][1]) / 2,
        ]
        xp_minus_info["size"] = [0, -farfield_y_max_abs - ym_info["center"][1]]
        xp_minus_info["direction"] = "x"
        xp_plus_info["xs"] = np.float32(xp_plus_info["center"][0])
        xp_plus_info["ys"] = (
            np.arange(
                int(
                    round(
                        (xp_plus_info["center"][1] - xp_plus_info["size"][1] / 2)
                        / self.grid_step
                    )
                ),
                int(
                    round(
                        (xp_plus_info["center"][1] + xp_plus_info["size"][1] / 2)
                        / self.grid_step
                    )
                ),
            )
            * self.grid_step
        )
        xp_minus_info["xs"] = np.float32(xp_minus_info["center"][0])
        xp_minus_info["ys"] = (
            np.arange(
                int(
                    round(
                        (xp_minus_info["center"][1] - xp_minus_info["size"][1] / 2)
                        / self.grid_step
                    )
                ),
                int(
                    round(
                        (xp_minus_info["center"][1] + xp_minus_info["size"][1] / 2)
                        / self.grid_step
                    )
                ),
            )
            * self.grid_step
        )

        self.port_monitor_slices_info[monitor_name + "_yp"] = yp_info
        self.port_monitor_slices_info[monitor_name + "_ym"] = ym_info
        self.port_monitor_slices_info[monitor_name + "_xp_plus"] = xp_plus_info
        self.port_monitor_slices_info[monitor_name + "_xp_minus"] = xp_minus_info
        print(yp_info)
        print(ym_info)
        print(xp_plus_info)
        print(xp_minus_info)

    def insert_modes(
        self,
        eps,
        slice: Slice,
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        grid_step=None,
        power_scales: dict = None,
        source_modes: Tuple[int] = ("Ez1",),
    ):
        grid_step = grid_step or self.grid_step
        dl = grid_step * MICRON_UNIT
        mode_profiles = {}
        for wl in np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl):
            for source_mode in source_modes:
                # there is no need to calculate the modes for different temperatures
                # since the eps is only modulated at active region
                # current_eps = get_temp_related_eps(eps, wl, temp)
                omega = 2 * np.pi * C_0 / (wl * MICRON_UNIT)
             
                ht_m, et_m, _, mode = insert_mode(
                    omega, dl, slice.x, slice.y, eps, m=source_mode
                )
                # print(ht_m)
                # ht_m, et_m, _, mode = insert_mode_spins(
                #     omega, dl, slice.x, slice.y, eps, m=source_mode
                # )
                # print(ht_m)
                # exit(0)
                if power_scales is not None:
                    power_scale = power_scales[(wl, source_mode)]
                    ht_m = ht_m * power_scale
                    et_m = et_m * power_scale
                    mode = mode * power_scale
                else:
                    power_scale = 1
                mode_profiles[(wl, source_mode)] = [mode, ht_m, et_m, power_scale]
        return mode_profiles

    def insert_plane_wave(
        self,
        eps,
        slice: Slice,
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        source_modes: Tuple[str] = ("Ez1",),
        grid_step=None,
        power_scales: dict = None,
        direction: str = "x+",
        custom_source: np.ndarray | torch.Tensor = None,
    ):
        if isinstance(custom_source, torch.Tensor):
            lib = torch
            eps = torch.tensor(eps, dtype=torch.float32, device=custom_source.device)
        elif isinstance(custom_source, np.ndarray) or custom_source is None:
            lib = np
        else:
            raise ValueError("custom_source must be either np.ndarray or torch.Tensor")
        grid_step = grid_step or self.grid_step
        source_profiles = {}
        offset = -1 if direction[1] == "+" else 1
        for wl in lib.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl):
            for source_mode in source_modes:
                source = lib.zeros_like(eps, dtype=lib.complex64)
                if lib == torch:
                    source = source.to(custom_source.device)
                if direction[0] == "y":  # horizontal slice
                    source[:, slice.y] = 1 if custom_source is None else custom_source
                    if lib == torch:
                        source[:, slice.y + offset] = lib.exp(
                            torch.tensor([-1j * 2 * lib.pi / wl_cen * grid_step - 1j * lib.pi,], device=source.device)
                        ) * (1 if custom_source is None else custom_source)
                    else:
                        source[:, slice.y + offset] = lib.exp(
                            -1j * 2 * lib.pi / wl_cen * grid_step - 1j * lib.pi
                        ) * (1 if custom_source is None else custom_source)
                elif direction[0] == "x":  # vertical slice
                    source[slice.x, :] = 1 if custom_source is None else custom_source
                    if lib == torch:
                        source[slice.x + offset, :] = lib.exp(
                            torch.tensor([-1j * 2 * lib.pi / wl_cen * grid_step - 1j * lib.pi,], device=source.device)
                        ) * (1 if custom_source is None else custom_source)
                    else:
                        source[slice.x + offset, :] = lib.exp(
                            -1j * 2 * lib.pi / wl_cen * grid_step - 1j * lib.pi
                        ) * (1 if custom_source is None else custom_source)

                ht_m = et_m = source.reshape(-1)
                if power_scales is not None:
                    power_scale = power_scales[
                        (wl, source_mode)
                    ]  # use direction as a placeholder for mode
                    ht_m = et_m = et_m * power_scale
                    source = source * power_scale
                else:
                    power_scale = 1
                if isinstance(wl, torch.Tensor):
                    wl = round(wl.item(), 2)
                source_profiles[(wl, source_mode)] = [source, ht_m, et_m, power_scale]
        return source_profiles

    def create_simulation(
        self, omega, dl, eps, NPML, solver="ceviche", pol: str = "Ez"
    ):
        if solver == "ceviche":
            if pol == "Ez":
                return fdfd_ez(omega, dl, eps, NPML)
            elif pol == "Hz":
                return fdfd_hz(omega, dl, eps, NPML)
            else:
                raise ValueError(f"Pol {pol} not supported")
        elif solver == "ceviche_torch":
            if pol == "Ez":
                fdfd_fn = fdfd_ez_torch
            elif pol == "Hz":
                fdfd_fn = fdfd_hz_torch
            return fdfd_fn(
                omega,
                dl,
                eps,
                NPML,
                neural_solver=self.sim_cfg.get("neural_solver", None),
                numerical_solver=self.sim_cfg.get("numerical_solver", "solve_direct"),
                use_autodiff=self.sim_cfg.get("use_autodiff", False),
            )
        else:
            raise ValueError(f"Solver {solver} not supported")

    def solve_ceviche(
        self,
        eps,
        source,
        wl: float = 1.55,
        grid_step=None,
        solver: str = "ceviche",
        pol: str = "Ez",
    ):
        """
        _summary_

        this is only called in the norm run through solve() in _norm_run(), so we can pass port_name and the mode to be 'Norm' directly
        and there is no need to run the backward to store the adjoint source and adjoint fields, so we enable torch.no_grad() environment
        """
        omega = 2 * np.pi * C_0 / (wl * MICRON_UNIT)
        grid_step = grid_step or self.grid_step
        dl = grid_step * MICRON_UNIT
        # simulation = fdfd_ez(omega, dl, eps, [self.NPML[0], self.NPML[1]])
        simulation = self.create_simulation(
            omega, dl, eps, self.NPML, solver=solver, pol=pol
        )

        if hasattr(simulation, "solver"):  # which means that it is a torch simulation
            with torch.no_grad():
                Fx, Fy, Fz = simulation.solve(source, slice_name="Norm", mode="Norm", temp="Norm")
        else:
            Fx, Fy, Fz = simulation.solve(source)

        if pol == "Ez":
            return {"Hx": Fx, "Hy": Fy, "Ez": Fz}
        elif pol == "Hz":
            return {"Ex": Fx, "Ey": Fy, "Hz": Fz}
        else:
            raise ValueError(f"Unknown simulation {type(simulation)} type")

    def solve(
        self,
        eps,
        source_profiles,
        solver="ceviche",
        grid_step=None,
    ):
        """_summary_

        Args:
            eps (_type_): _description_
            source_profiles (_type_): _description_
            solver (str, optional): _description_. Defaults to "ceviche".
            grid_step (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            fields: {(wl, mode): {"Hx": Hx, "Hy": Hy, "Ez": Ez}, ...}
        """
        grid_step = grid_step or self.grid_step
        fields = {}
        if solver in {"ceviche", "ceviche_torch"}:
            for (wl, mode), (source, _, _, _) in source_profiles.items():
                # current_eps = get_temp_related_eps(eps, wl, temp)
                current_eps = eps
                pol = mode[:2]  # "Ez1" -> "Ez"
                field_sol = self.solve_ceviche(
                    current_eps,
                    source,
                    wl=wl,
                    grid_step=grid_step,
                    solver=solver,
                    pol=pol,
                )
                fields[(wl, mode)] = field_sol
            return fields
        else:
            raise ValueError(f"Solver {solver} not supported")

    @lru_cache(maxsize=128)
    def build_norm_sources(
        self,
        source_modes: Tuple[str] = ("Ez1",),
        input_port_name: str = "in_port_1",
        input_slice_name: str = "in_slice_1",
        wl_cen=1.55,
        wl_width=0,
        n_wl=1,
        solver="ceviche",
        power: float = 1e-8,
        source_type: str = "mode",
        plot=False,
        require_sim: bool = False,
    ):
        assert source_type in {
            "mode",
            "plane_wave",
        }, f"Source type {source_type} not supported"

        input_slice = self.port_monitor_slices[input_slice_name]
        in_port = get_two_ports(self, port_name=input_port_name)
        in_port_eps = in_port.epsilon_map
        direction = self.port_monitor_slices_info[input_slice_name]["direction"]

        if direction[0] == "x":
            output_slice = Slice(x=self.Nx - input_slice.x, y=input_slice.y)
        elif direction[0] == "y":
            output_slice = Slice(x=input_slice.x, y=self.Ny - input_slice.y)

        def _norm_run(power_scales=None):
            if source_type == "mode":
                source_profiles = self.insert_modes(
                    in_port_eps,
                    input_slice,
                    wl_cen=wl_cen,
                    wl_width=wl_width,
                    n_wl=n_wl,
                    power_scales=power_scales,
                    source_modes=source_modes,
                )  # {(wl, mode): [source, ht_m, et_m, scale], ...}
                # print_stat(source_profiles[(1.55, 1)][0])
                # monitor_profiles = self.insert_modes(
                #     in_port_eps,
                #     output_slice,
                #     wl_cen=wl_cen,
                #     wl_width=wl_width,
                #     n_wl=n_wl,
                #     temp=temp,
                #     power_scales=power_scales,
                #     source_modes=source_modes,
                # )  # {(wl, mode): [monitor, ht_m, et_m, scale], ...}
            elif source_type == "plane_wave":
                source_profiles = self.insert_plane_wave(
                    in_port_eps,
                    input_slice,
                    wl_cen=wl_cen,
                    wl_width=wl_width,
                    n_wl=n_wl,
                    source_modes=source_modes,
                    power_scales=power_scales,
                    direction=direction,
                )

            # print_stat(monitor_profiles[(1.55, 1)][0])
            fields = self.solve(
                in_port_eps, source_profiles, solver=solver
            )  # [(wl, mode, Hx), ...], [(wl, mode, Hy), ...], [(wl, mode, Ez), ...]
            # print_stat(fields[(1.55, 1)]["Ez"])

            input_SCALE = {}
            for k in source_profiles:
                mode = k[1]
                pol = mode[:2]
                if pol == "Ez":
                    Fx, Fy, Fz = fields[k]["Hx"], fields[k]["Hy"], fields[k]["Ez"]
                elif pol == "Hz":
                    Fx, Fy, Fz = fields[k]["Ex"], fields[k]["Ey"], fields[k]["Hz"]
                else:
                    raise ValueError(f"Unknown polarization {pol}")

                # _, ht_m, et_m, _ = source_profiles[k]
                # print("this is the type of Hx:", type(Hx), flush=True)
                # print("this is the type of Hy:", type(Hy), flush=True)
                # print("this is the type of Ez:", type(Ez), flush=True)
                # print("this is the type of ht_m:", type(ht_m), flush=True)
                # print("this is the type of et_m:", type(et_m), flush=True)
                # ht_m = torch.from_numpy(ht_m).to(Ez.device)
                # et_m = torch.from_numpy(et_m).to(Ez.device)
                # eigen_energy = get_eigenmode_coefficients(
                #     Fx,
                #     Fy,
                #     Fz,
                #     ht_m,
                #     et_m,
                #     output_slice,
                #     grid_step=self.grid_step,
                #     direction=direction,
                #     energy=True,
                #     pol=pol,
                # )
                # print("eigen_energy:", eigen_energy)
                ## used to verify eigen mode coefficients, need to be the same as eigen energy
                flux = get_flux(
                    Fx,
                    Fy,
                    Fz,
                    output_slice,
                    grid_step=self.grid_step,
                    direction=direction,
                    pol=pol,
                )
                # print("norm flux:", flux)
                if isinstance(flux, torch.Tensor):
                    flux = flux.item()
                input_SCALE[k] = np.abs(flux)

            return input_SCALE, fields, source_profiles

        input_scale, fields, source_profiles = _norm_run()  # to get eigen energy
        input_scale = {
            k: (power / v) ** 0.5 for k, v in input_scale.items()
        }  # normalize the source power to target power for all wavelengths and modes

        pol = list(fields.keys())[0][1][:2]  # [wl, mode] get pol from mode
        Fz = list(fields.values())[0][pol]
        if isinstance(Fz, torch.Tensor):
            source_profiles = {
                k: [torch.from_numpy(i).to(Fz.device) for i in v[:-1]] + [v[-1]]
                for k, v in source_profiles.items()
            }
        source_profiles = {
            k: [e * input_scale[k] for e in v[:-1]] + [power] + [require_sim]
            for k, v in source_profiles.items()
        }
        # source_profiles["require_sim"] = require_sim
        # input_SCALE, fields, source_profiles = _norm_run(power_scales=input_scale)

        if plot:
            plot_eps_field(
                Fz * list(input_scale.values())[0],
                in_port_eps,
                zoom_eps_factor=1,
                filepath=os.path.join(
                    self.sim_cfg["plot_root"],
                    f"{self.config.device.type}_norm-{input_slice_name}.png",
                ),
                x_width=self.cell_size[0],
                y_height=self.cell_size[1],
                monitors=[(input_slice, "r"), (output_slice, "b")],
                title=f"|{pol}|^2, Norm run at {input_slice_name}",
                field_stat="intensity_real",
            )
        if self.port_sources_dict.get(input_slice_name) is not None:
            self.port_sources_dict[input_slice_name].update(source_profiles)
        else:
            self.port_sources_dict[input_slice_name] = source_profiles
        # print(source_profiles)
        # exit(0)
        return source_profiles  # {(wl, mode): [profile, ht_m, et_m, SCALE, require_sim], ...}

    def obtain_eps(self, permittivity: torch.Tensor):
        ## we need denormalized permittivity for the design region
        permittivity = permittivity.detach().cpu().numpy()
        eps_map = copy.deepcopy(self.epsilon_map)
        eps_map[self.design_region_mask] = permittivity.flatten()
        return eps_map  # return the copy of the permittivity map

    def copy(self, resolution: int = 310):
        sim_cfg = copy.deepcopy(self.sim_cfg)
        sim_cfg["resolution"] = resolution
        new_device = self.__class__()
        super(new_device.__class__, new_device).__init__(
            eps_bg=self.eps_bg,
            port_cfgs=self.port_cfgs,
            geometry_cfgs=self.geometry_cfgs,
            design_region_cfgs=self.design_region_cfgs,
            sim_cfg=sim_cfg,
            device=self.device,
        )
        return new_device

    def __str__(self):
        return f"{self.__class__.__name__}(size={self.cell_size}, Nx={self.Nx}, Ny={self.Ny})"
