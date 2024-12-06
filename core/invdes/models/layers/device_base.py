"""
Date: 2024-10-02 20:59:04
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-02 21:37:00
FilePath: /Metasurface-Opt/core/models/layers/device_base.py
"""

import copy
import math
import os
from functools import lru_cache
from typing import Tuple

import meep as mp
import numpy as np
import torch

from thirdparty.ceviche.ceviche import fdfd_ez
from core.fdfd import fdfd_ez as fdfd_ez_torch

# from ceviche.modes import insert_mode
from ceviche.constants import C_0
from pyutils.config import Config
from pyutils.general import ensure_dir

from .utils import (
    apply_regions_gpu,
    get_eigenmode_coefficients,
    get_flux,
    get_grid,
    insert_mode,
    plot_eps_field,
)
from core.utils import (
    Si_eps,
    SiO2_eps,
    Slice,
)

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
        self.Nx, self.Ny, self.Nz = [
            int(round(i * self.resolution)) for i in self.cell_size
        ] # change math.ceil to round since sometimes we will have like 10.200000000000001 in the cell_size which will cause a size mismatch
        self.NPML = [int(round(i * self.resolution)) for i in sim_cfg["PML"]]
        self.xs, self.ys = get_grid((self.Nx, self.Ny), self.grid_step)
        self.epsilon_map = self.get_epsilon_map(
            self.cell_size,
            self.geometry,
            sim_cfg["PML"],
            self.resolution,
            self.eps_bg,
        )
        self.design_region_masks = self.build_design_region_mask(design_region_cfgs)
        self.ports_regions = self.build_port_region(port_cfgs)

        self.port_monitor_slices = {}  # {port_name: Slice or mask}
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
            left = int(np.round(left / self.grid_step))
            right = int(np.round(right / self.grid_step))
            lower = int(np.round(lower / self.grid_step))
            upper = int(np.round(upper / self.grid_step))
            region = Slice(
                x=slice(left, right + 1), y=slice(lower, upper + 1)
            )  # a rectangular region
            design_region_masks[name] = region

        return design_region_masks

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
        assert size[0] == 1 or size[1] == 1, "Only 1D slice is supported"
        if direction is None:
            direction = "x" if size[0] == 1 else "y"

        if direction == "x":
            monitor_center = [int(round(c / self.grid_step)) for c in center]
            monitor_half_width = int(round(size[1] / 2 / self.grid_step))
            monitor_slice = Slice(
                x=np.array(monitor_center[0]),
                y=np.arange(
                    monitor_center[1] - monitor_half_width,
                    monitor_center[1] + monitor_half_width,
                ),
            )
        elif direction == "y":
            monitor_center = [int(round(c / self.grid_step)) for c in center]
            monitor_half_width = int(round(size[0] / 2 / self.grid_step))
            monitor_slice = Slice(
                x=np.arange(
                    monitor_center[0] - monitor_half_width,
                    monitor_center[0] + monitor_half_width,
                ),
                y=np.array(monitor_center[1]),
            )
        else:
            raise ValueError(f"Direction {direction} not supported")
        self.port_monitor_slices[slice_name] = monitor_slice
        return monitor_slice

    def build_port_monitor_slice(
        self,
        port_name: str = "in_port_1",
        slice_name: str = "in_port_1",
        rel_loc=0.2,
        rel_width=2,
    ):
        port_cfg = self.port_cfgs[port_name]
        direction = port_cfg["direction"]
        center = port_cfg["center"]
        size = port_cfg["size"]
        if direction == "x":
            monitor_center = [
                center[0] - size[0] / 2 + rel_loc * size[0] + self.cell_size[0] / 2,
                center[1] + self.cell_size[1] / 2,
            ]
            monitor_size = [1, size[1] * rel_width]
        elif direction == "y":
            monitor_center = [
                center[0] + self.cell_size[0] / 2,
                center[1] - size[1] / 2 + rel_loc * size[1] + self.cell_size[1] / 2,
            ]
            monitor_size = [size[0] * rel_width, 1]
        else:
            raise ValueError(f"Direction {direction} not supported")
        return self.add_monitor_slice(
            slice_name, monitor_center, monitor_size, direction
        )

    def build_radiation_monitor(
        self, monitor_name: str = "rad_monitor", distance_to_PML=[0.2, 0.2]
    ):
        radiation_monitor_xp = np.zeros_like(self.epsilon_map, dtype=np.bool_)
        radiation_monitor_xm = np.zeros_like(self.epsilon_map, dtype=np.bool_)
        radiation_monitor_yp = np.zeros_like(self.epsilon_map, dtype=np.bool_)
        radiation_monitor_ym = np.zeros_like(self.epsilon_map, dtype=np.bool_)
        distance_PML = [int(round(i / self.grid_step)) for i in distance_to_PML]
        left = self.NPML[0] + distance_PML[0]
        right = self.Nx - self.NPML[0] - distance_PML[0]
        lower = self.NPML[1] + distance_PML[1]
        upper = self.Ny - self.NPML[1] - distance_PML[1]
        radiation_monitor_xm[left : left + 1, lower:upper] = 1
        radiation_monitor_xp[right : right + 1, lower:upper] = 1
        radiation_monitor_ym[left:right, lower : lower + 1] = 1
        radiation_monitor_yp[left:right, upper : upper + 1] = 1
        radiation_monitor_xp[self.ports_regions] = 0
        radiation_monitor_xm[self.ports_regions] = 0
        radiation_monitor_yp[self.ports_regions] = 0
        radiation_monitor_ym[self.ports_regions] = 0
        self.port_monitor_slices[monitor_name + "_xp"] = radiation_monitor_xp
        self.port_monitor_slices[monitor_name + "_xm"] = radiation_monitor_xm
        self.port_monitor_slices[monitor_name + "_yp"] = radiation_monitor_yp
        self.port_monitor_slices[monitor_name + "_ym"] = radiation_monitor_ym

        return (
            radiation_monitor_xp,
            radiation_monitor_xm,
            radiation_monitor_yp,
            radiation_monitor_ym,
        )

    def insert_modes(
        self,
        eps,
        slice: Slice,
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        grid_step=None,
        power_scales: dict = None,
        source_modes: Tuple[int] = (1,),
    ):
        grid_step = grid_step or self.grid_step
        dl = grid_step * 1e-6
        mode_profiles = {}
        for wl in np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl):
            for source_mode in source_modes:
                omega = 2 * np.pi * C_0 / (wl * 1e-6)
                ht_m, et_m, _, mode = insert_mode(
                    omega, dl, slice.x, slice.y, eps, m=source_mode
                )
                if power_scales is not None:
                    power_scale = power_scales[(wl, source_mode)]
                    ht_m = ht_m * power_scale
                    et_m = et_m * power_scale
                    mode = mode * power_scale
                else:
                    power_scale = 1
                mode_profiles[(wl, source_mode)] = [mode, ht_m, et_m, power_scale]
        return mode_profiles

    def create_simulation(self, omega, dl, eps, NPML, solver="ceviche"):
        if solver == "ceviche":
            return fdfd_ez(omega, dl, eps, NPML)
        elif solver == "ceviche_torch":
            return fdfd_ez_torch(
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

    def solve_ceviche(self, eps, source, wl: float = 1.55, grid_step=None, solver: str="ceviche"):
        """
        _summary_
        
        this is only called in the norm run through solve() in _norm_run(), so we can pass port_name and the mode to be 'Norm' directly
        and there is no need to run the backward to store the adjoint source and adjoint fields, so we enable torch.no_grad() environment
        """
        omega = 2 * np.pi * C_0 / (wl * 1e-6)
        grid_step = grid_step or self.grid_step
        dl = grid_step * 1e-6
        # simulation = fdfd_ez(omega, dl, eps, [self.NPML[0], self.NPML[1]])
        simulation = self.create_simulation(omega, dl, eps, self.NPML, solver=solver)
        if hasattr(simulation, "solver"): # which means that it is a torch simulation
            with torch.no_grad():
                Hx, Hy, Ez = simulation.solve(source, port_name="Norm", mode="Norm")
        else:
            Hx, Hy, Ez = simulation.solve(source)

        return Hx, Hy, Ez

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
                Hx, Hy, Ez = self.solve_ceviche(eps, source, wl=wl, grid_step=grid_step, solver=solver)
                fields[(wl, mode)] = {"Hx": Hx, "Hy": Hy, "Ez": Ez}
            return fields
        else:
            raise ValueError(f"Solver {solver} not supported")

    @lru_cache(maxsize=128)
    def build_norm_sources(
        self,
        source_modes: Tuple[int] = (1,),
        input_port_name: str = "in_port_1",
        input_slice_name: str = "in_port_1",
        wl_cen=1.55,
        wl_width=0,
        n_wl=1,
        solver="ceviche",
        power: float = 1e-8,
        plot=False,
    ):
        input_slice = self.port_monitor_slices[input_slice_name]
        in_port = get_two_ports(self, port_name=input_port_name)
        in_port_eps = in_port.epsilon_map
        direction = in_port.port_cfgs[input_port_name]["direction"]

        if direction[0] == "x":
            output_slice = Slice(x=self.Nx - input_slice.x, y=input_slice.y)
        elif direction[0] == "y":
            output_slice = Slice(x=input_slice.x, y=self.Ny - input_slice.y)

        def _norm_run(power_scales=None):
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
            monitor_profiles = self.insert_modes(
                in_port_eps,
                output_slice,
                wl_cen=wl_cen,
                wl_width=wl_width,
                n_wl=n_wl,
                power_scales=power_scales,
                source_modes=source_modes,
            )  # {(wl, mode): [monitor, ht_m, et_m, scale], ...}
            # print_stat(monitor_profiles[(1.55, 1)][0])
            fields = self.solve(
                in_port_eps, source_profiles, solver=solver
            )  # [(wl, mode, Hx), ...], [(wl, mode, Hy), ...], [(wl, mode, Ez), ...]
            # print_stat(fields[(1.55, 1)]["Ez"])

            input_SCALE = {}
            for k in monitor_profiles:
                Hx, Hy, Ez = fields[k]["Hx"], fields[k]["Hy"], fields[k]["Ez"]
                # _, ht_m, et_m, _ = monitor_profiles[k]
                # print("this is the type of Hx:", type(Hx), flush=True)
                # print("this is the type of Hy:", type(Hy), flush=True)
                # print("this is the type of Ez:", type(Ez), flush=True)
                # print("this is the type of ht_m:", type(ht_m), flush=True)
                # print("this is the type of et_m:", type(et_m), flush=True)
                # ht_m = torch.from_numpy(ht_m).to(Ez.device)
                # et_m = torch.from_numpy(et_m).to(Ez.device)
                # eigen_energy = get_eigenmode_coefficients(
                #     Hx,
                #     Hy,
                #     Ez,
                #     ht_m,
                #     et_m,
                #     output_slice,
                #     grid_step=self.grid_step,
                #     direction=direction,
                #     energy=True,
                # )
                # print("eigen_energy:", eigen_energy)
                ## used to verify eigen mode coefficients, need to be the same as eigen energy
                flux = get_flux(
                    Hx,
                    Hy,
                    Ez,
                    output_slice,
                    grid_step=self.grid_step,
                    direction=direction,
                )
                # print("flux:", flux)
                if isinstance(flux, torch.Tensor):
                    flux = flux.item()
                input_SCALE[k] = np.abs(flux)

            return input_SCALE, fields, source_profiles

        input_scale, fields, source_profiles = _norm_run()  # to get eigen energy
        input_scale = {
            k: (power / v) ** 0.5 for k, v in input_scale.items()
        }  # normalize the source power to target power for all wavelengths and modes

        Ez = list(fields.values())[0]["Ez"]
        if isinstance(Ez, torch.Tensor):
            source_profiles = {
                k: [torch.from_numpy(i).to(Ez.device) for i in v[:-1]] + [v[-1]]
                for k, v in source_profiles.items()
            }
        source_profiles = {
            k: [e * input_scale[k] for e in v[:-1]] + [power]
            for k, v in source_profiles.items()
        }
        # input_SCALE, fields, source_profiles = _norm_run(power_scales=input_scale)

        if plot:
            plot_eps_field(
                Ez * list(input_scale.values())[0],
                in_port_eps,
                zoom_eps_factor=1,
                filepath=os.path.join(
                    self.sim_cfg["plot_root"],
                    f"{self.config.device.type}_norm-{input_slice_name}.png",
                ),
                x_width=self.cell_size[0],
                y_height=self.cell_size[1],
                monitors=[(input_slice, "r"), (output_slice, "b")],
                title=f"|Ez|^2, Norm run at {input_slice_name}",
            )

        self.port_sources_dict[input_slice_name] = source_profiles
        # print(source_profiles)
        # exit(0)
        return source_profiles  # {(wl, mode): [profile, ht_m, et_m, SCALE], ...}

    def obtain_eps(self, permittivity: torch.Tensor):
        ## we need denormalized permittivity for the design region
        permittivity = permittivity.detach().cpu().numpy()
        eps_map = copy.deepcopy(self.epsilon_map)
        eps_map[self.design_region_mask] = permittivity.flatten()
        return eps_map  # return the copy of the permittivity map

    def copy(self, resolution: int = 310):
        sim_cfg = copy.deepcopy(self.sim_cfg)
        print("finish deep copying...", flush=True)
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
