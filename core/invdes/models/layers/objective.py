import copy
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from autograd import numpy as npa
from torch import Tensor

from core.fdfd.near2far import (
    get_farfields_GreenFunction,
)
from core.utils import (
    Si_eps,
    get_eigenmode_coefficients,
    get_flux,
    get_shape_similarity,
)
from thirdparty.ceviche.ceviche import jacobian
from thirdparty.ceviche.ceviche.constants import MU_0


class EigenmodeObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        s_params: dict,
        port_profiles: dict,  # port monitor profiles {port_name: {(wl, mode): (profile, ht_m, et_m)}}
        port_slices: dict,
        in_port_name: str,
        out_port_name: str,
        in_mode: int,
        out_modes: Tuple[int],
        direction: str,
        name: str,
        target_wls: Tuple[float],
        target_temps: Tuple[float],
        grid_step: float,
        energy: bool = True,
        obj_type: str = "eigenmode",
    ):
        self.sims = sims
        self.s_params = s_params
        self.port_profiles = port_profiles
        self.port_slices = port_slices
        self.in_port_name = in_port_name
        self.out_port_name = out_port_name
        self.in_mode = in_mode
        self.out_modes = out_modes
        self.direction = direction
        self.name = name
        self.target_wls = target_wls
        self.target_temps = target_temps
        self.grid_step = grid_step
        self.energy = energy
        self.obj_type = obj_type

    def __call__(self, fields):
        s_list = []
        (
            target_wls,
            target_temps,
            in_port_name,
            out_port_name,
            in_mode,
            out_modes,
            direction,
            name,
            grid_step,
        ) = (
            self.target_wls,
            self.target_temps,
            self.in_port_name,
            self.out_port_name,
            self.in_mode,
            self.out_modes,
            self.direction,
            self.name,
            self.grid_step,
        )
        ## for each wavelength, we evaluate the objective
        for (wl, pol), sim in self.sims.items():
            ## we calculate the average eigen energy for all output modes
            if wl not in target_wls:
                continue
            if pol != in_mode[:2]:
                continue
            for out_mode in out_modes:
                for temp in target_temps:
                    src, ht_m, et_m, norm_p = self.port_profiles[out_port_name][
                        (wl, out_mode)
                    ]
                    norm_power = self.port_profiles[in_port_name][(wl, in_mode)][3]
                    monitor_slice = self.port_slices[out_port_name]
                    field = fields[(in_port_name, wl, in_mode, temp)]
                    pol = in_mode[:2]
                    if pol == "Ez":
                        fx, fy, fz = (
                            field["Hx"],
                            field["Hy"],
                            field["Ez"],
                        )  # fetch fields
                    elif pol == "Hz":
                        fx, fy, fz = (
                            field["Ex"],
                            field["Ey"],
                            field["Hz"],
                        )
                    if isinstance(ht_m, Tensor) and ht_m.device != fz.device:
                        ht_m = ht_m.to(fz.device)
                        et_m = et_m.to(fz.device)
                        self.port_profiles[out_port_name][(wl, out_mode)] = [
                            src.to(fz.device),
                            ht_m,
                            et_m,
                            norm_p,
                        ]
                    s_p, s_m = get_eigenmode_coefficients(
                        fx,
                        fy,
                        fz,
                        ht_m,
                        et_m,
                        monitor_slice,
                        grid_step=grid_step,
                        direction=direction[0],
                        autograd=True,
                        energy=self.energy,
                        pol=pol,
                    )
                    if direction[1] == "+":
                        s = s_p
                    elif direction[1] == "-":
                        s = s_m
                    else:
                        raise ValueError("Invalid direction")
                    # print(s, norm_power)
                    if self.energy:
                        s_list.append(s / norm_power)
                    else:
                        s_list.append(s / norm_power**0.5)
                    if self.obj_type == "eigenmode":
                        # only record the s parameters for eigenmode
                        # we don't need to record the s parameters if we calculate the phase
                        self.s_params[(self.out_port_name, wl, out_mode, temp)] = {
                            "s_p": s_p / norm_power
                            if self.energy
                            else s_p / norm_power**0.5,  # normalized by input power
                            "s_m": s_m / norm_power
                            if self.energy
                            else s_m / norm_power**0.5,  # normalized by input power
                        }
        if isinstance(s_list[0], Tensor):
            return torch.mean(torch.stack(s_list))
        else:
            return npa.mean(npa.array(s_list))


class FluxNear2FarObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        s_params: dict,
        port_profiles: dict,  # port monitor profiles {port_name: {(wl, mode): (profile, ht_m, et_m)}}
        port_slices: dict,
        port_slices_info: dict,
        in_port_name: str,
        out_port_name: str,
        in_mode: int,
        direction: str,
        name: str,
        target_temps: Tuple[float],
        grid_step: float,
        eps_bg: float,
        obj_type: str = "flux_near2far",
        total_farfield_region_solutions: dict = None,
    ):
        self.sims = sims
        self.s_params = s_params
        self.port_profiles = port_profiles
        self.port_slices = port_slices
        self.port_slices_info = port_slices_info
        self.in_port_name = in_port_name
        self.out_port_name = out_port_name
        self.in_mode = in_mode
        self.direction = direction
        self.name = name
        self.target_temps = target_temps
        self.grid_step = grid_step
        self.eps_bg = eps_bg
        self.obj_type = obj_type
        self.total_farfield_region_solutions = total_farfield_region_solutions

    def __call__(self, fields):
        s_list = []
        (
            target_temps,
            in_port_name,
            out_port_name,
            in_mode,
            direction,
            name,
            grid_step,
        ) = (
            self.target_temps,
            self.in_port_name,
            self.out_port_name,
            self.in_mode,
            self.direction,
            self.name,
            self.grid_step,
        )

        s_list = []
        ## for each wavelength, we evaluate the objective
        for (wl, pol), _ in self.sims.items():
            if pol != in_mode[:2]:
                continue
            for temp in target_temps:
                # monitor_slice = self.port_slices[out_port_name]
                norm_power = self.port_profiles[in_port_name][(wl, in_mode)][3]
                # this is how ez, hx and hy are calculated in regular simulation
                field = fields[(in_port_name, wl, in_mode, temp)]
                if pol == "Ez":
                    fx_near, fy_near, fz_near = (
                        field["Hx"],
                        field["Hy"],
                        field["Ez"],
                    )  # fetch fields
                elif pol == "Hz":
                    fx_near, fy_near, fz_near = (
                        field["Ex"],
                        field["Ey"],
                        field["Hz"],
                    )
                # print("this is the keys of the self.port_slices_info", list(self.port_slices.keys()))
                extended_farfield_slice_info = copy.deepcopy(
                    self.port_slices_info[out_port_name]
                )
                ## Ez, extend toward negative dir, Hz extend toward positive dir
                if direction[0] == "x":
                    xs = extended_farfield_slice_info["xs"]
                    if not xs.shape:
                        extended_farfield_slice_info["xs"] = np.array(
                            [xs - grid_step, xs] if pol == "Ez" else [xs, xs + grid_step]
                        )
                    else:
                        extended_farfield_slice_info["xs"] = np.concatenate(
                            [xs[0:1] - grid_step, xs] if pol == "Ez" else [xs, xs[-1:] + grid_step], axis=0
                        )
                elif direction[0] == "y":
                    ys = extended_farfield_slice_info["ys"]
                    if not ys.shape:
                        extended_farfield_slice_info["ys"] = np.array(
                            [ys - grid_step, ys] if pol == "Ez" else [ys, ys + grid_step]
                        )
                    else:
                        extended_farfield_slice_info["ys"] = np.concatenate(
                            [ys[0:1] - grid_step, ys] if pol == "Ez" else [ys, ys[-1:] + grid_step], axis=0
                        )
                if out_port_name == "total_farfield_region":
                    with torch.inference_mode():
                        farfield = get_farfields_GreenFunction(
                            nearfield_slices=[
                                self.port_slices[nearfield_slice_name]
                                for nearfield_slice_name in list(
                                    self.port_slices.keys()
                                )
                                if nearfield_slice_name.startswith("nearfield")
                            ],
                            nearfield_slices_info=[
                                self.port_slices_info[nearfield_slice_name]
                                for nearfield_slice_name in list(
                                    self.port_slices_info.keys()
                                )
                                if nearfield_slice_name.startswith("nearfield")
                            ],
                            Fz=fz_near[None, ..., None],
                            Fx=fx_near[None, ..., None],
                            Fy=fy_near[None, ..., None],
                            farfield_x=None,
                            farfield_slice_info=self.port_slices_info[out_port_name],
                            freqs=torch.tensor([1 / wl], device=fz_near.device),
                            eps=self.eps_bg,
                            mu=MU_0,
                            dL=self.grid_step,
                            component=pol,
                            decimation_factor=12,
                        )
                    if pol == "Ez":
                        fz = farfield["Ez"][0, ..., 0]
                        fx = farfield["Hx"][0, ..., 0]
                        fy = farfield["Hy"][0, ..., 0]
                        self.total_farfield_region_solutions[
                            (in_port_name, wl, in_mode, temp)
                        ] = {
                            "Ez": fz,
                            "Hx": fx,
                            "Hy": fy,
                        }
                    elif pol == "Hz":
                        fz = farfield["Hz"][0, ..., 0]
                        fx = farfield["Ex"][0, ..., 0]
                        fy = farfield["Ey"][0, ..., 0]
                        self.total_farfield_region_solutions[
                            (in_port_name, wl, in_mode, temp)
                        ] = {
                            "Hz": fz,
                            "Ex": fx,
                            "Ey": fy,
                        }
                    return torch.tensor(0.0).to(fz.device)
                else:
                    farfield = get_farfields_GreenFunction(
                        nearfield_slices=[
                            self.port_slices[nearfield_slice_name]
                            for nearfield_slice_name in list(self.port_slices.keys())
                            if nearfield_slice_name.startswith("nearfield")
                        ],
                        nearfield_slices_info=[
                            self.port_slices_info[nearfield_slice_name]
                            for nearfield_slice_name in list(
                                self.port_slices_info.keys()
                            )
                            if nearfield_slice_name.startswith("nearfield")
                        ],
                        Fz=fz_near[None, ..., None],
                        Fx=fx_near[None, ..., None],
                        Fy=fy_near[None, ..., None],
                        farfield_x=None,
                        farfield_slice_info=self.port_slices_info[out_port_name],
                        freqs=torch.tensor([1 / wl], device=fz_near.device),
                        eps=self.eps_bg,
                        mu=MU_0,
                        dL=self.grid_step,
                        component=pol,
                        decimation_factor=4,
                    )
                    if pol == "Ez":
                        fz = farfield["Ez"][0, ..., 0]
                        fx = farfield["Hx"][0, ..., 0]
                        fy = farfield["Hy"][0, ..., 0]
                    elif pol == "Hz":
                        fz = farfield["Hz"][0, ..., 0]
                        fx = farfield["Ex"][0, ..., 0]
                        fy = farfield["Ey"][0, ..., 0]

                if direction[0] == "x":  # Yee grid average
                    fz = (fz[:-1] + fz[1:]) / 2
                    if pol == "Ez":
                        fx = fx[1:]
                        fy = fy[1:]
                    elif pol == "Hz":
                        fx = fx[:-1]
                        fy = fy[:-1]
                else:
                    fz = (fz[:, :-1] + fz[:, 1:]) / 2
                    if pol == "Ez":
                        fx = fx[:, 1:]
                        fy = fy[:, 1:]
                    elif pol == "Hz":
                        fx = fx[:, :-1]
                        fy = fy[:, :-1]
                s = get_flux(
                    fx,
                    fy,
                    fz,
                    monitor=None,
                    grid_step=grid_step,
                    direction=direction[0],
                    autograd=True,
                    pol=pol,
                )
                if isinstance(s, Tensor):
                    abs = torch.abs
                else:
                    abs = npa.abs
                s = abs(s / norm_power)  # we only need absolute flux

                ## we need to average the flux across the region, which is treated as multiple slices
                s = s / (fz.shape[0] if direction[0] == "x" else fz.shape[1])

                s_list.append(s)
                self.s_params[(self.out_port_name, wl, self.obj_type, temp)] = {
                    "s": s,
                }
        if isinstance(s_list[0], Tensor):
            return torch.mean(torch.stack(s_list))
        else:
            return npa.mean(npa.array(s_list))  # we only need absolute flux


class FluxObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        s_params: dict,
        port_profiles: dict,  # port monitor profiles {port_name: {(wl, mode): (profile, ht_m, et_m)}}
        port_slices: dict,
        in_port_name: str,
        out_port_name: str,
        in_mode: int,
        direction: str,
        name: str,
        target_temps: Tuple[float],
        grid_step: float,
        minus_src: bool = False,
        obj_type: str = "flux",
    ):
        self.sims = sims
        self.s_params = s_params
        self.port_profiles = port_profiles
        self.port_slices = port_slices
        self.in_port_name = in_port_name
        self.out_port_name = out_port_name
        self.in_mode = in_mode
        self.direction = direction
        self.name = name
        self.target_temps = target_temps
        self.grid_step = grid_step
        self.minus_src = minus_src
        self.obj_type = obj_type

    def __call__(self, fields):
        s_list = []
        (
            target_temps,
            in_port_name,
            out_port_name,
            in_mode,
            direction,
            name,
            grid_step,
        ) = (
            self.target_temps,
            self.in_port_name,
            self.out_port_name,
            self.in_mode,
            self.direction,
            self.name,
            self.grid_step,
        )

        s_list = []
        ## for each wavelength, we evaluate the objective
        for (wl, pol), _ in self.sims.items():
            for temp in target_temps:
                monitor_slice = self.port_slices[out_port_name]
                norm_power = self.port_profiles[in_port_name][(wl, in_mode)][3]
                field = fields[(in_port_name, wl, in_mode, temp)]
                pol = in_mode[:2]
                if pol == "Ez":
                    fx, fy, fz = (
                        field["Hx"],
                        field["Hy"],
                        field["Ez"],
                    )  # fetch fields
                elif pol == "Hz":
                    fx, fy, fz = (
                        field["Ex"],
                        field["Ey"],
                        field["Hz"],
                    )
                s = get_flux(
                    fx,
                    fy,
                    fz,
                    monitor_slice,
                    grid_step=grid_step,
                    direction=direction[0],
                    autograd=True,
                    pol=pol,
                )
                if isinstance(s, Tensor):
                    abs = torch.abs
                else:
                    abs = npa.abs
                s = abs(s / norm_power)  # we only need absolute flux
                if self.minus_src:
                    s = abs(
                        s - 1
                    )  ## if it is larger than 1, then this slice must include source, we minus the power from source

                s_list.append(s)
                if self.minus_src:  # which means that we are calculating the reflection
                    self.s_params[(self.out_port_name, wl, self.obj_type, temp)] = {
                        "s_m": s,
                        "s_p": 1 - s,
                    }
                else:
                    self.s_params[(self.out_port_name, wl, self.obj_type, temp)] = {
                        "s": s,
                    }
        if isinstance(s_list[0], Tensor):
            return torch.mean(torch.stack(s_list))
        else:
            return npa.mean(npa.array(s_list))  # we only need absolute flux


class ShapeSimilarityObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        port_slices: dict,
        port_slices_info: dict,
        in_port_name: str,
        out_port_name: str,
        in_mode: int,
        out_modes: Tuple[int],
        name: str,
        target_wls: Tuple[float],
        target_temps: Tuple[float],
        shape_type: str,
        shape_cfg: dict,
        grid_step: float,
        intensity: bool = True,
        similarity: str = "angular",
        obj_type: str = "intensity_shape",
    ):
        self.sims = sims
        self.port_slices = port_slices
        self.port_slices_info = port_slices_info
        self.in_port_name = in_port_name
        self.out_port_name = out_port_name
        self.in_mode = in_mode
        self.out_modes = out_modes
        self.name = name
        self.target_wls = target_wls
        self.target_temps = target_temps
        self.shape_type = shape_type
        self.shape_cfg = shape_cfg
        self.grid_step = grid_step
        self.intensity = intensity
        self.similarity = similarity
        self.obj_type = obj_type

    def __call__(self, fields):
        (
            target_wls,
            target_temps,
            in_port_name,
            out_port_name,
            in_mode,
            out_modes,
            shape_type,
            shape_cfg,
        ) = (
            self.target_wls,
            self.target_temps,
            self.in_port_name,
            self.out_port_name,
            self.in_mode,
            self.out_modes,
            self.shape_type,
            self.shape_cfg,
        )

        similarity_list = []
        ## for each wavelength, we evaluate the objective
        for (wl, pol), sim in self.sims.items():
            ## we calculate the average eigen energy for all output modes
            if wl not in target_wls:
                continue
            if pol != in_mode[:2]:
                continue
            for out_mode in out_modes:
                for temp in target_temps:
                    monitor_slice = self.port_slices[out_port_name]
                    monitor_direction = self.port_slices_info[out_port_name][
                        "direction"
                    ]
                    fz = fields[(in_port_name, wl, in_mode, temp)][pol]
                    fz = fz[monitor_slice]
                    if (
                        len(monitor_slice.x.shape) <= 1
                        or len(monitor_slice.y.shape) <= 1
                    ):  # 1d slice
                        fz = fz.reshape(1, -1)
                    else:  # 2d slice
                        if monitor_direction[0] == "y":
                            fz = fz.t()
                    shape_similarity = get_shape_similarity(
                        fz,
                        grid_step=self.grid_step,
                        shape_type=shape_type,
                        shape_cfg=shape_cfg,
                        intensity=self.intensity,
                        similarity=self.similarity,
                    )
                    similarity_list.append(shape_similarity)
        if isinstance(similarity_list[0], Tensor):
            return torch.mean(torch.stack(similarity_list))
        else:
            return npa.mean(npa.array(similarity_list))


class ShapeSimilarityNear2FarObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        port_slices: dict,
        port_slices_info: dict,
        in_port_name: str,
        out_port_name: str,
        in_mode: int,
        out_modes: Tuple[int],
        name: str,
        target_wls: Tuple[float],
        target_temps: Tuple[float],
        shape_type: str,
        shape_cfg: dict,
        grid_step: float,
        eps_bg: float,
        intensity: bool = True,
        similarity: str = "angular",
        obj_type: str = "intensity_shape_near2far",
        total_farfield_region_solutions: dict = None,
    ):
        self.sims = sims
        self.port_slices = port_slices
        self.port_slices_info = port_slices_info
        self.in_port_name = in_port_name
        self.out_port_name = out_port_name
        self.in_mode = in_mode
        self.out_modes = out_modes
        self.name = name
        self.target_wls = target_wls
        self.target_temps = target_temps
        self.shape_type = shape_type
        self.shape_cfg = shape_cfg
        self.grid_step = grid_step
        self.eps_bg = eps_bg
        self.intensity = intensity
        self.similarity = similarity
        self.obj_type = obj_type
        self.total_farfield_region_solutions = total_farfield_region_solutions

    def __call__(self, fields):
        (
            target_wls,
            target_temps,
            in_port_name,
            out_port_name,
            in_mode,
            out_modes,
            shape_type,
            shape_cfg,
        ) = (
            self.target_wls,
            self.target_temps,
            self.in_port_name,
            self.out_port_name,
            self.in_mode,
            self.out_modes,
            self.shape_type,
            self.shape_cfg,
        )

        similarity_list = []
        ## for each wavelength, we evaluate the objective
        for (wl, pol), sim in self.sims.items():
            ## we calculate the average eigen energy for all output modes
            if wl not in target_wls:
                continue
            if pol != in_mode[:2]:
                continue
            for out_mode in out_modes:
                for temp in target_temps:
                    monitor_slice = self.port_slices[out_port_name]
                    monitor_direction = self.port_slices_info[out_port_name][
                        "direction"
                    ]
                    field = fields[(in_port_name, wl, in_mode, temp)]
                    if pol == "Ez":
                        fx_near, fy_near, fz_near = (
                            field["Hx"],
                            field["Hy"],
                            field["Ez"],
                        )
                    elif pol == "Hz":
                        fx_near, fy_near, fz_near = (
                            field["Ex"],
                            field["Ey"],
                            field["Hz"],
                        )

                    farfield = get_farfields_GreenFunction(
                        nearfield_slices=[
                            self.port_slices[nearfield_slice_name]
                            for nearfield_slice_name in list(self.port_slices.keys())
                            if nearfield_slice_name.startswith("nearfield")
                        ],
                        nearfield_slices_info=[
                            self.port_slices_info[nearfield_slice_name]
                            for nearfield_slice_name in list(
                                self.port_slices_info.keys()
                            )
                            if nearfield_slice_name.startswith("nearfield")
                        ],
                        Fz=fz_near[None, ..., None],
                        Fx=fx_near[None, ..., None],
                        Fy=fy_near[None, ..., None],
                        farfield_x=None,
                        farfield_slice_info=self.port_slices_info[out_port_name],
                        freqs=torch.tensor([1 / wl], device=fz_near.device),
                        eps=self.eps_bg,
                        mu=MU_0,
                        dL=self.grid_step,
                        component=pol,
                        decimation_factor=4,
                    )
                    
                    fz = farfield[pol][0, ..., 0]
                    # ez = ez[monitor_slice]
                    if (
                        len(monitor_slice.x.shape) <= 1
                        or len(monitor_slice.y.shape) <= 1
                    ):  # 1d slice
                        fz = fz.reshape(1, -1)
                    else:  # 2d slice
                        if monitor_direction[0] == "y":
                            fz = fz.t()
                    shape_similarity = get_shape_similarity(
                        fz,
                        grid_step=self.grid_step,
                        shape_type=shape_type,
                        shape_cfg=shape_cfg,
                        intensity=self.intensity,
                        similarity=self.similarity,
                    )
                    similarity_list.append(shape_similarity)
        if isinstance(similarity_list[0], Tensor):
            return torch.mean(torch.stack(similarity_list))
        else:
            return npa.mean(npa.array(similarity_list))


class ObjectiveFunc(object):
    def __init__(
        self,
        simulations: dict,
        port_profiles: dict,  # port monitor profiles {port_name: {(wl, mode): (profile, ht_m, et_m)}}
        port_slices: dict,
        port_slices_info: dict,
        grid_step: float,
        eps_bg: float,
        device,  # BaseDevice
        verbose=False,
    ):
        """_summary_

        Args:
            simulations (dict): {(wl, mode): Simulation}
            port_profiles (dict): port monitor profiles {port_name: {(wl, mode): (profile, ht_m, et_m)}}
            port_slices (dict): port slices {port_name: Slice}
            grid_step (float): um
        """
        self.sims = simulations
        self.port_profiles = port_profiles
        self.port_slices = port_slices
        self.port_slices_info = port_slices_info
        self.grid_step = grid_step
        self.eps_bg = eps_bg
        self.device = device  # BaseDevice

        self.eps = None
        self.Ez = None
        self.Js = {}  # forward from fields to foms
        self.adj_Js = {}  # Js for adjoint source calculation
        self.dJ = None  # backward from fom to permittivity
        self.breakdown = {}
        self.solutions = {}
        self.total_farfield_region_solutions = {}
        self.verbose = verbose
        self.obj_cfgs = dict()

    def switch_solver(self, neural_solver, numerical_solver, use_autodiff=False):
        for simulation in self.sims.values():
            simulation.switch_solver(neural_solver, numerical_solver, use_autodiff)

    def add_objective(
        self,
        cfgs: dict = dict(
            fwd_trans=dict(
                weight=1,
                type="eigenmode",
                #### objective is evaluated at this port
                in_port_name="in_port_1",
                out_port_name="out_port_1",
                #### objective is evaluated at all points by sweeping the wavelength and modes
                in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                out_modes=(
                    "Ez1",
                ),  # can evaluate on multiple output modes and get average transmission
                direction="x+",
            )
        ),
    ):
        self.s_params = {}
        self._obj_fusion_func = cfgs["_fusion_func"]
        cfgs = deepcopy(cfgs)
        del cfgs["_fusion_func"]
        self.obj_cfgs.update(cfgs)
        ### build objective functions from solved fields to fom
        for name, cfg in cfgs.items():
            obj_type = cfg["type"]
            in_port_name = cfg["in_port_name"]
            out_port_name = cfg["out_port_name"]
            in_mode = cfg["in_mode"]
            out_modes = cfg["out_modes"]
            direction = cfg["direction"]
            target_wls = cfg["wl"]
            target_temps = cfg["temp"]
            shape_type = cfg.get("shape_type", None)
            shape_cfg = cfg.get("shape_cfg", None)

            if obj_type == "eigenmode":
                objfn = EigenmodeObjective(
                    sims=self.sims,
                    s_params=self.s_params,
                    port_profiles=self.port_profiles,
                    port_slices=self.port_slices,
                    in_port_name=in_port_name,
                    out_port_name=out_port_name,
                    in_mode=in_mode,
                    out_modes=out_modes,
                    direction=direction,
                    name=name,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    grid_step=self.grid_step,
                    obj_type=obj_type,
                )
            elif obj_type in {"flux", "flux_minus_src"}:
                objfn = FluxObjective(
                    sims=self.sims,
                    s_params=self.s_params,
                    port_profiles=self.port_profiles,
                    port_slices=self.port_slices,
                    in_port_name=in_port_name,
                    out_port_name=out_port_name,
                    in_mode=in_mode,
                    direction=direction,
                    name=name,
                    target_temps=target_temps,
                    grid_step=self.grid_step,
                    minus_src=obj_type == "flux_minus_src",
                    obj_type=obj_type,
                )

            elif obj_type in {"flux_near2far"}:
                objfn = FluxNear2FarObjective(
                    sims=self.sims,
                    s_params=self.s_params,
                    port_profiles=self.port_profiles,
                    port_slices=self.port_slices,
                    port_slices_info=self.port_slices_info,
                    in_port_name=in_port_name,
                    out_port_name=out_port_name,
                    in_mode=in_mode,
                    direction=direction,
                    name=name,
                    target_temps=target_temps,
                    grid_step=self.grid_step,
                    eps_bg=self.eps_bg,
                    obj_type=obj_type,
                    total_farfield_region_solutions=self.total_farfield_region_solutions,
                )

            elif obj_type == "phase":
                # this is to make a equal phase MMI
                objfn = EigenmodeObjective(
                    sims=self.sims,
                    s_params=self.s_params,
                    port_profiles=self.port_profiles,
                    port_slices=self.port_slices,
                    in_port_name=in_port_name,
                    out_port_name=out_port_name,
                    in_mode=in_mode,
                    out_modes=out_modes,
                    direction=direction,
                    name=name,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    grid_step=self.grid_step,
                    energy=False,
                    obj_type=obj_type,
                )
            elif obj_type == "intensity_shape":
                objfn = ShapeSimilarityObjective(
                    sims=self.sims,
                    port_slices=self.port_slices,
                    port_slices_info=self.port_slices_info,
                    in_port_name=in_port_name,
                    out_port_name=out_port_name,
                    in_mode=in_mode,
                    out_modes=out_modes,
                    name=name,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    grid_step=self.grid_step,
                    shape_type=shape_type,
                    shape_cfg=shape_cfg,
                    intensity=True,
                    similarity="angular",
                    obj_type=obj_type,
                )

            elif obj_type == "intensity_shape_near2far":
                objfn = ShapeSimilarityNear2FarObjective(
                    sims=self.sims,
                    port_slices=self.port_slices,
                    port_slices_info=self.port_slices_info,
                    in_port_name=in_port_name,
                    out_port_name=out_port_name,
                    in_mode=in_mode,
                    out_modes=out_modes,
                    name=name,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    grid_step=self.grid_step,
                    shape_type=shape_type,
                    shape_cfg=shape_cfg,
                    eps_bg=self.eps_bg,
                    intensity=True,
                    similarity="angular",
                    obj_type=obj_type,
                    total_farfield_region_solutions=self.total_farfield_region_solutions,
                )
            else:
                raise ValueError("Invalid type")

            ### note that this is not the final objective! this is partial objective from fields to fom
            ### complete autograd graph is from permittivity (np.ndarray) to fields and to fom
            self.Js[name] = {"weight": cfg["weight"], "fn": objfn}

    def build_jacobian(self):
        ## obtain_objective is the complete forward function starts from permittivity to solved fields, then to fom
        self.dJ = jacobian(self.obtain_objective, mode="reverse")

    def build_adj_jacobian(self):
        self.dJ_dE = {}
        for name, obj in self.adj_Js.items():
            dJ_dE_fn = {}
            for (wl, out_mode), obj_fn in obj["fn"].items():
                dJ_dE = jacobian(obj_fn, mode="reverse")
                dJ_dE_fn[(wl, out_mode)] = dJ_dE
            self.dJ_dE[name] = {"weight": obj["weight"], "fn": dJ_dE_fn}

    def obtain_adj_srcs(self):
        # this should be called after obtain_objective, other wise self.solutions is empty
        adj_sources = {}
        field_adj = {}
        field_adj_normalizer = {}
        for key, sim in self.sims.items():
            adj_sources[key] = sim.solver.adj_src  # this is the b_adj
            fz_adj, fx_adj, fy_adj, flux = sim.norm_adj_power()
            # field_adj[key] = {"Ez": ez_adj, "Hx": hx_adj, "Hy": hy_adj}
            field_adj[key] = {}
            for (port_name, mode), _ in fz_adj.items():
                pol = mode[:2]
                if pol == "Ez":
                    field_adj[key][(port_name, mode)] = {
                        "Ez": fz_adj[(port_name, mode)],
                        "Hx": fx_adj[(port_name, mode)],
                        "Hy": fy_adj[(port_name, mode)],
                    }
                elif pol == "Hz":
                    field_adj[key][(port_name, mode)] = {
                        "Hz": fz_adj[(port_name, mode)],
                        "Ex": fx_adj[(port_name, mode)],
                        "Ey": fy_adj[(port_name, mode)],
                    }
            field_adj_normalizer[key] = flux
        return adj_sources, field_adj, field_adj_normalizer

    def obtain_objective(
        self, permittivity: np.ndarray | Tensor
    ) -> Tuple[dict, Tensor]:
        self.solutions = {}
        self.As = {}
        temperatures = []
        for _, cfg in self.obj_cfgs.items():
            temperatures = temperatures + cfg["temp"]
        temperatures = set(temperatures)
        for port_name, port_profile in self.port_profiles.items():
            for (wl, mode), (source, _, _, norm_power) in port_profile.items():
                ## here the source is already normalized during norm_run to make sure it has target power
                ## here is the key part that build the common "eps to field" autograd graph
                ## later on, multiple "field to fom" autograd graph(s) will be built inside of multiple obj_fn's
                pol = mode[:2]
                ## temperature is effective only when there is active region defined
                for temp in temperatures:
                    if getattr(self.device, "active_region_masks", None) is not None:
                        control_cfgs = {
                            name: {"T": temp}
                            for name in self.device.active_region_masks.keys()
                        }

                        self.sims[
                            (wl, pol)
                        ].eps_r = self.device.apply_active_modulation(
                            permittivity, control_cfgs
                        )
                    else:
                        self.sims[(wl, pol)].eps_r = permittivity
                    ## eps_r: permittivity tensor, denormalized

                    # self.sims[wl].eps_r = get_temp_related_eps(
                    #     eps=permittivity,
                    #     temp=temp,
                    #     temp_0=300,
                    #     eps_r_0=Si_eps(wl),
                    #     dn_dT=1.8e-4,
                    # )
                    Fx, Fy, Fz = self.sims[(wl, pol)].solve(
                        source, port_name=port_name, mode=mode
                    )
                    pol = mode[:2]
                    if pol == "Ez":
                        self.solutions[(port_name, wl, mode, temp)] = {
                            "Hx": Fx,
                            "Hy": Fy,
                            "Ez": Fz,
                        }
                    elif pol == "Hz":
                        self.solutions[(port_name, wl, mode, temp)] = {
                            "Ex": Fx,
                            "Ey": Fy,
                            "Hz": Fz,
                        }

                    self.As[(wl, temp)] = self.sims[(wl, pol)].A

        self.breakdown = {}
        for name, obj in self.Js.items():
            weight, value = obj["weight"], obj["fn"](fields=self.solutions)
            self.breakdown[name] = {
                "weight": weight,
                "value": value,
            }
        ## here we accept customized fusion function, e.g., weighted sum by default.
        fusion_results = self._obj_fusion_func(self.breakdown)
        if isinstance(fusion_results, tuple):
            total_loss, extra_breakdown = fusion_results
        else:
            total_loss = fusion_results
            extra_breakdown = {}
        self.breakdown.update(extra_breakdown)
        if self.verbose:
            print(f"Total loss: {total_loss}, Breakdown: {self.breakdown}")
        return total_loss

    def obtain_gradient(
        self, permittivity: np.ndarray, eps_shape: Tuple[int] = None
    ) -> np.ndarray:
        ## we need denormalized entire permittivity
        grad = np.squeeze(self.dJ(permittivity))

        grad = grad.reshape(eps_shape)
        return grad

    def __call__(
        self,
        permittivity: np.ndarray | Tensor,
        eps_shape: Tuple[int] = None,
        mode: str = "forward",
    ):
        if mode == "forward":
            objective = self.obtain_objective(permittivity)
            return objective
        elif mode == "backward":
            return self.obtain_gradient(permittivity, eps_shape)
