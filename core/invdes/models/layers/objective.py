import copy
import logging
import math
from copy import deepcopy
from typing import Any, Tuple

import numpy as np
import torch
from autograd import numpy as npa
from torch import Tensor

from core.fdfd.near2far import get_farfields_GreenFunction
from core.utils import (
    RectilinearGridMetadata,
    energy_constraint_penalty,
    get_eigenmode_coefficients,
    get_eigenmode_coefficients_3d,
    get_flux,
    get_flux_3d,
    get_shape_similarity,
    interface_field_penalty,
    print_stat,
    structure_simplify_penalty,
    yee_to_colocate_interpolate,
)
from thirdparty.ceviche import jacobian
from thirdparty.ceviche.constants import MU_0

from .thermal_control import currents_key_to_dict


class SMatrixObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        s_params: dict,
        port_profiles: dict,  # port monitor profiles {slice_name: {(wl, mode): (profile, ht_m, et_m)}}
        port_slices: dict,
        in_slice_names: Tuple[str],
        out_slice_names: Tuple[str],
        in_mode: int,
        out_modes: Tuple[int],
        directions: Tuple[str],
        name: str,
        target_wls: Tuple[float],
        target_temps: Tuple[float],
        grid_step: float,
        energy: bool = False,
        obj_type: str = "smatrix",
        grid_metadata: RectilinearGridMetadata | dict[str, Any] | None = None,
        cell_weights: Tensor | np.ndarray | None = None,
    ):
        self.sims = sims
        self.s_params = s_params
        self.port_profiles = port_profiles
        self.port_slices = port_slices
        self.in_slice_names = in_slice_names
        self.out_slice_names = out_slice_names
        self.in_mode = in_mode
        self.out_modes = out_modes
        self.directions = directions
        self.name = name
        self.target_wls = target_wls
        self.target_temps = target_temps
        self.grid_step = grid_step
        self.energy = energy
        self.obj_type = obj_type
        self.grid_metadata = grid_metadata
        self.cell_weights = cell_weights

    def __call__(self, fields):
        s_list = []
        (
            target_wls,
            target_temps,
            in_slice_names,
            out_slice_names,
            in_mode,
            out_modes,
            directions,
            name,
            grid_step,
        ) = (
            self.target_wls,
            self.target_temps,
            self.in_slice_names,
            self.out_slice_names,
            self.in_mode,
            self.out_modes,
            self.directions,
            self.name,
            self.grid_step,
        )
        target_temps = set(target_temps)

        wl = list(self.sims.keys())[0][0]
        if isinstance(wl, tuple) and len(wl) == 3:
            is_fdtdx3d = True
            all_raw_keys = list(self.sims.keys())
            all_keys = []
            for (wl_cen, wl_width, n_wl), pol, temp in all_raw_keys:
                wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)
                for wl in wls:
                    all_keys.append((wl, pol, temp))
        else:
            is_fdtdx3d = False
            all_keys = list(self.sims.keys())

        assert len(out_slice_names) == len(
            directions
        ), f"out_slice_names and directions must have the same length, but got {len(out_slice_names)} and {len(directions)}"
        ## for each wavelength, we evaluate the objective
        for in_slice_name in in_slice_names:
            for out_slice_name, direction in zip(out_slice_names, directions):
                ## for each wavelength, we evaluate the objective
                for wl, pol, temp in all_keys:
                    ## we calculate the average eigen energy for all output modes
                    if not any(
                        math.isclose(wl, target_wl, rel_tol=0, abs_tol=1e-4)
                        for target_wl in target_wls
                    ):
                        print(
                            f"skip wl {wl} for objective {name} since it's not in target_wls={target_wls}"
                        )
                        continue
                    if pol != in_mode[:2]:
                        print(
                            f"skip pol {pol} for objective {name} since it's different from in_mode {in_mode}"
                        )
                        continue
                    for out_mode in out_modes:
                        if temp in target_temps:
                            src, ht_m, et_m, norm_p, require_sim = self.port_profiles[
                                out_slice_name
                            ][(wl, out_mode)]
                            if is_fdtdx3d:
                                norm_power = 1
                            else:
                                norm_power = self.port_profiles[in_slice_name][
                                    (wl, in_mode)
                                ][3]
                            monitor_slice = self.port_slices[out_slice_name]

                            if (in_slice_name, wl, in_mode, temp) not in fields:
                                print(
                                    f"field for {(in_slice_name, wl, in_mode, temp)} not found in fields. keys are {list(fields.keys())}"
                                )

                            field = fields[(in_slice_name, wl, in_mode, temp)]
                            if not is_fdtdx3d:
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
                                if (
                                    isinstance(ht_m, Tensor)
                                    and ht_m.device != fz.device
                                ):
                                    ht_m = ht_m.to(fz.device)
                                    et_m = et_m.to(fz.device)
                                    self.port_profiles[out_slice_name][
                                        (wl, out_mode)
                                    ] = [
                                        src.to(fz.device),
                                        ht_m,
                                        et_m,
                                        norm_p,
                                        require_sim,
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
                                    cell_weights=self.cell_weights,
                                )
                            else:
                                Ex, Ey, Ez, Hx, Hy, Hz = (
                                    field["Ex"],
                                    field["Ey"],
                                    field["Ez"],
                                    field["Hx"],
                                    field["Hy"],
                                    field["Hz"],
                                )
                                if (
                                    isinstance(ht_m, Tensor)
                                    and ht_m.device != Ez.device
                                ):
                                    ht_m = ht_m.to(Ez.device)
                                    et_m = et_m.to(Ez.device)
                                    self.port_profiles[out_slice_name][
                                        (wl, out_mode)
                                    ] = [
                                        src.to(Ez.device),
                                        ht_m,
                                        et_m,
                                        norm_p,
                                        require_sim,
                                    ]
                                s_p, s_m = get_eigenmode_coefficients_3d(
                                    Ex,
                                    Ey,
                                    Ez,
                                    Hx,
                                    Hy,
                                    Hz,
                                    ht_m,
                                    et_m,
                                    monitor=monitor_slice,
                                    grid_step=grid_step,
                                    energy=False,
                                    direction=direction,
                                    grid_metadata=self.grid_metadata,
                                    cell_weights=self.cell_weights,
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

                            # only record the s parameters for eigenmode
                            # we don't need to record the s parameters if we calculate the phase
                            self.s_params[
                                (
                                    in_slice_name,
                                    out_slice_name,
                                    out_mode,
                                    wl,
                                    in_mode,
                                    temp,
                                )
                            ] = {
                                "s_p": (
                                    s_p / norm_power
                                    if self.energy
                                    else s_p / norm_power**0.5
                                ),  # normalized by input power
                                "s_m": (
                                    s_m / norm_power
                                    if self.energy
                                    else s_m / norm_power**0.5
                                ),  # normalized by input power
                            }

        if isinstance(s_list[0], Tensor):
            return torch.stack(s_list)
        else:
            return npa.array(s_list)


class EigenmodeObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        s_params: dict,
        port_profiles: dict,  # port monitor profiles {slice_name: {(wl, mode): (profile, ht_m, et_m)}}
        port_slices: dict,
        in_slice_name: str,
        out_slice_name: str,
        in_mode: int,
        out_modes: Tuple[int],
        direction: str,
        name: str,
        target_wls: Tuple[float],
        target_temps: Tuple[float],
        grid_step: float,
        energy: bool = True,
        obj_type: str = "eigenmode",
        grid_metadata: RectilinearGridMetadata | dict[str, Any] | None = None,
        cell_weights: Tensor | np.ndarray | None = None,
    ):
        self.sims = sims
        self.s_params = s_params
        self.port_profiles = port_profiles
        self.port_slices = port_slices
        self.in_slice_name = in_slice_name
        self.out_slice_name = out_slice_name
        self.in_mode = in_mode
        self.out_modes = out_modes
        self.direction = direction
        self.name = name
        self.target_wls = target_wls
        self.target_temps = target_temps
        self.grid_step = grid_step
        self.energy = energy
        self.obj_type = obj_type
        self.grid_metadata = grid_metadata
        self.cell_weights = cell_weights

    def __call__(self, fields):
        s_list = []
        (
            target_wls,
            target_temps,
            in_slice_name,
            out_slice_name,
            in_mode,
            out_modes,
            direction,
            name,
            grid_step,
        ) = (
            self.target_wls,
            self.target_temps,
            self.in_slice_name,
            self.out_slice_name,
            self.in_mode,
            self.out_modes,
            self.direction,
            self.name,
            self.grid_step,
        )
        target_temps = set(target_temps)
        wl = list(self.sims.keys())[0][0]
        if isinstance(wl, tuple) and len(wl) == 3:
            is_fdtdx3d = True
            all_raw_keys = list(self.sims.keys())
            all_keys = []
            for (wl_cen, wl_width, n_wl), pol, temp in all_raw_keys:
                wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)
                for wl in wls:
                    all_keys.append((wl, pol, temp))
        else:
            is_fdtdx3d = False
            all_keys = list(self.sims.keys())

        ## for each wavelength, we evaluate the objective
        for wl, pol, temp in all_keys:
            ## we calculate the average eigen energy for all output modes
            if not any(
                math.isclose(wl, target_wl, rel_tol=0, abs_tol=1e-4)
                for target_wl in target_wls
            ):
                # print(
                #     f"skip wl {wl} for objective {name} since it's not in target_wls={target_wls}"
                # )
                continue
            if pol != in_mode[:2]:
                continue
            for out_mode in out_modes:
                if temp in target_temps:
                    src, ht_m, et_m, norm_p, require_sim = self.port_profiles[
                        out_slice_name
                    ][(wl, out_mode)]
                    if is_fdtdx3d:
                        norm_power = 1
                    else:
                        norm_power = self.port_profiles[in_slice_name][(wl, in_mode)][3]
                    monitor_slice = self.port_slices[out_slice_name]

                    if (in_slice_name, wl, in_mode, temp) not in fields:
                        print(
                            f"field for {(in_slice_name, wl, in_mode, temp)} not found in fields. keys are {list(fields.keys())}"
                        )
                    field = fields[(in_slice_name, wl, in_mode, temp)]
                    if not is_fdtdx3d:
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
                            self.port_profiles[out_slice_name][(wl, out_mode)] = [
                                src.to(fz.device),
                                ht_m,
                                et_m,
                                norm_p,
                                require_sim,
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
                            cell_weights=self.cell_weights,
                        )
                    else:
                        Ex, Ey, Ez, Hx, Hy, Hz = (
                            field["Ex"],
                            field["Ey"],
                            field["Ez"],
                            field["Hx"],
                            field["Hy"],
                            field["Hz"],
                        )
                        if isinstance(ht_m, Tensor) and ht_m.device != Ez.device:
                            ht_m = ht_m.to(Ez.device)
                            et_m = et_m.to(Ez.device)
                            self.port_profiles[out_slice_name][(wl, out_mode)] = [
                                src.to(Ez.device),
                                ht_m,
                                et_m,
                                norm_p,
                                require_sim,
                            ]
                        s_p, s_m = get_eigenmode_coefficients_3d(
                            Ex,
                            Ey,
                            Ez,
                            Hx,
                            Hy,
                            Hz,
                            ht_m,
                            et_m,
                            monitor=monitor_slice,
                            grid_step=grid_step,
                            energy=self.energy,
                            direction=direction,
                            grid_metadata=self.grid_metadata,
                            cell_weights=self.cell_weights,
                        )

                    if direction[1] == "+":
                        s = s_p
                    elif direction[1] == "-":
                        s = s_m
                    else:
                        raise ValueError("Invalid direction")
                    # print(s, norm_power)
                    if self.energy:
                        s_list.append(s / abs(norm_power))
                    else:
                        s_list.append(s / abs(norm_power**0.5))
                    if self.obj_type == "eigenmode":
                        # only record the s parameters for eigenmode
                        # we don't need to record the s parameters if we calculate the phase
                        self.s_params[
                            (in_slice_name, out_slice_name, out_mode, wl, in_mode, temp)
                        ] = {
                            "s_p": (
                                s_p / norm_power
                                if self.energy
                                else s_p / norm_power**0.5
                            ),  # normalized by input power
                            "s_m": (
                                s_m / norm_power
                                if self.energy
                                else s_m / norm_power**0.5
                            ),  # normalized by input power
                        }
        if isinstance(s_list[0], Tensor):
            # return torch.mean(torch.stack(s_list))
            return torch.stack(s_list)
        else:
            return npa.array(s_list)
            # return npa.mean(npa.array(s_list))


class FluxNear2FarObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        s_params: dict,
        port_profiles: dict,  # port monitor profiles {port_name: {(wl, mode): (profile, ht_m, et_m)}}
        port_slices: dict,
        port_slices_info: dict,
        in_slice_name: str,
        out_slice_name: str,
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
        self.in_slice_name = in_slice_name
        self.out_slice_name = out_slice_name
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
            in_slice_name,
            out_slice_name,
            in_mode,
            direction,
            name,
            grid_step,
        ) = (
            self.target_temps,
            self.in_slice_name,
            self.out_slice_name,
            self.in_mode,
            self.direction,
            self.name,
            self.grid_step,
        )

        s_list = []
        target_temps = set(target_temps)
        ## for each wavelength, we evaluate the objective
        for (wl, pol, temp), _ in self.sims.items():
            if pol != in_mode[:2]:
                continue
            if temp in target_temps:
                # monitor_slice = self.port_slices[out_port_name]
                norm_power = self.port_profiles[in_slice_name][(wl, in_mode)][3]
                # this is how ez, hx and hy are calculated in regular simulation
                field = fields[(in_slice_name, wl, in_mode, temp)]
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
                    self.port_slices_info[out_slice_name]
                )
                ## Ez, extend toward negative dir, Hz extend toward positive dir
                if direction[0] == "x":
                    xs = extended_farfield_slice_info["xs"]
                    if not xs.shape:
                        extended_farfield_slice_info["xs"] = np.array(
                            [xs - grid_step, xs]
                            if pol == "Ez"
                            else [xs, xs + grid_step]
                        )
                    else:
                        extended_farfield_slice_info["xs"] = np.concatenate(
                            (
                                [xs[0:1] - grid_step, xs]
                                if pol == "Ez"
                                else [xs, xs[-1:] + grid_step]
                            ),
                            axis=0,
                        )
                elif direction[0] == "y":
                    ys = extended_farfield_slice_info["ys"]
                    if not ys.shape:
                        extended_farfield_slice_info["ys"] = np.array(
                            [ys - grid_step, ys]
                            if pol == "Ez"
                            else [ys, ys + grid_step]
                        )
                    else:
                        extended_farfield_slice_info["ys"] = np.concatenate(
                            (
                                [ys[0:1] - grid_step, ys]
                                if pol == "Ez"
                                else [ys, ys[-1:] + grid_step]
                            ),
                            axis=0,
                        )
                if out_slice_name == "total_farfield_region":
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
                            farfield_slice_info=self.port_slices_info[out_slice_name],
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
                            (in_slice_name, wl, in_mode, temp)
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
                            (in_slice_name, wl, in_mode, temp)
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
                        farfield_slice_info=self.port_slices_info[out_slice_name],
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
                    cell_weights=self.cell_weights,
                )
                if isinstance(s, Tensor):
                    abs = torch.abs
                else:
                    abs = npa.abs
                s = abs(s / norm_power)  # we only need absolute flux

                ## we need to average the flux across the region, which is treated as multiple slices
                s = s / (fz.shape[0] if direction[0] == "x" else fz.shape[1])

                s_list.append(s)
                self.s_params[
                    (in_slice_name, out_slice_name, self.obj_type, wl, in_mode, temp)
                ] = {
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
        in_slice_name: str,
        out_slice_name: str,
        in_mode: int,
        direction: str,
        name: str,
        target_temps: Tuple[float],
        target_wls: Tuple[float],
        grid_step: float,
        minus_src: bool = False,
        obj_type: str = "flux",
        grid_metadata: RectilinearGridMetadata | dict[str, Any] | None = None,
        cell_weights: Tensor | np.ndarray | None = None,
    ):
        self.sims = sims
        self.s_params = s_params
        self.port_profiles = port_profiles
        self.port_slices = port_slices
        self.in_slice_name = in_slice_name
        self.out_slice_name = out_slice_name
        self.in_mode = in_mode
        self.direction = direction
        self.name = name
        self.target_temps = target_temps
        self.target_wls = target_wls
        self.grid_step = grid_step
        self.minus_src = minus_src
        self.obj_type = obj_type
        self.grid_metadata = grid_metadata
        self.cell_weights = cell_weights

    def __call__(self, fields):
        s_list = []
        (
            target_temps,
            target_wls,
            in_slice_name,
            out_slice_name,
            in_mode,
            direction,
            name,
            grid_step,
        ) = (
            self.target_temps,
            self.target_wls,
            self.in_slice_name,
            self.out_slice_name,
            self.in_mode,
            self.direction,
            self.name,
            self.grid_step,
        )

        s_list = []
        target_temps = set(target_temps)
        target_wls = set(target_wls)
        ## for each wavelength, we evaluate the objective
        wl = list(self.sims.keys())[0][0]
        if isinstance(wl, tuple) and len(wl) == 3:
            is_fdtdx3d = True
            all_raw_keys = list(self.sims.keys())
            all_keys = []
            for (wl_cen, wl_width, n_wl), pol, temp in all_raw_keys:
                wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)
                for wl in wls:
                    all_keys.append((wl, pol, temp))
        else:
            is_fdtdx3d = False
            all_keys = list(self.sims.keys())

        ## for each wavelength, we evaluate the objective
        for wl, pol, temp in all_keys:
            if not any(
                math.isclose(wl, target_wl, rel_tol=0, abs_tol=1e-4)
                for target_wl in target_wls
            ):
                # print(
                #     f"skip wl {wl} for objective {name} since it's not in target_wls={target_wls}"
                # )
                continue
            if temp in target_temps:
                monitor_slice = self.port_slices[out_slice_name]
                if is_fdtdx3d:
                    norm_power = 1
                else:
                    norm_power = self.port_profiles[in_slice_name][(wl, in_mode)][3]

                field = fields[(in_slice_name, wl, in_mode, temp)]
                if not is_fdtdx3d:
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
                        cell_weights=self.cell_weights,
                    )
                else:
                    Ex, Ey, Ez, Hx, Hy, Hz = (
                        field["Ex"],
                        field["Ey"],
                        field["Ez"],
                        field["Hx"],
                        field["Hy"],
                        field["Hz"],
                    )
                    s = get_flux_3d(
                        Ex,
                        Ey,
                        Ez,
                        Hx,
                        Hy,
                        Hz,
                        grid_step=grid_step,
                        monitor=monitor_slice,
                        direction=direction[0],
                        grid_metadata=self.grid_metadata,
                        cell_weights=self.cell_weights,
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
                    self.s_params[
                        (
                            in_slice_name,
                            out_slice_name,
                            self.obj_type,
                            wl,
                            in_mode,
                            temp,
                        )
                    ] = {
                        "s_m": s,
                        "s_p": 1 - s,
                    }
                else:
                    self.s_params[
                        (
                            in_slice_name,
                            out_slice_name,
                            self.obj_type,
                            wl,
                            in_mode,
                            temp,
                        )
                    ] = {
                        "s": s,
                    }
        if isinstance(s_list[0], Tensor):
            # return torch.mean(torch.stack(s_list))
            return torch.stack(s_list)
        else:
            # return npa.mean(npa.array(s_list))  # we only need absolute flux
            return npa.array(s_list)  # we only need absolute flux


class ResponseRecorderObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        response: dict,
        port_slices: dict,
        in_slice_name: str,
        out_slice_name: str,
        in_mode: int,
        target_wls: Tuple[float],
        target_temps: Tuple[float],
        obj_type: str = "response_record",
        grid_metadata: RectilinearGridMetadata | dict[str, Any] | None = None,
        cell_weights: Tensor | np.ndarray | None = None,
    ):
        self.sims = sims
        self.response = response
        self.port_slices = port_slices
        self.in_slice_name = in_slice_name
        self.out_slice_name = out_slice_name
        self.in_mode = in_mode
        self.target_wls = target_wls
        self.target_temps = target_temps
        self.obj_type = obj_type
        self.grid_metadata = grid_metadata
        self.cell_weights = cell_weights

    def __call__(self, fields):
        (
            target_wls,
            target_temps,
            in_slice_name,
            out_slice_name,
            in_mode,
        ) = (
            self.target_wls,
            self.target_temps,
            self.in_slice_name,
            self.out_slice_name,
            self.in_mode,
        )
        mean_phase_list = []
        target_temps = set(target_temps)
        ## for each wavelength, we evaluate the objective
        for (wl, pol, temp), sim in self.sims.items():
            ## we calculate the average eigen energy for all output modes
            if not any(
                math.isclose(wl, target_wl, rel_tol=0, abs_tol=1e-4)
                for target_wl in target_wls
            ):
                continue
            if temp in target_temps:
                monitor_slice = self.port_slices[out_slice_name]
                fz = fields[(in_slice_name, wl, in_mode, temp)][pol]
                fz = fz[monitor_slice]
                assert (
                    len(monitor_slice.x.shape) <= 1 or len(monitor_slice.y.shape) <= 1
                ), "Only 1D slice is supported for phase recorder"
                fz = fz.reshape(1, -1)
                # calculate the phase of fz
                phase = torch.angle(fz)
                phase_std = torch.std(phase)
                phase_mean = torch.mean(phase)
                mag = torch.abs(fz)
                # phase_mean = torch.remainder(phase_mean, 2 * torch.pi)
                self.response[(in_slice_name, out_slice_name, wl, in_mode, temp)] = {
                    # "phase": torch.remainder(phase, 2 * torch.pi),
                    "fz": fz,
                    "mag": mag,
                    "phase": phase,
                    "phase_std": phase_std,
                    "phase_mean": phase_mean,
                }
                mean_phase_list.append(phase_mean)

        if isinstance(mean_phase_list[0], Tensor):
            return torch.mean(torch.stack(mean_phase_list))
        else:
            return npa.mean(npa.array(mean_phase_list))


class ShapeSimilarityObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        port_slices: dict,
        port_slices_info: dict,
        in_slice_name: str,
        out_slice_name: str,
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
        grid_metadata: RectilinearGridMetadata | dict[str, Any] | None = None,
        cell_weights: Tensor | np.ndarray | None = None,
    ):
        self.sims = sims
        self.port_slices = port_slices
        self.port_slices_info = port_slices_info
        self.in_slice_name = in_slice_name
        self.out_slice_name = out_slice_name
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
        self.grid_metadata = grid_metadata
        self.cell_weights = cell_weights

    def __call__(self, fields):
        (
            target_wls,
            target_temps,
            in_slice_name,
            out_slice_name,
            in_mode,
            out_modes,
            shape_type,
            shape_cfg,
        ) = (
            self.target_wls,
            self.target_temps,
            self.in_slice_name,
            self.out_slice_name,
            self.in_mode,
            self.out_modes,
            self.shape_type,
            self.shape_cfg,
        )

        similarity_list = []
        target_temps = set(target_temps)
        ## for each wavelength, we evaluate the objective
        for (wl, pol, temp), sim in self.sims.items():
            ## we calculate the average eigen energy for all output modes
            if not any(
                math.isclose(wl, target_wl, rel_tol=0, abs_tol=1e-4)
                for target_wl in target_wls
            ):
                continue
            if pol != in_mode[:2]:
                continue
            for out_mode in out_modes:
                if temp in target_temps:
                    monitor_slice = self.port_slices[out_slice_name]
                    monitor_direction = self.port_slices_info[out_slice_name][
                        "direction"
                    ]
                    fz = fields[(in_slice_name, wl, in_mode, temp)][pol]
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
        in_slice_name: str,
        out_slice_name: str,
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
        grid_metadata: RectilinearGridMetadata | dict[str, Any] | None = None,
        cell_weights: Tensor | np.ndarray | None = None,
    ):
        self.sims = sims
        self.port_slices = port_slices
        self.port_slices_info = port_slices_info
        self.in_slice_name = in_slice_name
        self.out_slice_name = out_slice_name
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
        self.grid_metadata = grid_metadata
        self.cell_weights = cell_weights

    def __call__(self, fields):
        (
            target_wls,
            target_temps,
            in_slice_name,
            out_slice_name,
            in_mode,
            out_modes,
            shape_type,
            shape_cfg,
        ) = (
            self.target_wls,
            self.target_temps,
            self.in_slice_name,
            self.out_slice_name,
            self.in_mode,
            self.out_modes,
            self.shape_type,
            self.shape_cfg,
        )

        similarity_list = []
        target_temps = set(target_temps)
        ## for each wavelength, we evaluate the objective
        for (wl, pol, temp), sim in self.sims.items():
            ## we calculate the average eigen energy for all output modes
            if not any(
                math.isclose(wl, target_wl, rel_tol=0, abs_tol=1e-4)
                for target_wl in target_wls
            ):
                continue
            if pol != in_mode[:2]:
                continue
            for out_mode in out_modes:
                if temp in target_temps:
                    monitor_slice = self.port_slices[out_slice_name]
                    monitor_direction = self.port_slices_info[out_slice_name][
                        "direction"
                    ]
                    field = fields[(in_slice_name, wl, in_mode, temp)]
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
                        farfield_slice_info=self.port_slices_info[out_slice_name],
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


class InterfacePenaltyObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        port_profiles: dict,  # port monitor profiles {slice_name: {(wl, mode): (profile, ht_m, et_m)}}
        port_slices: dict,
        in_slice_name: str,
        in_mode: int,
        target_wls: Tuple[float],
        target_temps: Tuple[float],
        interface_weights=(1.0, 1.0, 0.0),
        grid_metadata: RectilinearGridMetadata | dict[str, Any] | None = None,
        cell_weights: Tensor | np.ndarray | None = None,
    ):
        self.sims = sims
        self.port_profiles = port_profiles
        self.port_slices = port_slices
        self.in_slice_name = in_slice_name
        self.in_mode = in_mode
        self.target_wls = target_wls
        self.target_temps = target_temps
        self.interface_weights = interface_weights
        self.grid_metadata = grid_metadata
        self.cell_weights = cell_weights

    def __call__(self, fields):
        (
            target_wls,
            target_temps,
            in_slice_name,
            in_mode,
        ) = (
            self.target_wls,
            self.target_temps,
            self.in_slice_name,
            self.in_mode,
        )
        target_temps = set(target_temps)
        wl = list(self.sims.keys())[0][0]
        ## we need epsilon, we just use one epsilon from the first simulation as example.
        epsilon = self.sims[list(self.sims.keys())[0]].eps_r
        if isinstance(wl, tuple) and len(wl) == 3:
            is_fdtdx3d = True
            all_raw_keys = list(self.sims.keys())
            all_keys = []
            for (wl_cen, wl_width, n_wl), pol, temp in all_raw_keys:
                wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)
                for wl in wls:
                    all_keys.append((wl, pol, temp))
        else:
            is_fdtdx3d = False
            all_keys = list(self.sims.keys())

        ## for each wavelength, we evaluate the objective
        total_penalty = 0
        for wl, pol, temp in all_keys:
            ## we calculate the average eigen energy for all output modes
            if not any(
                math.isclose(wl, target_wl, rel_tol=0, abs_tol=1e-4)
                for target_wl in target_wls
            ):
                # print(
                #     f"skip wl {wl} for objective {name} since it's not in target_wls={target_wls}"
                # )
                continue
            if pol != in_mode[:2]:
                continue

            if temp in target_temps:
                if (in_slice_name, wl, in_mode, temp) not in fields:
                    print(
                        f"field for {(in_slice_name, wl, in_mode, temp)} not found in fields. keys are {list(fields.keys())}"
                    )
                field = fields[(in_slice_name, wl, in_mode, temp)]
                if not is_fdtdx3d:
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
                    f_xyz = torch.stack([fx, fy, fz], dim=0)[..., None]  # [3, X, Y, Z]

                    penalty = interface_field_penalty(
                        E=f_xyz,
                        eps=epsilon[..., None],  # [X,Y,Z]
                        interface_weights=self.interface_weights,
                    )
                else:
                    E_xyz = torch.stack(
                        [field["Ex"], field["Ey"], field["Ez"]], dim=0
                    )  # [3, X, Y, Z]

                    penalty = interface_field_penalty(
                        E=E_xyz,
                        eps=epsilon,  # [X,Y,Z]
                        interface_weights=self.interface_weights,
                    )
                total_penalty += penalty

        if isinstance(total_penalty, Tensor):
            return total_penalty
        else:
            return npa.array(total_penalty)


class EnergyConstraintObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        port_profiles: dict,  # port monitor profiles {slice_name: {(wl, mode): (profile, ht_m, et_m)}}
        port_slices: dict,
        in_slice_name: str,
        in_mode: int,
        target_wls: Tuple[float],
        target_temps: Tuple[float],
        design_region_masks: dict = None,
        design_region_cfgs: dict = None,
    ):
        self.sims = sims
        self.port_profiles = port_profiles
        self.port_slices = port_slices
        self.in_slice_name = in_slice_name
        self.in_mode = in_mode
        self.target_wls = target_wls
        self.target_temps = target_temps
        self.design_region_masks = design_region_masks
        self.design_region_cfgs = design_region_cfgs

    def __call__(self, fields):
        (
            target_wls,
            target_temps,
            in_slice_name,
            in_mode,
        ) = (
            self.target_wls,
            self.target_temps,
            self.in_slice_name,
            self.in_mode,
        )
        target_temps = set(target_temps)
        wl = list(self.sims.keys())[0][0]
        ## we need epsilon, we just use one epsilon from the first simulation as example.
        epsilon = self.sims[list(self.sims.keys())[0]].eps_r
        if isinstance(wl, tuple) and len(wl) == 3:
            is_fdtdx3d = True
            all_raw_keys = list(self.sims.keys())
            all_keys = []
            for (wl_cen, wl_width, n_wl), pol, temp in all_raw_keys:
                wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)
                for wl in wls:
                    all_keys.append((wl, pol, temp))
        else:
            is_fdtdx3d = False
            all_keys = list(self.sims.keys())

        ## for each wavelength, we evaluate the objective
        total_penalty = 0
        for wl, pol, temp in all_keys:
            ## we calculate the average eigen energy for all output modes
            if not any(
                math.isclose(wl, target_wl, rel_tol=0, abs_tol=1e-4)
                for target_wl in target_wls
            ):
                # print(
                #     f"skip wl {wl} for objective {name} since it's not in target_wls={target_wls}"
                # )
                continue
            if pol != in_mode[:2]:
                continue

            if temp in target_temps:
                if (in_slice_name, wl, in_mode, temp) not in fields:
                    print(
                        f"field for {(in_slice_name, wl, in_mode, temp)} not found in fields. keys are {list(fields.keys())}"
                    )
                field = fields[(in_slice_name, wl, in_mode, temp)]
                if not is_fdtdx3d:
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
                    # f_xyz = torch.stack([fx, fy, fz], dim=0)[..., None]  # [3, X, Y, Z]
                    penalty = 0
                    for region_name, region_mask in self.design_region_masks.items():
                        region_cfg = self.design_region_cfgs[region_name]
                        eps_bg = region_cfg.get("eps_bg", 1.0)
                        eps_r = region_cfg.get("eps_r", 12.0)

                        ## the gradient flow back to both fz and epsilon
                        penalty = penalty + energy_constraint_penalty(
                            fz=fz,  # [X, Y]
                            epsilon_map=epsilon,  # [X,Y]
                            eps_clad=eps_bg,
                            eps_core=eps_r,
                            design_region_mask=region_mask,  # [X,Y]
                        )
                else:
                    E_xyz = torch.stack(
                        [field["Ex"], field["Ey"], field["Ez"]], dim=0
                    )  # [3, X, Y, Z]
                    penalty = 0
                    for region_name, region_mask in self.design_region_masks.items():
                        region_cfg = self.design_region_cfgs[region_name]
                        eps_bg = region_cfg.get("eps_bg", 1.0)
                        eps_r = region_cfg.get("eps_r", 12.0)

                        ## the gradient flow back to both fz and epsilon
                        penalty = penalty + energy_constraint_penalty(
                            fz=E_xyz,  # [3, X, Y, Z]
                            epsilon_map=epsilon,  # [X, Y, Z]
                            eps_clad=eps_bg,
                            eps_core=eps_r,
                            design_region_mask=region_mask,  # [X, Y, Z]
                            is_3d=True,
                        )

                total_penalty += penalty

        if isinstance(total_penalty, Tensor):
            return total_penalty
        else:
            return npa.array(total_penalty)


class StructureSimplifyObjective(object):
    def __init__(
        self,
        sims: dict,  # {wl: Simulation}
        port_profiles: dict,  # port monitor profiles {slice_name: {(wl, mode): (profile, ht_m, et_m)}}
        port_slices: dict,
        in_slice_name: str,
        in_mode: int,
        target_wls: Tuple[float],
        target_temps: Tuple[float],
        design_region_masks: dict = None,
        design_region_cfgs: dict = None,
        energy_threshold: float = 0.01,
        soft: bool = True,
        temperature: float = 0.05,
    ):
        self.sims = sims
        self.port_profiles = port_profiles
        self.port_slices = port_slices
        self.in_slice_name = in_slice_name
        self.in_mode = in_mode
        self.target_wls = target_wls
        self.target_temps = target_temps
        self.design_region_masks = design_region_masks
        self.design_region_cfgs = design_region_cfgs
        self.energy_threshold = energy_threshold
        self.soft = soft
        self.temperature = temperature

    def __call__(self, fields):
        (
            target_wls,
            target_temps,
            in_slice_name,
            in_mode,
        ) = (
            self.target_wls,
            self.target_temps,
            self.in_slice_name,
            self.in_mode,
        )
        target_temps = set(target_temps)
        wl = list(self.sims.keys())[0][0]
        ## we need epsilon, we just use one epsilon from the first simulation as example.
        epsilon = self.sims[list(self.sims.keys())[0]].eps_r
        if isinstance(wl, tuple) and len(wl) == 3:
            is_fdtdx3d = True
            all_raw_keys = list(self.sims.keys())
            all_keys = []
            for (wl_cen, wl_width, n_wl), pol, temp in all_raw_keys:
                wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)
                for wl in wls:
                    all_keys.append((wl, pol, temp))
        else:
            is_fdtdx3d = False
            all_keys = list(self.sims.keys())

        ## for each wavelength, we evaluate the objective
        total_penalty = 0
        for wl, pol, temp in all_keys:
            ## we calculate the average eigen energy for all output modes
            if not any(
                math.isclose(wl, target_wl, rel_tol=0, abs_tol=1e-4)
                for target_wl in target_wls
            ):
                # print(
                #     f"skip wl {wl} for objective {name} since it's not in target_wls={target_wls}"
                # )
                continue
            if pol != in_mode[:2]:
                continue

            if temp in target_temps:
                if (in_slice_name, wl, in_mode, temp) not in fields:
                    print(
                        f"field for {(in_slice_name, wl, in_mode, temp)} not found in fields. keys are {list(fields.keys())}"
                    )
                field = fields[(in_slice_name, wl, in_mode, temp)]
                if not is_fdtdx3d:
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
                    # f_xyz = torch.stack([fx, fy, fz], dim=0)[..., None]  # [3, X, Y, Z]
                    penalty = 0
                    for region_name, region_mask in self.design_region_masks.items():
                        region_cfg = self.design_region_cfgs[region_name]
                        eps_bg = region_cfg.get("eps_bg", 1.0)
                        eps_r = region_cfg.get("eps_r", 12.0)

                        penalty = penalty + structure_simplify_penalty(
                            fz=fz,  # [X, Y]
                            epsilon_map=epsilon,  # [X,Y]
                            eps_clad=eps_bg,
                            eps_core=eps_r,
                            design_region_mask=region_mask,  # [X,Y]
                            energy_threshold=self.energy_threshold,
                            soft=self.soft,
                            temperature=self.temperature,
                        )
                else:
                    E_xyz = torch.stack(
                        [field["Ex"], field["Ey"], field["Ez"]], dim=0
                    )  # [3, X, Y, Z]
                    penalty = 0
                    for region_name, region_mask in self.design_region_masks.items():
                        region_cfg = self.design_region_cfgs[region_name]
                        eps_bg = region_cfg.get("eps_bg", 1.0)
                        eps_r = region_cfg.get("eps_r", 12.0)

                        ## the gradient flow back to both fz and epsilon
                        penalty = penalty + structure_simplify_penalty(
                            fz=E_xyz,  # [3, X, Y, Z]
                            epsilon_map=epsilon,  # [X, Y, Z]
                            eps_clad=eps_bg,
                            eps_core=eps_r,
                            design_region_mask=region_mask,  # [X, Y, Z]
                            energy_threshold=self.energy_threshold,
                            soft=self.soft,
                            temperature=self.temperature,
                            is_3d=True,
                        )
                total_penalty += penalty

        if isinstance(total_penalty, Tensor):
            return total_penalty
        else:
            return npa.array(total_penalty)


class ObjectiveFunc(object):
    def __init__(
        self,
        simulations: dict,
        port_profiles: dict,  # port monitor profiles {port_name: {(wl, mode): (profile, ht_m, et_m)}}
        port_slices: dict,
        port_slices_native: dict,
        port_slices_native_symmetry: dict,
        port_slices_info: dict,
        grid_step: float,
        eps_bg: float,
        device,  # BaseDevice
        control_states: dict | None = None,
        verbose=False,
        grid_metadata: RectilinearGridMetadata | dict[str, Any] | None = None,
        cell_weights: Tuple[Tensor | np.ndarray] | None = None,
        design_region_masks: dict = None,
        design_region_cfgs: dict = None,
    ):
        """_summary_

        Args:
            simulations (dict): {(wl, mode): Simulation}
            port_profiles (dict): port monitor profiles {port_name: {(wl, mode): (profile, ht_m, et_m)}}
            port_slices (dict): port slices {port_name: Slice}
            port_slices_native (dict): native port slices {port_name: Slice}
            port_slices_info (dict): port slice info {port_name: Info}
            grid_step (float): um
        """
        self.sims = simulations
        self.port_profiles = port_profiles
        self.port_slices = port_slices
        self.port_slices_native = port_slices_native
        self.port_slices_info = port_slices_info
        self.grid_step = grid_step
        self.eps_bg = eps_bg
        self.device = device  # BaseDevice
        self.control_states = control_states or {}
        self.runtime_material_maps = None
        self.grid_metadata = grid_metadata
        self.cell_weights = cell_weights
        self.port_slices_native_symmetry = port_slices_native_symmetry
        self.design_region_masks = design_region_masks
        self.design_region_cfgs = design_region_cfgs

        if self.grid_metadata is not None or self.cell_weights is None:
            logging.warning(
                "[ObjectiveFunc] grid_metadata or cell_weights is provided, objective calculation is on native grid if supported."
            )
            self._interpolate_fields_to_export_grid = False
            self.port_slices = self.port_slices_native
        else:
            self._interpolate_fields_to_export_grid = True

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

        wl = list(self.sims.keys())[0][0]
        if isinstance(wl, tuple) and len(wl) == 3:
            self.is_fdtdx3d = True
        else:
            self.is_fdtdx3d = False

    @staticmethod
    def _ensure_tuple(value):
        if isinstance(value, (tuple, list)):
            return tuple(value)
        return (value,)

    def _build_fdtdx_adjoint_group_info(
        self,
        *,
        sim_wl_key,
        pol: str,
        temp: float,
        input_slice_name: str,
        mode: str,
    ):
        if not (isinstance(sim_wl_key, tuple) and len(sim_wl_key) == 3):
            return []

        wl_cen, wl_width, n_wl = sim_wl_key
        sim_wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)
        group_map = {}

        for obj_name, cfg in self.obj_cfgs.items():
            if not cfg.get("requires_grad", True):
                continue
            if not cfg.get("requires_adjoint", True):
                continue
            if mode != cfg["in_mode"]:
                continue
            control_keys = cfg["control_keys"] if "control_keys" in cfg else cfg["temp"]
            if temp not in set(control_keys):
                continue
            if input_slice_name not in self._ensure_tuple(cfg["in_slice_name"]):
                continue

            target_wls = cfg.get("wl", ())
            freq_indices = [
                idx
                for idx, wl in enumerate(sim_wls)
                if any(
                    math.isclose(wl, target_wl, rel_tol=0, abs_tol=1e-4)
                    for target_wl in target_wls
                )
            ]
            if len(freq_indices) == 0:
                continue

            out_slice_names = self._ensure_tuple(cfg["out_slice_name"])
            direction_cfg = cfg.get("direction", ())
            if isinstance(direction_cfg, (tuple, list)):
                directions = tuple(direction_cfg)
            else:
                directions = tuple(direction_cfg for _ in out_slice_names)

            for idx, out_slice_name in enumerate(out_slice_names):
                if out_slice_name not in self.port_slices_native:
                    continue
                direction = directions[idx] if idx < len(directions) else directions[0]
                group_id = out_slice_name
                group = group_map.setdefault(
                    group_id,
                    {
                        "group_id": group_id,
                        "monitor_name": out_slice_name,
                        "monitor_slice": (
                            self.port_slices_native_symmetry[out_slice_name]
                            if self.port_slices_native_symmetry
                            else self.port_slices_native[out_slice_name]
                        ),  # this is used to check adjoint source group on the cotangents on native grid.
                        "port_slice_info": self.port_slices_info.get(out_slice_name),
                        "freq_indices": set(),
                        "objective_names": set(),
                        "obj_types": set(),
                        "directions": set(),
                        "pol": pol,
                        "input_slice_name": input_slice_name,
                        "mode": mode,
                    },
                )
                group["freq_indices"].update(freq_indices)
                group["objective_names"].add(obj_name)
                group["obj_types"].add(cfg["type"])
                group["directions"].add(direction)

        group_info = []
        for group in group_map.values():
            group_info.append(
                {
                    **group,
                    "freq_indices": tuple(sorted(group["freq_indices"])),
                    "objective_names": tuple(sorted(group["objective_names"])),
                    "obj_types": tuple(sorted(group["obj_types"])),
                    "directions": tuple(sorted(group["directions"])),
                }
            )
        return group_info

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
                in_slice_name="in_slice_1",
                out_slice_name="out_slice_1",
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
        self.response = {}
        self._obj_fusion_func = cfgs["_fusion_func"]
        cfgs = deepcopy(cfgs)
        del cfgs["_fusion_func"]
        cfgs = {name: cfg for name, cfg in cfgs.items() if isinstance(cfg, dict)}
        self.obj_cfgs.update(cfgs)
        ### build objective functions from solved fields to fom
        for name, cfg in cfgs.items():
            obj_type = cfg["type"]
            in_slice_name = cfg.get("in_slice_name", None)
            out_slice_name = cfg.get("out_slice_name", None)
            in_mode = cfg.get("in_mode", None)
            out_modes = cfg.get("out_modes", None)
            direction = cfg.get("direction", None)
            target_wls = cfg.get("wl", None)
            target_temps = cfg["control_keys"] if "control_keys" in cfg else cfg["temp"]
            shape_type = cfg.get("shape_type", None)
            shape_cfg = cfg.get("shape_cfg", None)

            if obj_type == "eigenmode":
                objfn = EigenmodeObjective(
                    sims=self.sims,
                    s_params=self.s_params,
                    port_profiles=self.port_profiles,
                    port_slices=self.port_slices,
                    in_slice_name=in_slice_name,
                    out_slice_name=out_slice_name,
                    in_mode=in_mode,
                    out_modes=out_modes,
                    direction=direction,
                    name=name,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    grid_step=self.grid_step,
                    energy=cfg.get("energy", True),
                    obj_type=obj_type,
                    grid_metadata=self.grid_metadata,
                    cell_weights=self.cell_weights,
                )
            elif obj_type == "smatrix":
                objfn = SMatrixObjective(
                    sims=self.sims,
                    s_params=self.s_params,
                    port_profiles=self.port_profiles,
                    port_slices=self.port_slices,
                    in_slice_names=in_slice_name,
                    out_slice_names=out_slice_name,
                    in_mode=in_mode,
                    out_modes=out_modes,
                    directions=direction,
                    name=name,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    grid_step=self.grid_step,
                    energy=False,
                    obj_type=obj_type,
                    grid_metadata=self.grid_metadata,
                    cell_weights=self.cell_weights,
                )
            elif obj_type in {"flux", "flux_minus_src"}:
                objfn = FluxObjective(
                    sims=self.sims,
                    s_params=self.s_params,
                    port_profiles=self.port_profiles,
                    port_slices=self.port_slices,
                    in_slice_name=in_slice_name,
                    out_slice_name=out_slice_name,
                    in_mode=in_mode,
                    direction=direction,
                    name=name,
                    target_temps=target_temps,
                    target_wls=target_wls,
                    grid_step=self.grid_step,
                    minus_src=obj_type == "flux_minus_src",
                    obj_type=obj_type,
                    grid_metadata=self.grid_metadata,
                    cell_weights=self.cell_weights,
                )

            elif obj_type in {"flux_near2far"}:
                objfn = FluxNear2FarObjective(
                    sims=self.sims,
                    s_params=self.s_params,
                    port_profiles=self.port_profiles,
                    port_slices=self.port_slices,
                    port_slices_info=self.port_slices_info,
                    in_slice_name=in_slice_name,
                    out_slice_name=out_slice_name,
                    in_mode=in_mode,
                    direction=direction,
                    name=name,
                    target_temps=target_temps,
                    grid_step=self.grid_step,
                    eps_bg=self.eps_bg,
                    obj_type=obj_type,
                    total_farfield_region_solutions=self.total_farfield_region_solutions,
                    grid_metadata=self.grid_metadata,
                    cell_weights=self.cell_weights,
                )

            elif obj_type == "phase":
                # this is to make a equal phase MMI
                objfn = EigenmodeObjective(
                    sims=self.sims,
                    s_params=self.s_params,
                    port_profiles=self.port_profiles,
                    port_slices=self.port_slices,
                    in_slice_name=in_slice_name,
                    out_slice_name=out_slice_name,
                    in_mode=in_mode,
                    out_modes=out_modes,
                    direction=direction,
                    name=name,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    grid_step=self.grid_step,
                    energy=False,
                    obj_type=obj_type,
                    grid_metadata=self.grid_metadata,
                    cell_weights=self.cell_weights,
                )
            elif obj_type == "response_record":
                objfn = ResponseRecorderObjective(
                    sims=self.sims,
                    response=self.response,
                    port_slices=self.port_slices,
                    in_slice_name=in_slice_name,
                    out_slice_name=out_slice_name,
                    in_mode=in_mode,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    obj_type=obj_type,
                    grid_metadata=self.grid_metadata,
                    cell_weights=self.cell_weights,
                )
            elif obj_type == "intensity_shape":
                objfn = ShapeSimilarityObjective(
                    sims=self.sims,
                    port_slices=self.port_slices,
                    port_slices_info=self.port_slices_info,
                    in_slice_name=in_slice_name,
                    out_slice_name=out_slice_name,
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
                    grid_metadata=self.grid_metadata,
                    cell_weights=self.cell_weights,
                )

            elif obj_type == "intensity_shape_near2far":
                objfn = ShapeSimilarityNear2FarObjective(
                    sims=self.sims,
                    port_slices=self.port_slices,
                    port_slices_info=self.port_slices_info,
                    in_slice_name=in_slice_name,
                    out_slice_name=out_slice_name,
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
                    grid_metadata=self.grid_metadata,
                    cell_weights=self.cell_weights,
                )
            elif obj_type == "interface_penalty":
                objfn = InterfacePenaltyObjective(
                    sims=self.sims,
                    port_profiles=self.port_profiles,
                    port_slices=self.port_slices,
                    in_slice_name=in_slice_name,
                    in_mode=in_mode,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    interface_weights=cfg.get("interface_weights", (1.0, 1.0, 0.0)),
                    grid_metadata=self.grid_metadata,
                    cell_weights=self.cell_weights,
                )
            elif obj_type == "energy_constraint":
                objfn = EnergyConstraintObjective(
                    sims=self.sims,
                    port_profiles=self.port_profiles,
                    port_slices=self.port_slices,
                    in_slice_name=in_slice_name,
                    in_mode=in_mode,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    design_region_masks=self.design_region_masks,
                    design_region_cfgs=self.design_region_cfgs,
                )
            elif obj_type == "structure_simplify":
                objfn = StructureSimplifyObjective(
                    sims=self.sims,
                    port_profiles=self.port_profiles,
                    port_slices=self.port_slices,
                    in_slice_name=in_slice_name,
                    in_mode=in_mode,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    design_region_masks=self.design_region_masks,
                    design_region_cfgs=self.design_region_cfgs,
                    energy_threshold=cfg.get("energy_threshold", 0.01),
                    soft=cfg.get("soft", True),
                    temperature=cfg.get("temperature", 0.05),
                )
            else:
                raise ValueError("Invalid type")

            ### note that this is not the final objective! this is partial objective from fields to fom
            ### complete autograd graph is from permittivity (np.ndarray) to fields and to fom
            self.Js[name] = {
                "weight": cfg["weight"],
                "fn": objfn,
                "requires_grad": cfg.get("requires_grad", True),
            }

    def build_jacobian(self):
        # deprecated
        ## obtain_objective is the complete forward function starts from permittivity to solved fields, then to fom
        self.dJ = jacobian(self.obtain_objective, mode="reverse")

    def build_adj_jacobian(self):
        # deprecated
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
            fz_adj, fx_adj, fy_adj, flux = sim.norm_adj_power()
            # field_adj[key] = {"Ez": ez_adj, "Hx": hx_adj, "Hy": hy_adj}
            field_adj[key] = {}
            adj_sources[key] = {}
            for (slice_name, mode, temp), _ in fz_adj.items():
                pol = mode[:2]
                if pol == "Ez":
                    field_adj[key][(slice_name, mode, temp)] = {
                        "Ez": fz_adj[(slice_name, mode, temp)],
                        "Hx": fx_adj[(slice_name, mode, temp)],
                        "Hy": fy_adj[(slice_name, mode, temp)],
                    }
                elif pol == "Hz":
                    field_adj[key][(slice_name, mode, temp)] = {
                        "Hz": fz_adj[(slice_name, mode, temp)],
                        "Ex": fx_adj[(slice_name, mode, temp)],
                        "Ey": fy_adj[(slice_name, mode, temp)],
                    }
                # convert the b_adj --> J_adj since I want uniform here, in forward, we store the J source
                # and this adj_src is normalized so that the power is 1e-8 matches the ez_adj
                adj_sources[key][(slice_name, mode, temp)] = (
                    sim.solver.adj_src[(slice_name, mode, temp)] / 1j / sim.omega
                )
            field_adj_normalizer[key] = flux
        return adj_sources, field_adj, field_adj_normalizer

    def read_gradient(self):
        gradients = {}
        for wl, sim in self.sims.items():
            gradients[wl] = sim.read_gradients()
        return gradients

    def _get_control_state(self, control_key):
        state = self.control_states.get(control_key)
        if state is not None:
            return state
        if isinstance(control_key, tuple):
            return {
                "mode": "currents",
                "control_key": control_key,
                "currents": currents_key_to_dict(control_key),
            }
        return {
            "mode": "legacy_temp",
            "control_key": control_key,
            "temp": control_key,
        }

    def _resolve_runtime_state(self, permittivity, control_key):
        if not hasattr(self, "_runtime_state_cache"):
            self._runtime_state_cache = {}
        if control_key in self._runtime_state_cache:
            return self._runtime_state_cache[control_key]
        control_state = self._get_control_state(control_key)
        mode = control_state["mode"]
        if mode == "legacy_temp":
            temp = control_state["temp"]
            runtime_material_maps = self.runtime_material_maps
            if getattr(self.device, "active_region_masks", None) is not None:
                control_cfgs = {
                    name: {"T": temp} for name in self.device.active_region_masks.keys()
                }
                eps = self.device.apply_active_modulation(permittivity, control_cfgs)
            else:
                eps = permittivity
            electrical_conductivity = None
            if runtime_material_maps is not None:
                electrical_conductivity = runtime_material_maps.get(
                    "electrical_conductivity"
                )
            state = {
                "mode": mode,
                "control_key": control_key,
                "eps": eps,
                "temperature": None,
                "optical_temperature": None,
                "conductivity": None,
                "electrical_conductivity": electrical_conductivity,
                "heat_capacity": None,
                "thermo_optic_coeff": None,
                "optical_thermo_optic_coeff": None,
                "q_map": None,
            }
            self._runtime_state_cache[control_key] = state
            return state

        runtime_material_maps = self.runtime_material_maps
        if self.is_fdtdx3d and self.device._optical_grid_uses_fdtdx_nonuniform():
            # state = self.device.build_runtime_fdtdx_state(
            #     permittivity,
            #     control_state["currents"],
            #     runtime_material_maps=runtime_material_maps,
            # )
            state = self.device.build_runtime_thermal_state(
                permittivity,
                control_state["currents"],
                runtime_material_maps=runtime_material_maps,
            )
        else:
            state = self.device.build_runtime_thermal_state(
                permittivity,
                control_state["currents"],
                runtime_material_maps=runtime_material_maps,
            )
        state["mode"] = mode
        state["control_key"] = control_key
        self._runtime_state_cache[control_key] = state
        return state

    def _resolve_runtime_eps(self, permittivity, control_key):
        state = self._resolve_runtime_state(permittivity, control_key)
        return state["eps"]

    def _apply_fdtdx_runtime_state(self, sim, state):
        sim.eps_r = state["eps"]
        if state.get("field_grid_metadata") is not None:
            sim.field_grid_metadata = state["field_grid_metadata"]
            sim.optical_grid_metadata = state["field_grid_metadata"]
        electrical_conductivity = state.get("electrical_conductivity")
        if (
            electrical_conductivity is not None
            or getattr(sim, "electrical_conductivity_map", None) is None
        ):
            sim.electrical_conductivity_map = electrical_conductivity

    def obtain_objective(
        self, permittivity: np.ndarray | Tensor, custom_source: dict = None
    ) -> Tuple[dict, Tensor]:
        self.solutions = {}
        self.As = {}
        self._runtime_state_cache = {}
        self.runtime_thermal_states = {}
        temperatures = []
        torch.cuda.empty_cache()
        for _, cfg in self.obj_cfgs.items():
            temperatures = temperatures + (
                cfg["control_keys"] if "control_keys" in cfg else cfg["temp"]
            )
        temperatures = set(temperatures)
        if custom_source is None:
            ### we only refactor the A matrix if A changes, i.e., different wl, eps, pol.
            ### If only source/mode changes, the A keeps the same, we do not need to refactor the matrix, we can reuse the pardiso solver.
            ### so here we need to reorder the simulation and cluster them into groups, so I can reuse the psolve if they have the same (temp, pol, and wl)
            _simulation_queues = {}

            for slice_name, port_profile in self.port_profiles.items():
                ## for 2D FDFD, wl here is one number; for 3D fdtdx, wl is a tuple of (wl_cen, wl_width, n_wl)
                for (wl, mode), (
                    source,
                    _,
                    _,
                    norm_power,
                    require_sim,
                ) in port_profile.items():
                    if not require_sim:
                        continue
                    ## here the source is already normalized during norm_run to make sure it has target power
                    ## here is the key part that build the common "eps to field" autograd graph
                    ## later on, multiple "field to fom" autograd graph(s) will be built inside of multiple obj_fn's
                    pol = mode[:2]
                    for temp in temperatures:
                        if (wl, pol, temp) not in _simulation_queues:
                            _simulation_queues[(wl, pol, temp)] = [
                                dict(mode=mode, slice_name=slice_name, source=source)
                            ]
                        else:
                            _simulation_queues[(wl, pol, temp)].append(
                                dict(mode=mode, slice_name=slice_name, source=source)
                            )

            if len(_simulation_queues) == 0:
                print(
                    "Warning: No simulation is required by the objective function. Please check require_sim flag in port profiles."
                )
            # print(_simulation_queues)

            ## after clustering, we can run simulation for each group, and only factorize matrix once per group
            if not self.is_fdtdx3d:
                solvers = {"Ez": None, "Hz": None}
                ### each polarization share the same solver as we can use pydiso to share the symbolic factorization
                for (wl, pol, temp), sim_inst_cfgs in _simulation_queues.items():
                    sim = self.sims[(wl, pol, temp)]
                    if solvers[pol] is None:
                        solvers[pol] = sim.solver
                        sim.solver.set_cache_mode(True)
                        sim.solver.clear_solver_cache()
                    else:
                        sim.solver = solvers[pol]

                ## for each polarization's solver, we only need to factorize once with pydiso
                ## we have three scenarios here:
                ## group 1: first solve (slow), then just solve(b): fast
                ## group 2...: first solve (refactor, middle), then just solve(b): fast
                for (wl, pol, temp), sim_inst_cfgs in _simulation_queues.items():
                    sim = self.sims[(wl, pol, temp)]
                    # sim.set_cache_mode(True)
                    # sim.clear_solver_cache()
                    for idx, sim_inst_cfg in enumerate(sim_inst_cfgs):
                        slice_name = sim_inst_cfg["slice_name"]
                        mode = sim_inst_cfg["mode"]
                        source = sim_inst_cfg["source"]
                        ## for each simulation instance, we run simulation
                        sim.eps_r = self._resolve_runtime_eps(permittivity, temp)

                        Fx, Fy, Fz = sim.solve(
                            source, slice_name=slice_name, mode=mode, temp=temp
                        )
                        if pol == "Ez":
                            self.solutions[(slice_name, wl, mode, temp)] = {
                                "Hx": Fx,
                                "Hy": Fy,
                                "Ez": Fz,
                            }
                        elif pol == "Hz":
                            self.solutions[(slice_name, wl, mode, temp)] = {
                                "Ex": Fx,
                                "Ey": Fy,
                                "Hz": Fz,
                            }
                        else:
                            raise ValueError("Invalid polarization")

                        self.As[(wl, temp)] = sim.A
                ## IMPORTANT: You can clear the cache after backward. OR in next iteration's forward, the above cache will be cleared.
                # Do not clear it here, as the pSolve can be reused by backward if symmetric A
            else:
                ## 3D FDTDX
                for (
                    (wl_cen, wl_width, n_wl),
                    pol,
                    temp,
                ), sim_inst_cfgs in _simulation_queues.items():
                    sim = self.sims[((wl_cen, wl_width, n_wl), pol, temp)]
                    for idx, sim_inst_cfg in enumerate(sim_inst_cfgs):
                        slice_name = sim_inst_cfg["slice_name"]
                        mode = sim_inst_cfg["mode"]
                        source = sim_inst_cfg["source"]
                        ## for each simulation instance, we run simulation
                        state = self._resolve_runtime_state(permittivity, temp)
                        self._apply_fdtdx_runtime_state(sim, state)
                        adjoint_group_info = self._build_fdtdx_adjoint_group_info(
                            sim_wl_key=(wl_cen, wl_width, n_wl),
                            pol=pol,
                            temp=temp,
                            input_slice_name=slice_name,
                            mode=mode,
                        )
                        Ex, Ey, Ez, Hx, Hy, Hz = sim.solve(
                            input_slice_name=slice_name,
                            wl_cen=wl_cen,
                            wl_width=wl_width,
                            n_wl=n_wl,
                            mode=mode,
                            adjoint_group_info=adjoint_group_info,
                            interpolate_to_export_grid=self._interpolate_fields_to_export_grid,
                        )
                        wls = np.linspace(
                            wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl
                        )
                        E = torch.stack([Ex, Ey, Ez], dim=1)  # [nfreq, 3, Nx, Ny, Nz]
                        H = torch.stack([Hx, Hy, Hz], dim=1)  # [nfreq, 3, Nx, Ny, Nz]
                        if not getattr(sim, "fields_are_colocated", False):
                            E, H = (
                                yee_to_colocate_interpolate(
                                    E, sim.objects, sim.config, is_E=True
                                ),
                                yee_to_colocate_interpolate(
                                    H, sim.objects, sim.config, is_E=False
                                ),
                            )
                        ## we split fields into individual wavelength
                        for wl_idx, wl in enumerate(wls):
                            self.solutions[(slice_name, wl, mode, temp)] = {
                                "Ex": E[wl_idx, 0],
                                "Ey": E[wl_idx, 1],
                                "Ez": E[wl_idx, 2],
                                "Hx": H[wl_idx, 0],
                                "Hy": H[wl_idx, 1],
                                "Hz": H[wl_idx, 2],
                            }

        else:  # we have a custom source to simulate
            slice_name = custom_source["slice_name"]
            src = custom_source["source"]
            mode = custom_source["mode"]
            wl = custom_source["wl"]
            direction = custom_source["direction"]
            pol = mode[:2]

            # build source from slice_name and source vector:
            source_profile = self.device.insert_plane_wave(
                eps=self.device.epsilon_map,
                slice=self.device.port_monitor_slices[slice_name],
                wl_cen=wl,
                source_modes=(mode,),
                direction=direction,
                custom_source=src,
            )
            source = source_profile[(wl, mode)][0]
            ## temperature is effective only when there is active region defined
            for temp in temperatures:
                self.sims[(wl, pol, temp)].eps_r = self._resolve_runtime_eps(
                    permittivity, temp
                )

                Fx, Fy, Fz = self.sims[(wl, pol, temp)].solve(
                    source, slice_name=slice_name, mode=mode, temp=temp
                )

                if pol == "Ez":
                    self.solutions[(slice_name, wl, mode, temp)] = {
                        "Hx": Fx,
                        "Hy": Fy,
                        "Ez": Fz,
                    }
                elif pol == "Hz":
                    self.solutions[(slice_name, wl, mode, temp)] = {
                        "Ex": Fx,
                        "Ey": Fy,
                        "Hz": Fz,
                    }

                self.As[(wl, temp)] = self.sims[(wl, pol, temp)].A
        self.breakdown = {}
        self.runtime_thermal_states = dict(self._runtime_state_cache)
        for name, obj in self.Js.items():
            weight, value = obj["weight"], obj["fn"](fields=self.solutions)
            if not obj.get("requires_grad", True) and isinstance(value, Tensor):
                value = value.detach()
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
        custom_source: dict = None,
        mode: str = "forward",
    ):
        if mode == "forward":
            objective = self.obtain_objective(permittivity, custom_source=custom_source)
            return objective
        elif mode == "backward":
            return self.obtain_gradient(permittivity, eps_shape)
