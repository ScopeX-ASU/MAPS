import copy
import logging
import math
from copy import deepcopy
from typing import Any, Sequence, Tuple

import numpy as np
import torch
from autograd import numpy as npa
from torch import Tensor, nn

from core.fdfd.near2far import get_farfields_GreenFunction
from core.nearfield import (
    angular_spectrum_torch,
    concatenate_surfaces,
    from_maps_fields,
    project,
)
from core.nearfield.projector import C_0 as C_0_SI
from core.nearfield.projector import EPSILON_0
from core.nearfield.projector import MU_0 as MU_0_SI
from core.utils import (
    RectilinearGridMetadata,
    energy_constraint_penalty,
    erosion_dilation_penalty,
    get_ed_kernel_radius_px,
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
    """
    Compute normalized S-parameter/eigenmode transmission objectives across wavelength, temperature, mode, and direction combinations.

    Args:
        sims: Simulation mapping keyed by wavelength/polarization/temperature or an FDTD-X broadband wavelength configuration.
        s_params: Mutable dictionary receiving normalized modal coefficients.
        port_profiles: Mapping ``slice_name -> (wavelength, mode) -> (source, H_mode, E_mode, normalization_power, requires_simulation)``.
        port_slices: Mapping from slice name to monitor-slice metadata.
        in_slice_names: Input slices.
        out_slice_names: Output slices.
        in_mode: Input mode identifier such as ``Ez1`` or ``Hz1`` in 2-D workflows.
        out_modes: Output mode indices.
        directions: Propagation directions such as ``x+`` or ``y-``; length must equal ``out_slice_names``.
        name: Objective identifier.
        target_wls: Target wavelengths in micrometers.
        target_temps: Target temperatures/control states.
        grid_step: Spatial step for 2-D modal overlap.
        energy: Whether to return power-like or amplitude-like normalized coefficients.
        grid_metadata: Optional 3-D rectilinear grid metadata.
        cell_weights: Optional physical integration weights for nonuniform grids.

    Returns:
        Vector containing one normalized objective sample for each valid combination.

    Scientific conventions:
        2-D ``Ez`` uses ``(Hx, Hy, Ez)``; ``Hz`` uses ``(Ex, Ey, Hz)``. 3-D evaluation uses all six colocated components.
    """

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
        """
        Evaluate all configured S-matrix objective combinations from simulated fields.

        Args:
            fields: Mapping keyed by ``(in_slice_name, wavelength, in_mode, temperature)``. 2-D values contain polarization-specific field components; 3-D values contain ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, and ``Hz``.

        Returns:
            Stacked Torch tensor for Torch fields, otherwise a NumPy/autograd array.

        Side Effects:
            Updates ``self.s_params`` with normalized forward/backward modal coefficients.
        """
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
    """
    Compute an input-to-output eigenmode objective over configured wavelengths, temperatures, output modes, and propagation direction.

    Args:
        sims: Simulation mapping keyed by wavelength/polarization/temperature.
        s_params: Mutable S-parameter bookkeeping dictionary.
        port_profiles: Port source/reference profiles keyed by ``(wavelength, mode)``.
        port_slices: Output monitor-slice mapping.
        in_slice_name: Input slice name.
        out_slice_name: Output slice name.
        in_mode: Input mode identifier.
        out_modes: Output mode indices.
        direction: Propagation direction such as ``x+`` or ``y-``.
        target_wls: Target wavelengths in micrometers.
        target_temps: Target temperature/control states.
        grid_step: 2-D modal-overlap grid spacing.
        energy: Whether to return power-like or amplitude-like values.
        grid_metadata: Optional 3-D rectilinear metadata.
        cell_weights: Optional integration weights.

    Returns:
        Torch or NumPy vector of normalized eigenmode objective values.
    """

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
        """
        Evaluate the eigenmode objective from simulated field dictionaries.

        Args:
            fields: Mapping keyed by ``(in_slice_name, wavelength, in_mode, temperature)`` with polarization-specific 2-D fields or all six 3-D field components.

        Returns:
            Stacked tensor/array containing normalized eigenmode objective values.
        """
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


class TargetIntensityObjective:
    """Score a sampled image-plane intensity against a target array.

    Args:
        sims: Simulation mapping retained for the standard objective interface.
        port_slices: Monitor slices on the field grid.
        in_slice_name: Source slice whose solved fields are selected.
        out_slice_name: Image-plane monitor slice.
        in_mode: Source mode label.
        target_wls: Selected wavelengths in micrometers.
        target_temps: Selected thermal/control states.
        target_intensity: Nonnegative two-dimensional target intensity.
        field_components: Electric-field components included in intensity.

    Notes:
        Measured intensity is scaled to the target mean, clipped to the unit
        target range, and scored as one minus the normalized L2 mismatch.
    """

    def __init__(
        self,
        sims,
        port_slices,
        in_slice_name,
        out_slice_name,
        in_mode,
        target_wls,
        target_temps,
        target_intensity,
        field_components=("Ex", "Ey", "Ez"),
    ):
        self.sims = sims
        self.port_slices = port_slices
        self.in_slice_name = in_slice_name
        self.out_slice_name = out_slice_name
        self.in_mode = in_mode
        self.target_wls = tuple(target_wls)
        self.target_temps = set(target_temps)
        self.target_intensity = torch.as_tensor(target_intensity, dtype=torch.float32)
        if self.target_intensity.ndim != 2:
            raise ValueError("target_intensity must be a two-dimensional array")
        if not torch.any(self.target_intensity > 0):
            raise ValueError("target_intensity must contain a positive value")
        if isinstance(field_components, str):
            field_components = (field_components,)
        self.field_components = tuple(field_components)
        if not self.field_components:
            raise ValueError("field_components must contain at least one component")
        self.last_intensity = None
        self.last_target = None
        self.last_error = None

    def _score_intensity(self, plane: Tensor) -> Tensor:
        """Score one real image-plane intensity and retain diagnostics."""
        target = self.target_intensity.to(device=plane.device, dtype=plane.dtype)
        if tuple(target.shape) != tuple(plane.shape):
            target = torch.nn.functional.interpolate(
                target[None, None],
                size=tuple(plane.shape),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        target = target.clamp(0, 1)
        measured = (
            plane * target.mean() / plane.mean().clamp_min(torch.finfo(plane.dtype).eps)
        )
        measured = measured.clamp(0, 1)
        mismatch = torch.linalg.vector_norm(measured - target) / target.numel() ** 0.5
        self.last_intensity = measured.detach()
        self.last_target = target.detach()
        self.last_error = mismatch.detach()
        return 1 - mismatch

    def __call__(self, fields):
        """Return a differentiable image similarity score in [0, 1]."""
        values = []
        for (source_name, wl, mode, temp), components in fields.items():
            if source_name != self.in_slice_name or mode != self.in_mode:
                continue
            if temp not in self.target_temps:
                continue
            if not any(
                math.isclose(wl, target, rel_tol=0, abs_tol=1e-4)
                for target in self.target_wls
            ):
                continue
            plane = None
            for component in self.field_components:
                if component in components:
                    energy = components[component].abs().square()
                    plane = energy if plane is None else plane + energy
            if plane is None:
                raise ValueError(
                    "No requested electric-field components in "
                    f"{(source_name, wl, mode, temp)}"
                )
            plane = plane[self.port_slices[self.out_slice_name]].squeeze()
            if plane.ndim != 2:
                raise ValueError(
                    f"Image-plane field must reduce to 2-D, got {tuple(plane.shape)}"
                )
            values.append(self._score_intensity(plane))
        if not values:
            raise ValueError(
                "No simulation fields matched the target-intensity objective"
            )
        return torch.stack(values).mean()


class TargetIntensityAngularSpectrumObjective(TargetIntensityObjective):
    """Score a periodic image plane propagated from a tangential near field.

    Args:
        projection_distance_um: Positive distance from the near-field plane to
            the virtual image plane, in micrometers.
        grid_step_um: Uniform transverse sampling pitch in micrometers.
        eps_bg: Relative permittivity above the metasurface.
        evanescent: Angular-spectrum evanescent-wave policy.
        **kwargs: Arguments accepted by :class:`TargetIntensityObjective`.

    Notes:
        The FFT represents one periodically continued x/y supercell. The input
        plane must be normal to +z and uses its colocated ``Ex`` and ``Ey``.
        ``field_components`` independently selects which propagated electric
        components form the supervised intensity and defaults to
        ``("Ex", "Ey", "Ez")``. The returned magnetic field is retained for
        diagnostics but is not part of the intensity objective.
    """

    def __init__(
        self,
        *,
        projection_distance_um: float,
        grid_step_um: float,
        eps_bg: float = 1.0,
        evanescent: str = "decay",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if projection_distance_um <= 0:
            raise ValueError("projection_distance_um must be positive")
        if grid_step_um <= 0:
            raise ValueError("grid_step_um must be positive")
        self.projection_distance_um = float(projection_distance_um)
        self.grid_step_um = float(grid_step_um)
        self.eps_bg = float(eps_bg)
        self.evanescent = evanescent
        unsupported_components = set(self.field_components) - {"Ex", "Ey", "Ez"}
        if unsupported_components:
            raise ValueError(
                "Angular-spectrum intensity supports projected electric components "
                f"Ex, Ey, and Ez; got {sorted(unsupported_components)}"
            )
        self.last_nearfield_intensity = None
        self.last_projected_intensity = None
        self.last_nearfield_fields = None
        self.last_projected_fields = None

    def __call__(self, fields):
        """Propagate each selected near field and return its image score."""
        values = []
        for (source_name, wl, mode, temp), components in fields.items():
            if source_name != self.in_slice_name or mode != self.in_mode:
                continue
            if temp not in self.target_temps:
                continue
            if not any(
                math.isclose(wl, target, rel_tol=0, abs_tol=1e-4)
                for target in self.target_wls
            ):
                continue
            if not all(component in components for component in ("Ex", "Ey")):
                raise ValueError("Angular-spectrum projection requires Ex and Ey")
            monitor_slice = self.port_slices[self.out_slice_name]
            ex = components["Ex"][monitor_slice].squeeze()
            ey = components["Ey"][monitor_slice].squeeze()
            if ex.ndim != 2 or ey.shape != ex.shape:
                raise ValueError(
                    "Near-field Ex and Ey must reduce to matching 2-D planes"
                )
            # MAPS arrays use (x, y); angular_spectrum_torch uses (v, u),
            # where u is the first transverse Cartesian axis.
            tangential = torch.stack((ex.mT, ey.mT), dim=-1)
            electric, magnetic = angular_spectrum_torch(
                tangential,
                C_0_SI / (float(wl) * 1e-6),
                self.grid_step_um * 1e-6,
                self.grid_step_um * 1e-6,
                self.projection_distance_um * 1e-6,
                normal_axis=2,
                normal_sign=1,
                eps_r=self.eps_bg,
                evanescent=self.evanescent,
            )
            electric = electric.movedim(0, 1)
            magnetic = magnetic.movedim(0, 1)
            component_indices = [
                ("Ex", "Ey", "Ez").index(component)
                for component in self.field_components
            ]
            intensity = electric[..., component_indices].abs().square().sum(dim=-1)
            self.last_projected_intensity = intensity.detach()
            self.last_nearfield_intensity = (
                ex.abs().square() + ey.abs().square()
            ).detach()
            self.last_nearfield_fields = {
                "Ex": ex.detach(),
                "Ey": ey.detach(),
                "spacing_um": (self.grid_step_um, self.grid_step_um),
                "plane_axes": ("x", "y"),
            }
            self.last_projected_fields = {
                component: values.detach()
                for component, values in zip(
                    ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                    (*electric.unbind(-1), *magnetic.unbind(-1)),
                )
            }
            self.last_projected_fields["spacing_um"] = (
                self.grid_step_um,
                self.grid_step_um,
            )
            self.last_projected_fields["plane_axes"] = ("x", "y")
            values.append(self._score_intensity(intensity))
        if not values:
            raise ValueError(
                "No simulation fields matched the target-intensity objective"
            )
        return torch.stack(values).mean()


class ShapeSimilarityNear2FarObjective(object):
    """Project near fields and optionally reduce one component to similarity.

    Args:
        sims: Simulation mapping. FDTD-X simulations use broadband wavelength
            tuples as their first key element.
        port_slices: Monitor slices on the field grid.
        port_slices_info: Physical monitor metadata in micrometers.
        in_slice_name: Source slice whose solved fields are selected.
        out_slice_name: Cartesian observation region.
        in_mode: Source mode label.
        out_modes: Retained for compatibility with the legacy 2-D objective.
        name: Objective name.
        target_wls: Selected wavelengths in micrometers.
        target_temps: Selected thermal/control states.
        shape_type: Target shape type used when ``return_field`` is unset.
        shape_cfg: Shape parameters used by ``get_shape_similarity``.
        grid_step: Legacy 2-D grid spacing in micrometers.
        eps_bg: Homogeneous projection-medium relative permittivity.
        intensity: Whether similarity compares intensity.
        similarity: Similarity metric name.
        obj_type: Objective type identifier.
        total_farfield_region_solutions: Legacy far-field result store.
        grid_metadata: Native 3-D rectilinear metadata in micrometers.
        cell_weights: Optional objective quadrature weights.
        return_field: ``None`` for shape similarity, ``"nearfield"`` for raw
            equivalent-surface samples, or ``"farfield"`` for raw projected
            fields.
        projection_backend: Backend passed to ``core.nearfield.project`` in 3-D.

    Notes:
        Three-dimensional MAPS magnetic phasors use the solver's impedance
        normalization and are divided by the vacuum impedance before entering
        the SI near-field projector. Raw results are dictionaries keyed by
        ``(in_slice_name, wavelength_um, in_mode, control_state)``.
    """

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
        return_field: str | None = None,
        projection_backend: str = "direct_torch",
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
        if return_field not in (None, "nearfield", "farfield"):
            raise ValueError("return_field must be None, 'nearfield', or 'farfield'")
        self.return_field = return_field
        self.projection_backend = projection_backend
        self.last_nearfield_fields = None
        self.last_projected_fields = None

    def _is_fdtdx3d(self) -> bool:
        """Return whether the simulation keys use the FDTD-X broadband form."""
        wavelength_key = next(iter(self.sims))[0]
        return isinstance(wavelength_key, tuple) and len(wavelength_key) == 3

    def _selected_keys(self):
        """Yield selected scalar-wavelength keys for 2-D or broadband 3-D sims."""
        target_temps = set(self.target_temps)
        for simulation_key in self.sims:
            wavelength_key, polarization, temperature = simulation_key
            wavelengths = (
                np.linspace(
                    wavelength_key[0] - wavelength_key[1] / 2,
                    wavelength_key[0] + wavelength_key[1] / 2,
                    wavelength_key[2],
                )
                if isinstance(wavelength_key, tuple) and len(wavelength_key) == 3
                else (wavelength_key,)
            )
            for wavelength in wavelengths:
                if (
                    polarization == self.in_mode[:2]
                    and temperature in target_temps
                    and any(
                        math.isclose(wavelength, target, rel_tol=0, abs_tol=1e-4)
                        for target in self.target_wls
                    )
                ):
                    yield float(wavelength), polarization, temperature

    def _native_grid(self) -> dict[str, Any]:
        """Return validated 3-D coordinate metadata for solved fields."""
        metadata = self.grid_metadata
        if isinstance(metadata, RectilinearGridMetadata):
            metadata = metadata.to_dict()
        if isinstance(metadata, dict) and "coords" not in metadata:
            metadata = metadata.get("native")
        if not isinstance(metadata, dict) or not all(
            key in metadata for key in ("coords", "boundaries")
        ):
            raise ValueError(
                "3-D near-to-far projection requires native grid coordinates and boundaries"
            )
        if len(metadata["coords"]) != 3 or len(metadata["boundaries"]) != 3:
            raise ValueError("3-D near-to-far grid metadata must contain x/y/z axes")
        return metadata

    @staticmethod
    def _monitor_bounds(monitor_slice, shape: tuple[int, int, int]):
        """Return validated half-open bounds for a rectangular monitor plane."""
        bounds = []
        for axis, size in zip("xyz", shape):
            indexer = getattr(monitor_slice, axis)
            if isinstance(indexer, slice):
                start, stop, step = indexer.indices(size)
                if step != 1:
                    raise ValueError("near-field monitor slices must be contiguous")
            else:
                index = int(indexer)
                if index < 0:
                    index += size
                start, stop = index, index + 1
            if start < 0 or stop > size or stop <= start:
                raise ValueError(
                    f"near-field monitor has invalid {axis}-axis bounds {(start, stop)}"
                )
            bounds.append((start, stop))
        return tuple(bounds)

    def _surface_3d(self, field: dict[str, Tensor], wavelength_um: float):
        """Adapt registered 3-D near-field planes to one SI Huygens surface."""
        grid = self._native_grid()
        reference = field["Ex"]
        real_dtype = reference.real.dtype
        device = reference.device
        coordinates_m = tuple(
            torch.as_tensor(axis, dtype=real_dtype, device=device) * 1e-6
            for axis in grid["coords"]
        )
        boundaries_m = tuple(
            torch.as_tensor(axis, dtype=real_dtype, device=device) * 1e-6
            for axis in grid["boundaries"]
        )
        impedance_0 = math.sqrt(MU_0_SI / EPSILON_0)
        fields = torch.stack(
            (
                field["Ex"],
                field["Ey"],
                field["Ez"],
                field["Hx"] / impedance_0,
                field["Hy"] / impedance_0,
                field["Hz"] / impedance_0,
            ),
            dim=0,
        ).unsqueeze(0)
        frequency = torch.as_tensor(
            [C_0_SI / (wavelength_um * 1e-6)], dtype=real_dtype, device=device
        )
        surfaces = []
        for slice_name, info in self.port_slices_info.items():
            if not slice_name.startswith("nearfield"):
                continue
            direction = info["direction"]
            normal_axis = "xyz".index(direction[0])
            normal_sign = -1 if direction.endswith("-") else 1
            bounds = self._monitor_bounds(
                self.port_slices[slice_name], tuple(fields.shape[-3:])
            )
            if bounds[normal_axis][1] - bounds[normal_axis][0] != 1:
                raise ValueError("near-field monitor must occupy one normal-axis cell")
            selection = tuple(slice(start, stop) for start, stop in bounds)
            patch_fields = fields[(slice(None), slice(None), *selection)]
            patch_coordinates = tuple(
                axis[start:stop] for axis, (start, stop) in zip(coordinates_m, bounds)
            )
            patch_boundaries = tuple(
                axis[start : stop + 1]
                for axis, (start, stop) in zip(boundaries_m, bounds)
            )
            surfaces.append(
                from_maps_fields(
                    patch_fields,
                    frequencies_hz=frequency,
                    coordinates_m=patch_coordinates,
                    boundaries_m=patch_boundaries,
                    normal_axis=normal_axis,
                    normal_sign=normal_sign,
                    index=0,
                    colocated=True,
                    temporally_aligned=True,
                    unfolded=True,
                )
            )
        if not surfaces:
            raise ValueError(
                "No monitor whose name starts with 'nearfield' is registered"
            )
        return concatenate_surfaces(surfaces)

    def _observation_points_3d(
        self, reference: Tensor
    ) -> tuple[Tensor, tuple[int, int, int], int]:
        """Return flattened SI observation points and their planar layout."""
        info = self.port_slices_info[self.out_slice_name]
        axes = []
        for axis in "xyz":
            values = np.atleast_1d(np.asarray(info[f"{axis}s"], dtype=np.float64))
            axes.append(
                torch.as_tensor(
                    values, dtype=reference.real.dtype, device=reference.device
                )
                * 1e-6
            )
        grids = torch.meshgrid(*axes, indexing="ij")
        points = torch.stack(grids, dim=-1).reshape(-1, 3)
        normal_axis = "xyz".index(info["direction"][0])
        if axes[normal_axis].numel() != 1:
            raise ValueError("3-D near-to-far observations must form a plane")
        shape = tuple(int(axis.numel()) for axis in axes)
        return points, shape, normal_axis

    @staticmethod
    def _surface_fields(surface) -> dict[str, Tensor]:
        """Return all SI surface components with geometry and quadrature."""
        return {
            "positions_m": surface.positions_m,
            "normals": surface.normals,
            "weights_m2": surface.weights_m2,
            "Ex": surface.electric_fields[0, :, 0],
            "Ey": surface.electric_fields[0, :, 1],
            "Ez": surface.electric_fields[0, :, 2],
            "Hx": surface.magnetic_fields[0, :, 0],
            "Hy": surface.magnetic_fields[0, :, 1],
            "Hz": surface.magnetic_fields[0, :, 2],
        }

    @staticmethod
    def _projected_fields(
        positions_m: Tensor,
        electric: Tensor,
        magnetic: Tensor,
        grid_shape: tuple[int, int, int],
        normal_axis: int,
    ) -> dict[str, Tensor]:
        """Return projected components on the transverse observation grid."""
        positions = positions_m.reshape(*grid_shape, 3).select(normal_axis, 0)
        electric = electric[0].reshape(*grid_shape, 3).select(normal_axis, 0)
        magnetic = magnetic[0].reshape(*grid_shape, 3).select(normal_axis, 0)
        return {
            "positions_m": positions,
            "Ex": electric[..., 0],
            "Ey": electric[..., 1],
            "Ez": electric[..., 2],
            "Hx": magnetic[..., 0],
            "Hy": magnetic[..., 1],
            "Hz": magnetic[..., 2],
        }

    def __call__(self, fields):
        similarity_list = []
        raw_results = {}
        self.last_nearfield_fields = None
        self.last_projected_fields = None
        is_3d = self._is_fdtdx3d()
        for wavelength, polarization, temperature in self._selected_keys():
            result_key = (
                self.in_slice_name,
                wavelength,
                self.in_mode,
                temperature,
            )
            field = fields[result_key]
            if is_3d:
                surface = self._surface_3d(field, wavelength)
                surface_fields = self._surface_fields(surface)
                self.last_nearfield_fields = {
                    name: value.detach() if isinstance(value, Tensor) else value
                    for name, value in surface_fields.items()
                }
                if self.return_field == "nearfield":
                    raw_results[result_key] = surface_fields
                    continue
                points, grid_shape, normal_axis = self._observation_points_3d(
                    field["Ex"]
                )
                electric, magnetic = project(
                    surface,
                    points,
                    backend=self.projection_backend,
                    eps_r=self.eps_bg,
                )
                projected = self._projected_fields(
                    points, electric, magnetic, grid_shape, normal_axis
                )
                self.last_projected_fields = {
                    name: value.detach() if isinstance(value, Tensor) else value
                    for name, value in projected.items()
                }
                if self.return_field == "farfield":
                    raw_results[result_key] = projected
                    continue
                fz = projected[polarization]
            else:
                if self.return_field == "nearfield":
                    raw_results[result_key] = dict(field)
                    continue
                if polarization == "Ez":
                    fx_near, fy_near, fz_near = (
                        field["Hx"],
                        field["Hy"],
                        field["Ez"],
                    )
                elif polarization == "Hz":
                    fx_near, fy_near, fz_near = (
                        field["Ex"],
                        field["Ey"],
                        field["Hz"],
                    )
                else:
                    raise ValueError(f"Unsupported 2-D polarization {polarization!r}")
                farfield = get_farfields_GreenFunction(
                    nearfield_slices=[
                        self.port_slices[name]
                        for name in self.port_slices
                        if name.startswith("nearfield")
                    ],
                    nearfield_slices_info=[
                        self.port_slices_info[name]
                        for name in self.port_slices_info
                        if name.startswith("nearfield")
                    ],
                    Fz=fz_near[None, ..., None],
                    Fx=fx_near[None, ..., None],
                    Fy=fy_near[None, ..., None],
                    farfield_x=None,
                    farfield_slice_info=self.port_slices_info[self.out_slice_name],
                    freqs=torch.tensor([1 / wavelength], device=fz_near.device),
                    eps=self.eps_bg,
                    mu=MU_0,
                    dL=self.grid_step,
                    component=polarization,
                    decimation_factor=4,
                )
                if self.return_field == "farfield":
                    raw_results[result_key] = farfield
                    continue
                fz = farfield[polarization][0, ..., 0]
                monitor_slice = self.port_slices[self.out_slice_name]
                monitor_direction = self.port_slices_info[self.out_slice_name][
                    "direction"
                ]
                if len(monitor_slice.x.shape) <= 1 or len(monitor_slice.y.shape) <= 1:
                    fz = fz.reshape(1, -1)
                elif monitor_direction[0] == "y":
                    fz = fz.t()
            similarity_list.append(
                get_shape_similarity(
                    fz,
                    grid_step=self.grid_step,
                    shape_type=self.shape_type,
                    shape_cfg=self.shape_cfg,
                    intensity=self.intensity,
                    similarity=self.similarity,
                )
            )
        if self.return_field is not None:
            return raw_results
        if not similarity_list:
            raise ValueError("No fields matched the near-to-far objective selection")
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
                        eps_r = region_cfg.get("eps_r", region_cfg.get("eps", 12.0))

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
                        eps_r = region_cfg.get("eps_r", region_cfg.get("eps", 12.0))

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
                        eps_r = region_cfg.get("eps_r", region_cfg.get("eps", 12.0))

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
                        eps_r = region_cfg.get("eps_r", region_cfg.get("eps", 12.0))

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


class ErosionDilationPenalty(nn.Module):
    """Differentiable erosion-dilation manufacturing penalty module.

    Enforces minimum feature size and minimum curvature radius constraints
    on photonic topology design regions by penalizing the RMS difference
    between morphological closing (dilated-eroded) and opening (eroded-dilated)
    occupancy fields.
    """

    def __init__(
        self,
        length_scale: float = 0.15,
        planes: str | Sequence[str] = ("middle",),
        plane_weights: dict[str, float] | Sequence[float] | None = None,
        aggregation: str = "mean",
        beta: float = 100.0,
        eta: float = 0.5,
        delta_eta: float = 0.01,
        evaluation_band: float = 0.0,
        evaluation_band_px: int = 0,
        extra_halo_px: int = 0,
        operator_mode: str = "tidy3d",
        sims: dict | None = None,
        device: Any | None = None,
        design_region_masks: dict | None = None,
        design_region_cfgs: dict | None = None,
        design_region_param_cfgs: dict | None = None,
        design_region_axis_weights: dict | None = None,
        grid_metadata: RectilinearGridMetadata | dict[str, Any] | None = None,
        grid_step: float | None = None,
        pixel_size: float | Sequence[float] | None = None,
        region_names: str | Sequence[str] | None = None,
        weight: float = 1.0,
        eta0: float | None = None,
        eps_core: complex | float | None = None,
        eps_clad: complex | float | None = None,
        objective_func: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.objective_func = objective_func
        self.length_scale = float(length_scale)
        if isinstance(planes, str):
            self.planes = (planes,)
        else:
            self.planes = tuple(planes)
        self.plane_weights = plane_weights
        self.aggregation = str(aggregation)
        self.beta = float(beta)
        self.eta = float(eta0 if eta0 is not None else eta)
        self.delta_eta = float(delta_eta)
        self.evaluation_band = float(evaluation_band)
        self.evaluation_band_px = int(evaluation_band_px)
        self.extra_halo_px = int(extra_halo_px)
        self.operator_mode = str(operator_mode)

        self.sims = sims or {}
        self.device = device
        self.design_region_masks = design_region_masks
        self.design_region_cfgs = design_region_cfgs or (
            getattr(device, "design_region_cfgs", None) or {}
        )
        self.design_region_param_cfgs = design_region_param_cfgs or (
            getattr(device, "design_region_param_cfgs", None) or {}
        )
        self.design_region_axis_weights = design_region_axis_weights or (
            getattr(device, "design_region_axis_weights", None) or {}
        )
        self.grid_metadata = grid_metadata
        self.grid_step = grid_step
        self._explicit_pixel_size = pixel_size is not None
        self.pixel_size = pixel_size if pixel_size is not None else grid_step
        if region_names is not None:
            self.region_names = (
                (region_names,)
                if isinstance(region_names, str)
                else tuple(region_names)
            )
        else:
            self.region_names = None
        self.weight = float(weight)
        self.eps_core = eps_core
        self.eps_clad = eps_clad

        self._context_cache: dict[tuple, dict[str, Any]] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.last_plane_penalties: dict[str, Any] = {}
        self.last_diagnostics: dict[str, Any] = {}

    def clear_cache(self) -> None:
        """Clear cached simulation context tensors and reset hit/miss counters."""
        self._context_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def _resolve_context_key(
        self,
        region_name: str | None,
        plane_name: str,
        sim_shape: tuple[int, ...],
        rx: int,
        ry: int,
        eval_bounds: tuple[int, int, int, int],
        extra_halo_px: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple:
        return (
            str(region_name),
            str(plane_name),
            tuple(sim_shape[-2:]),
            int(rx),
            int(ry),
            tuple(eval_bounds),
            int(extra_halo_px),
            str(device),
            str(dtype),
        )

    def evaluate(
        self,
        x: Tensor,
        pixel_size: float | Sequence[float] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Evaluate penalty on input tensor x (e.g. parameter or permittivity map)."""
        px = pixel_size if pixel_size is not None else self.pixel_size
        return self(epsilon_map=x, pixel_size=px, **kwargs)

    def forward(
        self,
        epsilon_map: Tensor | None = None,
        fields: dict | None = None,
        rho: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Evaluate erosion-dilation penalty across configured regions and planes."""
        target_eps = epsilon_map
        if target_eps is None and rho is None:
            if hasattr(self, "eps") and self.eps is not None:
                target_eps = self.eps
            elif (
                hasattr(self, "objective_func")
                and self.objective_func is not None
                and getattr(self.objective_func, "eps", None) is not None
            ):
                target_eps = self.objective_func.eps
            elif (
                self.device is not None
                and getattr(self.device, "epsilon_map", None) is not None
            ):
                target_eps = self.device.epsilon_map
            elif len(self.sims) > 0:
                first_sim = next(iter(self.sims.values()))
                target_eps = getattr(first_sim, "eps_r", None)

        masks = self.design_region_masks
        if masks is None and self.device is not None:
            masks = getattr(self.device, "design_region_masks", None)

        if self.region_names is not None:
            eval_regions = list(self.region_names)
        elif masks is not None and len(masks) > 0:
            eval_regions = list(masks.keys())
        else:
            eval_regions = [None]

        px = kwargs.get("pixel_size", self.pixel_size)
        if px is None and self.device is not None:
            px = getattr(self.device, "grid_step", 0.02)
        if px is None:
            px = 0.02
        dl_x = float(px if isinstance(px, (int, float)) else px[0])
        dl_y = float(px if isinstance(px, (int, float)) else px[1])
        rx, ry = get_ed_kernel_radius_px(self.length_scale, (dl_x, dl_y))

        region_penalties = []
        diag_by_region = {}

        for r_name in eval_regions:
            mask = (
                masks.get(r_name)
                if (masks is not None and r_name is not None)
                else kwargs.get("design_region_mask", None)
            )
            cfg = self.design_region_cfgs.get(r_name, {}) if r_name is not None else {}
            param_cfg = (
                self.design_region_param_cfgs.get(r_name, {})
                if r_name is not None
                else {}
            )

            eps_core = kwargs.get("eps_core", self.eps_core)
            if eps_core is None:
                eps_core = cfg.get("eps")
            eps_clad = kwargs.get("eps_clad", self.eps_clad)
            if eps_clad is None:
                eps_clad = cfg.get("eps_bg")
            if eps_clad is None and self.device is not None:
                eps_clad = getattr(self.device, "eps_bg", None)

            fixed_map = kwargs.get("fixed_epsilon_map", None)
            if fixed_map is None and self.device is not None:
                fixed_map = getattr(self.device, "epsilon_map", None)

            axis_weights = self.design_region_axis_weights.get(r_name, None)
            grid_meta = kwargs.get("grid_metadata", self.grid_metadata)
            if grid_meta is None and self.device is not None:
                grid_meta = getattr(self.device, "optical_grid_metadata", None)
                if grid_meta is None and hasattr(self.device, "grid_info_dict"):
                    grid_meta = self.device.grid_info_dict.get("epsilon_map", None)

            eval_band = kwargs.get("evaluation_band", self.evaluation_band)
            eval_band_px = kwargs.get("evaluation_band_px", self.evaluation_band_px)
            extra_halo = kwargs.get("extra_halo_px", self.extra_halo_px)
            op_mode = kwargs.get("operator_mode", self.operator_mode)
            use_cache = kwargs.get("use_cache", True)
            planes_arg = kwargs.get("planes", self.planes)
            plane_weights_arg = kwargs.get("plane_weights", self.plane_weights)
            agg_arg = kwargs.get("aggregation", self.aggregation)
            length_scale_arg = kwargs.get("length_scale", self.length_scale)
            beta_arg = kwargs.get("beta", self.beta)
            eta_arg = kwargs.get("eta", self.eta)
            delta_eta_arg = kwargs.get("delta_eta", self.delta_eta)
            rho_coords_arg = kwargs.get("rho_coords", None)

            cache_stats = {"hits": 0, "misses": 0}
            cache_prefix = (
                str(r_name),
                id(fixed_map) if fixed_map is not None else 0,
            )

            pen, details = erosion_dilation_penalty(
                epsilon_map=target_eps,
                rho=rho,
                fixed_epsilon_map=fixed_map,
                design_region_mask=mask,
                eps_core=eps_core,
                eps_clad=eps_clad,
                length_scale=length_scale_arg,
                pixel_size=px,
                planes=planes_arg,
                plane_weights=plane_weights_arg,
                aggregation=agg_arg,
                beta=beta_arg,
                eta=eta_arg,
                delta_eta=delta_eta_arg,
                axis_weights=axis_weights,
                evaluation_band=eval_band,
                evaluation_band_px=eval_band_px,
                extra_halo_px=extra_halo,
                operator_mode=op_mode,
                grid_metadata=grid_meta,
                design_region_cfg=cfg if cfg else kwargs.get("design_region_cfg", None),
                design_region_param_cfg=(
                    param_cfg
                    if (
                        param_cfg
                        and not self._explicit_pixel_size
                        and "pixel_size" not in kwargs
                    )
                    else kwargs.get("design_region_param_cfg", None)
                ),
                rho_coords=rho_coords_arg,
                return_details=True,
                _context_cache=self._context_cache if use_cache else None,
                _cache_key_prefix=cache_prefix if use_cache else None,
                _cache_stats=cache_stats if use_cache else None,
            )

            self.cache_hits += cache_stats["hits"]
            self.cache_misses += cache_stats["misses"]

            region_penalties.append(pen)
            diag_by_region[r_name or "default"] = details

        if len(region_penalties) == 1:
            total_penalty = region_penalties[0]
            self.last_plane_penalties = next(iter(diag_by_region.values()))[
                "plane_penalties"
            ]
        else:
            total_penalty = torch.stack(region_penalties).mean()
            self.last_plane_penalties = {
                f"{r}_{p}": d["plane_penalties"][p]
                for r, d in diag_by_region.items()
                for p in d["plane_penalties"]
            }

        self.last_diagnostics = diag_by_region
        return total_penalty


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
        design_region_param_cfgs: dict = None,
        design_region_axis_weights: dict = None,
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
        self.design_region_param_cfgs = (
            design_region_param_cfgs
            if design_region_param_cfgs is not None
            else getattr(device, "design_region_param_cfgs", None)
        )
        self.design_region_axis_weights = (
            design_region_axis_weights
            if design_region_axis_weights is not None
            else getattr(device, "design_region_axis_weights", None)
        )

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
            target_temps = cfg.get("control_keys", cfg.get("temp", (300,)))
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
                    energy=cfg.get("energy", False),
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

            elif obj_type == "target_intensity":
                objfn = TargetIntensityObjective(
                    sims=self.sims,
                    port_slices=self.port_slices,
                    in_slice_name=in_slice_name,
                    out_slice_name=out_slice_name,
                    in_mode=in_mode,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    target_intensity=cfg["target_intensity"],
                    field_components=cfg.get("field_components", ("Ex", "Ey", "Ez")),
                )

            elif obj_type == "target_intensity_angular_spectrum":
                objfn = TargetIntensityAngularSpectrumObjective(
                    sims=self.sims,
                    port_slices=self.port_slices,
                    in_slice_name=in_slice_name,
                    out_slice_name=out_slice_name,
                    in_mode=in_mode,
                    target_wls=target_wls,
                    target_temps=target_temps,
                    target_intensity=cfg["target_intensity"],
                    field_components=cfg.get("field_components", ("Ex", "Ey", "Ez")),
                    projection_distance_um=cfg["projection_distance_um"],
                    grid_step_um=self.grid_step,
                    eps_bg=self.eps_bg,
                    evanescent=cfg.get("evanescent", "decay"),
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
                    return_field=cfg.get("return_field"),
                    projection_backend=cfg.get("projection_backend", "direct_torch"),
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
            elif obj_type == "erosion_dilation_penalty":
                objfn = ErosionDilationPenalty(
                    sims=self.sims,
                    device=self.device,
                    grid_step=self.grid_step,
                    pixel_size=cfg.get("pixel_size", None),
                    grid_metadata=cfg.get("grid_metadata", self.grid_metadata),
                    design_region_masks=self.design_region_masks,
                    design_region_cfgs=self.design_region_cfgs,
                    design_region_param_cfgs=self.design_region_param_cfgs,
                    design_region_axis_weights=self.design_region_axis_weights,
                    length_scale=cfg.get("length_scale", 0.15),
                    planes=cfg.get("planes", ("middle",)),
                    plane_weights=cfg.get("plane_weights", None),
                    aggregation=cfg.get("aggregation", "mean"),
                    beta=cfg.get("beta", 100.0),
                    eta=cfg.get("eta", cfg.get("eta0", 0.5)),
                    delta_eta=cfg.get("delta_eta", 0.01),
                    evaluation_band=cfg.get("evaluation_band", 0.0),
                    evaluation_band_px=cfg.get("evaluation_band_px", 0),
                    operator_mode=cfg.get("operator_mode", "tidy3d"),
                    extra_halo_px=cfg.get("extra_halo_px", 0),
                    region_names=cfg.get("region_names", None),
                    weight=cfg.get("weight", 1.0),
                    objective_func=self,
                )
            else:
                raise ValueError("Invalid type")

            ### note that this is not the final objective! this is partial objective from fields to fom
            ### complete autograd graph is from permittivity (np.ndarray) to fields and to fom
            self.Js[name] = {
                "weight": cfg["weight"],
                "fn": objfn,
                "requires_grad": cfg.get("requires_grad", True),
                "include_in_fusion": cfg.get(
                    "include_in_fusion",
                    not (
                        obj_type == "intensity_shape_near2far"
                        and cfg.get("return_field") is not None
                    ),
                ),
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
        field_grid_metadata = getattr(sim, "optical_grid_metadata", None)
        if (
            electrical_conductivity is not None
            and isinstance(field_grid_metadata, dict)
            and "native" in field_grid_metadata
        ):
            electrical_conductivity = self.device._resample_map_to_fdtdx_native(
                electrical_conductivity,
                map_name="electrical_conductivity_map",
                native_grid=field_grid_metadata["native"],
            )
        if (
            electrical_conductivity is not None
            or getattr(sim, "electrical_conductivity_map", None) is None
        ):
            sim.electrical_conductivity_map = electrical_conductivity

    def obtain_objective(
        self, permittivity: np.ndarray | Tensor, custom_source: dict = None
    ) -> Tuple[dict, Tensor]:
        self.eps = permittivity
        self.solutions = {}
        self.As = {}
        self._runtime_state_cache = {}
        self.runtime_thermal_states = {}
        temperatures = []
        torch.cuda.empty_cache()
        for _, cfg in self.obj_cfgs.items():
            temps = cfg.get("control_keys", cfg.get("temp", [300]))
            if isinstance(temps, (int, float)):
                temps = [temps]
            temperatures = temperatures + list(temps)
        temperatures = set(temperatures)
        if custom_source is None:
            ### we only refactor the A matrix if A changes, i.e., different wl, eps, pol.
            ### If only source/mode changes, the A keeps the same, we do not need to refactor the matrix, we can reuse the pardiso solver.
            ### so here we need to reorder the simulation and cluster them into groups, so I can reuse the psolve if they have the same (temp, pol, and wl)
            _simulation_queues = {}

            for slice_name, port_profile in self.port_profiles.items():
                if slice_name.endswith("_field_monitor"):
                    continue
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
        fusion_breakdown = {
            name: value
            for name, value in self.breakdown.items()
            if self.Js[name].get("include_in_fusion", True)
        }
        fusion_results = self._obj_fusion_func(fusion_breakdown)
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
