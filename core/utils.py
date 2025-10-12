import collections
import logging
import math
import random
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Tuple

import autograd.numpy as npa
import matplotlib.pyplot as plt
import numpy as np
import ryaml
import torch
import torch.distributed as dist
import torch.fft
import torch.nn.functional as F
import torch.optim
import yaml
from pyutils.config import Config
from pyutils.general import TimerCtx
from torch import Tensor
from torch.types import Device
from torch_sparse import spmm

train_configs = Config()
from thirdparty.ceviche.constants import *

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


# def get_shape(shape_type, shape_cfg, field_size, grid_step):
#     shape_cfg["size"] = field_size
#     shape_cfg["grid_step"] = grid_step
#     return shape_dict[shape_type](**shape_cfg)


@lru_cache(maxsize=8)
def _gaussian(grid_step, size, width, device="cuda"):
    x_mesh = torch.linspace(
        -size * grid_step / 2, size * grid_step / 2, size, device=device
    )

    return torch.exp((x_mesh**2) / (-2 * width**2))


def gaussian(device="cuda", **kwargs):
    """Generate a gaussian shape.

    Parameters
    ----------
    x : torch.Tensor
        x coordinates.
    y : torch.Tensor
        y coordinates.
    sigma : float
        standard deviation of the gaussian.

    Returns
    -------
    torch.Tensor
        2D gaussian shape.
    """
    grid_step = kwargs["grid_step"]
    size = kwargs["size"]
    width = kwargs["width"]
    return _gaussian(grid_step, size, width, device=device)


shape_dict = {
    "gaussian": gaussian,
}


@torch.compile
# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def trapezoid_1d(integrand: Tensor, dx: float, dim: int = -1) -> Tensor:
    return torch.sum(integrand, dim=dim) * dx  # more efficient for uniform grid
    # slice1 = [slice(None)] * integrand.ndim
    # slice2 = [slice(None)] * integrand.ndim
    # slice1[dim] = slice(1, None)
    # slice2[dim] = slice(None, -1)
    # return torch.sum(
    #     (integrand[tuple(slice1)] + integrand[tuple(slice2)]), dim=dim
    # ) * (dx / 2)


def sph_2_car_field(
    f_r: Tensor,
    f_phi: Tensor,
    f_theta: Tensor,
    phi: Tensor,
    theta: Tensor = None,
    sin_phi: Tensor = None,
    cos_phi: Tensor = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert vector field components in spherical coordinates to cartesian.

    Parameters
    ----------
    f_r : float
        radial component of the vector field.
    f_theta : float
        polar angle component of the vector fielf.
    f_phi : float
        azimuthal angle component of the vector field.
    theta : float
        polar angle (rad) of location of the vector field.
    phi : float
        azimuthal angle (rad) of location of the vector field.

    Returns
    -------
    Tuple[float, float, float]
        x, y, and z components of the vector field in cartesian coordinates.
    """
    if sin_phi is None:
        sin_phi = torch.sin(phi)
    if cos_phi is None:
        cos_phi = torch.cos(phi)

    if theta is None:
        f_x = f_r * cos_phi - f_phi * sin_phi
        f_y = f_r * sin_phi + f_phi * cos_phi
        f_z = -f_theta
    else:
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        f_x = (
            f_r * sin_theta * cos_phi + f_theta * cos_theta * cos_phi - f_phi * sin_phi
        )
        f_y = (
            f_r * sin_theta * sin_phi + f_theta * cos_theta * sin_phi + f_phi * cos_phi
        )
        f_z = f_r * cos_theta - f_theta * sin_theta
    return f_x, f_y, f_z


@torch.compile
def car_2_sph(x: float, y: float, z: float = None) -> Tuple[float, float, float]:
    """Convert Cartesian to spherical coordinates.

    Parameters
    ----------
    x : float
        x coordinate relative to ``local_origin``.
    y : float
        y coordinate relative to ``local_origin``.
    z : float
        z coordinate relative to ``local_origin``.

    Returns
    -------
    Tuple[float, float, float]
        r, theta, and phi coordinates relative to ``local_origin``.
    """
    if z is None:
        r = x.square().add(y.square()).sqrt_()
        theta = None
    else:
        r = torch.sqrt(x.square() + y.square() + z.square())
        theta = torch.acos(z / r)
    phi = torch.arctan2(y, x)
    return r, phi, theta


@torch.compile
def sph_2_car(r: float, phi: float, theta: float = None) -> Tuple[float, float, float]:
    """Convert spherical to Cartesian coordinates.

    Parameters
    ----------
    r : float
        radius.
    theta : float
        polar angle (rad) downward from x=y=0 line.
    phi : float
        azimuthal (rad) angle from y=z=0 line.

    Returns
    -------
    Tuple[float, float, float]
        x, y, and z coordinates relative to ``local_origin``.
    """
    if theta is None:
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = None
    else:
        r_sin_theta = r * torch.sin(theta)
        x = r_sin_theta * torch.cos(phi)
        y = r_sin_theta * torch.sin(phi)
        z = r * torch.cos(theta)
    return x, y, z


def overlap(
    a,
    b,
    dl,
    direction: str = "x",
) -> float:
    """Numerically compute the overlap integral of two VectorFields.

    Args:
      a: `VectorField` specifying the first field.
      b: `VectorField` specifying the second field.
      normal: `Direction` specifying the direction normal to the plane (or slice)
        where the overlap is computed.

    Returns:
      Result of the overlap integral.
    """
    if any(isinstance(ai, torch.Tensor) for ai in a):
        conj = torch.conj
        sum = torch.sum
    else:
        conj = npa.conj
        sum = npa.sum
    # ac = tuple([conj(ai) for ai in a])
    ac = []
    for ai in a:
        if isinstance(ai, (torch.Tensor, np.ndarray)):
            ac.append(conj(ai))
        else:
            ac.append(npa.conj(ai))
    ac = tuple(ac)

    return sum(cross(ac, b, direction=direction)) * dl


def cross(a, b, direction="x"):
    """Compute the cross product between two VectorFields."""
    if direction[0] == "x":
        return a[1] * b[2] - a[2] * b[1]
    elif direction[0] == "y":
        return a[2] * b[0] - a[0] * b[2]
    elif direction[0] == "z":
        return a[0] * b[1] - a[1] * b[0]
    else:
        raise ValueError("Invalid direction")


def get_eigenmode_coefficients(
    hx,
    hy,
    ez,
    ht_m,
    et_m,
    monitor,
    grid_step,
    direction: str = "x",
    autograd=False,
    energy=False,
    pol: str = "Ez",
):
    ### for Ez polarization: hx, hy, ez, ht_m is hx or hy, et_m is ez
    ### for Hx polarization: ex, ey, hz, ht_m is hz, et_m is ex or ey
    if isinstance(ht_m, np.ndarray) and isinstance(hx, torch.Tensor):
        ht_m = torch.from_numpy(ht_m).to(ez.device)
        et_m = torch.from_numpy(et_m).to(ez.device)
    if autograd:
        abs = npa.abs
        ravel = npa.ravel
    else:
        abs = np.abs
        ravel = np.ravel
    if isinstance(ez, torch.Tensor):
        abs = torch.abs
        ravel = torch.ravel

    if direction[0] == "x":
        h = (0.0, ravel(hy[monitor]), 0)
        if pol == "Ez":
            hm = (0.0, ht_m, 0.0)
            em = (0.0, 0.0, et_m)
        elif pol == "Hz":
            hm = (0.0, 0.0, ht_m)
            em = (0.0, -et_m, 0.0)
        # The E-field is not co-located with the H-field in the Yee cell. Therefore,
        # we must sample at two neighboring pixels in the propataion direction and
        # then interpolate:
        e_yee_shifted = grid_average(ez, monitor, direction, autograd=autograd, pol=pol)

    elif direction[0] == "y":
        h = (ravel(hx[monitor]), 0, 0)
        if pol == "Ez":
            hm = (-ht_m, 0.0, 0.0)
            em = (0.0, 0.0, et_m)
        elif pol == "Hz":
            hm = (0.0, 0.0, ht_m)
            em = (et_m, 0.0, 0.0)
        # The E-field is not co-located with the H-field in the Yee cell. Therefore,
        # we must sample at two neighboring pixels in the propataion direction and
        # then interpolate:
        e_yee_shifted = grid_average(ez, monitor, direction, autograd=autograd, pol=pol)

    e = (0.0, 0.0, e_yee_shifted)

    if pol == "Hz":  # swap e and h
        e, h = h, e

    # print("this is the type of em: ", type(em[2])) # ndarray
    # print("this is the type of hy: ", type(hy[monitor])) # torch.Tensor

    dl = grid_step * MICRON_UNIT
    overlap1 = overlap(em, h, dl=dl, direction=direction)
    overlap2 = overlap(hm, e, dl=dl, direction=direction)
    normalization = overlap(em, hm, dl=dl, direction=direction)
    normalization = (2 * normalization) ** 0.5
    s_p = (overlap1 + overlap2) / 2 / normalization
    s_m = (overlap1 - overlap2) / 2 / normalization

    if energy:
        s_p = abs(s_p) ** 2
        s_m = abs(s_m) ** 2

    return s_p, s_m


def cal_fom_from_fields(
    Ez,
    Hx,
    Hy,
    ht_m,
    et_m,
    monitor_out,
    monitor_refl,
):
    fwd_trans, _ = get_eigenmode_coefficients(
        hx=Hx,
        hy=Hy,
        ez=Ez,
        ht_m=ht_m,
        et_m=et_m,
        monitor=monitor_out,
        grid_step=1 / 50,
        direction="y",
        energy=True,
    )
    fwd_trans = fwd_trans / 1e-8  # normalize with the input power

    refl_fom = get_flux(
        hx=Hx,
        hy=Hy,
        ez=Ez,
        monitor=monitor_refl,
        grid_step=1 / 50,
        direction="x",
    )
    refl_fom = torch.abs(
        torch.abs(refl_fom / 1e-8) - 1
    )  # normalize with the input power

    total_fom = -fwd_trans + 0.1 * refl_fom

    return total_fom


@lru_cache(maxsize=1024)
def _load_opt_cfgs(opt_cfg_file_path):
    with open(opt_cfg_file_path, "r") as opt_cfg_file:
        opt_cfgs = ryaml.load(opt_cfg_file)
        resolution = opt_cfgs["sim_cfg"]["resolution"]
        opt_cfgs = opt_cfgs["obj_cfgs"]
        # fusion_fn = opt_cfgs["_fusion_func"]
        del opt_cfgs["_fusion_func"]
    return opt_cfgs, resolution


def cal_fom_from_fwd_field(
    Ez4adj,
    Ez4fullfield,
    eps,
    ht_ms,
    et_ms,
    monitors,
    sim,
    opt_cfg_file_path,
    wl,
    mode,
    temp,
    src_in_slice_name,
):
    Ez4fullfield = Ez4fullfield.permute(0, 2, 3, 1).contiguous()
    Ez4fullfield = torch.view_as_complex(Ez4fullfield)
    Ez4adj = Ez4adj.permute(0, 2, 3, 1).contiguous()
    Ez4adj = torch.view_as_complex(Ez4adj)
    monitor_slice_list = []
    total_fom_list = []
    for i in range(Ez4adj.shape[0]):
        Ez4adj_i = Ez4adj[i].requires_grad_()
        Ez4fullfield_i = Ez4fullfield[i]
        sim[wl[i].item()].eps_r = eps[i]
        Hx4adj_vec, Hy4adj_vec = sim[wl[i].item()]._Ez_to_Hx_Hy(Ez4adj_i.flatten())
        Hx4adj_i = Hx4adj_vec.reshape(Ez4adj_i.shape)
        Hy4adj_i = Hy4adj_vec.reshape(Ez4adj_i.shape)
        if torch.equal(Ez4fullfield_i, Ez4adj_i):
            Hx4fullfield_i = Hx4adj_i
            Hy4fullfield_i = Hy4adj_i
        else:
            Hx4fullfield_vec, Hy4fullfield_vec = sim[wl[i].item()]._Ez_to_Hx_Hy(
                Ez4fullfield_i.flatten()
            )
            Hx4fullfield_i = Hx4fullfield_vec.reshape(Ez4fullfield_i.shape)
            Hy4fullfield_i = Hy4fullfield_vec.reshape(Ez4fullfield_i.shape)

        total_fom = torch.tensor(0.0, device=Ez4fullfield_i.device)
        opt_cfgs, resolution = _load_opt_cfgs(opt_cfg_file_path[i])

        for obj_name, opt_cfg in opt_cfgs.items():
            weight = float(opt_cfg["weight"])
            direction = opt_cfg["direction"]
            out_slice_name = opt_cfg["out_slice_name"]
            in_mode = opt_cfg["in_mode"]
            out_modes = opt_cfg.get("out_modes", [])
            out_modes = [int(mode) for mode in out_modes]
            assert (
                len(out_modes) == 1
            ), f"The code can handle multiple modes, but I have not check if it is correct"
            temperture = opt_cfg["temp"]
            temperture = [float(temp) for temp in temperture]
            wavelength = opt_cfg["wl"]
            assert (
                len(wavelength) == 1
            ), f"only support one wavelength for now but the wavelength is: {wavelength}"
            wavelength = wavelength[0]
            input_slice_name = opt_cfg["in_slice_name"]
            obj_type = opt_cfg["type"]
            monitor = Slice(
                x=monitors[f"port_slice-{out_slice_name}_x"][i],
                y=monitors[f"port_slice-{out_slice_name}_y"][i],
            )
            # print(f"this is the wl: {wl}, mode: {mode}, temp: {temp}, in_port_name: {in_port_name}")
            # print(f"this is the corresponding we read from current obj mode: {in_mode}, temp: {temperture}, in_port_name: {input_port_name}")
            if (
                weight == 0
                or in_mode != int(mode[i].item())
                or temp[i] not in temperture
                or input_slice_name != src_in_slice_name[i]
                or wavelength != float(wl[i].item())
            ):
                continue
            monitor_slice_list.append(
                monitor
            )  # add the monitor to the list so that we can later calculate the incident light field
            if obj_type == "eigenmode":
                fom = torch.tensor(0.0, device=Ez4fullfield_i.device)
                for output_mode in out_modes:
                    fom_inner, _ = get_eigenmode_coefficients(
                        hx=Hx4adj_i,
                        hy=Hy4adj_i,
                        ez=Ez4adj_i,
                        ht_m=ht_ms[
                            f"ht_m-wl-{float(wl[i].item())}-slice-{out_slice_name}-mode-{output_mode}"
                        ][i],
                        et_m=et_ms[
                            f"et_m-wl-{float(wl[i].item())}-slice-{out_slice_name}-mode-{output_mode}"
                        ][i],
                        monitor=monitor,
                        grid_step=1 / resolution,
                        direction=direction,
                        energy=True,
                    )
                    fom_inner = (
                        weight * fom_inner / 1e-8
                    )  # normalize with the input power
                    fom = fom + fom_inner
            elif "flux" in obj_type:  # flux or flux_minus_src
                fom = get_flux(
                    hx=Hx4adj_i,
                    hy=Hy4adj_i,
                    ez=Ez4adj_i,
                    monitor=monitor,
                    grid_step=1 / resolution,
                    direction=direction,
                )
                if "minus_src" in opt_cfg["type"]:
                    fom = torch.abs(torch.abs(fom / 1e-8) - 1)
                else:
                    fom = torch.abs(fom / 1e-8)
                fom = weight * fom
            else:
                raise ValueError(f"Unknown optimization type: {opt_cfg['type']}")
            total_fom = total_fom + fom
        total_fom_list.append(total_fom * (-1))
    return torch.stack(total_fom_list, dim=0)


def cal_total_field_adj_src_from_fwd_field(
    Ez4adj,
    Ez4fullfield,
    eps,
    ht_ms,
    et_ms,
    monitors,
    pml_mask,
    return_adj_src,
    sim,
    opt_cfg_file_path,
    wl,
    mode,
    temp,
    src_in_slice_name,
) -> Tensor:
    Ez4fullfield_copy = Ez4fullfield
    Ez4fullfield = Ez4fullfield.permute(0, 2, 3, 1).contiguous()
    Ez4fullfield = torch.view_as_complex(Ez4fullfield)
    Ez4adj = Ez4adj.permute(0, 2, 3, 1).contiguous()
    Ez4adj = torch.view_as_complex(Ez4adj)
    Hx_list = []
    Hy_list = []
    if not return_adj_src:
        for i in range(Ez4fullfield.shape[0]):
            sim[wl[i].item()].eps_r = eps[i]
            Hx_vec, Hy_vec = sim[wl[i].item()]._Ez_to_Hx_Hy(Ez4fullfield[i].flatten())
            Hx_i = Hx_vec.reshape(Ez4fullfield[i].shape)
            Hy_i = Hy_vec.reshape(Ez4fullfield[i].shape)

            Hx_to_append = torch.view_as_real(Hx_i).permute(2, 0, 1)
            Hy_to_append = torch.view_as_real(Hy_i).permute(2, 0, 1)

            Hx_list.append(Hx_to_append)
            Hy_list.append(Hy_to_append)
        Hx = torch.stack(Hx_list, dim=0)
        Hy = torch.stack(Hy_list, dim=0)
        total_field = torch.cat((Hx, Hy, Ez4fullfield_copy), dim=1)
        total_field = pml_mask.unsqueeze(0).unsqueeze(0) * total_field
        return total_field, None, None
    else:
        gradient_list = []
        monitor_slice_list = []
        for i in range(Ez4adj.shape[0]):
            Ez4adj_i = Ez4adj[i].requires_grad_()
            Ez4fullfield_i = Ez4fullfield[i]
            sim[wl[i].item()].eps_r = eps[i]
            Hx4adj_vec, Hy4adj_vec = sim[wl[i].item()]._Ez_to_Hx_Hy(Ez4adj_i.flatten())
            Hx4adj_i = Hx4adj_vec.reshape(Ez4adj_i.shape)
            Hy4adj_i = Hy4adj_vec.reshape(Ez4adj_i.shape)
            if torch.equal(Ez4fullfield_i, Ez4adj_i):
                Hx4fullfield_i = Hx4adj_i
                Hy4fullfield_i = Hy4adj_i
            else:
                Hx4fullfield_vec, Hy4fullfield_vec = sim[wl[i].item()]._Ez_to_Hx_Hy(
                    Ez4fullfield_i.flatten()
                )
                Hx4fullfield_i = Hx4fullfield_vec.reshape(Ez4fullfield_i.shape)
                Hy4fullfield_i = Hy4fullfield_vec.reshape(Ez4fullfield_i.shape)

            Hx_to_append = torch.view_as_real(Hx4fullfield_i).permute(2, 0, 1)
            Hy_to_append = torch.view_as_real(Hy4fullfield_i).permute(2, 0, 1)

            Hx_list.append(Hx_to_append)
            Hy_list.append(Hy_to_append)

            total_fom = torch.tensor(0.0, device=Ez4fullfield_i.device)
            opt_cfgs, resolution = _load_opt_cfgs(opt_cfg_file_path[i])

            for obj_name, opt_cfg in opt_cfgs.items():
                weight = float(opt_cfg["weight"])
                direction = opt_cfg["direction"]
                out_slice_name = opt_cfg["out_slice_name"]
                in_mode = opt_cfg["in_mode"]
                out_modes = opt_cfg.get("out_modes", [])
                out_modes = [int(mode) for mode in out_modes]
                assert (
                    len(out_modes) == 1
                ), f"The code can handle multiple modes, but I have not check if it is correct"
                temperture = opt_cfg["temp"]
                temperture = [float(temp) for temp in temperture]
                wavelength = opt_cfg["wl"]
                assert (
                    len(wavelength) == 1
                ), f"only support one wavelength for now but the wavelength is: {wavelength}"
                wavelength = wavelength[0]
                input_slice_name = opt_cfg["in_slice_name"]
                obj_type = opt_cfg["type"]
                monitor = Slice(
                    x=monitors[f"port_slice-{out_slice_name}_x"][i],
                    y=monitors[f"port_slice-{out_slice_name}_y"][i],
                )
                # print(f"this is the wl: {wl}, mode: {mode}, temp: {temp}, in_port_name: {in_port_name}")
                # print(f"this is the corresponding we read from current obj mode: {in_mode}, temp: {temperture}, in_port_name: {input_port_name}")
                if (
                    weight == 0
                    or in_mode != int(mode[i].item())
                    or temp[i] not in temperture
                    or input_slice_name != src_in_slice_name[i]
                    or wavelength != float(wl[i].item())
                ):
                    continue
                monitor_slice_list.append(
                    monitor
                )  # add the monitor to the list so that we can later calculate the incident light field
                if obj_type == "eigenmode":
                    fom = torch.tensor(0.0, device=Ez4fullfield_i.device)
                    for output_mode in out_modes:
                        fom_inner, _ = get_eigenmode_coefficients(
                            hx=Hx4adj_i,
                            hy=Hy4adj_i,
                            ez=Ez4adj_i,
                            ht_m=ht_ms[
                                f"ht_m-wl-{float(wl[i].item())}-slice-{out_slice_name}-mode-{output_mode}"
                            ][i],
                            et_m=et_ms[
                                f"et_m-wl-{float(wl[i].item())}-slice-{out_slice_name}-mode-{output_mode}"
                            ][i],
                            monitor=monitor,
                            grid_step=1 / resolution,
                            direction=direction,
                            energy=True,
                        )
                        fom_inner = (
                            weight * fom_inner / 1e-8
                        )  # normalize with the input power
                        fom = fom + fom_inner
                elif "flux" in obj_type:  # flux or flux_minus_src
                    fom = get_flux(
                        hx=Hx4adj_i,
                        hy=Hy4adj_i,
                        ez=Ez4adj_i,
                        monitor=monitor,
                        grid_step=1 / resolution,
                        direction=direction,
                    )
                    if "minus_src" in opt_cfg["type"]:
                        fom = torch.abs(torch.abs(fom / 1e-8) - 1)
                    else:
                        fom = torch.abs(fom / 1e-8)
                    fom = weight * fom
                else:
                    raise ValueError(f"Unknown optimization type: {opt_cfg['type']}")
                total_fom = total_fom + fom
            total_fom = total_fom * (-1)
            gradient = torch.autograd.grad(total_fom, Ez4adj_i, create_graph=True)[0]
            gradient_list.append(gradient)
        Hx = torch.stack(Hx_list, dim=0)
        Hy = torch.stack(Hy_list, dim=0)
        assert (
            Ez4fullfield_copy.dim() == 4
        ), f"Ez_copy should be 4D, Bs, 2, H, W but now {Ez4fullfield_copy.shape}"
        total_field = torch.cat((Hx, Hy, Ez4fullfield_copy), dim=1)
        total_field = pml_mask.unsqueeze(0).unsqueeze(0) * total_field
        adj_src = torch.conj(torch.stack(gradient_list, dim=0))
        return total_field, adj_src, monitor_slice_list


def plot_fourier_eps(
    eps_map: torch.Tensor,
    filepath: str,
) -> None:
    eps_map = 1 / eps_map[0]
    eps_map0 = eps_map.cpu().numpy()
    plt.imshow(eps_map0)
    plt.colorbar()
    plt.savefig(filepath.replace(".png", "_org_eps.png"))
    eps_map = torch.fft.fft2(eps_map)  # 2D FFT
    eps_map = torch.fft.fftshift(eps_map)  # Shift zero frequency to center
    eps_map = torch.abs(eps_map)
    print_stat(eps_map)
    eps_map = eps_map.cpu().numpy()
    plt.imshow(eps_map)
    plt.colorbar()
    plt.savefig(filepath)
    plt.close()


def plot_fouier_transform(
    field: torch.Tensor,
    filepath: str,
) -> None:
    field = (
        field.reshape(field.shape[0], -1, 2, field.shape[-2], field.shape[-1])
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )
    field = torch.abs(torch.view_as_complex(field)).squeeze()
    field = field[0]
    field0 = field.cpu().numpy()
    plt.imshow(field0)
    plt.savefig(filepath.replace(".png", "_org_field.png"))
    field = torch.fft.fft2(field)  # 2D FFT
    field = torch.fft.fftshift(field)  # Shift zero frequency to center
    field = torch.abs(field)
    field = field.cpu().numpy()
    plt.imshow(field)
    plt.savefig(filepath)
    plt.close()


def plot_fields(
    fields: Tensor,
    ground_truth: Tensor,
    cmap: str = "magma",
    filepath: str = "./figs/fields.png",
    **kwargs,
):
    # the field is of shape (batch, 6, x, y)
    fields = (
        fields.reshape(fields.shape[0], -1, 2, fields.shape[-2], fields.shape[-1])
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )
    fields = torch.view_as_complex(fields)
    ground_truth = (
        ground_truth.reshape(
            ground_truth.shape[0], -1, 2, ground_truth.shape[-2], ground_truth.shape[-1]
        )
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )
    ground_truth = torch.view_as_complex(ground_truth)
    fig, ax = plt.subplots(3, ground_truth.shape[1], figsize=(15, 10), squeeze=False)

    field_name = ["Hx", "Hy", "Ez"]
    for idx, field in enumerate(field_name):
        v_range = max(
            torch.abs(fields[0, idx]).max(), torch.abs(ground_truth[0, idx]).max()
        ).item()
        # Plot predicted fields in the first row
        im_pred = ax[0, idx].imshow(
            torch.abs(fields[0, idx]).cpu().numpy().T, vmin=0, vmax=v_range, cmap=cmap
        )
        ax[0, idx].set_title(f"Predicted Field {field}")
        fig.colorbar(im_pred, ax=ax[0, idx])

        # Plot ground truth fields in the second row
        im_gt = ax[1, idx].imshow(
            torch.abs(ground_truth[0, idx]).cpu().numpy().T,
            vmin=0,
            vmax=v_range,
            cmap=cmap,
        )
        ax[1, idx].set_title(f"Ground Truth {field}")
        fig.colorbar(im_gt, ax=ax[1, idx])

        # Plot the difference between the predicted and ground truth fields in the third row
        im_err = ax[2, idx].imshow(
            torch.abs(fields[0, idx] - ground_truth[0, idx]).cpu().numpy().T, cmap=cmap
        )
        ax[2, idx].set_title(f"Error {field}")
        fig.colorbar(im_err, ax=ax[2, idx])

    # Save the figure with high resolution
    plt.savefig(filepath, dpi=300)
    plt.close()


def resize_to_targt_size(image: Tensor, size: Tuple[int, int]) -> Tensor:
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)
    return F.interpolate(
        image, size=size, mode="bilinear", align_corners=False
    ).squeeze()


class DAdaptAdam(torch.optim.Optimizer):
    r"""
    Implements Adam with D-Adaptation automatic step-sizes.
    Leave LR set to 1 unless you encounter instability.

    To scale the learning rate differently for each layer, set the 'layer_scale'
    for each parameter group. Increase (or decrease) from its default value of 1.0
    to increase (or decrease) the learning rate for that layer relative to the
    other layers.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        log_every (int):
            Log using print every k steps, default 0 (no logging).
        decouple (boolean):
            Use AdamW style decoupled weight decay
        use_bias_correction (boolean):
            Turn on Adam's bias correction. Off by default.
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
    """

    def __init__(
        self,
        params,
        lr=1.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        log_every=0,
        decouple=False,
        use_bias_correction=False,
        d0=1e-6,
        growth_rate=float("inf"),
        fsdp_in_use=False,
    ):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple:
            print("Using decoupled weight decay")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            d=d0,
            k=0,
            layer_scale=1.0,
            numerator_weighted=0.0,
            log_every=log_every,
            growth_rate=growth_rate,
            use_bias_correction=use_bias_correction,
            decouple=decouple,
            fsdp_in_use=fsdp_in_use,
        )
        self.d0 = d0
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        sk_l1 = 0.0

        group = self.param_groups[0]
        use_bias_correction = group["use_bias_correction"]
        numerator_weighted = group["numerator_weighted"]
        beta1, beta2 = group["betas"]
        k = group["k"]

        d = group["d"]
        lr = max(group["lr"] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1

        dlr = d * lr * bias_correction

        growth_rate = group["growth_rate"]
        decouple = group["decouple"]
        log_every = group["log_every"]
        fsdp_in_use = group["fsdp_in_use"]

        sqrt_beta2 = beta2 ** (0.5)

        numerator_acum = 0.0

        for group in self.param_groups:
            decay = group["weight_decay"]
            k = group["k"]
            eps = group["eps"]
            group_lr = group["lr"]
            r = group["layer_scale"]

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(
                    "Setting different lr values in different parameter groups "
                    "is only supported for values of 0. To scale the learning "
                    "rate differently for each layer, set the 'layer_scale' value instead."
                )

            for p in group["params"]:
                if p.grad is None:
                    continue
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True

                grad = p.grad.data

                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if "step" not in state:
                    state["step"] = 0
                    state["s"] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data).detach()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                s = state["s"]

                if group_lr > 0.0:
                    denom = exp_avg_sq.sqrt().add_(eps)
                    numerator_acum += (
                        r
                        * dlr
                        * torch.dot(grad.flatten(), s.div(denom).flatten()).item()
                    )

                    # Adam EMA updates
                    exp_avg.mul_(beta1).add_(grad, alpha=r * dlr * (1 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    s.mul_(sqrt_beta2).add_(grad, alpha=dlr * (1 - sqrt_beta2))
                    sk_l1 += r * s.abs().sum().item()

            ######

        numerator_weighted = (
            sqrt_beta2 * numerator_weighted + (1 - sqrt_beta2) * numerator_acum
        )
        d_hat = d

        # if we have not done any progres, return
        # if we have any gradients available, will have sk_l1 > 0 (unless \|g\|=0)
        if sk_l1 == 0:
            return loss

        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = numerator_weighted
                dist_tensor[1] = sk_l1
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_numerator_weighted = dist_tensor[0]
                global_sk_l1 = dist_tensor[1]
            else:
                global_numerator_weighted = numerator_weighted
                global_sk_l1 = sk_l1

            d_hat = global_numerator_weighted / ((1 - sqrt_beta2) * global_sk_l1)
            d = max(d, min(d_hat, d * growth_rate))

        if log_every > 0 and k % log_every == 0:
            logging.info(
                f"lr: {lr} dlr: {dlr} d_hat: {d_hat}, d: {d}. sk_l1={global_sk_l1:1.1e} numerator_weighted={global_numerator_weighted:1.1e}"
            )

        for group in self.param_groups:
            group["numerator_weighted"] = numerator_weighted
            group["d"] = d

            decay = group["weight_decay"]
            k = group["k"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                denom = exp_avg_sq.sqrt().add_(eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)

                ### Take step
                p.data.addcdiv_(exp_avg, denom, value=-1)

            group["k"] = k + 1

        return loss


class DeterministicCtx:
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.random_state = None
        self.numpy_random_state = None
        self.torch_random_state = None
        self.torch_cuda_random_state = None

    def __enter__(self):
        # Save the current states
        self.random_state = random.getstate()
        self.numpy_random_state = np.random.get_state()
        self.torch_random_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self.torch_cuda_random_state = torch.cuda.get_rng_state()

        # Set deterministic behavior based on the seed
        set_torch_deterministic(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the saved states
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_random_state)
        torch.random.set_rng_state(self.torch_random_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(self.torch_cuda_random_state)


def set_torch_deterministic(seed: int = 0) -> None:
    seed = int(seed) % (2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def numerical_gradient_2d(phi, d_size):
    grad_x = torch.zeros_like(phi)
    grad_y = torch.zeros_like(phi)

    # Compute the gradient along the x direction (rows)
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if i == 0:
                grad_x[i, j] = (phi[i + 1, j] - phi[i, j]) / d_size
            elif i == phi.shape[0] - 1:
                grad_x[i, j] = (phi[i, j] - phi[i - 1, j]) / d_size
            else:
                grad_x[i, j] = (phi[i + 1, j] - phi[i - 1, j]) / (2 * d_size)

    # Compute the gradient along the y direction (columns)
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if j == 0:
                grad_y[i, j] = (phi[i, j + 1] - phi[i, j]) / d_size
            elif j == phi.shape[1] - 1:
                grad_y[i, j] = (phi[i, j] - phi[i, j - 1]) / d_size
            else:
                grad_y[i, j] = (phi[i, j + 1] - phi[i, j - 1]) / (2 * d_size)

    return grad_x, grad_y


# Auxiliary function to calculate first and second order partial derivatives.
def ls_derivatives(phi, d_size):
    SC = 1e-12

    # First-order derivatives
    phi_x, phi_y = numerical_gradient_2d(phi, d_size)
    phi_x += SC
    phi_y += SC

    # Second-order derivatives
    phi_2x_x, phi_2x_y = numerical_gradient_2d(phi_x, d_size)
    phi_2y_x, phi_2y_y = numerical_gradient_2d(phi_y, d_size)

    phi_xx = phi_2x_x
    phi_xy = phi_2x_y
    phi_yy = phi_2y_y

    return phi_x, phi_y, phi_xx, phi_xy, phi_yy


# Minimum gap size fabrication constraint integrand calculation.
# The "beta" parameter relax the constraint near the zero plane.
class fab_penalty_ls_gap(torch.nn.Module):
    def __init__(self, beta=1, min_feature_size=1):
        super(fab_penalty_ls_gap, self).__init__()
        self.beta = beta
        self.min_feature_size = min_feature_size

    def forward(self, data):
        params = data["params"]
        x_rho = data["x_rho"]
        y_rho = data["y_rho"]
        x_phi = data["x_phi"]
        y_phi = data["y_phi"]
        nx_phi = data["nx_phi"]
        ny_phi = data["ny_phi"]
        rho_size = data["rho_size"]
        grid_size = data["grid_size"]
        # Get the level set surface.
        phi_model = LevelSetInterp(x0=x_rho, y0=y_rho, z0=params, sigma=rho_size)
        phi = phi_model.get_ls(x1=x_phi, y1=y_phi)
        phi = torch.reshape(phi, (nx_phi, ny_phi))

        phi = torch.cat((phi, phi.flip(1)), dim=1)

        # Calculates their derivatives.
        phi_x, phi_y, phi_xx, phi_xy, phi_yy = ls_derivatives(phi, grid_size)

        # Calculates the gap penalty over the level set grid.
        pi_d = np.pi / (1.3 * self.min_feature_size)
        phi_v = torch.maximum(torch.sqrt(phi_x**2 + phi_y**2), torch.tensor(1e-8))
        phi_vv = (
            phi_x**2 * phi_xx + 2 * phi_x * phi_y * phi_xy + phi_y**2 * phi_yy
        ) / phi_v**2
        return torch.nansum(
            torch.maximum(
                (
                    torch.abs(phi_vv) / (pi_d * torch.abs(phi) + self.beta * phi_v)
                    - pi_d
                ),
                torch.tensor(0),
            )
            * grid_size**2
        )


# Minimum radius of curvature fabrication constraint integrand calculation.
# The "alpha" parameter controls its relative weight to the gap penalty.
# The "sharpness" parameter controls the smoothness of the surface near the zero-contour.
# def fab_penalty_ls_curve(params,
#                          alpha=1,
#                          sharpness = 1,
#                          min_feature_size=min_feature_size,
#                          grid_size=ls_grid_size):
class fab_penalty_ls_curve(torch.nn.Module):
    def __init__(self, alpha=1, min_feature_size=1):
        super(fab_penalty_ls_curve, self).__init__()
        self.alpha = alpha
        self.min_feature_size = min_feature_size

    def forward(self, data):
        params = data["params"]
        x_rho = data["x_rho"]
        y_rho = data["y_rho"]
        x_phi = data["x_phi"]
        y_phi = data["y_phi"]
        nx_rho = data["nx_rho"]
        ny_rho = data["ny_rho"]
        nx_phi = data["nx_phi"]
        ny_phi = data["ny_phi"]
        rho_size = data["rho_size"]
        grid_size = data["grid_size"]
        # Get the permittivity surface and calculates their derivatives.
        eps = get_eps(
            params, x_rho, y_rho, x_phi, y_phi, rho_size, nx_rho, ny_rho, nx_phi, ny_phi
        )
        eps = torch.cat((eps, eps.flip(1)), dim=1)
        eps_x, eps_y, eps_xx, eps_xy, eps_yy = ls_derivatives(eps, grid_size)

        # Calculates the curvature penalty over the permittivity grid.
        pi_d = np.pi / (1.1 * self.min_feature_size)
        eps_v = torch.maximum(
            torch.sqrt(eps_x**2 + eps_y**2), torch.tensor(1e-32**1 / 6)
        )
        k = (
            eps_x**2 * eps_yy - 2 * eps_x * eps_y * eps_xy + eps_y**2 * eps_xx
        ) / eps_v**3
        curve_const = torch.abs(k * torch.arctan(eps_v / eps)) - pi_d
        return torch.nansum(
            self.alpha * torch.maximum(curve_const, torch.tensor(0)) * grid_size**2
        )


def padding_to_tiles(x, tile_size):
    """
    Pads the input tensor to a size that is a multiple of the tile size.
    the input x should be a 2D tensor with shape x_dim, y_dim
    """
    pad_x = tile_size - x.size(0) % tile_size
    pad_y = tile_size - x.size(1) % tile_size
    pady_0 = pad_y // 2
    pady_1 = pad_y - pady_0
    padx_0 = pad_x // 2
    padx_1 = pad_x - padx_0
    if pad_x > 0 or pad_y > 0:
        x = torch.nn.functional.pad(x, (pady_0, pady_1, padx_0, padx_1))
    return x, pady_0, pady_1, padx_0, padx_1


def rip_padding(eps, pady_0, pady_1, padx_0, padx_1):
    """
    Removes the padding from the input tensor.
    """
    return eps[padx_0:-padx_1, pady_0:-pady_1]


class ComplexL1Loss(torch.nn.MSELoss):
    def __init__(self, norm=False) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """Complex L1 loss between the predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth complex-valued electric field intensity

        Returns:
            Tensor: loss
        """
        if self.norm:
            diff = torch.view_as_real(x - target)
            return (
                diff.norm(p=1, dim=[1, 2, 3, 4])
                .div(torch.view_as_real(target).norm(p=1, dim=[1, 2, 3, 4]))
                .mean()
            )
        return F.l1_loss(torch.view_as_real(x), torch.view_as_real(target))


class NormalizedMSELoss(torch.nn.modules.loss._Loss):
    def __init__(self, reduce="mean"):
        super(NormalizedMSELoss, self).__init__()

        self.reduce = reduce

    def forward(self, x, y, mask):
        one_mask = mask != 0

        error_energy = torch.norm((x - y) * one_mask, p=2, dim=(-1, -2))
        field_energy = torch.norm(y * one_mask, p=2, dim=(-1, -2)) + 1e-6
        return (error_energy / field_energy).mean()


class NL2NormLoss(torch.nn.modules.loss._Loss):
    def __init__(self, reduce="mean"):
        super(NL2NormLoss, self).__init__()

        self.reduce = reduce

    def forward(self, x, y, mask):
        one_mask = mask != 0

        error_energy = torch.norm((x - y) * one_mask, p=2, dim=(-1, -2))
        field_energy = torch.norm(y * one_mask, p=2, dim=(-1, -2)) + 1e-6
        return (error_energy / field_energy).mean()


class maskedNMSELoss(torch.nn.modules.loss._Loss):
    def __init__(
        self, reduce="mean", weighted_frames=10, weight=1, if_spatial_mask=True
    ):
        super(maskedNMSELoss, self).__init__()

        self.reduce = reduce
        self.weighted_frames = weighted_frames
        self.weight = weight
        self.if_spatial_mask = if_spatial_mask

    def forward(self, x, y, mask, num_iters=1):
        frame_mask = None
        ones_mask = mask != 0
        if self.weighted_frames != 0:
            assert x.shape[-3] // num_iters >= 10
            assert x.shape[-3] % num_iters == 0
            frame_mask = torch.ones((1, x.shape[1], 1, 1)).to(x.device)
            single_prediction_len = x.shape[-3] // num_iters
            for i in range(num_iters):
                frame_mask[
                    :,
                    -single_prediction_len * (i - 1)
                    - self.weighted_frames : -single_prediction_len * (i - 1),
                    :,
                    :,
                ] = 3
        assert (
            frame_mask is not None
        ) or self.if_spatial_mask, (
            "if mask NMSE, either frame_mask or spatial_mask should be True"
        )
        if self.if_spatial_mask:
            error_energy = torch.norm((x - y) * mask, p=2, dim=(-1, -2))
            field_energy = torch.norm(y * mask, p=2, dim=(-1, -2))
        else:
            error_energy = torch.norm((x - y) * ones_mask, p=2, dim=(-1, -2))
            field_energy = torch.norm(y * ones_mask, p=2, dim=(-1, -2))
        if frame_mask is not None:
            return (((error_energy / field_energy) * frame_mask).mean(dim=(-1))).mean()
        else:
            return ((error_energy / field_energy).mean(dim=(-1))).mean()


class maskedNL2NormLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self, reduce="mean", weighted_frames=10, weight=1, if_spatial_mask=True
    ):
        super(maskedNL2NormLoss, self).__init__()

        self.reduce = reduce
        self.weighted_frames = weighted_frames
        self.weight = weight
        self.if_spatial_mask = if_spatial_mask

    def forward(self, x, y, mask, num_iters=1):
        frame_mask = None
        ones_mask = mask != 0
        if self.weighted_frames != 0:
            assert x.shape[-3] // num_iters >= 10
            assert x.shape[-3] % num_iters == 0
            frame_mask = torch.ones((1, x.shape[1], 1, 1)).to(x.device)
            single_prediction_len = x.shape[-3] // num_iters
            for i in range(num_iters):
                frame_mask[
                    :,
                    -single_prediction_len * (i - 1)
                    - self.weighted_frames : -single_prediction_len * (i - 1),
                    :,
                    :,
                ] = 3
        assert (
            frame_mask is not None
        ) or self.if_spatial_mask, (
            "if mask nl2norm, either frame_mask or spatial_mask should be True"
        )
        if self.if_spatial_mask:
            error_energy = torch.norm((x - y) * mask, p=2, dim=(-1, -2))
            field_energy = torch.norm(y * mask, p=2, dim=(-1, -2))
        else:
            error_energy = torch.norm((x - y) * ones_mask, p=2, dim=(-1, -2))
            field_energy = torch.norm(y * ones_mask, p=2, dim=(-1, -2))
        if frame_mask is not None:
            return (((error_energy / field_energy) * frame_mask).mean(dim=(-1))).mean()
        else:
            return ((error_energy / field_energy).mean(dim=(-1))).mean()


def normalize(x):
    if isinstance(x, np.ndarray):
        x_min, x_max = np.percentile(x, 5), np.percentile(x, 95)
        x = np.clip(x, x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    else:
        x_min, x_max = torch.quantile(x, 0.05), torch.quantile(x, 0.95)
        x = x.clamp(x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    return x


def print_stat(x, dist=False):
    total_number = None
    distribution = None
    if dist:
        total_number = x.numel()
        distribution = torch.histc(x, bins=10, min=float(x.min()), max=float(x.max()))
    if isinstance(x, torch.Tensor):
        if torch.is_complex(x):
            x = x.abs()
        print(
            f"min = {x.min().data.item():-15g} max = {x.max().data.item():-15g} mean = {x.mean().data.item():-15g} std = {x.std().data.item():-15g}\n total num = {total_number} distribution = {distribution}"
        )
    elif isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            x = np.abs(x)
        print(
            f"min = {np.min(x):-15g} max = {np.max(x):-15g} mean = {np.mean(x):-15g} std = {np.std(x):-15g}"
        )


class TemperatureScheduler:
    def __init__(self, initial_T, final_T, total_steps):
        self.initial_T = initial_T
        self.final_T = final_T
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps
        cos_inner = (math.pi * self.current_step) / self.total_steps
        cos_out = math.cos(cos_inner) + 1
        self.current_T = self.final_T + 0.5 * (self.initial_T - self.final_T) * cos_out
        return self.current_T

    def get_temperature(self):
        return self.current_T


class SharpnessScheduler(object):
    __mode_list__ = {"cosine", "quadratic"}

    def __init__(
        self,
        initial_sharp: float,
        final_sharp: float,
        total_steps: int,
        mode: str = "cosine",
    ):
        super().__init__()
        self.initial_sharp = initial_sharp
        self.final_sharp = final_sharp
        self.total_steps = total_steps
        self.current_step = 0
        self.current_sharp = initial_sharp
        self.mode = mode
        assert (
            mode in self.__mode_list__
        ), f"mode should be one of {self.__mode_list__}, but got {mode}"

    def _step_cosine(self):
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps
        cos_inner = (math.pi * self.current_step) / self.total_steps
        cos_out = -math.cos(cos_inner) + 1
        self.current_sharp = (
            self.initial_sharp
            + (self.final_sharp - self.initial_sharp) * (cos_out / 2) ** 1
        )
        return self.current_sharp

    def _step_quadratic(self):
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps
        self.current_sharp = (
            self.initial_sharp
            + (self.final_sharp - self.initial_sharp)
            * (self.current_step / self.total_steps) ** 2
        )
        return self.current_sharp

    def step(self):
        if self.mode == "cosine":
            return self._step_cosine()
        elif self.mode == "quadratic":
            return self._step_quadratic()
        else:
            raise ValueError(
                f"mode should be one of {self.__mode_list__}, but got {self.mode}"
            )

    def get_sharpness(self):
        return self.current_sharp


class ResolutionScheduler:
    def __init__(self, initial_res, final_res, total_steps):
        self.initial_res = initial_res
        self.final_res = final_res
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps
        self.current_res = (
            self.initial_res
            + round(
                (self.final_res - self.initial_res)
                / self.total_steps
                * self.current_step
                / 10
            )
            * 10
        )
        return self.current_res

    def get_resolution(self):
        return self.current_res


class DistanceLoss(torch.nn.modules.loss._Loss):
    def __init__(self, min_distance=0.15):
        super(DistanceLoss, self).__init__()
        self.min_distance = min_distance

    def forward(self, hole_position):
        hole_position = torch.flatten(hole_position, start_dim=0, end_dim=1)
        distance = torch.zeros(hole_position.shape[0], hole_position.shape[0])
        for i in range(hole_position.shape[0]):
            for j in range(hole_position.shape[0]):
                distance[i, j] = torch.norm(
                    hole_position[i][:-1] - hole_position[j][:-1], p=2
                )
        distance_penalty = distance - self.min_distance
        distance_penalty = distance_penalty * (distance_penalty < 0)
        distance_penalty = distance_penalty.sum()
        distance_penalty = -1 * distance_penalty
        return distance_penalty


class AspectRatioLoss(torch.nn.modules.loss._Loss):
    def __init__(self, aspect_ratio=1):
        super(AspectRatioLoss, self).__init__()
        self.aspect_ratio = aspect_ratio

    def forward(self, input):
        height = input["height"]
        width = input["width"]
        period = input["period"]
        min_distance = height * self.aspect_ratio
        width_penalty = width - min_distance
        width_penalty = torch.minimum(
            width_penalty, torch.tensor(0.0, device=width.device)
        )
        width_penalty = width_penalty.abs().sum()

        # Compute gaps between consecutive widths across the batch
        gap = period - (width[:-1] / 2) - (width[1:] / 2)

        # Compute the gap penalty
        gap_penalty = gap - min_distance
        gap_penalty = torch.minimum(gap_penalty, torch.tensor(0.0, device=width.device))
        gap_penalty = gap_penalty.abs().sum()

        return gap_penalty + width_penalty


def padding_to_tiles(x, tile_size):
    """
    Pads the input tensor to a size that is a multiple of the tile size.
    the input x should be a 2D tensor with shape x_dim, y_dim
    """
    pad_x = tile_size - x.size(0) % tile_size
    pad_y = tile_size - x.size(1) % tile_size
    pady_0 = pad_y // 2
    pady_1 = pad_y - pady_0
    padx_0 = pad_x // 2
    padx_1 = pad_x - padx_0
    if pad_x > 0 or pad_y > 0:
        x = torch.nn.functional.pad(x, (pady_0, pady_1, padx_0, padx_1))
    return x, pady_0, pady_1, padx_0, padx_1


def rip_padding(eps, pady_0, pady_1, padx_0, padx_1):
    """
    Removes the padding from the input tensor.
    """
    return eps[padx_0:-padx_1, pady_0:-pady_1]


class MaxwellResidualLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        using_ALM: bool = False,
    ):
        super().__init__(size_average, reduce, reduction)
        self.wl_list = torch.linspace(
            wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl
        )
        self.omegas = 2 * np.pi * C_0 / (self.wl_list * MICRON_UNIT)
        self.using_ALM = using_ALM

    def forward(self, Ez: Tensor, source: Tensor, As, transpose_A):
        Ez = Ez[:, -2:, :, :]
        Ez = Ez.permute(0, 2, 3, 1).contiguous()
        Ez = torch.view_as_complex(Ez)  # convert Ez to the required complex format
        source = torch.view_as_real(source).permute(0, 3, 1, 2)  # B, 2, H, W
        source = source.permute(0, 2, 3, 1).contiguous()
        source = torch.view_as_complex(
            source
        )  # convert source to the required complex format

        # there is only one omega in this case
        Ez = Ez.unsqueeze(1)
        source = source.unsqueeze(1)

        free_space_mask = source.abs() <= 1e-10
        free_space_mask = free_space_mask.flatten(0, 1).flatten(1)

        ## Ez: [bs, n_wl, h, w] complex tensor
        ## eps_r: [bs, h, w] real tensor
        ## source: [bs, n_wl, h, w] complex tensor, source in sim.solve(source), not b, b = 1j * omega * source

        # step 2: calculate loss
        lhs = []
        if self.omegas.device != source.device:
            self.omegas = self.omegas.to(source.device)
        for i in range(Ez.shape[0]):  # loop over samples in a batch
            for j in range(Ez.shape[1]):  # loop over different wavelengths
                ez = Ez[i, j].flatten()
                # omega = 2 * np.pi * C_0 / (self.wl_list[j] * MICRON_UNIT)
                wl = round(self.wl_list[j].item() * 100) / 100
                entries = As[f"A-wl-{wl}-entries_a"][i]
                indices = As[f"A-wl-{wl}-indices_a"][i]
                # b = source[i, j].flatten() * (1j * omega)
                # print("this is the shape of the indices", indices.shape, flush=True) # this is the shape of the indices torch.Size([2, 405600])
                # assert len(indices.shape) == 3
                if transpose_A:
                    # print("this is the shape of the indices", indices.shape, flush=True) # this is the shape of the indices torch.Size([2, 405600])
                    indices = torch.flip(
                        indices, [0]
                    )  # considering the batch dimension, the axis set to 1 corresponds to axis = 0 in solver.
                    # b = b / 1j / omega
                A_by_e = spmm(
                    indices,
                    entries,
                    m=ez.shape[0],
                    n=ez.shape[0],
                    matrix=ez[:, None],
                )[:, 0]
                lhs.append(A_by_e)
        lhs = torch.stack(lhs, 0)  # [bs*n_wl, h*w]
        if not transpose_A:
            b = (
                (source * (1j * self.omegas[None, :, None, None]))
                .flatten(0, 1)
                .flatten(1)
            )  # [bs*n_wl, h*w]
        else:
            b = (source).flatten(0, 1).flatten(1)
        difference = lhs - b
        if (
            not self.using_ALM
        ):  # when we are not using ALM, we set the difference to zero in the free space region
            difference[~free_space_mask] = 0
        # b[~free_space_mask] = 0
        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # diff = ax[0].imshow(torch.abs(difference[0]).reshape(Ez.shape[-2], Ez.shape[-1]).detach().cpu().numpy())
        # lhs_plot = ax[1].imshow(torch.abs(lhs[0]).reshape(Ez.shape[-2], Ez.shape[-1]).detach().cpu().numpy())
        # b_plot = ax[2].imshow(torch.abs(b[0]).reshape(Ez.shape[-2], Ez.shape[-1]).detach().cpu().numpy())
        # plt.colorbar(diff, ax=ax[0])
        # plt.colorbar(lhs_plot, ax=ax[1])
        # plt.colorbar(b_plot, ax=ax[2])
        # plt.savefig("./figs/maxwell_residual_plot.png", dpi = 300)
        # plt.close()
        difference = torch.view_as_real(difference).double()
        b = torch.view_as_real(b).double()
        # print("this is the l2 norm of the b ", torch.norm(b, p=2, dim=(-2, -1)), flush=True) # ~e+22
        loss = (
            torch.norm(difference, p=2, dim=(-2, -1))
            / (torch.norm(b, p=2, dim=(-2, -1)) + 1e-6)
        ).mean()
        return loss


class GradientLoss(torch.nn.modules.loss._Loss):
    def __init__(self, reduce="mean"):
        super(GradientLoss, self).__init__()

        self.reduce = reduce

    def forward(
        self,
        forward_fields,
        adjoint_fields,
        target_gradient,
        gradient_multiplier,
        dr_mask=None,
    ):
        forward_fields_ez = forward_fields[
            :, -2:, :, :
        ]  # the forward fields has three components, we only need the Ez component
        forward_fields_ez = torch.view_as_complex(
            forward_fields_ez.permute(0, 2, 3, 1).contiguous()
        )
        adjoint_fields = adjoint_fields[
            :, -2:, :, :
        ]  # the adjoint fields has three components, we only need the Ez component
        adjoint_fields = torch.view_as_complex(
            adjoint_fields.permute(0, 2, 3, 1).contiguous()
        )  # adjoint fields only Ez
        gradient = -(adjoint_fields * forward_fields_ez).real
        batch_size = gradient.shape[0]
        for i in range(batch_size):
            gradient[i] = (
                gradient[i]
                / gradient_multiplier[
                    "field_adj_normalizer-wl-1.55-port-in_port_1-mode-1"
                ][i]
            )
        # Step 0: build one_mask from dr_mask
        ## This is not correct
        # need to build a design region mask whose size shold be b, H, W
        if dr_mask is not None:
            dr_masks = []
            for i in range(batch_size):
                mask = torch.zeros_like(gradient[i]).to(gradient.device)
                for key, value in dr_mask.items():
                    if key.endswith("x_start"):
                        x_start = value[i]
                    elif key.endswith("x_stop"):
                        x_stop = value[i]
                    elif key.endswith("y_start"):
                        y_start = value[i]
                    elif key.endswith("y_stop"):
                        y_stop = value[i]
                    else:
                        raise ValueError(f"Invalid key: {key}")
                mask[x_start:x_stop, y_start:y_stop] = 1
                dr_masks.append(mask)
            dr_masks = torch.stack(dr_masks, 0)
        else:
            dr_masks = torch.ones_like(gradient)

        x = -EPSILON_0 * (2 * torch.pi * C_0 / (1.55 * MICRON_UNIT)) ** 2 * (gradient)
        y = target_gradient
        error_energy = torch.norm((x - y) * dr_masks, p=2, dim=(-1, -2))
        field_energy = torch.norm(y * dr_masks, p=2, dim=(-1, -2)) + 1e-6
        return (error_energy / field_energy).mean()


class LevelSetInterp(object):
    """This class implements the level set surface using Gaussian radial basis functions."""

    def __init__(
        self,
        x0: Tensor = None,
        y0: Tensor = None,
        z0: Tensor = None,
        sigma: float = None,
        device: Device = torch.device("cuda:0"),
    ):
        # Input data.
        x, y = torch.meshgrid(y0, x0, indexing="ij")
        xy0 = torch.column_stack((x.reshape(-1), y.reshape(-1)))
        self.xy0 = xy0
        self.z0 = z0
        self.sig = sigma
        self.device = device

        # Builds the level set interpolation model.
        gauss_kernel = self.gaussian(self.xy0, self.xy0)
        self.model = torch.matmul(
            torch.linalg.inv(gauss_kernel), self.z0
        )  # Solve gauss_kernel @ model = z0
        # self.model = torch.linalg.solve(gauss_kernel, self.z0) # sees more stable

    def gaussian(self, xyi, xyj):
        dist = torch.sqrt(
            (xyi[:, 1].reshape(-1, 1) - xyj[:, 1].reshape(1, -1)) ** 2
            + (xyi[:, 0].reshape(-1, 1) - xyj[:, 0].reshape(1, -1)) ** 2
        )
        return torch.exp(-(dist**2) / (2 * self.sig**2)).to(self.device)

    def get_ls(self, x1, y1):
        xx, yy = torch.meshgrid(y1, x1, indexing="ij")
        xy1 = torch.column_stack((xx.reshape(-1), yy.reshape(-1)))
        ls = self.gaussian(self.xy0, xy1).T @ self.model
        return ls


def get_eps(
    design_param,
    x_rho,
    y_rho,
    x_phi,
    y_phi,
    rho_size,
    nx_rho,
    ny_rho,
    nx_phi,
    ny_phi,
    sharpness,
    plot_levelset=False,
):
    """Returns the permittivities defined by the zero level set isocontour"""
    phi_model = LevelSetInterp(
        x0=x_rho, y0=y_rho, z0=design_param, sigma=rho_size, device=design_param.device
    )
    phi = phi_model.get_ls(x1=x_phi, y1=y_phi)

    # the following is do the binarization projection, we have done it outside this function
    # # Calculates the permittivities from the level set surface.
    eps_phi = 0.5 * (torch.tanh(sharpness * phi) + 1)
    # eps = eps_min + (eps_max - eps_min) * eps_phi
    # eps = torch.maximum(eps, eps_min)
    # eps = torch.minimum(eps, eps_max)

    # Reshapes the design parameters into a 2D matrix.
    eps = torch.reshape(eps_phi, (nx_phi, ny_phi))

    # Plots the level set surface.
    if plot_levelset:
        rho = np.reshape(design_param, (nx_rho, ny_rho))
        phi = np.reshape(phi, (nx_phi, ny_phi))
        plot_level_set(x0=x_rho, y0=y_rho, rho=rho, x1=x_phi, y1=y_phi, phi=phi)

    return eps


def plot_level_set(x0, y0, rho, x1, y1, phi):
    y, x = np.meshgrid(y0, x0)
    yy, xx = np.meshgrid(y1, x1)

    fig = plt.figure(figsize=(12, 6), tight_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.view_init(elev=45, azim=-45, roll=0)
    ax1.plot_surface(xx, yy, phi, cmap="RdBu", alpha=0.8)
    ax1.contourf(
        xx,
        yy,
        phi,
        levels=[np.amin(phi), 0],
        zdir="z",
        offset=0,
        colors=["k", "w"],
        alpha=0.5,
    )
    ax1.contour3D(xx, yy, phi, 1, cmap="binary", linewidths=[2])
    ax1.scatter(x, y, rho, color="black", linewidth=1.0)
    ax1.set_title("Level set surface")
    ax1.set_xlabel("x ($\mu m$)")
    ax1.set_ylabel("y ($\mu m$)")
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor("w")
    ax1.yaxis.pane.set_edgecolor("w")
    ax1.zaxis.pane.set_edgecolor("w")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contourf(xx, yy, phi, levels=[0, np.amax(phi)], colors=[[0, 0, 0]])
    ax2.set_title("Zero level set contour")
    ax2.set_xlabel("x ($\mu m$)")
    ax2.set_ylabel("y ($\mu m$)")
    ax2.set_aspect("equal")
    plt.show()


def grid_average(e, monitor, direction: str = "x", autograd=False, pol: str = "Ez"):
    if autograd:
        mean = npa.mean
    else:
        mean = np.mean
    if isinstance(e, torch.Tensor):
        mean = lambda x, axis: torch.mean(x, dim=axis)

    if direction[0] == "x":
        if isinstance(monitor, Slice):
            if isinstance(monitor[0], torch.Tensor):
                e_monitor = (
                    monitor[0]
                    + torch.tensor([[-1], [0]], device=monitor[0].device)
                    + (0 if pol == "Ez" else 1),
                    monitor[1],
                )

                e_yee_shifted = torch.mean(e[e_monitor], dim=0)
            else:
                e_monitor = (
                    monitor[0] + np.array([[-1], [0]]) + (0 if pol == "Ez" else 1),
                    monitor[1],
                )

                e_yee_shifted = mean(e[e_monitor], axis=0)
        elif isinstance(monitor, np.ndarray):
            e_monitor = monitor.nonzero()
            e_monitor = (e_monitor[0] + (-1 if pol == "Ez" else 1), e_monitor[1])
            e_yee_shifted = (e[monitor] + e[e_monitor]) / 2
        elif isinstance(monitor, torch.Tensor):
            e_monitor = torch.nonzero(monitor, as_tuple=True)
            e_monitor_shifted = (
                e_monitor[0] + (-1 if pol == "Ez" else 1),
                e_monitor[1],
            )
            e_yee_shifted = (e[e_monitor] + e[e_monitor_shifted]) / 2
    elif direction[0] == "y":
        if isinstance(monitor, Slice):
            if isinstance(monitor[0], torch.Tensor):
                e_monitor = (
                    monitor[0],
                    monitor[1]
                    + torch.tensor([[-1], [0]], device=monitor[0].device)
                    + (0 if pol == "Ez" else 1),
                )
                e_yee_shifted = torch.mean(e[e_monitor], dim=0)
            else:
                e_monitor = (
                    monitor[0],
                    monitor[1] + np.array([[-1], [0]]) + (0 if pol == "Ez" else 1),
                )
                e_yee_shifted = mean(e[e_monitor], axis=0)
        elif isinstance(monitor, np.ndarray):
            e_monitor = monitor.nonzero()
            e_monitor = (e_monitor[0], e_monitor[1] + (-1 if pol == "Ez" else 1))
            e_yee_shifted = (e[monitor] + e[e_monitor]) / 2
        elif isinstance(monitor, torch.Tensor):
            e_monitor = torch.nonzero(monitor, as_tuple=True)
            e_monitor_shifted = (
                e_monitor[0],
                e_monitor[1] + (-1 if pol == "Ez" else 1),
            )
            e_yee_shifted = (e[e_monitor] + e[e_monitor_shifted]) / 2
    return e_yee_shifted


def get_flux(
    hx,
    hy,
    ez,
    monitor=None,
    grid_step: float = 0.05,
    direction: str = "x",
    autograd=False,
    pol: str = "Ez",
):
    if autograd:
        ravel = npa.ravel
        real = npa.real
    else:
        ravel = np.ravel
        real = np.real
    if isinstance(ez, torch.Tensor):
        ravel = torch.ravel
        real = torch.real

    if direction[0] == "x":
        if monitor is None:
            # no need to slice and ravel
            h = (0, hy, 0)

        else:
            h = (0, ravel(hy[monitor]), 0)
    elif direction[0] == "y":
        if monitor is None:
            h = (hx, 0, 0)
        else:
            h = (ravel(hx[monitor]), 0, 0)
    # The E-field is not co-located with the H-field in the Yee cell. Therefore,
    # we must sample at two neighboring pixels in the propataion direction and
    # then interpolate:
    if monitor is None:
        e_yee_shifted = ez
    else:
        e_yee_shifted = grid_average(ez, monitor, direction, autograd=autograd, pol=pol)

    e = (0.0, 0.0, e_yee_shifted)

    if pol == "Hz":
        ## swap e and h
        e, h = h, e
    s = 0.5 * real(overlap(e, h, dl=grid_step * MICRON_UNIT, direction=direction))

    return s


def get_shape(shape_type, shape_cfg, field_size, grid_step, device):
    shape_cfg["size"] = field_size[
        -1
    ]  # last dimension is the shape dimension, others are batch dimension
    shape_cfg["grid_step"] = grid_step
    return shape_dict[shape_type](**shape_cfg, device=device)


def get_shape_similarity(
    field,  # slices field
    grid_step,
    shape_type,
    shape_cfg,
    intensity: bool = True,
    similarity: str = "cosine",  # angular or cosine
    plot: bool = True,
):
    ## field: can support batch dimension, effective dimension is the last dimension
    ## e.g., field can be of shape [..., n]
    field = field.abs()
    field = field.reshape(-1, field.shape[-1])  # [b, n]

    if intensity:
        field = field.square()

    target_shape = get_shape(
        shape_type, shape_cfg, field.shape, grid_step, device=field.device
    )
    # return the angular similarity between the field intensity and the target shape
    if similarity == "cosine":
        return torch.nn.functional.cosine_similarity(
            field, target_shape.unsqueeze(0), dim=-1
        ).mean()
    elif similarity == "angular":
        return (
            1
            - torch.arccos(
                torch.nn.functional.cosine_similarity(
                    field, target_shape.unsqueeze(0), dim=-1
                )
            ).mul(1 / np.pi)
        ).mean()
    else:
        raise ValueError(f"Invalid similarity: {similarity}")


Slice = collections.namedtuple("Slice", "x y")


@lru_cache(maxsize=64)
def Si_eps(wavelength):
    """Returns the permittivity of silicon at the given wavelength"""
    return 3.48**2
    return Si.epsilon(1 / wavelength)[0, 0].real


@lru_cache(maxsize=64)
def Si_eff_eps(wavelength, width: float = 10, thickness: float = 0.22):
    """Returns the effective permittivity of silicon at the given wavelength"""
    if width > 1:  # very wide waveguidem we will treat it as infinite wide waveguide
        assert 0.15 <= thickness <= 0.22, "thickness should be between 0.15 and 0.22"
        # eps = 2.539683**2, # 150nm
        # eps = 2.594905**2, # 160nm
        # eps = 2.645874**2, # 170nm
        # eps = 2.692927**2, # 180nm
        permittivity = (
            -1.7708307286e01 * thickness**2
            + 1.0954869490e01 * thickness
            + 1.2951253477e00
        ) ** 2
    else:
        assert (
            thickness == 0.22
        ), f"only support thickness of 0.22 for narrow waveguide, but got {thickness}"
        match width:
            case 0.48:
                permittivity = 2.411707**2  # from lumerical
            case 0.8:
                permittivity = 2.688673**2  # from lumerical
            case _:
                raise ValueError(f"Invalid width: {width}, only support 0.48 and 0.8")
    return permittivity


@lru_cache(maxsize=64)
def SiO2_eps(wavelength):
    """Returns the permittivity of silicon at the given wavelength"""
    return 1.44**2
    return SiO2.epsilon(1 / wavelength)[0, 0].real


@lru_cache(maxsize=64)
def Air_eps(wavelength):
    """Returns the permittivity of silicon at the given wavelength"""
    return 1


@lru_cache(maxsize=64)
def SiN_eps(wavelength):
    """Returns the permittivity of silicon at the given wavelength"""
    return 2.45**2
    return SiO2.epsilon(1 / wavelength)[0, 0].real


@lru_cache(maxsize=64)
def TiO2_eps(wavelength):
    """Returns the permittivity of silicon at the given wavelength"""
    return 2.9**2


material_fn_dict = {
    "Si": Si_eps,
    "Si_eff": Si_eff_eps,
    "SiO2": SiO2_eps,
    "SiN": SiN_eps,
    "Air": Air_eps,
    "TiO2": TiO2_eps,
}


def get_material_fn(material: str | float) -> Callable:
    """Returns the permittivity of the given material"""
    if isinstance(material, float):  # permittivity

        def _material_fn(*args, **kwargs):
            return material

        return _material_fn
    if material not in material_fn_dict:
        raise ValueError(f"Invalid material: {material}")
    return material_fn_dict[material]


train_configs = Config()
inverse_configs = Config()
