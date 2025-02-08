"""
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-06 19:01:06
FilePath: /MAPS/core/fdfd/near2far.py
"""

from typing import Tuple

import numpy as np
import torch
from einops import einsum
from pyutils.general import TimerCtx
from torch import Tensor

from core.fdfd.utils import hankel
from core.utils import car_2_sph, sph_2_car_field, trapezoid_1d
from thirdparty.ceviche.constants import C_0, EPSILON_0, MU_0


# https://digitalcommons.calpoly.edu/cgi/viewcontent.cgi?article=1264&context=phy_fac
# full RS propagation equation
def RayleighSommerfieldPropagation(
    x: Tensor,  # far field location, um (x,y)
    freqs: Tensor,  # freqs = 1/lambda
    eps: float,  # eps_r in the homogeneous medium
    mu: float,  # mu_0 in the homogeneous medium
    x0: Tensor,  # nearfield monitor locations, um (x,y)
    fz: Tensor,  # nearfield fields, [bs, s, nf] complex
    z: Tensor,  # wave prop distance vector, um (x,y) [n,s,1]
    dL: float,  # grid size, um, used in nearfield integral
):
    # x: [n, 2], n far field target points, 2 dimension, x and y
    # freqs: [nf] frequencies
    # eps: scalar, permittivity in the homogeneous medium
    # mu: scalar, permeability in the homogeneous medium
    # x0: [s, 2] source near-field points, s near-field source points, 2 dimension, x and y
    # c0: field component direction X,Y,Z -> 0,1,2
    # fz: [bs, s, nf] a batch of DFT fields on near-field monitors, e.g., typically fz is Ez fields
    # z: [n,s,1] wave prop direction distance along x-axis
    # dL: grid size, um, used in nearfield integral

    # [n, 1, 2] - [1, s, 2] = [n, s, 2]
    r = x[..., None, :] - x0[None, ...]  # distance vector # [n, s, 2]
    r = torch.norm(r, p=2, dim=-1, keepdim=True)  # sqrt(dx^2 + dy^2) # [n, s, 1]
    k = 2 * np.pi * eps**0.5 * freqs  # wave number # [nf]
    factor = 1 - 1 / (1j * k * r)  # [n,s,nf]
    H = z / r**2 * torch.exp(1j * k * r)  # [n,s,1] * [n,s,nf] = [n,s,nf]
    farfield = einsum(
        freqs.mul(eps**0.5 * (-1j) * dL), fz, H * factor, "f, b s f, n s f -> b n f"
    )
    # this 2x is just a fix to match fdfd results.
    return farfield * 2


def get_farfields_Rayleigh_Sommerfeld(
    nearfield_slice,  # list of nearfield monitor
    nearfield_slice_info,  # list of nearfield monitor
    fields: Tensor,  # nearfield fields, entire fields
    farfield_x: Tensor | dict,  # farfield points physical locatinos, in um (x, y)
    freqs: Tensor,  # 1 / lambda, e.g., 1/1.55
    eps: float,  # eps_r
    mu: float,
    dL: float,  # grid size , um
    component: str = "Ez",
    farfield_slice_info: dict = None,
):
    """
    https://github.com/demroz/pinn-ms/blob/main/edofOpt/computeFarfields.py
    nearfield_regions: list of nearfield monitor, {monitor_name: {"slice": slice, "center": center, "size": size}}
    fields: [bs, X, Y, nf] a batch of nearfield fields, e.g., Ez
    x: [n, 2], batch size, n far field target points, 2 dimension, x and y
    freqs: [nf] frequencies
    eps: scalar, permittivity in the homogeneous medium
    mu: scalar, permeability in the homogeneous medium
    component: str, nearfield fields component
    """
    if farfield_x is None:
        assert farfield_slice_info is not None, "farfield_slice_info is required"
        far_xs, far_ys = farfield_slice_info["xs"], farfield_slice_info["ys"]
        if not isinstance(far_xs, np.ndarray):
            far_xs = np.array(far_xs.item())
        if not isinstance(far_ys, np.ndarray):
            far_ys = np.array(far_ys.item())
        far_xs = torch.from_numpy(far_xs).to(fields.device)
        far_ys = torch.from_numpy(far_ys).to(fields.device)
        print("this is the shape of far_xs", far_xs.shape)
        print("this is far_ys", far_ys)
        if len(far_xs.shape) == 0:  # vertical farfield slice
            far_xs = far_xs.reshape([1]).repeat(len(far_ys))
            # print(far_xs, far_ys)
        elif len(far_ys.shape) == 0:  # horizontal farfield slice
            far_ys = far_ys.reshape([1]).repeat(len(far_xs))

        farfield_shape = far_xs.shape
        far_xs = torch.stack((far_xs, far_ys), dim=-1).flatten(0, -2)  # [n, 2]
    else:
        far_xs = farfield_x

    far_fields = {"Ex": 0, "Ey": 0, "Ez": 0, "Hx": 0, "Hy": 0, "Hz": 0}

    fz = fields[..., *nearfield_slice, :]  # [bs, s, nf]
    near_xs, near_ys = nearfield_slice_info["xs"], nearfield_slice_info["ys"]
    if not isinstance(near_xs, np.ndarray):
        near_xs = np.array(near_xs.item())
    if not isinstance(near_ys, np.ndarray):
        near_ys = np.array(near_ys.item())

    xs = torch.from_numpy(near_xs).to(fields.device)
    ys = torch.from_numpy(near_ys).to(fields.device)
    direction = nearfield_slice_info["direction"]  # e.g., x+, x-, y+, y-
    if len(xs.shape) == 0:  # vertical monitor
        xs = xs.reshape([1]).expand_as(ys)
        xs = torch.stack((xs, ys), dim=-1)  # [num_src_points, 2]
        z = (
            far_xs[..., None, 0:1] - xs[None, ..., 0:1]
        )  # [n,s,1] # wave prop direction distance along x-axis
    else:
        ys = ys.reshape([1]).expand_as(xs)
        xs = torch.stack((xs, ys), dim=-1)  # [num_src_points, 2]
        z = (
            far_xs[..., None, 1:2] - xs[None, ..., 1:2]
        )  # [n,s,1] # wave prop direction distance along y-axis

    farfield = RayleighSommerfieldPropagation(
        x=far_xs, freqs=freqs, eps=eps, mu=mu, x0=xs, fz=fz, z=z, dL=dL
    )
    far_fields[component] = farfield.reshape(
        -1, *farfield_shape, len(freqs)
    )  # [bs, n, nf]

    return far_fields


def get_farfields_GreenFunction(
    nearfield_slices,  # list of nearfield monitor
    nearfield_slices_info,  # list of nearfield monitor
    Fz: Tensor,  # nearfield fields, entire fields
    Fx: Tensor,  # nearfield fields, entire fields
    Fy: Tensor,  # nearfield fields, entire fields
    farfield_x: Tensor | dict,  # farfield points physical locatinos, in um (x, y)
    freqs: Tensor,  # 1 / lambda, e.g., 1/1.55
    eps: float,  # eps_r
    mu: float,
    dL: float,  # grid size , um
    component: str = "Ez",
    farfield_slice_info: dict = None,
    decimation_factor: int = 1,  # subsampling rate on near field source
):
    """
    https://github.com/demroz/pinn-ms/blob/main/edofOpt/computeFarfields.py
    nearfield_regions: list of nearfield monitor, {monitor_name: {"slice": slice, "center": center, "size": size}}
    fields: [bs, X, Y, nf] a batch of nearfield fields, e.g., Ez
    x: [n, 2], batch size, n far field target points, 2 dimension, x and y
    freqs: [nf] frequencies
    eps: scalar, permittivity in the homogeneous medium
    mu: scalar, permeability in the homogeneous medium
    component: str, nearfield fields component
    """
    if farfield_x is None:
        assert farfield_slice_info is not None, "farfield_slice_info is required"
        far_xs, far_ys = farfield_slice_info["xs"], farfield_slice_info["ys"]
        if not isinstance(far_xs, np.ndarray):
            far_xs = np.array(far_xs.item())
        if not isinstance(far_ys, np.ndarray):
            far_ys = np.array(far_ys.item())
        far_xs = torch.from_numpy(far_xs.astype(np.float32)).to(Fz.device)
        far_ys = torch.from_numpy(far_ys.astype(np.float32)).to(Fz.device)
        if len(far_xs.shape) == 0:  # vertical farfield slice
            far_xs = far_xs.reshape([1]).repeat(len(far_ys))
            # print(far_xs, far_ys)
        elif len(far_ys.shape) == 0:  # horizontal farfield slice
            far_ys = far_ys.reshape([1]).repeat(len(far_xs))

        farfield_shape = far_xs.shape
        far_xs = torch.stack((far_xs, far_ys), dim=-1).flatten(0, -2)  # [n, 2]
    else:
        far_xs = farfield_x

    far_fields = {"Ex": 0, "Ey": 0, "Ez": 0, "Hx": 0, "Hy": 0, "Hz": 0}

    for nearfield_slice, nearfield_slice_info in zip(
        nearfield_slices, nearfield_slices_info
    ):
        fz = Fz[..., *nearfield_slice, :]  # [bs, s, nf]
        fx = Fx[..., *nearfield_slice, :]  # [bs, s, nf]
        fy = Fy[..., *nearfield_slice, :]  # [bs, s, nf]
        ## near field slice locations (um)
        near_xs, near_ys = nearfield_slice_info["xs"], nearfield_slice_info["ys"]
        if not isinstance(near_xs, np.ndarray):
            near_xs = np.array(near_xs.item())
        if not isinstance(near_ys, np.ndarray):
            near_ys = np.array(near_ys.item())

        xs = torch.from_numpy(near_xs.astype(np.float32)).to(Fz.device)
        ys = torch.from_numpy(near_ys.astype(np.float32)).to(Fz.device)
        direction = nearfield_slice_info["direction"]  # e.g., x+, x-, y+, y-
        if len(xs.shape) == 0:  # vertical monitor
            xs = xs.reshape([1]).expand_as(ys)
            xs = torch.stack((xs, ys), dim=-1)  # [num_src_points, 2]
        else:
            ys = ys.reshape([1]).expand_as(xs)
            xs = torch.stack((xs, ys), dim=-1)  # [num_src_points, 2]

        # with TimerCtx() as t:
        fx, fy, fz = GreenFunctionProjection(
            x=far_xs,
            freqs=freqs,
            eps_r=eps,
            mu=mu,
            x0=xs,
            Fz=fz,
            Fx=fx,
            Fy=fy,
            dL=dL,
            near_monitor_direction=direction,
            decimation_factor=decimation_factor,
            component=component,
        )
        #     torch.cuda.synchronize()
        # print(f"GreenFunctionProjection time: {t.interval} s")
        if component == "Ez":
            far_fields["Ez"] += fz.reshape(
                -1, *farfield_shape, len(freqs)
            )  # [bs, n, nf]
            far_fields["Hx"] += fx.reshape(
                -1, *farfield_shape, len(freqs)
            )  # [bs, n, nf]
            far_fields["Hy"] += fy.reshape(
                -1, *farfield_shape, len(freqs)
            )  # [bs, n, nf]
        elif component == "Hz":
            far_fields["Hz"] += fz.reshape(-1, *farfield_shape, len(freqs))
            far_fields["Ex"] += fx.reshape(-1, *farfield_shape, len(freqs))
            far_fields["Ey"] += fy.reshape(-1, *farfield_shape, len(freqs))

    return far_fields


# @torch.compile
def GreenFunctionProjection(
    x: Tensor,  # far field location, um (x,y)
    freqs: Tensor,  # freqs = 1/lambda
    eps_r: float,  # eps_r in the homogeneous medium
    mu: float,  # mu_0 in the homogeneous medium
    x0: Tensor,  # nearfield monitor locations, um (x,y)
    Fz: Tensor,  # nearfield fields, [bs, s, nf] complex
    Fx: Tensor,  # nearfield fields, [bs, s, nf] complex
    Fy: Tensor,  # nearfield fields, [bs, s, nf] complex
    dL: float,  # grid size, um, used in nearfield integral
    near_monitor_direction: str = "x+",
    decimation_factor: int = 1,  # subsampling rate on near field source
    maximum_batch_size: int = 500,
    component: str = "Ez",  # Ez, Hz
):
    x_ref = x
    num_iter = x_ref.shape[0] // maximum_batch_size + 1
    far_field_fz = []
    far_field_fx = []
    far_field_fy = []
    if decimation_factor > 1:
        Fz = Fz[:, decimation_factor // 2 :: decimation_factor, :].contiguous()
        Fx = Fx[:, decimation_factor // 2 :: decimation_factor, :].contiguous()
        Fy = Fy[:, decimation_factor // 2 :: decimation_factor, :].contiguous()
        x0 = x0[decimation_factor // 2 :: decimation_factor, :].contiguous()
        dL = dL * decimation_factor

    dtype = Fz.dtype
    freqs = freqs.float()
    x0 = x0.float()
    Fz = Fz.cfloat()
    Fx = Fx.cfloat()
    Fy = Fy.cfloat()

    i_omega = -2j * (np.pi * C_0) * freqs  # [nf]
    k = 2 * np.pi * eps_r**0.5 * freqs  # wave number # [nf]
    epsilon = EPSILON_0 * eps_r
    for i in range(num_iter):
        x = (
            x_ref[i * maximum_batch_size : (i + 1) * maximum_batch_size]
            if i < num_iter - 1
            else x_ref[i * maximum_batch_size :]
        )

        # transform the coordinate system so that the origin is at the source point
        # then the observation points in the new system are:
        r = x[..., None, :] - x0[None, ...]  # distance vector # [n, s, 2]

        # tangential source components to use

        if near_monitor_direction.startswith("x"):
            # n = torch.tensor([1.0, 0.0, 0.0], dtype=x.dtype, device=x.device) # surface normal
            # surface equivalence theory
            # J = n x H = nx.*Hy - ny.*Hx
            if near_monitor_direction[-1] == "+":
                if component == "Ez":
                    # [0, -Hz, Hy]
                    J = (0, 0, Fy[..., None, :, :])  # [bs, s, nf] -> [bs, 1, s, nf]
                    # M = -n x E
                    # (0, Ez, -Ey)
                    M = (0, Fz[..., None, :, :], 0)
                elif component == "Hz":
                    # [0, -Hz, Hy]
                    J = (0, -Fz[..., None, :, :], 0)
                    # (0, Ez, -Ey)
                    M = (0, 0, -Fy[..., None, :, :])
            else:
                if component == "Ez":
                    # [0, Hz, -Hy]
                    J = (0, 0, -Fy[..., None, :, :])
                    # M = -n x E
                    # (0, -Ez, Ey)
                    M = (0, -Fz[..., None, :, :], 0)
                elif component == "Hz":
                    # [0, Hz, -Hy]
                    J = (0, Fz[..., None, :, :], 0)
                    # (0, -Ez, Ey)
                    M = (0, 0, Fy[..., None, :, :])

        elif near_monitor_direction.startswith("y"):
            # n = torch.tensor([0.0, 1.0, 0.0], dtype=x.dtype, device=x.device) # surface normal
            if near_monitor_direction[-1] == "+":
                if component == "Ez":
                    # [Hz, 0, -Hx]
                    J = (0, 0, -Fx[..., None, :, :])
                    # M = -n x E
                    # (-Ez, 0, Ex)
                    M = (-Fz[..., None, :, :], 0, 0)
                elif component == "Hz":
                    # [Hz, 0, -Hx]
                    J = (Fz[..., None, :, :], 0, 0)
                    # M = -n x E
                    # (-Ez, 0, Ex)
                    M = (0, 0, Fx[..., None, :, :])
            else:
                if component == "Ez":
                    # [-Hz, 0, Hx]
                    J = (0, 0, Fx[..., None, :, :])
                    # M = -n x E
                    # (Ez, 0, -Ex)
                    M = (Fz[..., None, :, :], 0, 0)
                elif component == "Hz":
                    # [-Hz, 0, Hx]
                    J = (-Fz[..., None, :, :], 0, 0)
                    # M = -n x E
                    # (Ez, 0, -Ex)
                    M = (0, 0, -Fx[..., None, :, :])

        else:
            raise ValueError("Invalid near_monitor_direction")

        r_obs, phi_obs, theta_obs = car_2_sph(
            x=r[..., 0:1], y=r[..., 1:2], z=None
        )  # [n, s, 1]

        # angle terms
        sin_theta = 1  # sin(theta_obs), theta = 90
        cos_theta = 0  # cos(theta_obs), theta = 90
        sin_phi = torch.sin(phi_obs)  # [n, s, 1]
        cos_phi = torch.cos(phi_obs)  # [n, s, 1]

        # Green's function and terms related to its derivatives
        ### this is tidy3d implementation, have some approximation or error?
        # ikr = -1j * k * r_obs  # [n, s, nf]
        # G = torch.exp(ikr) / (4.0 * np.pi * r_obs)  # [n, s, nf]
        # tmp = (ikr - 1.0) / r_obs
        # dG_dr = G * tmp  # [n, s, nf]
        # d2G_dr2 = dG_dr * tmp + G / r_obs.square()  # [n, s, nf]

        ## this is exact hankel function implementation, more accurate!

        G, dG_dr, d2G_dr2 = _green_functions(k, r_obs)

        # operations between unit vectors and currents
        # @torch.compile
        def r_x_current(
            current: Tuple[Tensor, ...], is_2d: bool = True
        ) -> Tuple[Tensor, ...]:
            """Cross product between the r unit vector and the current."""
            if is_2d:
                return [
                    sin_phi * current[2],
                    -cos_phi * current[2],
                    cos_phi * current[1] - sin_phi * current[0],
                ]
            else:
                return [
                    sin_theta * sin_phi * current[2] - cos_theta * current[1],
                    cos_theta * current[0] - sin_theta * cos_phi * current[2],
                    sin_theta * cos_phi * current[1] - sin_theta * sin_phi * current[0],
                ]

        # @torch.compile
        def r_dot_current(current: Tuple[Tensor, ...], is_2d: bool = True) -> Tensor:
            """Dot product between the r unit vector and the current."""
            if is_2d:
                return cos_phi * current[0] + sin_phi * current[1]
            else:
                return (
                    sin_theta * cos_phi * current[0]
                    + sin_theta * sin_phi * current[1]
                    + cos_theta * current[2]
                )

        # @torch.compile
        def r_dot_current_dtheta(
            current: Tuple[Tensor, ...], is_2d: bool = True
        ) -> Tensor:
            """Theta derivative of the dot product between the r unit vector and the current."""
            if is_2d:
                return -current[2]
            else:
                return (
                    cos_theta * cos_phi * current[0]
                    + cos_theta * sin_phi * current[1]
                    - sin_theta * current[2]
                )

        # @torch.compile
        def r_dot_current_dphi_div_sin_theta(current: Tuple[Tensor, ...]) -> Tensor:
            """Phi derivative of the dot product between the r unit vector and the current,
            analytically divided by sin theta."""
            return -sin_phi * current[0] + cos_phi * current[1]

        def grad_Gr_r_dot_current(
            current: Tuple[Tensor, ...],
        ) -> Tuple[Tensor, ...]:
            """Gradient of the product of the gradient of the Green's function and the dot product
            between the r unit vector and the current."""
            dG_dr_by_r = dG_dr / r_obs
            temp = [
                d2G_dr2 * r_dot_current(current),
                dG_dr_by_r * r_dot_current_dtheta(current),
                dG_dr_by_r * r_dot_current_dphi_div_sin_theta(current),
            ]
            # convert to Cartesian coordinates
            # return surface.monitor.sph_2_car_field(temp[0], temp[1], temp[2], theta_obs, phi_obs)
            return sph_2_car_field(
                temp[0],
                temp[2],
                temp[1],
                phi_obs,
                theta_obs,
                sin_phi=sin_phi,
                cos_phi=cos_phi,
            )  # fx, fy, fz

        def potential_terms(current: Tuple[Tensor, ...], const: complex):
            """Assemble vector potential and its derivatives."""
            r_x_c = r_x_current(current)
            pot = [const * item * G for item in current]
            curl_pot = [const * item * dG_dr for item in r_x_c]
            grad_div_pot = grad_Gr_r_dot_current(current)
            grad_div_pot = [const * item for item in grad_div_pot]
            return pot, curl_pot, grad_div_pot

        # magnetic vector potential terms
        A, curl_A, grad_div_A = potential_terms(J, MU_0)

        # electric vector potential terms
        F, curl_F, grad_div_F = potential_terms(M, epsilon)

        if component == "Ez":
            # assemble the electric field components (Taflove 8.24, 8.27)
            # e_x_integrand, e_y_integrand, e_z_integrand = (
            #     i_omega * (a + grad_div_a / (k**2)) - curl_f / epsilon
            #     for a, grad_div_a, curl_f in zip(A, grad_div_A, curl_F)
            # )
            f_z_integrand = (
                i_omega * (A[2] + grad_div_A[2] / (k**2)) - curl_F[2] / epsilon
            )

            # assemble the magnetic field components (Taflove 8.25, 8.28)
            # h_x_integrand, h_y_integrand, h_z_integrand = (
            #     i_omega * (f + grad_div_f / (k**2)) + curl_a / MU_0
            #     for f, grad_div_f, curl_a in zip(F, grad_div_F, curl_A)
            # )  # [bs, n, s, nf]

            f_x_integrand, f_y_integrand = (
                i_omega * (f + grad_div_f / (k**2)) + curl_a / MU_0
                for f, grad_div_f, curl_a in zip(F[:2], grad_div_F[:2], curl_A[:2])
            )  # [bs, n, s, nf]
        elif component == "Hz":
            # assemble the electric field components (Taflove 8.24, 8.27)
            f_x_integrand, f_y_integrand = (
                i_omega * (a + grad_div_a / (k**2)) - curl_f / epsilon
                for a, grad_div_a, curl_f in zip(A[:2], grad_div_A[:2], curl_F[:2])
            )
            # e_z_integrand = i_omega * (A[2] + grad_div_A[2] / (k**2)) - curl_F[2] / epsilon

            # assemble the magnetic field components (Taflove 8.25, 8.28)
            # h_x_integrand, h_y_integrand, h_z_integrand = (
            #     i_omega * (f + grad_div_f / (k**2)) + curl_a / MU_0
            #     for f, grad_div_f, curl_a in zip(F, grad_div_F, curl_A)
            # )  # [bs, n, s, nf]

            # h_x_integrand, h_y_integrand = (
            #     i_omega * (f + grad_div_f / (k**2)) + curl_a / MU_0
            #     for f, grad_div_f, curl_a in zip(F[:2], grad_div_F[:2], curl_A[:2])
            # )  # [bs, n, s, nf]
            f_z_integrand = i_omega * (F[2] + grad_div_F[2] / (k**2)) + curl_A[2] / MU_0

        # integrate over the surface, sum over tagential dimensions
        # e.g., direciont = "x", sum over y (dim=3)
        # e_x = self.trapezoid(e_x_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        # e_y = self.trapezoid(e_y_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))

        f_z = trapezoid_1d(f_z_integrand, dx=dL, dim=-2).to(dtype)
        f_x = trapezoid_1d(f_x_integrand, dx=dL, dim=-2).to(dtype)
        f_y = trapezoid_1d(f_y_integrand, dx=dL, dim=-2).to(dtype)
        # h_z = self.trapezoid(h_z_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        far_field_fz.append(f_z)
        far_field_fx.append(f_x)
        far_field_fy.append(f_y)

    f_x = torch.cat(far_field_fx, dim=1)
    f_y = torch.cat(far_field_fy, dim=1)
    f_z = torch.cat(far_field_fz, dim=1)

    return f_x, f_y, f_z


# @torch.compile
def _green_functions(k, r_obs):
    kr = k * r_obs  # [n, s, nf]
    H0 = hankel(0, kr, kind=2)
    H1 = hankel(1, kr, kind=2)
    G = 0.25j * H0
    dG_dr = -0.25j * k * H1
    d2G_dr2 = -0.25j * k.square() * H0 + 0.25j * k / r_obs * H1
    return G, dG_dr, d2G_dr2
