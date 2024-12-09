"""
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-12-08 15:56:43
FilePath: /MAPS/core/fdfd/near2far.py
"""

import numpy as np
import torch
from einops import einsum
from torch import Tensor


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
        if len(far_xs.shape) == 0:  # vertical farfield slice
            far_xs = far_xs.reshape([1]).repeat(len(far_ys))
            # print(far_xs, far_ys)
            far_xs = torch.stack((far_xs, far_ys), dim=-1)
        else:
            far_ys = far_ys.reshape([1]).repeat(len(far_xs))
            far_xs = torch.stack((far_xs, far_ys), dim=-1)

    far_fields = {"Ex": 0, "Ey": 0, "Ez": 0, "Hx": 0, "Hy": 0, "Hz": 0}

    fz = fields[..., *nearfield_slice, :]  # [bs, s, nf]
    near_xs, near_ys = nearfield_slice_info["xs"], nearfield_slice_info["ys"]
    if not isinstance(near_xs, np.ndarray):
        near_xs = np.array(near_xs.item())
    if not isinstance(near_ys, np.ndarray):
        near_ys = np.array(near_ys.item())

    xs = torch.from_numpy(near_xs).to(fields.device)
    ys = torch.from_numpy(near_ys).to(fields.device)
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
    far_fields[component] = farfield  # [bs, n, nf]

    return far_fields
