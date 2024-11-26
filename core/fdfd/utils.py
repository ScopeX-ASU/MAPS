"""
Date: 2024-10-10 21:17:31
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-11 13:55:08
FilePath: /MAPS/core/models/fdfd/utils.py
"""

"""
Date: 2024-10-10 21:17:31
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-11 02:18:01
FilePath: /MAPS/core/models/fdfd/utils.py
"""
import autograd.numpy as npa
import numpy as np
import scipy.sparse as sp
import torch
from einops import einsum
from scipy.special import jn, yn
from torch import Tensor
from torch_sparse import spmm, spspmm
import collections

# def get_entries_indices(csr_matrix):
#     # takes sparse matrix and returns the entries and indeces in form compatible with 'make_sparse'
#     shape = csr_matrix.shape
#     coo_matrix = csr_matrix.tocoo()
#     entries = csr_matrix.data
#     cols = coo_matrix.col
#     rows = coo_matrix.row
#     indices = npa.vstack((rows, cols))
#     return entries, indices

__all__ = [
    "get_entries_indices",
    "torch_sparse_to_scipy_sparse",
    "real_sparse_mm",
    "sparse_mm",
    "real_sparse_mv",
    "sparse_mv",
    "hankel",
    "green2d",
    "get_farfields",
    "get_flux",
    "Slice",
    "overlap",
    "cross",
    "grid_average",
]


def get_entries_indices(coo_matrix):
    # takes sparse matrix and returns the entries and indeces in form compatible with 'make_sparse'
    entries = coo_matrix.data
    cols = coo_matrix.col
    rows = coo_matrix.row
    indices = torch.vstack((rows, cols))
    return entries, indices


def torch_sparse_to_scipy_sparse(
    A,
):  # input A is a tuple (entries_a, indices_a), do not have the coalesce function
    # A = A.coalesce()
    # return sp.coo_matrix(
    #     (A.values().cpu().numpy(), A.indices().cpu().numpy()), shape=tuple(A.shape)
    # )
    # If A is a PyTorch sparse tensor, coalesce it
    if isinstance(A, torch.sparse.Tensor):
        A = A.coalesce()

        # Convert PyTorch sparse tensor to SciPy COO format
        return sp.coo_matrix(
            (A.values().cpu().numpy(), A.indices().cpu().numpy()), shape=tuple(A.shape)
        )

    # If A is a SciPy sparse matrix, convert it to COO format
    elif isinstance(A, sp.spmatrix):
        return A.tocoo()  # Ensure the output is in COO format

    else:
        raise TypeError(
            f"Expected a PyTorch sparse tensor or a SciPy sparse matrix, but got {type(A)}"
        )


def real_sparse_mm(A, B):
    A = A.coalesce()
    B = B.coalesce()
    indices, values = spspmm(
        A.indices(),
        A.values(),
        B.indices(),
        B.values(),
        A.shape[0],
        A.shape[1],
        B.shape[1],
    )
    return torch.sparse_coo_tensor(indices, values, A.shape, device=A.device)


def sparse_mm(A, B):
    A = A.coalesce()
    B = B.coalesce()
    A_values = A.values()
    B_values = B.values()
    if torch.is_complex(A_values):
        A_real = A_values.real
        A_imag = A_values.imag
        B_real = B_values.real
        B_imag = B_values.imag

    indices, values_rr = spspmm(
        A.indices(),
        A_real,
        B.indices(),
        B_real,
        A.shape[0],
        A.shape[1],
        B.shape[1],
    )
    _, values_ri = spspmm(
        A.indices(),
        A_real,
        B.indices(),
        B_imag,
        A.shape[0],
        A.shape[1],
        B.shape[1],
    )
    _, values_ir = spspmm(
        A.indices(),
        A_imag,
        B.indices(),
        B_real,
        A.shape[0],
        A.shape[1],
        B.shape[1],
    )
    _, values_ii = spspmm(
        A.indices(),
        A_imag,
        B.indices(),
        B_imag,
        A.shape[0],
        A.shape[1],
        B.shape[1],
    )
    values = values_rr - values_ii + 1j * (values_ri + values_ir)
    return torch.sparse_coo_tensor(indices, values, A.shape, device=A.device)


def real_sparse_mv(A, x):
    A = A.coalesce()
    return spmm(A.indices(), A.values(), A.shape[0], A.shape[1], x[..., None]).squeeze(
        -1
    )


def sparse_mv(A, x):
    A = A.coalesce()
    A_values = A.values()
    if torch.is_complex(A_values):
        A_real = A_values.real
        A_imag = A_values.imag

    values_rr = spmm(A.indices(), A_real, A.shape[0], A.shape[1], x[..., None].real)
    values_ri = spmm(A.indices(), A_real, A.shape[0], A.shape[1], x[..., None].imag)
    values_ir = spmm(A.indices(), A_imag, A.shape[0], A.shape[1], x[..., None].real)
    values_ii = spmm(A.indices(), A_imag, A.shape[0], A.shape[1], x[..., None].imag)
    values = values_rr - values_ii + 1j * (values_ri + values_ir)
    return values.squeeze(-1)


def hankel(n: int, x: Tensor):
    ## hankel function J + iY
    # dtype = x.dtype
    # x = x.to(torch.float32)
    if n == 0:
        res = torch.complex(torch.special.bessel_j0(x), torch.special.bessel_y0(x))
    elif n == 1:
        res = torch.complex(torch.special.bessel_j1(x), torch.special.bessel_y1(x))
    elif n >= 2:  # must use scipy.special
        x_np = x.cpu().numpy()
        res = torch.complex(
            torch.from_numpy(jn(n, x_np)), torch.from_numpy(yn(n, x_np))
        ).to(x.device)
    else:
        raise ValueError("n must be a non-negative integer")
    return res


## https://github.com/NanoComp/meep/blob/master/src/near2far.cpp#L208
def green2d(
    x: Tensor,
    freqs: Tensor,
    eps: float,
    mu: float,
    x0: Tensor,  # nearfield monitor locations
    f0: Tensor,
    c0: str = "Ez",
):
    # x: [n, 2], n far field target points, 2 dimension, x and y
    # freqs: [nf] frequencies
    # eps: scalar, permittivity in the homogeneous medium
    # mu: scalar, permeability in the homogeneous medium
    # x0: [s, 2] source near-field points, s near-field source points, 2 dimension, x and y
    # c0: field component direction X,Y,Z -> 0,1,2
    # f0: [bs, s, nf] a batch of DFT fields on near-field monitors, e.g., typically f0 is Ez fields

    # [n, 1, 2] - [1, s, 2] = [n, s, 2]
    rhat = x[..., None, :] - x0[None, ...]  # distance vector # [n, s, 2]

    r = rhat.norm(p=2, dim=-1, keepdim=True)  # [n, s, 1]
    rhat = rhat / r  # unit vector # [n, s, 2]
    # print(rhat)
    omega = 2 * np.pi * freqs / 1e-6  # [nf] angular frequencies
    k = omega * (eps * mu) ** 0.5  # [nf] wave numbers
    ik = (1j * k).to(f0.dtype)  # [nf] imaginary wave numbers
    # [nf] * [n, s, 1] = [n, s, nf]
    kr = k * r
    # Z = (mu / eps) ** 0.5

    if c0 in {"Ez", "Hz"}:  # vertical source
        # [n, s, nf] * [bs, s, nf] = [bs, n, nf]
        print(kr.shape, kr.dtype, f0.shape, f0.dtype)
        # print(f0)
        H0 = einsum(hankel(0, kr).to(f0.dtype), f0, "n s f, b s f -> b n f")
        H1 = einsum(hankel(1, kr).to(f0.dtype), f0, "n s f, b s f -> b n s f")
        # ikH1 = 0.25 * ik * H1  # [bs, n, s, nf]
        ik_1_by_4 = 0.25 * ik

        if c0 == "Ez":  # Ez source
            Ex = Ey = Hz = 0
            # [nf] * [bs, n, nf] = [bs, n, nf]
            Ez = -0.25 * omega * mu * H0  # [bs, n, nf]
            # [bs, n, s] * [bs, n, s, nf] = [bs, n, nf]
            print(H1[0, 0, 0], ik_1_by_4[0])
            print(H1[0, 0, 0] * ik_1_by_4[0])
            Hx = torch.einsum(
                "ns,f,bnsf->bnf", -rhat[..., 1].to(ik_1_by_4.dtype), ik_1_by_4, H1
            )
            Hy = torch.einsum(
                "ns,f,bnsf->bnf", rhat[..., 0].to(ik_1_by_4.dtype), ik_1_by_4, H1
            )

        elif c0 == "Hz":  # Hz source
            Ex = torch.einsum(
                "ns,f,bnsf->bnf", rhat[..., 1].to(ik_1_by_4.dtype), ik_1_by_4, H1
            )
            Ey = torch.einsum(
                "ns,f,bnsf->bnf", -rhat[..., 0].to(ik_1_by_4.dtype), ik_1_by_4, H1
            )
            Hz = -0.25 * omega * eps * H0  # [bs, n, nf]
            Ez = Hx = Hy = 0
    elif c0 in {"Ex", "Ey", "Hx", "Hy"}:  # in-plane source
        Z = (mu / eps) ** 0.5  # [nf]
        # H0 = einsum(hankel(0, kr), f0, "n s f, b s f -> b n s f")
        H1 = einsum(hankel(1, kr), f0, "n s f, b s f -> b n s f")
        # H2 = einsum(hankel(2, kr), f0, "n s f, b s f -> b n s f")
        H0_minus_H2 = einsum(
            hankel(0, kr) - hankel(2, kr), f0, "n s f, b s f -> b n s f"
        )
        if c0 in {"Ex", "Hx"}:
            p = torch.tensor([1.0, 0.0], device=x.device)
        else:
            p = torch.tensor([0.0, 1.0], device=x.device)
        pdotrhat = einsum(p, rhat, "i, n s i -> n s")[..., None]  # [n, s, 1]
        rhatcrossp = rhat[..., 0] * p[1] - rhat[..., 1] * p[0]  # [n, s]
        if c0.startswith("E"):  # Exy source
            common_term_1 = (pdotrhat / r * 0.25) * Z
            common_term_2 = rhatcrossp[..., None] * omega * mu * 0.125
            Ex = einsum(
                -(rhat[..., 0:1] * common_term_1), H1, "n s f, b n s f -> b n f"
            ) + einsum(
                (rhat[..., 1:] * common_term_2), H0_minus_H2, "n s f, b n s f -> b n f"
            )  # [bs, n, nf]
            Ey = einsum(
                -(rhat[..., 1:] * common_term_1), H1, "n s f, b n s f -> b n f"
            ) - einsum(
                (rhat[..., 0:1] * common_term_2), H0_minus_H2, "n s f, b n s f -> b n f"
            )  # [bs, n, nf]
            Hx = Hy = Ez = 0

            Hz = einsum(
                -(rhatcrossp[..., None] * 0.25 * ik), H1, "n s f, b n s f -> b n f"
            )  # [bs, n, nf]
        elif c0.startswith("H"):  # Hxy source
            common_term_1 = (pdotrhat / r * 0.25) / Z
            common_term_2 = rhatcrossp[..., None] * (omega * eps * 0.125)
            Ex = Ey = Hz = 0
            Ez = einsum(
                (rhatcrossp[..., None] * 0.25 * ik), H1, "n s f, b n s f -> b n f"
            )  # [bs, n, nf]
            Hx = einsum(
                -(rhat[..., 0:1] * common_term_1), H1, "n s f, b n s f -> b n f"
            ) + einsum(
                (rhat[..., 1:] * common_term_2), H0_minus_H2, "n s f, b n s f -> b n f"
            )
            Hy = einsum(
                -(rhat[..., 1:] * common_term_1), H1, "n s f, b n s f -> b n f"
            ) - einsum(
                (rhat[..., 0:1] * common_term_2), H0_minus_H2, "n s f, b n s f -> b n f"
            )

    else:
        raise ValueError("c0 must be 'Ez' or 'Hz'")

    return Ex, Ey, Ez, Hx, Hy, Hz


def get_farfields(
    nearfield_regions,  # list of nearfield monitor
    fields: Tensor,  # nearfield fields, entire fields
    x: Tensor,  # farfield points physical locatinos, in um (x, y)
    freqs: Tensor,
    eps: float,
    mu: float,
    component: str = "Ez",  # nearfield fields component
):
    """
    nearfield_regions: list of nearfield monitor, {monitor_name: {"slice": slice, "center": center, "size": size}}
    fields: [bs, X, Y, nf] a batch of nearfield fields, e.g., Ez
    x: [n, 2], batch size, n far field target points, 2 dimension, x and y
    freqs: [nf] frequencies
    eps: scalar, permittivity in the homogeneous medium
    mu: scalar, permeability in the homogeneous medium
    component: str, nearfield fields component
    """
    far_fields = {"Ex": 0, "Ey": 0, "Ez": 0, "Hx": 0, "Hy": 0, "Hz": 0}
    for name, nearfield_region in nearfield_regions.items():
        nearfield_slice = nearfield_region["slice"]
        f0 = fields[..., *nearfield_slice, :]  # [bs, s, nf]
        center = nearfield_region["center"]
        size = nearfield_region["size"]
        if size[0] == 0:  # vertical monitor
            n_src_points = nearfield_slice[1].shape[0]

            xs_y = torch.linspace(
                center[1] - size[1] / 2, center[1] + size[1] / 2, n_src_points
            )
            xs_x = torch.empty_like(xs_y).fill_(center[0])
        else:
            n_src_points = nearfield_slice[0].shape[0]  # horizontal monitor
            xs_x = torch.linspace(
                center[0] - size[0] / 2, center[0] + size[0] / 2, n_src_points
            )
            xs_y = torch.empty_like(xs_x).fill_(center[1])

        xs = torch.stack((xs_x, xs_y), dim=-1)  # [num_src_points, 2]

        Ex, Ey, Ez, Hx, Hy, Hz = green2d(
            x=x, freqs=freqs, eps=eps, mu=mu, x0=xs, f0=f0, c0=component
        )
        far_fields["Ex"] += Ex
        far_fields["Ey"] += Ey
        far_fields["Ez"] += Ez
        far_fields["Hx"] += Hx
        far_fields["Hy"] += Hy
        far_fields["Hz"] += Hz
    return far_fields


def get_flux(hx, hy, ez, monitor, grid_step, direction: str = "x", autograd=False):
    if autograd:
        ravel = npa.ravel
        real = npa.real
    else:
        ravel = np.ravel
        real = np.real
    if isinstance(ez, torch.Tensor):
        ravel = torch.ravel
        real = torch.real

    if direction == "x":
        h = (0.0, ravel(hy[monitor]), 0)
    elif direction == "y":
        h = (ravel(hx[monitor]), 0, 0)
    # The E-field is not co-located with the H-field in the Yee cell. Therefore,
    # we must sample at two neighboring pixels in the propataion direction and
    # then interpolate:
    e_yee_shifted = grid_average(ez, monitor, direction, autograd=autograd)

    e = (0.0, 0.0, e_yee_shifted)

    s = 0.5 * real(overlap(e, h, dl=grid_step * 1e-6, direction=direction))

    return s

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

def grid_average(e, monitor, direction: str = "x", autograd=False):
    if autograd:
        mean = npa.mean
    else:
        mean = np.mean
    if isinstance(e, torch.Tensor):
        mean = lambda x, axis: torch.mean(x, dim=axis)

    if direction == "x":
        if isinstance(monitor, Slice):
            if isinstance(monitor[0], torch.Tensor):
                e_monitor = (monitor[0] + torch.tensor([[-1], [0]], device=monitor[0].device), monitor[1])
                
                e_yee_shifted = torch.mean(e[e_monitor], dim=0)
            else:
                e_monitor = (monitor[0] + np.array([[-1], [0]]), monitor[1])

                e_yee_shifted = mean(e[e_monitor], axis=0)
        elif isinstance(monitor, np.ndarray):
            e_monitor = monitor.nonzero()
            e_monitor = (e_monitor[0], e_monitor[1] - 1)
            e_yee_shifted = (e[monitor] + e[e_monitor]) / 2
    elif direction == "y":
        if isinstance(monitor, Slice):
            if isinstance(monitor[0], torch.Tensor):
                e_monitor = (monitor[0], monitor[1] + torch.tensor([[-1], [0]], device=monitor[0].device))
                e_yee_shifted = torch.mean(e[e_monitor], dim=0)
            else:
                e_monitor = (monitor[0], monitor[1] + np.array([[-1], [0]]))
                e_yee_shifted = mean(e[e_monitor], axis=0)
        elif isinstance(monitor, np.ndarray):
            e_monitor = monitor.nonzero()
            e_monitor = (e_monitor[0] - 1, e_monitor[1])
            e_yee_shifted = (e[monitor] + e[e_monitor]) / 2
    return e_yee_shifted

def cross(a, b, direction="x"):
    """Compute the cross product between two VectorFields."""
    if direction == "x":
        return a[1] * b[2] - a[2] * b[1]
    elif direction == "y":
        return a[2] * b[0] - a[0] * b[2]
    elif direction == "z":
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
):
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

    if direction == "x":
        h = (0.0, ravel(hy[monitor]), 0)
        hm = (0.0, ht_m, 0.0)
        # The E-field is not co-located with the H-field in the Yee cell. Therefore,
        # we must sample at two neighboring pixels in the propataion direction and
        # then interpolate:
        e_yee_shifted = grid_average(ez, monitor, direction, autograd=autograd)

    elif direction == "y":
        h = (ravel(hx[monitor]), 0, 0)
        hm = (-ht_m, 0.0, 0.0)
        # The E-field is not co-located with the H-field in the Yee cell. Therefore,
        # we must sample at two neighboring pixels in the propataion direction and
        # then interpolate:
        e_yee_shifted = grid_average(ez, monitor, direction, autograd=autograd)

    e = (0.0, 0.0, e_yee_shifted)
    em = (0.0, 0.0, et_m)

    # print("this is the type of em: ", type(em[2])) # ndarray
    # print("this is the type of hy: ", type(hy[monitor])) # torch.Tensor

    dl = grid_step * 1e-6
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


Slice = collections.namedtuple("Slice", "x y")