import numpy as np
import torch

from thirdparty.ceviche.constants import EPSILON_0, ETA_0

"""
This file contains functions related to performing derivative operations used in the simulation tools.
-  The FDTD method requires autograd-compatible curl operations, which are performed using numpy.roll
-  The FDFD method requires sparse derivative matrices, with PML added, which are constructed here.
"""


"""================================== CURLS FOR FDTD ======================================"""

__all__ = [
    "curl_E",
    "curl_H",
    "compute_derivative_matrices",
    "createDws",
    "make_Dxf",
    "make_Dxb",
    "make_Dyf",
    "make_Dyb",
    "create_S_matrices",
    "create_sfactor",
    "create_sfactor_f",
    "create_sfactor_b",
    "sig_w",
    "s_value",
]


def sparse_eye(n, device="cuda:0"):
    diags = torch.ones(n, dtype=torch.float64)
    return torch.sparse.spdiags(
        diags, torch.tensor([0]), (n, n), layout=torch.sparse_coo
    ).to(device)


def sparse_kron(input: torch.Tensor, other: torch.Tensor):
    assert input.ndim == other.ndim
    input = input.coalesce()
    other = other.coalesce()
    input_indices = input.indices()
    other_indices = other.indices()

    input_indices_expanded = input_indices.expand(
        other_indices.shape[1], *input_indices.shape
    ).T * torch.tensor(other.shape, device=input.device).reshape(1, -1, 1)
    other_indices_expanded = other_indices.expand(
        input_indices.shape[1], *other_indices.shape
    )
    new_indices = torch.permute(
        input_indices_expanded + other_indices_expanded, (1, 0, 2)
    ).reshape(input.ndim, -1)

    new_values = torch.kron(input.values(), other.values())

    if new_indices.ndim == 1:
        new_indices = new_indices.reshape([input.ndim, 0])

    new_shape = [n * m for n, m in zip(input.shape, other.shape)]
    return torch.sparse_coo_tensor(
        new_indices,
        new_values,
        new_shape,
        device=input.device,
        is_coalesced=True,
    )


def curl_E(axis, Ex, Ey, Ez, dL):
    if axis == 0:
        return (
            torch.roll(Ez, shifts=-1, dims=1)
            - Ez
            - (torch.roll(Ey, shifts=-1, dims=2) - Ey)
        ) / dL
    elif axis == 1:
        return (
            torch.roll(Ex, shifts=-1, dims=2)
            - Ex
            - (torch.roll(Ez, shifts=-1, dims=0) - Ez)
        ) / dL
    elif axis == 2:
        return (
            torch.roll(Ey, shifts=-1, dims=0)
            - Ey
            - (torch.roll(Ex, shifts=-1, dims=1) - Ex)
        ) / dL


def curl_H(axis, Hx, Hy, Hz, dL):
    if axis == 0:
        return (
            Hz
            - torch.roll(Hz, shifts=1, dims=1)
            - (Hy - torch.roll(Hy, shifts=1, dims=2))
        ) / dL
    elif axis == 1:
        return (
            Hx
            - torch.roll(Hx, shifts=1, dims=2)
            - (Hz - torch.roll(Hz, shifts=1, dims=0))
        ) / dL
    elif axis == 2:
        return (
            Hy
            - torch.roll(Hy, shifts=1, dims=0)
            - (Hx - torch.roll(Hx, shifts=1, dims=1))
        ) / dL


"""======================= STUFF THAT CONSTRUCTS THE DERIVATIVE MATRIX ==========================="""


def compute_derivative_matrices(
    omega, shape, npml, dL, bloch_x=0.0, bloch_y=0.0, device="cuda:0"
):
    """Returns sparse derivative matrices.  Currently works for 2D and 1D
    omega: angular frequency (rad/sec)
    shape: shape of the FDFD grid
    npml: list of number of PML cells in x and y.
    dL: spatial grid size (m)
    block_x: bloch phase (phase across periodic boundary) in x
    block_y: bloch phase (phase across periodic boundary) in y
    """

    # Construct derivate matrices without PML
    Dxf = createDws(
        "x", "f", shape, dL, bloch_x=bloch_x, bloch_y=bloch_y, device=device
    )
    Dxb = createDws(
        "x", "b", shape, dL, bloch_x=bloch_x, bloch_y=bloch_y, device=device
    )
    Dyf = createDws(
        "y", "f", shape, dL, bloch_x=bloch_x, bloch_y=bloch_y, device=device
    )
    Dyb = createDws(
        "y", "b", shape, dL, bloch_x=bloch_x, bloch_y=bloch_y, device=device
    )

    # make the S-matrices for PML
    (Sxf, Sxb, Syf, Syb) = create_S_matrices(omega, shape, npml, dL, device=device)

    # apply PML to derivative matrices
    # Dxf = torch.sparse.mm(Sxf, Dxf)
    # Dxb = torch.sparse.mm(Sxb, Dxb)
    # Dyf = torch.sparse.mm(Syf, Dyf)
    # Dyb = torch.sparse.mm(Syb, Dyb)

    Dxf = Sxf.unsqueeze(1) * Dxf
    Dxb = Sxb.unsqueeze(1) * Dxb
    Dyf = Syf.unsqueeze(1) * Dyf
    Dyb = Syb.unsqueeze(1) * Dyb

    return (
        Dxf.to_sparse_coo().coalesce(),
        Dxb.to_sparse_coo().coalesce(),
        Dyf.to_sparse_coo().coalesce(),
        Dyb.to_sparse_coo().coalesce(),
    )


""" Derivative Matrices (no PML) """


def createDws(component, dir, shape, dL, bloch_x=0.0, bloch_y=0.0, device="cuda:0"):
    """creates the derivative matrices
    component: one of 'x' or 'y' for derivative in x or y direction
    dir: one of 'f' or 'b', whether to take forward or backward finite difference
    shape: shape of the FDFD grid
    dL: spatial grid size (m)
    block_x: bloch phase (phase across periodic boundary) in x
    block_y: bloch phase (phase across periodic boundary) in y
    """

    Nx, Ny = shape

    # special case, a 1D problem
    if component == "x" and Nx == 1:
        return sparse_eye(Ny, device=device)

    if component == "y" and Ny == 1:
        return sparse_eye(Nx, device=device)
        # return sp.eye(Nx)

    # select a `make_D` function based on the component and direction
    component_dir = component + dir
    if component_dir == "xf":
        return make_Dxf(dL, shape, bloch_x=bloch_x, device=device)
    elif component_dir == "xb":
        return make_Dxb(dL, shape, bloch_x=bloch_x, device=device)
    elif component_dir == "yf":
        return make_Dyf(dL, shape, bloch_y=bloch_y, device=device)
    elif component_dir == "yb":
        return make_Dyb(dL, shape, bloch_y=bloch_y, device=device)
    else:
        raise ValueError(
            "component and direction {} and {} not recognized".format(component, dir)
        )


def make_Dxf(dL, shape, bloch_x=0.0, device="cuda:0"):
    """Forward derivative in x"""
    Nx, Ny = shape
    phasor_x = np.exp(1j * bloch_x)
    # Dxf = sp.diags(
    #     [-1, 1, phasor_x], [0, 1, -Nx + 1], shape=(Nx, Nx), dtype=np.complex128
    # )
    Dxf = (
        torch.sparse.spdiags(
            torch.vstack([-torch.ones(Nx), torch.ones(Nx), torch.ones(Nx) * phasor_x]),
            torch.tensor([0, 1, -Nx + 1]),
            (Nx, Nx),
            layout=torch.sparse_coo,
        )
        .to(device)
        .to(torch.complex128)
    )
    # Dxf = 1 / dL * sp.kron(Dxf, sp.eye(Ny))

    Dxf = 1 / dL * sparse_kron(Dxf, sparse_eye(Ny, device=device))

    return Dxf


def make_Dxb(dL, shape, bloch_x=0.0, device="cuda:0"):
    """Backward derivative in x"""
    Nx, Ny = shape
    phasor_x = np.exp(1j * bloch_x)
    # Dxb = sp.diags(
    #     [1, -1, -np.conj(phasor_x)],
    #     [0, -1, Nx - 1],
    #     shape=(Nx, Nx),
    #     dtype=np.complex128,
    # )
    Dxb = (
        torch.sparse.spdiags(
            torch.vstack(
                [torch.ones(Nx), -torch.ones(Nx), torch.ones(Nx) * (-np.conj(phasor_x))]
            ),
            torch.tensor([0, -1, Nx - 1]),
            (Nx, Nx),
            layout=torch.sparse_coo,
        )
        .to(device)
        .to(torch.complex128)
    )

    # Dxb = 1 / dL * sp.kron(Dxb, sp.eye(Ny))
    Dxb = 1 / dL * sparse_kron(Dxb, sparse_eye(Ny, device=device))
    return Dxb


def make_Dyf(dL, shape, bloch_y=0.0, device="cuda:0"):
    """Forward derivative in y"""
    Nx, Ny = shape
    phasor_y = np.exp(1j * bloch_y)
    # Dyf = sp.diags([-1, 1, phasor_y], [0, 1, -Ny + 1], shape=(Ny, Ny))
    Dyf = (
        torch.sparse.spdiags(
            torch.vstack([-torch.ones(Ny), torch.ones(Ny), torch.ones(Ny) * phasor_y]),
            torch.tensor([0, 1, -Ny + 1]),
            (Ny, Ny),
            layout=torch.sparse_coo,
        )
        .to(device)
        .to(torch.complex128)
    )
    # Dyf = 1 / dL * sp.kron(sp.eye(Nx), Dyf)
    Dyf = 1 / dL * sparse_kron(sparse_eye(Nx, device=device), Dyf)
    return Dyf


def make_Dyb(dL, shape, bloch_y=0.0, device="cuda:0"):
    """Backward derivative in y"""
    Nx, Ny = shape
    phasor_y = np.exp(1j * bloch_y)
    Dyb = (
        torch.sparse.spdiags(
            torch.vstack(
                [torch.ones(Ny), -torch.ones(Ny), torch.ones(Ny) * (-np.conj(phasor_y))]
            ),
            torch.tensor([0, -1, Ny - 1]),
            (Ny, Ny),
            layout=torch.sparse_coo,
        )
        .to(device)
        .to(torch.complex128)
    )
    # Dyb = sp.diags([1, -1, -np.conj(phasor_y)], [0, -1, Ny - 1], shape=(Ny, Ny))
    # Dyb = 1 / dL * sp.kron(sp.eye(Nx), Dyb)
    Dyb = 1 / dL * sparse_kron(sparse_eye(Nx, device=device), Dyb)
    return Dyb


""" PML Functions """


def create_S_matrices(omega, shape, npml, dL, device="cuda:0"):
    """Makes the 'S-matrices'.  When dotted with derivative matrices, they add PML"""

    # strip out some information needed
    Nx, Ny = shape
    N = Nx * Ny
    x_range = [0, float(dL * Nx)]
    y_range = [0, float(dL * Ny)]
    Nx_pml, Ny_pml = npml

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor("f", omega, dL, Nx, Nx_pml, device=device)
    s_vector_x_b = create_sfactor("b", omega, dL, Nx, Nx_pml, device=device)
    s_vector_y_f = create_sfactor("f", omega, dL, Ny, Ny_pml, device=device)
    s_vector_y_b = create_sfactor("b", omega, dL, Ny, Ny_pml, device=device)

    # Fill the 2D space with layers of appropriate s-factors
    # Sx_f_2D = torch.zeros(shape, dtype=torch.complex128, device=device)
    # Sx_b_2D = torch.zeros(shape, dtype=torch.complex128, device=device)
    # Sy_f_2D = torch.zeros(shape, dtype=torch.complex128, device=device)
    # Sy_b_2D = torch.zeros(shape, dtype=torch.complex128, device=device)

    # insert the cross sections into the S-grids (could be done more elegantly)
    Sx_f_2D = (1 / s_vector_x_f[:, None]).repeat(1, Ny)
    Sx_b_2D = (1 / s_vector_x_b[:, None]).repeat(1, Ny)
    Sy_f_2D = (1 / s_vector_y_f[None]).repeat(Nx, 1)
    Sy_b_2D = (1 / s_vector_y_b[None]).repeat(Nx, 1)
    # for i in range(0, Ny):
    #     Sx_f_2D[:, i] = 1 / s_vector_x_f
    #     Sx_b_2D[:, i] = 1 / s_vector_x_b
    # for i in range(0, Nx):
    #     Sy_f_2D[i, :] = 1 / s_vector_y_f
    #     Sy_b_2D[i, :] = 1 / s_vector_y_b

    # Reshape the 2D s-factors into a 1D s-vecay
    Sx_f_vec = Sx_f_2D.flatten()
    Sx_b_vec = Sx_b_2D.flatten()
    Sy_f_vec = Sy_f_2D.flatten()
    Sy_b_vec = Sy_b_2D.flatten()
    return Sx_f_vec, Sx_b_vec, Sy_f_vec, Sy_b_vec

    # Construct the 1D total s-vecay into a diagonal matrix
    # offset = torch.tensor([0])
    # Sx_f = torch.sparse.spdiags(Sx_f_vec, offset, [N, N], layout=torch.sparse_coo).to(device)
    # Sx_b = torch.sparse.spdiags(Sx_b_vec, offset, [N, N], layout=torch.sparse_coo).to(device)
    # Sy_f = torch.sparse.spdiags(Sy_f_vec, offset, [N, N], layout=torch.sparse_coo).to(device)
    # Sy_b = torch.sparse.spdiags(Sy_b_vec, offset, [N, N], layout=torch.sparse_coo).to(device)

    # Sx_f = sp.spdiags(Sx_f_vec, 0, N, N)
    # Sx_b = sp.spdiags(Sx_b_vec, 0, N, N)
    # Sy_f = sp.spdiags(Sy_f_vec, 0, N, N)
    # Sy_b = sp.spdiags(Sy_b_vec, 0, N, N)

    # return Sx_f, Sx_b, Sy_f, Sy_b


def create_sfactor(dir, omega, dL, N, N_pml, device="cuda:0"):
    """creates the S-factor cross section needed in the S-matrices"""

    #  for no PNL, this should just be zero
    if N_pml == 0:
        return torch.ones(N, dtype=torch.complex128, deice=device)

    # otherwise, get different profiles for forward and reverse derivative matrices
    dw = N_pml * dL
    if dir == "f":
        return create_sfactor_f(omega, dL, N, N_pml, dw, device=device)
    elif dir == "b":
        return create_sfactor_b(omega, dL, N, N_pml, dw, device=device)
    else:
        raise ValueError("Dir value {} not recognized".format(dir))


def create_sfactor_f(omega, dL, N, N_pml, dw, device="cuda:0"):
    """S-factor profile for forward derivative matrix"""
    sfactor_array = torch.ones(N, dtype=torch.complex128, device=device)
    ### Jiaqi: Why asymnetric?

    sfactor_array[: N_pml + 1] = s_value(
        dL * (N_pml - torch.arange(N_pml + 1, device=device) + 0.5), dw, omega
    )
    sfactor_array[N - N_pml + 1 :] = s_value(
        dL * (torch.arange(N - N_pml + 1, N, device=device) - (N - N_pml) - 0.5),
        dw,
        omega,
    )
    # for i in range(N):
    #     if i <= N_pml:
    #         sfactor_array[i] = s_value(dL * (N_pml - i + 0.5), dw, omega)
    #     elif i > N - N_pml:
    #         sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 0.5), dw, omega)
    return sfactor_array


def create_sfactor_b(omega, dL, N, N_pml, dw, device="cuda:0"):
    """S-factor profile for backward derivative matrix"""
    # sfactor_array = np.ones(N, dtype=np.complex128)
    sfactor_array = torch.ones(N, dtype=torch.complex128, device=device)
    sfactor_array[: N_pml + 1] = s_value(
        dL * (N_pml - torch.arange(N_pml + 1, device=device) + 1), dw, omega
    )
    sfactor_array[N - N_pml + 1 :] = s_value(
        dL * (torch.arange(N - N_pml + 1, N, device=device) - (N - N_pml) - 1),
        dw,
        omega,
    )

    # for i in range(N):
    #     if i <= N_pml:
    #         sfactor_array[i] = s_value(dL * (N_pml - i + 1), dw, omega)
    #     elif i > N - N_pml:
    #         sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 1), dw, omega)
    return sfactor_array


def sig_w(l, dw, m=3, lnR=-30):
    """Fictional conductivity, note that these values might need tuning"""
    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw) ** m


def s_value(l, dw, omega):
    """S-value to use in the S-matrices"""
    return 1 - 1j * sig_w(l, dw) / (omega * EPSILON_0)
