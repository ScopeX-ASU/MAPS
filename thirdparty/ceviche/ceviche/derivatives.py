import autograd.numpy as npa
import numpy as np
import scipy.sparse as sp

from .constants import *

"""
This file contains functions related to performing derivative operations used in the simulation tools.
-  The FDTD method requires autograd-compatible curl operations, which are performed using numpy.roll
-  The FDFD method requires sparse derivative matrices, with PML added, which are constructed here.
"""


"""================================== CURLS FOR FDTD ======================================"""


def curl_E(axis, Ex, Ey, Ez, dL):
    if axis == 0:
        return (npa.roll(Ez, shift=-1, axis=1) - Ez) / dL - (
            npa.roll(Ey, shift=-1, axis=2) - Ey
        ) / dL
    elif axis == 1:
        return (npa.roll(Ex, shift=-1, axis=2) - Ex) / dL - (
            npa.roll(Ez, shift=-1, axis=0) - Ez
        ) / dL
    elif axis == 2:
        return (npa.roll(Ey, shift=-1, axis=0) - Ey) / dL - (
            npa.roll(Ex, shift=-1, axis=1) - Ex
        ) / dL


def curl_H(axis, Hx, Hy, Hz, dL):
    if axis == 0:
        return (Hz - npa.roll(Hz, shift=1, axis=1)) / dL - (
            Hy - npa.roll(Hy, shift=1, axis=2)
        ) / dL
    elif axis == 1:
        return (Hx - npa.roll(Hx, shift=1, axis=2)) / dL - (
            Hz - npa.roll(Hz, shift=1, axis=0)
        ) / dL
    elif axis == 2:
        return (Hy - npa.roll(Hy, shift=1, axis=0)) / dL - (
            Hx - npa.roll(Hx, shift=1, axis=1)
        ) / dL


"""======================= STUFF THAT CONSTRUCTS THE DERIVATIVE MATRIX ==========================="""


def compute_derivative_matrices(omega, shape, npml, dL, bloch_x=0.0, bloch_y=0.0):
    """Returns sparse derivative matrices.  Currently works for 2D and 1D
    omega: angular frequency (rad/sec)
    shape: shape of the FDFD grid
    npml: list of number of PML cells in x and y.
    dL: spatial grid size (m)
    block_x: bloch phase (phase across periodic boundary) in x
    block_y: bloch phase (phase across periodic boundary) in y
    """
    if isinstance(dL, float):
        dL = [dL] * 3
    dx, dy, dz = dL

    if len(shape) == 2:
        shape = shape + (1,)

    # Construct derivate matrices without PML
    Dxf = createDws_new("x", "f", shape, dx)
    Dxb = createDws_new("x", "b", shape, dx)
    Dyf = createDws_new("y", "f", shape, dy)
    Dyb = createDws_new("y", "b", shape, dy)

    if shape[-1] > 1:  # have z dimension for 3D simulation
        Dzf = createDws_new("z", "f", shape, dz)
        Dzb = createDws_new("z", "b", shape, dz)
    else:
        Dzf = Dzb = None

    # make the S-matrices for PML
    (Sxf, Sxb, Syf, Syb, Szf, Szb) = create_S_matrices_new(omega, shape, npml, dL)
    # (Sxf, Sxb, Syf, Syb) = create_S_matrices(omega, shape[:2], npml, dL[0])

    # apply PML to derivative matrices
    Dxf = Sxf.dot(Dxf)
    Dxb = Sxb.dot(Dxb)
    Dyf = Syf.dot(Dyf)
    Dyb = Syb.dot(Dyb)

    Dzf = Szf = None

    if Dzf is not None and Szf is not None:
        Dzf = Szf.dot(Dzf)
        Dzb = Szb.dot(Dzb)

    return Dxf, Dxb, Dyf, Dyb, Dzf, Dzb


""" Derivative Matrices (no PML) """


def createDws_new(component: str, dir: str, shape, dL: float):
    """
    s = 'x' or 'y': x derivative or y derivative
    f = 'b' or 'f'
    catches exceptions if s and f are misspecified
    """
    M = np.prod(shape)

    sign = 1 if dir == "f" else -1

    indices = np.reshape(np.arange(M), shape, order="C")

    if component == "x":
        ind_adj = np.roll(indices, -sign, axis=0)
    elif component == "y":
        ind_adj = np.roll(indices, -sign, axis=1)
    elif component == "z":
        ind_adj = np.roll(indices, -sign, axis=2)

    # we could use flatten here since the indices are already in 'F' order
    indices_flatten = np.reshape(indices, (M,), order="F")
    indices_adj_flatten = np.reshape(ind_adj, (M,), order="F")
    on_inds = np.hstack((indices_flatten, indices_flatten))
    off_inds = np.concatenate((indices_flatten, indices_adj_flatten), axis=0)
    all_inds = np.concatenate(
        (np.expand_dims(on_inds, axis=1), np.expand_dims(off_inds, axis=1)), axis=1
    )
    data_1 = sign / dL * np.ones((M))
    data = np.concatenate((-data_1, data_1), axis=0)
    Dws = sp.csc_matrix((data, (all_inds[:, 0], all_inds[:, 1])), shape=(M, M))

    return Dws


""" PML Functions """


def create_S_matrices_new(omega, shape, npml, dL):
    """Makes the 'S-matrices'.  When dotted with derivative matrices, they add PML"""

    # strip out some information needed
    if isinstance(dL, float):
        dL = [dL] * 3

    dx, dy, dz = dL
    if len(shape) == 2:
        shape = tuple(shape) + (1,)
    if len(npml) == 2:
        npml = tuple(npml) + (0,)

    Nx, Ny, Nz = shape
    N = np.prod(shape)
    Nx_pml, Ny_pml, Nz_pml = npml

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor("f", omega, dx, Nx, Nx_pml)
    s_vector_x_b = create_sfactor("b", omega, dx, Nx, Nx_pml)
    s_vector_y_f = create_sfactor("f", omega, dy, Ny, Ny_pml)
    s_vector_y_b = create_sfactor("b", omega, dy, Ny, Ny_pml)

    if shape[-1] > 1:  # have z dimension for 3D simulation
        s_vector_z_f = create_sfactor("f", omega, dz, Nz, Nz_pml)
        s_vector_z_b = create_sfactor("b", omega, dz, Nz, Nz_pml)
        # now we create the matrix (i.e. repeat sxf Ny times repeat Syf Nx times)
        Sx_f, Sy_f, Sz_f = np.meshgrid(
            1 / s_vector_x_f, 1 / s_vector_y_f, 1 / s_vector_z_f, indexing="ij"
        )
        Sx_b, Sy_b, Sz_b = np.meshgrid(
            1 / s_vector_x_b, 1 / s_vector_y_b, 1 / s_vector_z_b, indexing="ij"
        )
    else:  # 2D simulation
        # now we create the matrix (i.e. repeat sxf Ny times repeat Syf Nx times)
        Sx_f, Sy_f = np.meshgrid(1 / s_vector_x_f, 1 / s_vector_y_f, indexing="ij")
        Sx_b, Sy_b = np.meshgrid(1 / s_vector_x_b, 1 / s_vector_y_b, indexing="ij")
        Sz_f = Sz_b = None

    # Reshape the 2D s-factors into a 1D s-vecay
    Sx_f = Sx_f.flatten(order="C")
    Sx_b = Sx_b.flatten(order="C")
    Sy_f = Sy_f.flatten(order="C")
    Sy_b = Sy_b.flatten(order="C")

    if Sz_f is not None:
        Sz_f = Sz_f.flatten(order="C")
        Sz_b = Sz_b.flatten(order="C")
        Sz_f = sp.spdiags(Sz_f, 0, N, N)
        Sz_b = sp.spdiags(Sz_b, 0, N, N)

    # Construct the 1D total s-vecay into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f, 0, N, N)
    Sx_b = sp.spdiags(Sx_b, 0, N, N)
    Sy_f = sp.spdiags(Sy_f, 0, N, N)
    Sy_b = sp.spdiags(Sy_b, 0, N, N)

    return Sx_f, Sx_b, Sy_f, Sy_b, Sz_f, Sz_b


def create_sfactor(dir, omega, dL, N, N_pml):
    """creates the S-factor cross section needed in the S-matrices"""

    #  for no PNL, this should just be zero
    if N_pml == 0:
        return np.ones(N, dtype=np.complex128)

    # otherwise, get different profiles for forward and reverse derivative matrices
    dw = N_pml * dL
    if dir == "f":
        return create_sfactor_f(omega, dL, N, N_pml, dw)
    elif dir == "b":
        return create_sfactor_b(omega, dL, N, N_pml, dw)
    else:
        raise ValueError("Dir value {} not recognized".format(dir))


def create_sfactor_f(omega, dL, N, N_pml, dw):
    """S-factor profile for forward derivative matrix"""
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= N_pml:
            sfactor_array[i] = s_value(dL * (N_pml - i + 0.5), dw, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 0.5), dw, omega)
    return sfactor_array


def create_sfactor_b(omega, dL, N, N_pml, dw):
    """S-factor profile for backward derivative matrix"""
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= N_pml:
            sfactor_array[i] = s_value(dL * (N_pml - i + 1), dw, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 1), dw, omega)
    return sfactor_array


def sig_w(l, dw, m=3, lnR=-30):
    """Fictional conductivity, note that these values might need tuning"""
    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw) ** m


def s_value(l, dw, omega):
    """S-value to use in the S-matrices"""
    return 1 - 1j * sig_w(l, dw) / (omega * EPSILON_0)
