import collections
import os
from copy import deepcopy
from functools import lru_cache
from typing import Callable, List, Tuple
from core.ceviche import ceviche
import matplotlib.patches as patches
import matplotlib.pylab as plt
import numpy as np
import torch
from autograd import numpy as npa
from core.ceviche.ceviche import constants, jacobian
from core.ceviche.ceviche.modes import get_modes
from meep.materials import Si, SiO2
from pyutils.general import ensure_dir
from scipy.ndimage import zoom
from torch import Tensor
from torch.types import Device
# from core.pytorch_sparse.torch_sparse import spmm
from torch_sparse import spmm
from pyutils.general import print_stat
from core.ceviche.ceviche.primitives import sp_solve, sp_mult, spsp_mult
from core.ceviche.ceviche.derivatives import compute_derivative_matrices

__all__ = [
    "material_fn_dict",
    "Slice",
    "get_grid",
    "apply_regions_gpu",
    "Si_eps",
    "SiO2_eps",
    "AdjointGradient",
    "differentiable_boundary",
    "BinaryProjection",
    "LevelSetInterp",
    "get_eps",
    "ApplyLowerLimit",
    "ApplyUpperLimit",
    "ApplyBothLimit",
    "HeavisideProjectionLayer",
    "heightProjectionLayer",
    "InsensitivePeriodLayer",
    "poynting_vector",
    "plot_eps_field",
    "get_eigenmode_coefficients",
    "insert_mode",
    "ObjectiveFunc",
]


Slice = collections.namedtuple("Slice", "x y")


def get_grid(shape, dl):
    # dl in um
    # computes the coordinates in the grid

    (Nx, Ny) = shape
    # if Ny % 2 == 0:
    #     Ny -= 1
    # coordinate vectors
    x_coord = np.linspace(-(Nx - 1) / 2 * dl, (Nx - 1) / 2 * dl, Nx)
    y_coord = np.linspace(-(Ny - 1) / 2 * dl, (Ny - 1) / 2 * dl, Ny)

    # x and y coordinate arrays
    xs, ys = np.meshgrid(x_coord, y_coord, indexing="ij")
    return (xs, ys)


def apply_regions_gpu(reg_list, xs, ys, eps_r_list, eps_bg, device="cuda"):
    # Convert inputs to tensors and move them to the GPU
    xs = torch.tensor(xs, device=device)
    ys = torch.tensor(ys, device=device)

    # Handle scalars to lists conversion
    if isinstance(eps_r_list, (int, float)):
        eps_r_list = [eps_r_list] * len(reg_list)
    if not isinstance(reg_list, (list, tuple)):
        reg_list = [reg_list]

    # Initialize permittivity tensor with background value
    eps_r = torch.full(xs.shape, eps_bg, device=device, dtype=torch.float32)

    # Convert region functions to a vectorized form using PyTorch operations
    for e, reg in zip(eps_r_list, reg_list):
        # Assume that reg is a lambda or function that can be applied to tensors
        material_mask = reg(xs, ys)  # This should return a boolean tensor
        # print("this is the dtype of the eps_r", eps_r.dtype)
        # print("this is the dtype of the e", e.dtype)
        eps_r[material_mask] = e

    return eps_r.cpu().numpy()


@lru_cache(maxsize=64)
def Si_eps(wavelength):
    """Returns the permittivity of silicon at the given wavelength"""
    return 3.48**2
    return Si.epsilon(1 / wavelength)[0, 0].real


@lru_cache(maxsize=64)
def SiO2_eps(wavelength):
    """Returns the permittivity of silicon at the given wavelength"""
    return 1.44**2
    return SiO2.epsilon(1 / wavelength)[0, 0].real


@lru_cache(maxsize=64)
def SiN_eps(wavelength):
    """Returns the permittivity of silicon at the given wavelength"""
    return 2.45**2
    return SiO2.epsilon(1 / wavelength)[0, 0].real


material_fn_dict = {
    "Si": Si_eps,
    "SiO2": SiO2_eps,
    "SiN": SiN_eps,
}


class AdjointGradient(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        obj_and_grad_fn: Callable,
        adjoint_mode: str,
        resolution: int,
        *permittivity_list: List[Tensor],
    ) -> Tensor:
        obj = obj_and_grad_fn(adjoint_mode, "need_value", resolution, permittivity_list)

        ctx.save_for_backward(*permittivity_list)
        ctx.save_adjoint_mode = adjoint_mode
        ctx.save_obj_and_grad_fn = obj_and_grad_fn
        ctx.save_resolution = resolution
        obj = torch.tensor(
            obj,
            device=permittivity_list[0].device,
            dtype=permittivity_list[0].dtype,
            requires_grad=True,
        )
        return obj

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        permittivity_list, adjoint_mode, obj_and_grad_fn, resolution = (
            ctx.saved_tensors,
            ctx.save_adjoint_mode,
            ctx.save_obj_and_grad_fn,
            ctx.save_resolution,
        )
        grad = obj_and_grad_fn(
            adjoint_mode, "need_gradient", resolution, permittivity_list
        )

        gradients = []
        if adjoint_mode == "reflection":
            if isinstance(grad, np.ndarray):  # make sure the gradient is torch tensor
                grad = (
                    torch.from_numpy(grad)
                    .to(permittivity_list[0].device)
                    .to(permittivity_list[0].dtype)
                )
            grad = grad.view_as(permittivity_list[0])
            gradients.append(grad_output * grad)
            return None, None, None, *gradients
        if adjoint_mode == "legume":
            if isinstance(grad, np.ndarray):  # make sure the gradient is torch tensor
                grad = (
                    torch.from_numpy(grad)
                    .to(permittivity_list[0].device)
                    .to(permittivity_list[0].dtype)
                )
            grad = grad.view_as(permittivity_list[0])
            gradients.append(grad_output * grad)
        else:
            if isinstance(
                grad, list
            ):  # which means that there are multiple design regions
                for i, g in enumerate(grad):
                    if isinstance(
                        g, np.ndarray
                    ):  # make sure the gradient is torch tensor
                        g = (
                            torch.from_numpy(g)
                            .to(permittivity_list[i].device)
                            .to(permittivity_list[i].dtype)
                        )

                    if (
                        len(g.shape) == 2
                    ):  # summarize the gradient along different frequencies
                        g = torch.sum(g, dim=-1)
                    g = g.view_as(permittivity_list[i])
                    gradients.append(grad_output * g)
            else:
                # there are two possibility:
                #   1. there is only one design region and the grad is a ndarray
                #   2. the mode is legume
                if isinstance(
                    grad, np.ndarray
                ):  # make sure the gradient is torch tensor
                    grad = (
                        torch.from_numpy(grad)
                        .to(permittivity_list[0].device)
                        .to(permittivity_list[0].dtype)
                    )

                # if len(grad.shape) == 2:  # summarize the gradient along different frquencies
                #     grad = torch.sum(grad, dim=-1)
                if adjoint_mode == "fdtd":
                    grad = grad.view_as(permittivity_list[0])
                elif adjoint_mode == "fdfd_angler":
                    Nx = int(grad.numel() // permittivity_list[0].shape[1])
                    grad = grad.view(Nx, permittivity_list[0].shape[1])
                elif "ceviche" in adjoint_mode:
                    if len(grad.shape) == 2:
                        Nx = round(grad.numel() // permittivity_list[0].shape[1])
                        grad = grad.view(Nx, permittivity_list[0].shape[1])
                        # print("this is the gradient in the custom function: ", grad)
                    elif len(grad.shape) == 3:
                        Nx = round(grad[0].numel() // permittivity_list[0].shape[1])
                        grad = grad.view(-1, Nx, permittivity_list[0].shape[1])
                else:
                    raise ValueError(f"mode {adjoint_mode} is not supported")
                gradients.append(grad_output * grad)
        return None, None, None, *gradients


class differentiable_boundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, total_length, T):
        ctx.save_for_backward(w)
        ctx.x = x
        ctx.total_length = total_length
        ctx.T = T
        w1 = total_length - w
        output = torch.where(
            x < -w / 2,
            1
            / (
                torch.exp(
                    -(((x + w / 2 + w1 / 2) ** 2 - (w1 / 2) ** 2) / T)
                    * (total_length / (3 * w1)) ** 2
                )
                + 1
            ),
            torch.where(
                x < w / 2,
                1
                / (
                    torch.exp(
                        ((x**2 - (w / 2) ** 2) / T) * (total_length / (3 * w)) ** 2
                    )
                    + 1
                ),
                1
                / (
                    torch.exp(
                        -(((x - w / 2 - w1 / 2) ** 2 - (w1 / 2) ** 2) / T)
                        * (total_length / (3 * w1)) ** 2
                    )
                    + 1
                ),
            ),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (w,) = ctx.saved_tensors
        x = ctx.x
        total_length = ctx.total_length
        T = ctx.T

        w1 = total_length - w

        # Precompute common expressions
        exp1 = torch.exp(
            -(((x + w / 2 + w1 / 2) ** 2 - (w1 / 2) ** 2) / T)
            * (total_length / (3 * w1)) ** 2
        )
        exp2 = torch.exp(((x**2 - (w / 2) ** 2) / T) * (total_length / (3 * w)) ** 2)
        exp3 = torch.exp(
            -(((x - w / 2 - w1 / 2) ** 2 - (w1 / 2) ** 2) / T)
            * (total_length / (3 * w1)) ** 2
        )

        denominator1 = (exp1 + 1) ** 2
        denominator2 = (exp2 + 1) ** 2
        denominator3 = (exp3 + 1) ** 2

        doutput_dw = torch.where(
            x < -w / 2,
            -exp1
            * (-2 * total_length**2 * (x + total_length / 2) ** 2)
            / (9 * w1**3 * T * denominator1),
            torch.where(
                x < w / 2,
                -exp2 * (-2 * total_length**2 * x**2) / (9 * w**3 * T * denominator2),
                -exp3
                * (-2 * total_length**2 * (x - total_length / 2) ** 2)
                / (9 * w1**3 * T * denominator3),
            ),
        )

        # not quite sure with the following code
        grad_w = (grad_output * doutput_dw).sum()

        return None, grad_w, None, None


class BinaryProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permittivity: Tensor, T_bny: float, T_threshold: float):
        ctx.T_bny = T_bny
        ctx.T_threshold = T_threshold
        ctx.save_for_backward(permittivity)
        result = (torch.tanh((0.5 - permittivity) / T_bny) + 1) / 2
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # if T_bny is larger than T_threshold, then use the automatic differentiation of the tanh function
        # if the T_bny is smaller than T_threshold, then use the gradient as if T_bny is T_threshold
        T_bny = ctx.T_bny
        T_threshold = ctx.T_threshold
        (permittivity,) = ctx.saved_tensors

        if T_bny > T_threshold:
            grad = (
                -grad_output
                * (1 - torch.tanh((0.5 - permittivity) / T_bny) ** 2)
                / T_bny
            )
        else:
            grad = (
                -grad_output
                * (1 - torch.tanh((0.5 - permittivity) / T_threshold) ** 2)
                / T_threshold
            )

        return grad, None, None


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


# Function to plot the level set surface.
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


class LevelSetInterp1D(object):
    """This class implements the level set surface using Gaussian radial basis functions in 1D."""

    def __init__(
        self,
        x0: Tensor = None,  # 1D input coordinates
        z0: Tensor = None,  # Corresponding level set values
        sigma: float = None,  # Gaussian RBF standard deviation
    ):
        # Input data
        self.x0 = x0  # 1D coordinates
        self.z0 = z0  # Level set values
        self.sig = sigma  # Gaussian kernel width

        # Builds the level set interpolation model
        gauss_kernel = self.gaussian(self.x0, self.x0)
        self.model = torch.linalg.solve(
            gauss_kernel, self.z0
        )  # Solving gauss_kernel @ model = z0

    def gaussian(self, xi, xj):
        # Compute the Gaussian RBF kernel
        dist = torch.abs(xi.reshape(-1, 1) - xj.reshape(1, -1))
        return torch.exp(-(dist**2) / (2 * self.sig**2))

    def get_ls(self, x1):
        # Interpolate the level set function at new points x1
        gauss_matrix = self.gaussian(self.x0, x1)
        ls = gauss_matrix.T @ self.model
        return ls


def get_eps_1d(
    design_param,
    x_rho,
    x_phi,
    rho_size,
    nx_rho,
    nx_phi,
    plot_levelset=False,
    sharpness=0.1,
):
    """Returns the permittivities defined by the zero level set isocontour for a 1D case"""

    # Initialize the LevelSetInterp model for 1D case
    phi_model = LevelSetInterp1D(x0=x_rho, z0=design_param, sigma=rho_size)

    # Obtain the level set function phi
    phi = phi_model.get_ls(x1=x_phi)

    eps_phi = 0.5 * (torch.tanh(sharpness * phi) + 1)

    # Reshape the design parameters into a 1D array
    eps = torch.reshape(eps_phi, (nx_phi,))

    # Plot the level set surface if required
    if plot_levelset:
        rho = np.reshape(design_param, (nx_rho,))
        phi = np.reshape(phi, (nx_phi,))
        plot_level_set_1d(x0=x_rho, rho=rho, x1=x_phi, phi=phi)

    return eps


# Function to plot the level set in 1D
def plot_level_set_1d(x0, rho, x1, phi, path="./1D_Level_Set_Plot.png"):
    """
    Plots the level set for the 1D case.

    x0: array-like, coordinates corresponding to design parameters
    rho: array-like, design parameters
    x1: array-like, coordinates where phi is evaluated
    phi: array-like, level set values
    """

    fig, ax1 = plt.subplots(figsize=(12, 6), tight_layout=True)

    # Plot the design parameters as scatter plot
    ax1.scatter(x0, rho, color="black", label="Design Parameters")

    # Plot the level set function
    ax1.plot(x1, phi, color="blue", label="Level Set Function")

    # Highlight the zero level set
    ax1.axhline(0, color="red", linestyle="--", label="Zero Level Set")

    ax1.set_title("1D Level Set Plot")
    ax1.set_xlabel("x ($\mu m$)")
    ax1.set_ylabel("Value")
    ax1.legend()

    plt.savefig(path)


class ApplyLowerLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lower_limit):
        ctx.save_for_backward(x)
        ctx.lower_limit = lower_limit
        return torch.maximum(x, lower_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors
        lower_limit = ctx.lower_limit

        # Compute gradient
        # If x > lower_limit, propagate grad_output normally
        # If x <= lower_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
        )  # None for lower_limit since it does not require gradients


class ApplyUpperLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, upper_limit):
        ctx.save_for_backward(x)
        ctx.upper_limit = upper_limit
        return torch.minimum(x, upper_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors
        upper_limit = ctx.upper_limit

        # Compute gradient
        # If x > upper_limit, propagate grad_output normally
        # If x <= upper_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
        )  # None for upper_limit since it does not require gradients


class ApplyBothLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, upper_limit, lower_limit):
        ctx.save_for_backward(x)
        ctx.upper_limit = upper_limit
        ctx.lower_limit = lower_limit
        return torch.minimum(torch.maximum(x, lower_limit), upper_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors
        upper_limit = ctx.upper_limit
        lower_limit = ctx.lower_limit

        # Compute gradient
        # If x > upper_limit, propagate grad_output normally
        # If x <= upper_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
            None,
        )  # None for upper_limit and lower_limit since they do not require gradients


class HeavisideProjectionLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, eta, threshold):
        ctx.save_for_backward(x, beta, eta)
        ctx.threshold = threshold
        return torch.where(
            x < eta,
            torch.tensor(0, dtype=torch.float32).to(x.device),
            torch.tensor(1, dtype=torch.float32).to(x.device),
        )
        if beta < threshold:
            return (torch.tanh(threshold * eta) + torch.tanh(threshold * (x - eta))) / (
                torch.tanh(threshold * eta) + torch.tanh(threshold * (1 - eta))
            )
        else:
            return (torch.tanh(beta * eta) + torch.tanh(beta * (x - eta))) / (
                torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
            )

    @staticmethod
    def backward(ctx, grad_output):
        x, beta, eta = ctx.saved_tensors

        threshold = ctx.threshold

        grad = (
            grad_output
            * (beta * (1 - (torch.tanh(beta * (x - eta))) ** 2))
            / (torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta)))
        )

        return grad, None, None, None


class heightProjectionLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ridge_height, height_mask, sharpness, threshold):
        ctx.save_for_backward(ridge_height, height_mask)
        ctx.sharpness = sharpness
        return torch.where(
            height_mask < ridge_height,
            torch.tensor(1, dtype=torch.float32).to(ridge_height.device),
            torch.tensor(0, dtype=torch.float32).to(ridge_height.device),
        )
        if sharpness < threshold:
            return torch.tanh(threshold * (ridge_height - height_mask)) / 2 + 0.5
        else:
            return torch.tanh(sharpness * (ridge_height - height_mask)) / 2 + 0.5

    @staticmethod
    def backward(ctx, grad_output):
        ridge_height, height_mask = ctx.saved_tensors
        sharpness = ctx.sharpness

        grad = (
            grad_output
            * sharpness
            * (1 - (torch.tanh(sharpness * (ridge_height - height_mask))) ** 2)
            / 2
        )

        return grad, None, None, None


class InsensitivePeriodLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, i):
        ctx.save_for_backward(x)
        ctx.i = i
        return x * i

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        i = ctx.i
        grad = grad_output

        return grad, None


def poynting_vector(
    Hx, Hy, Ez, grid_step, monitor=None, direction="x+", autograd=False
):
    if autograd:
        conj = npa.conj
        real = npa.real
        sum = npa.sum
    else:
        conj = np.conj
        real = np.real
        sum = np.sum
    if isinstance(monitor, (Slice, np.ndarray)):
        Hx = Hx[monitor]
        Hy = Hy[monitor]
        Ez_conj = conj(Ez[monitor])

    if direction == "x+":
        P = sum(real(Ez_conj * Hy)) * (-grid_step)
    elif direction == "x-":
        P = sum(real(Ez_conj * Hy)) * grid_step
    elif direction == "y+":
        P = sum(real(Ez_conj * Hx)) * grid_step
    elif direction == "y-":
        P = sum(real(Ez_conj * Hx)) * (-grid_step)
    else:
        raise ValueError("Invalid direction")
    return P


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


def plot_eps_field(
    Ez,
    eps,
    monitors=[],
    filepath=None,
    zoom_eps_factor=1,
    x_width=1,
    y_height=1,
    NPML=[0, 0],
    title: str = None,
):
    if isinstance(Ez, torch.Tensor):
        Ez = Ez.data.cpu().numpy()
    if isinstance(eps, torch.Tensor):
        eps = eps.data.cpu().numpy()

    if filepath is not None:
        ensure_dir(os.path.dirname(filepath))
    fig, ax = plt.subplots(
        1,
        2,
        constrained_layout=True,
        figsize=(7 * Ez.shape[0] / 600, 1.7 * Ez.shape[1] / 300),
    )
    ceviche.viz.abs(Ez, outline=None, ax=ax[0], cbar=True)
    ceviche.viz.abs(eps.astype(np.float64), ax=ax[0], cmap="Greys", alpha=0.2)
    if len(monitors) > 0:
        for m in monitors:
            if isinstance(m[0], Slice):
                m_slice, color = m
                if len(m_slice.x.shape) == 0:
                    xs = m_slice.x * np.ones(len(m_slice.y))
                    ys = m_slice.y
                else:
                    xs = m_slice.x
                    ys = m_slice.y * np.ones(len(m_slice.x))
                ax[0].plot(xs, ys, color, alpha=0.5)
            elif isinstance(m[0], np.ndarray):
                mask, color = m
                xs, ys = mask.nonzero()
                ax[0].scatter(xs, ys, c=color, s=1.5, alpha=0.5, linewidths=0)

    ## draw shaddow with NPML border
    ## left
    rect = patches.Rectangle(
        (0, 0), width=NPML[0], height=Ez.shape[1], facecolor="gray", alpha=0.5
    )
    ax[0].add_patch(rect)
    ## right
    rect = patches.Rectangle(
        (Ez.shape[0] - NPML[0], 0),
        width=NPML[0],
        height=Ez.shape[1],
        facecolor="gray",
        alpha=0.5,
    )
    ax[0].add_patch(rect)

    ## lower
    rect = patches.Rectangle(
        (NPML[0], 0),
        width=Ez.shape[0] - NPML[0] * 2,
        height=NPML[1],
        facecolor="gray",
        alpha=0.5,
    )
    ax[0].add_patch(rect)

    ## upper
    rect = patches.Rectangle(
        (NPML[0], Ez.shape[1] - NPML[1]),
        width=Ez.shape[0] - NPML[0] * 2,
        height=NPML[1],
        facecolor="gray",
        alpha=0.5,
    )
    ax[0].add_patch(rect)

    ## add title to ax[0]
    if title is not None:
        ax[0].set_title(title, fontsize=9, y=1.05)

    xlabel = np.linspace(-x_width / 2, x_width / 2, 5)
    ylabel = np.linspace(-y_height / 2, y_height / 2, 5)
    xticks = np.linspace(0, Ez.shape[0] - 1, 5)
    yticks = np.linspace(0, Ez.shape[1] - 1, 5)
    xlabel = [f"{x:.2f}" for x in xlabel]
    ylabel = [f"{y:.2f}" for y in ylabel]
    ax[0].set_xlabel(r"$x$ width ($\mu m$)")
    ax[0].set_ylabel(r"$y$ height ($\mu m$)")
    ax[0].set_xticks(xticks, xlabel)
    ax[0].set_yticks(yticks, ylabel)
    ax[0].set_xlim([0, Ez.shape[0]])
    ax[0].set_ylim([0, Ez.shape[1]])
    # ax[0].set_box_aspect(1)

    # for sl in slices:
    #     ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')

    size = eps.shape
    ## center crop of eps of size of new_size
    if zoom_eps_factor > 1:
        eps = zoom(eps, zoom_eps_factor)
        eps = eps[
            eps.shape[0] // 2 - size[0] // 2 : eps.shape[0] // 2 + size[0] // 2,
            eps.shape[1] // 2 - size[1] // 2 : eps.shape[1] // 2 + size[1] // 2,
        ]

    ceviche.viz.abs(eps, ax=ax[1], cmap="Greys", cbar=True)
    xlabel = np.linspace(
        -x_width / 2 / zoom_eps_factor, x_width / 2 / zoom_eps_factor, 5
    )
    ylabel = np.linspace(
        -y_height / 2 / zoom_eps_factor, y_height / 2 / zoom_eps_factor, 5
    )
    xlabel = [f"{x:.2f}" for x in xlabel]
    ylabel = [f"{y:.2f}" for y in ylabel]
    ax[1].set_xlabel(r"$x$ width ($\mu m$)")
    ax[1].set_ylabel(r"$y$ height ($\mu m$)")
    ax[1].set_xticks(xticks, xlabel)
    ax[1].set_yticks(yticks, ylabel)
    # ax[1].set_box_aspect(1)

    if filepath is not None:
        fig.savefig(filepath, dpi=600, bbox_inches="tight")
    plt.close()


def insert_mode(omega, dx, x, y, epsr, target=None, npml=0, m=1, filtering=False):
    """Solve for the modes in a cross section of epsr at the location defined by 'x' and 'y'

    The mode is inserted into the 'target' array if it is suppled, if the target array is not
    supplied, then a target array is created with the same shape as epsr, and the mode is
    inserted into it.
    """
    if target is None:
        target = np.zeros(epsr.shape, dtype=complex)
    epsr_cross = epsr[x, y]
    # Solves the eigenvalue problem:
    #    [ ∂²/∂x² / (k₀²) + εr ] E = (β²/k₀²) E
    vals, e = get_modes(epsr_cross, omega, dx, npml, m=m, filtering=filtering)
    # Compute transverse magnetic field as:
    #    H = β / (μ₀ ω) * E
    # where the β term originates from the spatial derivative in the propagation
    # direction.
    k0 = omega / constants.C_0
    beta = np.real(np.sqrt(vals, dtype=complex)) * k0
    h = beta / omega / constants.MU_0 * e
    target[x, y] = np.atleast_2d(e)[:, m - 1].squeeze()
    return h[:, m - 1], e[:, m - 1], beta, target

class sParamFromEfield(object):
    '''
    a callable class that calculates the s-parameters from the e_fields
    so that we can use jacobian reverse to calculate the gradient of the s-parameters 
    w.r.t the e-fields for adjoint source calculation
    '''
    def __init__(
            self, 
            port_profile, 
            port_slice, 
            grid_step, 
            sim, 
            direction, 
            in_port_name,
            out_port_name,
            wl,
            in_mode,
            out_mode,
            avg_coefficient,
            type,
        ):
        self.port_profile = port_profile
        self.port_slice = port_slice
        self.grid_step = grid_step
        self.sim = sim
        self.direction = direction
        self.monitor_slice = self.port_slice[out_port_name]
        self.norm_power = self.port_profile[in_port_name][
            (wl, in_mode)
        ][3]
        if type == "eigenmode":
            _, self.ht_m, self.et_m, _ = self.port_profile[out_port_name][
                (wl, out_mode)
            ]
        self.avg_coefficient = avg_coefficient
        self.type = type
        assert type in {"eigenmode", "flux", "flux_minus_src"}, f"type {type} not supported"
    
    def __call__(self, Efield):
        Efield_vec = self.sim._grid_to_vec(Efield)
        Hx_vec, Hy_vec = self.sim._Ez_to_Hx_Hy(Efield_vec)
        hx = self.sim._vec_to_grid(Hx_vec)
        hy = self.sim._vec_to_grid(Hy_vec)
        if self.type == "eigenmode":
            s_p, s_m = get_eigenmode_coefficients(
                hx,
                hy,
                Efield,
                self.ht_m,
                self.et_m,
                self.monitor_slice,
                grid_step=self.grid_step,
                direction=self.direction[0],
                autograd=True,
                energy=True,
            )
            if self.direction[1] == "+":
                s = s_p
            elif self.direction[1] == "-":
                s = s_m
            else:
                raise ValueError("Invalid direction")
            s = s / self.norm_power / self.avg_coefficient
        elif self.type in {"flux", "flux_minus_src"}:
            s = get_flux(
                hx,
                hy,
                Efield,
                self.monitor_slice,
                grid_step=self.grid_step,
                direction=self.direction[0],
                autograd=True,
            )
            s = npa.abs(s / self.norm_power / self.avg_coefficient)  # we only need absolute flux
            if type == "flux_minus_src":
                s = npa.abs(
                    s - 1
                )  ## if it is larger than 1, then this slice must include source, we minus the power from source
        return s

class ObjectiveFunc(object):
    def __init__(
        self,
        simulations: dict,
        port_profiles: dict,  # port monitor profiles {port_name: {(wl, mode): (profile, ht_m, et_m)}}
        port_slices: dict,
        grid_step: float,
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
        self.grid_step = grid_step

        self.eps = None
        self.Ez = None
        self.Js = {}  # forward from fields to foms
        self.adj_Js = {}  # Js for adjoint source calculation
        self.dJ = None  # backward from fom to permittivity
        self.breakdown = {}
        self.solutions = {}
        self.verbose = verbose

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
                in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                out_modes=(
                    1,
                ),  # can evaluate on multiple output modes and get average transmission
                direction="x+",
            )
        ),
    ):
        self.s_params = {}
        self._obj_fusion_func = cfgs["_fusion_func"]
        cfgs = deepcopy(cfgs)
        del cfgs["_fusion_func"]
        ### build objective functions from solved fields to fom
        for name, cfg in cfgs.items():
            type = cfg["type"]
            in_port_name = cfg["in_port_name"]
            out_port_name = cfg["out_port_name"]
            in_mode = cfg["in_mode"]
            out_modes = cfg["out_modes"]
            direction = cfg["direction"]

            if type == "eigenmode":

                def objfn(
                    fields,
                    in_port_name=in_port_name,
                    out_port_name=out_port_name,
                    in_mode=in_mode,
                    out_modes=out_modes,
                    direction=direction,
                    name=name,
                ):
                    s_list = []
                    ## for each wavelength, we evaluate the objective
                    for wl, sim in self.sims.items():
                        ## we calculate the average eigen energy for all output modes
                        for out_mode in out_modes:
                            src, ht_m, et_m, norm_p = self.port_profiles[out_port_name][
                                (wl, out_mode)
                            ]
                            norm_power = self.port_profiles[in_port_name][
                                (wl, in_mode)
                            ][3]
                            monitor_slice = self.port_slices[out_port_name]
                            field = fields[(in_port_name, wl, in_mode)]
                            hx, hy, ez = (
                                field["Hx"],
                                field["Hy"],
                                field["Ez"],
                            )  # fetch fields
                            if isinstance(ht_m, Tensor) and ht_m.device != ez.device:
                                ht_m = ht_m.to(ez.device)
                                et_m = et_m.to(ez.device)
                                self.port_profiles[out_port_name][(wl, out_mode)] = [
                                    src.to(ez.device),
                                    ht_m,
                                    et_m,
                                    norm_p,
                                ]
                            # if isinstance(ht_m, np.ndarray) and isinstance(ez, Tensor):
                            #     ht_m = torch.from_numpy(ht_m).to(ez.device)
                            #     et_m = torch.from_numpy(et_m).to(ez.device)
                            #     self.port_profiles[out_port_name][(wl, out_mode)] = [
                            #         torch.from_numpy(src).to(ez.device),
                            #         ht_m,
                            #         et_m,
                            #         norm_p,
                            #     ]

                            s_p, s_m = get_eigenmode_coefficients(
                                hx,
                                hy,
                                ez,
                                ht_m,
                                et_m,
                                monitor_slice,
                                grid_step=self.grid_step,
                                direction=direction[0],
                                autograd=True,
                                energy=True,
                            )
                            if direction[1] == "+":
                                s = s_p
                            elif direction[1] == "-":
                                s = s_m
                            else:
                                raise ValueError("Invalid direction")
                            s_list.append(s / norm_power)
                            self.s_params[(name, wl, out_mode)] = {
                                "s_p": s_p,
                                "s_m": s_m,
                            }
                    if isinstance(s_list[0], Tensor):
                        return torch.mean(torch.stack(s_list))
                    else:
                        return npa.mean(npa.array(s_list))
            elif type in {"flux", "flux_minus_src"}:

                def objfn(
                    fields,
                    in_port_name=in_port_name,
                    out_port_name=out_port_name,
                    in_mode=in_mode,
                    direction=direction,
                    type=type,
                    name = name,
                ):
                    s_list = []
                    ## for each wavelength, we evaluate the objective
                    for wl, sim in self.sims.items():
                        monitor_slice = self.port_slices[out_port_name]
                        norm_power = self.port_profiles[in_port_name][(wl, in_mode)][3]
                        field = fields[(in_port_name, wl, in_mode)]
                        hx, hy, ez = (
                            field["Hx"],
                            field["Hy"],
                            field["Ez"],
                        )  # fetch fields
                        s = get_flux(
                            hx,
                            hy,
                            ez,
                            monitor_slice,
                            grid_step=self.grid_step,
                            direction=direction[0],
                            autograd=True,
                        )
                        if isinstance(s, Tensor):
                            abs = torch.abs
                        else:
                            abs = npa.abs
                        s = abs(s / norm_power)  # we only need absolute flux
                        if type == "flux_minus_src":
                            s = abs(
                                s - 1
                            )  ## if it is larger than 1, then this slice must include source, we minus the power from source

                        s_list.append(s)
                        self.s_params[(name, wl, type)] = {
                            "s": s,
                        }
                    if isinstance(s_list[0], Tensor):
                        return torch.mean(torch.stack(s_list))
                    else:
                        return npa.mean(npa.array(s_list))  # we only need absolute flux
            else:
                raise ValueError("Invalid type")

            ### note that this is not the final objective! this is partial objective from fields to fom
            ### complete autograd graph is from permittivity (np.ndarray) to fields and to fom
            self.Js[name] = {"weight": cfg["weight"], "fn": objfn}

    def add_adj_objective(
        self,
        cfgs: dict = dict(
            fwd_trans=dict(
                weight=1,
                type="eigenmode",
                #### objective is evaluated at this port
                in_port_name="in_port_1",
                out_port_name="out_port_1",
                #### objective is evaluated at all points by sweeping the wavelength and modes
                in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                out_modes=(
                    1,
                ),  # can evaluate on multiple output modes and get average transmission
                direction="x+",
            )
        ),
    ):
        cfgs = deepcopy(cfgs)
        del cfgs["_fusion_func"]
        ### build objective functions from solved fields to fom
        for name, cfg in cfgs.items():
            type = cfg["type"]
            in_port_name = cfg["in_port_name"]
            out_port_name = cfg["out_port_name"]
            in_mode = cfg["in_mode"]
            out_modes = cfg["out_modes"]
            direction = cfg["direction"]

            adj_objfn = {}
            for wl, sim in self.sims.items():
                for out_mode in out_modes:
                    objfn = sParamFromEfield(
                        port_profile=self.port_profiles, 
                        port_slice=self.port_slices, 
                        grid_step=self.grid_step, 
                        sim=sim, 
                        direction=direction, 
                        in_port_name=in_port_name,
                        out_port_name=out_port_name,
                        wl=wl,
                        in_mode=in_mode,
                        out_mode=out_mode,
                        avg_coefficient=len(self.sims.keys()) * len(out_modes),
                        type=type,
                    )
                    adj_objfn[(wl, out_mode)] = objfn

            ### note that this is not the final objective! this is partial objective from fields to fom
            ### complete autograd graph is from permittivity (np.ndarray) to fields and to fom
            self.adj_Js[name] = {"weight": cfg["weight"], "fn": adj_objfn}

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
            adj_sources[key] = sim.solver.adj_src # this is the b_adj
            ez_adj, hx_adj, hy_adj, flux = sim.norm_adj_power()
            # field_adj[key] = {"Ez": ez_adj, "Hx": hx_adj, "Hy": hy_adj}
            field_adj[key] = {}
            for (port_name, mode), _ in ez_adj.items():
                field_adj[key][(port_name, mode)] = {
                    "Ez": ez_adj[(port_name, mode)],
                    "Hx": hx_adj[(port_name, mode)],
                    "Hy": hy_adj[(port_name, mode)],
                }
            field_adj_normalizer[key] = flux
        return adj_sources, field_adj, field_adj_normalizer

    def obtain_objective(
        self, permittivity: np.ndarray | Tensor
    ) -> Tuple[dict, Tensor]:
        self.solutions = {}
        self.As = {}
        for port_name, port_profile in self.port_profiles.items():
            for (wl, mode), (source, _, _, norm_power) in port_profile.items():
                ## here the source is already normalized during norm_run to make sure it has target power
                ## here is the key part that build the common "eps to field" autograd graph
                ## later on, multiple "field to fom" autograd graph(s) will be built inside of multiple obj_fn's
                self.sims[wl].eps_r = permittivity
                Hx, Hy, Ez = self.sims[wl].solve(source, port_name=port_name, mode=mode)
                self.solutions[(port_name, wl, mode)] = {
                    "Hx": Hx,
                    "Hy": Hy,
                    "Ez": Ez,
                }
                self.As[wl] = self.sims[wl].A

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


class MaxwellResidualLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        sim,
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ):
        super().__init__(size_average, reduce, reduction)
        self.sim = sim  #
        self.wl_list = torch.linspace(
            wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl
        )
        self.omegas = 2 * np.pi * constants.C_0 / (self.wl_list * 1e-6)
        self.As = (None, None)
        self.bs = {}

    def make_A(self, eps_r: torch.Tensor, wl_list: List[float]):
        ## eps_r: [bs, h, w] real tensor
        ## wl_list: [n_wl] list of wls in um, support spectral bundling
        eps_vec = eps_r.flatten(1)
        tot_entries_list = []
        tot_indices_list = []
        for eps_v in eps_vec:
            entries_list = []
            indices_list = []
            for wl in wl_list:
                self.sim.omega = 2 * np.pi * constants.C_0 / (wl.item() * 1e-6)
                self.sim._setup_derivatives() # need to setup derivatives after updating the omega
                entries_a, indices_a = self.sim._make_A(
                    eps_v
                )  # return scipy sparse indices and values
                entries_list.append(torch.from_numpy(entries_a))
                indices_list.append(torch.from_numpy(indices_a))
            entries_list = torch.stack(entries_list, 0)
            indices_list = torch.stack(indices_list, 0)
            tot_entries_list.append(entries_list)
            tot_indices_list.append(indices_list)
        tot_entries_list = torch.stack(tot_entries_list, 0).to(
            eps_r.device
        )  # [bs, n_wl, 5*h*w]
        tot_indices_list = (
            torch.stack(tot_indices_list, 0).to(eps_r.device).long()
        )  # [bs, n_wl, 2, 5*h*w]
        self.As = (tot_entries_list, tot_indices_list)
        return self.As

    def forward(self, Ez: Tensor, eps_r: Tensor, source: Tensor):
        ## Ez: [bs, n_wl, h, w] complex tensor
        ## eps_r: [bs, h, w] real tensor
        ## source: [bs, n_wl, h, w] complex tensor, source in sim.solve(source), not b, b = 1j * omega * source

        # step 1: make A
        self.make_A(eps_r, self.wl_list)

        # step 2: calculate loss
        entries, indices = self.As
        lhs = []
        if self.omegas.device != source.device:
            self.omegas = self.omegas.to(source.device)
        b = (
            (source * (1j * self.omegas[None, :, None, None])).flatten(0, 1).flatten(1)
        )  # [bs*n_wl, h*w]
        for i in range(Ez.shape[0]):
            for j in range(Ez.shape[1]):
                ez = Ez[i, j].flatten()
                omega = 2 * np.pi * constants.C_0 / (self.wl_list[j] * 1e-6)

                b = source[i, j].flatten() * (1j * omega)
                A_by_e = spmm(
                    indices[i, j],
                    entries[i, j],
                    m=ez.shape[0],
                    n=ez.shape[0],
                    matrix=ez[:, None],
                )[:, 0]
                lhs.append(A_by_e)
        lhs = torch.stack(lhs, 0)  # [bs*n_wl, h*w]

        loss = torch.nn.functional.mse_loss(lhs, b, reduction=self.reduction)

        return loss
