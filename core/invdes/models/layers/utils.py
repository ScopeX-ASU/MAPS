import math
import os
from typing import Callable, List, Tuple

import matplotlib.patches as patches
import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch
from autograd import numpy as npa
from pyutils.general import ensure_dir, logger
from scipy.ndimage import zoom
from torch import Tensor

from core.utils import (
    Slice,
    get_eigenmode_coefficients,
)
from thirdparty.ceviche.ceviche import constants
from thirdparty.ceviche.ceviche.fdfd import compute_derivative_matrices
from thirdparty.ceviche.ceviche.modes import filter_modes, normalize_modes

from .viz import abs as plot_abs
from .viz import real as plot_real

__all__ = [
    "get_grid",
    "apply_regions_gpu",
    "AdjointGradient",
    "differentiable_boundary",
    "BinaryProjection",
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
    "get_temp_related_eps",
    "modulation_fn_dict",
]


def get_temp_related_eps(
    eps, temp, temp_0: float = 300, eps_r_0: float = 3.48**2, dn_dT=1.8e-4
):
    # and we treat the air as it is independent of the temperature
    eps_max = eps.max()
    eps_min = eps.min()
    eps = (eps - eps_min) / (eps_max - eps_min)  # (0, 1)
    n_si = math.sqrt(eps_r_0) + (temp - temp_0) * dn_dT
    eps = eps * (n_si**2 / eps_r_0)
    eps = eps * (eps_max - eps_min) + eps_min
    return eps


def temperature_modulation(
    eps: float, T: float, T0: float = 300, dn_dT: float = 1.8e-4
):
    return (math.sqrt(eps) + (T - T0) * dn_dT) ** 2


modulation_fn_dict = {
    "temperature": temperature_modulation,
}


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


def loc2ind(
    loc: Tuple[float, float],
    box_size: Tuple[float, float],
    box_shape: Tuple[float, float],
    clip: bool = True,
):
    ## take arbitrary dimensions and return the index of the location in the box (if clip), otherwise can be outside of box
    indices = []
    ## box is in the center of the space, center aligns with origin
    assert (
        len(loc) == len(box_size) == len(box_shape)
    ), "The dimensions of loc, box_size, and box_shape should be the same"
    loc = np.array(loc)
    box_size = np.array(box_size)
    box_shape = np.array(box_shape)
    indices = np.round((loc + box_size / 2) / box_size * box_shape).astype(np.int32)

    if clip:
        indices = np.clip(indices, 0, box_shape - 1)
    return indices


def plot_eps_field(
    Ez,
    eps,
    monitors=[],
    filepath=None,
    zoom_eps_factor=1,
    zoom_eps_center=(0, 0),
    x_width=1,
    y_height=1,
    NPML=[0, 0],
    field_stat: str = "abs",  # "abs" or "real" or "abs_real"
    title: str = None,
    x_shift_coord: int = 0,
    x_shift_idx: int = 0,
):
    if isinstance(Ez, torch.Tensor):
        Ez = Ez.data.cpu().numpy()
    if isinstance(eps, torch.Tensor):
        eps = eps.data.cpu().numpy()

    if filepath is not None:
        ensure_dir(os.path.dirname(filepath))

    # Calculate dynamic font size based on eps dimensions
    base_fontsize = min(eps.shape[0], eps.shape[1]) / 30  # Scale factor can be adjusted
    title_fontsize = base_fontsize * 1.2
    label_fontsize = base_fontsize * 0.8
    tick_fontsize = base_fontsize * 0.5
    field_stat = field_stat.lower().split("_")
    fig, ax = plt.subplots(
        len(field_stat) + 1,
        1,
        constrained_layout=True,
        figsize=(
            7 * Ez.shape[0] / 600 / 2,
            1.7 * Ez.shape[1] / 300 * (len(field_stat) + 1),
        ),
        gridspec_kw={"wspace": 0.3},
    )
    for i, stat in enumerate(field_stat):
        if stat == "abs":
            plot_abs(Ez, outline=None, ax=ax[i], cbar=True, font_size=label_fontsize)
        elif stat == "real":
            plot_real(Ez, outline=None, ax=ax[i], cbar=True, font_size=label_fontsize)
        elif stat == "intensity":
            plot_abs(
                np.abs(Ez) ** 2,
                outline=None,
                ax=ax[i],
                cbar=True,
                font_size=label_fontsize,
            )
        plot_abs(
            eps.astype(np.float64),
            ax=ax[i],
            cmap="Greys",
            alpha=0.2,
            font_size=label_fontsize,
        )
        if len(monitors) > 0:
            for m in monitors:
                if isinstance(m[0], Slice):
                    m_slice, color = m
                    if len(m_slice.x.shape) == 0:
                        xs = m_slice.x * np.ones(len(m_slice.y))
                        ys = m_slice.y
                        ax[i].plot(xs, ys, color, alpha=0.5)
                    elif len(m_slice.y.shape) == 0:
                        xs = m_slice.x
                        ys = m_slice.y * np.ones(len(m_slice.x))
                        ax[i].plot(xs, ys, color, alpha=0.5)
                    else:  # two axis are all arrays, this is a box, we draw its 4 edges
                        xs, ys = m_slice.x[:, 0], m_slice.y[0]
                        left_xs = xs[0] * np.ones(len(ys))
                        left_ys = ys
                        ax[i].plot(left_xs, left_ys, color, alpha=0.5)
                        right_xs = xs[-1] * np.ones(len(ys))
                        right_ys = ys
                        ax[i].plot(right_xs, right_ys, color, alpha=0.5)
                        lower_xs = xs
                        lower_ys = ys[0] * np.ones(len(xs))
                        ax[i].plot(lower_xs, lower_ys, color, alpha=0.5)
                        upper_xs = xs
                        upper_ys = ys[-1] * np.ones(len(xs))
                        ax[i].plot(upper_xs, upper_ys, color, alpha=0.5)

                elif isinstance(m[0], np.ndarray):
                    mask, color = m
                    xs, ys = mask.nonzero()
                    ax[i].scatter(xs, ys, c=color, s=1.5, alpha=0.5, linewidths=0)

        ## draw shaddow with NPML border
        ## left
        rect = patches.Rectangle(
            (0, 0), width=NPML[0], height=Ez.shape[1], facecolor="gray", alpha=0.5
        )
        ax[i].add_patch(rect)
        ## right
        rect = patches.Rectangle(
            (Ez.shape[0] - NPML[0], 0),
            width=NPML[0],
            height=Ez.shape[1],
            facecolor="gray",
            alpha=0.5,
        )
        ax[i].add_patch(rect)

        ## lower
        rect = patches.Rectangle(
            (NPML[0], 0),
            width=Ez.shape[0] - NPML[0] * 2,
            height=NPML[1],
            facecolor="gray",
            alpha=0.5,
        )
        ax[i].add_patch(rect)

        ## upper
        rect = patches.Rectangle(
            (NPML[0], Ez.shape[1] - NPML[1]),
            width=Ez.shape[0] - NPML[0] * 2,
            height=NPML[1],
            facecolor="gray",
            alpha=0.5,
        )
        ax[i].add_patch(rect)

        ## add title to ax[0]
        if title is not None:
            # ax[0].set_title(title, fontsize=9, y=1.05)
            # ax[0].set_title(title, fontsize=title_fontsize, y=1.05)
            fig.suptitle(title, fontsize=title_fontsize, y=1.2, ha="center")

        xlabel = np.linspace(-x_width / 2, x_width / 2, 5)
        ylabel = np.linspace(-y_height / 2, y_height / 2, 5)
        xticks = np.linspace(0, Ez.shape[0] - 1, 5)
        yticks = np.linspace(0, Ez.shape[1] - 1, 5)
        xlabel = [f"{x:.2f}" for x in xlabel]
        ylabel = [f"{y:.2f}" for y in ylabel]
        ax[i].set_xlabel(r"$x$ width ($\mu m$)", fontsize=label_fontsize)
        ax[i].set_ylabel(r"$y$ height ($\mu m$)", fontsize=label_fontsize)
        ax[i].set_xticks(xticks)
        ax[i].set_yticks(yticks)
        ax[i].set_xticklabels(xlabel, fontsize=tick_fontsize)
        ax[i].set_yticklabels(ylabel, fontsize=tick_fontsize)
        # ax[0].set_xticks(xticks, xlabel)
        # ax[0].set_yticks(yticks, ylabel)
        ax[i].set_xlim([0, Ez.shape[0]])
        ax[i].set_ylim([0, Ez.shape[1]])
        ax[i].set_aspect("equal")

    # for sl in slices:
    #     ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')

    size = eps.shape
    ## center crop of eps of size of new_size
    ## find center pixel index based on zoom_eps_center
    patch_size = (x_width / zoom_eps_factor, y_height / zoom_eps_factor)
    if zoom_eps_factor > 1:
        ## move center to avoid exceeding the boundary
        zoom_eps_center = np.clip(
            zoom_eps_center,
            (-(x_width - patch_size[0]) / 2, -(y_height - patch_size[1]) / 2),
            ((x_width - patch_size[0]) / 2, (y_height - patch_size[1]) / 2),
        )
        zoom_eps_center_ind = np.round(
            loc2ind(zoom_eps_center, (x_width, y_height), size) * zoom_eps_factor
        ).astype(np.int32)

        eps = zoom(eps, zoom_eps_factor)
        eps = eps[
            zoom_eps_center_ind[0] - size[0] // 2 : zoom_eps_center_ind[0]
            + size[0] // 2,
            zoom_eps_center_ind[1] - size[1] // 2 : zoom_eps_center_ind[1]
            + size[1] // 2,
        ]
    else:
        zoom_eps_center = (0, 0)  # force to be origin if not zoomed

    # plot_abs(eps, ax=ax[-1], cmap="Greys", cbar=True, font_size=label_fontsize)
    plot_abs(
        torch.ones_like(torch.from_numpy(Ez)),
        ax=ax[-1],
        cmap="Greys",
        cbar=True,
        font_size=label_fontsize,
    )
    xlabel = np.linspace(
        zoom_eps_center[0] - patch_size[0] / 2,
        zoom_eps_center[0] + patch_size[0] / 2,
        5,
    )
    ylabel = np.linspace(
        zoom_eps_center[1] - patch_size[1] / 2,
        zoom_eps_center[1] + patch_size[1] / 2,
        5,
    )
    xlabel = [f"{x:.2f}" for x in xlabel]
    ylabel = [f"{y:.2f}" for y in ylabel]
    ax[-1].set_xlabel(r"$x$ width ($\mu m$)", fontsize=label_fontsize)
    ax[-1].set_ylabel(r"$y$ height ($\mu m$)", fontsize=label_fontsize)
    ax[-1].set_xticks(xticks)
    ax[-1].set_yticks(yticks)
    ax[-1].set_xticklabels(xlabel, fontsize=tick_fontsize)
    ax[-1].set_yticklabels(ylabel, fontsize=tick_fontsize)
    ax[-1].set_aspect("equal")
    # ax[1].set_xticks(xticks, xlabel)
    # ax[1].set_yticks(yticks, ylabel)
    area = Ez.shape[0] * Ez.shape[1]
    if area > 2000**2:
        dpi = 300
    else:
        dpi = 600
    if filepath is not None:
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()


def solver_eigs(A, Neigs, guess_value=1.0):
    """solves for `Neigs` eigenmodes of A
        A:            sparse linear operator describing modes
        Neigs:        number of eigenmodes to return
        guess_value:  estimate for the eigenvalues
    For more info, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
    """

    values, vectors = spl.eigs(A, k=Neigs, sigma=guess_value, v0=None, which="LM")

    return values, vectors


def get_modes(
    eps_cross, omega, dL, npml, m=1, filtering=True, eps_cross_xx=None, pol: str = "Ez"
):
    """Solve for the modes of a waveguide cross section
    ARGUMENTS
        eps_cross: the permittivity profile of the waveguide
        omega:     angular frequency of the modes
        dL:        grid size of the cross section
        npml:      number of PML points on each side of the cross section
        m:         number of modes to solve for
        filtering: whether to filter out evanescent modes
    RETURNS
        vals:      array of effective indeces of the modes
        vectors:   array containing the corresponding mode profiles
    """

    k0 = omega / constants.C_0

    N = eps_cross.size

    matrices = compute_derivative_matrices(omega, (N, 1), [npml, 0], dL=dL)

    Dxf, Dxb, Dyf, Dyb, Dzf, Dzb = matrices

    diag_eps_r = sp.spdiags(eps_cross.flatten(), [0], N, N)
    if pol == "Ez":
        A = diag_eps_r + Dxf.dot(Dxb) * (1 / k0) ** 2
    elif pol == "Hz":
        diag_eps_r_xx_inv = sp.spdiags(1 / eps_cross_xx.flatten(), [0], N, N)
        A = (
            diag_eps_r
            + diag_eps_r.dot(Dxf).dot(diag_eps_r_xx_inv).dot(Dxb) * (1 / k0) ** 2
        )

    n_max = np.sqrt(np.max(eps_cross)) * 0.92
    vals, vecs = solver_eigs(A, m, guess_value=n_max**2)

    if pol == "Hz":
        # vecs = (vecs + np.roll(vecs, shift=1, axis=0))/2
        vecs = np.roll(vecs, shift=1, axis=0)

    if filtering:
        filter_re = lambda vals: np.real(vals) > 0.0
        # filter_im = lambda vals: np.abs(np.imag(vals)) <= 1e-12
        filters = [filter_re]
        vals, vecs = filter_modes(vals, vecs, filters=filters)

    if vals.size == 0:
        raise BaseException("Could not find any eigenmodes for this waveguide")

    vecs = normalize_modes(vecs)

    return vals, vecs


def insert_mode(omega, dx, x, y, epsr, target=None, npml=0, m="Ez1", filtering=False):
    """Solve for the modes in a cross section of epsr at the location defined by 'x' and 'y'

    The mode is inserted into the 'target' array if it is suppled, if the target array is not
    supplied, then a target array is created with the same shape as epsr, and the mode is
    inserted into it.
    """
    if isinstance(m, int):
        pol = "Ez"  # by default Ez mode
        logger.warning("The mode is not specified, by default, it is Ez mode")
    pol = m[0:2]
    m = int(m[2:])
    if target is None:
        target = np.zeros(epsr.shape, dtype=complex)
    epsr_cross = epsr[x, y]

    if pol == "Hz":
        if len(x.shape) == 0:  # x direction slice
            epsr_cross_xx = epsr_cross
        elif len(y.shape) == 0:  # y direction slice
            epsr_cross_xx = (epsr_cross + np.roll(epsr_cross, shift=1)) / 2
    else:
        epsr_cross_xx = None

    ## see page 89 in https://empossible.net/wp-content/uploads/2019/08/Lecture-4f-FDFD-Extras.pdf
    ## E mode: -(Dxf @ Dxb + MU_0 * eps_0 * eps_r) Ez = gamma^2 Ez
    ## (Dxf @ Dxb / k0^2 + MU_0 * eps_0 / k0^2 * eps_r) Ez = (beta/k0)^2 Ez
    ## (Dxf @ Dxb / k0^2 + 1/omega^2 * eps_r) Ez = (beta/k0)^2 * Ez
    # gamma = j*k0*n_eff = j*beta
    ## beta = neff * k0 = neff * 2pi / lambda = neff * omega / c
    ## -(Dxf @ Dxb / k0^2 + eps_r) Ez = -beta*2 * Ez
    ## (Dxf @ Dxb + eps_r) Ez = beta*2 * Ez
    # Solves the eigenvalue problem:
    #    [ ∂²/∂x² / (k₀²) + εr ] E = (β²/k₀²) E
    #    [ ∂²/∂x² / (k₀²) + εr ] E = (β²/k₀²) E
    ## eigen value is effective index n_eff^2
    vals, fz = get_modes(
        epsr_cross,
        omega,
        dx,
        npml,
        m=m,
        filtering=filtering,
        eps_cross_xx=epsr_cross_xx,
        pol=pol,
    )
    # Compute transverse magnetic field as:
    #    H = β / (μ₀ ω) * E
    # where the β term originates from the spatial derivative in the propagation
    # direction.
    ## remove center phase
    if fz.shape[0] % 2 == 0:
        center_phase = np.exp(
            -1j
            * np.angle(
                (
                    fz[fz.shape[0] // 2 - 1 : fz.shape[0] // 2]
                    + fz[fz.shape[0] // 2 : fz.shape[0] // 2 + 1]
                )
                / 2
            )
        )
    else:
        center_phase = np.exp(
            -1j * np.angle(fz[fz.shape[0] // 2 : fz.shape[0] // 2 + 1])
        )
    fz = fz * center_phase

    ## for Ez pol, this e is Ez, h is tangential field, i.e., for x direction: hy, for y direction: hx
    k0 = omega / constants.C_0
    beta = np.real(np.sqrt(vals, dtype=complex)) * k0
    if pol == "Ez":
        e = fz
        h = beta / omega / constants.MU_0 * e
        target[x, y] = np.atleast_2d(e)[:, m - 1].squeeze()
    elif pol == "Hz":
        h = fz
        e = h * omega * constants.MU_0 / beta
        target[x, y] = np.atleast_2d(h)[:, m - 1].squeeze()

    return h[:, m - 1], e[:, m - 1], beta, target
