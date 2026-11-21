"""
Date: 2024-10-05 02:02:33
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-23 10:34:04
FilePath: /MAPS/core/invdes/models/layers/parametrization/levelset.py
"""

from functools import lru_cache
from typing import Callable, Tuple

import h5py
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.types import Device

from core.invdes.models.materialization.coverage import average_uniform_cells

from .base_parametrization import BaseParametrization, _convert_resolution
from .utils import HeavisideProjection, SubpixelSmoothedProjection

__all__ = ["LevelSetParameterization"]


@torch.no_grad()
def _reinit_sdf(phi: torch.Tensor, iters: int = 40, dt: float = 0.3) -> torch.Tensor:
    """
    Reinitialize a scalar level-set field toward signed-distance-function form with a Godunov-style update.

    Args:
        phi (torch.Tensor): Two-dimensional level-set field.
        iters (int): Number of reinitialization iterations. Defaults to 40.
        dt (float): Pseudo-time step for each reinitialization update. Defaults to 0.3.

    Returns:
        torch.Tensor: Detached reinitialized level-set field.
    """
    eps = 1e-6
    phi = phi.clone()
    S = phi / torch.sqrt(phi * phi + eps)

    def diffs(u):
        u_pad = F.pad(u[None, None, ...], (1, 1, 1, 1), mode="replicate")[0, 0]
        c = u_pad[..., 1:-1, 1:-1]
        xp = u_pad[..., 1:-1, 2:]
        xm = u_pad[..., 1:-1, 0:-2]
        yp = u_pad[..., 2:, 1:-1]
        ym = u_pad[..., 0:-2, 1:-1]
        Dx_f = xp - c
        Dx_b = c - xm
        Dy_f = yp - c
        Dy_b = c - ym
        return Dx_f, Dx_b, Dy_f, Dy_b

    for _ in range(iters):
        Dx_f, Dx_b, Dy_f, Dy_b = diffs(phi)
        a_pos = torch.maximum(
            torch.clamp(Dx_b, min=0) ** 2, torch.clamp(Dx_f, max=0) ** 2
        )
        b_pos = torch.maximum(
            torch.clamp(Dy_b, min=0) ** 2, torch.clamp(Dy_f, max=0) ** 2
        )
        grad_pos = torch.sqrt(a_pos + b_pos + eps)

        a_neg = torch.maximum(
            torch.clamp(Dx_f, min=0) ** 2, torch.clamp(Dx_b, max=0) ** 2
        )
        b_neg = torch.maximum(
            torch.clamp(Dy_f, min=0) ** 2, torch.clamp(Dy_b, max=0) ** 2
        )
        grad_neg = torch.sqrt(a_neg + b_neg + eps)

        G = torch.where(S >= 0, grad_pos, grad_neg)
        phi = phi - dt * S * (G - 1.0)
    return phi


def _grad_mag(phi: torch.Tensor):
    """
    Compute a forward finite-difference gradient magnitude with replicated boundary padding.

    Args:
        phi (torch.Tensor): Two-dimensional scalar field.

    Returns:
        torch.Tensor: Gradient-magnitude field with the same shape as ``phi``.
    """
    dx = phi[:, 1:] - phi[:, :-1]
    dy = phi[1:, :] - phi[:-1, :]
    dx = F.pad(dx[None, None, ...], (0, 1, 0, 0), mode="replicate")[0, 0]
    dy = F.pad(dy[None, None, ...], (0, 0, 0, 1), mode="replicate")[0, 0]
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


# --- Bilinear interpolation from knot grid (x0,y0) -> eval grid (x1,y1) ---
def _bilinear_from_knots(
    z: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate a knot-grid field onto an evaluation grid using bilinear interpolation, including degenerate one-dimensional cases.

    Args:
        z (torch.Tensor): Knot-grid values with shape ``[len(x0), len(y0)]``.
        x0 (torch.Tensor): Sorted source x coordinates.
        y0 (torch.Tensor): Sorted source y coordinates.
        x1 (torch.Tensor): Evaluation x coordinates.
        y1 (torch.Tensor): Evaluation y coordinates.

    Returns:
        torch.Tensor: Interpolated field with shape ``[len(x1), len(y1)]``.
    """
    H0, W0 = z.shape
    H1, W1 = len(x1), len(y1)

    # 1D cases
    if H0 == 1 and W0 >= 1:
        # interpolate along y only
        yy = y1.clamp(min=y0.min(), max=y0.max())
        j = torch.searchsorted(y0, yy).clamp(1, W0 - 1)
        y1l, y1u = y0[j - 1], y0[j]
        ty = (yy - y1l) / (y1u - y1l + 1e-12)
        z1 = z[0, j - 1]
        z2 = z[0, j]
        vals = (1 - ty) * z1 + ty * z2
        return vals.unsqueeze(0).repeat(H1, 1)

    if W0 == 1 and H0 >= 1:
        # interpolate along x only
        xx = x1.clamp(min=x0.min(), max=x0.max())
        i = torch.searchsorted(x0, xx).clamp(1, H0 - 1)
        x1l, x1u = x0[i - 1], x0[i]
        tx = (xx - x1l) / (x1u - x1l + 1e-12)
        z1 = z[i - 1, 0]
        z2 = z[i, 0]
        vals = (1 - tx) * z1 + tx * z2
        return vals.unsqueeze(1).repeat(1, W1)

    # 2D bilinear
    XX, YY = torch.meshgrid(x1, y1, indexing="ij")
    xf = XX.reshape(-1).clamp(min=x0.min(), max=x0.max())
    yf = YY.reshape(-1).clamp(min=y0.min(), max=y0.max())

    iu = torch.searchsorted(x0, xf).clamp(1, H0 - 1)
    ju = torch.searchsorted(y0, yf).clamp(1, W0 - 1)
    il = iu - 1
    jl = ju - 1

    xL, xU = x0[il], x0[iu]
    yL, yU = y0[jl], y0[ju]
    tx = (xf - xL) / (xU - xL + 1e-12)
    ty = (yf - yL) / (yU - yL + 1e-12)

    # gather corners
    il, iu, jl, ju = il.long(), iu.long(), jl.long(), ju.long()
    z11 = z[il, jl]
    z21 = z[iu, jl]
    z12 = z[il, ju]
    z22 = z[iu, ju]

    vals = (
        (1 - tx) * (1 - ty) * z11
        + tx * (1 - ty) * z21
        + (1 - tx) * ty * z12
        + tx * ty * z22
    )
    return vals.reshape(H1, W1)


# --- Public: linear-only projection of ls_knots to SDF ---
def project_ls_knots_to_sdf_linear(
    ls_knots: torch.Tensor,  # [len(x0), len(y0)]
    x0: torch.Tensor,
    y0: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
    *,
    reinit_iters: int = 40,
    fit_steps: int = 80,
    fit_lr: float = 5e-2,
    band_half_width: float = 2.0,
    w_fit: float = 1.0,
    w_sdf: float = 1e-3,
    clip_knots: float | None = 0.2,
) -> torch.Tensor:
    """
    Adjust level-set knots so their bilinear interpolation approximates a signed-distance function.

    Args:
        ls_knots (torch.Tensor): Knot-grid level-set parameters.
        x0 (torch.Tensor): Knot-grid x coordinates.
        y0 (torch.Tensor): Knot-grid y coordinates.
        x1 (torch.Tensor): Fine-grid x coordinates used to construct the SDF target.
        y1 (torch.Tensor): Fine-grid y coordinates used to construct the SDF target.
        reinit_iters (int): SDF reinitialization iterations. Defaults to 40.
        fit_steps (int): Inner Adam fitting iterations for the knot values. Defaults to 80.
        fit_lr (float): Inner Adam learning rate. Defaults to 5e-2.
        band_half_width (float): Half-width of the band around the zero level set used in the fitting losses. Defaults to 2.0.
        w_fit (float): Weight of the SDF-value fitting loss. Defaults to 1.0.
        w_sdf (float): Weight of the gradient-magnitude regularization loss. Defaults to 1e-3.
        clip_knots (float | None): Optional absolute clipping bound for fitted knots. Defaults to 0.2.

    Returns:
        torch.Tensor: Updated detached level-set knot tensor.
    """
    # 1) Build φ from current knots and reinitialize to SDF (target)
    with torch.no_grad():
        phi_raw = _bilinear_from_knots(ls_knots, x0, y0, x1, y1)
        phi_sdf = _reinit_sdf(phi_raw, iters=reinit_iters, dt=0.3)
        band = (phi_sdf.abs() <= band_half_width).to(phi_sdf.dtype)

    # 2) Inner optimization on knots with bilinear forward
    z = ls_knots.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=fit_lr)

    for _ in range(fit_steps):
        opt.zero_grad(set_to_none=True)
        phi_pred = _bilinear_from_knots(z, x0, y0, x1, y1)
        L_fit = ((phi_pred - phi_sdf).pow(2) * band).mean()
        L_sdf = ((_grad_mag(phi_pred) - 1.0).pow(2) * band).mean()
        (w_fit * L_fit + w_sdf * L_sdf).backward()
        opt.step()
        if clip_knots is not None:
            with torch.no_grad():
                z.clamp_(-clip_knots, clip_knots)

    return z.detach()


class LevelSetInterp(object):
    """This class implements the level set surface using various interpolation methods."""

    def __init__(
        self,
        x0: Tensor = None,
        y0: Tensor = None,
        z0: Tensor = None,
        sigma: float = 0.02,
        interpolation: str = "gaussian",
        device: Device = torch.device("cuda:0"),
    ):
        ## z0 is a tensor: first dimension is x-axis, second dimension is y-axis
        # Input data.
        """
        Initialize a level-set interpolator from a knot grid and select the interpolation backend.

        Args:
            x0 (torch.Tensor | None): One-dimensional source x coordinates. Defaults to None.
            y0 (torch.Tensor | None): One-dimensional source y coordinates. Defaults to None.
            z0 (torch.Tensor | None): Two-dimensional source level-set values. Defaults to None.
            sigma (float): Gaussian width used by Gaussian interpolation. Defaults to 0.02.
            interpolation (str): Interpolation backend: ``"gaussian"``, ``"linear"``, ``"bilinear"``, or ``"gaussian_linear"``. Defaults to ``"gaussian"``.
            device (torch.device): Device holding interpolation coordinates and kernels. Defaults to ``torch.device("cuda:0")``.
        """
        x0 = x0.to(device)
        y0 = y0.to(device)
        z0 = z0.to(device)
        self.n_phi = z0.shape
        x, y = torch.meshgrid(x0, y0, indexing="ij")
        xy0 = torch.column_stack((x.reshape(-1), y.reshape(-1)))
        self.xy0 = xy0
        self.z0 = z0
        self.sig = sigma
        self.interpolation = interpolation
        self.device = device

        # Builds the level set interpolation model.
        if self.interpolation == "gaussian":
            self.build_gaussian_model()
        elif self.interpolation == "linear":
            self.build_linear_model()
        elif self.interpolation == "bilinear":
            self.build_bilinear_model()
        elif self.interpolation == "gaussian_linear":
            self.build_gaussian_linear_model(x0, y0, sigma)

    def build_gaussian_model(self):
        """
        Build and cache the linear-system coefficients for Gaussian radial-basis interpolation.

        Returns:
            None: Stores the interpolation model in ``self.model``.
        """
        gauss_kernel = self.gaussian(self.xy0, self.xy0)
        self.model = torch.matmul(torch.linalg.inv(gauss_kernel), self.z0.flatten())

    def gaussian(self, xyi, xyj):
        """
        Evaluate the Gaussian radial-basis kernel between two coordinate sets.

        Args:
            xyi (torch.Tensor): First coordinate set with columns ``(x, y)``.
            xyj (torch.Tensor): Second coordinate set with columns ``(x, y)``.

        Returns:
            torch.Tensor: Pairwise Gaussian kernel matrix using ``self.sig``.
        """
        dist_sq = (xyi[:, 1].reshape(-1, 1) - xyj[:, 1].reshape(1, -1)).square_() + (
            xyi[:, 0].reshape(-1, 1) - xyj[:, 0].reshape(1, -1)
        ).square_()
        return dist_sq.mul_(-1 / (2 * self.sig**2)).exp_()

    def build_linear_model(self):
        # For linear interpolation, no precomputed model is required.
        """
        Select the linear interpolation backend without precomputing coefficients.

        Returns:
            None: No persistent model state is required.
        """
        pass

    def build_bilinear_model(self):
        # For bilinear interpolation, no precomputed model is required.
        """
        Select the bilinear interpolation backend without precomputing coefficients.

        Returns:
            None: No persistent model state is required.
        """
        pass

    def build_gaussian_linear_model(self, x0, y0, sigma):
        ### we use 3 sigma rule to build the convolution kernel
        ## sigma: typically the rho_size in um unit
        """
        Build and normalize a finite Gaussian convolution kernel used before linear interpolation.

        Args:
            x0 (torch.Tensor): Source x coordinates used to determine grid spacing.
            y0 (torch.Tensor): Source y coordinates used to determine grid spacing.
            sigma (float): Gaussian standard deviation in the same physical units as the coordinate grid.

        Returns:
            None: Stores the normalized kernel in ``self.gauss_kernel``.
        """
        grid_x, grid_y = x0[1] - x0[0], y0[1] - y0[0]
        ## then the kernel size in number of points
        kernel_size_x = int((3 * sigma) / grid_x) * 2 + 1
        kernel_size_y = int((3 * sigma) / grid_y) * 2 + 1
        ## create gaussian kernel
        grid_xx, grid_yy = torch.meshgrid(
            torch.linspace(
                -grid_x * (kernel_size_x // 2),
                grid_x * (kernel_size_x // 2),
                kernel_size_x,
                device=self.device,
            ),
            torch.linspace(
                -grid_y * (kernel_size_y // 2),
                grid_y * (kernel_size_y // 2),
                kernel_size_y,
                device=self.device,
            ),
            indexing="ij",
        )
        self.gauss_kernel = torch.exp(((grid_xx**2 + grid_yy**2) / (-2 * (sigma**2))))[
            None, None
        ]  # [1, 1, kx, ky]
        self.gauss_kernel = self.gauss_kernel / self.gauss_kernel.sum()

    def handle_constant_dimension(self, x1, y1):
        """
        Handle level-set grids with only one knot along one axis by linearly interpolating along the varying axis and repeating along the constant axis.

        Args:
            x1 (torch.Tensor): Target x coordinates.
            y1 (torch.Tensor): Target y coordinates.

        Returns:
            torch.Tensor | None: Interpolated constant-dimension field, or None when both source dimensions vary.
        """
        if self.n_phi[0] == 1:  # Single knot along the x-axis
            # Interpolate along the y-axis
            z_values = self.z0[0, :].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 134]
            z_interpolated = F.interpolate(
                z_values, size=(len(y1),), mode="linear", align_corners=True
            )
            z_const = z_interpolated.squeeze(0).repeat(
                len(x1), 1
            )  # Repeat along x-axis

        elif self.n_phi[1] == 1:  # Single knot along the y-axis
            # Interpolate along the x-axis
            z_values = self.z0[:, 0].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 134]
            z_interpolated = F.interpolate(
                z_values, size=(len(x1),), mode="linear", align_corners=True
            )

            z_const = (
                z_interpolated.squeeze(0).repeat(len(y1), 1).t()
            )  # Repeat along y-axis

        else:
            z_const = None  # No constant dimension
        return z_const

    def linear_interpolate(self, x1, y1, z0=None):
        # Perform 1D linear interpolation for each axis independently.
        """
        Evaluate piecewise bilinear interpolation from the source knot grid using independent coordinate searches.

        Args:
            x1 (torch.Tensor): Target x coordinates.
            y1 (torch.Tensor): Target y coordinates.
            z0 (torch.Tensor | None): Optional source field overriding ``self.z0``. Defaults to None.

        Returns:
            torch.Tensor: Interpolated target field.
        """
        x0 = self.xy0[:, 0].unique(sorted=True)
        y0 = self.xy0[:, 1].unique(sorted=True)
        if z0 is None:
            z0 = self.z0.reshape(len(x0), len(y0))

        x_idx = torch.searchsorted(x0, x1.clamp(min=x0.min(), max=x0.max()))
        y_idx = torch.searchsorted(y0, y1.clamp(min=y0.min(), max=y0.max()))

        x0_1 = x0[x_idx - 1]
        x0_2 = x0[x_idx]
        y0_1 = y0[y_idx - 1]
        y0_2 = y0[y_idx]

        z_x1_y1 = z0[x_idx - 1, y_idx - 1]
        z_x2_y1 = z0[x_idx, y_idx - 1]
        z_x1_y2 = z0[x_idx - 1, y_idx]
        z_x2_y2 = z0[x_idx, y_idx]

        wx1 = (x0_2 - x1) / (x0_2 - x0_1)
        wx2 = (x1 - x0_1) / (x0_2 - x0_1)
        wy1 = (y0_2 - y1) / (y0_2 - y0_1)
        wy2 = (y1 - y0_1) / (y0_2 - y0_1)

        return (
            wx1 * wy1 * z_x1_y1
            + wx2 * wy1 * z_x2_y1
            + wx1 * wy2 * z_x1_y2
            + wx2 * wy2 * z_x2_y2
        )

    def bilinear_interpolate(self, x1, y1):
        # Use bilinear interpolation, equivalent to linear in 2D.
        """
        Evaluate the class bilinear interpolation path, which delegates to ``linear_interpolate``.

        Args:
            x1 (torch.Tensor): Target x coordinates.
            y1 (torch.Tensor): Target y coordinates.

        Returns:
            torch.Tensor: Interpolated target field.
        """
        return self.linear_interpolate(x1, y1)

    def gaussian_linear_interpolate(self, x1, y1):
        ## first use the gaussian kernel to convolve the z0
        # z0: [len(x0), len(y0)]
        # gauss_kernel: [1, 1, kx, ky]
        """
        Smooth the source level-set grid with the cached Gaussian kernel and then perform linear interpolation.

        Args:
            x1 (torch.Tensor): Target x coordinates.
            y1 (torch.Tensor): Target y coordinates.

        Returns:
            torch.Tensor: Smoothed and linearly interpolated level-set field.
        """
        pad_x = self.gauss_kernel.shape[2] // 2
        pad_y = self.gauss_kernel.shape[3] // 2
        z0_padded = F.pad(
            self.z0.unsqueeze(0).unsqueeze(0),
            (pad_y, pad_y, pad_x, pad_x),
            mode="replicate",
        )
        z0 = F.conv2d(
            z0_padded,
            self.gauss_kernel,
        )[
            0, 0
        ]  # [H, W]
        return self.linear_interpolate(x1, y1, z0=z0)

    def get_ls(self, x1, y1, shape):
        """
        Evaluate the configured level-set interpolator on a target coordinate grid.

        Args:
            x1 (torch.Tensor): Target x coordinates.
            y1 (torch.Tensor): Target y coordinates.
            shape (tuple[int, int]): Output field shape.

        Returns:
            torch.Tensor: Level-set surface evaluated at the target coordinates.
        """
        xx, yy = torch.meshgrid(x1, y1, indexing="ij")
        xx = xx.to(self.device)
        yy = yy.to(self.device)

        # Handle the constant dimension case
        z_const = self.handle_constant_dimension(x1, y1)
        if z_const is not None:
            return z_const  # Return the constant level set surface with interpolation

        if self.interpolation == "gaussian":
            xy1 = torch.column_stack((xx.reshape(-1), yy.reshape(-1)))
            ls = self.gaussian(xy1, self.xy0) @ self.model
            ls = ls.reshape(shape)
        elif self.interpolation == "linear":
            ls = self.linear_interpolate(xx.flatten(), yy.flatten())
            ls = ls.reshape(shape)
        elif self.interpolation == "bilinear":
            ls = self.bilinear_interpolate(xx.flatten(), yy.flatten())
            ls = ls.reshape(shape)
        elif self.interpolation == "gaussian_linear":
            ls = self.gaussian_linear_interpolate(xx.flatten(), yy.flatten())
            ls = ls.reshape(shape)
        else:
            raise ValueError(f"Unsupported interpolation type: {self.interpolation}")

        return ls  # Level set surface with the same shape as z0


class GetLevelSetDensity(nn.Module):
    def __init__(self, fw_threshold, bw_threshold, mode, device):
        """
        Initialize the level-set density projection wrapper.

        Args:
            fw_threshold (float): Forward projection threshold.
            bw_threshold (float): Backward projection threshold.
            mode (str): Heaviside projection mode.
            device (torch.device): Device for projection state.
        """
        super().__init__()
        self.fw_threshold = fw_threshold
        self.bw_threshold = bw_threshold
        self.proj = HeavisideProjection(fw_threshold, bw_threshold, mode)
        self.device = device
        self.eta = torch.tensor(
            [
                0.5,
            ],
            device=device,
        )  # 0.5 is hard coded here since this is only the level set, don't need to consider value other than 0.5

    def forward(
        self,
        design_param,
        x_rho,
        y_rho,
        x_phi,
        y_phi,
        rho_size,
        nx_phi,
        ny_phi,
        sharpness,
    ):
        """
        Generate a normalized level-set density and its corresponding level-set field on the requested evaluation grid.

        Args:
            design_param (torch.Tensor): Level-set knot parameters.
            x_rho (torch.Tensor): Knot-grid x coordinates.
            y_rho (torch.Tensor): Knot-grid y coordinates.
            x_phi (torch.Tensor): Level-set evaluation-grid x coordinates.
            y_phi (torch.Tensor): Level-set evaluation-grid y coordinates.
            rho_size (float): Gaussian smoothing width passed to ``LevelSetInterp``.
            nx_phi (int): Output x dimension.
            ny_phi (int): Output y dimension.
            sharpness (float): Projection sharpness parameter.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Density field and reshaped level-set field.
        """
        phi_model = LevelSetInterp(
            x0=x_rho,
            y0=y_rho,
            z0=design_param,
            sigma=rho_size,
            device=design_param.device,
        )
        phi = phi_model.get_ls(x1=x_phi, y1=y_phi)
        # Calculates the normalized density from the level set surface.
        phi = phi + self.eta
        density_phi = self.proj(phi, sharpness, self.eta)

        # Reshapes the design parameters into a 2D matrix.
        density = torch.reshape(density_phi, (nx_phi, ny_phi))
        phi = torch.reshape(phi, (nx_phi, ny_phi))

        return density, phi


class LevelSetParameterization(BaseParametrization):
    def __init__(
        self,
        *args,
        cfgs: dict = dict(
            method="levelset",
            rho_resolution=[50, 0],  #  50 knots per um, 0 means reduced dimention
            phi_resolution=[50, 50],  # uniform physical level-set evaluation grid
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
            transform=[
                dict(
                    type="mirror_symmetry",  # Mirror symmetry
                    dims=[],  # Symmetry dimensions
                ),
                dict(type="transpose_symmetry", flag=False),  # Transpose symmetry
            ],
            init_method="random",
            denorm_mode="linear_eps",
            dims=(0, 1),  # the levelset2d describes x-y plane.
            extrude_direction="-",  # extrude in the negative z direction
            extrude_angle=0.0,  # extrude vertically
        ),
        **kwargs,
    ):
        """
        Initialize the level-set parameterization, build its learnable knots, and configure binary/subpixel projection state.

        Args:
            *args: Positional arguments forwarded to ``BaseParametrization``.
            cfgs (dict): Level-set configuration containing interpolation resolution, projection, transform, initialization, dimensions, and extrusion settings. Defaults to the built-in level-set configuration defined in the signature.
            **kwargs: Additional keyword arguments forwarded to ``BaseParametrization``.
        """
        super().__init__(*args, cfgs=cfgs, **kwargs)

        method = cfgs["method"]

        self.register_parameter_build_per_region_fn(
            method, self._build_parameters_levelset
        )
        self.register_parameter_reset_per_region_fn(
            method, self._reset_parameters_levelset
        )

        self.build_parameters(cfgs, self.design_region_cfg)
        self.reset_parameters(cfgs, self.design_region_cfg)
        # self.binary_projection = HeavisideProjection(**self.cfgs["binary_projection"])
        self.binary_projection = SubpixelSmoothedProjection(
            projection_midpoint=0.5, resolution=None
        )
        self.eta = torch.tensor(
            [
                0.5,
            ],
            device=self.operation_device,
        )

    @lru_cache(maxsize=3)
    def _prepare_parameters_levelset(
        self,
        rho_resolution: Tuple[int, int],
        region_size: Tuple[float, float] | Tuple[float, float, float],
        dims: Tuple[int, int] = (0, 1),
        phi_resolution: Tuple[int, int] | None = None,
    ):
        """
        Construct level-set knot/evaluation coordinates and dimensions from physical region size and requested resolutions.

        Args:
            rho_resolution (tuple[int, int]): Knot resolution in points per physical unit; non-positive entries create a single evaluation coordinate along that axis.
            region_size (tuple[float, float] | tuple[float, float, float]): Physical design-region dimensions.
            dims (tuple[int, int]): Physical axes represented by the 2-D level set. Defaults to ``(0, 1)``.
            phi_resolution (tuple[int, int] | None): Optional evaluation-grid resolution. Defaults to None, which uses ``rho_resolution``.

        Returns:
            dict: Dictionary containing ``n_rho``, ``n_phi``, ``rho``, and ``phi`` coordinate metadata.

        Raises:
            ValueError: If ``phi_resolution`` has the wrong dimensionality.
        """
        region_size = tuple(region_size)
        region_size = tuple([region_size[d] for d in dims])
        n_rho = [
            int(region_s * res) + 1
            for region_s, res in zip(region_size, rho_resolution)
        ]
        phi_resolution = tuple(phi_resolution or rho_resolution)
        if len(phi_resolution) != len(region_size):
            raise ValueError(
                "phi_resolution must have one entry for each level-set dimension"
            )
        n_phi = [
            1 if res <= 0 else max(2, int(round(region_s * res)) + 1)
            for region_s, res in zip(region_size, phi_resolution)
        ]

        rho = [
            torch.linspace(-region_s / 2, region_s / 2, n, device=self.operation_device)
            for region_s, n in zip(region_size, n_rho)
        ]
        ## if one dimension has rho_resolution=0, then this dimension need to be duplicated, e.g., ridge
        ## then all n_phi points need to be the same number, which is -region_s/2
        phi = [
            torch.linspace(
                -region_s / 2,
                region_s / 2 if phi_res > 0 else -region_s / 2,
                n,
                device=self.operation_device,
            )
            for region_s, n, phi_res in zip(region_size, n_phi, phi_resolution)
        ]

        param_dict = dict(
            n_rho=n_rho,
            n_phi=n_phi,
            rho=rho,
            phi=phi,
        )

        return param_dict

    def _build_parameters_levelset(self, param_cfg, region_cfg):
        """
        Build the learnable level-set knot parameter and associated coordinate metadata for one design region.

        Args:
            param_cfg (dict): Level-set parameterization configuration.
            region_cfg (dict): Design-region geometry configuration.

        Returns:
            tuple[dict, dict]: Parameter mapping containing ``ls_knots`` and coordinate metadata.
        """
        dims = sorted(
            self.cfgs.get("dims", (0, 1))
        )  # the dimensions that the level set is defined on, default to [0, 1] for x-y plane

        param_dict = self._prepare_parameters_levelset(
            tuple(param_cfg["rho_resolution"]),
            tuple(region_cfg["size"]),
            dims=tuple(dims),
            phi_resolution=tuple(
                param_cfg.get("phi_resolution", param_cfg["rho_resolution"])
            ),
        )
        n_rho = param_dict["n_rho"]
        ls_knots = nn.Parameter(torch.zeros(*n_rho, device=self.operation_device))

        weight_dict = dict(ls_knots=ls_knots)
        return weight_dict, param_dict

    def reset_levelset_sdf(self):
        """
        Project the current level-set knots toward a signed-distance-function representation in place.

        Returns:
            None: Updates ``self.weights["ls_knots"]`` under ``torch.no_grad``.
        """
        weights = self.weights
        rho = self.params["rho"]
        phi = self.params["phi"]
        x0, y0 = rho[0], rho[1]
        x1, y1 = phi[0], phi[1]
        weights["ls_knots"].data.copy_(
            project_ls_knots_to_sdf_linear(
                weights["ls_knots"],
                x0,
                y0,
                x1,
                y1,
                reinit_iters=40,
                fit_steps=80,
                fit_lr=0.05,
            )
        )

    def _reset_parameters_levelset(
        self, weight_dict, param_cfg, region_cfg, init_method: str = "random"
    ):
        """
        Initialize level-set knot values according to the configured initialization method or external HDF5 source.

        Args:
            weight_dict (dict[str, torch.nn.Parameter]): Learnable level-set parameters.
            param_cfg (dict): Level-set parameterization configuration.
            region_cfg (dict): Design-region geometry configuration.
            init_method (str | Callable): Initialization strategy. Defaults to ``"random"``.

        Raises:
            ValueError: If the requested initialization method is unsupported.
        """
        init_file_path = param_cfg.get("initialization_file", None)
        if isinstance(init_method, Callable):
            init_method = init_method(weight_dict, param_cfg, region_cfg)
            return
        if init_file_path is not None:
            assert (
                init_method == "grating_1d"
            ), "Only grating_1d init method is supported with given initialization file"
            with h5py.File(init_file_path, "r") as f:
                level_set_knots = f["Si_width"][:]
                level_set_knots = (
                    torch.tensor(level_set_knots, device=self.operation_device) * 1e6
                )
                # level_set_knots.fill_(0.05)
                # print("this is the shape of level_set_knots", level_set_knots.shape, flush=True)
        if init_method == "random":
            nn.init.normal_(weight_dict["ls_knots"], mean=0, std=0.0001)
        elif init_method.startswith("random_"):
            p = init_method.split("_")
            if len(p) > 1:
                p = float(p[-1])
            else:
                p = 0.01
            nn.init.uniform_(weight_dict["ls_knots"], a=-p, b=p)
        elif init_method == "ones":
            nn.init.normal_(weight_dict["ls_knots"], mean=0.05, std=0.01)
        elif init_method.startswith("constant"):
            p = init_method.split("_")
            if len(p) > 1:
                p = float(p[-1])
            else:
                p = 0.0  # 0 means middle point
            weight_dict["ls_knots"].data.fill_(p)
        elif init_method == "checkerboard":
            ## make a checkerboard pattern
            weight_dict["ls_knots"].data.fill_(-0.05)
            period = 16
            for i in range(period // 2):
                for j in range(period // 2):
                    weight_dict["ls_knots"].data[i::period, j::period] = 0.05
            for i in range(period // 2, period):
                for j in range(period // 2, period):
                    weight_dict["ls_knots"].data[i::period, j::period] = 0.05
        elif init_method == "ball":  ## same as diamond_2
            nn.init.normal_(weight_dict["ls_knots"], mean=0, std=0.01)
            ### create a map with center high, edge low as a ball using mesh grid
            x, y = torch.meshgrid(
                torch.linspace(-1, 1, weight_dict["ls_knots"].shape[0]),
                torch.linspace(-1, 1, weight_dict["ls_knots"].shape[1]),
            )
            mask = (x.square() + y.square()) < 1
            weight_dict["ls_knots"].data += -0.05
            weight_dict["ls_knots"].data[mask] += 0.1
        elif init_method.startswith("diamond"):
            p = init_method.split("_")
            if len(p) > 1:
                p = float(p[-1])
            else:
                p = 1
            # nn.init.normal_(weight_dict["ls_knots"], mean=-0.1, std=0.0001)
            nn.init.normal_(weight_dict["ls_knots"], mean=-0.3, std=0.01)
            ### create a map with center high, edge low as a ball using mesh grid
            x, y = torch.meshgrid(
                torch.linspace(
                    -1,
                    1,
                    weight_dict["ls_knots"].shape[0],
                    device=self.operation_device,
                ),
                torch.linspace(
                    -1,
                    1,
                    weight_dict["ls_knots"].shape[1],
                    device=self.operation_device,
                ),
            )
            # P=1, 1.2, p=0.5, 1.3, P=0.3, 1.5
            mask = (x.abs() ** p + y.abs() ** p) < 1.5
            # mask_ring = ((x.abs() ** 2 + y.abs() ** 2) < 0.6) & ((x.abs() ** 2 + y.abs() ** 2) > 0.3)
            # mask = mask | mask_ring
            # weight_dict["ls_knots"].data[mask] = nn.init.normal_(weight_dict["ls_knots"].data[mask], mean=0.05, std=0.01)
            weight_dict["ls_knots"].data[mask] = nn.init.normal_(
                weight_dict["ls_knots"].data[mask], mean=0.3, std=0.01
            )
            # weight_dict["ls_knots"].data[mask] = (1-(x.abs()**p + y.abs()**p)[mask]) * 0.1
        elif init_method == "zeros":
            # nn.init.normal_(weight_dict["ls_knots"], mean=0, std=0.01)
            weight_dict["ls_knots"].data.fill_(-0.25)
        elif init_method == "rectangle":
            weight = weight_dict["ls_knots"]
            # weight.data.fill_(-0.2)
            # weight.data[:, weight.shape[1] // 4 : 3 * weight.shape[1] // 4] = 0.05
            # weight.data += torch.randn_like(weight) * 0.01
            weight.data.fill_(-1)
            weight.data[:, weight.shape[1] // 4 : 3 * weight.shape[1] // 4] = 1
        elif init_method.startswith("grating_1d"):
            method = init_method.split("_")
            if len(method) > 1:
                method = method[-1]
            else:
                method = ""
            weight = weight_dict["ls_knots"]
            weight.data.fill_(-0.2)
            rho_res = self.cfgs["rho_resolution"]
            if weight.shape[0] == 1:
                rho_res = rho_res[1]
                n_gratings = weight.shape[1] // 2  # 0 1 0 1 0, 2 gratings
            elif weight.shape[1] == 1:
                rho_res = rho_res[0]
                n_gratings = weight.shape[0] // 2  # 0 1 0 1 0, 2 gratings
                grating_widths = torch.linspace(0, 2 / rho_res, n_gratings)
            else:
                raise ValueError("Unsupported grating initialization method")

            if init_file_path is not None:
                grating_widths = level_set_knots.squeeze()
            elif method == "random":
                grating_widths = torch.empty(n_gratings).uniform_(0, 2 / rho_res)
            elif method == "minmax":
                grating_widths = torch.empty(n_gratings)
                grating_widths[0::2] = 0.05
                grating_widths[1::2] = 0.08
            elif method in {""}:
                grating_widths = torch.linspace(0, 2 / rho_res, n_gratings)
            else:
                try:
                    grating_widths = float(method)
                except ValueError:
                    raise ValueError(
                        f"Unsupported grating initialization method: {method}"
                    )

            weight_values = (
                0.2 * grating_widths / 2 / (1 / rho_res - grating_widths / 2)
            )
            if weight.shape[0] == 1:
                weight.data[:, 1::2] = weight_values
            elif weight.shape[1] == 1:
                weight.data[1::2, :] = weight_values
        elif init_method == "ring":
            # region_cfg
            #     design_region_cfgs["bending_region"] = dict(
            #         type="box",
            #         center=[0, 0],
            #         size=box_size,
            #         eps=eps_r_fn(wl_cen),
            #         eps_bg=eps_bg_fn(wl_cen),
            #     )
            # param_cfg
            #     design_region_param_cfgs[region_name] = dict(
            #         method="levelset",
            #         rho_resolution=[25, 25],
            #         transform=[dict(type="transpose_symmetry", flag=True), dict(type="blur", mfs=0.1, resolutions=[310, 310])],
            #         # init_method="random", # this can only converge to fwd transmission ~ 25% in TE1 mode
            #         init_method="ring",
            #         binary_projection=dict(
            #             fw_threshold=100,
            #             bw_threshold=100,
            #             mode="regular",
            #         ),
            #     )
            weight = weight_dict["ls_knots"]
            weight.data.fill_(-0.2)
            # print("this is the shape of weight.data", weight.data.shape, flush=True) #(66, 66)
            box_size_x = region_cfg["size"][0]
            box_size_y = region_cfg["size"][1]
            x_ax = torch.linspace(
                0, box_size_x, weight.data.shape[0], device=self.operation_device
            )
            y_ax = torch.linspace(
                0, box_size_y, weight.data.shape[1], device=self.operation_device
            )
            x_ax, y_ax = torch.meshgrid(x_ax, y_ax)
            r = torch.sqrt(x_ax**2 + y_ax**2)
            half_wg_width = (
                0.48 / 2
            )  # 0.48 is hard coded here since for now we only need to consider the wg that support TE1 mode
            quater_ring_mask = torch.logical_and(
                r < (box_size_x / 2 + half_wg_width),
                r > (box_size_x / 2 - half_wg_width),
            )
            weight.data[quater_ring_mask] = 0.05
        elif init_method.startswith("bending"):
            width = init_method.split("_")
            if len(width) > 1:
                width = float(width[-1])
            else:
                width = 0.5
            rho_res = self.cfgs["rho_resolution"]
            half_wg_width_x = int((width / 2) * rho_res[0])
            half_wg_width_y = int((width / 2) * rho_res[1])

            weight = weight_dict["ls_knots"]
            weight.data.fill_(-0.3)
            weight.data[
                weight.shape[0] // 2
                - half_wg_width_x : weight.shape[0] // 2
                + half_wg_width_x,
                :,
            ] = 0.3
            weight.data[
                :,
                weight.shape[1] // 2
                - half_wg_width_y : weight.shape[1] // 2
                + half_wg_width_y,
            ] = 0.3
        elif init_method.startswith("bending"):
            # Add bending initialization logic here, e.g., bending_0.5, 0.5um width circular bending
            width = init_method.split("_")
            if len(width) > 1:
                width = float(width[-1])
            else:
                width = 0.5
            rho_res = self.cfgs["rho_resolution"]
            weight = weight_dict["ls_knots"]
            weight.data.fill_(-0.2)
            box_size_x = region_cfg["size"][0]
            box_size_y = region_cfg["size"][1]
            x_ax = torch.linspace(
                0, box_size_x, weight.data.shape[0], device=self.operation_device
            )
            y_ax = torch.linspace(
                0, box_size_y, weight.data.shape[1], device=self.operation_device
            )
            x_ax, y_ax = torch.meshgrid(x_ax, y_ax)
            r = torch.sqrt((x_ax) ** 2 + (y_ax) ** 2)

            ## bending radius is determined by the region region size
            bending_radius = min(box_size_x, box_size_y) / 2
            half_wg_width = width / 2  # 0.48um waveguide width
            bending_mask = torch.logical_and(
                r < (bending_radius + half_wg_width),
                r > (bending_radius - half_wg_width),
            )
            weight.data[bending_mask] = 0.2

        else:
            raise ValueError(f"Unsupported initialization method: {init_method}")

    def _build_density(self, weights, rho, phi, n_phi, sharpness: float, ls_knots=None):
        """
        Interpolate level-set knots onto the evaluation grid and produce the normalized level-set density field.

        Args:
            weights (dict[str, torch.Tensor]): Level-set learnable weights.
            rho (list[torch.Tensor]): Knot-grid coordinate tensors.
            phi (list[torch.Tensor]): Evaluation-grid coordinate tensors.
            n_phi (list[int]): Evaluation-grid shape.
            sharpness (float): Projection sharpness parameter.
            ls_knots (torch.Tensor | None): Optional alternative knot tensor used for sensitivity/evaluation. Defaults to None.

        Returns:
            torch.Tensor: High-resolution normalized density field.
        """
        if ls_knots is not None:
            assert (
                ls_knots.shape == weights["ls_knots"].shape
            ), f"the shape of ls_knots {ls_knots.shape} should be the same as the shape of ls_knots in weights {weights['ls_knots'].shape}"
        sigma = self.cfgs.get("sigma", 1 / max(self.cfgs["rho_resolution"]))
        interpolation = self.cfgs.get("interpolation", "gaussian")
        design_param = weights["ls_knots"] if ls_knots is None else ls_knots
        ### to avoid all knots becoming unreasonable large to make it stable
        ### also to avoid most knots concentrating near threshold, otherwise, binarization will not work
        # design_param = design_param / (design_param.std() + 1e-6) * 1 / 4
        phi_model = LevelSetInterp(
            x0=rho[0],
            y0=rho[1],
            z0=design_param,
            sigma=sigma,
            interpolation=interpolation,
            device=design_param.device,
        )

        phi = phi_model.get_ls(x1=phi[0], y1=phi[1], shape=n_phi)

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # axes[0].imshow(design_param.data.cpu().numpy())
        # axes[1].imshow(phi.data.cpu().numpy())
        # plt.savefig("phi.png")
        # input("Press Enter to continue...")
        # exit(0)

        ## This is used to constrain the value to be [0, 1] for heaviside input
        # phi = torch.tanh(phi) * 0.5
        phi = phi.to(self.operation_device)
        phi = phi + self.eta  # already in [0, 1]
        density_phi = phi
        # density_phi = self.binary_projection(phi, sharpness, self.eta)

        self.phi = torch.reshape(phi, n_phi)

        return density_phi

    def _region_coordinates(self, device, region_mask):
        """
        Extract physical optical-grid coordinates inside a design-region mask relative to the region center.

        Args:
            device (object): Device-like object exposing ``optical_grid_metadata``.
            region_mask (tuple[slice, ...]): Per-axis mask selecting the design region.

        Returns:
            tuple[torch.Tensor, ...] | None: Center-relative coordinates on the configured level-set dimensions, or None when grid metadata is unavailable.
        """
        metadata = getattr(device, "optical_grid_metadata", None)
        if metadata is None:
            return None
        dims = sorted(self.cfgs.get("dims", (0, 1)))
        coords = metadata.get("coords")
        if coords is None:
            return None
        region_center = self.design_region_cfg["center"]
        return tuple(
            torch.as_tensor(
                coords[dim][region_mask[dim]] - float(region_center[dim]),
                dtype=torch.float32,
                device=self.operation_device,
            )
            for dim in dims
        )

    def _resample_density_to_hr(self, density, target_size):
        """
        Resample an evaluation-grid density onto the high-resolution design-region grid.

        Args:
            density (torch.Tensor): Density on the level-set evaluation grid.
            target_size (tuple[int, ...]): Target high-resolution spatial size.

        Returns:
            torch.Tensor: Density sampled on the high-resolution grid.
        """
        source_coords = self.params["phi"]
        if source_coords is not None:
            dims = sorted(self.cfgs.get("dims", (0, 1)))
            absolute_coords = tuple(
                coord + float(self.design_region_cfg["center"][dim])
                for coord, dim in zip(source_coords, dims)
            )
            averaged = self._resample_volume_fraction_on_grid(
                density, absolute_coords, self.hr_device, self.hr_design_region_mask
            )
            if averaged is not None:
                return averaged
        target_coords = self._region_coordinates(
            self.hr_device, self.hr_design_region_mask
        )
        if target_coords is None:
            return _convert_resolution(
                density, intplt_mode="nearest", target_size=target_size
            )
        return _bilinear_from_knots(
            density,
            source_coords[0],
            source_coords[1],
            target_coords[0],
            target_coords[1],
        )

    def _resample_density_to_native(self, density, target_size, eps_r, eps_bg):
        """
        Resample an evaluation-grid density onto the native simulation grid, optionally preserving material-aware subpixel interpolation.

        Args:
            density (torch.Tensor): Density on the level-set evaluation grid.
            target_size (tuple[int, ...]): Target native-grid spatial size.
            eps_r (float | torch.Tensor): Design-region material permittivity.
            eps_bg (float | torch.Tensor): Background material permittivity.

        Returns:
            torch.Tensor: Density sampled on the native grid.
        """
        source_coords = self.params["phi"]
        target_coords = self._region_coordinates(self.device, self.design_region_mask)
        if source_coords is not None:
            dims = sorted(self.cfgs.get("dims", (0, 1)))
            absolute_coords = tuple(
                coord + float(self.design_region_cfg["center"][dim])
                for coord, dim in zip(source_coords, dims)
            )
            averaged = self._resample_volume_fraction_on_grid(
                density, absolute_coords, self.device, self.design_region_mask
            )
            if averaged is not None:
                return averaged
        if source_coords is None or target_coords is None:
            return average_uniform_cells(density, target_size)
        return _bilinear_from_knots(
            density,
            source_coords[0],
            source_coords[1],
            target_coords[0],
            target_coords[1],
        )

    def build_density(self, weights, sharpness: float, ls_knots=None):
        ## this is the high resolution, e.g., res=200, 310 normalized density
        """
        Build the high-resolution normalized density field from the current level-set parameters.

        Args:
            weights (dict[str, torch.Tensor]): Learnable level-set weights.
            sharpness (float): Projection sharpness parameter.
            ls_knots (torch.Tensor | None): Optional alternative knot tensor for evaluation. Defaults to None.

        Returns:
            torch.Tensor: High-resolution normalized density field.
        """
        hr_density = self._build_density(
            weights,
            self.params["rho"],
            self.params["phi"],
            self.params["n_phi"],
            sharpness,
            ls_knots=ls_knots,
        )

        return hr_density

    def add_roughness(
        self,
        density: Tensor,
        *,
        A: float,
        Lc: float,
        seed: int | None = None,
        **kwargs,
    ) -> Tensor:
        """
        Apply correlated stochastic boundary roughness along the level-set normal direction while preserving the binary-density envelope.

        Args:
            density (torch.Tensor): Two-dimensional density field.
            A (float): Roughness amplitude in the design-region physical units.
            Lc (float): Correlation length in the design-region physical units.
            seed (int | None): Optional random seed for deterministic roughness. Defaults to None.
            **kwargs: Additional implementation-specific options forwarded to the roughness helper.
        """
        if density.ndim != 2 or not hasattr(self, "phi"):
            raise ValueError(
                "LevelSet add_roughness requires a 2D density and level-set field"
            )
        phi = self.phi.to(device=density.device, dtype=density.dtype)
        if tuple(phi.shape) != tuple(density.shape):
            phi = F.interpolate(
                phi[None, None],
                size=tuple(density.shape),
                mode="bilinear",
                align_corners=True,
            )[0, 0]

        region_size = tuple(float(v) for v in self.design_region_cfg["size"][:2])
        spacing = tuple(
            size / max(length - 1, 1)
            for size, length in zip(region_size, density.shape)
        )
        phi_signed = phi - self.eta.to(device=phi.device, dtype=phi.dtype)
        phi_grad_x = torch.gradient(
            phi_signed, spacing=(spacing[0],), dim=(0,), edge_order=1
        )[0]
        phi_grad_y = torch.gradient(
            phi_signed, spacing=(spacing[1],), dim=(1,), edge_order=1
        )[0]
        grad_mag = torch.sqrt(phi_grad_x.square() + phi_grad_y.square() + 1e-12)
        normal_x = phi_grad_x / grad_mag
        normal_y = phi_grad_y / grad_mag
        density_grad_x = torch.gradient(
            density, spacing=(spacing[0],), dim=(0,), edge_order=1
        )[0]
        density_grad_y = torch.gradient(
            density, spacing=(spacing[1],), dim=(1,), edge_order=1
        )[0]
        displacement = self._correlated_roughness_2d(
            density,
            A=float(A),
            Lc=float(Lc),
            seed=seed,
        )
        normal_density_gradient = normal_x * density_grad_x + normal_y * density_grad_y

        class ClipSTEFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                return input_tensor.clamp(0, 1)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        return ClipSTEFunc.apply(
            density - (displacement * normal_density_gradient).detach()
        )

    def build_permittivity(self, weights, sharpness: float, ls_knots=None):
        return self.build_density(weights, sharpness, ls_knots=ls_knots)
