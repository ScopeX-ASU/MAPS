"""
Date: 2024-10-05 02:02:33
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-16 16:54:26
FilePath: /MAPS/core/invdes/models/layers/parametrization/levelset.py
"""

from functools import lru_cache
from typing import Tuple

import h5py
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.types import Device

from .base_parametrization import BaseParametrization
from .utils import HeavisideProjection

__all__ = ["LeveSetParameterization"]


# class LevelSetInterp(object):
#     """This class implements the level set surface using Gaussian radial basis functions."""

#     def __init__(
#         self,
#         x0: Tensor = None,
#         y0: Tensor = None,
#         z0: Tensor = None,
#         sigma: float = 0.02,
#         interpolation: str = "gaussian",
#         device: Device = torch.device("cuda:0"),
#     ):
#         ## z0 is a tensor: first dimension is x-axis, second dimension is y-axis
#         # Input data.
#         x0 = x0.to(device)
#         y0 = y0.to(device)
#         z0 = z0.to(device)
#         self.n_phi = z0.shape
#         x, y = torch.meshgrid(x0, y0, indexing="ij")
#         xy0 = torch.column_stack((x.reshape(-1), y.reshape(-1)))
#         self.xy0 = xy0
#         self.z0 = z0
#         self.sig = sigma
#         self.interpolation = interpolation # TODO
#         self.device = device

#         # Builds the level set interpolation model.
#         if self.interpolation == "gaussian":
#             self.build_gaussian_model()

#     def build_gaussian_model(self):
#         gauss_kernel = self.gaussian(self.xy0, self.xy0)
#         self.model = torch.matmul(torch.linalg.inv(gauss_kernel), self.z0.flatten())

#     def gaussian(self, xyi, xyj):
#         # TODO no need to recalculate the distance matrix
#         dist_sq = (xyi[:, 1].reshape(-1, 1) - xyj[:, 1].reshape(1, -1)).square_() + (
#             xyi[:, 0].reshape(-1, 1) - xyj[:, 0].reshape(1, -1)
#         ).square_()
#         # return torch.exp(-dist_sq / (2 * self.sig**2))
#         return dist_sq.mul_(-1 / (2 * self.sig**2)).exp_()
#         # return dist_sq.mul_(-1 / (2 * 0.03**2)).exp_()

#     def get_ls(self, x1, y1, shape):
#         xx, yy = torch.meshgrid(x1, y1, indexing="ij")
#         xx = xx.to(self.device)
#         yy = yy.to(self.device)
#         xy1 = torch.column_stack((xx.reshape(-1), yy.reshape(-1)))
#         # ls = self.gaussian(self.xy0, xy1).T @ self.model
#         ls = self.gaussian(xy1, self.xy0) @ self.model
#         ls = ls.reshape(shape)
#         return ls  ## level set surface with the same shape as z0


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

    def build_gaussian_model(self):
        gauss_kernel = self.gaussian(self.xy0, self.xy0)
        self.model = torch.matmul(torch.linalg.inv(gauss_kernel), self.z0.flatten())

    def gaussian(self, xyi, xyj):
        dist_sq = (xyi[:, 1].reshape(-1, 1) - xyj[:, 1].reshape(1, -1)).square_() + (
            xyi[:, 0].reshape(-1, 1) - xyj[:, 0].reshape(1, -1)
        ).square_()
        return dist_sq.mul_(-1 / (2 * self.sig**2)).exp_()

    def build_linear_model(self):
        # For linear interpolation, no precomputed model is required.
        pass

    def build_bilinear_model(self):
        # For bilinear interpolation, no precomputed model is required.
        pass

    def handle_constant_dimension(self, x1, y1):
        """Handle cases where one dimension of z0 has only one knot."""
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

    def linear_interpolate(self, x1, y1):
        # Perform 1D linear interpolation for each axis independently.
        x0 = self.xy0[:, 0].unique(sorted=True)
        y0 = self.xy0[:, 1].unique(sorted=True)
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
        return self.linear_interpolate(x1, y1)

    def get_ls(self, x1, y1, shape):
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
        else:
            raise ValueError(f"Unsupported interpolation type: {self.interpolation}")

        return ls  # Level set surface with the same shape as z0


class GetLevelSetEps(nn.Module):
    def __init__(self, fw_threshold, bw_threshold, mode, device):
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
        phi_model = LevelSetInterp(
            x0=x_rho,
            y0=y_rho,
            z0=design_param,
            sigma=rho_size,
            device=design_param.device,
        )
        phi = phi_model.get_ls(x1=x_phi, y1=y_phi)
        # # Calculates the permittivities from the level set surface.
        phi = phi + self.eta
        eps_phi = self.proj(phi, sharpness, self.eta)

        # Reshapes the design parameters into a 2D matrix.
        eps = torch.reshape(eps_phi, (nx_phi, ny_phi))
        phi = torch.reshape(phi, (nx_phi, ny_phi))

        return eps, phi


class LeveSetParameterization(BaseParametrization):
    def __init__(
        self,
        *args,
        cfgs: dict = dict(
            method="levelset",
            rho_resolution=[50, 0],  #  50 knots per um, 0 means reduced dimention
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
        ),
        **kwargs,
    ):
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
        self.binary_projection = HeavisideProjection(**self.cfgs["binary_projection"])
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
        region_size: Tuple[float, float],
    ):
        n_rho = [
            int(region_s * res) + 1
            for region_s, res in zip(region_size, rho_resolution)
        ]
        ### this makes sure n_phi is the same as design_region_mask
        ## add 1 here due to leveset needs to have one more point than the design region
        n_phi = [(m.stop - m.start) for m in self.design_region_mask]

        n_hr_phi = [(m.stop - m.start) for m in self.hr_design_region_mask]

        rho = [
            torch.linspace(-region_s / 2, region_s / 2, n, device=self.operation_device)
            for region_s, n in zip(region_size, n_rho)
        ]
        ## if one dimension has rho_resolution=0, then this dimension need to be duplicated, e.g., ridge
        ## then all n_phi points need to be the same number, which is -region_s/2
        phi = [
            torch.linspace(
                -region_s / 2,
                region_s / 2 if rho_res > 0 else -region_s / 2,
                n,
                device=self.operation_device,
            )
            for region_s, n, rho_res in zip(region_size, n_phi, rho_resolution)
        ]

        ## if one dimension has rho_resolution=0, then this dimension need to be duplicated, e.g., ridge
        ## then all n_phi points need to be the same number, which is -region_s/2
        hr_phi = [
            torch.linspace(
                -region_s / 2,
                region_s / 2 if rho_res > 0 else -region_s / 2,
                n,
                device=self.operation_device,
            )
            for region_s, n, rho_res in zip(region_size, n_hr_phi, rho_resolution)
        ]

        param_dict = dict(
            n_rho=n_rho, n_phi=n_phi, rho=rho, phi=phi, n_hr_phi=n_hr_phi, hr_phi=hr_phi
        )

        return param_dict

    def _build_parameters_levelset(self, param_cfg, region_cfg):
        param_dict = self._prepare_parameters_levelset(
            tuple(param_cfg["rho_resolution"]),
            tuple(region_cfg["size"]),
        )
        n_rho = param_dict["n_rho"]
        ls_knots = nn.Parameter(
            -0.05 * torch.ones(*n_rho, device=self.operation_device)
        )

        weight_dict = dict(ls_knots=ls_knots)
        return weight_dict, param_dict

    def _reset_parameters_levelset(
        self, weight_dict, param_cfg, region_cfg, init_method: str = "random"
    ):
        init_file_path = param_cfg.get("initialization_file", None)
        if init_file_path is not None:
            assert init_method == "grating_1d", (
                "Only grating_1d init method is supported with given initialization file"
            )
            with h5py.File(init_file_path, "r") as f:
                level_set_knots = f["Si_width"][:]
                level_set_knots = (
                    torch.tensor(level_set_knots, device=self.operation_device) * 1e6
                )
                # level_set_knots.fill_(0.05)
                # print("this is the shape of level_set_knots", level_set_knots.shape, flush=True)
        if init_method == "random":
            nn.init.normal_(weight_dict["ls_knots"], mean=0, std=0.01)
        elif init_method == "ones":
            nn.init.normal_(weight_dict["ls_knots"], mean=0.05, std=0.01)
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
            nn.init.normal_(weight_dict["ls_knots"], mean=-0.05, std=0.01)
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
                weight_dict["ls_knots"].data[mask], mean=0.06, std=0.01
            )
            # weight_dict["ls_knots"].data[mask] = (1-(x.abs()**p + y.abs()**p)[mask]) * 0.1
        elif init_method == "zeros":
            nn.init.normal_(weight_dict["ls_knots"], mean=0, std=0.01)
            weight_dict["ls_knots"].data -= 0.05
        elif init_method == "rectangle":
            weight = weight_dict["ls_knots"]
            weight.data.fill_(-0.2)
            weight.data[:, weight.shape[1] // 4 : 3 * weight.shape[1] // 4] = 0.05
            weight.data += torch.randn_like(weight) * 0.01
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
        elif init_method == "crossing":
            rho_res = self.cfgs["rho_resolution"]
            half_wg_width_x = int((0.48 / 2) * rho_res[0])
            half_wg_width_y = int((0.48 / 2) * rho_res[1])

            weight = weight_dict["ls_knots"]
            weight.data.fill_(-0.2)
            weight.data[
                weight.shape[0] // 2 - half_wg_width_x : weight.shape[0] // 2
                + half_wg_width_x,
                :,
            ] = 0.05
            weight.data[
                :,
                weight.shape[1] // 2 - half_wg_width_y : weight.shape[1] // 2
                + half_wg_width_y,
            ] = 0.05
            weight.data += torch.randn_like(weight) * 0.01
        else:
            raise ValueError(f"Unsupported initialization method: {init_method}")

    def _build_permittivity(
        self, weights, rho, phi, n_phi, sharpness: float, ls_knots=None
    ):
        if ls_knots is not None:
            assert ls_knots.shape == weights["ls_knots"].shape, (
                f"the shape of ls_knots {ls_knots.shape} should be the same as the shape of ls_knots in weights {weights['ls_knots'].shape}"
            )
        sigma = getattr(self.cfgs, "sigma", 1 / max(self.cfgs["rho_resolution"]))
        interpolation = getattr(self.cfgs, "interpolation", "gaussian")
        design_param = weights["ls_knots"] if ls_knots is None else ls_knots
        ### to avoid all knots becoming unreasonable large to make it stable
        ### also to avoid most knots concentrating near threshold, otherwise, binarization will not work
        design_param = design_param / (design_param.std() + 1e-6) * 1 / 4
        phi_model = LevelSetInterp(
            x0=rho[0],
            y0=rho[1],
            z0=design_param,
            sigma=sigma,
            interpolation=interpolation,
            device=design_param.device,
        )
        phi = phi_model.get_ls(x1=phi[0], y1=phi[1], shape=n_phi)  # [76, 2001]

        ## This is used to constrain the value to be [0, 1] for heaviside input
        phi = torch.tanh(phi) * 0.5
        phi = phi.to(self.operation_device)
        phi = phi + self.eta
        eps_phi = self.binary_projection(phi, sharpness, self.eta)

        self.phi = torch.reshape(phi, n_phi)

        return eps_phi

    def build_permittivity(self, weights, sharpness: float, ls_knots=None):
        ## this is the high resolution, e.g., res=200, 310 permittivity
        ## return:
        #   1. we need the first one for gds dump out
        #   2. we need the second one for evaluation, do not need to downsample it here. transform will handle it.
        hr_permittivity = self._build_permittivity(
            weights,
            self.params["rho"],
            self.params["hr_phi"],
            self.params["n_hr_phi"],
            sharpness,
            ls_knots=ls_knots,
        )

        return hr_permittivity
