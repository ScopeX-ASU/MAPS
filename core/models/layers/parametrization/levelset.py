"""
Date: 2024-10-05 02:02:33
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-05 02:06:40
FilePath: /Metasurface-Opt/core/models/parametrization/levelset.py
"""

from functools import lru_cache
from typing import Tuple
import torch

from torch import nn, Tensor
from torch.types import Device

from core.utils import print_stat
from .base_parametrization import BaseParametrization
from .utils import HeavisideProjection

__all__ = ["LeveSetParameterization"]


class LevelSetInterp(object):
    """This class implements the level set surface using Gaussian radial basis functions."""

    def __init__(
        self,
        x0: Tensor = None,
        y0: Tensor = None,
        z0: Tensor = None,
        sigma: float = 0.02,
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
        self.device = device

        # Builds the level set interpolation model.
        gauss_kernel = self.gaussian(self.xy0, self.xy0)
        self.model = torch.matmul(torch.linalg.inv(gauss_kernel), self.z0.flatten())

        # Solve gauss_kernel @ model = z0
        # self.model = torch.linalg.solve(gauss_kernel, self.z0) # sees more stable

    def gaussian(self, xyi, xyj):
        dist_sq = (xyi[:, 1].reshape(-1, 1) - xyj[:, 1].reshape(1, -1)).square_() + (
            xyi[:, 0].reshape(-1, 1) - xyj[:, 0].reshape(1, -1)
        ).square_()
        # return torch.exp(-dist_sq / (2 * self.sig**2))
        return dist_sq.mul_(-1 / (2 * self.sig**2)).exp_()
        # return dist_sq.mul_(-1 / (2 * 0.03**2)).exp_()

    def get_ls(self, x1, y1, shape):
        xx, yy = torch.meshgrid(x1, y1, indexing="ij")
        xy1 = torch.column_stack((xx.reshape(-1), yy.reshape(-1)))
        # ls = self.gaussian(self.xy0, xy1).T @ self.model
        ls = self.gaussian(xy1, self.xy0) @ self.model
        ls = ls.reshape(shape)
        return ls ## level set surface with the same shape as z0


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
        n_phi = [
            (m.stop - m.start)
            for m in self.device.design_region_masks[self.region_name]
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
                region_s / 2 if rho_res > 0 else -region_s / 2,
                n,
                device=self.operation_device,
            )
            for region_s, n, rho_res in zip(region_size, n_phi, rho_resolution)
        ]
        param_dict = dict(n_rho=n_rho, n_phi=n_phi, rho=rho, phi=phi)

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
        if init_method == "random":
            nn.init.normal_(weight_dict["ls_knots"], mean=0, std=0.01)
            weight_dict["ls_knots"].data += 0.05
        elif init_method == "rectangle":
            weight = weight_dict["ls_knots"]
            weight.data.fill_(-0.2)
            weight.data[
                :, weight.shape[1] // 4 : 3 * weight.shape[1] // 4
            ] = 0.05
            # weight.data += torch.randn_like(weight) * 0.01
        else:
            raise ValueError(f"Unsupported initialization method: {init_method}")

    def build_permittivity(self, weights, sharpness: float):
        rho, phi = self.params["rho"], self.params["phi"]
        sigma = 1 / max(self.cfgs["rho_resolution"])
        design_param = weights["ls_knots"]
        ### to avoid all knots becoming unreasonable large to make it stable
        ### also to avoid most knots concentrating near threshold, otherwise, binarization will not work
        design_param = design_param / design_param.std() * 1/4
        phi_model = LevelSetInterp(
            x0=rho[0],
            y0=rho[1],
            z0=design_param,
            sigma=sigma,
            device=design_param.device,
        )

        n_phi = self.params["n_phi"]

        phi = phi_model.get_ls(x1=phi[0], y1=phi[1], shape=n_phi)
   
        


        phi = phi + self.eta
        eps_phi = self.binary_projection(phi, sharpness, self.eta)

        self.phi = torch.reshape(phi, n_phi)

        return eps_phi
