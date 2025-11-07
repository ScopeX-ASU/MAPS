"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2025-10-31 14:34:17
LastEditors: Jiaqi Gu (jiaqigu@asu.edu)
LastEditTime: 2025-11-02 16:52:26
FilePath: /MAPS_ilt/core/invdes/models/layers/parametrization/geometry.py
"""

import logging
from typing import Any, Dict
from torch.types import Device
from torch import nn, Tensor
import torch
import torch.nn.functional as F
import numpy as np


def gradImage(image):
    GRAD_STEPSIZE = 1.0
    image = image.view([-1, 1, image.shape[-2], image.shape[-1]])
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")[:, 0].detach()
    gradX = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * GRAD_STEPSIZE)
    gradY = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * GRAD_STEPSIZE)
    return gradX.view(image.shape), gradY.view(image.shape)


def get_grid(shape, dl):
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
    eps_r = torch.full(xs.shape, eps_bg, device=device)

    # Convert region functions to a vectorized form using PyTorch operations
    for e, reg in zip(eps_r_list, reg_list):
        # Assume that reg is a lambda or function that can be applied to tensors
        material_mask = reg(xs, ys)  # This should return a boolean tensor
        eps_r[material_mask] = e

    return eps_r  # Move the result back to CPU and convert to numpy array


class BatchGeometry(nn.Module):
    def __init__(
        self,
        sim_cfg: Dict[str, Any],
        geometry_cfgs: Dict[str, Any],
        region_name: str = "design_region_1",
        design_region_mask=None,
        design_region_cfg=None,
        operation_device: Device = torch.device("cuda:0"),
    ):
        super().__init__()
        self.sim_cfg = sim_cfg
        self.geometry_cfgs = geometry_cfgs
        batch_dims = self.geometry_cfgs[
            "batch_dims"
        ]  # can be a number of multiple batch dimensions
        if isinstance(batch_dims, int):
            batch_dims = (batch_dims,)
        size_dim = self.geometry_cfgs.get(
            "size_dim", 2
        )  # number of parameters to determine the size, default 2, e.g., 2 for box, 1 for circle
        assert isinstance(size_dim, int) and size_dim >= 1, (
            "size_dim should be a positive integer."
        )
        self.centers = nn.Parameter(
            torch.zeros(*batch_dims, 2)
        )  # (num_geometries, 2) # x, y coordinates, unit in um
        self.sizes = nn.Parameter(
            torch.ones(*batch_dims, size_dim)
        )  # (num_geometries, size_dim), unit in um (for height/width/radius)

        self.region_name = region_name
        self.design_region_mask = design_region_mask  # (H, W), bool tensor
        self.design_region_cfg = design_region_cfg
        self.operation_device = operation_device

    def compute_derivatives(self, grad_eps: Tensor):
        # grad_eps: (B, 1, H, W)
        # Placeholder implementation, actual implementation depends on geometry type and simulation details
        grad_centers = torch.zeros_like(self.centers)
        grad_sizes = torch.zeros_like(self.sizes)
        return grad_centers, grad_sizes

    def discretize_as_permittivity(
        self, grid_x: Tensor, grid_y: Tensor, permittivity: Tensor = None
    ) -> Tensor:
        # grid_x, grid_y: (H, W)
        # given the interested grid points, we discretize and draw this continuous geometries on to this grid/mesh
        eps = torch.zeros_like(grid_x) if permittivity is None else permittivity
        return eps