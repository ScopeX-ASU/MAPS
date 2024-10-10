"""
Date: 2024-10-04 23:22:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-04 23:35:34
FilePath: /Metasurface-Opt/core/models/base_parametrization.py
"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch.types import Device
from core.utils import padding_to_tiles, rip_padding
from core.NVILT_Share.photonic_model import *


def mirror_symmetry(x, dims):
    for dim in dims:
        y1, y2 = x.chunk(2, dim=dim)
        if x.shape[dim] % 2 != 0:
            if dim == 0:
                x = torch.cat([y1, y1[:-1].flip(dims=[dim])], dim=dim)
            elif dim == 1:
                x = torch.cat([y1, y1[:, :-1].flip(dims=[dim])], dim=dim)
        else:
            x = torch.cat([y1, y1.flip(dims=[dim])], dim=dim)
    return x


def transpose_symmetry(x, flag):
    assert x.shape[0] == x.shape[1], "Only support square matrix for transpose symmetry"
    if flag:
        x_t = torch.transpose(x, 0, 1)
        x = torch.tril(x, -1) + torch.triu(x_t)

    return x


def convert_resolution(
    x, source_resolution: int, target_resolution: int, intplt_mode="nearest"
):
    target_nx, target_ny = [
        int(round(i * target_resolution / source_resolution)) for i in x.shape
    ]

    if len(x.shape) == 2:
        x = (
            F.interpolate(
                x.unsqueeze(0).unsqueeze(0),
                size=(target_nx, target_ny),
                mode=intplt_mode,
            )
            .squeeze(0)
            .squeeze(0)
        )
    elif len(x.shape) == 3:
        x = F.interpolate(
            x.unsqueeze(0), size=(target_nx, target_ny), mode=intplt_mode
        ).squeeze(0)
    return x

def litho(x, mask_steepness, resist_steepness, device):
    # in this case, we only consider the nominal corner of lithography
    # TODO ensure that the input x is a (0, 1) pattern
    entire_eps, pady_0, pady_1, padx_0, padx_1 = padding_to_tiles(
        x, 620
    )
    # remember to set the resist_steepness to a smaller value so that the output three mask is not strictly binarized for later etching
    nvilt = My_nvilt2(
        target_img_shape=entire_eps.shape,
        mask_steepness=mask_steepness,
        resist_steepness=resist_steepness,
        avepool_kernel=5,
        morph=0,
        scale_factor=1,
        device=device,
    )
    x_out, _, _ = nvilt.forward_batch(
        batch_size=1, target_img=entire_eps
    )

    x_out_norm = rip_padding(x_out.squeeze(), pady_0, pady_1, padx_0, padx_1)

    return x_out_norm

def etching(x, sharpness, eta, binary_projection):
    # in this case, we only consider the nominal corner for etching
    sharpness = torch.tensor([sharpness,], device=x.device)
    eta = torch.tensor([eta,], device=x.device)
    x = binary_projection(x, sharpness, eta)

    return x

permittivity_transform_collections = dict(
    mirror_symmetry=mirror_symmetry,
    transpose_symmetry=transpose_symmetry,
    convert_resolution=convert_resolution,
    litho=litho,
    etching=etching,
)


class BaseParametrization(nn.Module):
    def __init__(
        self,
        device,  # BaseDevice
        sim_cfg: dict,
        region_name: str = "design_region_1",
        cfgs: dict = dict(
            method="levelset",
            rho_resolution=[50, 0],  #  50 knots per um, 0 means reduced dimention
            transform=dict(),
            init_method="random",
        ),
        operation_device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.region_name = region_name
        self.sim_cfg = sim_cfg
        self.cfgs = cfgs
        self.device = device
        self.design_region_mask = device.design_region_masks[region_name]
        self.design_region_cfg = device.design_region_cfgs[region_name]
        self.operation_device = operation_device
        self._parameter_build_per_region_fns = {}
        self._parameter_reset_per_region_fns = {}
        # self.build_parameters(cfgs, self.design_region_cfg)
        # self.reset_parameters(cfgs, self.design_region_cfg)

    def register_parameter_build_per_region_fn(self, method, fn):
        self._parameter_build_per_region_fns[method] = fn

    def register_parameter_reset_per_region_fn(self, method, fn):
        self._parameter_reset_per_region_fns[method] = fn

    def build_parameters(self, cfgs, design_region_cfg, *args, **kwargs):
        method = cfgs["method"]
        _build_fn = self._parameter_build_per_region_fns.get(method, None)
        if _build_fn is not None:
            weight_dict, param_dict = _build_fn(
                cfgs, design_region_cfg, *args, **kwargs
            )
        else:
            raise ValueError(f"Unsupported parameterization build method: {method}")

        self.weights = nn.ParameterDict(weight_dict)

        self.params = param_dict

    def reset_parameters(self, cfgs, design_region_cfgs, *args, **kwargs):
        method = cfgs["method"]
        init_method = cfgs["init_method"]

        _reset_fn = self._parameter_reset_per_region_fns.get(method, None)
        if _reset_fn is not None:
            _reset_fn(self.weights, cfgs, design_region_cfgs, init_method)
        else:
            raise ValueError(f"Unsupported parameterization reset method: {method}")

    def build_permittivity(self, weights, sharpness: float):
        ### return: permittivity
        raise NotImplementedError

    def permittivity_transform(self, permittivity, cfgs):
        transform_cfg_list = cfgs["transform"]

        for transform_cfg in transform_cfg_list:
            transform_type = transform_cfg["type"]
            cfg = deepcopy(transform_cfg)
            del cfg["type"]
            if "device" in cfg.keys():
                assert cfg["device"] == 'cuda', "running on cpu is not supported"
                cfg["device"] = self.operation_device
            if "binary_proj_layer" in cfg.keys():
                cfg["binary_projection"] = self.binary_projection
            permittivity = permittivity_transform_collections[transform_type](
                permittivity, **cfg
            )

        return permittivity

    def denormalize_permittivity(self, permittivity):
        eps_r = self.design_region_cfg["eps"]
        eps_bg = self.design_region_cfg["eps_bg"]

        permittivity = permittivity * (eps_r - eps_bg) + eps_bg
        return permittivity

    def forward(self, sharpness: float):
        ## first build the normalized device permittivity using weights
        permittivity = self.build_permittivity(self.weights, sharpness)

        # I swap the order of the denormalize and transform, it should be fine

        ### then transform the permittivity for all regions using transform settings
        ## e.g., mirror symmetry, transpose symmetry, convert resolution, ...
        permittivity = self.permittivity_transform(permittivity, self.cfgs)

        ## we need to denormalize the permittivity to the real permittivity values
        ## for the simulation
        permittivity = self.denormalize_permittivity(permittivity)

        return permittivity
