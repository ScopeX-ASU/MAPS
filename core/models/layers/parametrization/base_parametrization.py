"""
Date: 2024-10-04 23:22:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-04 23:35:34
FilePath: /Metasurface-Opt/core/models/base_parametrization.py
"""

from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.types import Device

from core.NVILT_Share.photonic_model import *
from core.utils import padding_to_tiles, rip_padding


def _mirror_symmetry(x, dims):
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


def mirror_symmetry(xs: Tuple | List, dims):
    xs = [_mirror_symmetry(x, dims) for x in xs]
    return xs


def _transpose_symmetry(x, flag: bool = True):
    assert x.shape[0] == x.shape[1], "Only support square matrix for transpose symmetry"
    if flag:
        x_t = torch.transpose(x, 0, 1)
        x = torch.tril(x, -1) + torch.triu(x_t)

    return x


def transpose_symmetry(xs: Tuple | List, flag: bool = True) -> List:
    xs = [_transpose_symmetry(x, flag=flag) for x in xs]
    return xs


def _convert_resolution(
    x,
    source_resolution: int = None,
    target_resolution: int = None,
    intplt_mode="nearest",
    subpixel_smoothing: bool = False,
    eps_r: float = None,
    eps_bg: float = None,
    target_size=None,
):
    if target_size is None:
        target_nx, target_ny = [
            int(round(i * target_resolution / source_resolution)) for i in x.shape[-2:]
        ]
        target_size = (target_nx, target_ny)
    if x.shape[-2:] == tuple(target_size):
        return x

    if (
        target_size[0] < x.shape[-2]
        and target_size[1] < x.shape[-1]
        and subpixel_smoothing
    ):
        assert (
            x.shape[-2] % target_size[0] == 0 and x.shape[-1] % target_size[1] == 0
        ), f"source size should be multiples of target size, got {x.shape[-2:]} and {target_size}"
        x = eps_bg + (eps_r - eps_bg) * x
        x = 1 / x
        # avg_pool_stride = [int(round(s / r)) for s, r in zip(x.shape[-2:], target_size)]
        # avg_pool_kernel_size = [s + 1 for s in avg_pool_stride]
        # pad_size = []
        # x = F.pad(
        #     x, (pad_size[1], pad_size[1], pad_size[0], pad_size[0]), mode="constant"
        # )
        # print(x.shape, avg_pool_kernel_size, avg_pool_stride)
        x = F.adaptive_avg_pool2d(
            x[None, None],
            output_size=target_size,
        )[0, 0]
        x = 1 / x
        x = (x - eps_bg) / (eps_r - eps_bg)
        return x

    if len(x.shape) == 2:
        x = (
            F.interpolate(
                x.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode=intplt_mode,
            )
            .squeeze(0)
            .squeeze(0)
        )
    elif len(x.shape) == 3:
        x = F.interpolate(x.unsqueeze(0), size=target_size, mode=intplt_mode).squeeze(0)

    return x


def convert_resolution(
    xs: Tuple | List,
    source_resolution: int = None,
    target_resolution: int = None,
    intplt_mode="nearest",
    subpixel_smoothing: bool = False,
    eps_r: float = None,
    eps_bg: float = None,
    target_size=None,
):
    x = _convert_resolution(
        xs[1],
        source_resolution=source_resolution,
        target_resolution=target_resolution,
        intplt_mode=intplt_mode,
        subpixel_smoothing=subpixel_smoothing,
        eps_r=eps_r,
        eps_bg=eps_bg,
        target_size=target_size,
    )
    return list(xs[:-1]) + [x]


def _litho(x_310, mask_steepness, resist_steepness, device):
    ## hr_x is the high resolution pattern 1 nm/pixel, x is the low resolution pattern following sim_cfg resolution
    # in this case, we only consider the nominal corner of lithography
    # TODO ensure that the input x is a (0, 1) pattern
    ### make sure input x is the correct resolution=310!
    entire_eps, pady_0, pady_1, padx_0, padx_1 = padding_to_tiles(x_310, 620)
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
    x_out, _, _ = nvilt.forward_batch(batch_size=1, target_img=entire_eps)

    x_out = rip_padding(x_out.squeeze(), pady_0, pady_1, padx_0, padx_1)

    ### x_outis also resolution=310

    return x_out


def litho(xs, mask_steepness, resist_steepness, device):
    outs = [xs[0]]
    outs += [_litho(x, mask_steepness, resist_steepness, device) for x in xs[1:]]
    return outs


def _etching(x, sharpness, eta, binary_projection):
    # in this case, we only consider the nominal corner for etching
    sharpness = torch.tensor(
        [
            sharpness,
        ],
        device=x.device,
    )
    eta = torch.tensor(
        [
            eta,
        ],
        device=x.device,
    )
    x = binary_projection(x, sharpness, eta)

    return x


def etching(xs, sharpness, eta, binary_projection):
    outs = [xs[0]]
    outs += [_etching(x, sharpness, eta, binary_projection) for x in xs[1:]]
    return outs

def _blur(x, mfs, res):
    # in this case, we only consider the nominal corner for etching
    mfs_px = int(mfs * res) + 1 # ceiling the mfs to the nearest pixel
    if mfs_px % 2 == 0:
        mfs_px += 1 # ensure mfs is odd
    # build the kernel
    mfs_kernel = 1 - torch.abs(torch.linspace(-1, 1, steps=mfs_px, device=x.device))
    # x is a 2D tensor
    # convolve the kernel with the x along the second dimension
    x = F.conv1d(x.unsqueeze(1), mfs_kernel.unsqueeze(0).unsqueeze(0), padding=mfs_px//2).squeeze(1)
    return x

def blur(xs, mfs, resolutions):
    xs = [_blur(x, mfs, res) for x, res in zip(xs, resolutions)]
    return xs


permittivity_transform_collections = dict(
    mirror_symmetry=mirror_symmetry,
    transpose_symmetry=transpose_symmetry,
    convert_resolution=convert_resolution,
    litho=litho,
    etching=etching,
    blur=blur,
)


class BaseParametrization(nn.Module):
    def __init__(
        self,
        device,  # BaseDevice
        hr_device,  # BaseDevice
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
        self.hr_device = hr_device
        self.design_region_mask = device.design_region_masks[region_name]
        self.design_region_cfg = device.design_region_cfgs[region_name]

        self.hr_design_region_mask = hr_device.design_region_masks[region_name]
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
        ### return: permittivity that you would like to dumpout as final solution, typically should be high resolution
        raise NotImplementedError

    def permittivity_transform(self, hr_permittivity, permittivity, cfgs):
        transform_cfg_list = cfgs["transform"]

        for transform_cfg in transform_cfg_list:
            transform_type = transform_cfg["type"]
            cfg = deepcopy(transform_cfg)
            del cfg["type"]
            if "device" in cfg.keys():
                assert cfg["device"] == "cuda", "running on cpu is not supported"
                cfg["device"] = self.operation_device
            if "binary_proj_layer" in cfg.keys():
                cfg["binary_projection"] = self.binary_projection
            hr_permittivity, permittivity = permittivity_transform_collections[
                transform_type
            ]((hr_permittivity, permittivity), **cfg)

        ### we have to match the design region size to be able to be placed in the design region with subpixel smoothing
        target_size = [(m.stop - m.start) for m in self.design_region_mask]

        ## first we upsample to ~1nm resolution with nearest interpolation to maintain the geometry

        src_res = self.hr_device.sim_cfg["resolution"]  # e.g., 310
        tar_res = int(round(1000 / src_res)) * src_res
        ## it also needs to be multiples of the sim resolution to enable subpixel smoothing
        hr_size = [int(round(i * tar_res / src_res)) for i in permittivity.shape[-2:]]
        hr_size = [int(round(i / j) * j) for i, j in zip(hr_size, target_size)]
        # print(permittivity.shape)
        permittivity = _convert_resolution(
            permittivity,
            intplt_mode="nearest",
            target_size=hr_size,
        )
        # print(permittivity.shape)
        # then we convert the resolution to the sim_cfg resolution with subpixeling smoothing, if we use res=50, 100, then we can use pooling
        permittivity = _convert_resolution(
            permittivity,
            subpixel_smoothing=True,
            eps_r=self.design_region_cfg["eps"],
            eps_bg=self.design_region_cfg["eps_bg"],
            target_size=target_size,
        )
        # print(permittivity.shape)

        with torch.inference_mode():
            target_size = [(m.stop - m.start) for m in self.hr_design_region_mask]

            hr_permittivity = _convert_resolution(
                hr_permittivity,
                intplt_mode="nearest",
                target_size=target_size,
            )

        return hr_permittivity, permittivity

    def denormalize_permittivity(self, permittivity):
        eps_r = self.design_region_cfg["eps"]
        eps_bg = self.design_region_cfg["eps_bg"]

        permittivity = permittivity * (eps_r - eps_bg) + eps_bg
        return permittivity

    def forward(self, sharpness: float):
        ## first build the normalized device permittivity using weights
        ## the built one is the high resolution permittivity for evaluation
        permittivity = self.build_permittivity(self.weights, sharpness)

        ## this is the cloned and detached permittivity for gds dumpout
        hr_permittivity = permittivity.detach().clone()

        # I swap the order of the denormalize and transform, it should be fine

        ### then transform the permittivity for all regions using transform settings
        ## e.g., mirror symmetry, transpose symmetry, convert resolution, ...

        ## after this, permittivity will be downsampled to match the sim_cfg resolution, e.g., res=50 or 100
        ## hr_permittivity will maintain the high resolution
        hr_permittivity, permittivity = self.permittivity_transform(
            hr_permittivity, permittivity, self.cfgs
        )

        ## we need to denormalize the permittivity to the real permittivity values
        ## for the simulation
        permittivity = self.denormalize_permittivity(permittivity)
        hr_permittivity = self.denormalize_permittivity(hr_permittivity)

        return hr_permittivity, permittivity
