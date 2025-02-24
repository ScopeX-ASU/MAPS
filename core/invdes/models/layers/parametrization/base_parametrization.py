"""
Date: 2024-10-04 23:22:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-16 16:55:16
FilePath: /MAPS/core/invdes/models/layers/parametrization/base_parametrization.py
"""

from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.types import Device

from core.inv_litho.photonic_model import *
from core.utils import padding_to_tiles, rip_padding


def cvt_res(
    x,
    source_resolution: int = None,
    target_resolution: int = None,
    intplt_mode="nearest",
    target_size=None,
):
    if target_size is None:
        target_nx, target_ny = [
            int(round(i * target_resolution / source_resolution)) for i in x.shape[-2:]
        ]
        target_size = (target_nx, target_ny)
    if x.shape[-2:] == tuple(target_size):
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


def _transpose_symmetry(x, rot_k: int = 3):
    assert x.shape[0] == x.shape[1], "Only support square matrix for transpose symmetry"
    x_t = torch.transpose(x, 0, 1)
    x = torch.tril(x, -1) + torch.triu(x_t)
    x = torch.rot90(x, k=rot_k, dims=[-2, -1])

    return x


def transpose_symmetry(xs: Tuple | List, rot_k: int = 3) -> List:
    xs = [_transpose_symmetry(x, rot_k=rot_k) for x in xs]
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
            max(1, int(round(i * target_resolution / source_resolution)))
            for i in x.shape[-2:]
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
        ), (
            f"source size should be multiples of target size, got {x.shape[-2:]} and {target_size}"
        )
        x = eps_bg + (eps_r - eps_bg) * x
        # x = 1 / x
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
        # x = 1 / x
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


def _litho(x_310, res, entire_eps, dr_mask, device):
    ## hr_x is the high resolution pattern 1 nm/pixel, x is the low resolution pattern following sim_cfg resolution
    # in this case, we only consider the nominal corner of lithography
    # x_310 is the (hr_permittivity and lr_permittivity)
    # we need to calculate the 310 resolution pattern for both of them
    # TODO ensure that the input x is a (0, 1) pattern
    entire_eps[dr_mask] = x_310
    origion_shape = entire_eps.shape
    entire_eps = cvt_res(entire_eps, source_resolution=res, target_resolution=310)
    entire_eps, pady_0, pady_1, padx_0, padx_1 = padding_to_tiles(entire_eps, 620)
    # remember to set the resist_steepness to a smaller value so that the output three mask is not strictly binarized for later etching
    litho = litho_model(  # reimplement from arixv https://arxiv.org/abs/2411.07311
        target_img_shape=entire_eps.shape,
        avepool_kernel=5,
        device=device,
    )
    x_out, _, _ = litho.forward_batch(batch_size=1, target_img=entire_eps)
    x_out = rip_padding(x_out.squeeze(), pady_0, pady_1, padx_0, padx_1)
    x_out = cvt_res(x_out, target_size=origion_shape)[dr_mask]
    return x_out


def litho(xs, res, entire_eps, dr_mask, device):
    # the res of the two xs are the same
    hr_out = _litho(xs[0], res, entire_eps, dr_mask, device)
    out = _litho(xs[1], res, entire_eps, dr_mask, device)
    return [hr_out, out]


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


def _blur(x, mfs, res, entire_eps, dr_mask, dim="xy"):
    """
    Apply MFS-based blur to a 2D tensor along specified dimension(s).

    Parameters:
    - x: 2D tensor to blur.
    - mfs: Minimum feature size in physical units.
    - res: Resolution to convert mfs into pixels.
    - dim: Dimension to blur ("x", "y", or "xy").

    Returns:
    - Blurred 2D tensor.
    """
    mfs_px = (
        int(2 * mfs * res) + 1
    )  # Convert mfs to pixels and round up, 1.2 here is a margin coefficient
    if mfs_px % 2 == 0:
        mfs_px += 1  # Ensure kernel size is odd

    # Build the 1D blur kernel
    # mfs_kernel_1d = 1 - torch.abs(torch.linspace(-1, 1, steps=mfs_px, device=x.device))
    mfs_kernel_1d = torch.ones(mfs_px, device=x.device)
    mfs_kernel_1d = mfs_kernel_1d / mfs_kernel_1d.sum()  # Normalize the kernel

    entire_eps[dr_mask] = x

    if dim == "x":
        # Blur along the "x" (columns)
        entire_eps = F.conv1d(
            entire_eps.unsqueeze(1),  # Add a channel dimension for conv1d
            mfs_kernel_1d.unsqueeze(0).unsqueeze(0),  # Shape (1, 1, kernel_size)
            padding=mfs_px // 2,
        ).squeeze(1)  # Remove the channel dimension
    elif dim == "y":
        # Blur along the "y" (rows)
        entire_eps = (
            F.conv1d(
                entire_eps.t().unsqueeze(1),  # Transpose to blur rows as columns
                mfs_kernel_1d.unsqueeze(0).unsqueeze(0),  # Shape (1, 1, kernel_size)
                padding=mfs_px // 2,
            )
            .squeeze(1)
            .t()
        )  # Undo the transpose
    elif dim == "xy":
        # # Build the 2D blur kernel from the 1D kernel
        # mfs_kernel_2d = torch.outer(mfs_kernel_1d, mfs_kernel_1d)
        # # the mfs 2d kernel should be a circle like kernel instead of a square kernel
        # for i in range(mfs_px):
        #     for j in range(mfs_px):
        #         if (i - mfs_px // 2) ** 2 + (j - mfs_px // 2) ** 2 > (mfs_px // 2) ** 2:
        #             mfs_kernel_2d[i, j] = 0
        # mfs_kernel_2d = mfs_kernel_2d / mfs_kernel_2d.sum()  # Normalize the 2D kernel
        # Build a circular averaging kernel directly with PyTorch
        y, x = torch.meshgrid(
            torch.arange(mfs_px, device=x.device),
            torch.arange(mfs_px, device=x.device),
            indexing="ij",
        )
        x = x.to(entire_eps.device)
        y = y.to(entire_eps.device)
        center = mfs_px // 2
        distance = (y - center) ** 2 + (x - center) ** 2
        radius = (mfs_px // 2) ** 2

        # Generate circular kernel mask
        mfs_kernel_2d = (distance <= radius).float().to(x.device)

        # Normalize the kernel to ensure it sums to 1
        mfs_kernel_2d /= mfs_kernel_2d.sum()
        # Blur using 2D convolution
        entire_eps = (
            F.conv2d(
                entire_eps.unsqueeze(0).unsqueeze(
                    0
                ),  # Add batch and channel dimensions
                mfs_kernel_2d.unsqueeze(0).unsqueeze(
                    0
                ),  # Shape (1, 1, kernel_size, kernel_size)
                padding=mfs_px // 2,
            )
            .squeeze(0)
            .squeeze(0)
        )  # Remove batch and channel dimensions
    else:
        raise ValueError(f"Invalid dim argument: {dim}. Must be 'x', 'y', or 'xy'.")

    x = entire_eps[dr_mask]
    return x


def blur(xs, mfs, resolutions, entire_eps, dr_mask, dim="xy"):
    """
    Apply MFS-based blur to a list of 2D tensors along specified dimension(s).

    Parameters:
    - xs: List of 2D tensors to blur.
    - mfs: Minimum feature size in physical units.
    - resolutions: Resolutions to convert mfs into pixels.
    - dim: Dimension to blur ("x", "y", or "xy").

    Returns:
    - List of blurred 2D tensors.
    """
    xs = [
        _blur(x, mfs, res, entire_eps, dr_mask, dim) for x, res in zip(xs, resolutions)
    ]
    return xs


def _fft(x, mfs, res, entire_eps, dr_mask, dim="xy"):
    entire_eps[dr_mask] = x
    assert dim == "xy", "Only 2D FFT filtering is supported for now"

    # Calculate the number of frequencies to keep
    height, width = entire_eps.shape
    cutoff_y = int(height / (2 * mfs * res))
    cutoff_x = int(width / (2 * mfs * res))

    # Apply 2D FFT
    freq = torch.fft.fft2(entire_eps)

    # Create a mask to keep only the low frequencies
    mask = torch.zeros_like(freq)
    mask[:cutoff_y, :cutoff_x] = 1  # Top-left corner
    mask[:cutoff_y, -cutoff_x:] = 1  # Top-right corner
    mask[-cutoff_y:, :cutoff_x] = 1  # Bottom-left corner
    mask[-cutoff_y:, -cutoff_x:] = 1  # Bottom-right corner

    # Apply the mask to the frequency domain
    filtered_freq = freq * mask

    # Inverse FFT to get the filtered design
    filtered_spatial = torch.fft.ifft2(filtered_freq).real

    # Update the original tensor where dr_mask is True
    return filtered_spatial[dr_mask]


def fft(xs, mfs, resolutions, entire_eps, dr_mask, dim="xy"):
    """
    apply fft to filter out the high frequency components for minimum feature size control
    """
    xs = [
        _fft(x, mfs, res, entire_eps, dr_mask, dim) for x, res in zip(xs, resolutions)
    ]
    return xs


permittivity_transform_collections = dict(
    mirror_symmetry=mirror_symmetry,
    transpose_symmetry=transpose_symmetry,
    convert_resolution=convert_resolution,
    litho=litho,
    etching=etching,
    blur=blur,
    fft=fft,
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
            rho_resolution=[50, 0],  #  50 knots per um, 0 means reduced dimension
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

    def permittivity_transform(
        self, hr_permittivity, permittivity, cfgs, sharpness, hr_entire_eps, hr_dr_mask
    ):
        transform_cfg_list = cfgs["transform"]
        # print(permittivity)
        # print("this is the transform cfg list", transform_cfg_list, flush=True)
        # plt.figure()
        # plt.imshow(1 - np.rot90(hr_permittivity.cpu().numpy()), cmap="gray")
        # plt.savefig(f"./figs/origion_hr.png")
        # plt.close()
        for transform_cfg in transform_cfg_list:
            transform_type = transform_cfg["type"]
            if transform_type == "binarize":
                hr_permittivity = self.binary_projection(
                    hr_permittivity, sharpness, self.eta
                )
                permittivity = self.binary_projection(permittivity, sharpness, self.eta)
                # plt.figure()
                # plt.imshow(1 - np.rot90(hr_permittivity.cpu().numpy()), cmap="gray")
                # plt.savefig(f"./figs/binarize_hr.png")
                # plt.close()
                continue
            cfg = deepcopy(transform_cfg)
            del cfg["type"]
            if "device" in cfg.keys():
                assert cfg["device"] == "cuda", "running on cpu is not supported"
                cfg["device"] = self.operation_device
            if "binary_proj_layer" in cfg.keys():
                cfg["binary_projection"] = self.binary_projection
            if "litho" in transform_type:
                cfg["device"] = self.operation_device
                # hr_res, res should be contained in the cfgs
                cfg["entire_eps"] = (hr_entire_eps - hr_entire_eps.min()) / (
                    hr_entire_eps.max() - hr_entire_eps.min()
                )  # normalize the eps
                cfg["dr_mask"] = hr_dr_mask
            if "blur" in transform_type or "fft" in transform_type:
                cfg["entire_eps"] = (hr_entire_eps - hr_entire_eps.min()) / (
                    hr_entire_eps.max() - hr_entire_eps.min()
                )  # normalize the eps
                cfg["dr_mask"] = hr_dr_mask
            hr_permittivity, permittivity = permittivity_transform_collections[
                transform_type
            ]((hr_permittivity, permittivity), **cfg)
            # plt.figure()
            # plt.imshow(1 - np.rot90(hr_permittivity.cpu().numpy()), cmap="gray")
            # plt.savefig(f"./figs/{transform_type}_hr.png")
            # plt.close()
        # plt.figure()
        # plt.imshow(1 - np.rot90(hr_permittivity.cpu().numpy()), cmap="gray")
        # plt.savefig(f"./figs/eps_final.png")
        # plt.close()
        # quit()
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
        # plt.figure()
        # plt.imshow(1 - np.rot90(hr_permittivity.cpu().numpy()), cmap="gray")
        # plt.savefig(f"./figs/smoothing_lr.png")
        # plt.close()
        # print(permittivity)
        return hr_permittivity, permittivity

    def denormalize_permittivity(self, permittivity):
        eps_r = self.design_region_cfg["eps"]
        eps_bg = self.design_region_cfg["eps_bg"]

        permittivity = permittivity * (eps_r - eps_bg) + eps_bg
        return permittivity

    def forward(
        self,
        sharpness: float,
        hr_entire_eps: torch.Tensor,
        hr_dr_mask: torch.Tensor,
        ls_knots=None,
    ):
        ## first build the normalized device permittivity using weights
        ## the built one is the high resolution permittivity for evaluation
        permittivity = self.build_permittivity(self.weights, sharpness, ls_knots)

        ## this is the cloned and detached permittivity for gds dumpout
        hr_permittivity = permittivity.detach().clone()

        # I swap the order of the denormalize and transform, it should be fine

        ### then transform the permittivity for all regions using transform settings
        ## e.g., mirror symmetry, transpose symmetry, convert resolution, ...

        ## after this, permittivity will be downsampled to match the sim_cfg resolution, e.g., res=50 or 100
        ## hr_permittivity will maintain the high resolution
        hr_permittivity, permittivity = self.permittivity_transform(
            hr_permittivity,
            permittivity,
            self.cfgs,
            sharpness,
            hr_entire_eps,
            hr_dr_mask,
        )

        ## we need to denormalize the permittivity to the real permittivity values
        ## for the simulation
        permittivity = self.denormalize_permittivity(permittivity)
        hr_permittivity = self.denormalize_permittivity(hr_permittivity)

        return hr_permittivity, permittivity
