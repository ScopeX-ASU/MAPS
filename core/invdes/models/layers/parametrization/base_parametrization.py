"""
Date: 2024-10-04 23:22:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-23 10:33:07
FilePath: /MAPS/core/invdes/models/layers/parametrization/base_parametrization.py
"""

import warnings
from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.types import Device

from core.inv_litho.photonic_model import *
from core.utils import padding_to_tiles, rip_padding

from .extrude import extrude


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


class DecayGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, decay_factor):
        ctx.save_for_backward(input)
        ctx.decay_factor = decay_factor
        return input

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        decay_factor = ctx.decay_factor
        grad_input = grad_output * decay_factor
        return grad_input, None


def _mirror_symmetry(x, dims):
    ## the flipped part should decay gradient by half.
    ## otherwise the shared parameter will double accumulate the gradients.
    for dim in dims:
        y1, y2 = x.chunk(2, dim=dim)
        if x.shape[dim] % 2 != 0:
            if dim == 0:
                y1_mirror = DecayGradient.apply(y1[:-1], 0.5)
                x = torch.cat([y1_mirror, y1[-1:], y1_mirror.flip(dims=[dim])], dim=dim)
            elif dim == 1:
                y1_mirror = DecayGradient.apply(y1[:, :-1], 0.5)
                x = torch.cat(
                    [y1_mirror, y1[:, -1:], y1_mirror.flip(dims=[dim])], dim=dim
                )
        else:
            x = torch.cat(
                [
                    DecayGradient.apply(y1, 0.5),
                    DecayGradient.apply(y1.flip(dims=[dim]), 0.5),
                ],
                dim=dim,
            )
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


def transpose_symmetry(xs: Tuple | List, rot_k: int = 2) -> List:
    xs = [_transpose_symmetry(x, rot_k=rot_k) for x in xs]
    return xs


def _rotation_symmetry(x, rot_k: int = 2):
    ## do not use rotation and average, it will get gray pixel
    assert x.shape[0] == x.shape[1], "Only support square matrix for rotation symmetry"
    n = x.shape[-1]
    half = n // 2
    odd = n % 2 == 1

    if rot_k == 1:
        # 90-degree rotational symmetry: x == rot90(x, 1)
        q = half
        a = x[..., :q, :q]

        if odd:
            b = x[..., :q, q : q + 1]  # top half of center column
            d = x[..., q : q + 1, q : q + 1]  # center pixel

            top = torch.cat(
                [
                    a,
                    b,
                    torch.rot90(a, k=3, dims=[-2, -1]),
                ],
                dim=-1,
            )

            mid = torch.cat(
                [
                    b.transpose(-2, -1),
                    d,
                    b.flip(-2).transpose(-2, -1),
                ],
                dim=-1,
            )

            bot = torch.cat(
                [
                    torch.rot90(a, k=1, dims=[-2, -1]),
                    b.flip(-2),
                    torch.rot90(a, k=2, dims=[-2, -1]),
                ],
                dim=-1,
            )

            x = torch.cat([top, mid, bot], dim=-2)

        else:
            x = torch.cat(
                [
                    torch.cat(
                        [a, torch.rot90(a, k=3, dims=[-2, -1])],
                        dim=-1,
                    ),
                    torch.cat(
                        [
                            torch.rot90(a, k=1, dims=[-2, -1]),
                            torch.rot90(a, k=2, dims=[-2, -1]),
                        ],
                        dim=-1,
                    ),
                ],
                dim=-2,
            )
    elif rot_k == 2:
        ## 180-deg rotation symmetry: upper half is free; odd sizes keep a symmetrized middle row.
        ## i.e., x == torch.rot90(x, 2)
        top = x[:half, :]
        bottom = top.flip(dims=[-2, -1])

        if odd:
            mid = x[half : half + 1, :]
            mid_left = mid[:, :half]
            mid_center = mid[:, half : half + 1]
            mid = torch.cat(
                [mid_left, mid_center, torch.flip(mid_left, dims=[-1])], dim=-1
            )
            x = torch.cat([top, mid, bottom], dim=-2)
        else:
            x = torch.cat([top, bottom], dim=-2)
    else:
        raise ValueError("Only support 90-deg and 180-deg rotation symmetry")

    return x


def rotation_symmetry(xs: Tuple | List, rot_k: int = 2) -> List:
    xs = [_rotation_symmetry(x, rot_k=rot_k) for x in xs]
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
        # assert (
        #     x.shape[-2] % target_size[0] == 0 and x.shape[-1] % target_size[1] == 0
        # ), (
        #     f"source size should be multiples of target size, got {x.shape[-2:]} and {target_size}"
        # )
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
        # x = F.interpolate(
        #     x.unsqueeze(0).unsqueeze(0),
        #     size=target_size,
        #     mode="area",
        # )[0, 0]
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
    # x_310 is the high/low-resolution density representation
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
        ).squeeze(
            1
        )  # Remove the channel dimension
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
            torch.arange(mfs_px, device=entire_eps.device),
            torch.arange(mfs_px, device=entire_eps.device),
            indexing="ij",
        )
        center = mfs_px // 2
        distance = (y - center) ** 2 + (x - center) ** 2
        radius = (mfs_px // 2) ** 2

        # Generate circular kernel mask
        mfs_kernel_2d = (distance <= radius).float()

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
    rotation_symmetry=rotation_symmetry,
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
            denorm_mode="linear_eps",  # linear_eps, inverse_eps, linear_index
            dims=(0, 1),  # the levelset2d describes x-y plane
            extrude_direction="-",
            extrude_angle=90.0,
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
        self.design_region_axis_weights = getattr(
            device, "design_region_axis_weights", {}
        ).get(region_name)
        self.design_region_cfg = device.design_region_cfgs[region_name]

        self.hr_design_region_mask = hr_device.design_region_masks[region_name]
        self.hr_design_region_axis_weights = getattr(
            hr_device, "design_region_axis_weights", {}
        ).get(region_name)
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

        if all(isinstance(p, nn.Parameter) for p in weight_dict.values()):
            self.weights = nn.ParameterDict(weight_dict)
        elif all(isinstance(p, nn.Module) for p in weight_dict.values()):
            self.weights = nn.ModuleDict(weight_dict)
        else:
            raise ValueError(
                "The weight_dict should contain all nn.Parameter or all nn.Module"
            )

        self.params = param_dict

    def reset_parameters(self, cfgs, design_region_cfgs, *args, **kwargs):
        method = cfgs["method"]
        init_method = cfgs["init_method"]

        _reset_fn = self._parameter_reset_per_region_fns.get(method, None)
        if _reset_fn is not None:
            _reset_fn(self.weights, cfgs, design_region_cfgs, init_method)
        else:
            raise ValueError(f"Unsupported parameterization reset method: {method}")

    def build_density(self, weights, sharpness: float, **kwargs):
        ### return: normalized design density in [0, 1], typically at high resolution
        raise NotImplementedError

    def build_permittivity(self, weights, sharpness: float, **kwargs):
        return self.build_density(weights, sharpness, **kwargs)

    def _base_region_tensor(self, eps_map, region_mask, like):
        base_region = eps_map[region_mask]
        if not isinstance(base_region, torch.Tensor):
            base_region = torch.as_tensor(base_region)
        if torch.is_complex(base_region) and not torch.is_complex(like):
            base_region = base_region.real
        return base_region.to(dtype=like.dtype, device=like.device)

    def density_transform(self, hr_density, cfgs, sharpness, hr_entire_eps, hr_dr_mask):
        def _real_transform_reference_map(reference_map: torch.Tensor) -> torch.Tensor:
            normalized = self.normalize_permittivity(reference_map)
            if torch.is_complex(normalized):
                normalized = normalized.real
            return normalized.to(dtype=torch.float32)

        # hr_density: high-resolution 2D normalized density with requires_grad=True
        ## input density is always a 2D design density in the design region
        ## for 3D case, we need to apply extra extrude operation at the end
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
                hr_density = self.binary_projection(
                    hr_density,
                    sharpness,
                    self.eta,
                    resolution=self.hr_device.resolution,
                )
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
                cfg["entire_eps"] = _real_transform_reference_map(hr_entire_eps)
                cfg["dr_mask"] = hr_dr_mask
            if "blur" in transform_type or "fft" in transform_type:
                if len(hr_dr_mask) == 3:
                    ## for 3D, we use the middle slice as representation for blurring
                    dims = sorted(self.cfgs.get("dims", (0, 1)))
                    extrude_dim = list(set(range(3)) - set(dims))[0]

                    hr_entire_eps_2d = hr_entire_eps.select(
                        extrude_dim,
                        int(
                            hr_dr_mask[extrude_dim].start + hr_dr_mask[extrude_dim].stop
                        )
                        // 2,
                    )
                    hr_dr_mask_2d = tuple([hr_dr_mask[d] for d in dims])
                else:
                    hr_entire_eps_2d = hr_entire_eps
                    hr_dr_mask_2d = hr_dr_mask
                hr_entire_eps_2d = _real_transform_reference_map(hr_entire_eps_2d)

                cfg["entire_eps"] = hr_entire_eps_2d  # normalize the eps
                cfg["dr_mask"] = hr_dr_mask_2d
                # print(hr_entire_eps_2d.shape, hr_dr_mask_2d)

            ## apply differentiable transformation to the high-resolution density
            (hr_density,) = permittivity_transform_collections[transform_type](
                (hr_density,), **cfg
            )
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

        if 1:
            ### now we have two tasks:
            dims = sorted(self.cfgs.get("dims", (0, 1)))
            design_region_mask = [self.design_region_mask[d] for d in dims]

            target_size = [(m.stop - m.start) for m in design_region_mask]

            ## task 1: fit the hr_density to hr_design_region as they might not match
            ## this does not requires any grad
            with torch.inference_mode():
                hr_design_region_mask = [self.hr_design_region_mask[d] for d in dims]
                hr_target_size = [(m.stop - m.start) for m in hr_design_region_mask]

                hr_density_nograd = _convert_resolution(
                    hr_density.detach(),
                    intplt_mode="nearest",
                    target_size=hr_target_size,
                )

            ## task 2: convert the hr_density to the sim_cfg resolution for simulation
            lr_density_grad = _convert_resolution(
                hr_density,
                subpixel_smoothing=True,
                eps_r=self.design_region_cfg["eps"],
                eps_bg=self.design_region_cfg["eps_bg"],
                target_size=target_size,
            )

            return hr_density_nograd, lr_density_grad

        else:
            #### deprecated in 2026/05/06: the upsampling and downsampling causes gradient issue ##########{{
            # ## first we upsample to ~1nm resolution with nearest interpolation to maintain the geometry
            dims = sorted(self.cfgs.get("dims", (0, 1)))
            design_region_mask = [self.design_region_mask[d] for d in dims]

            target_size = [(m.stop - m.start) for m in design_region_mask]
            src_res = self.hr_device.sim_cfg["resolution"]  # e.g., 310
            tar_res = int(round(1000 / src_res)) * src_res
            # ## it also needs to be multiples of the sim resolution to enable subpixel smoothing

            hr_size = [int(round(i * tar_res / src_res)) for i in hr_density.shape[-2:]]

            hr_size = [int(round(i / j) * j) for i, j in zip(hr_size, target_size)]

            # print(permittivity.shape)
            hr_density_nograd = hr_density.detach()
            hr_density = _convert_resolution(
                hr_density,
                intplt_mode="nearest",
                target_size=hr_size,
            )
            # print(permittivity.shape)
            # then we convert the resolution to the sim_cfg resolution with subpixeling smoothing, if we use res=50, 100, then we can use pooling
            lr_density_grad = _convert_resolution(
                hr_density,
                subpixel_smoothing=True,
                eps_r=self.design_region_cfg["eps"],
                eps_bg=self.design_region_cfg["eps_bg"],
                target_size=target_size,
            )

            # print(permittivity.shape)

            with torch.inference_mode():
                hr_design_region_mask = [self.hr_design_region_mask[d] for d in dims]
                target_size = [(m.stop - m.start) for m in hr_design_region_mask]

                hr_density_nograd = _convert_resolution(
                    hr_density_nograd,
                    intplt_mode="nearest",
                    target_size=target_size,
                )
            ### deprecated in 2026/05/06: the upsampling and downsampling causes gradient issue ##########}}

            # plt.figure()
            # plt.imshow(1 - np.rot90(hr_permittivity.cpu().numpy()), cmap="gray")
            # plt.savefig(f"./figs/smoothing_lr.png")
            # plt.close()
            # print(permittivity)
            # those are 2D permittivity in the desion region in "dims" plane

            return hr_density_nograd, lr_density_grad

    def permittivity_transform(
        self, hr_permittivity, cfgs, sharpness, hr_entire_eps, hr_dr_mask
    ):
        return self.density_transform(
            hr_permittivity, cfgs, sharpness, hr_entire_eps, hr_dr_mask
        )

    def normalize_permittivity(self, permittivity, mode: str | None = None):
        ## this is the inverse process of denormalization, mainly for visualization purpose
        eps_r = self.design_region_cfg["eps"]
        eps_bg = self.design_region_cfg["eps_bg"]
        mode = mode or self.cfgs.get("denorm_mode", "linear_eps")

        alg, exp = mode.split("_")

        if exp == "eps":
            exp = 1
        elif exp == "index":
            exp = 0.5
        else:
            exp = float(exp)

        if alg == "linear":
            pass
        elif alg == "inverse":
            exp = -exp
        else:
            raise ValueError(f"Unsupported permittivity normalization mode: {mode}")

        has_complex_endpoint = torch.is_complex(
            torch.as_tensor(eps_r)
        ) or torch.is_complex(torch.as_tensor(eps_bg))
        if has_complex_endpoint and mode != "linear_eps":
            raise ValueError(
                "Complex eps_r/eps_bg endpoints are only supported with denorm_mode='linear_eps'. "
                f"Got mode={mode!r}."
            )
        if has_complex_endpoint and torch.is_complex(torch.as_tensor(permittivity)):
            warnings.warn(
                "normalize_permittivity() is receiving complex permittivity. "
                "Only linear_eps is supported for complex endpoint normalization.",
                stacklevel=2,
            )

        permittivity = (permittivity**exp - eps_bg**exp) / (
            eps_r**exp - eps_bg**exp + 1e-8
        )

        return permittivity

    def denormalize_permittivity(self, permittivity, mode: str | None = None):
        ## input normalized permittivity is from [0,1]
        ## this is called interpolation process, linear interpolation is one common method
        ## however, we can also use other methods such as nonlinear interpolation, e.g., create absorption (imag part of eps) for intermediate permittivity density
        eps_r = self.design_region_cfg["eps"]
        eps_bg = self.design_region_cfg["eps_bg"]
        mode = mode or self.cfgs.get("denorm_mode", "linear_eps")

        alg, exp = mode.split("_")

        if exp == "eps":
            exp = 1
        elif exp == "index":
            exp = 0.5
        else:
            exp = float(exp)

        if alg == "linear":
            pass
        elif alg == "inverse":
            exp = -exp
        else:
            raise ValueError(f"Unsupported permittivity denormalization mode: {mode}")

        has_complex_endpoint = torch.is_complex(
            torch.as_tensor(eps_r)
        ) or torch.is_complex(torch.as_tensor(eps_bg))
        if has_complex_endpoint and mode != "linear_eps":
            raise ValueError(
                "Complex eps_r/eps_bg endpoints are only supported with denorm_mode='linear_eps'. "
                f"Got mode={mode!r}."
            )

        permittivity = (
            permittivity * (eps_r**exp - eps_bg**exp) + eps_bg**exp
        ) ** (1 / exp)

        return permittivity

    def denormalize_design_property(self, density, property_name: str):
        if property_name == "permittivity":
            return self.denormalize_permittivity(density)
        if property_name == "conductivity":
            return self.device.denormalize_design_region_conductivity(
                density, region_name=self.region_name
            )
        if property_name == "electrical_conductivity":
            return self.device.denormalize_design_region_electrical_conductivity(
                density, region_name=self.region_name
            )
        if property_name == "heat_capacity":
            return self.device.denormalize_design_region_heat_capacity(
                density, region_name=self.region_name
            )
        if property_name == "thermo_optic_coeff":
            return self.device.denormalize_design_region_thermo_optic_coeff(
                density, region_name=self.region_name
            )
        raise ValueError(f"Unsupported design property: {property_name}")

    def _is_design_property_enabled(self, property_name: str) -> bool:
        if property_name == "permittivity":
            return True
        probe = torch.tensor([0.0], dtype=torch.float32, device=self.operation_device)
        try:
            value = self.denormalize_design_property(probe, property_name)
        except Exception:
            return False
        return value is not None

    def _extrude_density_tensor(
        self,
        density: torch.Tensor,
        *,
        entire_eps: torch.Tensor,
        region_mask,
        axis_weights,
        grid_step: float,
    ) -> torch.Tensor:
        dims = sorted(self.cfgs.get("dims", (0, 1)))
        extrude_dim = list(set(range(3)) - set(dims))[0]
        extrude_grid_size = region_mask[extrude_dim]
        extrude_grid_size = extrude_grid_size.stop - extrude_grid_size.start
        extrude_angle = self.cfgs.get("extrude_angle", 90.0)
        extrude_direction = self.cfgs.get("extrude_direction", "-")
        extrude_z_downsample_factor = self.cfgs.get("extrude_z_downsample_factor", 1)
        physical_thickness = float(self.design_region_cfg["size"][extrude_dim])
        extrude_weights = None if axis_weights is None else axis_weights[extrude_dim]
        base_density = self._base_region_tensor(entire_eps, region_mask, density)
        base_density = self.normalize_permittivity(base_density)

        return extrude(
            density,
            extrude_dim=extrude_dim,
            extrude_size=extrude_grid_size,
            extrude_angle=extrude_angle,
            extrude_direction=extrude_direction,
            extrude_weights=extrude_weights,
            base_permittivity=base_density,
            grid_step=grid_step,
            physical_thickness=physical_thickness,
            z_downsample_factor=extrude_z_downsample_factor,
        )

    def forward(
        self,
        sharpness: float,
        hr_entire_eps: torch.Tensor,
        hr_dr_mask: torch.Tensor,
        *args,
        **kwargs,
    ):
        if args:
            if len(args) != 1 or not isinstance(args[0], dict):
                raise ValueError(
                    "Unexpected positional arguments to parametrization.forward(); "
                    "expected at most one dict of extra keyword arguments."
                )
            kwargs = {**args[0], **kwargs}
        ## first build the normalized design density using weights
        hr_density = self.build_density(self.weights, sharpness, **kwargs)

        ### then transform the density for all regions using transform settings
        ## e.g., mirror symmetry, transpose symmetry, convert resolution, ...

        ## after this, density will be downsampled to match the sim_cfg resolution
        ## while hr_density maintains the high-resolution representation
        hr_density, density = self.density_transform(
            hr_density,
            self.cfgs,
            sharpness,
            hr_entire_eps,
            hr_dr_mask,
        )

        if len(self.design_region_mask) == 3:
            ### 3D design region, extrude the density first so all material properties
            ### share the exact same transformed geometry.
            density = self._extrude_density_tensor(
                density,
                entire_eps=torch.as_tensor(
                    np.real(self.device.epsilon_map),
                    dtype=density.dtype,
                    device=density.device,
                ),
                region_mask=self.design_region_mask,
                axis_weights=self.design_region_axis_weights,
                grid_step=self.device.grid_step,
            )
            hr_density = self._extrude_density_tensor(
                hr_density,
                entire_eps=hr_entire_eps,
                region_mask=self.hr_design_region_mask,
                axis_weights=self.hr_design_region_axis_weights,
                grid_step=self.hr_device.grid_step,
            )

        ## we need to denormalize the density to the real material values for simulation
        region_maps = {
            "density": density,
            "permittivity": self.denormalize_design_property(density, "permittivity"),
            "conductivity": (
                self.denormalize_design_property(density, "conductivity")
                if self._is_design_property_enabled("conductivity")
                else None
            ),
            "electrical_conductivity": (
                self.denormalize_design_property(density, "electrical_conductivity")
                if self._is_design_property_enabled("electrical_conductivity")
                else None
            ),
            "heat_capacity": (
                self.denormalize_design_property(density, "heat_capacity")
                if self._is_design_property_enabled("heat_capacity")
                else None
            ),
            "thermo_optic_coeff": (
                self.denormalize_design_property(density, "thermo_optic_coeff")
                if self._is_design_property_enabled("thermo_optic_coeff")
                else None
            ),
        }
        hr_region_maps = {
            "density": hr_density,
            "permittivity": self.denormalize_design_property(
                hr_density, "permittivity"
            ),
            "conductivity": None,
            "electrical_conductivity": None,
            "heat_capacity": None,
            "thermo_optic_coeff": None,
        }
        ## check tensor shape
        # for property_name in region_maps.keys():
        #     if region_maps[property_name] is not None:
        #         print(f"Shape of {property_name}: {region_maps[property_name].shape}")
        return hr_region_maps, region_maps
