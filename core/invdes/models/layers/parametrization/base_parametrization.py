"""
Date: 2024-10-04 23:22:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-23 10:33:07
FilePath: /MAPS/core/invdes/models/layers/parametrization/base_parametrization.py
"""

import secrets
import warnings
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.types import Device

from core.inv_litho.photonic_model import *
from core.invdes.models.materialization.coverage import (
    average_region_cells,
    average_uniform_cells,
)
from core.utils import padding_to_tiles, resample_rectilinear_tensor, rip_padding

from .extrude import extrude, get_required_lateral_halo


def cvt_res(
    x,
    source_resolution: int = None,
    target_resolution: int = None,
    intplt_mode="nearest",
    target_size=None,
):
    """
    Convert the last two spatial dimensions of a tensor to a target resolution.

    Args:
        x: 2-D or 3-D tensor whose final two dimensions are spatial.
        source_resolution: Source resolution in pixels per micrometer.
        target_resolution: Target resolution in pixels per micrometer.
        intplt_mode: PyTorch interpolation mode, typically ``nearest`` or ``bilinear``.
        target_size: Optional explicit ``(nx, ny)`` output size; overrides resolution-based sizing.

    Returns:
        Tensor with the same leading dimensions as ``x`` and spatial shape ``target_size``. If the input already has the target shape, it is returned unchanged.
    """
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
    """
    Construct a mirrored design tensor while reducing the mirrored contribution to the shared-parameter gradient.

    Args:
        x: Design tensor with symmetry applied along the requested spatial dimensions.
        dims: Iterable of spatial axes to mirror; the implementation supports axes ``0`` and ``1``.

    Returns:
        Tensor containing the original half plus its mirrored copy. For even sizes each half is connected through ``DecayGradient`` with factor ``0.5``; odd sizes retain a central row/column.
    """
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
    """
    Apply mirror symmetry to every tensor in a parameter list.

    Args:
        xs: Sequence of design tensors.
        dims: Spatial axes along which each tensor is mirrored.

    Returns:
        List of mirrored tensors with the same ordering as ``xs``.
    """
    xs = [_mirror_symmetry(x, dims) for x in xs]
    return xs


def _transpose_symmetry(x, rot_k: int = 3):
    """
    Construct a transpose-reflected square design and optionally rotate it by 90-degree increments.

    Args:
        x: Square 2-D design tensor.
        rot_k: Number of quarter-turn rotations applied after transpose symmetry.

    Raises:
        AssertionError: If ``x`` is not square.

    Returns:
        Symmetry-constrained square tensor with the same spatial shape as ``x``.
    """
    assert (
        x.shape[0] == x.shape[1]
    ), "Only support square matrix for transpose symmetry, but got shape {}".format(
        x.shape
    )
    x_t = torch.transpose(x, 0, 1)
    x = torch.tril(x, -1) + torch.triu(x_t)
    x = torch.rot90(x, k=rot_k, dims=[-2, -1])

    return x


def transpose_symmetry(xs: Tuple | List, rot_k: int = 2) -> List:
    """
    Apply transpose symmetry to every tensor in a sequence.

    Args:
        xs: Sequence of square design tensors.
        rot_k: Number of quarter-turn rotations passed to ``_transpose_symmetry``.

    Returns:
        List of symmetry-constrained tensors.
    """
    xs = [_transpose_symmetry(x, rot_k=rot_k) for x in xs]
    return xs


def _rotation_symmetry(x, rot_k: int = 2):
    ## do not use rotation and average, it will get gray pixel
    """
    Construct 90-degree or 180-degree rotational symmetry for a square design.

    Args:
        x: Square spatial design tensor.
        rot_k: ``1`` for 90-degree symmetry or ``2`` for 180-degree symmetry.

    Raises:
        AssertionError: If ``x`` is not square.
        ValueError: If ``rot_k`` is not 1 or 2.

    Returns:
        Tensor satisfying the requested rotational symmetry while preserving odd-size center pixels/rows where applicable.
    """
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
    """
    Apply rotational symmetry to every tensor in a sequence.

    Args:
        xs: Sequence of square design tensors.
        rot_k: ``1`` for 90-degree or ``2`` for 180-degree symmetry.

    Returns:
        List of symmetry-constrained tensors.
    """
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
    """
    Resample a design tensor from one spatial resolution to another.

    Args:
        x: 2-D or 3-D tensor. Its final two dimensions are resampled; a leading channel/dimension is preserved.
        source_resolution: Source pixels-per-micrometer resolution.
        target_resolution: Target pixels-per-micrometer resolution.
        intplt_mode: PyTorch interpolation mode.
        subpixel_smoothing: If true while downsampling, perform area averaging in permittivity space using ``eps_r`` and ``eps_bg``.
        eps_r: High-permittivity/material value used for subpixel smoothing.
        eps_bg: Background permittivity used for subpixel smoothing.
        target_size: Optional explicit output spatial shape ``(nx, ny)``.

    Returns:
        Tensor with final spatial shape ``target_size``.

    Notes:
        With ``subpixel_smoothing=True`` and downsampling, the code maps normalized density to permittivity, applies adaptive average pooling, then maps back to normalized density.
    """
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
        # Density is a real design variable. With complex material endpoints,
        # the material-aware averaging above is complex-valued, but its real
        # normalized component is the physical density that must be passed to
        # geometry operations and extrusion.
        return x.real if torch.is_complex(x) else x

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
    """
    Convert a parameterization tuple by resampling its design-density entry.

    Args:
        xs: Sequence containing parameterization outputs/metadata. The current
            implementation resamples ``xs[1]`` and replaces the final element of
            the returned sequence with the resampled tensor. This implies the
            current expected structure has the design tensor at index 1 and
            preserves all elements before the final element.
        source_resolution: Source pixels-per-micrometer resolution.
        target_resolution: Target pixels-per-micrometer resolution.
        intplt_mode: Interpolation mode.
        subpixel_smoothing: Whether to use permittivity-space area averaging on downsampling.
        eps_r: Relative permittivity of the high-index material for smoothing.
        eps_bg: Background relative permittivity.
        target_size: Optional explicit target shape.

    Returns:
        A list formed as ``list(xs[:-1]) + [resampled_x]``. The resampled tensor
    has the requested spatial resolution.
    """
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
    """
    Apply the nominal lithography model to a high-resolution design region.

    Args:
        x_310: Normalized density tensor representing the high/low-resolution design pattern.
        res: Simulation resolution in pixels per micrometer.
        entire_eps: Full-device epsilon tensor modified at ``dr_mask`` during the intermediate lithography simulation.
        dr_mask: Design-region spatial mask/slice applied to ``entire_eps``.
        device: Torch device used by the lithography model.

    Returns:
        Lithography-filtered density restricted back to the original design-region shape.
    """
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
    """
    Apply the nominal lithography model to every member of a two-pattern parameter tuple.

    Args:
        xs: Sequence containing the high-resolution and simulation-resolution design patterns.
        res: Common simulation resolution.
        entire_eps: Full-device epsilon workspace.
        dr_mask: Design-region mask.
        device: Torch device for the lithography model.

    Returns:
        Two filtered design-region tensors in the same ordering as the two input patterns.
    """
    hr_out = _litho(xs[0], res, entire_eps, dr_mask, device)
    out = _litho(xs[1], res, entire_eps, dr_mask, device)
    return [hr_out, out]


def _etching(x, sharpness, eta, binary_projection):
    # in this case, we only consider the nominal corner for etching
    """
    Apply binary projection to a single normalized design tensor for nominal etching.

    Args:
        x: Normalized density tensor.
        sharpness: Scalar projection sharpness.
        eta: Projection threshold parameter.
        binary_projection: Callable implementing the configured binary projection.

    Returns:
        Projected density tensor with the same spatial shape as ``x``.
    """
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
    """
    Apply the configured binary-projection etching transform to a sequence of densities.

    Args:
        xs: Sequence of normalized design-density tensors. The first entry is preserved unchanged; projection is applied to the remaining entries.
        sharpness: Binary-projection sharpness.
        eta: Binary-projection threshold.
        binary_projection: Projection callable.

    Returns:
        List whose first element is ``xs[0]`` and remaining elements are projected tensors.
    """
    outs = [xs[0]]
    outs += [_etching(x, sharpness, eta, binary_projection) for x in xs[1:]]
    return outs


def _blur(x, mfs, res, entire_eps, dr_mask, dim="xy", ste: bool = False):
    """
    Apply minimum-feature-size averaging blur along one or more spatial axes.

    Args:
        x: 2-D normalized density tensor.
        mfs: Minimum feature size in physical units.
        res: Grid resolution used to convert ``mfs`` to pixels.
        entire_eps: Full-device epsilon workspace.
        dr_mask: Design-region mask.
        dim: ``x``, ``y``, or ``xy``; ``xy`` uses a circular 2-D averaging kernel.
        ste: If true, use a straight-through estimator that returns the unblurred forward tensor but routes gradient through the blurred result.

    Returns:
        Blurred design-region tensor, or an STE-wrapped tensor when ``ste=True``.
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
    x_ste = x

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

    class STEFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, input_blur):
            return input

        @staticmethod
        def backward(ctx, grad_output):
            return None, grad_output

    if ste:
        return STEFunction.apply(x_ste, x)
    return x


def blur(xs, mfs, resolutions, entire_eps, dr_mask, dim="xy", ste: bool = False):
    """
    Apply minimum-feature-size blur to multiple design-density tensors.

    Args:
        xs: Sequence of 2-D density tensors.
        mfs: Minimum feature size in physical units.
        resolutions: Per-tensor grid resolutions, one value per entry of ``xs``.
        entire_eps: Full-device epsilon workspace.
        dr_mask: Design-region mask.
        dim: Blur direction ``x``, ``y``, or ``xy``.
        ste: Whether to use straight-through differentiation.

    Returns:
        List of blurred tensors in the same ordering as ``xs``.
    """
    xs = [
        _blur(x, mfs, res, entire_eps, dr_mask, dim, ste=ste)
        for x, res in zip(xs, resolutions)
    ]
    return xs


def _fft(x, mfs, res, entire_eps, dr_mask, dim="xy"):
    """
    Filter a design density in the spatial-frequency domain for minimum-feature-size control.

    Args:
        x: 2-D design tensor.
        mfs: Minimum feature size in physical units.
        res: Resolution used to determine the Fourier cutoff.
        entire_eps: Full-device epsilon workspace.
        dr_mask: Design-region mask.
        dim: Must be ``xy`` in the current implementation.

    Returns:
        Real-valued filtered design tensor restricted to ``dr_mask``.
    """
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
    Apply the FFT-based minimum-feature-size filter to multiple design tensors.

    Args:
        xs: Sequence of 2-D design tensors.
        mfs: Minimum feature size in physical units.
        resolutions: Per-tensor resolutions.
        entire_eps: Full-device epsilon workspace.
        dr_mask: Design-region mask.
        dim: Currently only ``xy`` is supported.

    Returns:
        List of filtered tensors.
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
    """
    Base class for differentiable MAPS inverse-design parameterizations.

    The class owns the native/high-resolution design-region masks, simulation configuration, operation device, learnable parameter containers, and dispatch tables that map a parameterization ``method`` to build/reset functions.

    Args:
        device: Native-resolution device exposing ``design_region_masks`` and design-region configuration.
        hr_device: High-resolution device used for parameterization and export.
        sim_cfg: Simulation configuration dictionary.
        region_name: Design-region key used to select the native/high-resolution masks.
        cfgs: Parameterization configuration. Important entries include ``method``, ``rho_resolution``, ``transform``, ``init_method``, ``denorm_mode``, ``dims``, and extrusion settings.
        operation_device: Torch device used for learnable parameters and derived tensors.

    Key attributes:
        ``weights``: ``nn.ParameterDict`` or ``nn.ModuleDict`` created by ``build_parameters``.
        ``params``: Auxiliary parameter/configuration dictionary returned by the build function.
        ``design_region_mask`` / ``hr_design_region_mask``: Native/high-resolution spatial masks.

    Shape convention:
        2-D parameterization tensors generally use ``(nx, ny)``; 3-D extensions may add a third spatial dimension according to ``dims`` and extrusion configuration.
    """

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
            extrude_angle=0.0,
            reference_plane="middle",
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
        """
        Register the factory used to build learnable parameters for a parameterization method.

        Args:
            method: String identifier matching ``cfgs["method"]``.
            fn: Callable receiving configuration, design-region configuration, and optional extra arguments; it must return ``(weight_dict, param_dict)``.

        Returns:
            None.
        """
        self._parameter_build_per_region_fns[method] = fn

    def register_parameter_reset_per_region_fn(self, method, fn):
        """
        Register the initializer/reset function for a parameterization method.

        Args:
            method: String identifier matching ``cfgs["method"]``.
            fn: Callable invoked as ``fn(weights, cfgs, design_region_cfgs, init_method)``.

        Returns:
            None.
        """
        self._parameter_reset_per_region_fns[method] = fn

    def build_parameters(self, cfgs, design_region_cfg, *args, **kwargs):
        """
        Build and register learnable parameter containers for the configured parameterization method.

        Args:
            cfgs: Parameterization configuration; ``cfgs["method"]`` selects the registered build function.
            design_region_cfg: Device design-region configuration.
            *args: Additional build-function arguments.
            **kwargs: Additional build-function keyword arguments.

        Raises:
            ValueError: If the method has no registered builder or if the builder returns a mixture of ``nn.Parameter`` and ``nn.Module`` values.

        Side Effects:
            Stores the learnable state in ``self.weights`` as an ``nn.ParameterDict`` or ``nn.ModuleDict`` and stores auxiliary outputs in ``self.params``.
        """
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
        """
        Reset learnable parameters using the registered initializer for the configured method.

        Args:
            cfgs: Parameterization configuration.
            design_region_cfgs: Design-region configuration.
            *args: Reserved extra arguments.
            **kwargs: Reserved extra keyword arguments.

        Raises:
            ValueError: If no reset function is registered for ``cfgs["method"]``.

        Returns:
            None.
        """
        method = cfgs["method"]
        init_method = cfgs["init_method"]

        _reset_fn = self._parameter_reset_per_region_fns.get(method, None)
        if _reset_fn is not None:
            _reset_fn(self.weights, cfgs, design_region_cfgs, init_method)
        else:
            raise ValueError(f"Unsupported parameterization reset method: {method}")

    def build_density(self, weights, sharpness: float, **kwargs):
        ### return: normalized design density in [0, 1], typically at high resolution
        """
        Build a normalized design density from the parameterization weights.

        Args:
            weights: Learnable parameter container produced by ``build_parameters``.
            sharpness: Binarization/projection sharpness used by parameterizations that support continuation.
            **kwargs: Method-specific arguments.

        Returns:
            Normalized design density, typically with values in ``[0, 1]`` and at the high-resolution parameterization grid. Concrete subclasses implement the actual mapping.
        """
        raise NotImplementedError

    def build_permittivity(self, weights, sharpness: float, **kwargs):
        """
        Build the design permittivity representation from parameterization weights.

        Args:
            weights: Learnable parameter container.
            sharpness: Projection/sharpness control.
            **kwargs: Method-specific options.

        Returns:
            Current implementation delegates to ``build_density``; concrete subclasses may override this to return material-property values directly.
        """
        return self.build_density(weights, sharpness, **kwargs)

    def add_roughness(self, density, *, A, Lc, seed=None, **kwargs):
        """
        Apply a roughness model to a parameterized design density.

        Args:
            density: Normalized design-density tensor.
            A: Roughness amplitude parameter.
            Lc: Roughness correlation length.
            seed: Optional random seed controlling stochastic roughness.
            **kwargs: Additional roughness-model parameters.

        Raises:
            NotImplementedError: BaseParametrization provides no roughness implementation; subclasses must override it.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement add_roughness"
        )

    def _uniform_transform_reference(self, density, entire_eps, region_mask):
        """Return a full uniform reference map and mask for density transforms."""
        source_map = getattr(self.hr_device, "export_epsilon_map", None)
        source_metadata = getattr(self.hr_device, "export_grid_metadata", None)
        if source_map is None or source_metadata is None:
            source_map = entire_eps
            source_metadata = getattr(self.hr_device, "optical_grid_metadata", None)

        source_map = torch.as_tensor(
            source_map, dtype=density.dtype, device=density.device
        )
        dims = sorted(self.cfgs.get("dims", (0, 1)))
        if source_map.ndim == 3:
            extrude_dim = list(set(range(3)) - set(dims))[0]
            source_map = source_map.select(
                extrude_dim, source_map.shape[extrude_dim] // 2
            )

        if source_metadata is None or source_metadata.get("coords") is None:
            target_shape = tuple(
                max(int(source_map.shape[d]), int(n))
                for d, n in enumerate(density.shape)
            )
            reference = F.interpolate(
                source_map[None, None],
                size=target_shape,
                mode="bilinear",
                align_corners=True,
            )[0, 0]
            return (
                reference,
                tuple(slice(0, n) for n in density.shape),
                tuple(
                    float(n - 1) / max(float(size), 1e-12)
                    for n, size in zip(density.shape, self.design_region_cfg["size"])
                ),
            )

        coords = source_metadata["coords"]
        region_center = self.design_region_cfg["center"]
        region_size = self.design_region_cfg["size"]
        target_shape = []
        target_mask = []
        resolutions = []
        for axis_index, dim in enumerate(dims):
            # source_coords = torch.from_numpy(coords[dim])
            source_coords = coords[dim]
            source_span = float(source_coords[-1] - source_coords[0])
            density_count = int(density.shape[axis_index])
            density_spacing = float(region_size[dim]) / max(density_count - 1, 1)
            full_count = max(2, int(round(source_span / density_spacing)) + 1)
            start = int(
                round(
                    (
                        float(region_center[dim])
                        - float(region_size[dim]) / 2.0
                        - float(source_coords[0])
                    )
                    / max(density_spacing, 1e-12)
                )
            )
            start = max(0, min(start, full_count - density_count))
            target_shape.append(full_count)
            target_mask.append(slice(start, start + density_count))
            resolutions.append(1.0 / max(density_spacing, 1e-12))

        reference = F.interpolate(
            source_map[None, None],
            size=tuple(target_shape),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        return reference, tuple(target_mask), tuple(resolutions)

    def _correlated_roughness_2d(self, density, *, A, Lc, seed=None):
        if A < 0 or Lc <= 0:
            raise ValueError("add_roughness requires A >= 0 and Lc > 0")
        if A == 0:
            return torch.zeros_like(density)
        if density.ndim != 2:
            raise ValueError("2D correlated roughness requires a 2D density")

        if seed is None:
            seed = getattr(self, "_add_roughness_seed", None)
            if seed is None:
                seed = secrets.randbits(63)
                self._add_roughness_seed = seed

        region_size = tuple(float(v) for v in self.design_region_cfg["size"][:2])
        spacing = tuple(
            size / max(length - 1, 1)
            for size, length in zip(region_size, density.shape)
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        noise = torch.randn(density.shape, generator=generator, dtype=torch.float64)
        noise = noise.to(device=density.device)
        frequencies = [
            torch.fft.fftfreq(length, d=step, device=density.device)
            for length, step in zip(density.shape, spacing)
        ]
        fx, fy = torch.meshgrid(frequencies[0], frequencies[1], indexing="ij")
        spectrum = torch.fft.fft2(noise)
        gaussian_filter = torch.exp(
            -0.5 * (2.0 * torch.pi * Lc) ** 2 * (fx.square() + fy.square())
        )
        roughness = torch.fft.ifft2(spectrum * gaussian_filter).real
        roughness = roughness - roughness.mean()
        roughness = roughness / roughness.std().clamp_min(1e-12) * A
        return roughness.to(dtype=density.dtype)

    def _base_region_tensor(self, eps_map, region_mask, like):
        base_region = eps_map[region_mask]
        if not isinstance(base_region, torch.Tensor):
            base_region = torch.as_tensor(base_region)
        if torch.is_complex(base_region) and not torch.is_complex(like):
            base_region = base_region.real
        return base_region.to(dtype=like.dtype, device=like.device)

    def _uniform_parameter_grid_coords(self, density):
        """Return physical coordinates for an optional uniform parameter grid.

        Geometry parameterizations that rasterize on an explicit uniform grid can
        override this hook. The returned coordinates must correspond to the final
        two-dimensional tensor axes of ``density`` and use the same physical units
        as the device grid metadata.

        Args:
            density (torch.Tensor): Internal parameterization density.

        Returns:
            tuple[torch.Tensor, torch.Tensor] | None: Strictly increasing x/y
            coordinates, or ``None`` when the parameterization uses the legacy
            shape-only conversion path.
        """
        del density
        return None

    def _resample_volume_fraction_on_grid(
        self, density, source_coords, target_device, target_region_mask
    ):
        """Conservatively average geometric fractions using physical cell edges.

        Args:
            density: Two-dimensional arithmetic material fractions.
            source_coords: Source sample coordinates in absolute physical units.
            target_device: Device owning target optical cell boundaries.
            target_region_mask: Target region slices in the full device.

        Returns:
            Averaged fractions, or None if boundary metadata is unavailable.
        """
        metadata = getattr(target_device, "grid_info_dict", {}).get("epsilon_map")
        if metadata is None or metadata.get("boundaries") is None:
            metadata = getattr(target_device, "optical_grid_metadata", None)
        if metadata is None or metadata.get("boundaries") is None:
            return None
        dims = sorted(self.cfgs.get("dims", (0, 1)))
        boundaries = tuple(
            np.asarray(metadata["boundaries"][dim])[
                target_region_mask[dim].start : target_region_mask[dim].stop + 1
            ]
            for dim in dims
        )
        center, size = self.design_region_cfg["center"], self.design_region_cfg["size"]
        bounds = tuple((center[d] - size[d] / 2, center[d] + size[d] / 2) for d in dims)
        return average_region_cells(density, source_coords, boundaries, bounds)

    def _resample_uniform_parameter_density(
        self,
        density,
        target_device,
        target_region_mask,
    ):
        """Resample geometric material fractions onto a device-region grid.

        Cell-boundary metadata selects conservative overlap averaging. Older
        coordinate-only metadata retains interpolation in arithmetic epsilon
        space. The returned fractions remain differentiable.

        Args:
            density (torch.Tensor): Arithmetic fractions on the uniform grid,
                already converted from nonlinear optimization coordinates.
            target_device: Device owning the target raster-grid metadata.
            target_region_mask: Region selector on ``target_device``.

        Returns:
            torch.Tensor | None: Density on the selected target region, or ``None``
            when coordinate metadata or the parameter-grid hook is unavailable.
        """
        source_coords = self._uniform_parameter_grid_coords(density)
        if source_coords is not None:
            averaged = self._resample_volume_fraction_on_grid(
                density, source_coords, target_device, target_region_mask
            )
            if averaged is not None:
                return averaged
        grid_info = getattr(target_device, "grid_info_dict", {}).get("epsilon_map")
        if source_coords is None or grid_info is None:
            return None

        dims = sorted(self.cfgs.get("dims", (0, 1)))
        if len(dims) != 2 or density.ndim != 2:
            return None
        try:
            target_coords = tuple(
                torch.as_tensor(
                    np.asarray(grid_info["coords"][dim])[target_region_mask[dim]],
                    dtype=density.real.dtype,
                    device=density.device,
                )
                for dim in dims
            )
        except (KeyError, IndexError, TypeError):
            return None

        source_coords = tuple(
            torch.as_tensor(coord, dtype=density.real.dtype, device=density.device)
            for coord in source_coords
        )
        if any(
            coord.numel() != density.shape[axis]
            for axis, coord in enumerate(source_coords)
        ):
            raise ValueError(
                "Uniform parameter-grid coordinates must match density shape; "
                f"got shape={tuple(density.shape)} and "
                f"coord lengths={tuple(coord.numel() for coord in source_coords)}."
            )

        # Design-region masks may include partially intersecting edge cells whose
        # centers lie just outside the physical region. Clamp those samples to the
        # parameter-grid endpoints instead of extrapolating material values.
        target_coords = tuple(
            coord.clamp(source[0], source[-1])
            for coord, source in zip(target_coords, source_coords)
        )
        # This helper resamples geometric fractions, not optimization variables.
        physical_density = self.denormalize_permittivity(density, mode="linear_eps")
        physical_target = resample_rectilinear_tensor(
            physical_density,
            src_coords=source_coords,
            dst_coords=target_coords,
            axes=(0, 1),
            mode="cartesian",
        )
        normalized_target = self.normalize_permittivity(
            physical_target, mode="linear_eps"
        )
        # Keep the geometry density real even when the intermediate
        # permittivity uses complex PCM endpoints.
        return (
            normalized_target.real
            if torch.is_complex(normalized_target)
            else normalized_target
        )

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

        roughness_cfgs = []

        def _restore_transform_boundaries(transformed, original, boundaries):
            if boundaries is None:
                return transformed
            restored = transformed.clone()
            for boundary in boundaries:
                if boundary == "x":
                    restored[0] = original[0]
                    restored[-1] = original[-1]
                elif boundary == "y":
                    restored[:, 0] = original[:, 0]
                    restored[:, -1] = original[:, -1]
                else:
                    raise ValueError(
                        f"Unsupported transform boundary {boundary!r}; "
                        "expected 'x' or 'y'."
                    )
            return restored

        transform_index = 0
        while transform_index < len(transform_cfg_list):
            transform_cfg = transform_cfg_list[transform_index]
            transform_type = transform_cfg["type"]
            if transform_type == "binarize":
                preserve_boundaries = transform_cfg.get("preserve_boundaries")
                original_density = hr_density
                hr_density = self.binary_projection(
                    hr_density,
                    sharpness,
                    self.eta,
                    resolution=self.hr_device.resolution,
                )
                hr_density = _restore_transform_boundaries(
                    hr_density, original_density, preserve_boundaries
                )
                # plt.figure()
                # plt.imshow(1 - np.rot90(hr_permittivity.cpu().numpy()), cmap="gray")
                # plt.savefig(f"./figs/binarize_hr.png")
                # plt.close()
                transform_index += 1
                continue
            if transform_type == "add_roughness":
                roughness_cfg = dict(transform_cfg)
                del roughness_cfg["type"]
                if "amplitude" in roughness_cfg and "A" not in roughness_cfg:
                    roughness_cfg["A"] = roughness_cfg.pop("amplitude")
                if "correlation_length" in roughness_cfg and "Lc" not in roughness_cfg:
                    roughness_cfg["Lc"] = roughness_cfg.pop("correlation_length")
                if "A" not in roughness_cfg or "Lc" not in roughness_cfg:
                    raise ValueError(
                        "add_roughness requires both A and Lc (or amplitude and "
                        "correlation_length)"
                    )
                roughness_cfgs.append(roughness_cfg)
                transform_index += 1
                continue
            cfg = deepcopy(transform_cfg)
            del cfg["type"]
            preserve_boundaries = cfg.pop("preserve_boundaries", None)
            original_density = hr_density
            if "device" in cfg.keys():
                assert cfg["device"] == "cuda", "running on cpu is not supported"
                cfg["device"] = self.operation_device
            if "binary_proj_layer" in cfg.keys():
                cfg["binary_projection"] = self.binary_projection
            if "litho" in transform_type:
                cfg["device"] = self.operation_device
                transform_reference, transform_mask, _ = (
                    self._uniform_transform_reference(
                        hr_density, hr_entire_eps, hr_dr_mask
                    )
                )
                cfg["entire_eps"] = _real_transform_reference_map(transform_reference)
                cfg["dr_mask"] = transform_mask
            if "blur" in transform_type or "fft" in transform_type:
                transform_reference, transform_mask, transform_resolutions = (
                    self._uniform_transform_reference(
                        hr_density, hr_entire_eps, hr_dr_mask
                    )
                )
                cfg["entire_eps"] = _real_transform_reference_map(transform_reference)
                cfg["dr_mask"] = transform_mask
                if "blur" in transform_type and "resolutions" in cfg:
                    cfg["resolutions"] = list(transform_resolutions)

            ## apply differentiable transformation to the high-resolution density
            (hr_density,) = permittivity_transform_collections[transform_type](
                (hr_density,), **cfg
            )
            hr_density = _restore_transform_boundaries(
                hr_density, original_density, preserve_boundaries
            )
            transform_index += 1
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
            # Nonlinear optimization interpolation must precede geometric
            # averaging. All resampling hooks below consume volume fractions.
            hr_density = self._density_to_volume_fraction(hr_density)
            ### now we have two tasks:
            dims = sorted(self.cfgs.get("dims", (0, 1)))
            design_region_mask = [self.design_region_mask[d] for d in dims]

            target_size = [(m.stop - m.start) for m in design_region_mask]

            ## task 1: fit the hr_density to hr_design_region as they might not match
            ## this does not requires any grad
            with torch.inference_mode():
                hr_design_region_mask = [self.hr_design_region_mask[d] for d in dims]
                hr_target_size = [(m.stop - m.start) for m in hr_design_region_mask]

                resample_to_hr = getattr(self, "_resample_density_to_hr", None)
                if resample_to_hr is None:
                    uniform_hr_density = self._resample_uniform_parameter_density(
                        hr_density.detach(),
                        self.hr_device,
                        self.hr_design_region_mask,
                    )
                else:
                    uniform_hr_density = None
                if uniform_hr_density is None:
                    hr_density_nograd = _convert_resolution(
                        hr_density.detach(),
                        intplt_mode="nearest",
                        target_size=hr_target_size,
                    )
                else:
                    hr_density_nograd = uniform_hr_density
                if resample_to_hr is not None:
                    hr_density_nograd = resample_to_hr(
                        hr_density.detach(), target_size=hr_target_size
                    )

            ## task 2: convert the hr_density to the sim_cfg resolution for simulation
            resample_to_native = getattr(self, "_resample_density_to_native", None)
            if resample_to_native is None:
                uniform_lr_density = self._resample_uniform_parameter_density(
                    hr_density,
                    self.device,
                    self.design_region_mask,
                )
            else:
                uniform_lr_density = None
            if uniform_lr_density is None:
                lr_density_grad = average_uniform_cells(hr_density, target_size)
            else:
                lr_density_grad = uniform_lr_density
            if resample_to_native is not None:
                lr_density_grad = resample_to_native(
                    hr_density,
                    target_size=target_size,
                    eps_r=self.design_region_cfg["eps"],
                    eps_bg=self.design_region_cfg["eps_bg"],
                )

            for roughness_cfg in roughness_cfgs:
                lr_density_grad = self.add_roughness(
                    lr_density_grad,
                    **roughness_cfg,
                )

            return (
                self._volume_fraction_to_density(hr_density_nograd),
                self._volume_fraction_to_density(lr_density_grad),
            )

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

    def _density_to_volume_fraction(self, density: torch.Tensor) -> torch.Tensor:
        """Convert optimization density to the arithmetic epsilon fraction.

        Args:
            density: Optimization variables after projection and transforms.

        Returns:
            Real geometric fractions corresponding to the physical epsilon map.
            Linear-epsilon parametrizations are returned unchanged.
        """
        if self.cfgs.get("denorm_mode", "linear_eps") == "linear_eps":
            return density
        eps = self.denormalize_permittivity(density)
        lo, hi = self.design_region_cfg["eps_bg"], self.design_region_cfg["eps"]
        fraction = (eps - lo) / (hi - lo)
        return fraction.real if torch.is_complex(fraction) else fraction

    def _volume_fraction_to_density(self, fraction: torch.Tensor) -> torch.Tensor:
        """Encode averaged epsilon in the configured optimization coordinate.

        Args:
            fraction: Real arithmetic material fractions after geometric averaging.

        Returns:
            Equivalent optimization-coordinate density. This is a change of
            representation only; it does not apply a second material mixture.
        """
        if self.cfgs.get("denorm_mode", "linear_eps") == "linear_eps":
            return fraction
        eps = self.denormalize_permittivity(fraction, mode="linear_eps")
        # Use the exact inverse, without the visualization normalizer's epsilon
        # regularizer: repeated encode/decode must preserve physical epsilon.
        mode = self.cfgs["denorm_mode"]
        algorithm, exponent = mode.split("_")
        exponent = {"eps": 1.0, "index": 0.5}.get(exponent, exponent)
        exponent = float(exponent) * (-1 if algorithm == "inverse" else 1)
        lo, hi = self.design_region_cfg["eps_bg"], self.design_region_cfg["eps"]
        return (eps**exponent - lo**exponent) / (hi**exponent - lo**exponent)

    def _extrude_density_tensor(
        self,
        density: torch.Tensor,
        *,
        entire_eps: torch.Tensor,
        region_mask,
        axis_weights,
        grid_step: float,
    ) -> tuple[torch.Tensor, tuple[slice, ...], np.ndarray | None]:
        """Extrude a design density using fixed geometry around its lateral border.

        Args:
            density (torch.Tensor): Two-dimensional optimized reference-plane
                density in the original design region.
            entire_eps (torch.Tensor): Full-device permittivity tensor supplying
                fixed reference-plane context and the merge background.
            region_mask: Three-dimensional design-region slices.
            axis_weights: Optional per-axis cell-coverage weights for the design
                region.
            grid_step (float): Device grid spacing in physical length units.

        Returns:
            tuple: ``(density_3d, support_mask, support_weight)``. For nonzero
            sidewall angles, ``support_mask`` includes the required lateral halo;
            otherwise the original design-region mask is returned. The optional
            support weight preserves the original lateral cell-coverage weights
            while assigning full coverage to halo cells.

        Notes:
            The fixed context is detached before composition, so gradients flow
            only through ``density``. Coverage and morphology operate on arithmetic
            epsilon fractions, never nonlinear optimization coordinates. The result
            is converted back only to preserve the caller's density contract.
            The extruded density is unioned with the
            fixed support density to prevent background in the extrusion result
            from erasing existing same-material geometry.
        """
        dims = sorted(self.cfgs.get("dims", (0, 1)))
        extrude_dim = list(set(range(3)) - set(dims))[0]
        extrude_grid_size = region_mask[extrude_dim]
        extrude_grid_size = extrude_grid_size.stop - extrude_grid_size.start
        extrude_arbitrary = self.cfgs.get("extrude_arbitrary", False)
        extrude_angle = self.cfgs.get("extrude_angle", 0.0)
        extrude_direction = self.cfgs.get("extrude_direction", "-")
        reference_plane = self.cfgs.get("reference_plane", "middle")
        extrude_z_downsample_factor = self.cfgs.get("extrude_z_downsample_factor", 1)
        physical_thickness = float(self.design_region_cfg["size"][extrude_dim])
        extrude_weights = None if axis_weights is None else axis_weights[extrude_dim]
        halo_px = get_required_lateral_halo(
            extrude_size=extrude_grid_size,
            extrude_angle=extrude_angle,
            extrude_direction=extrude_direction,
            reference_plane=reference_plane,
            grid_step=grid_step,
            physical_thickness=physical_thickness,
            extrude_weights=extrude_weights,
        )

        support_mask = tuple(region_mask)
        density = self._density_to_volume_fraction(density)
        context_density = density
        support_weight = None
        if halo_px > 0:
            support_slices = list(region_mask)
            local_slices = []
            for dim in dims:
                region_slice = region_mask[dim]
                start = max(0, int(region_slice.start) - halo_px)
                stop = min(int(entire_eps.shape[dim]), int(region_slice.stop) + halo_px)
                support_slices[dim] = slice(start, stop)
                local_slices.append(
                    slice(
                        int(region_slice.start) - start, int(region_slice.stop) - start
                    )
                )
            support_mask = tuple(support_slices)

            extrude_slice = region_mask[extrude_dim]
            if reference_plane == "bottom":
                reference_index = int(extrude_slice.start)
            elif reference_plane == "top":
                reference_index = int(extrude_slice.stop) - 1
            else:
                reference_index = (
                    int(extrude_slice.start) + int(extrude_slice.stop) - 1
                ) // 2
            plane_index = list(support_mask)
            plane_index[extrude_dim] = reference_index
            fixed_context = self._base_region_tensor(
                entire_eps, tuple(plane_index), density
            )
            fixed_context = self.normalize_permittivity(
                fixed_context, mode="linear_eps"
            ).detach()
            pad = (
                local_slices[1].start,
                fixed_context.shape[1] - local_slices[1].stop,
                local_slices[0].start,
                fixed_context.shape[0] - local_slices[0].stop,
            )
            padded_density = F.pad(density, pad)
            local_design_mask = F.pad(torch.ones_like(density), pad)
            context_density = (
                fixed_context * (1 - local_design_mask)
                + padded_density * local_design_mask
            )

            support_weight = np.ones(
                tuple(slc.stop - slc.start for slc in support_mask),
                dtype=np.float32,
            )
            if axis_weights is not None:
                expanded_axis_weights = []
                for dim, support_slice in enumerate(support_mask):
                    length = int(support_slice.stop) - int(support_slice.start)
                    expanded = np.ones(length, dtype=np.float32)
                    if dim in dims:
                        local_slice = local_slices[dims.index(dim)]
                        expanded[local_slice] = np.asarray(
                            axis_weights[dim], dtype=np.float32
                        )
                    else:
                        expanded = np.asarray(axis_weights[dim], dtype=np.float32)
                    expanded_axis_weights.append(expanded)
                for dim, weight in enumerate(expanded_axis_weights):
                    if dim == extrude_dim:
                        continue
                    shape = [1] * 3
                    shape[dim] = len(weight)
                    support_weight *= weight.reshape(shape)

        base_density = self._base_region_tensor(entire_eps, support_mask, density)
        base_density = self.normalize_permittivity(
            base_density, mode="linear_eps"
        ).detach()

        extruded_density = extrude(
            context_density,
            extrude_dim=extrude_dim,
            extrude_size=extrude_grid_size,
            extrude_arbitrary=extrude_arbitrary,
            extrude_angle=extrude_angle,
            extrude_direction=extrude_direction,
            reference_plane=reference_plane,
            extrude_weights=extrude_weights,
            # The region replaces material inside its physical slab. The fixed
            # map is already cell-averaged and is not the material outside that
            # slab: blending it here would count port overlap a second time.
            # Zero fraction is the region's physical background. Fixed geometry
            # in an expanded sidewall halo is restored by the union below.
            base_permittivity=torch.zeros_like(base_density),
            grid_step=grid_step,
            physical_thickness=physical_thickness,
            z_downsample_factor=extrude_z_downsample_factor,
        )
        if halo_px > 0:
            extruded_density = torch.maximum(base_density, extruded_density)

        return (
            self._volume_fraction_to_density(extruded_density),
            support_mask,
            support_weight,
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
            density, region_support_mask, region_support_weight = (
                self._extrude_density_tensor(
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
            )
            hr_density, hr_region_support_mask, hr_region_support_weight = (
                self._extrude_density_tensor(
                    hr_density,
                    entire_eps=hr_entire_eps,
                    region_mask=self.hr_design_region_mask,
                    axis_weights=self.hr_design_region_axis_weights,
                    grid_step=self.hr_device.grid_step,
                )
            )
        else:
            region_support_mask = self.design_region_mask
            hr_region_support_mask = self.hr_design_region_mask
            region_support_weight = None
            hr_region_support_weight = None

        ## we need to denormalize the density to the real material values for simulation
        region_maps = {
            "_region_mask": region_support_mask,
            "_region_weight": region_support_weight,
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
            "_region_mask": hr_region_support_mask,
            "_region_weight": hr_region_support_weight,
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
