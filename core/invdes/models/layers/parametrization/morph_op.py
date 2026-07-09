from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

__all__ = [
    "dilation",
    "erosion",
    "opening",
    "closing",
]


def _build_binary_kernel_offsets(
    kernel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"kernel must be square [K, K], got {tuple(kernel.shape)}")
    active = torch.nonzero(kernel != 0, as_tuple=False)
    if active.numel() == 0:
        raise ValueError("kernel has no active entries")
    return (
        active[:, 0].to(torch.int32).contiguous(),
        active[:, 1].to(torch.int32).contiguous(),
    )


def _build_grayscale_kernel_data(
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor],
    max_val: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_y, active_x = _build_binary_kernel_offsets(kernel)
    if structuring_element is None:
        active_w = torch.zeros(
            active_y.numel(), device=kernel.device, dtype=torch.float32
        )
    else:
        if structuring_element.shape != kernel.shape:
            raise ValueError(
                "structuring_element must match kernel shape, got "
                f"{tuple(structuring_element.shape)} vs {tuple(kernel.shape)}"
            )
        neighborhood = structuring_element.to(dtype=torch.float32).clone()
        neighborhood = neighborhood.masked_fill(kernel == 0, -max_val)
        active = torch.nonzero(kernel != 0, as_tuple=False)
        active_w = neighborhood[active[:, 0], active[:, 1]].contiguous()
    return active_y, active_x, active_w


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=8),
        triton.Config({"BLOCK_HW": 512}, num_warps=8),
    ],
    key=["H", "W", "N_ACTIVE"],
)
@triton.jit
def _binary_dilation2d_fwd_kernel(
    x_ptr,
    oy_ptr,
    ox_ptr,
    out_ptr,
    stride_xb,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_ob,
    stride_oc,
    stride_oh,
    stride_ow,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    origin_y: tl.constexpr,
    origin_x: tl.constexpr,
    border_value,
    HW,
    N_ACTIVE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_bc = tl.program_id(1)
    b = pid_bc // C
    c = pid_bc % C

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs < HW
    h = offs // W
    w = offs % W

    best = tl.full([BLOCK_HW], -float("inf"), tl.float32)
    base_x = b * stride_xb + c * stride_xc
    base_o = b * stride_ob + c * stride_oc

    for k in range(N_ACTIVE):
        ky = tl.load(oy_ptr + k)
        kx = tl.load(ox_ptr + k)
        ih = h + ky - origin_y
        iw = w + kx - origin_x
        inside = mask_hw & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        x_ptrs = x_ptr + base_x + ih * stride_xh + iw * stride_xw
        xval = tl.load(x_ptrs, mask=inside, other=border_value)
        best = tl.maximum(best, xval.to(tl.float32))

    out_ptrs = out_ptr + base_o + h * stride_oh + w * stride_ow
    tl.store(out_ptrs, best.to(tl.float32), mask=mask_hw)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=8),
        triton.Config({"BLOCK_HW": 512}, num_warps=8),
    ],
    key=["H", "W", "N_ACTIVE"],
)
@triton.jit
def _binary_dilation2d_fwd_arg_kernel(
    x_ptr,
    oy_ptr,
    ox_ptr,
    out_ptr,
    arg_ptr,
    stride_xb,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_ob,
    stride_oc,
    stride_oh,
    stride_ow,
    stride_ab,
    stride_ac,
    stride_ah,
    stride_aw,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    origin_y: tl.constexpr,
    origin_x: tl.constexpr,
    border_value,
    HW,
    N_ACTIVE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_bc = tl.program_id(1)
    b = pid_bc // C
    c = pid_bc % C

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs < HW
    h = offs // W
    w = offs % W

    best = tl.full([BLOCK_HW], -float("inf"), tl.float32)
    best_idx = tl.full([BLOCK_HW], -1, tl.int32)
    base_x = b * stride_xb + c * stride_xc
    base_o = b * stride_ob + c * stride_oc
    base_a = b * stride_ab + c * stride_ac

    for k in range(N_ACTIVE):
        ky = tl.load(oy_ptr + k)
        kx = tl.load(ox_ptr + k)
        ih = h + ky - origin_y
        iw = w + kx - origin_x
        inside = mask_hw & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        x_ptrs = x_ptr + base_x + ih * stride_xh + iw * stride_xw
        xval = tl.load(x_ptrs, mask=inside, other=border_value).to(tl.float32)
        better = xval > best
        best = tl.where(better, xval, best)
        best_idx = tl.where(better, k, best_idx)

    out_ptrs = out_ptr + base_o + h * stride_oh + w * stride_ow
    arg_ptrs = arg_ptr + base_a + h * stride_ah + w * stride_aw
    tl.store(out_ptrs, best.to(tl.float32), mask=mask_hw)
    tl.store(arg_ptrs, best_idx, mask=mask_hw)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=8),
        triton.Config({"BLOCK_HW": 512}, num_warps=8),
    ],
    key=["H", "W", "N_ACTIVE"],
)
@triton.jit
def _binary_erosion2d_fwd_kernel(
    x_ptr,
    oy_ptr,
    ox_ptr,
    out_ptr,
    stride_xb,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_ob,
    stride_oc,
    stride_oh,
    stride_ow,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    origin_y: tl.constexpr,
    origin_x: tl.constexpr,
    border_value,
    HW,
    N_ACTIVE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_bc = tl.program_id(1)
    b = pid_bc // C
    c = pid_bc % C

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs < HW
    h = offs // W
    w = offs % W

    best = tl.full([BLOCK_HW], float("inf"), tl.float32)
    base_x = b * stride_xb + c * stride_xc
    base_o = b * stride_ob + c * stride_oc

    for k in range(N_ACTIVE):
        ky = tl.load(oy_ptr + k)
        kx = tl.load(ox_ptr + k)
        ih = h + ky - origin_y
        iw = w + kx - origin_x
        inside = mask_hw & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        x_ptrs = x_ptr + base_x + ih * stride_xh + iw * stride_xw
        xval = tl.load(x_ptrs, mask=inside, other=border_value)
        best = tl.minimum(best, xval.to(tl.float32))

    out_ptrs = out_ptr + base_o + h * stride_oh + w * stride_ow
    tl.store(out_ptrs, best.to(tl.float32), mask=mask_hw)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=8),
        triton.Config({"BLOCK_HW": 512}, num_warps=8),
    ],
    key=["H", "W", "N_ACTIVE"],
)
@triton.jit
def _binary_erosion2d_fwd_arg_kernel(
    x_ptr,
    oy_ptr,
    ox_ptr,
    out_ptr,
    arg_ptr,
    stride_xb,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_ob,
    stride_oc,
    stride_oh,
    stride_ow,
    stride_ab,
    stride_ac,
    stride_ah,
    stride_aw,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    origin_y: tl.constexpr,
    origin_x: tl.constexpr,
    border_value,
    HW,
    N_ACTIVE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_bc = tl.program_id(1)
    b = pid_bc // C
    c = pid_bc % C

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs < HW
    h = offs // W
    w = offs % W

    best = tl.full([BLOCK_HW], float("inf"), tl.float32)
    best_idx = tl.full([BLOCK_HW], -1, tl.int32)
    base_x = b * stride_xb + c * stride_xc
    base_o = b * stride_ob + c * stride_oc
    base_a = b * stride_ab + c * stride_ac

    for k in range(N_ACTIVE):
        ky = tl.load(oy_ptr + k)
        kx = tl.load(ox_ptr + k)
        ih = h + ky - origin_y
        iw = w + kx - origin_x
        inside = mask_hw & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        x_ptrs = x_ptr + base_x + ih * stride_xh + iw * stride_xw
        xval = tl.load(x_ptrs, mask=inside, other=border_value).to(tl.float32)
        better = xval < best
        best = tl.where(better, xval, best)
        best_idx = tl.where(better, k, best_idx)

    out_ptrs = out_ptr + base_o + h * stride_oh + w * stride_ow
    arg_ptrs = arg_ptr + base_a + h * stride_ah + w * stride_aw
    tl.store(out_ptrs, best.to(tl.float32), mask=mask_hw)
    tl.store(arg_ptrs, best_idx, mask=mask_hw)


@triton.jit
def _grayscale_morphology2d_fwd_kernel(
    x_ptr,
    oy_ptr,
    ox_ptr,
    w_ptr,
    out_ptr,
    arg_ptr,
    stride_xb,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_ob,
    stride_oc,
    stride_oh,
    stride_ow,
    stride_ab,
    stride_ac,
    stride_ah,
    stride_aw,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    origin_y: tl.constexpr,
    origin_x: tl.constexpr,
    border_value,
    morph_sign: tl.constexpr,
    HW,
    N_ACTIVE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_bc = tl.program_id(1)
    b = pid_bc // C
    c = pid_bc % C

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs < HW
    h = offs // W
    w = offs % W

    best = tl.full([BLOCK_HW], -float("inf"), tl.float32)
    best_idx = tl.full([BLOCK_HW], -1, tl.int32)
    base_x = b * stride_xb + c * stride_xc
    base_o = b * stride_ob + c * stride_oc
    base_a = b * stride_ab + c * stride_ac

    for k in range(N_ACTIVE):
        ky = tl.load(oy_ptr + k)
        kx = tl.load(ox_ptr + k)
        kw = tl.load(w_ptr + k)
        ih = h + ky - origin_y
        iw = w + kx - origin_x
        inside = mask_hw & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        x_ptrs = x_ptr + base_x + ih * stride_xh + iw * stride_xw
        xval = tl.load(x_ptrs, mask=inside, other=border_value)
        score = morph_sign * (xval + morph_sign * kw)
        better = score > best
        best = tl.where(better, score, best)
        best_idx = tl.where(better, k, best_idx)

    out_ptrs = out_ptr + base_o + h * stride_oh + w * stride_ow
    arg_ptrs = arg_ptr + base_a + h * stride_ah + w * stride_aw
    tl.store(out_ptrs, morph_sign * best, mask=mask_hw)
    tl.store(arg_ptrs, best_idx, mask=mask_hw)


class _BinaryMorphology2dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        active_y: torch.Tensor,
        active_x: torch.Tensor,
        origin: tuple[int, int],
        border_value: float,
        op: str,
    ):
        if not x.is_contiguous():
            x = x.contiguous()
        B, C, H, W = x.shape
        hw = H * W
        active_y = active_y.to(device=x.device, dtype=torch.int32).contiguous()
        active_x = active_x.to(device=x.device, dtype=torch.int32).contiguous()

        out = torch.empty_like(x)
        arg = torch.empty_like(x, dtype=torch.int32)
        grid = lambda meta: (triton.cdiv(hw, meta["BLOCK_HW"]), B * C)
        kernel_fn = (
            _binary_dilation2d_fwd_arg_kernel
            if op == "dilation"
            else _binary_erosion2d_fwd_arg_kernel
        )
        kernel_fn[grid](
            x,
            active_y,
            active_x,
            out,
            arg,
            stride_xb=x.stride(0),
            stride_xc=x.stride(1),
            stride_xh=x.stride(2),
            stride_xw=x.stride(3),
            stride_ob=out.stride(0),
            stride_oc=out.stride(1),
            stride_oh=out.stride(2),
            stride_ow=out.stride(3),
            stride_ab=arg.stride(0),
            stride_ac=arg.stride(1),
            stride_ah=arg.stride(2),
            stride_aw=arg.stride(3),
            C=C,
            H=H,
            W=W,
            origin_y=int(origin[0]),
            origin_x=int(origin[1]),
            border_value=float(border_value),
            HW=hw,
            N_ACTIVE=int(active_y.numel()),
        )
        ctx.save_for_backward(arg, active_y, active_x)
        ctx.origin = origin
        ctx.input_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        arg, active_y, active_x = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        B, C, H, W = ctx.input_shape
        device = grad_out.device
        grad_dtype = grad_out.dtype

        arg_flat = arg.view(-1)
        safe_arg = arg_flat.clamp_min(0).to(torch.long)
        ky = active_y.to(device=device, dtype=torch.long)[safe_arg]
        kx = active_x.to(device=device, dtype=torch.long)[safe_arg]

        hw_idx = torch.arange(H * W, device=device, dtype=torch.long).repeat(B * C)
        h = hw_idx // W
        w = hw_idx % W

        ih = h + ky - int(ctx.origin[0])
        iw = w + kx - int(ctx.origin[1])
        valid = arg_flat >= 0
        valid = valid & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

        grad_x_accum = torch.zeros((B * C, H * W), device=device, dtype=torch.float32)
        src_lin = ih * W + iw
        grad_vals = grad_out.view(-1).to(torch.float32)
        grad_x_accum.scatter_add_(
            1,
            src_lin.view(B * C, H * W).masked_fill(~valid.view(B * C, H * W), 0),
            grad_vals.view(B * C, H * W) * valid.view(B * C, H * W).to(torch.float32),
        )
        grad_x = grad_x_accum.view(B, C, H, W).to(grad_dtype)
        return grad_x, None, None, None, None, None


class _GrayscaleMorphology2dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        active_y: torch.Tensor,
        active_x: torch.Tensor,
        active_w: torch.Tensor,
        origin: tuple[int, int],
        border_value: float,
        block_hw: int,
        morph_sign: int,
    ):
        if not x.is_contiguous():
            x = x.contiguous()
        B, C, H, W = x.shape
        n_active = int(active_y.numel())

        active_y = active_y.to(device=x.device, dtype=torch.int32).contiguous()
        active_x = active_x.to(device=x.device, dtype=torch.int32).contiguous()
        active_w = active_w.to(device=x.device, dtype=x.dtype).contiguous()

        out = torch.empty_like(x)
        arg = torch.empty_like(x, dtype=torch.int32)
        hw = H * W
        grid = (triton.cdiv(hw, block_hw), B * C)
        _grayscale_morphology2d_fwd_kernel[grid](
            x,
            active_y,
            active_x,
            active_w,
            out,
            arg,
            stride_xb=x.stride(0),
            stride_xc=x.stride(1),
            stride_xh=x.stride(2),
            stride_xw=x.stride(3),
            stride_ob=out.stride(0),
            stride_oc=out.stride(1),
            stride_oh=out.stride(2),
            stride_ow=out.stride(3),
            stride_ab=arg.stride(0),
            stride_ac=arg.stride(1),
            stride_ah=arg.stride(2),
            stride_aw=arg.stride(3),
            C=C,
            H=H,
            W=W,
            origin_y=int(origin[0]),
            origin_x=int(origin[1]),
            border_value=float(border_value),
            morph_sign=int(morph_sign),
            HW=hw,
            N_ACTIVE=n_active,
            BLOCK_HW=int(block_hw),
        )

        ctx.save_for_backward(arg, active_y, active_x)
        ctx.origin = origin
        ctx.input_shape = x.shape
        ctx.morph_sign = int(morph_sign)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        arg, active_y, active_x = ctx.saved_tensors
        _, _, H, W = ctx.input_shape
        grad_x = torch.zeros_like(grad_out)
        grad_w = torch.zeros(
            active_y.numel(), device=grad_out.device, dtype=grad_out.dtype
        )

        for k in range(active_y.numel()):
            mask = arg == k
            if not mask.any():
                continue
            ky = int(active_y[k].item())
            kx = int(active_x[k].item())
            dy = ky - ctx.origin[0]
            dx = kx - ctx.origin[1]
            g = grad_out * mask.to(grad_out.dtype)
            grad_w[k] = g.sum() * ctx.morph_sign

            h_src0 = max(0, -dy)
            h_src1 = min(H, H - dy) if dy >= 0 else H
            h_dst0 = h_src0 + dy
            h_dst1 = h_src1 + dy
            w_src0 = max(0, -dx)
            w_src1 = min(W, W - dx) if dx >= 0 else W
            w_dst0 = w_src0 + dx
            w_dst1 = w_src1 + dx
            grad_x[:, :, h_dst0:h_dst1, w_dst0:w_dst1] += g[
                :, :, h_src0:h_src1, w_src0:w_src1
            ]

        return grad_x, None, None, grad_w, None, None, None, None


def _normalize_border(
    op: str,
    border_type: str,
    border_value: float,
    max_val: float,
) -> float:
    if border_type == "geodesic":
        return -max_val if op == "dilation" else max_val
    if border_type != "constant":
        raise NotImplementedError(
            "Only border_type='constant' or 'geodesic' is supported."
        )
    return float(border_value)


def _binary_morphology_inference(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    origin: tuple[int, int],
    border_value: float,
    op: str,
) -> torch.Tensor:
    active_y, active_x = _build_binary_kernel_offsets(kernel)
    active_y = active_y.to(device=tensor.device, dtype=torch.int32).contiguous()
    active_x = active_x.to(device=tensor.device, dtype=torch.int32).contiguous()

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    out = torch.empty_like(tensor)
    B, C, H, W = tensor.shape
    hw = H * W
    grid = lambda meta: (triton.cdiv(hw, meta["BLOCK_HW"]), B * C)
    kernel_fn = (
        _binary_dilation2d_fwd_kernel
        if op == "dilation"
        else _binary_erosion2d_fwd_kernel
    )
    kernel_fn[grid](
        tensor,
        active_y,
        active_x,
        out,
        stride_xb=tensor.stride(0),
        stride_xc=tensor.stride(1),
        stride_xh=tensor.stride(2),
        stride_xw=tensor.stride(3),
        stride_ob=out.stride(0),
        stride_oc=out.stride(1),
        stride_oh=out.stride(2),
        stride_ow=out.stride(3),
        C=C,
        H=H,
        W=W,
        origin_y=int(origin[0]),
        origin_x=int(origin[1]),
        border_value=float(border_value),
        HW=hw,
        N_ACTIVE=int(active_y.numel()),
    )
    return out


def _morphology(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[list[int] | tuple[int, int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "auto",
    op: str = "dilation",
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")
    if tensor.ndim != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")
    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")
    if kernel.ndim != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")
    if not tensor.is_cuda:
        raise ValueError("tensor must be a CUDA tensor")
    if not tensor.is_floating_point():
        raise TypeError("tensor must be floating point")
    if op not in {"dilation", "erosion"}:
        raise ValueError(f"Unsupported morphology op: {op}")
    if engine not in {"auto", "triton"}:
        raise ValueError(f"Unsupported engine: {engine}")

    se_h, se_w = kernel.shape
    if origin is None:
        origin_ = (se_h // 2, se_w // 2)
    else:
        origin_ = (int(origin[0]), int(origin[1]))

    border_value_ = _normalize_border(op, border_type, border_value, max_val)

    # Fast path for flat binary morphology. Training uses exact hard max/min
    # backward via saved winners; inference skips arg storage entirely.
    if structuring_element is None:
        if tensor.requires_grad:
            active_y, active_x = _build_binary_kernel_offsets(kernel)
            return _BinaryMorphology2dFn.apply(
                tensor,
                active_y,
                active_x,
                origin_,
                float(border_value_),
                op,
            )
        return _binary_morphology_inference(
            tensor=tensor,
            kernel=kernel,
            origin=origin_,
            border_value=float(border_value_),
            op=op,
        )

    active_y, active_x, active_w = _build_grayscale_kernel_data(
        kernel=kernel,
        structuring_element=structuring_element,
        max_val=max_val,
    )
    morph_sign = 1 if op == "dilation" else -1
    return _GrayscaleMorphology2dFn.apply(
        tensor,
        active_y,
        active_x,
        active_w,
        origin_,
        float(border_value_),
        256,
        morph_sign,
    )


def dilation(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[list[int] | tuple[int, int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "auto",
) -> torch.Tensor:
    return _morphology(
        tensor=tensor,
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
        op="dilation",
    )


def erosion(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[list[int] | tuple[int, int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "auto",
) -> torch.Tensor:
    return _morphology(
        tensor=tensor,
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
        op="erosion",
    )


def opening(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[list[int] | tuple[int, int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "auto",
) -> torch.Tensor:
    return dilation(
        erosion(
            tensor,
            kernel=kernel,
            structuring_element=structuring_element,
            origin=origin,
            border_type=border_type,
            border_value=border_value,
            max_val=max_val,
            engine=engine,
        ),
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
    )


def closing(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[list[int] | tuple[int, int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "auto",
) -> torch.Tensor:
    return erosion(
        dilation(
            tensor,
            kernel=kernel,
            structuring_element=structuring_element,
            origin=origin,
            border_type=border_type,
            border_value=border_value,
            max_val=max_val,
            engine=engine,
        ),
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
    )
