# edge_op.py
from __future__ import annotations

import math

import torch

try:
    import cupy as cp
    from numba import cuda
except:
    cp = None
    cuda = None

__all__ = ["edge_mean_sample"]


def torch_to_cupy(x: torch.Tensor) -> cp.ndarray:
    """Convert a CUDA Torch tensor to a contiguous CuPy array without a copy when possible.

    Args:
        x (torch.Tensor): CUDA Torch tensor to expose through DLPack. The tensor
            is made contiguous when necessary.

    Returns:
        cp.ndarray: CuPy view backed by the same CUDA allocation.

    Raises:
        ValueError: If ``x`` is not a CUDA tensor.
    """
    if not x.is_cuda:
        raise ValueError("Expected a CUDA tensor.")
    if not x.is_contiguous():
        x = x.contiguous()
    return cp.fromDlpack(torch.utils.dlpack.to_dlpack(x))


def cupy_to_torch(x: cp.ndarray) -> torch.Tensor:
    """Convert a CuPy array to a Torch tensor through DLPack.

    Args:
        x (cp.ndarray): CuPy array on a supported device.

    Returns:
        torch.Tensor: Torch tensor sharing the underlying device storage when
        supported by the DLPack bridge.
    """
    return torch.from_dlpack(x)


@cuda.jit(device=True, inline=True)
def bilinear_sample(img, x, y):
    """Sample a 2-D CUDA array at floating-point pixel coordinates by bilinear interpolation.

    Args:
        img (device array): 2-D image-like array with shape ``[H, W]``.
        x (float): Horizontal pixel coordinate.
        y (float): Vertical pixel coordinate.

    Returns:
        float: Bilinearly interpolated value. Coordinates outside the image
        bounds return ``0.0``.
    """
    H, W = img.shape
    if x < 0.0 or x > W - 1 or y < 0.0 or y > H - 1:
        return 0.0

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)

    tx = x - x0
    ty = y - y0

    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]

    v0 = (1.0 - tx) * v00 + tx * v10
    v1 = (1.0 - tx) * v01 + tx * v11
    return (1.0 - ty) * v0 + ty * v1


@cuda.jit
def edge_mean_sample_kernel(img, edges_xyxy, out):
    """Compute one bilinearly sampled mean along each supplied pixel-space edge.

    Args:
        img (device array): 2-D CUDA array with shape ``[H, W]``.
        edges_xyxy (device array): CUDA array with shape ``[E, 4]``; each row
            stores ``(x0, y0, x1, y1)`` in pixel coordinates.
        out (device array): Preallocated CUDA output array with shape ``[E]``.
    """
    e = cuda.grid(1)
    if e >= edges_xyxy.shape[0]:
        return

    x0 = edges_xyxy[e, 0]
    y0 = edges_xyxy[e, 1]
    x1 = edges_xyxy[e, 2]
    y1 = edges_xyxy[e, 3]

    dx = x1 - x0
    dy = y1 - y0

    n = int(max(abs(dx), abs(dy))) + 1
    if n < 1:
        n = 1

    acc = 0.0
    for i in range(n):
        t = 0.0 if n == 1 else i / (n - 1)
        x = x0 + t * dx
        y = y0 + t * dy
        acc += bilinear_sample(img, x, y)

    out[e] = acc / n


def edge_mean_sample(img: torch.Tensor, edges_xyxy: torch.Tensor) -> torch.Tensor:
    """Sample the mean value along multiple 2-D edges on the CUDA device.

    Args:
        img (torch.Tensor): CUDA Torch tensor with shape ``[H, W]``. Values are
            converted to ``torch.float32`` before sampling when needed.
        edges_xyxy (torch.Tensor): CUDA Torch tensor with shape ``[E, 4]``;
            each row stores ``(x0, y0, x1, y1)`` in pixel coordinates. Values
            are converted to ``torch.float32`` when needed.

    Returns:
        torch.Tensor: CUDA tensor with shape ``[E]`` containing the mean sampled
        value for each edge, returned as ``torch.float32``.

    Raises:
        ValueError: If ``img`` is not 2-D, ``edges_xyxy`` is not shaped ``[E, 4]``,
            or either input is not a CUDA tensor.
    """
    if img.ndim != 2:
        raise ValueError(f"img must be [H, W], got {tuple(img.shape)}")
    if edges_xyxy.ndim != 2 or edges_xyxy.shape[-1] != 4:
        raise ValueError(f"edges_xyxy must be [E, 4], got {tuple(edges_xyxy.shape)}")
    if not img.is_cuda or not edges_xyxy.is_cuda:
        raise ValueError("Both img and edges_xyxy must be CUDA tensors.")
    if img.dtype != torch.float32:
        img = img.float()
    if edges_xyxy.dtype != torch.float32:
        edges_xyxy = edges_xyxy.float()

    img_cp = torch_to_cupy(img)
    edges_cp = torch_to_cupy(edges_xyxy)
    out_cp = cp.empty((edges_xyxy.shape[0],), dtype=cp.float32)

    threads_per_block = 256
    blocks = (edges_xyxy.shape[0] + threads_per_block - 1) // threads_per_block

    edge_mean_sample_kernel[blocks, threads_per_block](img_cp, edges_cp, out_cp)

    # Keep synchronization here so the wrapper is safe for immediate consumption and timing.
    cp.cuda.runtime.deviceSynchronize()

    return cupy_to_torch(out_cp)
