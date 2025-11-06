"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2025-10-31 15:11:34
LastEditors: Jiaqi Gu (jiaqigu@asu.edu)
LastEditTime: 2025-10-31 15:24:08
FilePath: /MAPS_ilt/core/invdes/models/layers/parametrization/box.py
"""

from typing import Any, Dict

import torch
from torch import nn
from torch.types import Device

from .base_parametrization import BaseParametrization
from .geometry import BatchGeometry

__all__ = ["BoxParameterization", "Boxes"]


class DiffWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centers, sizes, build_fn, grad_fn):
        eps = build_fn()
        ctx.grad_fn = grad_fn
        ctx.save_for_backward(eps)
        return eps

    @staticmethod
    def backward(ctx, grad_output):
        grad_fn = ctx.grad_fn
        eps = ctx.saved_tensors[0]
        grad_centers, grad_sizes = grad_fn(grad_output, eps)
        return grad_centers, grad_sizes, None, None


def fill_rectangles_thresh_um(
    boxes_um: torch.Tensor,  # [N,4]: (xmin, xmax, ymin, ymax) in µm
    region_center_um: tuple,  # (cx, cy) in µm
    region_size_um: tuple,  # (sx, sy) in µm
    grid_step_um: float,  # µm per pixel (e.g., 0.01 => 10 nm)
    color,  # scalar or [C]  (C=1 or 3)
    coverage_thresh: float = 0.5,  # fraction of pixel area that must be covered (> threshold)
    alpha: float = 1.0,
    img: torch.Tensor | None = None,  # optional existing canvas [H,W] or [C,H,W]
    y_up: bool = True,  # True: +y points up (Cartesian); False: image coords (+y down)
    device: str | torch.device = None,
    dtype: torch.dtype | None = None,
):
    """
    Exact area-based thresholding: a pixel (cell) of size g×g is filled if there exists a rectangle
    whose overlap area with the pixel is > coverage_thresh * g*g.

    Memory: O(HW) for the canvas + small per-rectangle temporaries (no [N,H,W] allocations).
    """
    g = float(grid_step_um)
    cx, cy = float(region_center_um[0]), float(region_center_um[1])
    sx, sy = float(region_size_um[0]), float(region_size_um[1])

    # Create/normalize canvas
    if img is None:
        W = int(round(sx / g))
        H = int(round(sy / g))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float32
        canvas = torch.zeros(1, H, W, device=device, dtype=dtype)
    else:
        canvas = img if img.ndim == 3 else img.unsqueeze(0)
        if device is None:
            device = canvas.device
        if dtype is None:
            dtype = canvas.dtype

    C, W, H = canvas.shape
    dev, dt = device, dtype

    if boxes_um.numel() == 0:
        return (
            img.clone() if img is not None else (canvas if C > 1 else canvas.squeeze(0))
        )

    # Physical canvas bounds
    x_left = cx - sx / 2.0
    x_right = cx + sx / 2.0
    y_bottom = cy - sy / 2.0
    y_top = cy + sy / 2.0

    # Target “on” mask (we OR rectangles that pass the threshold)
    on = torch.zeros((W, H), device=dev, dtype=torch.bool)

    # Helper to compute 1D overlap length between many pixel-centered intervals and a single [a,b]
    # For pixel centers u and cell half-size g/2, the pixel interval is [u - g/2, u + g/2].
    def overlap_1d(centers: torch.Tensor, a: float, b: float, half: float):
        # returns length of intersection in [0, g]
        left = torch.maximum(
            centers - half, torch.tensor(a, device=centers.device, dtype=centers.dtype)
        )
        right = torch.minimum(
            centers + half, torch.tensor(b, device=centers.device, dtype=centers.dtype)
        )
        return torch.clamp(right - left, min=0.0, max=2 * half)

    half = g * 0.5
    area_thresh = coverage_thresh * (g * g)

    # Process each rectangle; per-rect work is limited to its pixel span
    # b = boxes_um.to(device=dev, dtype=torch.float64)
    b = boxes_um.flatten(0, -2)
    xmin, xmax = torch.minimum(b[:, 0], b[:, 1]), torch.maximum(b[:, 0], b[:, 1])
    ymin, ymax = torch.minimum(b[:, 2], b[:, 3]), torch.maximum(b[:, 2], b[:, 3])

    # Clip to canvas to avoid empty/huge spans
    xmin = torch.clamp(xmin, x_left, x_right)
    xmax = torch.clamp(xmax, x_left, x_right)
    ymin = torch.clamp(ymin, y_bottom, y_top)
    ymax = torch.clamp(ymax, y_bottom, y_top)

    for i in range(b.shape[0]):
        ax, bx = float(xmin[i]), float(xmax[i])
        ay, by = float(ymin[i]), float(ymax[i])
        if ax >= bx or ay >= by:
            continue  # fully outside or degenerate after clipping

        # Compute pixel index ranges touched by this rectangle (conservative)
        # We consider pixels whose centers could yield >0 overlap.
        # Column indices 0..W-1 have centers at: x_left + (j+0.5)*g
        # Any pixel whose center is within [a - g/2, b + g/2] can have nonzero overlap.
        j0 = int(
            max(0, min(W - 1, int((ax - x_left) / g - 0.5)))
        )  # floor((a - x_left)/g - 0.5)
        j1 = int(
            max(0, min(W - 1, int((bx - x_left) / g + 0.5)))
        )  # ceil((b - x_left)/g + 0.5) - 1
        if j1 < j0:
            continue

        if y_up:
            # row r center at y_top - (r+0.5)g
            # Nonzero overlap if center in [ay - g/2, by + g/2]
            # Translate to r indices accordingly.
            # y_center(r) = y_top - (r+0.5)g ∈ [ay - g/2, by + g/2]
            # => r ∈ [ (y_top - (by + g/2))/g - 0.5 , (y_top - (ay - g/2))/g - 0.5 ]
            r0 = int(max(0, min(H - 1, int((y_top - (by + half)) / g - 0.5))))
            r1 = int(max(0, min(H - 1, int((y_top - (ay - half)) / g - 0.5))))
        else:
            # row r center at y_bottom + (r+0.5)g
            r0 = int(max(0, min(H - 1, int(((ay - half) - y_bottom) / g - 0.5))))
            r1 = int(max(0, min(H - 1, int(((by + half) - y_bottom) / g - 0.5))))
        if r1 < r0:
            continue

        # Build 1D vectors of pixel centers in this subgrid (double precision for accurate thresholds)
        x_centers = (
            x_left
            + (torch.arange(j0, j1 + 1, device=dev, dtype=torch.float64) + 0.5) * g
        )
        if y_up:
            y_centers = (
                y_top
                - (torch.arange(r0, r1 + 1, device=dev, dtype=torch.float64) + 0.5) * g
            )
        else:
            y_centers = (
                y_bottom
                + (torch.arange(r0, r1 + 1, device=dev, dtype=torch.float64) + 0.5) * g
            )

        # Exact 1D overlaps (lengths in µm), bounded to [0, g]
        ox = overlap_1d(x_centers, ax, bx, half)  # [W_sub]
        oy = overlap_1d(y_centers, ay, by, half)  # [H_sub]

        if not (ox.any() and oy.any()):
            continue

        # Per-pixel overlap area via outer product: area = ox[j] * oy[r]
        # Compare to threshold; set mask region
        A = torch.outer(ox, oy)  # [W_sub, H_sub]
        on[j0 : j1 + 1, r0 : r1 + 1] |= A > area_thresh

    # Composite onto canvas
    mask = on.unsqueeze(0).to(canvas.dtype)  # [1,W,H]
    col = torch.as_tensor(color, device=dev, dtype=dt).view(-1, 1, 1)
    if col.shape[0] == 1 and C > 1:
        col = col.repeat(C, 1, 1)
    elif col.shape[0] not in (1, C):
        raise ValueError(f"Color channel mismatch: {col.shape[0]} vs canvas {C}")

    if alpha >= 1.0:
        out = torch.where(mask.bool(), col, canvas)
    else:
        out = canvas * (1 - alpha * mask) + col * (alpha * mask)

    return (
        out
        if (img is not None and img.ndim == 3) or (img is None and C > 1)
        else out.squeeze(0)
    )


def _overlap_1d_centers(centers: torch.Tensor, a: float, b: float, g: float):
    """
    centers: [K] physical centers along one axis (µm)
    interval: [a,b] (µm), with a <= b
    g: pixel size (µm)
    returns: [K] overlap lengths in µm, each clamped to [0, g]
    """
    half = g * 0.5
    left = torch.maximum(
        centers - half, torch.tensor(a, device=centers.device, dtype=centers.dtype)
    )
    right = torch.minimum(
        centers + half, torch.tensor(b, device=centers.device, dtype=centers.dtype)
    )
    return torch.clamp(right - left, min=0.0, max=g)


def rect_edge_grads_um(
    boxes_um: torch.Tensor,  # [N,4] => (xmin, xmax, ymin, ymax) in µm
    grad_eps: torch.Tensor,  # [H,W] or [C,H,W] (will be reduced to scalar per pixel)
    center_um: tuple,  # (cx, cy) in µm
    size_um: tuple,  # (sx, sy) in µm
    grid_step_um: float,  # µm per pixel
    y_up: bool = True,  # Cartesian +y up (True) vs image +y down (False)
):
    """
    Returns:
      edge_grads: [N,4] tensor with gradients [d/dxmin, d/dxmax, d/dymin, d/dymax]
      cxcy_sxsy_grads: [N,4] tensor with gradients [d/dcx, d/dcy, d/dsx, d/dsy]

    Notes:
      - Uses exact overlap lengths along the axis orthogonal to each edge (in µm).
      - Samples grad_eps at the pixel column/row where the edge lies.
      - Works on CPU or GPU; loops over boxes but with vectorized inner ops per edge span.
    """
    g = float(grid_step_um)
    cx, cy = float(center_um[0]), float(center_um[1])
    sx, sy = float(size_um[0]), float(size_um[1])

    # Canonicalize grad_eps to [H,W], scalar field
    grad_eps = grad_eps.t()
    H, W = grad_eps.shape
    dev = grad_eps.device
    f64 = torch.float64

    x_left = cx - sx / 2.0
    x_right = cx + sx / 2.0
    y_bottom = cy - sy / 2.0
    y_top = cy + sy / 2.0

    boxes = boxes_um.flatten(0, -2)  # [N,4]
    xmin = torch.minimum(boxes[:, 0], boxes[:, 1])
    xmax = torch.maximum(boxes[:, 0], boxes[:, 1])
    ymin = torch.minimum(boxes[:, 2], boxes[:, 3])
    ymax = torch.maximum(boxes[:, 2], boxes[:, 3])

    # Clip to canvas bounds to avoid empty spans
    xmin = xmin.clamp(x_left, x_right)
    xmax = xmax.clamp(x_left, x_right)
    ymin = ymin.clamp(y_bottom, y_top)
    ymax = ymax.clamp(y_bottom, y_top)

    N = boxes.shape[0]
    edge_grads = torch.zeros((N, 4), device=dev)  # [d_xmin, d_xmax, d_ymin, d_ymax]

    # Precompute coordinate helpers
    # Column j center: x_left + (j+0.5)*g
    # Row    r center: y_top - (r+0.5)*g  if y_up else y_bottom + (r+0.5)*g
    for i in range(N):
        ax, bx = float(xmin[i]), float(xmax[i])
        ay, by = float(ymin[i]), float(ymax[i])
        if ax >= bx or ay >= by:
            continue  # degenerate/empty after clipping

        # ---------- Vertical edges (x = xmin, xmax) ----------
        # Rows potentially intersected by the rectangle ± half pixel:
        # y-centers within [ay - g/2, by + g/2]
        if y_up:
            r0 = int(max(0, min(H - 1, int((y_top - (by + g * 0.5)) / g - 0.5))))
            r1 = int(max(0, min(H - 1, int((y_top - (ay - g * 0.5)) / g - 0.5))))
            y_centers = (
                y_top - (torch.arange(r0, r1 + 1, device=dev, dtype=f64) + 0.5) * g
            )
        else:
            r0 = int(max(0, min(H - 1, int(((ay - g * 0.5) - y_bottom) / g - 0.5))))
            r1 = int(max(0, min(H - 1, int(((by + g * 0.5) - y_bottom) / g - 0.5))))
            y_centers = (
                y_bottom + (torch.arange(r0, r1 + 1, device=dev, dtype=f64) + 0.5) * g
            )

        oy = _overlap_1d_centers(y_centers, ay, by, g)  # [H_sub], µm
        if oy.numel() > 0 and oy.max() > 0:
            # Columns containing the edges
            j_min = int(max(0, min(W - 1, int((ax - x_left) / g))))
            j_max = int(max(0, min(W - 1, int((bx - x_left) / g))))
            gy_min = grad_eps[r0 : r1 + 1, j_min]  # [H_sub]
            gy_max = grad_eps[r0 : r1 + 1, j_max]  # [H_sub]

            d_xmin = -(gy_min * oy).sum()  # minus sign: increasing xmin shrinks area
            d_xmax = (gy_max * oy).sum()  # plus sign: increasing xmax grows area

            edge_grads[i, 0] = d_xmin
            edge_grads[i, 1] = d_xmax

        # ---------- Horizontal edges (y = ymin, ymax) ----------
        # Columns potentially intersected: x-centers within [ax - g/2, bx + g/2]
        j0 = int(max(0, min(W - 1, int((ax - x_left) / g - 0.5))))
        j1 = int(max(0, min(W - 1, int((bx - x_left) / g + 0.5))))
        x_centers = x_left + (torch.arange(j0, j1 + 1, device=dev) + 0.5) * g
        ox = _overlap_1d_centers(x_centers, ax, bx, g)  # [W_sub], µm

        if ox.numel() > 0 and ox.max() > 0:
            if y_up:
                r_min = int(
                    max(0, min(H - 1, int((y_top - ay) / g)))
                )  # row containing ymin
                r_max = int(
                    max(0, min(H - 1, int((y_top - by) / g)))
                )  # row containing ymax
            else:
                r_min = int(max(0, min(H - 1, int((ay - y_bottom) / g))))
                r_max = int(max(0, min(H - 1, int((by - y_bottom) / g))))

            gx_min = grad_eps[r_min, j0 : j1 + 1]  # [W_sub]
            gx_max = grad_eps[r_max, j0 : j1 + 1]  # [W_sub]

            d_ymin = -(gx_min * ox).sum()  # increasing ymin shrinks area
            d_ymax = (gx_max * ox).sum()  # increasing ymax grows area

            edge_grads[i, 2] = d_ymin
            edge_grads[i, 3] = d_ymax

    # Convert edge grads -> center/size grads
    # xmin = cx - sx/2,  xmax = cx + sx/2  => dcx = d_xmin + d_xmax,  dsx = 0.5*d_xmax - 0.5*d_xmin
    # ymin = cy - sy/2,  ymax = cy + sy/2  => dcy = d_ymin + d_ymax,  dsy = 0.5*d_ymax - 0.5*d_ymin
    return edge_grads[:, 0], edge_grads[:, 1], edge_grads[:, 2], edge_grads[:, 3]


class Boxes(BatchGeometry):
    def __init__(
        self,
        sim_cfg: Dict[str, Any],
        geometry_cfgs,
        region_name: str = "design_region_1",
        design_region_mask=None,
        design_region_cfg=None,
        operation_device: Device = torch.device("cuda:0"),
    ):
        super().__init__(
            sim_cfg,
            geometry_cfgs,
            region_name=region_name,
            design_region_mask=design_region_mask,
            design_region_cfg=design_region_cfg,
            operation_device=operation_device,
        )

        ## centers [x, y], sizes [size_x, size_y]
        self.grid_step = 1 / self.sim_cfg["resolution"]

    def compute_derivatives(self, grad_eps: torch.Tensor, eps: torch.Tensor):
        # grad_eps: (H, W) design region gradient
        # eps: (H, W) design region permittivity
        ## we use 4 edges' gradient to compute the gradients w.r.t. center and size
        xmin = self.centers[..., 0] - self.sizes[..., 0] / 2
        xmax = self.centers[..., 0] + self.sizes[..., 0] / 2
        ymin = self.centers[..., 1] - self.sizes[..., 1] / 2
        ymax = self.centers[..., 1] + self.sizes[..., 1] / 2
        boxes_um = torch.stack([xmin, xmax, ymin, ymax], dim=-1)  # [N,4]
        region_size = self.design_region_cfg[
            "size"
        ]  # [reg_size_x, reg_size_y] in um unit
        region_center = self.design_region_cfg[
            "center"
        ]  # [reg_center_x, reg_center_y] in um unit
        # print(grad_eps.shape, eps_grad_x.shape)
        ## assume dL/deps on the right edge is negative, it means we should increase the permittivity there
        ## here the rectangle is assumed to have eps_inner = 1, eps_outer = 0
        ## then the right edge should move to the right to increase the permittivity
        ## then the grad_xmax should be negative, same sign as grad_eps
        ## for grad_xmin, it should have opposite sign as grad_eps
        ## this is handled in the rect_edge_grads_um function
        grad_xmin, grad_xmax, grad_ymin, grad_ymax = rect_edge_grads_um(
            boxes_um=boxes_um,
            grad_eps=grad_eps,
            center_um=region_center,
            size_um=region_size,
            grid_step_um=self.grid_step,
            y_up=True,
        )

        grad_centers = torch.stack(
            [
                grad_xmax - grad_xmin,
                grad_ymax - grad_ymin,
            ],
            dim=-1,
        )  # d/dcx, d/dcy

        grad_sizes = torch.stack(
            [
                0.5 * (grad_xmax + grad_xmin),
                0.5 * (grad_ymax + grad_ymin),
            ],
            dim=-1,
        )  # d/dsx, d/dsy
        return grad_centers.reshape_as(self.centers).zero_(), grad_sizes.reshape_as(
            self.sizes
        )

    def build_pattern(
        self,
        permittivity: torch.Tensor = None,
    ) -> torch.Tensor:
        # return a normalized/boolean pattern where we fill 1 inside the geometry and 0 outside
        # we can add various constraints here, do not need to worry about the differentiability of the function
        # eps_bg = self.design_region_cfg["eps_bg"]
        # eps_r = self.design_region_cfg["eps"]

        eps_r = 1.0  # inside geometry, we use 1.0 as normalized permittivity

        region_px_size = [(m.stop - m.start) for m in self.design_region_mask]
        eps = (
            torch.zeros(region_px_size, device=self.operation_device)
            if permittivity is None
            else permittivity
        )

        region_center = self.design_region_cfg["center"]
        region_size = self.design_region_cfg["size"]
        n_boxes = self.centers.shape[0:2]  # Nx, Ny
        ## uniform spread the box centers in the design region using meshgrid
        border = 0.2
        max_box_size_x = (region_size[0] - border * 2) / n_boxes[0]
        max_box_size_y = (region_size[1] - border * 2) / n_boxes[1]
        self.sizes.data.clamp_(min=0.0)  # ensure sizes are non-negative
        self.sizes.data[..., 0].clamp_(max=max_box_size_x)
        self.sizes.data[..., 1].clamp_(max=max_box_size_y)

        xmin = self.centers[..., 0] - self.sizes[..., 0] / 2
        xmax = self.centers[..., 0] + self.sizes[..., 0] / 2
        ymin = self.centers[..., 1] - self.sizes[..., 1] / 2
        ymax = self.centers[..., 1] + self.sizes[..., 1] / 2
        boxes_um = torch.stack([xmin, xmax, ymin, ymax], dim=-1)  # [N,4]
        region_size = self.design_region_cfg[
            "size"
        ]  # [reg_size_x, reg_size_y] in um unit
        region_center = self.design_region_cfg[
            "center"
        ]  # [reg_center_x, reg_center_y] in um unit

        eps = fill_rectangles_thresh_um(
            boxes_um,
            region_center,
            region_size,
            self.grid_step,
            color=eps_r,
            alpha=1.0,
            img=eps,
            device=self.operation_device,
        )
        # import matplotlib.pyplot as plt
        # plt.imshow(eps.cpu().numpy())
        # plt.savefig("debug_box.png")
        # plt.close()
        # exit(0)

        return eps

    def forward(self, permittivity: torch.Tensor = None) -> torch.Tensor:
        return DiffWrapper.apply(
            self.centers,
            self.sizes,
            lambda eps=permittivity: self.build_pattern(permittivity=eps),
            self.compute_derivatives,
        )


class BoxParameterization(BaseParametrization):
    def __init__(
        self,
        *args,
        cfgs: dict = dict(
            method="box",
            geometry_cfgs=dict(
                batch_dims=(16, 16),  # Nx, Ny number of boxes in x and y directions
                size_dim=2,  # number of parameters to determine the size, default 2, e.g., 2 for box (size_x, size_y)
            ),
            transform=[],
            init_method="2d_array",
            denorm_mode="linear_eps",
        ),
        **kwargs,
    ):
        super().__init__(*args, cfgs=cfgs, **kwargs)

        method = cfgs["method"]

        self.register_parameter_build_per_region_fn(method, self._build_parameters_box)
        self.register_parameter_reset_per_region_fn(method, self._reset_parameters_box)

        self.build_parameters(cfgs, self.design_region_cfg)
        self.reset_parameters(cfgs, self.design_region_cfg)

    def _prepare_parameters_box(
        self,
    ):
        param_dict = dict()

        return param_dict

    def _build_parameters_box(self, param_cfg, region_cfg):
        param_dict = self._prepare_parameters_box()
        boxes = Boxes(
            sim_cfg=self.hr_device.sim_cfg,
            geometry_cfgs=param_cfg["geometry_cfgs"],
            region_name=self.region_name,
            design_region_mask=self.hr_design_region_mask,
            design_region_cfg=self.hr_device.design_region_cfgs[self.region_name],
            operation_device=self.operation_device,
        )

        weight_dict = dict(boxes=boxes)
        return weight_dict, param_dict

    def _reset_parameters_box(
        self, weight_dict, param_cfg, region_cfg, init_method: str = "2d_array"
    ):
        if init_method.startswith("2d_array"):
            density = param_cfg.get("density", 0.5)
            region_center = region_cfg["center"]
            region_size = region_cfg["size"]
            n_boxes = weight_dict["boxes"].centers.shape[0:2]  # Nx, Ny
            ## uniform spread the box centers in the design region using meshgrid
            border = 0.2
            max_box_size_x = (region_size[0] - border * 2) / n_boxes[0]
            max_box_size_y = (region_size[1] - border * 2) / n_boxes[1]

            weight_dict["boxes"].centers.data = torch.stack(
                torch.meshgrid(
                    torch.linspace(
                        region_center[0]
                        - region_size[0] / 2
                        + max_box_size_x / 2
                        + border,
                        region_center[0]
                        + region_size[0] / 2
                        - max_box_size_x / 2
                        - border,
                        steps=n_boxes[0],
                        device=self.operation_device,
                    ),
                    torch.linspace(
                        region_center[1]
                        - region_size[1] / 2
                        + max_box_size_y / 2
                        + border,
                        region_center[1]
                        + region_size[1] / 2
                        - max_box_size_y / 2
                        - border,
                        steps=n_boxes[1],
                        device=self.operation_device,
                    ),
                    indexing="ij",
                ),
                dim=-1,
            )
            weight_dict["boxes"].sizes.data[..., 0].uniform_(
                max_box_size_x / 3, max_box_size_x / 2
            )
            weight_dict["boxes"].sizes.data[..., 1].uniform_(
                max_box_size_y / 3, max_box_size_y / 2
            )
            weight_dict["boxes"].sizes.data = weight_dict["boxes"].sizes.data.to(
                device=self.operation_device
            )
            ## randomly set 1-density boxes to zero size
            if density < 1.0:
                random_matrix = torch.rand(n_boxes, device=self.operation_device)
                mask = random_matrix > density
                weight_dict["boxes"].sizes.data.masked_fill_(mask.unsqueeze(-1), 0.0)
        else:
            raise ValueError(f"Unsupported initialization method: {init_method}")

    def _build_permittivity(self, weights, sharpness: float):
        boxes: Boxes = weights["boxes"]
        eps = boxes.forward()

        return eps

    def build_permittivity(self, weights, sharpness: float | None = None):
        ## this is the high resolution, e.g., res=200, 310 permittivity
        ## return:
        #   1. we need the first one for gds dump out
        #   2. we need the second one for evaluation, do not need to downsample it here. transform will handle it.
        hr_permittivity = self._build_permittivity(
            weights,
            sharpness,
        )

        return hr_permittivity