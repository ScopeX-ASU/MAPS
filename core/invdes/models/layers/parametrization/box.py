"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2025-10-31 15:11:34
LastEditors: Jiaqi Gu (jiaqigu@asu.edu)
LastEditTime: 2025-10-31 15:24:08
FilePath: /MAPS_ilt/core/invdes/models/layers/parametrization/box.py
"""

from typing import Any, Callable, Dict

import numpy as np
import torch
from torch import nn
from torch.types import Device

from .base_parametrization import BaseParametrization
from .geometry import BatchGeometry
from .material_projection import EffectiveMediumProjector

__all__ = ["BoxParameterization", "Boxes"]


def _gpu_raster_to_torch_xy(raster: np.ndarray) -> np.ndarray:
    """Convert the rasterizer's ascending-y array to Torch's Cartesian layout.

    Args:
        raster: CuPy/NumPy raster with shape ``(nx, ny, nz)``. The local GPU
            rasterizer indexes ``y`` from the lower boundary upward, whereas
            the Torch rectangle rasterizer uses ``y_up=True`` and indexes its
            second axis from the upper boundary downward.

    Returns:
        NumPy array with the second axis reversed and contiguous storage.

    Raises:
        ValueError: If ``raster`` is not a three-dimensional array.
    """
    raster = np.asarray(raster)
    if raster.ndim != 3:
        raise ValueError(
            "Expected a three-dimensional GPU raster (nx, ny, nz), "
            f"got shape {raster.shape}."
        )
    return np.flip(raster, axis=1).copy()


class DiffWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centers, sizes, material_values, build_fn, grad_fn):
        eps = build_fn()
        ctx.grad_fn = grad_fn
        ctx.save_for_backward(eps)
        return eps

    @staticmethod
    def backward(ctx, grad_output):
        grad_fn = ctx.grad_fn
        eps = ctx.saved_tensors[0]
        grad_centers, grad_sizes, grad_material = grad_fn(grad_output, eps)
        return grad_centers, grad_sizes, grad_material, None, None


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
    """Fill a canvas where rectangle overlap with a pixel exceeds a coverage threshold.

    Args:
        boxes_um (torch.Tensor): Rectangles with trailing coordinates ``(xmin, xmax, ymin, ymax)``
            in micrometers; leading dimensions are flattened to a rectangle list.
        region_center_um (tuple[float, float]): Physical canvas center in micrometers.
        region_size_um (tuple[float, float]): Physical canvas size ``(sx, sy)`` in micrometers.
        grid_step_um (float): Pixel/cell size in micrometers.
        color (float | Sequence[float] | torch.Tensor): Scalar, channel vector,
            or tensor with one value per rectangle used for filled pixels.
        coverage_thresh (float): Minimum fractional pixel-area coverage; the implementation uses a
            strict ``>`` comparison. Defaults to 0.5.
        alpha (float): Blend factor applied to the filled mask. Values at least 1 use replacement;
            lower values blend with the canvas. Defaults to 1.0.
        img (torch.Tensor | None): Optional existing canvas with shape ``[H, W]`` or ``[C, H, W]``.
            Defaults to None.
        y_up (bool): Whether image rows map to Cartesian ``+y`` upward rather than image ``+y``
            downward. Defaults to True.
        device (str | torch.device | None): Device for an automatically created canvas. Defaults to None;
            an omitted device falls back to CUDA when available, otherwise CPU.
        dtype (torch.dtype | None): Dtype for an automatically created canvas, or for the color tensor
            when an existing canvas is supplied. Defaults to None; an omitted dtype falls back to
            ``torch.float32`` for a new canvas or the existing canvas dtype.

    Returns:
        torch.Tensor: Canvas after rectangle compositing. A single-channel newly created canvas is
        squeezed back to ``[H, W]``; otherwise the channel dimension is preserved.

    Notes:
        Rectangle spans are clipped to the physical canvas. Pixel overlap is computed as the product
        of two exact 1-D overlap lengths, avoiding an ``[N, H, W]`` temporary.
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

    # Target “on” mask (we OR rectangles that pass the threshold). When color
    # has one value per rectangle, owner records the last covering rectangle.
    on = torch.zeros((W, H), device=dev, dtype=torch.bool)
    owner = None

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
    color_tensor = torch.as_tensor(color, device=dev, dtype=dt).flatten()
    color_shape = torch.as_tensor(color).shape
    per_box_color = color_tensor.numel() == b.shape[0] and (
        b.shape[0] != C or len(color_shape) > 1
    )
    if color_tensor.numel() != 1 and not per_box_color and color_tensor.numel() != C:
        raise ValueError(
            "Color must be a scalar, a channel vector, or have one value for each "
            f"rectangle; got {color_tensor.numel()} values for {b.shape[0]} rectangles."
        )
    if per_box_color and b.shape[0] > 1:
        owner = torch.full((W, H), -1, device=dev, dtype=torch.long)
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
        box_on = A > area_thresh
        on[j0 : j1 + 1, r0 : r1 + 1] |= box_on
        if owner is not None:
            owner[j0 : j1 + 1, r0 : r1 + 1] = torch.where(
                box_on,
                torch.as_tensor(i, device=dev, dtype=torch.long),
                owner[j0 : j1 + 1, r0 : r1 + 1],
            )

    # Composite onto canvas
    mask = on.unsqueeze(0).to(canvas.dtype)  # [1,W,H]
    if owner is not None:
        owner_clamped = owner.clamp_min(0)
        per_pixel_color = color_tensor[owner_clamped].unsqueeze(0)
        if alpha >= 1.0:
            out = torch.where(mask.bool(), per_pixel_color, canvas)
        else:
            out = canvas * (1 - alpha * mask) + per_pixel_color * (alpha * mask)
    else:
        col = color_tensor.view(-1, 1, 1)
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
    """Compute physical overlap between pixel-centered cells and a 1-D interval.

    Args:
        centers (torch.Tensor): One-dimensional physical pixel centers.
        a (float): Interval start coordinate.
        b (float): Interval end coordinate.
        g (float): Pixel size in the same physical units as ``centers``, ``a``, and ``b``.

    Returns:
        torch.Tensor: Per-center overlap lengths, clamped to ``[0, g]``.
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
    """Estimate rectangle edge gradients from a rasterized permittivity gradient.

    Args:
        boxes_um (torch.Tensor): Rectangles with trailing coordinates ``(xmin, xmax, ymin, ymax)``
            in micrometers; leading dimensions are flattened to ``[N, 4]``.
        grad_eps (torch.Tensor): Raster gradient expected as ``[H, W]`` or a shape whose
            transpose yields the scalar ``[H, W]`` field used by the implementation.
        center_um (tuple[float, float]): Physical design-region center in micrometers.
        size_um (tuple[float, float]): Physical design-region size in micrometers.
        grid_step_um (float): Pixel size in micrometers.
        y_up (bool): Whether rows map to Cartesian ``+y`` upward. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Per-box gradients
        with respect to ``xmin``, ``xmax``, ``ymin``, and ``ymax``. Exact orthogonal edge
        overlap lengths weight sampled raster gradients and are normalized by one pixel
        area for the area-fraction raster.
    """
    g = float(grid_step_um)
    cx, cy = float(center_um[0]), float(center_um[1])
    sx, sy = float(size_um[0]), float(size_um[1])

    # Canonicalize grad_eps to [H,W], scalar field
    grad_eps = grad_eps.t()
    H, W = grad_eps.shape
    dev = grad_eps.device
    f64 = torch.float64
    # The raster stores an area fraction.  An edge overlap length therefore
    # contributes to the fraction derivative after division by one pixel area.
    edge_scale = 1.0 / (g * g)

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
    # Row    r center: y_top - (r+0.5)g  if y_up else y_bottom + (r+0.5)g
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

            # The raster stores an area fraction, so convert the overlap
            # length into a density derivative by the pixel area.
            d_xmin = -(gy_min * oy).sum() * edge_scale
            d_xmax = (gy_max * oy).sum() * edge_scale

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

            d_ymin = -(gx_min * ox).sum() * edge_scale
            d_ymax = (gx_max * ox).sum() * edge_scale

            edge_grads[i, 2] = d_ymin
            edge_grads[i, 3] = d_ymax

    # Convert edge grads -> center/size grads
    # xmin = cx - sx/2,  xmax = cx + sx/2  => dcx = d_xmin + d_xmax,  dsx = 0.5*d_xmax - 0.5*d_xmin
    # ymin = cy - sy/2,  ymax = cy + sy/2  => dcy = d_ymin + d_ymax,  dsy = 0.5*d_ymax - 0.5*d_ymin
    return edge_grads[:, 0], edge_grads[:, 1], edge_grads[:, 2], edge_grads[:, 3]


class Boxes(BatchGeometry):
    """Parameterize a batch of axis-aligned rectangular boxes in a 2-D design region.

    The box centers and sizes are learnable tensors inherited through ``BatchGeometry``.
    Rasterization uses normalized inside-box permittivity ``1.0`` by default. With
    material optimization enabled, each box owns a normalized material variable
    during the material and joint stages.
    """

    def __init__(
        self,
        sim_cfg: Dict[str, Any],
        geometry_cfgs,
        region_name: str = "design_region_1",
        design_region_mask=None,
        design_region_cfg=None,
        operation_device: Device = torch.device("cuda:0"),
    ):
        """Initialize batched box geometry state.

        Args:
            sim_cfg (dict[str, Any]): Simulation configuration containing ``resolution``.
            geometry_cfgs (dict[str, Any]): Box geometry configuration passed to the base
                geometry class.
            region_name (str): Design-region identifier. Defaults to ``\"design_region_1\"``.
            design_region_mask (Any | None): Optional design-region index mask used to determine
                the raster shape. Defaults to None.
            design_region_cfg (dict[str, Any] | None): Physical design-region configuration.
                Defaults to None.
            operation_device (torch.device): Device for learnable centers and sizes.
                Defaults to ``torch.device(\"cuda:0\")``.

        Attributes:
            grid_step (float): Physical pixel spacing derived as ``1 / sim_cfg[\"resolution\"]``.
        """
        super().__init__(
            sim_cfg,
            geometry_cfgs,
            region_name=region_name,
            design_region_mask=design_region_mask,
            design_region_cfg=design_region_cfg,
            operation_device=operation_device,
        )

        ## centers [x, y], sizes [size_x, size_y]
        phi_resolution = geometry_cfgs.get("phi_resolution", self.sim_cfg["resolution"])
        if isinstance(phi_resolution, (int, float)):
            phi_resolution = (float(phi_resolution), float(phi_resolution))
        else:
            phi_resolution = tuple(float(value) for value in phi_resolution)
        if len(phi_resolution) != 2 or any(value <= 0 for value in phi_resolution):
            raise ValueError(
                "phi_resolution must contain two positive values in pixels per um"
            )
        if phi_resolution[0] != phi_resolution[1]:
            raise ValueError(
                "Boxes currently require an isotropic phi_resolution because the "
                "rectangle rasterizer uses one scalar grid step."
            )
        self.phi_resolution = phi_resolution
        region_size = self.design_region_cfg["size"]
        self.phi_shape = tuple(
            max(1, int(round(float(region_size[axis]) * phi_resolution[axis])))
            for axis in range(2)
        )
        self.grid_step = 1 / phi_resolution[0]
        self.min_size_um = self._parse_size_bound(
            geometry_cfgs.get("min_size_um", 0.0), "min_size_um"
        )
        configured_max_size = geometry_cfgs.get("max_size_um", None)
        self.max_size_um = (
            None
            if configured_max_size is None
            else self._parse_size_bound(configured_max_size, "max_size_um")
        )
        self.small_box_policy = str(
            geometry_cfgs.get("small_box_policy", "remove")
        ).lower()
        valid_small_box_policies = {
            "remove",
            "clamp",
            "snap_by_area",
            "snap_by_min_side",
        }
        if self.small_box_policy not in valid_small_box_policies:
            raise ValueError(
                "small_box_policy must be 'remove', 'clamp', 'snap_by_area', "
                "or 'snap_by_min_side'"
            )
        self.share_size = bool(geometry_cfgs.get("share_size", False))
        if self.small_box_policy != "remove":
            effective_max = self._max_box_sizes()
            if any(
                minimum > maximum
                for minimum, maximum in zip(self.min_size_um, effective_max.tolist())
            ):
                raise ValueError(
                    "min_size_um cannot exceed the effective size maximum for "
                    "clamping or snapping policies"
                )
            if self.share_size and max(self.min_size_um) > min(effective_max.tolist()):
                raise ValueError(
                    "shared box sizing requires a nonempty intersection between "
                    "the x and y size bounds"
                )
        self.use_gpu_subpixel = bool(geometry_cfgs.get("gpu_subpixel", False))
        self._gpu_subpixel_grid = None
        self._gpu_subpixel_rasterizer = None
        self.material_projector = EffectiveMediumProjector(
            geometry_cfgs.get("material_model", "linear_eps")
        )

    def uniform_parameter_grid_coords(self):
        """Return cell-center coordinates of the internal uniform box grid."""
        center = self.design_region_cfg["center"]
        size = self.design_region_cfg["size"]
        return tuple(
            torch.linspace(
                float(center[axis]) - float(size[axis]) / 2 + self.grid_step / 2,
                float(center[axis]) + float(size[axis]) / 2 - self.grid_step / 2,
                self.phi_shape[axis],
                device=self.centers.device,
                dtype=self.centers.dtype,
            )
            for axis in range(2)
        )

    @staticmethod
    def _parse_size_bound(value, name):
        """Normalize a scalar or two-axis size bound to ``(x, y)``."""
        if isinstance(value, (int, float)):
            bound = (float(value), float(value))
        else:
            try:
                bound = tuple(float(axis_value) for axis_value in value)
            except TypeError as error:
                raise ValueError(
                    f"{name} must be a nonnegative scalar or a two-element sequence"
                ) from error
            if len(bound) != 2:
                raise ValueError(
                    f"{name} must be a nonnegative scalar or a two-element sequence"
                )
        if any(axis_value < 0 for axis_value in bound):
            raise ValueError(f"{name} must be nonnegative")
        return bound

    def _material_values(self):
        """Return the normalized value rasterized inside each box."""
        if not self.material_enabled:
            return None
        if self.optimization_stage in {"material", "joint"}:
            return self.material_rho
        return self.material_state

    def _max_box_sizes(self):
        """Return configured and lattice-derived per-axis size upper bounds."""
        lattice_max = self._lattice_max_box_sizes()
        if self.max_size_um is None:
            return lattice_max
        return torch.minimum(
            lattice_max,
            torch.as_tensor(
                self.max_size_um,
                device=self.sizes.device,
                dtype=self.sizes.dtype,
            ),
        )

    def _lattice_max_box_sizes(self):
        """Return the per-axis maximum dimensions allowed by the lattice."""
        region_size = self.design_region_cfg["size"]
        n_boxes = self.centers.shape[0:2]
        border = float(self.geometry_cfgs.get("border", 0.2))
        return torch.as_tensor(
            [
                (region_size[0] - border * 2) / n_boxes[0],
                (region_size[1] - border * 2) / n_boxes[1],
            ],
            device=self.sizes.device,
            dtype=self.sizes.dtype,
        )

    def _effective_box_sizes(self, sizes=None):
        """Apply size bounds, small-box policy, and optional square sharing."""
        sizes = self.sizes if sizes is None else sizes
        max_sizes = self._max_box_sizes().to(device=sizes.device, dtype=sizes.dtype)
        min_sizes = torch.as_tensor(
            self.min_size_um, device=sizes.device, dtype=sizes.dtype
        )
        sizes = sizes.clamp(min=0.0)
        sizes = torch.minimum(sizes, max_sizes)

        if self.share_size:
            shared_min = min_sizes.max()
            shared_max = max_sizes.min()
            sizes = sizes[..., 0].clamp(min=0.0, max=shared_max)
            min_sizes = torch.stack((shared_min, shared_min))
            sizes = torch.stack((sizes, sizes), dim=-1)

        if self.small_box_policy == "clamp":
            return torch.maximum(sizes, min_sizes)

        if self.small_box_policy == "snap_by_area":
            minimum_area = min_sizes.prod()
            remove = sizes.prod(dim=-1, keepdim=True) < 0.5 * minimum_area
            snapped = torch.maximum(sizes, min_sizes)
            return torch.where(remove, torch.zeros_like(sizes), snapped)

        if self.small_box_policy == "snap_by_min_side":
            minimum_side = min_sizes.min()
            remove = sizes.min(dim=-1, keepdim=True).values < 0.5 * minimum_side
            snapped = torch.maximum(sizes, min_sizes)
            return torch.where(remove, torch.zeros_like(sizes), snapped)

        remove = (sizes < min_sizes).any(dim=-1, keepdim=True)
        return torch.where(remove, torch.zeros_like(sizes), sizes)

    def project_size_constraints_(self):
        """Project stored box sizes to the configured legal geometry range."""
        with torch.no_grad():
            self.sizes.copy_(self._effective_box_sizes())

    def _build_gpu_subpixel_pattern(
        self, material_values: torch.Tensor | None, background_value: float
    ) -> torch.Tensor:
        """Rasterize the box array with the CuPy subpixel volume-average kernel.

        Args:
            material_values: Per-box normalized material values, or ``None`` for
                legacy unit-valued boxes.
            background_value: Normalized background value outside the boxes.

        Returns:
            Tensor: Differentiable-wrapper output containing the normalized
            subpixel-averaged density on the uniform ``phi_resolution`` grid.

        Raises:
            RuntimeError: If GPU subpixel rasterization is requested without CUDA.
            ImportError: If Tidy3D or CuPy is unavailable.

        Notes:
            The returned tensor is intentionally detached from the Torch graph.
            ``DiffWrapper.backward`` supplies the existing approximate box
            derivatives.
        """
        if not self.centers.is_cuda:
            raise RuntimeError(
                "gpu_subpixel=True requires box parameters on a CUDA device"
            )

        import cupy as cp
        import tidy3d as td

        from gpu_subpixel import GPUSubpixelRasterizer

        # Tidy3D requires scalar relative permittivities >= 1. Encode the
        # normalized density with an affine offset; geometric averaging and
        # subtraction of this offset recover the desired density exactly.
        encoded_offset = 1.0

        if self._gpu_subpixel_rasterizer is None:
            center = self.design_region_cfg["center"]
            size = self.design_region_cfg["size"]
            x_bounds = np.linspace(
                float(center[0]) - float(size[0]) / 2,
                float(center[0]) + float(size[0]) / 2,
                self.phi_shape[0] + 1,
                dtype=np.float64,
            )
            y_bounds = np.linspace(
                float(center[1]) - float(size[1]) / 2,
                float(center[1]) + float(size[1]) / 2,
                self.phi_shape[1] + 1,
                dtype=np.float64,
            )
            # A single unit-thickness z cell makes the 2-D area average a
            # regular 3-D volume average without changing the x-y result.
            grid = td.Grid(boundaries=td.Coords(x=x_bounds, y=y_bounds, z=(-0.5, 0.5)))
            self._gpu_subpixel_grid = grid
            self._gpu_subpixel_rasterizer = GPUSubpixelRasterizer(
                grid, dtype=cp.float32
            )

        centers = self.centers.detach().reshape(-1, 2).cpu().numpy()
        sizes = self._effective_box_sizes().detach().reshape(-1, 2).cpu().numpy()
        if material_values is None:
            values = np.ones(centers.shape[0], dtype=np.float64)
        else:
            values = material_values.detach().reshape(-1).cpu().numpy()

        structures = []
        for (cx, cy), (sx, sy), value in zip(centers, sizes, values):
            if sx <= 0.0 or sy <= 0.0:
                continue
            structures.append(
                td.Structure(
                    geometry=td.Box(
                        center=(float(cx), float(cy), 0.0),
                        size=(float(sx), float(sy), 1.0),
                    ),
                    medium=td.Medium(permittivity=float(value) + encoded_offset),
                )
            )

        raster = self._gpu_subpixel_rasterizer.rasterize(
            structures,
            background_eps=float(background_value) + encoded_offset,
        )

        # out = cp.asnumpy(raster).copy()

        # print(
        #     "RASTER:",
        #     "min =", out.min(),
        #     "max =", out.max(),
        #     "mean =", out.mean(),
        #     "sum =", out.sum(),
        # )

        # if hasattr(self, "_debug_prev"):
        #     diff = out - self._debug_prev
        #     print(
        #         "RASTER DIFF:",
        #         "max =", np.abs(diff).max(),
        #         "mean =", np.abs(diff).mean(),
        #         "sum =", np.abs(diff).sum(),
        #         "nchanged =", np.count_nonzero(np.abs(diff) > 1e-7),
        #     )

        # self._debug_prev = out

        # The rasterizer launches a CuPy RawKernel, while the solver consumes
        # the result through PyTorch.  DLPack transfers ownership, but it does
        # not make the producer's CuPy stream a dependency of arbitrary later
        # CUDA work in every supported CuPy/PyTorch combination.  Synchronize
        # the producer stream before exposing the array to the solver; without
        # this, the solver can repeatedly see the previous iteration's raster
        # even though the custom backward receives a nonzero gradient.
        cp.cuda.get_current_stream().synchronize()
        # Use an explicit copy here instead of the zero-copy DLPack path.  The
        # raster is only the 260 x 260 x 1 design-region map, so this transfer
        # is small, while it avoids backend-dependent lifetime/stream behavior
        # for a CuPy allocation that is immediately consumed by Ceviche.
        density = torch.as_tensor(
            _gpu_raster_to_torch_xy(cp.asnumpy(raster)),
            device=self.operation_device,
        )
        density = density[..., 0] - encoded_offset
        return density.to(device=self.operation_device, dtype=torch.float32)

    def compute_derivatives(self, grad_eps: torch.Tensor, eps: torch.Tensor):
        """Estimate box center and size gradients from four rectangle edges.

        Args:
            grad_eps (torch.Tensor): Design-region raster gradient consumed by
                ``rect_edge_grads_um``.
            eps (torch.Tensor): Rasterized permittivity passed by the custom autograd
                wrapper; the current implementation does not read this tensor directly.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: Center, size,
            and optional material gradients reshaped to match their parameters.
            Center gradients remain zero to preserve the legacy box behavior.
        """
        # grad_eps: (H, W) design region gradient
        # eps: (H, W) design region permittivity
        ## we use 4 edges' gradient to compute the gradients w.r.t. center and size
        effective_sizes = self._effective_box_sizes().detach()
        centers = self.centers.detach()
        xmin = centers[..., 0] - effective_sizes[..., 0] / 2
        xmax = centers[..., 0] + effective_sizes[..., 0] / 2
        ymin = centers[..., 1] - effective_sizes[..., 1] / 2
        ymax = centers[..., 1] + effective_sizes[..., 1] / 2
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

        material_values = self._material_values()
        if material_values is None:
            shape_values = torch.ones(
                self.centers.numel() // 2,
                device=grad_eps.device,
                dtype=grad_eps.dtype,
            )
        else:
            shape_values = material_values.reshape(-1).to(
                device=grad_eps.device, dtype=grad_eps.dtype
            )
            shape_values = shape_values - float(
                self.geometry_cfgs.get("material_background_value", 0.0)
            )

        grad_centers = torch.stack(
            [
                grad_xmin + grad_xmax,
                grad_ymin + grad_ymax,
            ],
            dim=-1,
        )  # d/dcx, d/dcy

        grad_sizes = torch.stack(
            [
                0.5 * (grad_xmax - grad_xmin),
                0.5 * (grad_ymax - grad_ymin),
            ],
            dim=-1,
        )  # d/dsx, d/dsy; edge minima already carry the negative sign
        grad_centers = grad_centers.reshape_as(self.centers).zero_()
        shape_values = shape_values.reshape(self.centers.shape[:-1]).unsqueeze(-1)
        grad_sizes = grad_sizes.reshape_as(self.sizes) * shape_values
        if self.share_size:
            grad_sizes[..., 0].add_(grad_sizes[..., 1])
            grad_sizes[..., 1].zero_()

        grad_material = None
        if material_values is not None and self.optimization_stage in {
            "material",
            "joint",
        }:
            # Integrate per-box contributions without constructing an N,H,W
            # tensor. Match the rasterizer's deterministic last-box-wins rule
            # for the uncommon case where primitives overlap.
            grad_material = torch.zeros_like(material_values)
            covered_later = torch.zeros_like(grad_eps, dtype=torch.bool)
            boxes_flat = boxes_um.flatten(0, -2)
            for i in range(boxes_flat.shape[0] - 1, -1, -1):
                one_mask = fill_rectangles_thresh_um(
                    boxes_flat[i : i + 1],
                    region_center,
                    region_size,
                    self.grid_step,
                    color=1.0,
                    alpha=1.0,
                    img=torch.zeros_like(grad_eps),
                    device=grad_eps.device,
                    dtype=grad_eps.dtype,
                )
                assigned_mask = one_mask.bool() & ~covered_later
                grad_material.reshape(-1)[i] = (grad_eps * assigned_mask).sum()
                covered_later |= one_mask.bool()

        return grad_centers, grad_sizes, grad_material

    def transfer_material_to_shape(
        self,
        *,
        eps_host=None,
        eps_inclusion=None,
        eps_low=None,
        eps_high=None,
        beta=None,
        eta=0.5,
        mapping="isotropic",
    ):
        """Project material variables, assign binary material state, and resize boxes.

        Args:
            eps_host: Host relative permittivity. Defaults to ``eps_bg``.
            eps_inclusion: Primitive relative permittivity. Defaults to ``eps``.
            eps_low: Permittivity represented by normalized zero. Defaults to
                ``eps_bg``.
            eps_high: Permittivity represented by normalized one. Defaults to ``eps``.
            beta: Optional projection sharpness.
            eta: Projection midpoint. Defaults to 0.5.
            mapping: ``"isotropic"``, ``"x_only"``, or ``"y_only"``.

        Returns:
            Tensor: Inclusion fill fractions for all boxes.

        Raises:
            RuntimeError: If material optimization is not enabled.
        """
        if self.material_rho is None:
            raise RuntimeError("Material optimization is not enabled for these boxes")
        eps_low = self.design_region_cfg["eps_bg"] if eps_low is None else eps_low
        eps_high = self.design_region_cfg["eps"] if eps_high is None else eps_high
        eps_host = eps_low if eps_host is None else eps_host
        eps_inclusion = eps_high if eps_inclusion is None else eps_inclusion

        with torch.no_grad():
            projected_rho = self.material_projector.project_rho(
                self.material_rho, beta=beta, eta=eta
            )
            fill_fraction = self.material_projector.rho_to_fill_fraction(
                self.material_rho,
                eps_low=eps_low,
                eps_high=eps_high,
                eps_host=eps_host,
                eps_inclusion=eps_inclusion,
                beta=beta,
                eta=eta,
            )
            eta_t = torch.as_tensor(
                eta, dtype=projected_rho.dtype, device=projected_rho.device
            )
            self.material_state.copy_((projected_rho >= eta_t).to(self.material_state))
            scales = self.material_projector.fill_fraction_to_scale(
                fill_fraction, mapping=mapping
            )
            self.sizes.copy_(self._effective_box_sizes(self._max_box_sizes() * scales))
        return fill_fraction

    def project_material_to_shape(self, **kwargs):
        """Compatibility alias for :meth:`transfer_material_to_shape`."""
        return self.transfer_material_to_shape(**kwargs)

    def build_pattern(
        self,
        permittivity: torch.Tensor = None,
    ) -> torch.Tensor:
        """Build a normalized box-density raster by filling rectangle interiors.

        Args:
            permittivity (torch.Tensor | None): Optional existing design-region raster
                used as the compositing background. Defaults to None; a zero tensor is
                allocated from ``design_region_mask`` on the operation device.

        Returns:
            torch.Tensor: Raster whose box-covered cells are set to normalized
            active normalized material values and whose uncovered cells retain the
            background values.
            Box sizes are projected onto the configured legal range before
            rasterization. The stored parameters are projected by
            ``project_size_constraints_`` or an after-step callback.
        """
        # return a normalized/boolean pattern where we fill 1 inside the geometry and 0 outside
        # we can add various constraints here, do not need to worry about the differentiability of the function
        # eps_bg = self.design_region_cfg["eps_bg"]
        # eps_r = self.design_region_cfg["eps"]

        material_values = self._material_values()
        eps_r = 1.0 if material_values is None else material_values

        # Boxes are rasterized in the design x-y plane. A 3-D design region is
        # extruded by BaseParametrization after this 2-D density is built.
        region_px_size = self.phi_shape
        background_value = float(
            self.geometry_cfgs.get("material_background_value", 0.0)
        )
        eps = (
            torch.full(
                region_px_size,
                background_value,
                device=self.operation_device,
            )
            if permittivity is None
            else permittivity
        )

        region_center = self.design_region_cfg["center"]
        region_size = self.design_region_cfg["size"]
        effective_sizes = self._effective_box_sizes()

        if self.use_gpu_subpixel and permittivity is None:
            eps = self._build_gpu_subpixel_pattern(material_values, background_value)
            # import matplotlib.pyplot as plt
            # plt.imshow(eps.cpu().numpy())
            # plt.savefig("debug_box.png")
            # plt.close()
            return eps

        xmin = self.centers[..., 0] - effective_sizes[..., 0] / 2
        xmax = self.centers[..., 0] + effective_sizes[..., 0] / 2
        ymin = self.centers[..., 1] - effective_sizes[..., 1] / 2
        ymax = self.centers[..., 1] + effective_sizes[..., 1] / 2
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
            dtype=eps.dtype,
        )
        # import matplotlib.pyplot as plt
        # plt.imshow(eps.cpu().numpy())
        # plt.savefig("debug_box.png")
        # plt.close()
        # exit(0)

        return eps

    def forward(self, permittivity: torch.Tensor = None) -> torch.Tensor:
        """Run box rasterization through the custom differentiable autograd wrapper.

        Args:
            permittivity (torch.Tensor | None): Optional background raster forwarded to
                ``build_pattern``. Defaults to None.

        Returns:
            torch.Tensor: Normalized box-density raster produced by ``build_pattern``
            with the custom backward path defined by ``compute_derivatives``.
        """
        return DiffWrapper.apply(
            self.centers,
            self.sizes,
            self._material_values(),
            lambda eps=permittivity: self.build_pattern(permittivity=eps),
            self.compute_derivatives,
        )


class BoxParameterization(BaseParametrization):
    """Register the box geometry as a MAPS design-region parameterization method.

    The default configuration creates a ``16 x 16`` array of box parameters and uses
    the ``2d_array`` initialization strategy.
    """

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
        """Initialize and register box parameter build/reset callbacks.

        Args:
            *args (Any): Positional arguments forwarded to ``BaseParametrization``.
            cfgs (dict): Box parameterization configuration. Defaults to the built-in
                configuration declared in the function signature.
            **kwargs (Any): Keyword arguments forwarded to ``BaseParametrization``.
        """
        super().__init__(*args, cfgs=cfgs, **kwargs)

        method = cfgs["method"]

        self.register_parameter_build_per_region_fn(method, self._build_parameters_box)
        self.register_parameter_reset_per_region_fn(method, self._reset_parameters_box)

        self.build_parameters(cfgs, self.design_region_cfg)
        self.reset_parameters(cfgs, self.design_region_cfg)

    def _prepare_parameters_box(
        self,
    ):
        """Prepare auxiliary parameter state for the box parameterization.

        Returns:
            dict[str, Any]: Empty auxiliary-parameter mapping in the current implementation.
        """
        param_dict = dict()

        return param_dict

    def _build_parameters_box(self, param_cfg, region_cfg):
        """Construct the ``Boxes`` geometry module for one design region.

        Args:
            param_cfg (dict[str, Any]): Region parameterization configuration containing
                ``geometry_cfgs``.
            region_cfg (dict[str, Any]): Design-region configuration supplied by the base
                parameterization lifecycle.

        Returns:
            tuple[dict[str, Boxes], dict[str, Any]]: Parameter mapping containing the
            ``boxes`` module and the auxiliary mapping from ``_prepare_parameters_box``.
        """
        param_dict = self._prepare_parameters_box()
        geometry_cfgs = dict(param_cfg["geometry_cfgs"])
        for key in (
            "material_optimization",
            "enable_material",
            "material_enabled",
            "material_init",
            "material_model",
            "material_background_value",
            "border",
            "phi_resolution",
            "min_size_um",
            "max_size_um",
            "small_box_policy",
            "share_size",
            "gpu_subpixel",
        ):
            if key in param_cfg:
                geometry_cfgs.setdefault(key, param_cfg[key])
        boxes = Boxes(
            sim_cfg=self.hr_device.sim_cfg,
            geometry_cfgs=geometry_cfgs,
            region_name=self.region_name,
            design_region_mask=self.hr_design_region_mask,
            design_region_cfg=self.hr_device.design_region_cfgs[self.region_name],
            operation_device=self.operation_device,
        )

        weight_dict = dict(boxes=boxes)
        return weight_dict, param_dict

    def _uniform_parameter_grid_coords(self, density):
        """Return the box density's uniform physical sampling coordinates."""
        del density
        return self.weights["boxes"].uniform_parameter_grid_coords()

    def set_optimization_stage(
        self,
        stage: str,
        *,
        optimize_centers: bool = False,
    ) -> None:
        """Set the optimization stage on this region's box geometry.

        Args:
            stage: ``"material"``, ``"shape"``, ``"joint"``, or ``"frozen"``.
            optimize_centers: Whether centers should remain trainable. Defaults
                to False.
        """
        self.weights["boxes"].set_optimization_stage(
            stage, optimize_centers=optimize_centers
        )

    def transfer_material_to_shape(self, **kwargs):
        """Transfer this region's material variables into box sizes and state."""
        return self.weights["boxes"].transfer_material_to_shape(**kwargs)

    def project_material_to_shape(self, **kwargs):
        """Compatibility alias for :meth:`transfer_material_to_shape`."""
        return self.transfer_material_to_shape(**kwargs)

    def _reset_parameters_box(
        self,
        weight_dict,
        param_cfg,
        region_cfg,
        init_method: str | Callable = "2d_array",
    ):
        """Initialize box centers and sizes for the selected strategy.

        Args:
            weight_dict (dict[str, Any]): Parameter dictionary containing ``weight_dict[\"boxes\"]``.
            param_cfg (dict[str, Any]): Box parameterization configuration; optional ``density``
                controls random removal of boxes.
            region_cfg (dict[str, Any]): Physical design-region center and size.
            init_method (str | Callable): Initialization strategy name or callable.
                A callable receives ``(weight_dict, param_cfg, region_cfg)`` and is
                responsible for initializing the box parameters. Defaults to
                ``\"2d_array\"``.

        Raises:
            ValueError: If the initialization method is unsupported.

        Notes:
            The optional ``density`` configuration defaults to 0.5 and randomly sets selected
            box sizes to zero.
        """
        if isinstance(init_method, Callable):
            init_method(weight_dict, param_cfg, region_cfg)
            return

        if init_method.startswith("2d_array"):
            density = param_cfg.get("density", 0.5)
            region_center = region_cfg["center"]
            region_size = region_cfg["size"]
            n_boxes = weight_dict["boxes"].centers.shape[0:2]  # Nx, Ny
            ## uniform spread the box centers in the design region using meshgrid
            border = float(weight_dict["boxes"].geometry_cfgs.get("border", 0.2))
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
            if init_method == "2d_array_full":
                weight_dict["boxes"].sizes.data[..., 0].fill_(max_box_size_x)
                weight_dict["boxes"].sizes.data[..., 1].fill_(max_box_size_y)
            else:
                weight_dict["boxes"].sizes.data[..., 0].uniform_(
                    max_box_size_x / 3, max_box_size_x / 2
                )
                weight_dict["boxes"].sizes.data[..., 1].uniform_(
                    max_box_size_y / 3, max_box_size_y / 2
                )
            weight_dict["boxes"].sizes.data = weight_dict["boxes"].sizes.data.to(
                device=self.operation_device
            )
            if (
                weight_dict["boxes"].material_rho is not None
                and init_method == "2d_array_full"
            ):
                material_init = float(param_cfg.get("material_init", 0.5))
                weight_dict["boxes"].material_rho.data.fill_(material_init)

            ## randomly set 1-density boxes to zero size for the legacy mode
            if init_method != "2d_array_full" and density < 1.0:
                random_matrix = torch.rand(n_boxes, device=self.operation_device)
                mask = random_matrix > density
                weight_dict["boxes"].sizes.data.masked_fill_(mask.unsqueeze(-1), 0.0)
        else:
            raise ValueError(f"Unsupported initialization method: {init_method}")

    def _build_density(self, weights, sharpness: float):
        """Build the normalized box density from the geometry module.

        Args:
            weights (dict[str, Any]): Parameter mapping containing the ``boxes`` geometry module.
            sharpness (float): Projection/sharpness argument accepted for the common
                parameterization interface; the box implementation does not directly use it.

        Returns:
            torch.Tensor: Normalized box-density raster.
        """
        boxes: Boxes = weights["boxes"]
        density = boxes.forward()

        return density

    def build_density(self, weights, sharpness: float | None = None):
        """Build the high-resolution normalized box density for a design region.

        Args:
            weights (dict[str, Any]): Parameter mapping containing the ``boxes`` geometry module.
            sharpness (float | None): Common parameterization sharpness argument. Defaults to None.

        Returns:
            torch.Tensor: High-resolution normalized density raster.
        """
        ## this is the high resolution, e.g., res=200, 310 normalized density
        hr_density = self._build_density(
            weights,
            sharpness,
        )

        return hr_density

    def build_permittivity(self, weights, sharpness: float | None = None):
        """Build the normalized box permittivity pattern.

        Args:
            weights (dict[str, Any]): Parameter mapping containing the ``boxes`` geometry module.
            sharpness (float | None): Common parameterization sharpness argument forwarded to
                ``build_density``. Defaults to None.

        Returns:
            torch.Tensor: Normalized box pattern returned by ``build_density``.
        """
        return self.build_density(weights, sharpness)
