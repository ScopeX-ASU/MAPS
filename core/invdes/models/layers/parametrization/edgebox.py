from __future__ import annotations

import secrets
import warnings
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.types import Device

from core.invdes.models.materialization.coverage import average_uniform_cells

from .base_parametrization import BaseParametrization, _convert_resolution
from .levelset import _bilinear_from_knots
from .utils import HeavisideProjection, SubpixelSmoothedProjection

__all__ = [
    "EdgeBoxGeometry",
    "EdgeBoxParameterization",
    "EdgeSpline",
    "polygon_area",
    "polygon_centroid",
]


_EDGE_ORDER = ("x-", "x+", "y-", "y+")


class EdgeBoxDiffWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, build_fn, grad_fn, *raw_params):
        eps = build_fn()
        ctx.grad_fn = grad_fn
        ctx.n_inputs = len(raw_params)
        ctx.save_for_backward(eps)
        return eps

    @staticmethod
    def backward(ctx, grad_output):
        (eps,) = ctx.saved_tensors
        grad_fn = ctx.grad_fn
        grad_raws = grad_fn(grad_output, eps)
        if not isinstance(grad_raws, (tuple, list)):
            raise ValueError(
                "grad_fn must return a tuple/list of gradients, one per active edge spline."
            )
        if len(grad_raws) != ctx.n_inputs:
            raise ValueError(
                f"grad_fn returned {len(grad_raws)} gradients, but {ctx.n_inputs} were expected."
            )
        return (None, None, *grad_raws)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------


def polygon_area(vertices: torch.Tensor) -> torch.Tensor:
    """
    Compute the signed shoelace area of one or more closed 2-D polygons.

    Args:
        vertices (torch.Tensor): Polygon vertices with final dimension 2; leading dimensions are treated as batch dimensions.

    Returns:
        torch.Tensor: Signed polygon area for each leading batch entry.
    """
    x = vertices[..., 0]
    y = vertices[..., 1]
    x1 = torch.roll(x, shifts=-1, dims=-1)
    y1 = torch.roll(y, shifts=-1, dims=-1)
    return 0.5 * torch.sum(x * y1 - x1 * y, dim=-1)


def polygon_centroid(vertices: torch.Tensor) -> torch.Tensor:
    """
    Compute the area-weighted centroid of a nondegenerate 2-D polygon, with the vertex mean as a degenerate fallback.

    Args:
        vertices (torch.Tensor): Polygon vertices with shape ``[V, 2]``.

    Returns:
        torch.Tensor: Centroid with shape ``[2]`` in the same coordinate system as ``vertices``.

    Raises:
        ValueError: If ``vertices`` is not two-dimensional with final dimension 2.
    """
    if vertices.ndim != 2 or vertices.shape[-1] != 2:
        raise ValueError(f"vertices must be [V, 2], got shape {tuple(vertices.shape)}")

    x = vertices[:, 0]
    y = vertices[:, 1]
    x1 = torch.roll(x, shifts=-1, dims=0)
    y1 = torch.roll(y, shifts=-1, dims=0)
    cross = x * y1 - x1 * y
    area = 0.5 * cross.sum()
    if torch.abs(area) < 1e-12:
        return vertices.mean(dim=0)
    cx = ((x + x1) * cross).sum() / (6.0 * area)
    cy = ((y + y1) * cross).sum() / (6.0 * area)
    return torch.stack([cx, cy])


def _point_in_polygon(
    xx: torch.Tensor, yy: torch.Tensor, vertices: torch.Tensor
) -> torch.Tensor:
    vx = vertices[:, 0]
    vy = vertices[:, 1]
    vx1 = torch.roll(vx, shifts=-1, dims=0)
    vy1 = torch.roll(vy, shifts=-1, dims=0)

    x = xx[..., None]
    y = yy[..., None]
    cond = ((vy > y) != (vy1 > y)) & (
        x < (vx1 - vx) * (y - vy) / (vy1 - vy + 1e-12) + vx
    )
    return (cond.sum(dim=-1) % 2) == 1


# ----------------------------------------------------------------------------
# Spline helpers
# ----------------------------------------------------------------------------


def _constraint_map(
    u: torch.Tensor, constraint: Optional[Tuple[float, float]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map raw spline control values into a bounded physical offset interval while preserving zero offset at zero raw value.

    Args:
        u (torch.Tensor): Raw unconstrained control values.
        constraint (tuple[float, float] | None): Physical lower and upper offset bounds. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Bounded physical offsets and the elementwise derivative with respect to ``u``.

    Raises:
        ValueError: If a finite interval does not contain zero.
    """
    if constraint is None:
        return u, torch.ones_like(u)

    low, high = constraint
    if low is None or high is None:
        return u, torch.ones_like(u)

    # Require the interval to contain zero for zero-init = original edge.
    if not (low <= 0.0 <= high):
        raise ValueError(
            f"Constraint interval {constraint} must contain 0.0 for zero-init to mean the original edge."
        )

    t = torch.tanh(u)

    # Piecewise scaling around 0.
    # t in [-1, 0] -> [low, 0]
    # t in [ 0, 1] -> [0, high]
    out = torch.where(t >= 0, high * t, low * (-t))
    deriv = torch.where(t >= 0, high * (1.0 - t * t), (-low) * (1.0 - t * t))
    return out, deriv


@lru_cache(maxsize=128)
def _open_uniform_knot_vector(n_ctrl: int, degree: int) -> torch.Tensor:
    """
    Construct an open uniform B-spline knot vector on ``[0, 1]``.

    Args:
        n_ctrl (int): Number of control points.
        degree (int): Polynomial degree.

    Returns:
        torch.Tensor: One-dimensional float32 knot vector.

    Raises:
        ValueError: If ``n_ctrl < degree + 1``.
    """
    if n_ctrl < degree + 1:
        raise ValueError(
            f"Need n_ctrl >= degree+1, got n_ctrl={n_ctrl}, degree={degree}"
        )
    n_internal = n_ctrl - degree - 1
    start = torch.zeros(degree + 1, dtype=torch.float32)
    end = torch.ones(degree + 1, dtype=torch.float32)
    if n_internal > 0:
        internal = torch.linspace(0.0, 1.0, steps=n_internal + 2, dtype=torch.float32)[
            1:-1
        ]
        return torch.cat([start, internal, end], dim=0)
    return torch.cat([start, end], dim=0)


@lru_cache(maxsize=256)
def _bspline_basis_uniform(n_ctrl: int, degree: int, n_samples: int) -> torch.Tensor:
    """
    Evaluate a uniform open B-spline basis on a regular parameter grid.

    Args:
        n_ctrl (int): Number of control points.
        degree (int): B-spline degree.
        n_samples (int): Number of samples in the parameter domain.

    Returns:
        torch.Tensor: Basis matrix with shape ``[n_samples, n_ctrl]``.
    """
    t = torch.linspace(0.0, 1.0, steps=n_samples, dtype=torch.float32)
    U = _open_uniform_knot_vector(n_ctrl, degree)
    N = torch.zeros(n_samples, n_ctrl, dtype=torch.float32)

    for i in range(n_ctrl):
        left = U[i]
        right = U[i + 1]
        if i == n_ctrl - 1:
            mask = (t >= left) & (t <= right)
        else:
            mask = (t >= left) & (t < right)
        N[:, i] = mask.to(torch.float32)

    for p in range(1, degree + 1):
        N_next = torch.zeros_like(N)
        for i in range(n_ctrl):
            term1 = 0.0
            denom1 = U[i + p] - U[i]
            if float(denom1) > 1e-12:
                term1 = ((t - U[i]) / denom1) * N[:, i]

            term2 = 0.0
            if i + 1 < n_ctrl:
                denom2 = U[i + p + 1] - U[i + 1]
                if float(denom2) > 1e-12:
                    term2 = ((U[i + p + 1] - t) / denom2) * N[:, i + 1]

            N_next[:, i] = term1 + term2
        N = N_next

    return N


try:
    import cupy as cp
    from cupyx.scipy.ndimage import (
        distance_transform_edt as cupy_distance_transform_edt,
    )

    _HAS_CUPY = True
except Exception:
    cp = None
    cupy_distance_transform_edt = None
    _HAS_CUPY = False

try:
    from scipy.ndimage import distance_transform_edt as scipy_distance_transform_edt

    _HAS_SCIPY = True
except Exception:
    scipy_distance_transform_edt = None
    _HAS_SCIPY = False


def _sdf_from_near_binary_density(
    density: torch.Tensor, smooth_ks: int = 3
) -> torch.Tensor:
    """
    Construct a signed distance field from a near-binary density using CuPy or SciPy Euclidean distance transforms.

    Args:
        density (torch.Tensor): Two-dimensional density field. Values above 0.5 are treated as inside.
        smooth_ks (int): Optional average-pooling kernel used before thresholding. Defaults to 3.

    Returns:
        torch.Tensor: Signed distance field with positive values outside and negative values inside.

    Raises:
        ValueError: If ``density`` is not two-dimensional.
        ImportError: If neither CuPy nor SciPy is available.
    """
    if density.ndim != 2:
        raise ValueError(f"Expected 2D density, got shape {tuple(density.shape)}")

    rho = density.detach().float()

    if smooth_ks and smooth_ks > 1:
        rho = F.avg_pool2d(
            rho[None, None],
            kernel_size=smooth_ks,
            stride=1,
            padding=smooth_ks // 2,
        )[0, 0]

    mask = rho > 0.5

    if _HAS_CUPY:
        # Keep the expensive SDF estimation on GPU via CuPy.
        mask_np = mask.cpu().numpy().astype(np.bool_)
        mask_cp = cp.asarray(mask_np)

        inside = cupy_distance_transform_edt(mask_cp)
        outside = cupy_distance_transform_edt(~mask_cp)
        phi_cp = outside - inside  # positive outside, negative inside

        phi = torch.utils.dlpack.from_dlpack(phi_cp.toDlpack())
        return phi.to(device=density.device, dtype=density.dtype)

    if _HAS_SCIPY:
        mask_np = mask.cpu().numpy().astype(np.bool_)
        inside = scipy_distance_transform_edt(mask_np)
        outside = scipy_distance_transform_edt(~mask_np)
        phi_np = outside - inside
        return torch.from_numpy(phi_np).to(device=density.device, dtype=density.dtype)

    raise ImportError("Need either CuPy or SciPy available for SDF construction.")


def _sample_correlated_roughness_1d(
    coords: torch.Tensor,
    *,
    amplitude: float,
    correlation_length: float,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Sample a zero-mean correlated 1-D Gaussian-process roughness profile with squared-exponential covariance.

    Args:
        coords (torch.Tensor): One-dimensional physical coordinates.
        amplitude (float): Target roughness standard-deviation scale.
        correlation_length (float): Correlation length in the same physical units as ``coords``.
        seed (int | None): Optional random seed. Defaults to None.

    Returns:
        torch.Tensor: Roughness samples with the same length, device, and dtype family as ``coords``.

    Raises:
        ValueError: If ``amplitude < 0`` or ``correlation_length <= 0``.
    """
    coords = torch.as_tensor(
        coords, dtype=torch.float32, device=coords.device
    ).flatten()
    if coords.numel() < 2:
        return torch.zeros_like(coords)

    if amplitude < 0 or correlation_length <= 0:
        raise ValueError("amplitude must be >= 0 and correlation_length must be > 0")

    separation = coords[:, None] - coords[None, :]
    cov = (amplitude**2) * torch.exp(
        -2.0 * separation.square() / (correlation_length**2)
    )

    # Small jitter for numerical stability
    jitter = max(amplitude**2, 1.0) * 1e-12
    cov = cov + jitter * torch.eye(
        coords.numel(), device=coords.device, dtype=coords.dtype
    )

    gen = torch.Generator(device=coords.device)
    if seed is not None:
        gen.manual_seed(int(seed))

    L = torch.linalg.cholesky(cov)
    z = torch.randn(
        coords.numel(), generator=gen, device=coords.device, dtype=coords.dtype
    )
    rough = L @ z

    rough = rough - rough.mean()
    rough = rough / (rough.std(unbiased=False) + 1e-8)
    return amplitude * rough


def _soft_density_from_sdf(
    phi_rough: torch.Tensor,
    *,
    x_spacing: float,
    y_spacing: float,
    subpixel_tau_scale: float = 0.25,
) -> torch.Tensor:
    """
    Convert a signed-distance field into a smooth gray occupancy using a sigmoid transition.

    Args:
        phi_rough (torch.Tensor): Signed distance field with negative values inside the structure.
        x_spacing (float): Physical grid spacing along x.
        y_spacing (float): Physical grid spacing along y.
        subpixel_tau_scale (float): Fraction of the smaller grid spacing used as sigmoid width. Defaults to 0.25.

    Returns:
        torch.Tensor: Smooth density field in ``(0, 1)``.
    """
    tau = subpixel_tau_scale * min(float(x_spacing), float(y_spacing))
    tau = max(tau, 1e-8)
    return torch.sigmoid(-phi_rough / tau)


class EdgeSpline(nn.Module):
    """
    A latent 1D spline field used to parametrize a single active box edge.

    For x-sides, the spline returns x(y).
    For y-sides, the spline returns y(x).
    """

    def __init__(
        self,
        *,
        side: str,
        n_ctrl: int,
        degree: int,
        constraint: Optional[Tuple[float, float]],
        operation_device: Device,
    ):
        """
        Initialize a bounded 1-D B-spline control field for one box edge.

        Args:
            side (str): Active edge identifier: ``"x-"``, ``"x+"``, ``"y-"``, or ``"y+"``.
            n_ctrl (int): Number of spline control points.
            degree (int): Requested B-spline degree; it is capped at ``n_ctrl - 1``.
            constraint (tuple[float, float] | None): Optional physical offset bounds.
            operation_device (torch.device): Device holding the learnable raw controls.
        """
        super().__init__()
        assert side in _EDGE_ORDER
        self.side = side
        self.n_ctrl = int(n_ctrl)
        self.degree = int(min(degree, self.n_ctrl - 1))
        self.constraint = constraint
        self.operation_device = operation_device
        self.raw_ctrl = nn.Parameter(
            torch.zeros(self.n_ctrl, device=self.operation_device)
        )

    def physical_ctrl(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the constrained physical edge offsets and raw-to-physical derivative.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Physical control offsets and their elementwise derivative with respect to ``raw_ctrl``.
        """
        return _constraint_map(self.raw_ctrl, self.constraint)

    # def interpolate(
    #     self,
    #     coord_grid: torch.Tensor,
    #     box_min: float,
    #     box_max: float,
    #     base_value: float,
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     ctrl_offset, dctrl_draw = self.physical_ctrl()

    #     denom = float(box_max - box_min)
    #     if abs(denom) < 1e-12:
    #         t = torch.zeros_like(coord_grid)
    #     else:
    #         t = ((coord_grid - box_min) / (denom + 1e-12)).clamp(0.0, 1.0)

    #     basis = _bspline_basis_uniform(
    #         self.n_ctrl, self.degree, t.numel()
    #     ).to(device=ctrl_offset.device, dtype=ctrl_offset.dtype)

    #     taper = torch.sin(torch.pi * t).pow(2)

    #     # Option 2: offset field
    #     offset_curve = basis @ ctrl_offset
    #     base = torch.as_tensor(base_value, device=ctrl_offset.device, dtype=ctrl_offset.dtype)
    #     profile = base + taper * offset_curve

    #     return profile, taper, basis, dctrl_draw

    def interpolate(
        self,
        coord_grid: torch.Tensor,
        box_min: float,
        box_max: float,
        base_value: float,
    ):
        """
        Evaluate the spline edge profile over a physical coordinate grid.

        Args:
            coord_grid (torch.Tensor): Coordinates along the edge tangent direction.
            box_min (float): Lower coordinate bound of the box edge domain.
            box_max (float): Upper coordinate bound of the box edge domain.
            base_value (float): Nominal physical edge location.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Edge profile, taper field, B-spline basis matrix, and raw-control derivative factor.
        """
        ctrl_offset, dctrl_draw = self.physical_ctrl()

        denom = float(box_max - box_min)
        if abs(denom) < 1e-12:
            t = torch.zeros_like(coord_grid)
        else:
            t = ((coord_grid - box_min) / (denom + 1e-12)).clamp(0.0, 1.0)

        basis = _bspline_basis_uniform(self.n_ctrl, self.degree, t.numel()).to(
            device=ctrl_offset.device, dtype=ctrl_offset.dtype
        )

        offset_curve = basis @ ctrl_offset
        base = torch.as_tensor(
            base_value, device=ctrl_offset.device, dtype=ctrl_offset.dtype
        )
        profile = base + offset_curve

        return profile, torch.ones_like(t), basis, dctrl_draw


class EdgeBoxGeometry(nn.Module):
    """
    A fixed box scaffold inside the design region, where the four edges can be
    optimized as spline fields.

    The box is defined in the design-region local frame by default:
      - box_center=(0, 0) means the box is centered at the design-region center
      - box_size is a fraction of the design-region size by default

    Coordinate convention is [x, y]: x is the first dimension, y is the second.
    """

    def __init__(
        self,
        sim_cfg: Dict[str, Any],
        geometry_cfgs: Dict[str, Any],
        region_name: str = "design_region_1",
        design_region_mask=None,
        design_region_cfg=None,
        operation_device: Device = torch.device("cuda:0"),
    ):
        """
        Initialize a spline-controlled box scaffold inside a design region.

        Args:
            sim_cfg (dict[str, Any]): Simulation configuration; ``resolution`` determines the geometry grid step.
            geometry_cfgs (dict[str, Any]): EdgeBox geometry, symmetry, box-size, control-resolution, and roughness settings.
            region_name (str): Name of the associated design region. Defaults to ``"design_region_1"``.
            design_region_mask (tuple[slice, ...] | None): Native/high-resolution design-region mask. Defaults to None.
            design_region_cfg (dict | None): Design-region center/size configuration. Defaults to None.
            operation_device (torch.device): Device for learnable spline parameters and intermediate geometry tensors. Defaults to ``torch.device("cuda:0")``.
        """
        super().__init__()
        self.sim_cfg = sim_cfg
        self.geometry_cfgs = geometry_cfgs
        self.region_name = region_name
        self.design_region_mask = design_region_mask
        self.design_region_cfg = design_region_cfg
        self.operation_device = operation_device

        self.grid_step = 1.0 / float(self.sim_cfg["resolution"])
        self.degree = int(self.geometry_cfgs.get("degree", 3))
        self.pair_symmetry = self.geometry_cfgs.get("pair_symmetry", "mirror")
        if self.pair_symmetry not in {"mirror", "shared", "independent"}:
            raise ValueError(f"Unsupported pair_symmetry={self.pair_symmetry!r}")

        self.enabled_edges = set(self.geometry_cfgs.get("enabled_edges", _EDGE_ORDER))
        self.edge_constraints = dict(self.geometry_cfgs.get("constraints", {}))

        self.box_center = tuple(self.geometry_cfgs.get("box_center", (0.0, 0.0)))
        self.box_size = tuple(self.geometry_cfgs.get("box_size", (0.5, 0.5)))
        self.box_center_unit = self.geometry_cfgs.get("box_center_unit", "relative")
        self.box_size_unit = self.geometry_cfgs.get("box_size_unit", "relative")

        self.rho_resolution = dict(self.geometry_cfgs.get("rho_resolution", {}))
        self.default_rho_resolution = float(
            self.geometry_cfgs.get("default_rho_resolution", 50.0)
        )
        self.phi_resolution = self.geometry_cfgs.get(
            "phi_resolution",
            self.geometry_cfgs.get("default_phi_resolution", 50.0),
        )
        self.n_ctrl_by_edge = dict(self.geometry_cfgs.get("n_ctrl_by_edge", {}))
        self.line_edge_roughness = self.geometry_cfgs.get(
            "line_edge_roughness", self.geometry_cfgs.get("roughness")
        )
        roughness_cfg = self.line_edge_roughness
        if isinstance(roughness_cfg, dict):
            configured_seed = roughness_cfg.get("seed", None)
            self._roughness_seed = (
                secrets.randbits(63)
                if configured_seed is None
                else int(configured_seed)
            )
        else:
            self._roughness_seed = None
        self._roughness_samples: Dict[
            str, Tuple[Tuple[float, float, int, int], torch.Tensor, torch.Tensor]
        ] = {}

        self._active_side_by_axis: Dict[str, Optional[str]] = {"x": None, "y": None}
        self._canonical_pair_specs: Dict[str, Dict[str, Any]] = {}

        self._build_edge_modules()

    # ---------------------- coordinate conversions ----------------------
    def _relative_to_absolute_center(self) -> Tuple[float, float]:
        """
        Convert the configured box center into absolute physical coordinates.

        Returns:
            tuple[float, float]: Absolute x/y box center in the same physical coordinate system as the design region.

        Raises:
            ValueError: If ``box_center_unit`` is unsupported.
        """
        dr_center = self.design_region_cfg["center"]
        dr_size = self.design_region_cfg["size"]
        if self.box_center_unit == "relative":
            return (
                float(dr_center[0] + self.box_center[0] * dr_size[0]),
                float(dr_center[1] + self.box_center[1] * dr_size[1]),
            )
        if self.box_center_unit == "um":
            return float(self.box_center[0]), float(self.box_center[1])
        raise ValueError(f"Unsupported box_center_unit={self.box_center_unit!r}")

    def _relative_to_absolute_size(self) -> Tuple[float, float]:
        """
        Convert the configured box size into absolute physical dimensions.

        Returns:
            tuple[float, float]: Absolute x/y box size.

        Raises:
            ValueError: If ``box_size_unit`` is unsupported.
        """
        dr_size = self.design_region_cfg["size"]
        if self.box_size_unit == "relative":
            return (
                float(self.box_size[0] * dr_size[0]),
                float(self.box_size[1] * dr_size[1]),
            )
        if self.box_size_unit == "um":
            return float(self.box_size[0]), float(self.box_size[1])
        raise ValueError(f"Unsupported box_size_unit={self.box_size_unit!r}")

    def _design_bounds(self, axis: str) -> Tuple[float, float]:
        """
        Return the absolute design-region bounds along one axis.

        Args:
            axis (str): ``"x"`` or ``"y"``.

        Returns:
            tuple[float, float]: Lower and upper physical bounds.
        """
        center = self.design_region_cfg["center"]
        size = self.design_region_cfg["size"]
        if axis == "x":
            return float(center[0] - size[0] / 2.0), float(center[0] + size[0] / 2.0)
        return float(center[1] - size[1] / 2.0), float(center[1] + size[1] / 2.0)

    def _box_bounds(self, axis: str) -> Tuple[float, float]:
        """
        Return the absolute spline-controlled box bounds along one axis before edge deformation.

        Args:
            axis (str): ``"x"`` or ``"y"``.

        Returns:
            tuple[float, float]: Nominal lower and upper physical box bounds.
        """
        box_center = self._relative_to_absolute_center()
        box_size = self._relative_to_absolute_size()
        if axis == "x":
            return float(box_center[0] - box_size[0] / 2.0), float(
                box_center[0] + box_size[0] / 2.0
            )
        return float(box_center[1] - box_size[1] / 2.0), float(
            box_center[1] + box_size[1] / 2.0
        )

    def _nominal_edge_base(self, side: str) -> float:
        """
        Return the nominal absolute coordinate of one undeformed box edge.

        Args:
            side (str): Edge identifier from ``"x-"``, ``"x+"``, ``"y-"``, ``"y+"``.

        Returns:
            float: Physical coordinate of the requested edge.

        Raises:
            ValueError: If ``side`` is unsupported.
        """
        box_center = self._relative_to_absolute_center()
        box_size = self._relative_to_absolute_size()
        if side == "x-":
            return float(box_center[0] - box_size[0] / 2.0)
        if side == "x+":
            return float(box_center[0] + box_size[0] / 2.0)
        if side == "y-":
            return float(box_center[1] - box_size[1] / 2.0)
        if side == "y+":
            return float(box_center[1] + box_size[1] / 2.0)
        raise ValueError(f"Unsupported side: {side}")

    # ---------------------- active edge bookkeeping ----------------------
    def _edge_present(self, side: str) -> bool:
        return side in self.enabled_edges

    def _canonical_side_for_axis(self, axis: str) -> Optional[str]:
        neg = f"{axis}-"
        pos = f"{axis}+"
        if self._edge_present(pos):
            return pos
        if self._edge_present(neg):
            return neg
        return None

    def _edge_length_um(self, side: str) -> float:
        abs_size = self._relative_to_absolute_size()
        return float(abs_size[1] if side.startswith("x") else abs_size[0])

    def _edge_rho_resolution(self, side: str) -> float:
        """
        Resolve the control-point density used along one edge.

        Args:
            side (str): Edge identifier.

        Returns:
            float: Requested control resolution along the edge.
        """
        if side in self.rho_resolution:
            return float(self.rho_resolution[side])
        axis = side[0]
        if axis in self.rho_resolution:
            return float(self.rho_resolution[axis])
        return float(self.default_rho_resolution)

    def _edge_n_ctrl(self, side: str) -> int:
        """
        Compute the number of B-spline control points required for one edge.

        Args:
            side (str): Edge identifier.

        Returns:
            int: Control-point count, at least ``degree + 1``.
        """
        if side in self.n_ctrl_by_edge:
            return int(self.n_ctrl_by_edge[side])
        rho = self._edge_rho_resolution(side)
        length_um = self._edge_length_um(side)
        return max(self.degree + 1, int(round(length_um * rho)) + 1)

    def _adjust_constraint_for_side(
        self, side: str, constraint: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """
        Clip an edge displacement constraint to remain inside the design region and preserve the opposite-edge ordering.

        Args:
            side (str): Edge identifier.
            constraint (tuple[float, float] | None): Requested absolute physical bounds.

        Returns:
            tuple[float, float] | None: Feasible adjusted bounds, or None when unconstrained.

        Side Effects:
            Emits a warning when the requested interval is modified.
        """
        if constraint is None:
            return None

        low, high = constraint
        if low is None or high is None:
            return None
        if low > high:
            low, high = high, low

        reg_low, reg_high = self._design_bounds(side[0])
        opp_side = {"x-": "x+", "x+": "x-", "y-": "y+", "y+": "y-"}[side]
        opp_base = self._nominal_edge_base(opp_side)
        gap = max(self.grid_step, 1e-6)

        if side.endswith("-"):
            feasible_low = reg_low
            feasible_high = min(reg_high, opp_base - gap)
        else:
            feasible_low = max(reg_low, opp_base + gap)
            feasible_high = reg_high

        adj_low = max(low, feasible_low)
        adj_high = min(high, feasible_high)

        if adj_low > adj_high:
            base = self._nominal_edge_base(side)
            center = min(max(base, feasible_low), feasible_high)
            half = 0.5 * gap
            adj_low = max(feasible_low, center - half)
            adj_high = min(feasible_high, center + half)

        if (adj_low, adj_high) != (low, high):
            warnings.warn(
                f"Adjusted constraint for {side} from {(low, high)} to {(adj_low, adj_high)} to keep it within the design region and box ordering.",
                stacklevel=2,
            )
        return (float(adj_low), float(adj_high))

    def _build_edge_modules(self):
        """
        Construct the active ``EdgeSpline`` modules and canonical pair/symmetry bookkeeping.

        Returns:
            None: Populates ``edge_modules``, ``_active_side_by_axis``, and ``_canonical_pair_specs``.
        """
        self.edge_modules = nn.ModuleDict()

        for axis in ("x", "y"):
            neg_side = f"{axis}-"
            pos_side = f"{axis}+"

            if self.pair_symmetry == "independent":
                independent_sides = tuple(
                    side for side in (neg_side, pos_side) if self._edge_present(side)
                )
                for side in independent_sides:
                    n_ctrl = self._edge_n_ctrl(side)
                    constraint = self._adjust_constraint_for_side(
                        side, self.edge_constraints.get(side, None)
                    )
                    self.edge_modules[side] = EdgeSpline(
                        side=side,
                        n_ctrl=n_ctrl,
                        degree=self.degree,
                        constraint=constraint,
                        operation_device=self.operation_device,
                    )
                self._canonical_pair_specs[axis] = {
                    "active_side": None,
                    "derived_side": None,
                    "independent_sides": independent_sides,
                    "symmetry": self.pair_symmetry,
                }
                continue

            active_side = self._canonical_side_for_axis(axis)
            self._active_side_by_axis[axis] = active_side

            if active_side is None:
                self._canonical_pair_specs[axis] = {
                    "active_side": None,
                    "derived_side": None,
                    "independent_sides": (),
                    "symmetry": self.pair_symmetry,
                }
                continue

            inactive_side = f"{axis}-" if active_side == f"{axis}+" else f"{axis}+"
            if self._edge_present(inactive_side):
                warnings.warn(
                    f"Both {axis}- and {axis}+ were enabled; using {active_side} as canonical and deriving {inactive_side}.",
                    stacklevel=2,
                )

            n_ctrl = self._edge_n_ctrl(active_side)
            constraint = self._adjust_constraint_for_side(
                active_side, self.edge_constraints.get(active_side, None)
            )
            self.edge_modules[active_side] = EdgeSpline(
                side=active_side,
                n_ctrl=n_ctrl,
                degree=self.degree,
                constraint=constraint,
                operation_device=self.operation_device,
            )

            self._canonical_pair_specs[axis] = {
                "active_side": active_side,
                "derived_side": inactive_side,
                "independent_sides": (),
                "symmetry": self.pair_symmetry,
            }

    def _pack_raw_params(self) -> Tuple[Tensor, ...]:
        return tuple(mod.raw_ctrl for mod in self.edge_modules.values())

    def _roughness_profile(
        self,
        side: str,
        coord_grid: torch.Tensor,
        box_min: float,
        box_max: float,
        amplitude: float | None = None,
        correlation_length: float | None = None,
        seed: int | None = None,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        """
        Generate or reuse a correlated tangential roughness profile for one edge.

        Args:
            side (str): Edge identifier used for deterministic seed offsets and cache selection.
            coord_grid (torch.Tensor): Tangential physical coordinates.
            box_min (float): Lower edge-domain coordinate.
            box_max (float): Upper edge-domain coordinate.
            amplitude (float | None): Roughness amplitude override. Defaults to None, which reads configuration.
            correlation_length (float | None): Correlation-length override. Defaults to None, which reads configuration.
            seed (int | None): Optional deterministic seed override. Defaults to None.
            num_samples (int | None): Number of Gaussian-process samples. Defaults to None, which uses the configured value or the coordinate count.

        Returns:
            torch.Tensor: Roughness profile sampled on ``coord_grid``.
        """
        cfg = self.line_edge_roughness
        if cfg in (None, False):
            cfg = {}
        elif cfg is True or not isinstance(cfg, dict):
            raise TypeError("line_edge_roughness must be a mapping or None")

        amplitude = float(cfg.get("amplitude", 0.0) if amplitude is None else amplitude)
        correlation_length = float(
            cfg.get("correlation_length", 0.0)
            if correlation_length is None
            else correlation_length
        )
        if amplitude == 0.0:
            return torch.zeros_like(coord_grid)
        if amplitude < 0.0 or correlation_length <= 0.0:
            raise ValueError(
                "line_edge_roughness requires non-negative amplitude and positive correlation_length"
            )

        num_samples = int(
            cfg.get("num_samples", max(32, coord_grid.numel()))
            if num_samples is None
            else num_samples
        )
        if num_samples < 3:
            raise ValueError("line_edge_roughness num_samples must be at least 3")

        covariance_dtype = torch.float64
        sample_coords = torch.linspace(
            box_min,
            box_max,
            num_samples,
            device=coord_grid.device,
            dtype=covariance_dtype,
        )
        separation = sample_coords[:, None] - sample_coords[None, :]
        covariance = amplitude**2 * torch.exp(
            -2.0 * separation.square() / correlation_length**2
        )
        identity = torch.eye(
            num_samples, device=coord_grid.device, dtype=covariance_dtype
        )
        roughness_seed = self._roughness_seed if seed is None else int(seed)
        if roughness_seed is None:
            roughness_seed = secrets.randbits(63)
            self._roughness_seed = roughness_seed
        cache_key = (
            amplitude,
            correlation_length,
            num_samples,
            roughness_seed,
        )
        cached = self._roughness_samples.get(side)
        cache_valid = cached is not None and cached[0] == cache_key
        if cache_valid:
            _, cached_coords, samples = cached
            cache_valid = cached_coords.device == sample_coords.device and torch.equal(
                cached_coords, sample_coords
            )

        if not cache_valid:
            sample_seed = roughness_seed + _EDGE_ORDER.index(side)
            generator = torch.Generator(device=coord_grid.device)
            generator.manual_seed(sample_seed)
            noise = torch.randn(
                num_samples,
                generator=generator,
                device=coord_grid.device,
                dtype=covariance_dtype,
            )
            jitter = max(amplitude**2, 1.0) * 1e-12
            for _ in range(8):
                try:
                    factor = torch.linalg.cholesky(covariance + jitter * identity)
                    break
                except torch.linalg.LinAlgError:
                    jitter *= 10.0
            else:
                raise ValueError(
                    "Unable to factorize line_edge_roughness covariance; "
                    "reduce num_samples or increase correlation_length"
                )
            samples = factor @ noise
            self._roughness_samples[side] = (
                cache_key,
                sample_coords,
                samples.detach(),
            )

        sample_coords = sample_coords.to(dtype=coord_grid.dtype)
        samples = samples.to(device=coord_grid.device, dtype=coord_grid.dtype)
        indices = torch.searchsorted(sample_coords, coord_grid).clamp(
            1, num_samples - 1
        )
        lower = indices - 1
        upper = indices
        fraction = (coord_grid - sample_coords[lower]) / (
            sample_coords[upper] - sample_coords[lower] + 1e-12
        )
        return samples[lower] * (1.0 - fraction) + samples[upper] * fraction

    def _normal_roughness_profile(
        self,
        side: str,
        coord_grid: torch.Tensor,
        box_min: float,
        box_max: float,
        profile: torch.Tensor,
        amplitude: float | None = None,
        correlation_length: float | None = None,
        seed: int | None = None,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        """
        Scale an edge-tangent roughness profile by the local profile arclength factor to obtain normal displacement.

        Args:
            side (str): Edge identifier.
            coord_grid (torch.Tensor): Tangential coordinates.
            box_min (float): Lower domain bound.
            box_max (float): Upper domain bound.
            profile (torch.Tensor): Nominal edge profile.
            amplitude (float | None): Roughness amplitude override. Defaults to None.
            correlation_length (float | None): Correlation-length override. Defaults to None.
            seed (int | None): Optional random seed. Defaults to None.
            num_samples (int | None): Optional roughness sample count. Defaults to None.

        Returns:
            torch.Tensor: Roughness expressed as an outward-normal displacement.
        """
        roughness = self._roughness_profile(
            side,
            coord_grid,
            box_min,
            box_max,
            amplitude=amplitude,
            correlation_length=correlation_length,
            seed=seed,
            num_samples=num_samples,
        )
        if coord_grid.numel() < 2:
            return roughness

        profile_detached = profile.detach()
        slope = torch.empty_like(profile_detached)
        slope[1:-1] = (profile_detached[2:] - profile_detached[:-2]) / (
            coord_grid[2:] - coord_grid[:-2] + 1e-12
        )
        slope[0] = (profile_detached[1] - profile_detached[0]) / (
            coord_grid[1] - coord_grid[0] + 1e-12
        )
        slope[-1] = (profile_detached[-1] - profile_detached[-2]) / (
            coord_grid[-1] - coord_grid[-2] + 1e-12
        )
        normal_projection = torch.sqrt(1.0 + slope.square())
        return roughness * normal_projection

    # ---------------------- forward rasterization ----------------------
    def _resolve_pair_profiles(
        self,
        axis: str,
        coord_grid: torch.Tensor,
        box_min: float,
        box_max: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """
        Resolve the negative/positive edge profiles for one axis according to the configured pair-symmetry policy.

        Args:
            axis (str): ``"x"`` or ``"y"``.
            coord_grid (torch.Tensor): Coordinates along the edge pair.
            box_min (float): Nominal box lower coordinate along the edge axis.
            box_max (float): Nominal box upper coordinate along the edge axis.
            device (torch.device): Output device.
            dtype (torch.dtype): Output dtype.

        Returns:
            dict: Pair profiles plus spline basis, taper, active/derived edge, and independent-side metadata.
        """
        if self.pair_symmetry == "independent":
            base_profiles = {
                side: torch.full_like(
                    coord_grid,
                    self._nominal_edge_base(side),
                    device=device,
                    dtype=dtype,
                )
                for side in (f"{axis}-", f"{axis}+")
            }
            independent_profiles = {}
            for side in self._canonical_pair_specs[axis]["independent_sides"]:
                edge_module: EdgeSpline = self.edge_modules[side]
                base = self._nominal_edge_base(side)
                profile, taper, basis, dctrl_draw = edge_module.interpolate(
                    coord_grid, box_min, box_max, base
                )
                independent_profiles[side] = {
                    "profile": profile,
                    "taper": taper,
                    "basis": basis,
                    "dctrl_draw": dctrl_draw,
                }
                base_profiles[side] = profile

            return {
                "neg": base_profiles[f"{axis}-"],
                "pos": base_profiles[f"{axis}+"],
                "active_side": None,
                "other_side": None,
                "independent_profiles": independent_profiles,
                "coord_grid": coord_grid,
            }

        active_side = self._canonical_pair_specs[axis]["active_side"]
        symmetry = self._canonical_pair_specs[axis]["symmetry"]
        base_neg = self._nominal_edge_base(f"{axis}-")
        base_pos = self._nominal_edge_base(f"{axis}+")

        if active_side is None:
            return {
                "neg": torch.full_like(
                    coord_grid, base_neg, device=device, dtype=dtype
                ),
                "pos": torch.full_like(
                    coord_grid, base_pos, device=device, dtype=dtype
                ),
                "active_side": None,
                "other_side": None,
                "coord_grid": coord_grid,
                "taper": torch.zeros_like(coord_grid, device=device, dtype=dtype),
                "basis": torch.zeros(coord_grid.numel(), 1, device=device, dtype=dtype),
                "dctrl_draw": torch.ones(1, device=device, dtype=dtype),
            }

        active_mod: EdgeSpline = self.edge_modules[active_side]
        active_base = self._nominal_edge_base(active_side)
        other_side = self._canonical_pair_specs[axis]["derived_side"]
        other_base = self._nominal_edge_base(other_side)

        active_profile, taper, basis, dctrl_draw = active_mod.interpolate(
            coord_grid, box_min, box_max, active_base
        )
        active_profile = active_profile
        displacement = active_profile - torch.as_tensor(
            active_base, device=device, dtype=dtype
        )

        if symmetry == "mirror":
            other_profile = (
                torch.as_tensor(other_base, device=device, dtype=dtype) - displacement
            )
        elif symmetry == "shared":
            other_profile = (
                torch.as_tensor(other_base, device=device, dtype=dtype) + displacement
            )
        else:
            raise ValueError(f"Unsupported pair_symmetry={symmetry!r}")

        if active_side.endswith("+"):
            neg, pos = other_profile, active_profile
        else:
            neg, pos = active_profile, other_profile

        return {
            "neg": neg,
            "pos": pos,
            "active_side": active_side,
            "other_side": other_side,
            "independent_profiles": {},
            "coord_grid": coord_grid,
            "taper": taper,
            "basis": basis,
            "dctrl_draw": dctrl_draw,
        }

    def _build_profiles(self, X: int, Y: int, device: torch.device, dtype: torch.dtype):
        """
        Build spatial edge profiles and coordinate grids for the full 2-D design region.

        Args:
            X (int): Number of x samples.
            Y (int): Number of y samples.
            device (torch.device): Output device.
            dtype (torch.dtype): Output dtype.

        Returns:
            dict: Left/right/top/bottom profiles, pair metadata, box bounds, and coordinate vectors.
        """
        x0, x1 = self._box_bounds("x")
        y0, y1 = self._box_bounds("y")
        x_coords = torch.linspace(
            self.design_region_cfg["center"][0]
            - self.design_region_cfg["size"][0] / 2.0,
            self.design_region_cfg["center"][0]
            + self.design_region_cfg["size"][0] / 2.0,
            steps=X,
            device=device,
            dtype=dtype,
        )
        y_coords = torch.linspace(
            self.design_region_cfg["center"][1]
            - self.design_region_cfg["size"][1] / 2.0,
            self.design_region_cfg["center"][1]
            + self.design_region_cfg["size"][1] / 2.0,
            steps=Y,
            device=device,
            dtype=dtype,
        )

        x_pair = self._resolve_pair_profiles("x", y_coords, y0, y1, device, dtype)
        y_pair = self._resolve_pair_profiles("y", x_coords, x0, x1, device, dtype)

        return {
            "x_left": x_pair["neg"],
            "x_right": x_pair["pos"],
            "y_bottom": y_pair["neg"],
            "y_top": y_pair["pos"],
            "x_pair": x_pair,
            "y_pair": y_pair,
            "box_xmin": x0,
            "box_xmax": x1,
            "box_ymin": y0,
            "box_ymax": y1,
            "x_coords": x_coords,
            "y_coords": y_coords,
        }

    def build_pattern(self, permittivity: torch.Tensor = None) -> torch.Tensor:
        """
        Rasterize the spline-controlled box into a binary/material pattern on the configured design-region grid.

        Args:
            permittivity (torch.Tensor | None): Optional preallocated output tensor to populate. Defaults to None, which creates a zero-valued float32 tensor.

        Returns:
            torch.Tensor: Rasterized 2-D pattern with the same spatial shape as the generated design grid.
        """
        eps_r = 1.0
        dims = sorted(
            self.geometry_cfgs.get("dims", (0, 1))
        )  # the dimensions that the geometry is defined on, default to [0, 1] for x-y plane

        region_size = self.design_region_cfg["size"]
        if isinstance(self.phi_resolution, dict):
            phi_resolution = [
                float(self.phi_resolution.get(axis, 50.0)) for axis in ("x", "y")
            ]
        else:
            phi_resolution = [float(self.phi_resolution)] * 2
        region_px_size = [
            max(2, int(round(float(region_size[d]) * phi_resolution[d])) + 1)
            for d in dims
        ]
        eps = (
            torch.zeros(
                region_px_size, device=self.operation_device, dtype=torch.float32
            )
            if permittivity is None
            else permittivity
        )

        center = self.design_region_cfg["center"]
        size = self.design_region_cfg["size"]
        X, Y = eps.shape[-2], eps.shape[-1]

        x_coords = torch.linspace(
            center[0] - size[0] / 2.0,
            center[0] + size[0] / 2.0,
            steps=X,
            device=eps.device,
            dtype=eps.dtype,
        )
        y_coords = torch.linspace(
            center[1] - size[1] / 2.0,
            center[1] + size[1] / 2.0,
            steps=Y,
            device=eps.device,
            dtype=eps.dtype,
        )
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing="ij")

        profiles = self._build_profiles(X=X, Y=Y, device=eps.device, dtype=eps.dtype)
        x_left = profiles["x_left"][None, :]
        x_right = profiles["x_right"][None, :]
        y_bottom = profiles["y_bottom"][:, None]
        y_top = profiles["y_top"][:, None]

        # box_mask = (
        #     (xx >= profiles["box_xmin"]) & (xx <= profiles["box_xmax"]) &
        #     (yy >= profiles["box_ymin"]) & (yy <= profiles["box_ymax"])
        # )
        inside_x = (xx >= x_left) & (xx <= x_right)
        inside_y = (yy >= y_bottom) & (yy <= y_top)
        # mask = box_mask & inside_x & inside_y
        # eps = torch.where(mask, torch.full_like(eps, eps_r), eps)

        mask = inside_x & inside_y
        eps = torch.where(mask, torch.full_like(eps, eps_r), eps)
        return eps

    def add_roughness(
        self,
        density: torch.Tensor,
        *,
        A: float,
        Lc: float,
        seed: int | None = None,
        num_samples: int | None = None,
        subpixel_tau_scale: float = 0.25,
        use_ste: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply correlated stochastic edge roughness as a differentiable straight-through perturbation of the density field.

        Args:
            density (torch.Tensor): Two-dimensional near-binary density.
            A (float): Roughness amplitude.
            Lc (float): Correlation length.
            seed (int | None): Optional random seed. Defaults to None.
            num_samples (int | None): Optional roughness sampling resolution per edge. Defaults to None.
            subpixel_tau_scale (float): Fraction of grid spacing used as the SDF-to-density sigmoid width. Defaults to 0.25.

        Returns:
            torch.Tensor: Roughened density with the original tensor's gradient connection preserved through a straight-through term.
        """
        if A < 0 or Lc <= 0:
            raise ValueError("add_roughness requires A >= 0 and Lc > 0")
        if density.ndim != 2:
            raise ValueError(
                f"EdgeBox roughness requires a 2D density, got shape {tuple(density.shape)}"
            )

        X, Y = density.shape
        profiles = self._build_profiles(
            X=X,
            Y=Y,
            device=density.device,
            dtype=density.dtype,
        )

        x_spacing = float(self.design_region_cfg["size"][0]) / max(X - 1, 1)
        y_spacing = float(self.design_region_cfg["size"][1]) / max(Y - 1, 1)

        # Signed distance field from the near-binary density
        phi = _sdf_from_near_binary_density(density, smooth_ks=3)

        # Normal derivatives of the SDF
        phi_x = torch.gradient(phi, spacing=(x_spacing,), dim=(0,), edge_order=1)[0]
        phi_y = torch.gradient(phi, spacing=(y_spacing,), dim=(1,), edge_order=1)[0]

        edge_band = max(
            float(self.geometry_cfgs.get("border_band_um", 2.0 * self.grid_step)),
            self.grid_step,
        )

        # Collect signed normal displacement field.
        # Sign convention:
        #   phi > 0 outside, phi < 0 inside
        #   phi_rough = phi - delta_field
        # For a positive roughness displacement along the outward normal:
        #   x- edge outward is -x  -> sign = -1
        #   x+ edge outward is +x  -> sign = +1
        #   y- edge outward is -y  -> sign = -1
        #   y+ edge outward is +y  -> sign = +1
        delta_field = torch.zeros_like(phi)

        seed_offset = {"x-": 11, "x+": 23, "y-": 37, "y+": 53}

        # Left/right edges: tangential variation along y
        for side, x_profile in (
            ("x-", profiles["x_left"]),
            ("x+", profiles["x_right"]),
        ):
            if side not in self.enabled_edges:
                continue

            n_samples = Y if num_samples is None else int(num_samples)
            y_coords = profiles["y_coords"]
            if n_samples != y_coords.numel():
                y_coords = torch.linspace(
                    float(y_coords[0]),
                    float(y_coords[-1]),
                    n_samples,
                    device=density.device,
                    dtype=density.dtype,
                )

            delta_y = _sample_correlated_roughness_1d(
                y_coords,
                amplitude=A,
                correlation_length=Lc,
                seed=None if seed is None else seed + seed_offset[side],
            )

            if delta_y.numel() != profiles["y_coords"].numel():
                delta_y = F.interpolate(
                    delta_y[None, None, :],
                    size=profiles["y_coords"].numel(),
                    mode="linear",
                    align_corners=True,
                )[0, 0]

            x_distance = profiles["x_coords"][:, None] - x_profile[None, :]
            x_weight = torch.clamp(1.0 - x_distance.abs() / edge_band, min=0.0)

            outward_sign = -1.0 if side == "x-" else +1.0
            delta_field = delta_field + outward_sign * (delta_y[None, :] * x_weight)

        # Bottom/top edges: tangential variation along x
        for side, y_profile in (
            ("y-", profiles["y_bottom"]),
            ("y+", profiles["y_top"]),
        ):
            if side not in self.enabled_edges:
                continue

            n_samples = X if num_samples is None else int(num_samples)
            x_coords = profiles["x_coords"]
            if n_samples != x_coords.numel():
                x_coords = torch.linspace(
                    float(x_coords[0]),
                    float(x_coords[-1]),
                    n_samples,
                    device=density.device,
                    dtype=density.dtype,
                )

            delta_x = _sample_correlated_roughness_1d(
                x_coords,
                amplitude=A,
                correlation_length=Lc,
                seed=None if seed is None else seed + seed_offset[side],
            )

            if delta_x.numel() != profiles["x_coords"].numel():
                delta_x = F.interpolate(
                    delta_x[None, None, :],
                    size=profiles["x_coords"].numel(),
                    mode="linear",
                    align_corners=True,
                )[0, 0]

            y_distance = profiles["y_coords"][None, :] - y_profile[:, None]
            y_weight = torch.clamp(1.0 - y_distance.abs() / edge_band, min=0.0)

            outward_sign = -1.0 if side == "y-" else +1.0
            delta_field = delta_field + outward_sign * (delta_x[:, None] * y_weight)

        phi_rough = phi - delta_field

        # Subpixel gray rasterization
        density_soft = _soft_density_from_sdf(
            phi_rough,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            subpixel_tau_scale=subpixel_tau_scale,
        )

        return density + (density_soft - density).detach()

        if use_ste:
            # Hard forward, soft backward
            hard = (phi_rough <= 0).to(density_soft.dtype)
            return hard + (density_soft - density_soft.detach())

        # Gray-valued forward; detach because you said you do not need differentiability
        return density_soft.detach()

    def add_roughness_old(
        self,
        density: torch.Tensor,
        *,
        A: float,
        Lc: float,
        seed: int | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply the legacy first-order density-gradient roughness correction near enabled edges.

        Args:
            density (torch.Tensor): Two-dimensional density field.
            A (float): Roughness amplitude.
            Lc (float): Correlation length.
            seed (int | None): Optional random seed. Defaults to None.
            num_samples (int | None): Optional sampling resolution. Defaults to None.
            **kwargs: Additional legacy options accepted for compatibility.

        Returns:
            torch.Tensor: Clipped roughened density using a straight-through custom autograd function.
        """
        if A < 0 or Lc <= 0:
            raise ValueError("add_roughness requires A >= 0 and Lc > 0")
        if density.ndim != 2:
            raise ValueError(
                "EdgeBox roughness requires a 2D density, "
                f"got shape {tuple(density.shape)}"
            )

        X, Y = density.shape
        profiles = self._build_profiles(
            X=X,
            Y=Y,
            device=density.device,
            dtype=density.dtype,
        )
        x_spacing = float(self.design_region_cfg["size"][0]) / max(X - 1, 1)
        y_spacing = float(self.design_region_cfg["size"][1]) / max(Y - 1, 1)
        density_grad_x = torch.gradient(
            density, spacing=(x_spacing,), dim=(0,), edge_order=1
        )[0]
        density_grad_y = torch.gradient(
            density, spacing=(y_spacing,), dim=(1,), edge_order=1
        )[0]
        rough_correction = torch.zeros_like(density)
        edge_band = max(
            float(self.geometry_cfgs.get("border_band_um", 2.0 * self.grid_step)),
            self.grid_step,
        )

        # A displaced interface satisfies rho(x - delta) ~= rho(x) -
        # delta * d rho / d x. Apply that first-order displacement only in a
        # narrow band around explicitly enabled edges. Smooth pair symmetry
        # controls the nominal profiles, but never copies roughness.
        for x_side, x_profile in (
            ("x-", profiles["x_left"]),
            ("x+", profiles["x_right"]),
        ):
            if x_side not in self.enabled_edges:
                continue
            x_roughness = self._normal_roughness_profile(
                x_side,
                profiles["y_coords"],
                profiles["box_ymin"],
                profiles["box_ymax"],
                x_profile,
                A,
                Lc,
                seed,
                num_samples,
            )
            x_distance = profiles["x_coords"][:, None] - x_profile[None, :]
            x_weight = torch.clamp(
                1.0 - x_distance.abs() / edge_band,
                min=0.0,
            )
            rough_correction = rough_correction - density_grad_x * (
                x_roughness[None, :] * x_weight
            )

        for y_side, y_profile in (
            ("y-", profiles["y_bottom"]),
            ("y+", profiles["y_top"]),
        ):
            if y_side not in self.enabled_edges:
                continue
            y_roughness = self._normal_roughness_profile(
                y_side,
                profiles["x_coords"],
                profiles["box_xmin"],
                profiles["box_xmax"],
                y_profile,
                A,
                Lc,
                seed,
                num_samples,
            )
            y_distance = profiles["y_coords"][None, :] - y_profile[:, None]
            y_weight = torch.clamp(
                1.0 - y_distance.abs() / edge_band,
                min=0.0,
            )
            rough_correction = rough_correction - density_grad_y * (
                y_roughness[:, None] * y_weight
            )

        class ClipSTEFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                return input_tensor.clamp(0, 1)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        return ClipSTEFunc.apply(density + rough_correction.detach())

    def apply_post_projection_roughness(self, density: torch.Tensor) -> torch.Tensor:
        """
        Apply configured line-edge roughness after projection, or return the density unchanged when roughness is disabled.

        Args:
            density (torch.Tensor): Density field to perturb.

        Returns:
            torch.Tensor: Post-projection roughened density or the original density.
        """
        cfg = self.line_edge_roughness
        if cfg in (None, False):
            return density
        return self.add_roughness(
            density,
            A=float(cfg["amplitude"]),
            Lc=float(cfg["correlation_length"]),
            seed=cfg.get("seed"),
            num_samples=cfg.get("num_samples"),
        )

    def _boundary_profile_grad_x(
        self,
        side: str,
        profile: torch.Tensor,
        taper: torch.Tensor,
        grad_eps: torch.Tensor,
    ) -> torch.Tensor:
        X, Y = grad_eps.shape
        device = grad_eps.device
        dtype = grad_eps.dtype
        x_coords = torch.linspace(
            self.design_region_cfg["center"][0]
            - self.design_region_cfg["size"][0] / 2.0,
            self.design_region_cfg["center"][0]
            + self.design_region_cfg["size"][0] / 2.0,
            steps=X,
            device=device,
            dtype=dtype,
        )
        half_band = max(
            0.5 * self.grid_step,
            float(self.geometry_cfgs.get("border_band_um", 2.0 * self.grid_step)),
        )
        side_sign = 1.0 if side.endswith("+") else -1.0
        dist = x_coords[:, None] - profile[None, :]
        w = torch.clamp(1.0 - dist.abs() / (half_band + 1e-12), min=0.0)
        grad_profile = (
            side_sign * (grad_eps * w).sum(dim=0) * (self.grid_step * self.grid_step)
        )
        return taper * grad_profile

    def _boundary_profile_grad_y(
        self,
        side: str,
        profile: torch.Tensor,
        taper: torch.Tensor,
        grad_eps: torch.Tensor,
    ) -> torch.Tensor:
        X, Y = grad_eps.shape
        device = grad_eps.device
        dtype = grad_eps.dtype
        y_coords = torch.linspace(
            self.design_region_cfg["center"][1]
            - self.design_region_cfg["size"][1] / 2.0,
            self.design_region_cfg["center"][1]
            + self.design_region_cfg["size"][1] / 2.0,
            steps=Y,
            device=device,
            dtype=dtype,
        )
        half_band = max(
            0.5 * self.grid_step,
            float(self.geometry_cfgs.get("border_band_um", 2.0 * self.grid_step)),
        )
        side_sign = 1.0 if side.endswith("+") else -1.0
        dist = y_coords[None, :] - profile[:, None]
        w = torch.clamp(1.0 - dist.abs() / (half_band + 1e-12), min=0.0)
        grad_profile = (
            side_sign * (grad_eps * w).sum(dim=1) * (self.grid_step * self.grid_step)
        )
        return taper * grad_profile

    def compute_derivatives(
        self, grad_eps: torch.Tensor, eps: torch.Tensor
    ) -> Tuple[Tensor, ...]:
        """
        Compute gradients of the rasterized pattern with respect to the active spline raw control parameters.

        Args:
            grad_eps (torch.Tensor): Adjoint gradient of the rasterized pattern with shape ``[X, Y]``.
            eps (torch.Tensor): Rasterized pattern tensor; retained for API compatibility.

        Returns:
            tuple[torch.Tensor, ...]: One gradient tensor per active edge spline, ordered by ``edge_modules``.

        Raises:
            ValueError: If ``grad_eps`` is not a 2-D ``[X, Y]`` tensor.
        """
        if grad_eps.ndim != 2:
            raise ValueError(
                f"grad_eps must be 2D [X, Y], got shape {tuple(grad_eps.shape)}"
            )

        X, Y = grad_eps.shape
        device = grad_eps.device
        dtype = grad_eps.dtype
        profiles = self._build_profiles(X=X, Y=Y, device=device, dtype=dtype)
        x_pair = profiles["x_pair"]
        y_pair = profiles["y_pair"]

        grads: list[Tensor] = []
        if self.pair_symmetry == "independent":
            for side, module in self.edge_modules.items():
                axis = side[0]
                pair = x_pair if axis == "x" else y_pair
                side_data = pair["independent_profiles"][side]
                if axis == "x":
                    grad_profile = self._boundary_profile_grad_x(
                        side,
                        side_data["profile"],
                        side_data["taper"],
                        grad_eps,
                    )
                else:
                    grad_profile = self._boundary_profile_grad_y(
                        side,
                        side_data["profile"],
                        side_data["taper"],
                        grad_eps,
                    )
                grad_ctrl_phys = side_data["basis"].transpose(0, 1) @ (
                    side_data["taper"] * grad_profile
                )
                grads.append(grad_ctrl_phys * side_data["dctrl_draw"])
            return tuple(grads)

        for axis in ("x", "y"):
            spec = self._canonical_pair_specs[axis]
            active_side = spec["active_side"]
            if active_side is None:
                continue

            if axis == "x":
                active_profile = x_pair["pos"] if active_side == "x+" else x_pair["neg"]
                derived_profile = (
                    x_pair["neg"] if active_side == "x+" else x_pair["pos"]
                )
                active_grad = self._boundary_profile_grad_x(
                    active_side, active_profile, x_pair["taper"], grad_eps
                )
                derived_grad = self._boundary_profile_grad_x(
                    spec["derived_side"], derived_profile, x_pair["taper"], grad_eps
                )
                basis = x_pair["basis"]
                taper = x_pair["taper"]
                dctrl_draw = x_pair["dctrl_draw"]
            else:
                active_profile = y_pair["pos"] if active_side == "y+" else y_pair["neg"]
                derived_profile = (
                    y_pair["neg"] if active_side == "y+" else y_pair["pos"]
                )
                active_grad = self._boundary_profile_grad_y(
                    active_side, active_profile, y_pair["taper"], grad_eps
                )
                derived_grad = self._boundary_profile_grad_y(
                    spec["derived_side"], derived_profile, y_pair["taper"], grad_eps
                )
                basis = y_pair["basis"]
                taper = y_pair["taper"]
                dctrl_draw = y_pair["dctrl_draw"]

            sym_sign = -1.0 if self.pair_symmetry == "mirror" else 1.0
            total_profile_grad = active_grad + sym_sign * derived_grad
            grad_ctrl_phys = basis.transpose(0, 1) @ (taper * total_profile_grad)
            grad_ctrl_raw = grad_ctrl_phys * dctrl_draw
            grads.append(grad_ctrl_raw)

        return tuple(grads)

    def forward(self, permittivity: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluate the spline-controlled edge-box pattern through the custom autograd wrapper.

        Args:
            permittivity (torch.Tensor | None): Optional output tensor passed to ``build_pattern``. Defaults to None.

        Returns:
            torch.Tensor: Rasterized design-region pattern with custom gradients to the spline controls.
        """
        return EdgeBoxDiffWrapper.apply(
            lambda: self.build_pattern(permittivity=permittivity),
            self.compute_derivatives,
            *self._pack_raw_params(),
        )


class EdgeBoxParameterization(BaseParametrization):
    def __init__(
        self,
        *args,
        cfgs: dict = dict(
            method="edgebox",
            geometry_cfgs=dict(
                enabled_edges=("y-", "y+"),
                pair_symmetry="mirror",
                box_center=(0.0, 0.0),
                box_center_unit="relative",
                box_size=(1.0, 0.5),
                box_size_unit="relative",
                rho_resolution={"y-": 50, "y+": 50},
                default_rho_resolution=50,
                degree=3,
                border_band_um=0.02,
                constraints={},
            ),
            transform=[],
            init_method="zeros",
            denorm_mode="linear_eps",
        ),
        **kwargs,
    ):
        """
        Initialize the EdgeBox parameterization and construct its geometry module.

        Args:
            *args: Positional arguments forwarded to ``BaseParametrization``.
            cfgs (dict): EdgeBox configuration containing ``geometry_cfgs``, transforms, initialization, and denormalization settings. Defaults to the built-in EdgeBox configuration defined in the signature.
            **kwargs: Additional arguments forwarded to ``BaseParametrization``.
        """
        super().__init__(*args, cfgs=cfgs, **kwargs)

        method = cfgs["method"]
        self.register_parameter_build_per_region_fn(
            method, self._build_parameters_edgebox
        )
        self.register_parameter_reset_per_region_fn(
            method, self._reset_parameters_edgebox
        )

        self.build_parameters(cfgs, self.design_region_cfg)
        self.reset_parameters(cfgs, self.design_region_cfg)

        self.binary_projection = SubpixelSmoothedProjection(
            projection_midpoint=0.5, resolution=None
        )
        self.eta = torch.tensor([0.5], device=self.operation_device)

    def _build_parameters_edgebox(self, param_cfg, region_cfg):
        """
        Build the per-region ``EdgeBoxGeometry`` module and its parameter metadata.

        Args:
            param_cfg (dict): EdgeBox parameterization configuration.
            region_cfg (dict): Design-region configuration.

        Returns:
            tuple[dict, dict]: Weight mapping containing the ``edgebox`` module and an empty parameter-metadata mapping.
        """
        gcfg = dict(param_cfg["geometry_cfgs"])
        if "phi_resolution" in param_cfg and "phi_resolution" not in gcfg:
            gcfg["phi_resolution"] = param_cfg["phi_resolution"]
        geom = EdgeBoxGeometry(
            sim_cfg=self.hr_device.sim_cfg,
            geometry_cfgs=gcfg,
            region_name=self.region_name,
            design_region_mask=self.hr_design_region_mask,
            design_region_cfg=self.hr_device.design_region_cfgs[self.region_name],
            operation_device=self.operation_device,
        )
        return dict(edgebox=geom), dict()

    def _reset_parameters_edgebox(
        self, weight_dict, param_cfg, region_cfg, init_method: str = "zeros"
    ):
        """
        Reset all active EdgeBox spline control parameters using the selected initialization strategy.

        Args:
            weight_dict (dict[str, torch.nn.Module]): Per-region EdgeBox weights.
            param_cfg (dict): EdgeBox parameterization configuration.
            region_cfg (dict): Design-region configuration.
            init_method (str): Initialization strategy, either ``"zeros"`` or ``"random"``. Defaults to ``"zeros"``.

        Raises:
            ValueError: If ``init_method`` is unsupported.
        """
        geom: EdgeBoxGeometry = weight_dict["edgebox"]
        if init_method not in {"zeros", "random"}:
            raise ValueError(f"Unsupported init_method={init_method!r}")

        for side, mod in geom.edge_modules.items():
            if init_method == "zeros":
                mod.raw_ctrl.data.zero_()
            elif init_method == "random":
                mod.raw_ctrl.data.normal_(mean=0.0, std=0.02)

    def build_density(self, weights, sharpness: float | None = None):
        """
        Build the EdgeBox normalized density by rasterizing the underlying geometry module.

        Args:
            weights (dict[str, torch.nn.Module]): Per-region EdgeBox weight mapping.
            sharpness (float | None): Projection sharpness retained for BaseParametrization compatibility. Defaults to None.

        Returns:
            torch.Tensor: Rasterized EdgeBox density.
        """
        return weights["edgebox"].forward()

    def _uniform_region_coordinates(self, density):
        """
        Construct uniform physical coordinate vectors for the dimensions represented by the EdgeBox density.

        Args:
            density (torch.Tensor): Density whose last spatial dimensions determine the sample counts.

        Returns:
            tuple[torch.Tensor, ...]: Centered physical coordinate vectors for the configured dimensions.
        """
        dims = sorted(self.cfgs.get("dims", (0, 1)))
        center = self.design_region_cfg["center"]
        size = self.design_region_cfg["size"]
        return tuple(
            torch.linspace(
                float(center[dim] - size[dim] / 2.0),
                float(center[dim] + size[dim] / 2.0),
                density.shape[index],
                device=density.device,
                dtype=density.dtype,
            )
            for index, dim in enumerate(dims)
        )

    def _region_coordinates(self, device, region_mask):
        """
        Read native/high-resolution optical-grid coordinates inside a design-region mask.

        Args:
            device (object): Device-like object exposing ``optical_grid_metadata``.
            region_mask (tuple[slice, ...]): Mask selecting the design-region coordinates.

        Returns:
            tuple[torch.Tensor, ...] | None: Physical coordinate vectors or None when metadata is unavailable.
        """
        metadata = getattr(device, "optical_grid_metadata", None)
        if metadata is None or metadata.get("coords") is None:
            return None
        dims = sorted(self.cfgs.get("dims", (0, 1)))
        return tuple(
            torch.as_tensor(
                metadata["coords"][dim][region_mask[dim]],
                dtype=torch.float32,
                device=self.operation_device,
            )
            for dim in dims
        )

    def _resample_density_to_hr(self, density, target_size):
        """
        Resample EdgeBox density from its uniform parameterization grid to the high-resolution simulation grid.

        Args:
            density (torch.Tensor): Parameterization-grid density.
            target_size (tuple[int, ...]): Target spatial size.

        Returns:
            torch.Tensor: High-resolution density.
        """
        source_coords = self._uniform_region_coordinates(density)
        target_coords = self._region_coordinates(
            self.hr_device, self.hr_design_region_mask
        )
        averaged = self._resample_volume_fraction_on_grid(
            density, source_coords, self.hr_device, self.hr_design_region_mask
        )
        if averaged is not None:
            return averaged
        if target_coords is None:
            return _convert_resolution(
                density, intplt_mode="nearest", target_size=target_size
            )
        return _bilinear_from_knots(
            density,
            source_coords[0],
            source_coords[1],
            target_coords[0],
            target_coords[1],
        )

    def _resample_density_to_native(self, density, target_size, eps_r, eps_bg):
        """
        Resample EdgeBox density to the native simulation grid, using material-aware interpolation when coordinate metadata is unavailable.

        Args:
            density (torch.Tensor): Parameterization-grid density.
            target_size (tuple[int, ...]): Target native spatial size.
            eps_r (float | torch.Tensor): Design-region permittivity.
            eps_bg (float | torch.Tensor): Background permittivity.

        Returns:
            torch.Tensor: Native-grid density.
        """
        source_coords = self._uniform_region_coordinates(density)
        target_coords = self._region_coordinates(self.device, self.design_region_mask)
        averaged = self._resample_volume_fraction_on_grid(
            density, source_coords, self.device, self.design_region_mask
        )
        if averaged is not None:
            return averaged
        if target_coords is None:
            return average_uniform_cells(density, target_size)
        return _bilinear_from_knots(
            density,
            source_coords[0],
            source_coords[1],
            target_coords[0],
            target_coords[1],
        )

    def add_roughness(
        self,
        density: torch.Tensor,
        *,
        A: float,
        Lc: float,
        seed: int | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Delegate post-projection edge roughness generation to the underlying EdgeBoxGeometry module.

        Args:
            density (torch.Tensor): Density field.
            A (float): Roughness amplitude.
            Lc (float): Correlation length.
            seed (int | None): Optional random seed. Defaults to None.
            num_samples (int | None): Optional roughness sample count. Defaults to None.
            **kwargs: Additional roughness options.

        Returns:
            torch.Tensor: Roughened density.
        """
        return self.weights["edgebox"].add_roughness(
            density,
            A=A,
            Lc=Lc,
            seed=seed,
            num_samples=num_samples,
            **kwargs,
        )

    def post_projection_transform(self, density):
        """
        Apply the geometry module's configured post-projection roughness transform.

        Args:
            density (torch.Tensor): Projected density field.

        Returns:
            torch.Tensor: Transformed density.
        """
        return self.weights["edgebox"].apply_post_projection_roughness(density)

    def build_permittivity(self, weights, sharpness: float | None = None):
        """
        Build the EdgeBox permittivity representation using the same rasterization path as density construction.

        Args:
            weights (dict[str, torch.nn.Module]): EdgeBox weight mapping.
            sharpness (float | None): Projection sharpness retained for interface compatibility. Defaults to None.

        Returns:
            torch.Tensor: Rasterized EdgeBox density/permittivity representation.
        """
        return self.build_density(weights, sharpness)
