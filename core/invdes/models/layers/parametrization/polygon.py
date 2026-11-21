from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.types import Device

from .base_parametrization import BaseParametrization

__all__ = ["PolygonGeometry", "PolygonParameterization"]


class PolygonDiffWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_vertices, build_fn, grad_fn):
        eps = build_fn()
        ctx.grad_fn = grad_fn
        ctx.save_for_backward(raw_vertices.detach(), eps)
        return eps

    @staticmethod
    def backward(ctx, grad_output):
        raw_vertices, eps = ctx.saved_tensors
        grad_fn = ctx.grad_fn
        grad_raw_vertices = grad_fn(grad_output, eps, raw_vertices)
        return grad_raw_vertices, None, None


def _polygon_signed_area(vertices: torch.Tensor) -> torch.Tensor:
    """Compute signed polygon area; positive values indicate counter-clockwise ordering.

    Args:
        vertices (torch.Tensor): Tensor with a trailing vertex-coordinate dimension
            ``[..., V, 2]``.

    Returns:
        torch.Tensor: Signed area for each leading polygon index.
    """
    x = vertices[..., 0]
    y = vertices[..., 1]
    x1 = torch.roll(x, shifts=-1, dims=-1)
    y1 = torch.roll(y, shifts=-1, dims=-1)
    return 0.5 * torch.sum(x * y1 - x1 * y, dim=-1)


def _point_in_polygon(
    xx: torch.Tensor, yy: torch.Tensor, vertices: torch.Tensor
) -> torch.Tensor:
    """Evaluate polygon membership on a 2-D coordinate grid with ray casting.

    Args:
        xx (torch.Tensor): ``[H, W]`` grid of ``x`` coordinates.
        yy (torch.Tensor): ``[H, W]`` grid of ``y`` coordinates.
        vertices (torch.Tensor): ``[V, 2]`` polygon vertices ordered around the
            boundary.

    Returns:
        torch.Tensor: ``[H, W]`` boolean tensor indicating points inside the polygon.
    """
    vx = vertices[:, 0]
    vy = vertices[:, 1]
    vx1 = torch.roll(vx, shifts=-1, dims=0)
    vy1 = torch.roll(vy, shifts=-1, dims=0)

    x = xx[..., None]
    y = yy[..., None]

    # Half-open interval on y to avoid double counting vertices.
    cond = ((vy > y) != (vy1 > y)) & (
        x < (vx1 - vx) * (y - vy) / (vy1 - vy + 1e-12) + vx
    )
    return (cond.sum(dim=-1) % 2) == 1


def _catmull_rom_boundary(
    control_points: torch.Tensor, samples_per_segment: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a closed uniform Catmull-Rom boundary and return interpolation weights.

    Args:
        control_points (torch.Tensor): ``[V, 2]`` polygon control vertices.
        samples_per_segment (int): Number of boundary samples generated per control
            segment.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Sampled boundary points and a basis matrix
        mapping control vertices to sampled boundary coordinates.

    Raises:
        ValueError: If ``samples_per_segment`` is less than 1.
    """
    if samples_per_segment < 1:
        raise ValueError("samples_per_segment must be at least 1")

    n_points = control_points.shape[0]
    device = control_points.device
    dtype = control_points.dtype
    t = (
        torch.arange(samples_per_segment, device=device, dtype=dtype)
        / samples_per_segment
    )
    t2 = t * t
    t3 = t2 * t
    weights = torch.stack(
        (
            -0.5 * t3 + t2 - 0.5 * t,
            1.5 * t3 - 2.5 * t2 + 1.0,
            -1.5 * t3 + 2.0 * t2 + 0.5 * t,
            0.5 * t3 - 0.5 * t2,
        ),
        dim=-1,
    )

    segment_indices = torch.arange(n_points, device=device)
    sample_indices = (
        segment_indices[:, None] + torch.tensor([-1, 0, 1, 2], device=device)
    ) % n_points
    basis = torch.zeros(
        n_points * samples_per_segment,
        n_points,
        device=device,
        dtype=dtype,
    )
    basis.scatter_add_(
        1,
        sample_indices[:, None, :].expand(-1, samples_per_segment, -1).reshape(-1, 4),
        weights[None, :, :].expand(n_points, -1, -1).reshape(-1, 4),
    )
    boundary = basis @ control_points
    return boundary, basis


def _overlap_1d_centers(centers: torch.Tensor, a: float, b: float, g: float):
    """Compute physical overlap length between pixel cells and a 1-D interval.

    Args:
        centers (torch.Tensor): ``[K]`` pixel-center coordinates.
        a (float): Interval start coordinate.
        b (float): Interval end coordinate.
        g (float): Pixel size in the same physical units as ``centers``, ``a``, and ``b``.

    Returns:
        torch.Tensor: ``[K]`` overlap lengths, clamped to ``[0, g]``.
    """
    half = g * 0.5
    left = torch.maximum(
        centers - half, torch.tensor(a, device=centers.device, dtype=centers.dtype)
    )
    right = torch.minimum(
        centers + half, torch.tensor(b, device=centers.device, dtype=centers.dtype)
    )
    return torch.clamp(right - left, min=0.0, max=g)


def _polygon_edge_band_grads(
    vertices: torch.Tensor,
    grad_eps: torch.Tensor,
    region_center: Tuple[float, float],
    region_size: Tuple[float, float],
    grid_step: float,
    band_um: float,
) -> torch.Tensor:
    """Approximate vertex gradients by integrating a narrow band around polygon edges.

    Args:
        vertices (torch.Tensor): ``[V, 2]`` polygon vertices in physical coordinates.
        grad_eps (torch.Tensor): ``[H, W]`` upstream permittivity gradient on the
            design-region grid.
        region_center (tuple[float, float]): Physical design-region center.
        region_size (tuple[float, float]): Physical design-region size.
        grid_step (float): Grid spacing in physical units.
        band_um (float): Target edge-band width in physical units.

    Returns:
        torch.Tensor: ``[V, 2]`` approximate gradients for vertex coordinates.
    """
    device = vertices.device
    dtype = vertices.dtype
    H, W = grad_eps.shape

    cx, cy = float(region_center[0]), float(region_center[1])
    sx, sy = float(region_size[0]), float(region_size[1])
    g = float(grid_step)

    x_left = cx - sx / 2.0
    x_right = cx + sx / 2.0
    y_bottom = cy - sy / 2.0
    y_top = cy + sy / 2.0

    x_coords = torch.linspace(x_left, x_right, steps=H, device=device, dtype=dtype)
    y_coords = torch.linspace(y_bottom, y_top, steps=W, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing="ij")

    half_band = max(0.5 * band_um, 0.5 * g)
    cell_area = g * g

    area = _polygon_signed_area(vertices)
    ccw = bool(area >= 0)

    V = vertices.shape[0]
    edge_forces = torch.zeros((V, 2), device=device, dtype=dtype)

    for i in range(V):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % V]
        e = p1 - p0
        L = torch.linalg.norm(e)
        if float(L) < 1e-12:
            continue

        t = e / (L + 1e-12)
        # Outward normal for CCW polygons is the right normal.
        n = torch.stack([t[1], -t[0]])
        if not ccw:
            n = -n

        relx = xx - p0[0]
        rely = yy - p0[1]
        proj = relx * t[0] + rely * t[1]
        dist = relx * n[0] + rely * n[1]

        band = (proj >= 0) & (proj <= L) & (dist.abs() <= half_band)
        # triangular taper in the edge-normal direction
        w = torch.clamp(1.0 - dist.abs() / (half_band + 1e-12), min=0.0)

        edge_scalar = (grad_eps * w * band).sum() * cell_area
        edge_forces[i] = edge_scalar * n

    grad_vertices = torch.zeros_like(vertices)
    for i in range(V):
        grad_vertices[i] += 0.5 * edge_forces[i - 1] + 0.5 * edge_forces[i]

    return grad_vertices


class PolygonGeometry(nn.Module):
    """Batched polygon geometry with learnable vertices and differentiable rasterization.

    Args:
        sim_cfg (dict[str, Any]): Simulation configuration containing the base resolution.
        geometry_cfgs (dict[str, Any]): Polygon configuration, including optional
            ``batch_dims``, ``n_vertices``, ``rho_resolution``, border-band width,
            interpolation mode, sample count, and fixed vertices.
        region_name (str): Design-region identifier.
        design_region_cfg (dict[str, Any]): Physical design-region configuration containing
            ``center`` and ``size``.
        operation_device (torch.device): Device for learnable vertex parameters.
            Defaults to ``torch.device("cuda:0")``.

    Attributes:
        raw_vertices (nn.Parameter): Relative vertex coordinates with shape
            ``batch_dims + (n_vertices, 2)`` in micrometers.
        n_vertices (int): Number of polygon control vertices.
        grid_step (float): Base pixel spacing derived as ``1 / sim_cfg["resolution"]``.
        boundary_interpolation (str): ``"linear"`` or ``"catmull_rom"``.
        samples_per_segment (int): Catmull-Rom samples per control segment.
        fixed_vertices (dict[int, tuple[float, float]]): Vertices restored after clipping.
    """

    def __init__(
        self,
        sim_cfg: Dict[str, Any],
        geometry_cfgs: Dict[str, Any],
        region_name: str,
        design_region_cfg: Dict[str, Any],
        operation_device: Device = torch.device("cuda:0"),
    ):
        """Initialize polygon geometry parameters and rasterization configuration.

        Args:
            sim_cfg (dict[str, Any]): Simulation configuration containing ``resolution``.
            geometry_cfgs (dict[str, Any]): Polygon geometry configuration.
            region_name (str): Design-region identifier.
            design_region_cfg (dict[str, Any]): Physical region configuration containing
                ``center`` and ``size``.
            operation_device (torch.device): Device for learnable vertex parameters.
                Defaults to ``torch.device("cuda:0")``.

        Raises:
            AssertionError: If fewer than three polygon vertices are requested.
            ValueError: If interpolation mode, sample count, or fixed-vertex configuration
                is invalid.
        """
        super().__init__()
        self.sim_cfg = sim_cfg
        self.geometry_cfgs = geometry_cfgs
        self.region_name = region_name
        self.design_region_cfg = design_region_cfg
        self.operation_device = operation_device

        batch_dims = self.geometry_cfgs.get("batch_dims", (1,))
        if isinstance(batch_dims, int):
            batch_dims = (batch_dims,)
        self.batch_dims = tuple(batch_dims)

        self.n_vertices = int(self.geometry_cfgs.get("n_vertices", 6))
        assert self.n_vertices >= 3, "Polygon requires at least 3 vertices."

        self.grid_step = 1.0 / float(self.sim_cfg["resolution"])
        self.rho_resolution = self.geometry_cfgs.get(
            "rho_resolution", self.sim_cfg["resolution"]
        )
        self.border_band_um = float(
            self.geometry_cfgs.get("border_band_um", 2.0 * self.grid_step)
        )
        self.boundary_interpolation = str(
            self.geometry_cfgs.get("boundary_interpolation", "linear")
        ).lower()
        if self.boundary_interpolation not in {"linear", "catmull_rom"}:
            raise ValueError("boundary_interpolation must be 'linear' or 'catmull_rom'")
        self.samples_per_segment = int(self.geometry_cfgs.get("samples_per_segment", 4))
        if self.samples_per_segment < 1:
            raise ValueError("samples_per_segment must be at least 1")
        self.fixed_vertices = {
            int(index): tuple(float(value) for value in position)
            for index, position in self.geometry_cfgs.get("fixed_vertices", {}).items()
        }
        for index, position in self.fixed_vertices.items():
            if not 0 <= index < self.n_vertices:
                raise ValueError(f"fixed vertex index {index} is outside the polygon")
            if len(position) != 2:
                raise ValueError("fixed vertex positions must be [x, y] pairs")

        self.raw_vertices = nn.Parameter(
            torch.zeros(
                *self.batch_dims, self.n_vertices, 2, device=self.operation_device
            )
        )

    def _physical_vertices(self) -> torch.Tensor:
        """Convert learnable relative vertices into physical coordinates.

        Returns:
            torch.Tensor: Vertex tensor with the same batch shape as ``raw_vertices``;
            the design-region center is added to the final ``[V, 2]`` coordinates.
        """
        center = torch.as_tensor(
            self.design_region_cfg["center"][:2],
            device=self.raw_vertices.device,
            dtype=self.raw_vertices.dtype,
        )
        return center + self.raw_vertices

    def _boundary_vertices(self) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Build the linear or Catmull-Rom polygon boundary from physical vertices.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Boundary coordinates and,
            for Catmull-Rom interpolation, the basis matrix mapping control vertices
            to sampled boundary points. Linear interpolation returns ``None`` for the basis.
        """
        physical = self._physical_vertices()
        if self.boundary_interpolation == "linear":
            return physical, None

        flat = physical.reshape(-1, self.n_vertices, 2)
        boundaries = []
        bases = []
        for control_points in flat:
            boundary, basis = _catmull_rom_boundary(
                control_points, self.samples_per_segment
            )
            boundaries.append(boundary)
            bases.append(basis)
        return torch.stack(boundaries).reshape(
            *physical.shape[:-2], -1, 2
        ), torch.stack(bases).reshape(*physical.shape[:-2], -1, self.n_vertices)

    def clip_vertices_(self) -> None:
        """Clip free vertices to the design-region bounds and restore fixed vertices.

        The operation is performed in-place under ``torch.no_grad()``.
        """
        size = torch.as_tensor(
            self.design_region_cfg["size"][:2],
            device=self.raw_vertices.device,
            dtype=self.raw_vertices.dtype,
        )
        with torch.no_grad():
            self.raw_vertices.clamp_(-0.5 * size, 0.5 * size)
            for index, position in self.fixed_vertices.items():
                self.raw_vertices[..., index, :] = torch.as_tensor(
                    position,
                    device=self.raw_vertices.device,
                    dtype=self.raw_vertices.dtype,
                )

    def build_pattern(self, permittivity: torch.Tensor = None) -> torch.Tensor:
        """Rasterize polygon interiors into a relative-permittivity pattern.

        Args:
            permittivity (torch.Tensor | None): Optional existing raster tensor to modify
                where the polygon mask is true. Defaults to None; a zero tensor is allocated.

        Returns:
            torch.Tensor: Rasterized tensor whose polygon-covered cells are set to
            relative permittivity ``1.0`` and whose other cells retain the supplied
            background tensor values.
        """
        eps_r = 1.0
        region_size = self.design_region_cfg["size"][:2]
        if isinstance(self.rho_resolution, dict):
            resolutions = [
                float(self.rho_resolution.get(axis, self.sim_cfg["resolution"]))
                for axis in ("x", "y")
            ]
        else:
            resolutions = [float(self.rho_resolution)] * 2
        region_px_size = [
            max(2, int(round(float(region_size[axis]) * resolutions[axis])))
            for axis in range(2)
        ]
        eps = (
            torch.zeros(
                region_px_size,
                device=self.operation_device,
                dtype=self.raw_vertices.dtype,
            )
            if permittivity is None
            else permittivity
        )

        center = self.design_region_cfg["center"][:2]
        size = self.design_region_cfg["size"][:2]
        H, W = eps.shape[-2], eps.shape[-1]

        x_left = center[0] - size[0] / 2.0
        y_top = center[1] + size[1] / 2.0

        x_coords = torch.linspace(
            x_left,
            center[0] + size[0] / 2.0,
            steps=H,
            device=eps.device,
            dtype=eps.dtype,
        )
        y_coords = torch.linspace(
            center[1] - size[1] / 2.0,
            y_top,
            steps=W,
            device=eps.device,
            dtype=eps.dtype,
        )
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing="ij")

        mask = torch.zeros((H, W), device=eps.device, dtype=torch.bool)
        vertices, _ = self._boundary_vertices()
        vertices = vertices.reshape(-1, vertices.shape[-2], 2)

        for poly in vertices:
            mask |= _point_in_polygon(xx, yy, poly)

        eps = torch.where(mask, torch.full_like(eps, eps_r), eps)
        return eps

    def compute_derivatives(
        self,
        grad_eps: torch.Tensor,
        eps: torch.Tensor,
        raw_vertices: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate gradients with respect to polygon vertex coordinates.

        Args:
            grad_eps (torch.Tensor): Upstream raster gradient, expected as ``[H, W]``.
            eps (torch.Tensor): Rasterized polygon pattern saved by the custom autograd
                wrapper.
            raw_vertices (torch.Tensor): Original learnable relative vertices whose
                shape determines the returned gradient layout.

        Returns:
            torch.Tensor: Gradient tensor reshaped like ``raw_vertices``. Catmull-Rom
            interpolation applies the sampled-boundary basis transpose to map gradients
            back to control vertices.
        """
        center = self.design_region_cfg["center"][:2]
        size = self.design_region_cfg["size"][:2]

        phys, basis = self._boundary_vertices()
        phys = phys.reshape(-1, phys.shape[-2], 2)
        if basis is not None:
            basis = basis.reshape(-1, basis.shape[-2], basis.shape[-1])

        grad_phys = torch.zeros_like(phys)
        for i, poly in enumerate(phys):
            grad_phys[i] = _polygon_edge_band_grads(
                vertices=poly,
                grad_eps=grad_eps,
                region_center=tuple(center),
                region_size=tuple(size),
                grid_step=self.grid_step,
                band_um=self.border_band_um,
            )

        if basis is not None:
            grad_control = torch.bmm(basis.transpose(1, 2), grad_phys)
            return grad_control.reshape_as(raw_vertices)
        return grad_phys.reshape_as(raw_vertices)

    def forward(self, permittivity: torch.Tensor = None) -> torch.Tensor:
        """Run differentiable polygon rasterization through the custom autograd wrapper.

        Args:
            permittivity (torch.Tensor | None): Optional background raster passed to
                ``build_pattern``. Defaults to None.

        Returns:
            torch.Tensor: Rasterized relative-permittivity pattern with a custom backward
            path that maps border-band raster gradients to vertex gradients.
        """
        return PolygonDiffWrapper.apply(
            self.raw_vertices,
            lambda: self.build_pattern(permittivity=permittivity),
            self.compute_derivatives,
        )


class PolygonParameterization(BaseParametrization):
    """Register polygon geometry as a MAPS design-region parameterization method.

    The parameterization builds a ``PolygonGeometry`` module per design region and
    supports initialization modes such as ``random`` and ``regular_ngon``.
    """

    def __init__(
        self,
        *args,
        cfgs: dict = dict(
            method="polygon",
            geometry_cfgs=dict(
                batch_dims=(1,),
                n_vertices=6,
                border_band_um=0.02,
            ),
            transform=[],
            init_method="regular_ngon",
            denorm_mode="linear_eps",
        ),
        **kwargs,
    ):
        """Initialize and register the polygon parameter build/reset callbacks.

        Args:
            *args: Positional arguments forwarded to ``BaseParametrization``.
            cfgs (dict): Polygon parameterization configuration. Defaults to the
                built-in polygon configuration declared in the function signature.
            **kwargs: Keyword arguments forwarded to ``BaseParametrization``.
        """
        super().__init__(*args, cfgs=cfgs, **kwargs)

        method = cfgs["method"]
        self.register_parameter_build_per_region_fn(
            method, self._build_parameters_polygon
        )
        self.register_parameter_reset_per_region_fn(
            method, self._reset_parameters_polygon
        )

        self.build_parameters(cfgs, self.design_region_cfg)
        self.reset_parameters(cfgs, self.design_region_cfg)

    def _build_parameters_polygon(self, param_cfg, region_cfg):
        """Build the ``PolygonGeometry`` module and per-region metadata.

        Args:
            param_cfg (dict[str, Any]): Polygon parameterization configuration containing
                the ``geometry_cfgs`` mapping.
            region_cfg (dict[str, Any]): Design-region configuration supplied by the
                base parameterization lifecycle.

        Returns:
            tuple[dict[str, PolygonGeometry], dict[str, Any]]: Parameter dictionary
            containing the ``polygon`` module and an empty auxiliary-state dictionary.
        """
        polygon = PolygonGeometry(
            sim_cfg=self.hr_device.sim_cfg,
            geometry_cfgs=param_cfg["geometry_cfgs"],
            region_name=self.region_name,
            design_region_cfg=self.hr_device.design_region_cfgs[self.region_name],
            operation_device=self.operation_device,
        )
        return dict(polygon=polygon), dict()

    def _reset_parameters_polygon(
        self, weight_dict, param_cfg, region_cfg, init_method: str = "regular_ngon"
    ):
        """Reset polygon vertices using a callable or named initialization strategy.

        Args:
            weight_dict (dict[str, Any]): Parameter dictionary containing the ``polygon``
                geometry module.
            param_cfg (dict[str, Any]): Polygon parameterization configuration.
            region_cfg (dict[str, Any]): Physical design-region configuration containing
                center and size.
            init_method (str | callable): Initialization mode. Supported strings are
                ``"random"`` and ``"regular_ngon"``. Defaults to ``"regular_ngon"``.

        Raises:
            ValueError: If ``init_method`` is not supported.
        """
        poly: PolygonGeometry = weight_dict["polygon"]

        if callable(init_method):
            init_method(weight_dict, param_cfg, region_cfg)
            return

        center = torch.as_tensor(
            region_cfg["center"][:2],
            device=self.operation_device,
            dtype=poly.raw_vertices.dtype,
        )
        size = torch.as_tensor(
            region_cfg["size"][:2],
            device=self.operation_device,
            dtype=poly.raw_vertices.dtype,
        )
        half = 0.5 * size

        if init_method == "random":
            nn.init.normal_(poly.raw_vertices, mean=0.0, std=0.1)
            return

        if init_method == "regular_ngon":
            V = poly.n_vertices
            theta = torch.linspace(
                0.0,
                2.0 * torch.pi,
                steps=V + 1,
                device=self.operation_device,
                dtype=poly.raw_vertices.dtype,
            )[:-1]
            radius = 0.35 * torch.min(half)
            verts = torch.stack(
                [radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1
            )
            verts = verts.unsqueeze(0).repeat(*poly.batch_dims, 1, 1)
            verts = center + verts

            # Invert the tanh map back to raw coordinates.
            poly.raw_vertices.data.copy_(verts - center)
            return

        raise ValueError(f"Unsupported polygon init method: {init_method}")

    def build_density(self, weights, sharpness: float, **kwargs):
        """Build the polygon density pattern used by the parameterization lifecycle.

        Args:
            weights (dict[str, Any]): Parameter dictionary containing the ``polygon``
                geometry module.
            sharpness (float): Projection/sharpness parameter accepted for interface
                compatibility; polygon rasterization does not currently use it.
            **kwargs: Additional lifecycle keyword arguments accepted for compatibility.

        Returns:
            torch.Tensor: Polygon rasterized density returned by the geometry module.
        """
        return weights["polygon"].forward()

    def build_permittivity(self, weights, sharpness: float, **kwargs):
        """Build the polygon permittivity pattern through the density implementation.

        Args:
            weights (dict[str, Any]): Parameter dictionary containing the ``polygon``
                geometry module.
            sharpness (float): Projection/sharpness parameter accepted for interface
                compatibility. The current implementation delegates directly to density.
            **kwargs: Additional lifecycle keyword arguments accepted for compatibility.

        Returns:
            torch.Tensor: Polygon rasterized pattern returned by ``build_density``.
        """
        return self.build_density(weights, sharpness, **kwargs)
