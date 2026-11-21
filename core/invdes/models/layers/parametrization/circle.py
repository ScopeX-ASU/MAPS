"""Circular and z-axis cylindrical digital geometry parameterizations."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import torch
from torch import Tensor, nn
from torch.types import Device

from .base_parametrization import BaseParametrization
from .geometry import BatchGeometry
from .material_projection import EffectiveMediumProjector

__all__ = [
    "CircleParameterization",
    "CylinderParameterization",
    "Cylinders",
    "Circles",
]


class CircleDiffWrapper(torch.autograd.Function):
    """Attach an aggregate geometry derivative to a detached circle raster."""

    @staticmethod
    def forward(ctx, centers, sizes, material_values, build_fn, grad_fn):
        density = build_fn()
        ctx.grad_fn = grad_fn
        ctx.save_for_backward(density)
        return density

    @staticmethod
    def backward(ctx, grad_output):
        density = ctx.saved_tensors[0]
        grad_centers, grad_sizes, grad_material = ctx.grad_fn(grad_output, density)
        return grad_centers, grad_sizes, grad_material, None, None


def _circle_coverage(
    center: Tensor,
    radius: Tensor,
    *,
    region_center: tuple[float, float],
    region_size: tuple[float, float],
    grid_step: float,
    shape: tuple[int, int],
    samples_per_pixel: int,
) -> Tensor:
    """Approximate a circle's area fraction in each Cartesian image cell.

    Args:
        center: Circle center ``[x, y]`` in micrometers.
        radius: Circle radius in micrometers.
        region_center: Physical center of the raster region.
        region_size: Physical ``(x, y)`` size of the raster region.
        grid_step: Uniform cell pitch in micrometers.
        shape: Raster shape ``(nx, ny)``.
        samples_per_pixel: Number of midpoint samples along each cell axis.

    Returns:
        Tensor: Per-cell coverage fractions in Torch's ``y_up`` layout, with
        shape ``(nx, ny)`` and values in ``[0, 1]``.

    Notes:
        This is deliberately a uniform-grid area-fraction model. It is used by
        the Torch fallback and by the manually aggregated geometry derivatives;
        it does not differentiate through the sampling operation itself.
    """
    if samples_per_pixel < 1:
        raise ValueError("samples_per_pixel must be at least one")

    nx, ny = shape
    x_left = float(region_center[0]) - float(region_size[0]) / 2.0
    y_top = float(region_center[1]) + float(region_size[1]) / 2.0
    cx, cy = center.unbind()
    radius = radius.clamp_min(0.0)

    # Restrict the supersampling work to cells that can intersect the circle.
    r = float(radius.detach())
    if r <= 0.0:
        return torch.zeros(shape, device=center.device, dtype=center.dtype)
    x_min = float(cx.detach()) - r
    x_max = float(cx.detach()) + r
    y_min = float(cy.detach()) - r
    y_max = float(cy.detach()) + r
    i0 = max(0, min(nx - 1, int((x_min - x_left) / grid_step) - 1))
    i1 = max(0, min(nx - 1, int((x_max - x_left) / grid_step) + 1))
    j0 = max(0, min(ny - 1, int((y_top - y_max) / grid_step) - 1))
    j1 = max(0, min(ny - 1, int((y_top - y_min) / grid_step) + 1))
    if i1 < i0 or j1 < j0:
        return torch.zeros(shape, device=center.device, dtype=center.dtype)

    x = (
        x_left
        + (torch.arange(i0, i1 + 1, device=center.device, dtype=center.dtype) + 0.5)
        * grid_step
    )
    y = (
        y_top
        - (torch.arange(j0, j1 + 1, device=center.device, dtype=center.dtype) + 0.5)
        * grid_step
    )
    offsets = (
        (
            torch.arange(samples_per_pixel, device=center.device, dtype=center.dtype)
            + 0.5
        )
        / samples_per_pixel
        - 0.5
    ) * grid_step
    sample_x = x[:, None, None, None] + offsets[None, None, :, None]
    sample_y = y[None, :, None, None] + offsets[None, None, None, :]
    inside = (sample_x - cx).square() + (sample_y - cy).square() <= radius.square()
    coverage = inside.to(center.dtype).mean(dim=(-1, -2))

    result = torch.zeros(shape, device=center.device, dtype=center.dtype)
    result[i0 : i1 + 1, j0 : j1 + 1] = coverage
    return result


class Circles(BatchGeometry):
    """Parameterize a batch of circular footprints or z-axis cylinders.

    The stored center is ``(x, y)`` and ``sizes[..., 0]`` is the radius, all in
    micrometers. For a 3-D device, the resulting 2-D density is extruded by the
    existing parameterization pipeline; the cylinder height and etch profile
    therefore remain properties of the device/materialization configuration.
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
        super().__init__(
            sim_cfg,
            {**geometry_cfgs, "size_dim": 1},
            region_name=region_name,
            design_region_mask=design_region_mask,
            design_region_cfg=design_region_cfg,
            operation_device=operation_device,
        )
        phi_resolution = geometry_cfgs.get("phi_resolution", sim_cfg["resolution"])
        if isinstance(phi_resolution, (int, float)):
            phi_resolution = (float(phi_resolution), float(phi_resolution))
        else:
            phi_resolution = tuple(float(value) for value in phi_resolution)
        if len(phi_resolution) != 2 or any(value <= 0 for value in phi_resolution):
            raise ValueError(
                "phi_resolution must contain two positive values in pixels per um"
            )
        if phi_resolution[0] != phi_resolution[1]:
            raise ValueError("Circles currently require an isotropic phi_resolution")
        self.phi_resolution = phi_resolution
        region_size = self.design_region_cfg["size"]
        self.phi_shape = tuple(
            max(1, int(round(float(region_size[axis]) * phi_resolution[axis])))
            for axis in range(2)
        )
        self.grid_step = 1.0 / phi_resolution[0]
        self.coverage_samples = int(geometry_cfgs.get("coverage_samples", 8))
        self.gradient_step_um = float(
            geometry_cfgs.get("gradient_step_um", 0.25 * self.grid_step)
        )
        if self.gradient_step_um <= 0:
            raise ValueError("gradient_step_um must be positive")
        self.min_radius_um = float(geometry_cfgs.get("min_radius_um", 1.0e-4))
        if self.min_radius_um < 0:
            raise ValueError("min_radius_um must be nonnegative")
        configured_max_radius = geometry_cfgs.get("max_radius_um", None)
        self.max_radius_um = (
            None if configured_max_radius is None else float(configured_max_radius)
        )
        if self.max_radius_um is not None and self.max_radius_um < 0:
            raise ValueError("max_radius_um must be nonnegative")
        if self.max_radius_um is not None and self.min_radius_um > self.max_radius_um:
            raise ValueError("min_radius_um cannot exceed max_radius_um")
        self.small_circle_policy = str(
            geometry_cfgs.get("small_circle_policy", "remove")
        ).lower()
        if self.small_circle_policy not in {
            "remove",
            "clamp",
            "snap_by_radius",
            "snap_by_area",
        }:
            raise ValueError(
                "small_circle_policy must be 'remove', 'clamp', "
                "'snap_by_radius', or 'snap_by_area'"
            )
        if (
            self.small_circle_policy in {"clamp", "snap_by_radius", "snap_by_area"}
            and self.min_radius_um > self._max_radius()
        ):
            raise ValueError(
                "min_radius_um cannot exceed the effective maximum radius "
                "when small_circle_policy is 'clamp' or a snap policy"
            )
        self.zero_radius_gradient_scale = float(
            geometry_cfgs.get("zero_radius_gradient_scale", 1.0 / self.grid_step)
        )
        if self.zero_radius_gradient_scale < 0:
            raise ValueError("zero_radius_gradient_scale must be nonnegative")
        self.use_gpu_subpixel = bool(geometry_cfgs.get("gpu_subpixel", False))
        self.gpu_circle_sides = int(geometry_cfgs.get("gpu_circle_sides", 128))
        if self.gpu_circle_sides < 8:
            raise ValueError("gpu_circle_sides must be at least eight")
        self._gpu_subpixel_grid = None
        self._gpu_subpixel_rasterizer = None
        self.material_projector = EffectiveMediumProjector(
            geometry_cfgs.get("material_model", "linear_eps")
        )

    def uniform_parameter_grid_coords(self):
        """Return cell-center coordinates for the internal uniform circle grid."""
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

    @property
    def radius(self) -> Tensor:
        """Return the learnable radius tensor in micrometers."""
        return self.sizes[..., 0]

    def _material_values(self):
        """Return the normalized material values currently used by rasterization."""
        if not self.material_enabled:
            return None
        if self.optimization_stage in {"material", "joint"}:
            return self.material_rho
        return self.material_state

    def _max_radius(self) -> float:
        """Return the configured and lattice-derived radius upper bound."""
        size = self.design_region_cfg["size"]
        n_x, n_y = self.centers.shape[:2]
        border = float(self.geometry_cfgs.get("border", 0.2))
        pitch_x = (float(size[0]) - 2.0 * border) / n_x
        pitch_y = (float(size[1]) - 2.0 * border) / n_y
        lattice_max = max(0.0, 0.5 * min(pitch_x, pitch_y))
        if self.max_radius_um is None:
            return lattice_max
        return min(lattice_max, self.max_radius_um)

    def _effective_radii(self, radii: Tensor | None = None) -> Tensor:
        """Apply configured lower and upper radius bounds before rasterization."""
        radii = self.radius if radii is None else radii
        max_radius = torch.as_tensor(
            self._max_radius(), device=radii.device, dtype=radii.dtype
        )
        radii = radii.clamp(min=0.0, max=max_radius)
        threshold = torch.as_tensor(
            self.min_radius_um, device=radii.device, dtype=radii.dtype
        )
        if self.small_circle_policy == "clamp":
            return radii.clamp_min(threshold)
        if self.small_circle_policy == "snap_by_radius":
            half_threshold = threshold / 2.0
            return torch.where(
                radii < half_threshold,
                torch.zeros_like(radii),
                radii.clamp_min(threshold),
            )
        if self.small_circle_policy == "snap_by_area":
            half_area_radius = threshold * (0.5**0.5)
            return torch.where(
                radii < half_area_radius,
                torch.zeros_like(radii),
                radii.clamp_min(threshold),
            )
        return torch.where(radii >= threshold, radii, torch.zeros_like(radii))

    def _coverage_stack(
        self,
        centers: Tensor | None = None,
        radii: Tensor | None = None,
    ) -> Tensor:
        """Rasterize all circles into a flattened ``[n, nx, ny]`` coverage stack."""
        centers = self.centers if centers is None else centers
        radii = self._effective_radii(self.radius if radii is None else radii)
        flat_centers = centers.reshape(-1, 2)
        flat_radii = radii.reshape(-1)
        region_center = tuple(self.design_region_cfg["center"][:2])
        region_size = tuple(self.design_region_cfg["size"][:2])
        return torch.stack(
            [
                _circle_coverage(
                    center,
                    radius,
                    region_center=region_center,
                    region_size=region_size,
                    grid_step=self.grid_step,
                    shape=self.phi_shape,
                    samples_per_pixel=self.coverage_samples,
                )
                for center, radius in zip(flat_centers, flat_radii)
            ],
            dim=0,
        )

    @staticmethod
    def _compose(coverages: Tensor, values: Tensor, background: float):
        """Compose coverage maps with the same sequential alpha rule as the rasterizer."""
        state = torch.full(
            coverages.shape[1:],
            background,
            device=coverages.device,
            dtype=coverages.dtype,
        )
        before = []
        for coverage, value in zip(coverages, values):
            before.append(state)
            state = state + coverage * (value - state)
        return state, torch.stack(before, dim=0)

    def _values_for_raster(self, dtype, device):
        values = self._material_values()
        if values is None:
            values = torch.ones(self.centers.shape[:-1], device=device, dtype=dtype)
        return values.reshape(-1).to(device=device, dtype=dtype)

    def _build_gpu_subpixel_pattern(
        self, material_values: Tensor | None, background_value: float
    ) -> Tensor:
        """Rasterize z-axis cylinders with the CuPy subpixel area-average kernel.

        The local rasterizer accepts polygonal extrusions, so each circle is
        represented by a configurable regular polygon. With the default 128
        sides, the polygon error is far below a typical 10 nm simulation cell
        for the radii used by the digital lattice.
        """
        if not self.centers.is_cuda:
            raise RuntimeError(
                "gpu_subpixel=True requires circle parameters on a CUDA device"
            )

        import cupy as cp
        import tidy3d as td

        from gpu_subpixel import GPUSubpixelRasterizer

        from .box import _gpu_raster_to_torch_xy

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
            grid = td.Grid(boundaries=td.Coords(x=x_bounds, y=y_bounds, z=(-0.5, 0.5)))
            self._gpu_subpixel_grid = grid
            self._gpu_subpixel_rasterizer = GPUSubpixelRasterizer(
                grid, dtype=cp.float32
            )

        centers = self.centers.detach().reshape(-1, 2).cpu().numpy()
        radii = self._effective_radii(self.radius).detach().reshape(-1).cpu().numpy()
        if material_values is None:
            values = np.ones(centers.shape[0], dtype=np.float64)
        else:
            values = material_values.detach().reshape(-1).cpu().numpy()

        theta = np.arange(self.gpu_circle_sides, dtype=np.float64) * (
            2.0 * np.pi / self.gpu_circle_sides
        )
        structures = []
        for (cx, cy), radius, value in zip(centers, radii, values):
            if radius <= 0.0:
                continue
            vertices = np.stack(
                (cx + radius * np.cos(theta), cy + radius * np.sin(theta)),
                axis=-1,
            )
            structures.append(
                td.Structure(
                    geometry=td.PolySlab(vertices=vertices, slab_bounds=(-0.5, 0.5)),
                    medium=td.Medium(permittivity=float(value) + encoded_offset),
                )
            )

        raster = self._gpu_subpixel_rasterizer.rasterize(
            structures,
            background_eps=float(background_value) + encoded_offset,
        )
        cp.cuda.get_current_stream().synchronize()
        density = torch.as_tensor(
            _gpu_raster_to_torch_xy(cp.asnumpy(raster)),
            device=self.operation_device,
        )
        return (density[..., 0] - encoded_offset).to(
            device=self.operation_device, dtype=torch.float32
        )

    def build_pattern(self, permittivity: Tensor | None = None) -> Tensor:
        """Build a uniform-grid circle density using subpixel area fractions.

        Args:
            permittivity: Optional existing raster used as the background map.

        Returns:
            Tensor: Normalized density with shape ``phi_shape``. Circle overlap
            follows deterministic sequential alpha compositing; normal digital
            configurations should keep neighboring circles non-overlapping.
        """
        dtype = self.centers.dtype
        device = self.centers.device
        material_values = self._material_values()
        background = float(self.geometry_cfgs.get("material_background_value", 0.0))
        if self.use_gpu_subpixel and permittivity is None:
            return self._build_gpu_subpixel_pattern(material_values, background)
        coverages = self._coverage_stack()
        values = self._values_for_raster(dtype, device)
        density, _ = self._compose(coverages, values, background)
        if permittivity is not None:
            # ``permittivity`` is a legacy compositing argument. Circle maps are
            # normally built without it, but honoring it keeps the geometry API
            # compatible with the other parameterizations.
            density = permittivity + density - background
        return density

    def compute_derivatives(self, grad_eps: Tensor, eps: Tensor):
        """Aggregate material, radius, and center gradients from circle coverage.

        Args:
            grad_eps: Upstream gradient with respect to the normalized density map.
            eps: Forward density map saved by :class:`CircleDiffWrapper`.

        Returns:
            Tuple of center gradients, size/radius gradients, and material
            gradients. Radius and center derivatives use central finite
            differences of the same area-fraction rasterizer used in the Torch
            forward path. Later overlapping circles receive the same sequential
            visibility weighting as the forward composition.

        Notes:
            For an isolated circle, the geometric derivatives are multiplied by
            ``inner_value - background_value``. Thus a lower-index hole has the
            opposite radius and center-gradient polarity from a higher-index
            inclusion. For overlapping circles, the same contrast is computed
            against the already-composited material beneath the current circle,
            followed by the visibility factor of later circles.
            When ``small_circle_policy`` removes a circle and makes its
            effective radius zero, the ordinary area derivative is also zero.
            In that case, the cell containing the circle center receives a
            configurable surrogate radius derivative so the circle can re-enter
            the design. With ``gpu_subpixel=True``, the forward uses the
            configured regular-polygon approximation and the backward uses the
            sampled circular coverage model, so ``gpu_circle_sides`` and
            ``coverage_samples`` control the corresponding approximation errors.
        """
        del eps
        grad_eps = grad_eps.real if torch.is_complex(grad_eps) else grad_eps
        dtype = grad_eps.dtype
        device = grad_eps.device
        coverages = self._coverage_stack().to(device=device, dtype=dtype)
        values = self._values_for_raster(dtype, device)
        background = float(self.geometry_cfgs.get("material_background_value", 0.0))
        _, before = self._compose(coverages, values, background)

        flat_centers = (
            self.centers.detach().reshape(-1, 2).to(device=device, dtype=dtype)
        )
        flat_radii = (
            self._effective_radii(self.radius)
            .detach()
            .reshape(-1)
            .to(device=device, dtype=dtype)
        )
        grad_centers = torch.zeros_like(self.centers, device=device, dtype=dtype)
        grad_sizes = torch.zeros_like(self.sizes, device=device, dtype=dtype)
        grad_centers_flat = grad_centers.reshape(-1, 2)
        grad_radius = grad_sizes[..., 0].reshape(-1)

        grad_material = None
        if self.material_rho is not None and self.optimization_stage in {
            "material",
            "joint",
        }:
            grad_material = torch.zeros_like(
                self.material_rho, device=device, dtype=dtype
            )
            grad_material_flat = grad_material.reshape(-1)

        # ``suffix`` is the fraction not replaced by all later circles. This is
        # the key term needed to aggregate derivatives for the rasterizer's
        # sequential overlap rule.
        suffix = torch.ones_like(grad_eps)
        optimize_shape = self.optimization_stage in {"shape", "joint"} or bool(
            self.centers.requires_grad
        )
        for index in range(coverages.shape[0] - 1, -1, -1):
            visible = suffix
            if grad_material is not None:
                grad_material_flat[index] = (
                    grad_eps * coverages[index] * visible
                ).sum()

            if optimize_shape:
                center = flat_centers[index]
                radius = flat_radii[index]
                step = self.gradient_step_um
                delta_maps = []
                for axis in (0, 1):
                    plus_center = center.clone()
                    minus_center = center.clone()
                    plus_center[axis] += step
                    minus_center[axis] -= step
                    plus = _circle_coverage(
                        plus_center,
                        radius,
                        region_center=tuple(self.design_region_cfg["center"][:2]),
                        region_size=tuple(self.design_region_cfg["size"][:2]),
                        grid_step=self.grid_step,
                        shape=self.phi_shape,
                        samples_per_pixel=self.coverage_samples,
                    )
                    minus = _circle_coverage(
                        minus_center,
                        radius,
                        region_center=tuple(self.design_region_cfg["center"][:2]),
                        region_size=tuple(self.design_region_cfg["size"][:2]),
                        grid_step=self.grid_step,
                        shape=self.phi_shape,
                        samples_per_pixel=self.coverage_samples,
                    )
                    delta_maps.append((plus - minus) / (2.0 * step))
                plus = _circle_coverage(
                    center,
                    self._effective_radii(radius + step),
                    region_center=tuple(self.design_region_cfg["center"][:2]),
                    region_size=tuple(self.design_region_cfg["size"][:2]),
                    grid_step=self.grid_step,
                    shape=self.phi_shape,
                    samples_per_pixel=self.coverage_samples,
                )
                minus = _circle_coverage(
                    center,
                    self._effective_radii(radius - step),
                    region_center=tuple(self.design_region_cfg["center"][:2]),
                    region_size=tuple(self.design_region_cfg["size"][:2]),
                    grid_step=self.grid_step,
                    shape=self.phi_shape,
                    samples_per_pixel=self.coverage_samples,
                )
                delta_maps.append((plus - minus) / (2.0 * step))
                if radius.item() == 0.0 and self.small_circle_policy in {
                    "remove",
                    "snap_by_radius",
                    "snap_by_area",
                }:
                    # A removed circle has no ordinary area derivative at r=0.
                    # Seed its radius with the upstream gradient at the cell
                    # containing its center so a discrete primitive can recover.
                    delta_maps[2] = delta_maps[2].clone()
                    x_left = (
                        float(self.design_region_cfg["center"][0])
                        - float(self.design_region_cfg["size"][0]) / 2.0
                    )
                    y_top = (
                        float(self.design_region_cfg["center"][1])
                        + float(self.design_region_cfg["size"][1]) / 2.0
                    )
                    cell_x = int((center[0].detach().item() - x_left) / self.grid_step)
                    cell_y = int((y_top - center[1].detach().item()) / self.grid_step)
                    cell_x = max(0, min(self.phi_shape[0] - 1, cell_x))
                    cell_y = max(0, min(self.phi_shape[1] - 1, cell_y))
                    delta_maps[2][cell_x, cell_y] = self.zero_radius_gradient_scale
                coefficient = (values[index] - before[index]) * visible
                grad_centers_flat[index, 0] = (
                    grad_eps * coefficient * delta_maps[0]
                ).sum()
                grad_centers_flat[index, 1] = (
                    grad_eps * coefficient * delta_maps[1]
                ).sum()
                grad_radius[index] = (grad_eps * coefficient * delta_maps[2]).sum()

            suffix = suffix * (1.0 - coverages[index])

        return grad_centers, grad_sizes, grad_material

    def forward(self, permittivity: Tensor | None = None) -> Tensor:
        """Rasterize circles with the custom aggregate-gradient backward pass."""
        return CircleDiffWrapper.apply(
            self.centers,
            self.sizes,
            self._material_values(),
            lambda: self.build_pattern(permittivity=permittivity),
            self.compute_derivatives,
        )

    def set_optimization_stage(self, stage: str, *, optimize_centers: bool = False):
        """Select material, shape, joint, or frozen optimization for circles."""
        super().set_optimization_stage(stage, optimize_centers=optimize_centers)

    def transfer_material_to_shape(self, **kwargs):
        """Project material values and transfer area fractions into radii."""
        if self.material_rho is None:
            raise RuntimeError("Material optimization is not enabled for these circles")
        eps_low = kwargs.pop("eps_low", self.design_region_cfg["eps_bg"])
        eps_high = kwargs.pop("eps_high", self.design_region_cfg["eps"])
        eps_host = kwargs.pop("eps_host", eps_low)
        eps_inclusion = kwargs.pop("eps_inclusion", eps_high)
        beta = kwargs.pop("beta", None)
        eta = kwargs.pop("eta", 0.5)
        with torch.no_grad():
            projected = self.material_projector.project_rho(
                self.material_rho, beta=beta, eta=eta
            )
            fraction = self.material_projector.rho_to_fill_fraction(
                self.material_rho,
                eps_low=eps_low,
                eps_high=eps_high,
                eps_host=eps_host,
                eps_inclusion=eps_inclusion,
                beta=beta,
                eta=eta,
            )
            self.material_state.copy_((projected >= eta).to(self.material_state))
            ## the shrinking is based on the current circle radius, not max radius
            radii = self.sizes[..., 0] * torch.sqrt(fraction)
            self.sizes[..., 0].copy_(self._effective_radii(radii))
        return fraction


class CircleParameterization(BaseParametrization):
    """Register circular footprints as a MAPS parameterization method."""

    def __init__(
        self,
        *args,
        cfgs: dict = dict(
            method="circle",
            geometry_cfgs=dict(batch_dims=(16, 16), size_dim=1),
            transform=[],
            init_method="2d_array",
            denorm_mode="linear_eps",
        ),
        **kwargs,
    ):
        super().__init__(*args, cfgs=cfgs, **kwargs)
        method = cfgs["method"]
        self.geometry_key = "cylinders" if method == "cylinder" else "circles"
        self.register_parameter_build_per_region_fn(
            method, self._build_parameters_circle
        )
        self.register_parameter_reset_per_region_fn(
            method, self._reset_parameters_circle
        )
        self.build_parameters(cfgs, self.design_region_cfg)
        self.reset_parameters(cfgs, self.design_region_cfg)

    def _build_parameters_circle(self, param_cfg, region_cfg):
        geometry_cfgs = dict(param_cfg["geometry_cfgs"])
        geometry_cfgs["size_dim"] = 1
        for key in (
            "material_optimization",
            "enable_material",
            "material_enabled",
            "material_init",
            "material_model",
            "material_background_value",
            "border",
            "phi_resolution",
            "coverage_samples",
            "gradient_step_um",
            "min_radius_um",
            "max_radius_um",
            "small_circle_policy",
            "zero_radius_gradient_scale",
            "gpu_subpixel",
            "gpu_circle_sides",
        ):
            if key in param_cfg:
                geometry_cfgs.setdefault(key, param_cfg[key])
        circles = Circles(
            sim_cfg=self.hr_device.sim_cfg,
            geometry_cfgs=geometry_cfgs,
            region_name=self.region_name,
            design_region_mask=self.hr_design_region_mask,
            design_region_cfg=self.hr_device.design_region_cfgs[self.region_name],
            operation_device=self.operation_device,
        )
        return {self.geometry_key: circles}, dict()

    def _reset_parameters_circle(
        self,
        weight_dict,
        param_cfg,
        region_cfg,
        init_method: str | Callable = "2d_array",
    ):
        """Initialize circle centers and radii on a regular lattice."""
        circles: Circles = weight_dict[self.geometry_key]
        if callable(init_method):
            init_method(weight_dict, param_cfg, region_cfg)
            return
        if not init_method.startswith("2d_array"):
            raise ValueError(f"Unsupported circle initialization method: {init_method}")

        n_x, n_y = circles.centers.shape[:2]
        center = region_cfg["center"]
        size = region_cfg["size"]
        border = float(circles.geometry_cfgs.get("border", 0.2))
        pitch_x = (float(size[0]) - 2.0 * border) / n_x
        pitch_y = (float(size[1]) - 2.0 * border) / n_y
        x = torch.linspace(
            center[0] - size[0] / 2 + border + pitch_x / 2,
            center[0] + size[0] / 2 - border - pitch_x / 2,
            n_x,
            device=self.operation_device,
        )
        y = torch.linspace(
            center[1] - size[1] / 2 + border + pitch_y / 2,
            center[1] + size[1] / 2 - border - pitch_y / 2,
            n_y,
            device=self.operation_device,
        )
        circles.centers.data.copy_(
            torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1)
        )
        max_radius = circles._max_radius()
        if init_method == "2d_array_full":
            circles.sizes.data[..., 0].fill_(max_radius)
        else:
            circles.sizes.data[..., 0].uniform_(max_radius / 3.0, max_radius / 2.0)
            density = float(param_cfg.get("density", 0.5))
            if density < 1.0:
                mask = torch.rand((n_x, n_y), device=self.operation_device) > density
                circles.sizes.data[..., 0].masked_fill_(mask, 0.0)
        if circles.material_rho is not None and init_method == "2d_array_full":
            circles.material_rho.data.fill_(float(param_cfg.get("material_init", 0.5)))

    def _uniform_parameter_grid_coords(self, density):
        """Return physical coordinates for the circle density's uniform grid."""
        del density
        return self.weights[self.geometry_key].uniform_parameter_grid_coords()

    def build_density(self, weights, sharpness: float | None = None, **kwargs):
        """Build the normalized circle density on its internal uniform grid."""
        del sharpness, kwargs
        return weights[self.geometry_key]()

    def build_permittivity(self, weights, sharpness: float | None = None, **kwargs):
        """Build the normalized circle permittivity map."""
        return self.build_density(weights, sharpness, **kwargs)

    def set_optimization_stage(self, stage: str, *, optimize_centers: bool = False):
        """Set the optimization stage on this region's circle geometry."""
        self.weights[self.geometry_key].set_optimization_stage(
            stage, optimize_centers=optimize_centers
        )

    def transfer_material_to_shape(self, **kwargs):
        """Transfer material variables into circle radii and binary state."""
        return self.weights[self.geometry_key].transfer_material_to_shape(**kwargs)

    def project_material_to_shape(self, **kwargs):
        """Compatibility alias for :meth:`transfer_material_to_shape`."""
        return self.transfer_material_to_shape(**kwargs)


class CylinderParameterization(CircleParameterization):
    """Register z-axis cylindrical footprints as a MAPS parameterization."""


Cylinders = Circles
