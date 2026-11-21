"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2025-10-31 14:34:17
LastEditors: Jiaqi Gu (jiaqigu@asu.edu)
LastEditTime: 2025-11-02 16:52:26
FilePath: /MAPS_ilt/core/invdes/models/layers/parametrization/geometry.py
"""

import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.types import Device


def gradImage(image):
    """Compute centered finite-difference image gradients with replicated padding.

    Args:
        image (Tensor): Image tensor whose final two dimensions are spatial dimensions.
            The tensor is reshaped internally to ``[-1, 1, H, W]`` before finite
            differences are evaluated.

    Returns:
        tuple[Tensor, Tensor]: ``(grad_x, grad_y)`` tensors with the internal
        ``[batch, 1, H, W]`` layout used by the implementation.
    """
    GRAD_STEPSIZE = 1.0
    image = image.view([-1, 1, image.shape[-2], image.shape[-1]])
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")[:, 0].detach()
    gradX = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * GRAD_STEPSIZE)
    gradY = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * GRAD_STEPSIZE)
    return gradX.view(image.shape), gradY.view(image.shape)


def get_grid(shape, dl):
    """Construct centered Cartesian coordinate arrays for a 2-D grid.

    Args:
        shape (tuple[int, int]): Number of grid points along the ``x`` and ``y``
            axes, respectively.
        dl (float): Uniform grid spacing in the coordinate units of the caller.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(xs, ys)`` coordinate arrays with
        shape ``(shape[0], shape[1])`` and ``numpy.meshgrid(..., indexing="ij")``
        axis ordering.
    """
    # computes the coordinates in the grid

    (Nx, Ny) = shape
    # if Ny % 2 == 0:
    #     Ny -= 1
    # coordinate vectors
    x_coord = np.linspace(-(Nx - 1) / 2 * dl, (Nx - 1) / 2 * dl, Nx)
    y_coord = np.linspace(-(Ny - 1) / 2 * dl, (Ny - 1) / 2 * dl, Ny)

    # x and y coordinate arrays
    xs, ys = np.meshgrid(x_coord, y_coord, indexing="ij")
    return (xs, ys)


def apply_regions_gpu(reg_list, xs, ys, eps_r_list, eps_bg, device="cuda"):
    """Rasterize region functions into a GPU permittivity tensor.

    Args:
        reg_list (callable | Sequence[callable]): Region predicate functions.
            Each callable receives ``(xs, ys)`` tensors and returns a boolean mask.
        xs (ArrayLike): ``x`` coordinates used to evaluate region predicates.
        ys (ArrayLike): ``y`` coordinates used to evaluate region predicates.
        eps_r_list (float | Sequence[float]): Relative-permittivity value for each
            region. A scalar is replicated across all regions.
        eps_bg (float): Background relative permittivity.
        device (str | torch.device): Torch device on which coordinates and the
            resulting tensor are allocated. Defaults to ``"cuda"``.

    Returns:
        Tensor: Relative-permittivity tensor initialized to ``eps_bg`` and
        overwritten by the supplied region values on their boolean masks.
    """
    # Convert inputs to tensors and move them to the GPU
    xs = torch.tensor(xs, device=device)
    ys = torch.tensor(ys, device=device)

    # Handle scalars to lists conversion
    if isinstance(eps_r_list, (int, float)):
        eps_r_list = [eps_r_list] * len(reg_list)
    if not isinstance(reg_list, (list, tuple)):
        reg_list = [reg_list]

    # Initialize permittivity tensor with background value
    eps_r = torch.full(xs.shape, eps_bg, device=device)

    # Convert region functions to a vectorized form using PyTorch operations
    for e, reg in zip(eps_r_list, reg_list):
        # Assume that reg is a lambda or function that can be applied to tensors
        material_mask = reg(xs, ys)  # This should return a boolean tensor
        eps_r[material_mask] = e

    return eps_r  # Move the result back to CPU and convert to numpy array


class BatchGeometry(nn.Module):
    """Represent a batch of parameterized 2-D geometries and their rasterization state.

    Args:
        sim_cfg (dict[str, Any]): Simulation configuration mapping retained by the
            geometry object.
        geometry_cfgs (dict[str, Any]): Geometry configuration containing the required
            ``batch_dims`` entry and optional ``size_dim`` entry.
        region_name (str): Design-region identifier. Defaults to ``"design_region_1"``.
        design_region_mask (Tensor | None): Optional ``[H, W]`` boolean mask identifying
            the active design region. Defaults to None.
        design_region_cfg (dict[str, Any] | None): Optional design-region configuration.
            Defaults to None.
        operation_device (torch.device): Device associated with optimization state.
            Defaults to ``torch.device("cuda:0")``.

    Attributes:
        centers (nn.Parameter): Geometry centers with a trailing dimension of size 2
            representing ``x`` and ``y`` coordinates in micrometers.
        sizes (nn.Parameter): Geometry size parameters in micrometers; the trailing
            dimension is controlled by ``size_dim``.
        region_name (str): Design-region identifier.
        design_region_mask (Tensor | None): Optional boolean design-region mask.
        design_region_cfg (dict[str, Any] | None): Optional region configuration.
        operation_device (torch.device): Optimization-state device.

    Note:
        ``compute_derivatives`` and ``discretize_as_permittivity`` currently contain
        placeholder behavior; their docstrings describe the implemented behavior rather
        than implying a completed geometry-specific derivative or rasterizer.
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
        """Initialize batched geometry parameters and design-region state.

        Args:
            sim_cfg (dict[str, Any]): Simulation configuration mapping.
            geometry_cfgs (dict[str, Any]): Geometry configuration containing
                ``batch_dims`` and optionally ``size_dim``.
            region_name (str): Design-region identifier. Defaults to ``"design_region_1"``.
            design_region_mask (Tensor | None): Optional ``[H, W]`` boolean design mask.
                Defaults to None.
            design_region_cfg (dict[str, Any] | None): Optional design-region configuration.
                Defaults to None.
            operation_device (torch.device): Device associated with the optimization
                state. Defaults to ``torch.device("cuda:0")``.

        Raises:
            AssertionError: If ``size_dim`` is not a positive integer.
            KeyError: If ``geometry_cfgs`` does not contain ``batch_dims``.
        """
        super().__init__()
        self.sim_cfg = sim_cfg
        self.geometry_cfgs = geometry_cfgs
        batch_dims = self.geometry_cfgs[
            "batch_dims"
        ]  # can be a number of multiple batch dimensions
        if isinstance(batch_dims, int):
            batch_dims = (batch_dims,)
        size_dim = self.geometry_cfgs.get(
            "size_dim", 2
        )  # number of parameters to determine the size, default 2, e.g., 2 for box, 1 for circle
        assert (
            isinstance(size_dim, int) and size_dim >= 1
        ), "size_dim should be a positive integer."
        self.batch_dims = tuple(batch_dims)
        self.centers = nn.Parameter(
            torch.zeros(*batch_dims, 2)
        )  # (num_geometries, 2) # x, y coordinates, unit in um
        self.sizes = nn.Parameter(
            torch.ones(*batch_dims, size_dim)
        )  # (num_geometries, size_dim), unit in um (for height/width/radius)

        self.material_enabled = bool(
            self.geometry_cfgs.get(
                "material_optimization",
                self.geometry_cfgs.get(
                    "enable_material",
                    self.geometry_cfgs.get(
                        "material_enabled", "material_init" in self.geometry_cfgs
                    ),
                ),
            )
        )
        self.optimization_stage = "legacy"
        self.register_buffer(
            "material_state",
            torch.ones(*batch_dims),
        )
        if self.material_enabled:
            material_init = float(self.geometry_cfgs.get("material_init", 0.5))
            if not 0.0 <= material_init <= 1.0:
                raise ValueError("material_init must be between 0 and 1")
            self.material_rho = nn.Parameter(
                torch.full(
                    batch_dims,
                    material_init,
                )
            )
        else:
            self.material_rho = None

        self.region_name = region_name
        self.design_region_mask = design_region_mask  # (H, W), bool tensor
        self.design_region_cfg = design_region_cfg
        self.operation_device = operation_device
        if self.material_enabled:
            self.set_optimization_stage("material")

    def set_optimization_stage(
        self,
        stage: str,
        *,
        optimize_centers: bool = False,
    ) -> None:
        """Select which geometry parameter family is trainable.

        Args:
            stage: One of ``"material"``, ``"shape"``, ``"joint"``, or
                ``"frozen"``.
            optimize_centers: Whether center parameters are trainable in the
                selected stage. Defaults to False because digital arrays normally
                keep their lattice positions fixed.

        Raises:
            ValueError: If ``stage`` is unsupported or material optimization is
                requested without an enabled material parameter.
        """
        valid_stages = {"material", "shape", "joint", "frozen"}
        if stage not in valid_stages:
            raise ValueError(f"Unsupported optimization stage: {stage!r}")
        if stage in {"material", "joint"} and self.material_rho is None:
            raise ValueError(
                "Material optimization is disabled; set material_optimization=True "
                "in geometry_cfgs before selecting this stage."
            )

        self.optimization_stage = stage
        self.centers.requires_grad_(bool(optimize_centers) and stage != "frozen")
        self.sizes.requires_grad_(stage in {"shape", "joint"})
        if self.material_rho is not None:
            self.material_rho.requires_grad_(stage in {"material", "joint"})

    def transfer_material_to_shape(
        self,
        *,
        eps_host: Tensor | float,
        eps_inclusion: Tensor | float,
        eps_low: Tensor | float,
        eps_high: Tensor | float,
        beta: float | Tensor | None = None,
        eta: float | Tensor = 0.5,
        mapping: str = "isotropic",
    ) -> Tensor:
        """Project material variables and transfer them to shape parameters.

        Geometry subclasses implement the shape-specific size update. This base
        method owns the shared material projection and binary material state.

        Args:
            eps_host: Host material permittivity for the EMT relation.
            eps_inclusion: Primitive inclusion permittivity for the EMT relation.
            eps_low: Physical endpoint represented by normalized material zero.
            eps_high: Physical endpoint represented by normalized material one.
            beta: Optional projection sharpness.
            eta: Projection midpoint. Defaults to 0.5.
            mapping: Shape-specific fill-fraction mapping name.

        Returns:
            Tensor: Inclusion fill fractions used by the subclass transfer.

        Raises:
            NotImplementedError: Always; subclasses must update their shape.
            RuntimeError: If material optimization is disabled.
        """
        raise NotImplementedError

    def project_material_to_shape(self, **kwargs):
        """Compatibility alias for :meth:`transfer_material_to_shape`."""
        return self.transfer_material_to_shape(**kwargs)

    def compute_derivatives(self, grad_eps: Tensor):
        """Compute placeholder geometry derivatives from an epsilon-gradient tensor.

        Args:
            grad_eps (Tensor): Upstream permittivity gradient, expected by the current
                implementation in a ``[B, 1, H, W]`` layout.

        Returns:
            tuple[Tensor, Tensor]: Zero tensors matching ``self.centers`` and
            ``self.sizes``. The current implementation does not yet derive geometry-
            specific gradients from ``grad_eps``.
        """
        # grad_eps: (B, 1, H, W)
        # Placeholder implementation, actual implementation depends on geometry type and simulation details
        grad_centers = torch.zeros_like(self.centers)
        grad_sizes = torch.zeros_like(self.sizes)
        return grad_centers, grad_sizes

    def discretize_as_permittivity(
        self, grid_x: Tensor, grid_y: Tensor, permittivity: Tensor = None
    ) -> Tensor:
        """Return the current placeholder permittivity raster for the supplied grid.

        Args:
            grid_x (Tensor): ``[H, W]`` grid-point ``x`` coordinates.
            grid_y (Tensor): ``[H, W]`` grid-point ``y`` coordinates.
            permittivity (Tensor | None): Optional tensor used as the returned
                permittivity state instead of allocating zeros. Defaults to None.

        Returns:
            Tensor: ``permittivity`` when supplied; otherwise a zero tensor with
            the same shape, dtype, and device as ``grid_x``.

        Note:
            The current implementation does not yet rasterize the stored geometry
            parameters onto ``grid_x``/``grid_y``.
        """
        # grid_x, grid_y: (H, W)
        # given the interested grid points, we discretize and draw this continuous geometries on to this grid/mesh
        eps = torch.zeros_like(grid_x) if permittivity is None else permittivity
        return eps
