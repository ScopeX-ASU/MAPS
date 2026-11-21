import math

import torch

from .morph_op import dilation, erosion


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def _make_disk_kernel(radius_int, device, dtype):
    """Build a binary disk structuring element for morphology operations.

    Args:
        radius_int (int): Integer radius in pixels. Non-positive values produce
            a ``1 x 1`` kernel.
        device (torch.device | str): Device for the returned kernel.
        dtype (torch.dtype): Torch dtype for the returned kernel.

    Returns:
        torch.Tensor: Square binary disk kernel with side length
        ``2 * radius_int + 1`` for positive radii, or ``1 x 1`` otherwise.
    """
    radius_int = int(radius_int)
    if radius_int <= 0:
        return torch.ones((1, 1), device=device, dtype=dtype)

    coords = torch.arange(-radius_int, radius_int + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return ((xx**2 + yy**2) <= (float(radius_int) + 0.5) ** 2).to(dtype)


def _build_radius_schedule(
    extrude_size,
    extrude_angle,
    extrude_direction,
    reference_plane,
    grid_step,
    physical_thickness=None,
    extrude_weights=None,
    dtype=torch.float32,
    device=None,
):
    """Compute the lateral sidewall offset for each extrusion slice.

    Args:
        extrude_size (int): Number of output slices along the extrusion axis.
        extrude_angle (float): Sidewall angle in radians.
        extrude_direction (str): Extrusion direction sign, ``"+"`` or ``"-"``.
            The current implementation validates this value but does not use it
            to flip the computed lateral offset.
        reference_plane (str): Reference plane for the extrusion, one of
            ``"top"``, ``"middle"``, or ``"bottom"``.
        grid_step (float): Lateral grid spacing used to convert physical offsets
            into pixel radii.
        physical_thickness (float | None): Total physical extrusion thickness.
            Defaults to None; when omitted, thickness is ``extrude_size * grid_step``.
        extrude_weights (Sequence[float] | None): Per-slice coverage weights
            multiplied by ``grid_step`` to form slice thicknesses. Defaults to None.
        dtype (torch.dtype): Torch dtype for generated tensors. Defaults to
            ``torch.float32``.
        device (torch.device | str | None): Device for generated tensors.
            Defaults to None.

    Returns:
        torch.Tensor: One lateral offset in pixels for each extrusion slice.

    Raises:
        ValueError: If the extrusion size, grid spacing, direction, reference plane,
            angle, or coverage-weight length is invalid.
    """
    if extrude_size < 1:
        raise ValueError(f"extrude_size must be at least 1, got {extrude_size}")
    if grid_step is None or grid_step <= 0:
        raise ValueError(f"grid_step must be positive, got {grid_step}")
    if extrude_direction not in {"+", "-"}:
        raise ValueError(
            f"extrude_direction should be '+' or '-', but got {extrude_direction}"
        )
    if reference_plane not in {"top", "middle", "bottom"}:
        raise ValueError(
            "reference_plane must be 'top', 'middle', or 'bottom', "
            f"got {reference_plane!r}"
        )
    if abs(extrude_angle) > math.pi / 2:
        raise ValueError(
            f"sidewall angle must be in [-pi/2, pi/2] radians, got {extrude_angle}"
        )

    if extrude_weights is not None:
        slice_thickness = torch.as_tensor(
            extrude_weights, dtype=dtype, device=device
        ).flatten() * float(grid_step)
        if slice_thickness.numel() != extrude_size:
            raise ValueError(
                "Extrusion coverage weight size mismatch: "
                f"{slice_thickness.numel()} vs {extrude_size}"
            )
    else:
        total_thickness = (
            float(physical_thickness)
            if physical_thickness is not None
            else float(extrude_size) * float(grid_step)
        )
        slice_thickness = torch.full(
            (extrude_size,),
            total_thickness / float(extrude_size),
            dtype=dtype,
            device=device,
        )

    boundaries = torch.cat(
        [
            torch.zeros(1, dtype=dtype, device=device),
            torch.cumsum(slice_thickness, dim=0),
        ]
    )
    centers = (boundaries[:-1] + boundaries[1:]) / 2.0
    total_thickness = boundaries[-1]
    if reference_plane == "bottom":
        distance_from_reference = centers
    elif reference_plane == "top":
        distance_from_reference = total_thickness - centers
    else:
        # With a middle reference, positive z distance means erosion above
        # the plane and negative z distance means dilation below it.
        distance_from_reference = centers - total_thickness / 2.0

    if math.isclose(extrude_angle, 0.0, abs_tol=1e-8):
        return torch.zeros_like(distance_from_reference)

    lateral_offset = distance_from_reference * math.tan(float(extrude_angle))
    return lateral_offset / float(grid_step)


def _pchip_endpoint_slope(
    first_interval,
    second_interval,
    first_secant,
    second_secant,
):
    """Compute a shape-preserving slope at one end of a PCHIP curve."""
    candidate = (
        (2.0 * first_interval + second_interval) * first_secant
        - first_interval * second_secant
    ) / (first_interval + second_interval)
    candidate = torch.where(
        candidate * first_secant > 0,
        candidate,
        torch.zeros_like(candidate),
    )
    needs_limit = (first_secant * second_secant <= 0) & (
        candidate.abs() > 3.0 * first_secant.abs()
    )
    return torch.where(needs_limit, 3.0 * first_secant, candidate)


def _monotone_cubic_interpolate(
    control_values,
    control_parameters,
    sample_parameters,
):
    """Interpolate values with differentiable, shape-preserving cubic PCHIP."""
    control_values = control_values.flatten()
    control_parameters = control_parameters.flatten()
    if control_values.numel() == 0:
        raise ValueError("control_values must not be empty")
    if control_parameters.numel() != control_values.numel():
        raise ValueError(
            "control_parameters and control_values must have the same length"
        )
    if control_values.numel() == 1:
        return control_values.expand_as(sample_parameters)

    interval_widths = control_parameters[1:] - control_parameters[:-1]
    if torch.any(interval_widths.detach() <= 0).item():
        raise ValueError("control_parameters must be strictly increasing")
    if (
        not torch.isclose(
            control_parameters[0].detach(), control_parameters.new_tensor(0.0)
        ).item()
        or not torch.isclose(
            control_parameters[-1].detach(), control_parameters.new_tensor(1.0)
        ).item()
    ):
        raise ValueError("control_parameters must start at 0 and end at 1")

    secants = (control_values[1:] - control_values[:-1]) / interval_widths
    if control_values.numel() == 2:
        slopes = torch.stack((secants[0], secants[0]))
    else:
        left_secants = secants[:-1]
        right_secants = secants[1:]
        same_direction = left_secants * right_secants > 0
        safe_left_secants = torch.where(
            same_direction, left_secants, torch.ones_like(left_secants)
        )
        safe_right_secants = torch.where(
            same_direction, right_secants, torch.ones_like(right_secants)
        )
        left_widths = interval_widths[:-1]
        right_widths = interval_widths[1:]
        left_weights = 2.0 * right_widths + left_widths
        right_weights = right_widths + 2.0 * left_widths
        interior_slopes = torch.where(
            same_direction,
            (left_weights + right_weights)
            / (left_weights / safe_left_secants + right_weights / safe_right_secants),
            torch.zeros_like(left_secants),
        )
        slopes = torch.cat(
            (
                _pchip_endpoint_slope(
                    interval_widths[0],
                    interval_widths[1],
                    secants[0],
                    secants[1],
                )[None],
                interior_slopes,
                _pchip_endpoint_slope(
                    interval_widths[-1],
                    interval_widths[-2],
                    secants[-1],
                    secants[-2],
                )[None],
            )
        )

    sample_parameters = sample_parameters.flatten().clamp(0.0, 1.0)
    interval_indices = (
        torch.searchsorted(control_parameters, sample_parameters, right=True) - 1
    ).clamp_(0, control_values.numel() - 2)
    local_widths = interval_widths[interval_indices]
    local_parameter = (
        sample_parameters - control_parameters[interval_indices]
    ) / local_widths
    local_squared = local_parameter.square()
    local_cubed = local_squared * local_parameter

    return (
        (2.0 * local_cubed - 3.0 * local_squared + 1.0)
        * control_values[interval_indices]
        + (local_cubed - 2.0 * local_squared + local_parameter)
        * local_widths
        * slopes[interval_indices]
        + (-2.0 * local_cubed + 3.0 * local_squared)
        * control_values[interval_indices + 1]
        + (local_cubed - local_squared) * local_widths * slopes[interval_indices + 1]
    )


def _build_arbitrary_adius_schedule(
    extrude_size,
    control_points,
    grid_step,
    control_u=None,
    anchor_top=True,
    dtype=torch.float32,
    device=None,
):
    """Build a signed morphology-radius schedule from profile control points.

    Args:
        extrude_size (int): Number of output slices along the extrusion axis.
        control_points (Sequence[float] | torch.Tensor): Physical lateral offsets
            ordered from top to bottom.
        grid_step (float): Lateral grid spacing used to convert physical offsets
            into pixel radii.
        control_u (Sequence[float] | torch.Tensor | None): Normalized depths paired
            with ``control_points``. Defaults to equally spaced depths.
        anchor_top (bool): Whether to subtract the top offset so the input contour
            is unchanged at the top slice. Defaults to True.
        dtype (torch.dtype): Tensor dtype for the schedule. Defaults to
            ``torch.float32``.
        device (torch.device | str | None): Device for the schedule. Defaults to
            None.

    Returns:
        torch.Tensor: Signed morphology radii in pixels, ordered top to bottom.

    Raises:
        TypeError: If ``anchor_top`` is not a bool.
        ValueError: If the size, spacing, or control-point parameters are invalid.
    """
    if extrude_size < 1:
        raise ValueError(f"extrude_size must be at least 1, got {extrude_size}")
    if grid_step is None or grid_step <= 0:
        raise ValueError(f"grid_step must be positive, got {grid_step}")
    if not isinstance(anchor_top, bool):
        raise TypeError("anchor_top must be a bool")

    control_points = torch.as_tensor(
        control_points, dtype=dtype, device=device
    ).flatten()
    if control_points.numel() == 0:
        raise ValueError("control_points must not be empty")
    if control_u is None:
        control_u = torch.linspace(
            0.0,
            1.0,
            control_points.numel(),
            dtype=dtype,
            device=device,
        )
    else:
        control_u = torch.as_tensor(control_u, dtype=dtype, device=device).flatten()

    sample_u = torch.linspace(
        0.0,
        1.0,
        extrude_size,
        dtype=dtype,
        device=device,
    )
    offsets = _monotone_cubic_interpolate(control_points, control_u, sample_u)
    if anchor_top:
        offsets = offsets - offsets[0]
    return offsets / float(grid_step)


def get_required_lateral_halo(
    extrude_size,
    extrude_angle,
    extrude_direction,
    reference_plane,
    grid_step,
    physical_thickness=None,
    extrude_weights=None,
):
    """Return the lateral context needed by a sidewall extrusion.

    Args:
        extrude_size (int): Number of output slices along the extrusion axis.
        extrude_angle (float): Sidewall angle in degrees, matching :func:`extrude`.
        extrude_direction (str): Extrusion direction sign, ``"+"`` or ``"-"``.
        reference_plane (str): Reference plane, one of ``"top"``, ``"middle"``,
            or ``"bottom"``.
        grid_step (float): Lateral grid spacing used to convert physical offsets
            into pixel radii.
        physical_thickness (float | None): Total physical extrusion thickness.
            Defaults to None.
        extrude_weights (Sequence[float] | torch.Tensor | None): Optional
            per-slice coverage weights. Defaults to None.

    Returns:
        int: Maximum absolute sidewall displacement rounded up to pixels, plus
        one safety pixel. Vertical extrusion returns zero because it does not
        require lateral context.

    Raises:
        ValueError: If any extrusion argument is invalid.
    """
    angle_radians = math.radians(float(extrude_angle))
    radius_px = _build_radius_schedule(
        extrude_size=extrude_size,
        extrude_angle=angle_radians,
        extrude_direction=extrude_direction,
        reference_plane=reference_plane,
        grid_step=grid_step,
        physical_thickness=physical_thickness,
        extrude_weights=extrude_weights,
    )
    if torch.count_nonzero(radius_px).item() == 0:
        return 0
    return int(torch.ceil(radius_px.abs().max()).item()) + 1


def _build_integer_dilation_cache(field_4d, radius_px, op):
    """Cache morphology results for the integer radii needed by fractional blending.

    Args:
        field_4d (torch.Tensor): 4-D input plane with leading singleton batch/channel
            dimensions, typically shaped ``[1, 1, H, W]``.
        radius_px (torch.Tensor): Non-negative per-slice pixel radii.
        op (str): Morphology operation selector; ``"dilation"`` selects dilation,
            otherwise erosion is used.

    Returns:
        dict[int, torch.Tensor]: Morphology results keyed by the integer radii required
        by the supplied schedule.
    """
    r_floor = torch.floor(radius_px).to(torch.long).clamp_min(0)
    r_ceil = torch.ceil(radius_px).to(torch.long).clamp_min(0)
    needed_radii = torch.unique(torch.cat([r_floor, r_ceil], dim=0))

    cache = {}
    for radius in needed_radii.detach().cpu().tolist():
        kernel = _make_disk_kernel(
            radius_int=radius,
            device=field_4d.device,
            dtype=field_4d.dtype,
        )
        morphology = dilation if op == "dilation" else erosion
        cache[int(radius)] = morphology(
            field_4d,
            kernel=kernel,
            structuring_element=None,
            border_type="geodesic",
            engine="triton",
        )
    return cache


def _fractional_radius_blend(radius_px_scalar, morphology_cache):
    """Linearly interpolate morphology outputs between adjacent integer radii.

    Args:
        radius_px_scalar (torch.Tensor): Scalar, non-negative pixel radius.
        morphology_cache (dict[int, torch.Tensor]): Cached morphology results keyed
            by integer radius.

    Returns:
        torch.Tensor: Integer-radius result when the radius is integral; otherwise
        the linear interpolation between floor and ceiling radius results.
    """
    radius_px_scalar = radius_px_scalar.clamp_min(0.0)
    r0_float = torch.floor(radius_px_scalar)
    r1_float = torch.ceil(radius_px_scalar)
    r0 = int(r0_float.item())
    r1 = int(r1_float.item())

    if r0 == r1:
        return morphology_cache[r0]

    blend = (radius_px_scalar - r0_float).clamp(0.0, 1.0)
    return (1.0 - blend) * morphology_cache[r0] + blend * morphology_cache[r1]


def _extrude_vertical(permittivity, extrude_dim, extrude_size):
    """Replicate a 2-D permittivity plane uniformly along an extrusion axis.

    Args:
        permittivity (torch.Tensor): Input permittivity tensor.
        extrude_dim (int): Axis along which to insert the extrusion dimension.
        extrude_size (int): Number of repetitions along the inserted axis.

    Returns:
        torch.Tensor: Tensor with ``extrude_size`` replicated slices along
        ``extrude_dim``.
    """
    return permittivity.unsqueeze(extrude_dim).repeat_interleave(
        extrude_size, dim=extrude_dim
    )


def _extrude_sidewall_morphology(
    permittivity,
    extrude_dim,
    extrude_size,
    extrude_angle,
    extrude_direction,
    reference_plane,
    grid_step,
    physical_thickness=None,
    extrude_weights=None,
    z_downsample_factor=1,
    interpolation_mode="trilinear",
):
    """Construct non-vertical extrusion by slice-wise morphology.

    Args:
        permittivity (torch.Tensor): 2-D CUDA permittivity plane.
        extrude_dim (int): Output axis receiving the generated extrusion slices.
        extrude_size (int): Number of generated slices.
        extrude_angle (float): Sidewall angle in radians.
        extrude_direction (str): Extrusion direction sign, ``"+"`` or ``"-"``.
            The current implementation validates this value but does not use it
            to flip the computed lateral offset. Defaults to ``"+"`` at the public API.
        reference_plane (str): Reference plane, one of ``"top"``, ``"middle"``,
            or ``"bottom"``.
        grid_step (float): Physical grid spacing used to convert lateral offsets to
            pixels.
        physical_thickness (float | None): Total physical extrusion thickness.
            Defaults to None.
        extrude_weights (Sequence[float] | None): Per-slice coverage weights.
            Defaults to None.
        z_downsample_factor (int): Number of slices processed together as a chunk.
            Defaults to 1.
        interpolation_mode (str): Interpolation mode retained by the public API;
            the current morphology path does not otherwise use it. Defaults to
            ``"trilinear"``.

    Returns:
        torch.Tensor: Extruded tensor with the new slice axis moved to ``extrude_dim``.

    Raises:
        ValueError: If the input is not a 2-D CUDA tensor or the slice/chunk settings
            are invalid.
    """
    if not permittivity.is_cuda:
        raise ValueError("Non-vertical extrusion currently requires a CUDA tensor")
    if permittivity.ndim != 2:
        raise ValueError(
            f"Sidewall extrusion expects a 2D permittivity plane, got {permittivity.ndim}D"
        )
    if z_downsample_factor < 1:
        raise ValueError(
            f"z_downsample_factor must be at least 1, got {z_downsample_factor}"
        )

    radius_px = _build_radius_schedule(
        extrude_size=extrude_size,
        extrude_angle=extrude_angle,
        extrude_direction=extrude_direction,
        reference_plane=reference_plane,
        grid_step=grid_step,
        physical_thickness=physical_thickness,
        extrude_weights=extrude_weights,
        dtype=permittivity.dtype,
        device=permittivity.device,
    )

    if torch.count_nonzero(radius_px).item() == 0:
        return _extrude_vertical(permittivity, extrude_dim, extrude_size)

    plane_4d = permittivity.unsqueeze(0).unsqueeze(0)
    dilation_cache = _build_integer_dilation_cache(
        plane_4d, radius_px.abs(), "dilation"
    )
    erosion_cache = _build_integer_dilation_cache(plane_4d, radius_px.abs(), "erosion")

    _, _, h, w = plane_4d.shape
    volume = torch.empty(
        (extrude_size, h, w),
        dtype=permittivity.dtype,
        device=permittivity.device,
    )

    # Build every physical z slice from the continuous radius schedule.
    # This avoids artificial smoothing from z interpolation while keeping
    # the expensive morphology calls cached by integer radius.
    chunk_z = max(1, int(z_downsample_factor))
    for start in range(0, extrude_size, chunk_z):
        stop = min(start + chunk_z, extrude_size)
        for k in range(start, stop):
            radius = radius_px[k]
            cache = erosion_cache if radius >= 0 else dilation_cache
            volume[k] = _fractional_radius_blend(
                radius_px_scalar=radius.abs(),
                morphology_cache=cache,
            )[0, 0]

    return torch.movedim(volume, 0, extrude_dim)


def _extrude_arbitrary_morphology(
    permittivity,
    extrude_dim,
    extrude_size,
    control_points,
    grid_step,
    control_u=None,
    anchor_top=True,
    z_downsample_factor=1,
    interpolation_mode="trilinear",
):
    """Construct a control-point extrusion by slice-wise morphology.

    Args:
        permittivity (torch.Tensor): 2-D CUDA permittivity plane.
        extrude_dim (int): Output axis receiving the generated extrusion slices.
        extrude_size (int): Number of generated slices.
        control_points (Sequence[float] | torch.Tensor): Physical lateral offsets
            ordered from top to bottom. Positive offsets dilate the input contour;
            negative offsets erode it.
        grid_step (float): Physical grid spacing used to convert lateral offsets to
            pixels.
        control_u (Sequence[float] | torch.Tensor | None): Normalized depths paired
            with ``control_points``. Defaults to equally spaced depths.
        anchor_top (bool): Whether to subtract the top control-point offset.
            Defaults to True.
        z_downsample_factor (int): Number of slices processed together as a chunk.
            Defaults to 1.
        interpolation_mode (str): Interpolation mode retained by the public API;
            the current morphology path does not otherwise use it. Defaults to
            ``"trilinear"``.

    Returns:
        torch.Tensor: Extruded tensor with the new slice axis moved to ``extrude_dim``.

    Raises:
        ValueError: If the input is not a 2-D CUDA tensor or the slice/chunk settings
            are invalid.
    """
    if not permittivity.is_cuda:
        raise ValueError("Non-vertical extrusion currently requires a CUDA tensor")
    if permittivity.ndim != 2:
        raise ValueError(
            f"Sidewall extrusion expects a 2D permittivity plane, got {permittivity.ndim}D"
        )
    if z_downsample_factor < 1:
        raise ValueError(
            f"z_downsample_factor must be at least 1, got {z_downsample_factor}"
        )

    radius_px = _build_arbitrary_adius_schedule(
        extrude_size=extrude_size,
        control_points=control_points,
        grid_step=grid_step,
        control_u=control_u,
        anchor_top=anchor_top,
        dtype=permittivity.dtype,
        device=permittivity.device,
    )
    # Control points are specified from physical top to bottom, while tensor
    # axes follow increasing coordinates (bottom to top).
    radius_px = torch.flip(radius_px, dims=(0,))

    if torch.count_nonzero(radius_px).item() == 0:
        return _extrude_vertical(permittivity, extrude_dim, extrude_size)

    plane_4d = permittivity.unsqueeze(0).unsqueeze(0)
    dilation_cache = _build_integer_dilation_cache(
        plane_4d, radius_px[radius_px >= 0], "dilation"
    )
    erosion_cache = _build_integer_dilation_cache(
        plane_4d, radius_px[radius_px < 0].abs(), "erosion"
    )

    _, _, h, w = plane_4d.shape
    volume = torch.empty(
        (extrude_size, h, w),
        dtype=permittivity.dtype,
        device=permittivity.device,
    )

    # Build every physical z slice from the continuous radius schedule.
    # This avoids artificial smoothing from z interpolation while keeping
    # the expensive morphology calls cached by integer radius.
    chunk_z = max(1, int(z_downsample_factor))
    for start in range(0, extrude_size, chunk_z):
        stop = min(start + chunk_z, extrude_size)
        for k in range(start, stop):
            radius = radius_px[k]
            cache = dilation_cache if radius >= 0 else erosion_cache
            volume[k] = _fractional_radius_blend(
                radius_px_scalar=radius.abs(),
                morphology_cache=cache,
            )[0, 0]

    return torch.movedim(volume, 0, extrude_dim)


def extrude(
    permittivity,
    extrude_dim: int,
    extrude_size: int,
    extrude_arbitrary: bool = False,
    extrude_angle: float = 0.0,
    extrude_direction: str = "+",
    reference_plane: str = "middle",
    extrude_weights=None,
    base_permittivity=None,
    grid_step: float | None = None,
    physical_thickness: float | None = None,
    z_downsample_factor: int = 1,
    interpolation_mode: str = "trilinear",
):
    """Extrude a 2-D permittivity plane into a 3-D volume, optionally with sidewalls.

    Args:
        permittivity (torch.Tensor): 2-D permittivity plane to extrude. Non-vertical
            sidewall extrusion requires this tensor to be on CUDA.
        extrude_dim (int): Output axis along which the new extrusion dimension is
            inserted; must be ``0``, ``1``, or ``2``.
        extrude_size (int): Number of slices in the extrusion direction.
        extrude_angle (float): Sidewall angle in degrees. Zero produces uniform
            replication. Defaults to 0.0.
        extrude_direction (str): Sidewall direction sign, validated as ``"+"`` or
            ``"-"``. The current sidewall morphology path passes this value through
            but does not use it to flip the computed lateral offset. Defaults to ``"+"``.
        reference_plane (str): Reference plane for sidewall geometry: ``"top"``,
            ``"middle"``, or ``"bottom"``. Defaults to ``"middle"``.
        extrude_weights (Sequence[float] | torch.Tensor | None): Optional per-slice
            coverage weights. When supplied, they are also used for the final
            differentiable weighted blend. Defaults to None.
        base_permittivity (torch.Tensor | None): Optional background/base tensor
            used for the weighted blend when ``extrude_weights`` is supplied.
            Defaults to None.
        grid_step (float | None): Lateral physical grid spacing used for sidewall
            radius conversion. Defaults to None.
        physical_thickness (float | None): Total physical extrusion thickness.
            When omitted, the default is ``extrude_size * grid_step`` in the
            non-vertical path. Defaults to None.
        z_downsample_factor (int): Processing chunk size for sidewall slices.
            Defaults to 1.
        interpolation_mode (str): Interpolation mode parameter retained by the
            public API; the current sidewall morphology implementation does not
            otherwise use it. Defaults to ``"trilinear"``.

    Returns:
        torch.Tensor: Extruded permittivity volume with the new slice axis at
        ``extrude_dim``. When coverage weights are supplied, a gradient-scaling
        autograd wrapper applies ``1 / sum(weights)`` to the backward pass.

    Raises:
        ValueError: If the extrusion axis, direction, size, reference plane, angle,
            or coverage-weight/base-tensor shapes are invalid.
    """
    if extrude_dim not in [0, 1, 2]:
        raise ValueError(f"extrude_dim should be 0, 1 or 2, but got {extrude_dim}")
    if extrude_direction not in ["+", "-"]:
        raise ValueError(
            f"extrude_direction should be '+' or '-', but got {extrude_direction}"
        )
    if extrude_size < 1:
        raise ValueError(f"extrude_size should be at least 1, but got {extrude_size}")
    if reference_plane not in {"top", "middle", "bottom"}:
        raise ValueError(
            "reference_plane must be 'top', 'middle', or 'bottom', "
            f"got {reference_plane!r}"
        )
    if not -90.0 <= extrude_angle <= 90.0:
        raise ValueError(
            f"extrude_angle must be in the interval [-90, 90] degrees, got {extrude_angle}"
        )

    if extrude_arbitrary:
        extruded_permittivity = _extrude_arbitrary_morphology(
            permittivity=permittivity,
            extrude_dim=extrude_dim,
            extrude_size=extrude_size,
            # control_points=[0.0, -0.075, 0.075],
            control_points=[0.0, -0.01, -0.02, -0.01, 0, 0.01],
            # control_points=[-0.03, -0.01, 0, 0.01, 0.02],
            grid_step=grid_step,
            z_downsample_factor=z_downsample_factor,
            interpolation_mode=interpolation_mode,
        )
    else:
        angle_radians = math.radians(float(extrude_angle))
        if math.isclose(angle_radians, 0.0, abs_tol=1e-8):
            extruded_permittivity = _extrude_vertical(
                permittivity=permittivity,
                extrude_dim=extrude_dim,
                extrude_size=extrude_size,
            )
        else:
            extruded_permittivity = _extrude_sidewall_morphology(
                permittivity=permittivity,
                extrude_dim=extrude_dim,
                extrude_size=extrude_size,
                extrude_angle=angle_radians,
                extrude_direction=extrude_direction,
                reference_plane=reference_plane,
                grid_step=grid_step,
                physical_thickness=physical_thickness,
                extrude_weights=extrude_weights,
                z_downsample_factor=z_downsample_factor,
                interpolation_mode=interpolation_mode,
            )

    if extrude_weights is not None:
        weights = torch.as_tensor(
            extrude_weights,
            dtype=extruded_permittivity.dtype,
            device=extruded_permittivity.device,
        )
        if weights.numel() != extrude_size:
            raise ValueError(
                "Extrusion coverage weight size mismatch: "
                f"{weights.numel()} vs {extrude_size}"
            )
        weight_sum = float(weights.sum().item())
        grad_decay = 0.0 if weight_sum <= 0.0 else 1.0 / weight_sum
        weight_shape = [1] * extruded_permittivity.ndim
        weight_shape[extrude_dim] = extrude_size
        weights = weights.reshape(weight_shape)
        if base_permittivity is None:
            base_permittivity = torch.zeros_like(extruded_permittivity)
        else:
            base_permittivity = base_permittivity.to(
                dtype=extruded_permittivity.dtype,
                device=extruded_permittivity.device,
            )
            if tuple(base_permittivity.shape) != tuple(extruded_permittivity.shape):
                raise ValueError(
                    "Extrusion base permittivity shape mismatch: "
                    f"{tuple(base_permittivity.shape)} vs "
                    f"{tuple(extruded_permittivity.shape)}"
                )
        extruded_permittivity = (
            base_permittivity * (1 - weights) + extruded_permittivity * weights
        )

    return _ScaleGradient.apply(extruded_permittivity, grad_decay)
