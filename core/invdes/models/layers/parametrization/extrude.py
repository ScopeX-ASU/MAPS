import math

import torch

try:
    from .morph_op import dilation

    HAS_MORPH_OP = True
except:
    from kornia.morphology import dilation

    HAS_MORPH_OP = False


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def _make_disk_kernel(radius_int, device, dtype):
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
    grid_step,
    physical_thickness=None,
    extrude_weights=None,
    dtype=torch.float32,
    device=None,
):
    if extrude_size < 1:
        raise ValueError(f"extrude_size must be at least 1, got {extrude_size}")
    if grid_step is None or grid_step <= 0:
        raise ValueError(f"grid_step must be positive, got {grid_step}")
    if extrude_direction not in {"+", "-"}:
        raise ValueError(
            f"extrude_direction should be '+' or '-', but got {extrude_direction}"
        )
    if extrude_angle <= 0.0 or extrude_angle > 90.0:
        raise NotImplementedError(
            f"Only support sidewall angles in (0, 90], got {extrude_angle}"
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

    depths = torch.zeros(extrude_size, dtype=dtype, device=device)
    if extrude_size > 1:
        cumulative = torch.cumsum(slice_thickness, dim=0)
        if extrude_direction == "+":
            depths[1:] = cumulative[:-1]
        else:
            trailing = torch.flip(
                torch.cumsum(torch.flip(slice_thickness, dims=[0]), dim=0), dims=[0]
            )
            depths[:-1] = trailing[1:]

    if math.isclose(extrude_angle, 90.0, abs_tol=1e-8):
        return torch.zeros_like(depths)

    lateral_expansion = depths / math.tan(math.radians(float(extrude_angle)))
    return lateral_expansion / float(grid_step)


def _build_integer_dilation_cache(field_4d, radius_px):
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
        if HAS_MORPH_OP:
            cache[int(radius)] = dilation(
                field_4d,
                kernel=kernel,
                structuring_element=None,
                border_type="geodesic",
                engine="triton",
            )
        else:
            cache[int(radius)] = dilation(
                field_4d,
                kernel=kernel,
                structuring_element=None,
                border_type="geodesic",
                engine="convolution",
            )
    return cache


def _fractional_radius_blend(radius_px_scalar, dilation_cache):
    radius_px_scalar = radius_px_scalar.clamp_min(0.0)
    r0_float = torch.floor(radius_px_scalar)
    r1_float = torch.ceil(radius_px_scalar)
    r0 = int(r0_float.item())
    r1 = int(r1_float.item())

    if r0 == r1:
        return dilation_cache[r0]

    blend = (radius_px_scalar - r0_float).clamp(0.0, 1.0)
    return (1.0 - blend) * dilation_cache[r0] + blend * dilation_cache[r1]


def _extrude_vertical(permittivity, extrude_dim, extrude_size):
    return permittivity.unsqueeze(extrude_dim).repeat_interleave(
        extrude_size, dim=extrude_dim
    )


def _extrude_sidewall_morphology(
    permittivity,
    extrude_dim,
    extrude_size,
    extrude_angle,
    extrude_direction,
    grid_step,
    physical_thickness=None,
    extrude_weights=None,
    z_downsample_factor=1,
    interpolation_mode="trilinear",
):
    if not permittivity.is_cuda:
        raise ValueError("Non-90-degree extrusion currently requires a CUDA tensor")
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
        grid_step=grid_step,
        physical_thickness=physical_thickness,
        extrude_weights=extrude_weights,
        dtype=permittivity.dtype,
        device=permittivity.device,
    )

    if torch.count_nonzero(radius_px).item() == 0:
        return _extrude_vertical(permittivity, extrude_dim, extrude_size)

    plane_4d = permittivity.unsqueeze(0).unsqueeze(0)
    dilation_cache = _build_integer_dilation_cache(plane_4d, radius_px)

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
            volume[k] = _fractional_radius_blend(
                radius_px_scalar=radius_px[k],
                dilation_cache=dilation_cache,
            )[0, 0]

    return torch.movedim(volume, 0, extrude_dim)


def extrude(
    permittivity,
    extrude_dim: int,
    extrude_size: int,
    extrude_angle: float = 90.0,
    extrude_direction: str = "+",
    extrude_weights=None,
    base_permittivity=None,
    grid_step: float | None = None,
    physical_thickness: float | None = None,
    z_downsample_factor: int = 1,
    interpolation_mode: str = "trilinear",
):
    if extrude_dim not in [0, 1, 2]:
        raise ValueError(f"extrude_dim should be 0, 1 or 2, but got {extrude_dim}")
    if extrude_direction not in ["+", "-"]:
        raise ValueError(
            f"extrude_direction should be '+' or '-', but got {extrude_direction}"
        )
    if extrude_size < 1:
        raise ValueError(f"extrude_size should be at least 1, but got {extrude_size}")

    grad_decay = 1.0 / float(extrude_size)
    if math.isclose(extrude_angle, 90.0, abs_tol=1e-8):
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
            extrude_angle=extrude_angle,
            extrude_direction=extrude_direction,
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
