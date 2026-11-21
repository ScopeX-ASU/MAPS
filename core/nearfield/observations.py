"""Observation coordinates, polarization bases, and explicit angular quadrature."""

import torch
from torch import Tensor


def _real_finite(values: Tensor, name: str) -> None:
    """Validate a finite real floating-point coordinate tensor.

    Args:
        values: Coordinate or quadrature tensor.
        name: Argument name used in validation errors.

    Raises:
        ValueError: If values are not finite real floating-point samples.
    """
    if not isinstance(values, Tensor) or not values.is_floating_point():
        raise ValueError(f"{name} must be a real floating-point tensor")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")


def direction_cosines(
    u: Tensor, v: Tensor, *, normal_axis: int = 2, normal_sign: int = 1
) -> Tensor:
    """Construct unit observation directions from transverse direction cosines.

    Args:
        u: First transverse direction cosine, a finite real tensor.
        v: Second transverse cosine, broadcastable with ``u`` and sharing its
            dtype and device. Transverse axes follow increasing x/y/z order.
        normal_axis: Normal axis, 0 for x, 1 for y, or 2 for z.
        normal_sign: Hemisphere orientation, -1 or +1.

    Returns:
        Real unit vectors with the broadcast shape of u/v followed by 3.

    Raises:
        ValueError: If coordinates are invalid or u squared plus v squared
            exceeds one, corresponding to a nonpropagating direction.
    """
    _real_finite(u, "u")
    _real_finite(v, "v")
    if u.device != v.device or u.dtype != v.dtype:
        raise ValueError("u and v must share dtype and device")
    if normal_axis not in (0, 1, 2) or normal_sign not in (-1, 1):
        raise ValueError("Invalid normal axis or sign")
    u, v = torch.broadcast_tensors(u, v)
    transverse_squared = u.square() + v.square()
    if (transverse_squared > 1).any():
        raise ValueError(
            "Nonpropagating direction: u squared plus v squared exceeds one"
        )
    components = [u, v]
    components.insert(normal_axis, normal_sign * torch.sqrt(1 - transverse_squared))
    return torch.stack(components, dim=-1)


def cartesian_observation_plane(
    u_coordinates_m: Tensor,
    v_coordinates_m: Tensor,
    *,
    normal_axis: int,
    position_m: float | Tensor,
) -> Tensor:
    """Create a Cartesian plane grid using explicit physical sample positions.

    Args:
        u_coordinates_m: Increasing first transverse coordinate vector, in m.
        v_coordinates_m: Increasing second transverse vector, in m, sharing
            dtype/device with u. Transverse axes follow increasing x/y/z order.
        normal_axis: Plane normal axis, 0, 1, or 2.
        position_m: Scalar plane position along the normal axis, in meters.

    Returns:
        Real positions ``(Nu*Nv,3)``, with ``meshgrid(indexing='ij')`` ordering:
        the v coordinate varies fastest.

    Raises:
        ValueError: If axes, positions, or vector shapes are invalid.
    """
    _real_finite(u_coordinates_m, "u_coordinates_m")
    _real_finite(v_coordinates_m, "v_coordinates_m")
    if normal_axis not in (0, 1, 2):
        raise ValueError("normal_axis must be 0, 1, or 2")
    if (
        u_coordinates_m.device != v_coordinates_m.device
        or u_coordinates_m.dtype != v_coordinates_m.dtype
    ):
        raise ValueError("Coordinate vectors must share dtype and device")
    for axis in (u_coordinates_m, v_coordinates_m):
        if axis.ndim != 1 or axis.numel() == 0 or (axis[1:] <= axis[:-1]).any():
            raise ValueError(
                "Coordinate vectors must be nonempty, one-dimensional and increasing"
            )
    position = torch.as_tensor(
        position_m, device=u_coordinates_m.device, dtype=u_coordinates_m.dtype
    )
    if position.numel() != 1 or not torch.isfinite(position).all():
        raise ValueError("position_m must be a finite scalar")
    u_grid, v_grid = torch.meshgrid(u_coordinates_m, v_coordinates_m, indexing="ij")
    components = [u_grid, v_grid]
    components.insert(normal_axis, position.reshape(()).expand_as(u_grid))
    return torch.stack(components, dim=-1).reshape(-1, 3)


def spherical_components(vector: Tensor, directions: Tensor) -> Tensor:
    """Resolve Cartesian vectors into radial, polar-theta, azimuthal-phi components.

    Args:
        vector: Complex vectors shaped ``(...,M,3)``.
        directions: Real unit vectors shaped ``(M,3)`` or broadcastable with
            vector, sharing its device and real dtype. Theta is measured from
            +z and phi from +x toward +y.

    Returns:
        Complex ``(radial,theta,phi)`` components with the broadcast vector shape.
        At either z pole, phi is defined as zero.

    Raises:
        ValueError: If layout, dtype, device, finiteness or unit length is invalid.
    """
    if (
        not isinstance(vector, Tensor)
        or not vector.is_complex()
        or vector.shape[-1:] != (3,)
    ):
        raise ValueError("vector must be complex with final dimension 3")
    _real_finite(directions, "directions")
    if directions.shape[-1:] != (3,):
        raise ValueError("directions must have final dimension 3")
    if vector.device != directions.device or vector.real.dtype != directions.dtype:
        raise ValueError("Directions must match the vector device and real dtype")
    if not torch.isfinite(vector).all():
        raise ValueError("vector must be finite")
    norm = torch.linalg.vector_norm(directions, dim=-1)
    if not torch.allclose(norm, torch.ones_like(norm), atol=2e-6, rtol=2e-6):
        raise ValueError("directions must be unit length")
    x, y, z = directions.unbind(-1)
    at_pole = (x == 0) & (y == 0)
    phi = torch.atan2(
        torch.where(at_pole, torch.zeros_like(y), y),
        torch.where(at_pole, torch.ones_like(x), x),
    )
    sin_theta = torch.linalg.vector_norm(directions[..., :2], dim=-1)
    theta_basis = torch.stack(
        (z * torch.cos(phi), z * torch.sin(phi), -sin_theta), dim=-1
    )
    phi_basis = torch.stack(
        (-torch.sin(phi), torch.cos(phi), torch.zeros_like(phi)), dim=-1
    )
    return torch.stack(
        tuple(
            (vector * basis).sum(-1) for basis in (directions, theta_basis, phi_basis)
        ),
        dim=-1,
    )


def integrated_directional_power(
    intensity: Tensor, solid_angle_weights: Tensor
) -> Tensor:
    """Integrate directional radiation intensity using explicit steradian weights.

    Args:
        intensity: Real nonnegative radiation intensity ``(F,M)``, in W/sr.
        solid_angle_weights: Nonnegative quadrature weights ``(M,)``, in sr,
            sharing intensity dtype and device. These must include the angular
            Jacobian appropriate to the observation grid. Zero weights at
            spherical-grid poles are allowed; the total weight must be positive.

    Returns:
        Real integrated power per frequency ``(F,)``, in watts.

    Raises:
        ValueError: If values, shapes, dtype/device, or quadrature are invalid.
    """
    _real_finite(intensity, "intensity")
    _real_finite(solid_angle_weights, "solid_angle_weights")
    if intensity.ndim != 2 or min(intensity.shape) == 0:
        raise ValueError("intensity must have shape (F,M) with nonempty axes")
    if solid_angle_weights.shape != (intensity.shape[1],):
        raise ValueError("solid_angle_weights must match the observation dimension")
    if (
        intensity.dtype != solid_angle_weights.dtype
        or intensity.device != solid_angle_weights.device
    ):
        raise ValueError(
            "Intensity and solid angle weights must share dtype and device"
        )
    if (
        (intensity < 0).any()
        or (solid_angle_weights < 0).any()
        or not (solid_angle_weights > 0).any()
    ):
        raise ValueError(
            "Intensity and weights must be nonnegative with a positive total solid angle"
        )
    return (intensity * solid_angle_weights).sum(-1)
