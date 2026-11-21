"""Native PyTorch near-to-far field projection for sampled surfaces.

Fields use SI units and the ``exp(-i omega t)`` phasor convention. The outgoing
homogeneous-medium Green function is therefore ``exp(+i k R) / (4 pi R)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.checkpoint import checkpoint

EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6
C_0 = 1.0 / math.sqrt(EPSILON_0 * MU_0)


@dataclass(frozen=True)
class Surface:
    """Tangential field samples and quadrature on an oriented surface.

    Args:
        positions_m: Sample positions, shape ``(P, 3)``, in meters.
        normals: Unit surface normals pointing out of the enclosed region,
            shape ``(P, 3)``.
        weights_m2: Surface quadrature weights, shape ``(P,)``, in square
            meters.
        electric_fields: Complex electric field samples, shape ``(F, P, 3)``,
            in V/m.
        magnetic_fields: Complex magnetic field samples, shape ``(F, P, 3)``,
            in A/m.
        frequencies_hz: Positive frequency samples, shape ``(F,)``, in Hz.

    Notes:
        All tensors must share a device. Geometry and frequencies are real;
        fields are complex. The surface equivalence currents are
        ``J_s = n x H`` and ``M_s = -n x E``.
    """

    positions_m: torch.Tensor
    normals: torch.Tensor
    weights_m2: torch.Tensor
    electric_fields: torch.Tensor
    magnetic_fields: torch.Tensor
    frequencies_hz: torch.Tensor

    def __post_init__(self) -> None:
        """Validate geometry, common device/dtype, and finite phasor samples.

        Raises:
            TypeError: If inputs violate the tensor or real/complex dtype contract.
            ValueError: If shapes, units-bearing values, or unit normals are invalid.
        """
        values = {
            "positions_m": self.positions_m,
            "normals": self.normals,
            "weights_m2": self.weights_m2,
            "electric_fields": self.electric_fields,
            "magnetic_fields": self.magnetic_fields,
            "frequencies_hz": self.frequencies_hz,
        }
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if self.positions_m.ndim != 2 or self.positions_m.shape[-1] != 3:
            raise ValueError("positions_m must have shape (P, 3)")
        points = self.positions_m.shape[0]
        if points == 0:
            raise ValueError("a surface must contain at least one sample")
        if self.normals.shape != (points, 3):
            raise ValueError("normals must have shape (P, 3)")
        if self.weights_m2.shape != (points,):
            raise ValueError("weights_m2 must have shape (P,)")
        if self.electric_fields.ndim != 3 or self.electric_fields.shape[1:] != (
            points,
            3,
        ):
            raise ValueError("electric_fields must have shape (F, P, 3)")
        if self.magnetic_fields.shape != self.electric_fields.shape:
            raise ValueError(
                "magnetic_fields must have the same shape as electric_fields"
            )
        frequencies = self.electric_fields.shape[0]
        if frequencies == 0 or self.frequencies_hz.shape != (frequencies,):
            raise ValueError("frequencies_hz must have shape (F,) with F > 0")
        if any(t.device != self.positions_m.device for t in values.values()):
            raise ValueError("all Surface tensors must be on the same device")
        if (
            self.positions_m.is_complex()
            or self.normals.is_complex()
            or self.weights_m2.is_complex()
            or self.frequencies_hz.is_complex()
        ):
            raise TypeError("positions, normals, weights, and frequencies must be real")
        if (
            not self.electric_fields.is_complex()
            or not self.magnetic_fields.is_complex()
        ):
            raise TypeError("electric_fields and magnetic_fields must be complex")
        if self.magnetic_fields.dtype != self.electric_fields.dtype:
            raise TypeError(
                "electric_fields and magnetic_fields must have the same complex dtype"
            )
        if any(
            t.dtype != self.electric_fields.real.dtype
            for t in (
                self.positions_m,
                self.normals,
                self.weights_m2,
                self.frequencies_hz,
            )
        ):
            raise TypeError(
                "real geometry, quadrature, and frequency tensors must match the real field dtype"
            )
        if (
            not torch.isfinite(self.positions_m).all()
            or not torch.isfinite(self.normals).all()
        ):
            raise ValueError("surface positions and normals must be finite")
        if not torch.isfinite(self.weights_m2).all() or torch.any(self.weights_m2 <= 0):
            raise ValueError("quadrature weights must be finite and positive")
        if not torch.isfinite(self.frequencies_hz).all() or torch.any(
            self.frequencies_hz <= 0
        ):
            raise ValueError("frequencies_hz must be finite and positive")
        if (
            not torch.isfinite(self.electric_fields).all()
            or not torch.isfinite(self.magnetic_fields).all()
        ):
            raise ValueError("electric and magnetic field samples must be finite")
        norm = torch.linalg.vector_norm(self.normals, dim=-1)
        if not torch.allclose(norm, torch.ones_like(norm), atol=2e-5, rtol=2e-5):
            raise ValueError("normals must be unit length")


def _trap_weights(coordinates: torch.Tensor) -> torch.Tensor:
    """Return composite trapezoid weights for ascending nodes.

    Args:
        coordinates: Real sample positions, shape ``(N,)`` with N >= 2.

    Returns:
        Positive nodal weights in the same length units, shape ``(N,)``.

    Raises:
        ValueError: If coordinates are not a strictly increasing vector.
    """
    if coordinates.ndim != 1 or coordinates.numel() < 2:
        raise ValueError("each planar coordinate vector must have at least two nodes")
    differences = coordinates[1:] - coordinates[:-1]
    if not torch.all(differences > 0):
        raise ValueError("planar coordinate vectors must be strictly increasing")
    weights = torch.empty_like(coordinates)
    weights[0] = differences[0] / 2
    weights[-1] = differences[-1] / 2
    if coordinates.numel() > 2:
        weights[1:-1] = (differences[:-1] + differences[1:]) / 2
    return weights


def make_planar_surface(
    origin_m: torch.Tensor,
    u_axis: torch.Tensor,
    v_axis: torch.Tensor,
    u_coordinates_m: torch.Tensor,
    v_coordinates_m: torch.Tensor,
    electric_fields: torch.Tensor,
    magnetic_fields: torch.Tensor,
    frequencies_hz: torch.Tensor,
) -> Surface:
    """Build a tensor-product planar surface with trapezoidal quadrature.

    Args:
        origin_m: Position of the coordinate-grid origin, shape ``(3,)``.
        u_axis: Direction vector for ``u_coordinates_m``, shape ``(3,)``.
        v_axis: Direction vector for ``v_coordinates_m``, shape ``(3,)``.
        u_coordinates_m: Strictly increasing coordinates in meters.
        v_coordinates_m: Strictly increasing coordinates in meters.
        electric_fields: Complex samples shaped ``(F, Nv, Nu, 3)`` or
            ``(F, Nv * Nu, 3)``.
        magnetic_fields: Complex samples with the same shape as electric_fields.
        frequencies_hz: Positive frequencies shaped ``(F,)``.

    Returns:
        A ``Surface`` with points ordered by v-major then u-major flattening.

    Raises:
        ValueError: If axes are collinear, coordinates are invalid, or field
            shapes do not match the tensor-product grid.
    """
    tensors = (origin_m, u_axis, v_axis, u_coordinates_m, v_coordinates_m)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("planar geometry arguments must be tensors")
    if origin_m.shape != (3,) or u_axis.shape != (3,) or v_axis.shape != (3,):
        raise ValueError("origin_m, u_axis, and v_axis must have shape (3,)")
    cross = torch.linalg.cross(u_axis, v_axis)
    area_scale = torch.linalg.vector_norm(cross)
    if not torch.isfinite(area_scale) or area_scale <= 0:
        raise ValueError("u_axis and v_axis must be finite and non-collinear")
    nu, nv = u_coordinates_m.numel(), v_coordinates_m.numel()
    if u_coordinates_m.ndim != 1 or v_coordinates_m.ndim != 1:
        raise ValueError("planar coordinate vectors must be one-dimensional")
    u_weights = _trap_weights(u_coordinates_m)
    v_weights = _trap_weights(v_coordinates_m)
    positions = (
        origin_m
        + v_coordinates_m[:, None, None] * v_axis
        + u_coordinates_m[None, :, None] * u_axis
    )
    positions = positions.reshape(-1, 3)
    normals = (cross / area_scale).expand(positions.shape[0], 3)
    weights = (area_scale * v_weights[:, None] * u_weights[None, :]).reshape(-1)
    expected = (frequencies_hz.numel(), nv, nu, 3)
    if magnetic_fields.shape != electric_fields.shape:
        raise ValueError("magnetic_fields must have the same shape as electric_fields")
    if electric_fields.shape == expected:
        electric_fields = electric_fields.reshape(frequencies_hz.numel(), -1, 3)
        magnetic_fields = magnetic_fields.reshape(frequencies_hz.numel(), -1, 3)
    elif electric_fields.shape != (frequencies_hz.numel(), nu * nv, 3):
        raise ValueError(f"field samples must have shape {expected} or (F, Nv*Nu, 3)")
    return Surface(
        positions, normals, weights, electric_fields, magnetic_fields, frequencies_hz
    )


def concatenate_surfaces(surfaces: Sequence[Surface]) -> Surface:
    """Concatenate surface patches that share a frequency grid and device.

    Args:
        surfaces: Non-empty sequence of independently sampled surface patches.

    Returns:
        A single surface with all samples in input order.

    Raises:
        ValueError: If frequency grids, field dtypes, or devices differ.
    """
    if not surfaces:
        raise ValueError("at least one surface is required")
    first = surfaces[0]
    for surface in surfaces[1:]:
        if (
            surface.frequencies_hz.shape != first.frequencies_hz.shape
            or not torch.equal(surface.frequencies_hz, first.frequencies_hz)
        ):
            raise ValueError("all surfaces must use the same frequency grid")
        if surface.positions_m.device != first.positions_m.device:
            raise ValueError("all surfaces must be on the same device")
        if (
            surface.electric_fields.dtype != first.electric_fields.dtype
            or surface.magnetic_fields.dtype != first.magnetic_fields.dtype
        ):
            raise ValueError("all surfaces must use the same field dtypes")
    return Surface(
        torch.cat([s.positions_m for s in surfaces]),
        torch.cat([s.normals for s in surfaces]),
        torch.cat([s.weights_m2 for s in surfaces]),
        torch.cat([s.electric_fields for s in surfaces], dim=1),
        torch.cat([s.magnetic_fields for s in surfaces], dim=1),
        first.frequencies_hz,
    )


def _medium_parameters(
    frequencies_hz: torch.Tensor,
    eps_r: float | torch.Tensor,
    mu_r: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve homogeneous lossless medium parameters on the field device.

    Args:
        frequencies_hz: Positive frequency vector, shape ``(F,)``.
        eps_r: Real positive scalar relative permittivity.
        mu_r: Real positive scalar relative permeability.

    Returns:
        Angular frequencies in rad/s, absolute permittivity in F/m,
        permeability in H/m, wave numbers in rad/m, and impedance in ohms.
        Frequency-dependent quantities have shape ``(F,)``; others are scalars.

    Raises:
        ValueError: If a material parameter is complex, nonfinite, or nonpositive.
    """
    dtype = frequencies_hz.dtype
    device = frequencies_hz.device
    if any(torch.is_complex(torch.as_tensor(value)) for value in (eps_r, mu_r)):
        raise ValueError("eps_r and mu_r must be real lossless material parameters")
    eps = torch.as_tensor(eps_r, dtype=dtype, device=device)
    mu = torch.as_tensor(mu_r, dtype=dtype, device=device)
    if (
        eps.numel() != 1
        or mu.numel() != 1
        or not torch.isfinite(eps).all()
        or not torch.isfinite(mu).all()
        or eps <= 0
        or mu <= 0
    ):
        raise ValueError("eps_r and mu_r must be finite positive scalars")
    omega = 2 * math.pi * frequencies_hz
    epsilon = EPSILON_0 * eps
    permeability = MU_0 * mu
    k = omega * torch.sqrt(epsilon * permeability)
    eta = torch.sqrt(permeability / epsilon)
    return omega, epsilon, permeability, k, eta


def _chunk_nearfield(
    observation_points: torch.Tensor,
    source_positions: torch.Tensor,
    electric_current: torch.Tensor,
    magnetic_current: torch.Tensor,
    weights: torch.Tensor,
    omega: torch.Tensor,
    epsilon: torch.Tensor,
    permeability: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate and integrate one nonsingular dyadic Green-function block.

    Args:
        observation_points: Real target positions ``(M,3)`` in meters.
        source_positions: Real source positions ``(P,3)`` in meters.
        electric_current: Complex equivalent electric current ``(P,3)`` in A/m.
        magnetic_current: Complex equivalent magnetic current ``(P,3)`` in V/m.
        weights: Source quadrature areas ``(P,)`` in square meters.
        omega: Scalar angular frequency in rad/s.
        epsilon: Absolute permittivity in F/m.
        permeability: Absolute permeability in H/m.
        k: Scalar wave number in rad/m.

    Returns:
        Integrated electric and magnetic fields ``(M,3)`` in V/m and A/m.

    Raises:
        ValueError: If a target coincides with a source sample.
    """
    displacement = observation_points[:, None, :] - source_positions[None, :, :]
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    if torch.any(radius == 0):
        raise ValueError("observation points must not coincide with surface samples")
    radial = displacement / radius[..., None]
    green = torch.exp(1j * k * radius) / (4 * math.pi * radius)
    inverse_kr = 1 / (k * radius)
    a = 1 + 1j * inverse_kr - inverse_kr.square()
    b = -1 - 3j * inverse_kr + 3 * inverse_kr.square()

    def dyadic_apply(vector: torch.Tensor) -> torch.Tensor:
        """Apply G I + Hess(G)/k² to a sampled Cartesian current.

        Args:
            vector: Complex source vectors, shape ``(P,3)``.

        Returns:
            Complex dyadic products, shape ``(M,P,3)``.
        """
        projection = torch.sum(radial * vector[None, :, :], dim=-1)
        return green[..., None] * (
            a[..., None] * vector[None, :, :]
            + b[..., None] * radial * projection[..., None]
        )

    grad_green = green[..., None] * (1j * k - 1 / radius)[..., None] * radial
    cross_j = torch.linalg.cross(
        grad_green, electric_current[None, :, :].expand_as(grad_green), dim=-1
    )
    cross_m = torch.linalg.cross(
        grad_green, magnetic_current[None, :, :].expand_as(grad_green), dim=-1
    )
    electric = 1j * omega * permeability * dyadic_apply(electric_current) - cross_m
    magnetic = cross_j + 1j * omega * epsilon * dyadic_apply(magnetic_current)
    return (electric * weights[None, :, None]).sum(dim=1), (
        magnetic * weights[None, :, None]
    ).sum(dim=1)


def project_nearfield(
    surface: Surface,
    observation_points_m: torch.Tensor,
    eps_r: float | torch.Tensor = 1.0,
    mu_r: float | torch.Tensor = 1.0,
    *,
    source_chunk_size: int = 256,
    observation_chunk_size: int = 256,
    checkpoint_chunks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project equivalent currents to arbitrary observation points.

    Args:
        surface: Oriented sampled surface containing complex E/H fields.
        observation_points_m: Real observation locations, shape ``(M, 3)``.
            Points must not coincide with a surface sample.
        eps_r: Homogeneous, lossless relative permittivity (positive scalar).
        mu_r: Homogeneous, lossless relative permeability (positive scalar).
        source_chunk_size: Maximum source samples evaluated together.
        observation_chunk_size: Maximum observation points evaluated together.
        checkpoint_chunks: Recompute block kernels during backward to bound
            saved Green-function memory when gradients are requested.

    Returns:
        Tuple ``(electric, magnetic)`` with complex tensors of shape ``(F,M,3)``.

    Notes:
        Uses ``exp(-i omega t)``, ``G=exp(+i k R)/(4 pi R)``, and surface
        currents ``J_s=n x H`` and ``M_s=-n x E``. Differentiation through
        fields, observation geometry, quadrature, and scalar material inputs
        uses native PyTorch autograd.
    """
    if observation_points_m.ndim != 2 or observation_points_m.shape[-1] != 3:
        raise ValueError("observation_points_m must have shape (M, 3)")
    if observation_points_m.shape[0] == 0:
        raise ValueError("at least one observation point is required")
    if observation_points_m.device != surface.positions_m.device:
        raise ValueError("observation points and surface must share a device")
    if observation_points_m.is_complex():
        raise TypeError("observation_points_m must be real")
    if observation_points_m.dtype != surface.positions_m.dtype:
        raise TypeError("observation_points_m must match the surface real dtype")
    if source_chunk_size <= 0 or observation_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if not torch.isfinite(observation_points_m).all():
        raise ValueError("observation points must be finite")
    omega_all, epsilon, permeability, k_all, _ = _medium_parameters(
        surface.frequencies_hz, eps_r, mu_r
    )
    normals_complex = surface.normals.to(dtype=surface.electric_fields.dtype)
    currents_j = torch.linalg.cross(
        normals_complex[None, :, :], surface.magnetic_fields, dim=-1
    )
    currents_m = -torch.linalg.cross(
        normals_complex[None, :, :], surface.electric_fields, dim=-1
    )
    field_e_f, field_h_f = [], []
    for frequency_index in range(surface.frequencies_hz.numel()):
        obs_e, obs_h = [], []
        omega = omega_all[frequency_index]
        k = k_all[frequency_index]
        for obs_start in range(
            0, observation_points_m.shape[0], observation_chunk_size
        ):
            obs = observation_points_m[obs_start : obs_start + observation_chunk_size]
            total_e = torch.zeros(
                (obs.shape[0], 3),
                dtype=surface.electric_fields.dtype,
                device=obs.device,
            )
            total_h = torch.zeros_like(total_e)
            for source_start in range(
                0, surface.positions_m.shape[0], source_chunk_size
            ):
                source_stop = min(
                    source_start + source_chunk_size, surface.positions_m.shape[0]
                )
                source_pos = surface.positions_m[source_start:source_stop]
                current_j = currents_j[frequency_index, source_start:source_stop]
                current_m = currents_m[frequency_index, source_start:source_stop]
                weights = surface.weights_m2[source_start:source_stop]
                args = (
                    obs,
                    source_pos,
                    current_j,
                    current_m,
                    weights,
                    omega,
                    epsilon,
                    permeability,
                    k,
                )
                if (
                    checkpoint_chunks
                    and torch.is_grad_enabled()
                    and any(t.requires_grad for t in args)
                ):
                    block_e, block_h = checkpoint(
                        _chunk_nearfield, *args, use_reentrant=False
                    )
                else:
                    block_e, block_h = _chunk_nearfield(*args)
                total_e = total_e + block_e
                total_h = total_h + block_h
            obs_e.append(total_e)
            obs_h.append(total_h)
        field_e_f.append(torch.cat(obs_e, dim=0))
        field_h_f.append(torch.cat(obs_h, dim=0))
    return torch.stack(field_e_f), torch.stack(field_h_f)


def direction_vectors_spherical(
    theta_rad: torch.Tensor, phi_rad: torch.Tensor
) -> torch.Tensor:
    """Convert polar and azimuth angles into Cartesian unit directions.

    Args:
        theta_rad: Polar angle from +z, arbitrary shape.
        phi_rad: Azimuth from +x toward +y, broadcastable with theta_rad.

    Returns:
        Unit direction vectors with the broadcast angle shape followed by 3.
    """
    theta, phi = torch.broadcast_tensors(theta_rad, phi_rad)
    return torch.stack(
        (
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ),
        dim=-1,
    )


def _farfield_current_integrals(
    directions: torch.Tensor,
    positions: torch.Tensor,
    electric_current: torch.Tensor,
    magnetic_current: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute area-weighted Fourier integrals for one angular/source block.

    Args:
        directions: Real unit observation vectors ``(M,3)``.
        positions: Source coordinates ``(P,3)`` in meters.
        electric_current: Complex surface electric currents ``(P,3)`` in A/m.
        magnetic_current: Complex surface magnetic currents ``(P,3)`` in V/m.
        weights: Source areas ``(P,)`` in square meters.
        k: Scalar wave number in rad/m.

    Returns:
        Complex Fourier integrals of J and M, each ``(M,3)``, with units A m
        and V m. The source phase is ``exp(-i k direction dot position)``.
    """
    phase = torch.exp(-1j * k * (directions @ positions.T))
    weighted_j = electric_current * weights[:, None]
    weighted_m = magnetic_current * weights[:, None]
    return phase @ weighted_j, phase @ weighted_m


def project_farfield(
    surface: Surface,
    directions: torch.Tensor,
    eps_r: float | torch.Tensor = 1.0,
    mu_r: float | torch.Tensor = 1.0,
    *,
    source_chunk_size: int = 512,
    observation_chunk_size: int = 256,
    checkpoint_chunks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project equivalent currents into the homogeneous-medium radiation zone.

    Args:
        surface: Oriented sampled surface containing complex E/H fields.
        directions: Real unit observation directions, shape ``(M,3)``.
        eps_r: Homogeneous, lossless relative permittivity.
        mu_r: Homogeneous, lossless relative permeability.
        source_chunk_size: Number of source samples evaluated in each block.
        observation_chunk_size: Number of directions evaluated in each block.
        checkpoint_chunks: Recompute Fourier kernels during backward to limit
            saved phase-array memory when gradients are requested.

    Returns:
        Tuple ``(electric, magnetic)`` of complex far-field amplitudes, shape
        ``(F,M,3)``. Electric and magnetic amplitudes have units V and A,
        respectively, with the radial factor ``exp(+ikr)/r`` omitted.
    """
    if directions.ndim != 2 or directions.shape[-1] != 3 or directions.shape[0] == 0:
        raise ValueError("directions must have shape (M, 3) with M > 0")
    if directions.device != surface.positions_m.device or directions.is_complex():
        raise ValueError("directions must be real and share the surface device")
    norm = torch.linalg.vector_norm(directions, dim=-1)
    if not torch.allclose(norm, torch.ones_like(norm), atol=2e-5, rtol=2e-5):
        raise ValueError("directions must be unit length")
    if source_chunk_size <= 0 or observation_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    omega_all, epsilon, permeability, k_all, eta = _medium_parameters(
        surface.frequencies_hz, eps_r, mu_r
    )
    normals_complex = surface.normals.to(dtype=surface.electric_fields.dtype)
    j = torch.linalg.cross(normals_complex[None], surface.magnetic_fields, dim=-1)
    m = -torch.linalg.cross(normals_complex[None], surface.electric_fields, dim=-1)
    electric_f, magnetic_f = [], []
    for fi in range(surface.frequencies_hz.numel()):
        electric_blocks, magnetic_blocks = [], []
        k = k_all[fi]
        for obs_start in range(0, directions.shape[0], observation_chunk_size):
            direction_block = directions[obs_start : obs_start + observation_chunk_size]
            j_integrals, m_integrals = [], []
            for start in range(0, surface.positions_m.shape[0], source_chunk_size):
                stop = min(start + source_chunk_size, surface.positions_m.shape[0])
                args = (
                    direction_block,
                    surface.positions_m[start:stop],
                    j[fi, start:stop],
                    m[fi, start:stop],
                    surface.weights_m2[start:stop],
                    k,
                )
                if (
                    checkpoint_chunks
                    and torch.is_grad_enabled()
                    and any(t.requires_grad for t in args)
                ):
                    current_j_integral, current_m_integral = checkpoint(
                        _farfield_current_integrals, *args, use_reentrant=False
                    )
                else:
                    current_j_integral, current_m_integral = (
                        _farfield_current_integrals(*args)
                    )
                j_integrals.append(current_j_integral)
                m_integrals.append(current_m_integral)
            current_j_integral = torch.stack(j_integrals).sum(dim=0)
            current_m_integral = torch.stack(m_integrals).sum(dim=0)
            projected_j = current_j_integral - direction_block * torch.sum(
                direction_block * current_j_integral, dim=-1, keepdim=True
            )
            direction_complex = direction_block.to(current_m_integral.dtype)
            amplitude = (
                1j
                * k
                * (
                    eta * projected_j
                    - torch.linalg.cross(direction_complex, current_m_integral, dim=-1)
                )
                / (4 * math.pi)
            )
            magnetic = torch.linalg.cross(direction_complex, amplitude, dim=-1) / eta
            electric_blocks.append(amplitude)
            magnetic_blocks.append(magnetic)
        electric_f.append(torch.cat(electric_blocks, dim=0))
        magnetic_f.append(torch.cat(magnetic_blocks, dim=0))
    return torch.stack(electric_f), torch.stack(magnetic_f)


def directional_power(
    electric_farfield: torch.Tensor,
    mu_r: float | torch.Tensor = 1.0,
    eps_r: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Return time-average radiation intensity ``r^2 <S_r>`` per direction.

    Args:
        electric_farfield: Complex electric amplitude, shape ``(..., 3)``.
        mu_r: Positive relative permeability of the observation medium.
        eps_r: Positive relative permittivity of the observation medium.

    Returns:
        Real intensity in W/sr, shape ``electric_farfield.shape[:-1]``.
    """
    if electric_farfield.shape[-1] != 3 or not electric_farfield.is_complex():
        raise ValueError("electric_farfield must be complex with final dimension 3")
    if any(torch.is_complex(torch.as_tensor(value)) for value in (eps_r, mu_r)):
        raise ValueError("eps_r and mu_r must be real lossless material parameters")
    eps = torch.as_tensor(
        eps_r, dtype=electric_farfield.real.dtype, device=electric_farfield.device
    )
    mu = torch.as_tensor(
        mu_r, dtype=electric_farfield.real.dtype, device=electric_farfield.device
    )
    if (
        eps.numel() != 1
        or mu.numel() != 1
        or not torch.isfinite(eps).all()
        or not torch.isfinite(mu).all()
        or eps <= 0
        or mu <= 0
    ):
        raise ValueError("eps_r and mu_r must be finite positive scalars")
    impedance = math.sqrt(MU_0 / EPSILON_0) * torch.sqrt(mu / eps)
    return torch.sum(torch.abs(electric_farfield) ** 2, dim=-1) / (2 * impedance)


def projection_adjoint(
    surface: Surface,
    observation_points: torch.Tensor,
    electric_cotangent: torch.Tensor,
    magnetic_cotangent: torch.Tensor,
    *,
    mode: str = "near",
    eps_r: float | torch.Tensor = 1.0,
    mu_r: float | torch.Tensor = 1.0,
    source_chunk_size: int = 256,
    observation_chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the field-space adjoint of a fixed-geometry projection.

    Args:
        surface: Sampled surface; only its field samples are differentiated.
        observation_points: Near-field points in meters for ``mode="near"``
            or unit directions for ``mode="far"``.
        electric_cotangent: Complex output cotangent with shape ``(F,M,3)``.
        magnetic_cotangent: Complex output cotangent with shape ``(F,M,3)``.
        mode: ``"near"`` or ``"far"`` projection.
        eps_r: Homogeneous lossless relative permittivity.
        mu_r: Homogeneous lossless relative permeability.
        source_chunk_size: Source block size for the selected projection.
        observation_chunk_size: Observation block size.

    Returns:
        Complex cotangents for the sampled electric and magnetic fields, each
        with the corresponding input shape ``(F,P,3)``.

    Notes:
        This is a matrix-free vector-Jacobian product. Geometry, quadrature,
        directions/points, and material parameters are treated as fixed; the
        routine recomputes the forward projection and asks PyTorch autograd for
        its VJP, so it does not form or retain a dense projection matrix.
    """
    if mode not in ("near", "far"):
        raise ValueError("mode must be 'near' or 'far'")
    if not electric_cotangent.is_complex() or not magnetic_cotangent.is_complex():
        raise TypeError("electric and magnetic cotangents must be complex")
    if electric_cotangent.shape != magnetic_cotangent.shape:
        raise ValueError("electric and magnetic cotangents must have matching shapes")
    if (
        electric_cotangent.device != surface.electric_fields.device
        or magnetic_cotangent.device != surface.electric_fields.device
    ):
        raise ValueError("cotangents and surface must share a device")
    if (
        electric_cotangent.dtype != surface.electric_fields.dtype
        or magnetic_cotangent.dtype != surface.magnetic_fields.dtype
    ):
        raise TypeError("cotangents must match surface field dtypes")

    electric_input = surface.electric_fields.detach().clone().requires_grad_(True)
    magnetic_input = surface.magnetic_fields.detach().clone().requires_grad_(True)
    fixed_surface = Surface(
        surface.positions_m.detach(),
        surface.normals.detach(),
        surface.weights_m2.detach(),
        electric_input,
        magnetic_input,
        surface.frequencies_hz.detach(),
    )
    with torch.enable_grad():
        if mode == "near":
            projected_e, projected_h = project_nearfield(
                fixed_surface,
                observation_points.detach(),
                eps_r=eps_r,
                mu_r=mu_r,
                source_chunk_size=source_chunk_size,
                observation_chunk_size=observation_chunk_size,
            )
        else:
            projected_e, projected_h = project_farfield(
                fixed_surface,
                observation_points.detach(),
                eps_r=eps_r,
                mu_r=mu_r,
                source_chunk_size=source_chunk_size,
                observation_chunk_size=observation_chunk_size,
            )
        if (
            projected_e.shape != electric_cotangent.shape
            or projected_h.shape != magnetic_cotangent.shape
        ):
            raise ValueError(
                "cotangents must have shape matching projected fields (F,M,3)"
            )
        grad_e, grad_h = torch.autograd.grad(
            (projected_e, projected_h),
            (electric_input, magnetic_input),
            grad_outputs=(electric_cotangent, magnetic_cotangent),
        )
    return grad_e, grad_h
