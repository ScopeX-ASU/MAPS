"""Separable Fourier and angular-spectrum operations for planar fields.

All operations use SI units and the ``exp(-i omega t)`` phasor convention
documented by :mod:`core.nearfield.projector`.
"""

from __future__ import annotations

import math

import torch

from .projector import EPSILON_0, MU_0, Surface


def project_farfield_separable(
    surface: Surface,
    u_axis: torch.Tensor,
    v_axis: torch.Tensor,
    u_coordinates_m: torch.Tensor,
    v_coordinates_m: torch.Tensor,
    u_directions: torch.Tensor,
    v_directions: torch.Tensor,
    origin_m: torch.Tensor,
    *,
    normal_sign: int = 1,
    eps_r: float | torch.Tensor = 1.0,
    mu_r: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a tensor-product planar surface using two Fourier matrix products.

    Args:
        surface: Planar field samples in v-major/u-minor order.
        u_axis: Unit first in-plane axis, shape ``(3,)``.
        v_axis: Unit second in-plane axis, shape ``(3,)`` and orthogonal to u.
        u_coordinates_m: Strictly increasing source coordinates along u,
            shape ``(Nu,)``.
        v_coordinates_m: Strictly increasing source coordinates along v,
            shape ``(Nv,)``.
        u_directions: First direction-cosine samples, shape ``(Ku,)``.
        v_directions: Second direction-cosine samples, shape ``(Kv,)``.
        origin_m: Position corresponding to source coordinate ``(0, 0)``.
        normal_sign: Surface normal and outgoing hemisphere sign, either -1 or 1.
        eps_r: Positive relative permittivity of the homogeneous medium.
        mu_r: Positive relative permeability of the homogeneous medium.

    Returns:
        Electric and magnetic far-field amplitudes with shape ``(F,Kv,Ku,3)``.
        Both are flattened only conceptually: v-direction varies before
        u-direction, matching ``make_planar_surface``'s source ordering.

    Raises:
        ValueError: If the supplied grid does not describe ``surface`` or any
            requested direction cosine lies outside the propagating disk.

    Notes:
        The source Fourier transform is evaluated as a pair of matrix
        multiplications. Surface quadrature weights are used as supplied.
        This routine requires an orthonormal planar coordinate frame.
    """
    tensors = (
        u_axis,
        v_axis,
        u_coordinates_m,
        v_coordinates_m,
        u_directions,
        v_directions,
        origin_m,
    )
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError(
            "planar axes, coordinates, directions, and origin must be tensors"
        )
    if normal_sign not in (-1, 1):
        raise ValueError("normal_sign must be -1 or +1")
    if any(isinstance(value, complex) for value in (eps_r, mu_r)) or any(
        isinstance(value, torch.Tensor) and value.is_complex()
        for value in (eps_r, mu_r)
    ):
        raise TypeError("eps_r and mu_r must be real")
    dtype, device = surface.positions_m.dtype, surface.positions_m.device
    if any(t.device != device or t.dtype != dtype or t.is_complex() for t in tensors):
        raise TypeError(
            "planar geometry and direction tensors must match the Surface real dtype and device"
        )
    if u_axis.shape != (3,) or v_axis.shape != (3,) or origin_m.shape != (3,):
        raise ValueError("u_axis, v_axis, and origin_m must have shape (3,)")
    if u_coordinates_m.ndim != 1 or v_coordinates_m.ndim != 1:
        raise ValueError("source coordinate vectors must be one-dimensional")
    if u_directions.ndim != 1 or v_directions.ndim != 1:
        raise ValueError("direction cosine vectors must be one-dimensional")
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("planar arguments must be finite")
    if (u_coordinates_m[1:] <= u_coordinates_m[:-1]).any() or (
        v_coordinates_m[1:] <= v_coordinates_m[:-1]
    ).any():
        raise ValueError("source coordinate vectors must be strictly increasing")
    if u_coordinates_m.numel() < 2 or v_coordinates_m.numel() < 2:
        raise ValueError("each planar source axis must contain at least two samples")
    for values, name in (
        (u_directions, "u_directions"),
        (v_directions, "v_directions"),
    ):
        if values.numel() == 0:
            raise ValueError(f"{name} must be nonempty")
    u_norm = torch.linalg.vector_norm(u_axis)
    v_norm = torch.linalg.vector_norm(v_axis)
    dot_uv = torch.dot(u_axis, v_axis)
    if (
        not torch.allclose(u_norm, u_norm.new_ones(()), atol=2e-6, rtol=2e-6)
        or not torch.allclose(v_norm, v_norm.new_ones(()), atol=2e-6, rtol=2e-6)
        or abs(dot_uv) > 2e-6
    ):
        raise ValueError("u_axis and v_axis must be orthonormal")
    nu, nv = u_coordinates_m.numel(), v_coordinates_m.numel()
    points = surface.positions_m.reshape(nv, nu, 3)
    expected = (
        origin_m
        + v_coordinates_m[:, None, None] * v_axis
        + u_coordinates_m[None, :, None] * u_axis
    )
    scale = torch.maximum(
        surface.positions_m.abs().max(), origin_m.abs().max()
    ).clamp_min(1e-12)
    atol = 64 * torch.finfo(dtype).eps * scale
    if not torch.allclose(
        points, expected, atol=float(atol.detach()), rtol=64 * torch.finfo(dtype).eps
    ):
        raise ValueError(
            "Surface positions do not match the explicit v-major/u-minor planar grid"
        )
    normal = normal_sign * torch.linalg.cross(u_axis, v_axis)
    if not torch.allclose(
        surface.normals, normal.expand_as(surface.normals), atol=2e-5, rtol=2e-5
    ):
        raise ValueError(
            "Surface normals do not match normal_sign * (u_axis cross v_axis)"
        )
    U, V = torch.meshgrid(u_directions, v_directions, indexing="xy")
    transverse2 = U.square() + V.square()
    if (transverse2 > 1).any():
        raise ValueError(
            "direction cosine grid must lie inside the propagating unit disk"
        )
    w = torch.sqrt(1 - transverse2)
    directions = (
        U[..., None] * u_axis
        + V[..., None] * v_axis
        + (normal_sign * w)[..., None] * torch.linalg.cross(u_axis, v_axis)
    )

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
    omega = 2 * math.pi * surface.frequencies_hz
    epsilon = EPSILON_0 * eps
    permeability = MU_0 * mu
    k_all = omega * torch.sqrt(epsilon * permeability)
    eta = torch.sqrt(permeability / epsilon)
    field_complex = surface.electric_fields.dtype
    normals = surface.normals.to(field_complex)
    j = torch.linalg.cross(normals[None], surface.magnetic_fields, dim=-1).reshape(
        -1, nv, nu, 3
    )
    m = -torch.linalg.cross(normals[None], surface.electric_fields, dim=-1).reshape(
        -1, nv, nu, 3
    )
    weighted_j = j * surface.weights_m2.reshape(nv, nu)[None, :, :, None]
    weighted_m = m * surface.weights_m2.reshape(nv, nu)[None, :, :, None]
    electric_f, magnetic_f = [], []
    for fi, k in enumerate(k_all):
        phase_u = torch.exp(-1j * k * u_directions[:, None] * u_coordinates_m[None, :])
        phase_v = torch.exp(-1j * k * v_directions[:, None] * v_coordinates_m[None, :])
        phase_origin = torch.exp(
            -1j * k * torch.einsum("...c,c->...", directions, origin_m)
        )
        # The first product transforms the v axis, and the second transforms u.
        j_int = torch.einsum(
            "bi,fiuc,au->fbac", phase_v, weighted_j[fi : fi + 1], phase_u
        )
        m_int = torch.einsum(
            "bi,fiuc,au->fbac", phase_v, weighted_m[fi : fi + 1], phase_u
        )
        j_int = j_int[0] * phase_origin[..., None]
        m_int = m_int[0] * phase_origin[..., None]
        d = directions.to(field_complex)
        projected_j = j_int - d * torch.sum(d * j_int, dim=-1, keepdim=True)
        e = (
            1j
            * k
            * (eta * projected_j - torch.linalg.cross(d, m_int, dim=-1))
            / (4 * math.pi)
        )
        h = torch.linalg.cross(d, e, dim=-1) / eta
        electric_f.append(e)
        magnetic_f.append(h)
    return torch.stack(electric_f), torch.stack(magnetic_f)


def angular_spectrum_torch(
    electric_tangential: torch.Tensor,
    frequency_hz: float | torch.Tensor,
    spacing_u_m: float | torch.Tensor,
    spacing_v_m: float | torch.Tensor,
    distance_m: float | torch.Tensor,
    *,
    normal_axis: int = 2,
    normal_sign: int = 1,
    eps_r: float | torch.Tensor = 1.0,
    mu_r: float | torch.Tensor = 1.0,
    evanescent: str = "decay",
    grazing_tolerance: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate a one-sided sampled tangential E field with an FFT spectrum.

    Args:
        electric_tangential: Complex tangential E samples shaped ``(Nv,Nu,2)``
            or ``(F,Nv,Nu,2)``. Components follow the two transverse Cartesian
            axes in increasing axis order; ``Nu`` indexes the first axis.
        frequency_hz: Positive scalar frequency or one frequency per leading F.
        spacing_u_m: Positive sample spacing on the first transverse axis.
        spacing_v_m: Positive sample spacing on the second transverse axis.
        distance_m: Nonnegative distance along the selected ``normal_sign``.
        normal_axis: Cartesian normal axis, 0, 1, or 2.
        normal_sign: Propagation hemisphere, -1 or +1.
        eps_r: Positive relative permittivity.
        mu_r: Positive relative permeability.
        evanescent: ``"decay"`` keeps evanescent FFT modes with exponential
            decay; ``"discard"`` zeros those modes.
        grazing_tolerance: Relative cutoff for modes where longitudinal wave
            number approaches zero and transversality reconstruction is singular.

    Returns:
        Complex ``(Nv,Nu,3)`` or ``(F,Nv,Nu,3)`` propagated electric and
        magnetic fields at the requested plane.

    Raises:
        ValueError: If dimensions, frequency, sampling, propagation direction,
            or grazing modes are invalid.

    Notes:
        FFT frequencies use the unshifted PyTorch ordering and are bounded by
        the sample Nyquist frequencies ``pi/spacing``. Modes outside that
        sampling passband are not represented by the sampled data. Each input
        frequency represents a one-sided wave traveling along ``normal_sign``;
        ``H=k cross E/(omega mu)``. For evanescent modes the positive-imaginary
        square-root branch is used and ``distance_m`` is unsigned, so decay is
        stable for either normal sign.
    """
    if (
        not isinstance(electric_tangential, torch.Tensor)
        or not electric_tangential.is_complex()
    ):
        raise TypeError("electric_tangential must be a complex tensor")
    if electric_tangential.ndim not in (3, 4) or electric_tangential.shape[-1] != 2:
        raise ValueError("electric_tangential must have shape (Nv,Nu,2) or (F,Nv,Nu,2)")
    if normal_axis not in (0, 1, 2) or normal_sign not in (-1, 1):
        raise ValueError("normal_axis must be 0..2 and normal_sign must be -1 or +1")
    if evanescent not in ("decay", "discard"):
        raise ValueError("evanescent must be 'decay' or 'discard'")
    if any(
        isinstance(value, complex)
        for value in (frequency_hz, spacing_u_m, spacing_v_m, distance_m, eps_r, mu_r)
    ) or any(
        isinstance(value, torch.Tensor) and value.is_complex()
        for value in (frequency_hz, spacing_u_m, spacing_v_m, distance_m, eps_r, mu_r)
    ):
        raise TypeError(
            "frequency, sampling, distance, and medium parameters must be real"
        )
    if not math.isfinite(grazing_tolerance) or grazing_tolerance <= 0:
        raise ValueError("grazing_tolerance must be finite and positive")
    if electric_tangential.shape[-3] == 0 or electric_tangential.shape[-2] == 0:
        raise ValueError("FFT plane dimensions must be nonempty")
    if not torch.isfinite(electric_tangential).all():
        raise ValueError("electric_tangential must be finite")
    real_dtype = electric_tangential.real.dtype
    device = electric_tangential.device

    def as_real(value, name):
        """Convert a named scalar to the field real dtype and device."""
        return torch.as_tensor(value, dtype=real_dtype, device=device)

    freq = as_real(frequency_hz, "frequency_hz").reshape(-1)
    spacing_u = as_real(spacing_u_m, "spacing_u_m")
    spacing_v = as_real(spacing_v_m, "spacing_v_m")
    distance = as_real(distance_m, "distance_m")
    eps = as_real(eps_r, "eps_r")
    mu = as_real(mu_r, "mu_r")
    if any(t.numel() != 1 for t in (spacing_u, spacing_v, distance, eps, mu)):
        raise ValueError("spacings, distance, eps_r, and mu_r must be scalars")
    if freq.numel() == 0 or not torch.isfinite(freq).all() or (freq <= 0).any():
        raise ValueError("frequency_hz must contain positive finite values")
    if (
        not torch.isfinite(spacing_u).all()
        or not torch.isfinite(spacing_v).all()
        or spacing_u <= 0
        or spacing_v <= 0
    ):
        raise ValueError("sample spacings must be finite and positive")
    if not torch.isfinite(distance).all() or distance < 0:
        raise ValueError("distance_m must be finite and nonnegative")
    if (
        not torch.isfinite(eps).all()
        or not torch.isfinite(mu).all()
        or eps <= 0
        or mu <= 0
    ):
        raise ValueError("eps_r and mu_r must be finite and positive")
    batched = electric_tangential.ndim == 4
    fields = electric_tangential if batched else electric_tangential.unsqueeze(0)
    if freq.numel() not in (1, fields.shape[0]):
        raise ValueError(
            "frequency_hz must be scalar or have one value per field batch"
        )
    if freq.numel() == 1 and fields.shape[0] > 1:
        freq = freq.expand(fields.shape[0])
    if fields.shape[0] != freq.numel():
        raise ValueError("field batch and frequency count differ")
    nu, nv = fields.shape[-2], fields.shape[-3]
    transverse = [axis for axis in range(3) if axis != normal_axis]
    ku = (
        2
        * math.pi
        * torch.fft.fftfreq(nu, d=1.0, device=device, dtype=real_dtype)
        / spacing_u
    )
    kv = (
        2
        * math.pi
        * torch.fft.fftfreq(nv, d=1.0, device=device, dtype=real_dtype)
        / spacing_v
    )
    k_u, k_v = torch.meshgrid(ku, kv, indexing="xy")
    k_t2 = k_u.square() + k_v.square()
    omega = 2 * math.pi * freq
    k = omega * torch.sqrt(EPSILON_0 * eps * MU_0 * mu)
    # complex sqrt selects nonnegative real and imaginary parts for real input.
    kz = torch.sqrt(
        (k[:, None, None].square() - k_t2[None]).to(
            torch.complex64 if real_dtype == torch.float32 else torch.complex128
        )
    )
    near_grazing = torch.abs(kz) <= (grazing_tolerance * k[:, None, None])
    spectra = torch.fft.fft2(fields.movedim(-1, -3), dim=(-2, -1))
    if torch.any(near_grazing & (torch.abs(spectra).amax(dim=1) > 0)):
        raise ValueError(
            "nonzero grazing spectral modes cannot be reconstructed from tangential E"
        )
    # Remove numerically zero grazing bins before the division.
    signed_kz = normal_sign * kz
    denom = torch.where(near_grazing, torch.ones_like(signed_kz), signed_kz)
    e_spec_components = [None, None, None]
    e_spec_components[transverse[0]] = spectra[:, 0]
    e_spec_components[transverse[1]] = spectra[:, 1]
    e_normal = -(k_u[None] * spectra[:, 0] + k_v[None] * spectra[:, 1]) / denom
    e_spec_components[normal_axis] = e_normal
    e_spec = torch.stack(e_spec_components, dim=-1)
    wave_components = [None, None, None]
    wave_components[transverse[0]] = k_u.to(kz.dtype).expand_as(kz)
    wave_components[transverse[1]] = k_v.to(kz.dtype).expand_as(kz)
    wave_components[normal_axis] = signed_kz
    wave = torch.stack(wave_components, dim=-1)
    h_spec = torch.linalg.cross(wave, e_spec, dim=-1) / (
        omega[:, None, None, None] * MU_0 * mu
    )
    phase = torch.exp(1j * kz * distance)
    if evanescent == "discard":
        phase = torch.where(
            (k_t2[None] <= k[:, None, None].square()), phase, torch.zeros_like(phase)
        )
    phase = torch.where(near_grazing, torch.zeros_like(phase), phase)
    e_out = torch.fft.ifft2(e_spec * phase[..., None], dim=(-3, -2))
    h_out = torch.fft.ifft2(h_spec * phase[..., None], dim=(-3, -2))
    if not batched:
        return e_out[0], h_out[0]
    return e_out, h_out
