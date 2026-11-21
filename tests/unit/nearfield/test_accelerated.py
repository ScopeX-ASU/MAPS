"""Contracts for planar Fourier and FFT near-field accelerators."""

# ruff: noqa: E402

import math

import pytest

torch = pytest.importorskip("torch")

from core.nearfield import Surface, make_planar_surface, project_farfield
from core.nearfield.accelerated import (
    angular_spectrum_torch,
    project_farfield_separable,
)
from core.nearfield.projector import C_0

pytestmark = pytest.mark.unit


def test_separable_farfield_matches_direct_with_varying_quadrature():
    dtype = torch.float64
    coords_u = torch.tensor([-0.5, -0.1, 0.4, 0.9], dtype=dtype) * 1e-6
    coords_v = torch.tensor([-0.6, 0.0, 0.5], dtype=dtype) * 1e-6
    u_axis = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    v_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    origin = torch.tensor([0.3e-6, -0.2e-6, 0.7e-6], dtype=dtype)
    generator = torch.Generator().manual_seed(43)
    e = torch.randn((2, 3, 4, 3), generator=generator, dtype=dtype) + 1j * torch.randn(
        (2, 3, 4, 3), generator=generator, dtype=dtype
    )
    h = torch.randn((2, 3, 4, 3), generator=generator, dtype=dtype) + 1j * torch.randn(
        (2, 3, 4, 3), generator=generator, dtype=dtype
    )
    frequencies = torch.tensor([2.1e14, 2.25e14], dtype=dtype)
    surface = make_planar_surface(
        origin, u_axis, v_axis, coords_u, coords_v, e, h, frequencies
    )
    # Keep all corners within the unit disk while sampling both transverse axes.
    u_directions = torch.tensor([-0.18, 0.0, 0.23], dtype=dtype)
    v_directions = torch.tensor([-0.12, 0.16], dtype=dtype)
    fast = project_farfield_separable(
        surface,
        u_axis,
        v_axis,
        coords_u,
        coords_v,
        u_directions,
        v_directions,
        origin,
    )
    u_grid, v_grid = torch.meshgrid(u_directions, v_directions, indexing="xy")
    directions = torch.stack(
        (u_grid, v_grid, torch.sqrt(1 - u_grid.square() - v_grid.square())), dim=-1
    ).reshape(-1, 3)
    reference = project_farfield(surface, directions)
    torch.testing.assert_close(
        fast[0], reference[0].reshape(2, 2, 3, 3), rtol=2e-11, atol=2e-11
    )
    torch.testing.assert_close(
        fast[1], reference[1].reshape(2, 2, 3, 3), rtol=2e-11, atol=2e-11
    )


def test_separable_farfield_differentiates_fields_and_direction_cosines():
    dtype = torch.float64
    u = torch.tensor([-1e-6, 0.0, 1e-6], dtype=dtype)
    v = torch.tensor([-0.8e-6, 0.2e-6, 1.1e-6], dtype=dtype)
    ua = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    va = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    origin = torch.zeros(3, dtype=dtype)
    e = torch.randn(
        (1, 3, 3, 3), dtype=torch.complex128, generator=torch.Generator().manual_seed(2)
    ).requires_grad_()
    h = torch.randn(
        (1, 3, 3, 3), dtype=torch.complex128, generator=torch.Generator().manual_seed(3)
    ).requires_grad_()
    freq = torch.tensor([1.7e14], dtype=dtype)
    surface = make_planar_surface(origin, ua, va, u, v, e, h, freq)
    ud = torch.tensor([-0.1, 0.2], dtype=dtype, requires_grad=True)
    vd = torch.tensor([-0.12, 0.11], dtype=dtype, requires_grad=True)

    def objective(electric, magnetic, u_values, v_values):
        surf = Surface(
            surface.positions_m,
            surface.normals,
            surface.weights_m2,
            electric.reshape(1, -1, 3),
            magnetic.reshape(1, -1, 3),
            freq,
        )
        out_e, out_h = project_farfield_separable(
            surf, ua, va, u, v, u_values, v_values, origin
        )
        return out_e.abs().square().sum() + 0.3 * out_h.abs().square().sum()

    loss = objective(e, h, ud, vd)
    grads = torch.autograd.grad(loss, (e, h, ud, vd))
    assert all(
        torch.isfinite(grad).all() and torch.linalg.vector_norm(grad) > 0
        for grad in grads
    )


def _plane_wave_grid(normal_axis: int):
    dtype = torch.float64
    nu, nv = 32, 24
    du, dv = 0.19, 0.23
    f = 1.1e9
    omega = 2 * math.pi * f
    k = omega / C_0
    # Exact FFT bins give a single Fourier mode without interpolation error.
    ix, iy = 2, -1
    ku = 2 * math.pi * ix / (nu * du)
    kv = 2 * math.pi * iy / (nv * dv)
    kz = math.sqrt(k * k - ku * ku - kv * kv)
    x = torch.arange(nu, dtype=dtype) * du
    y = torch.arange(nv, dtype=dtype) * dv
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    transverse_axes = [axis for axis in range(3) if axis != normal_axis]
    polarization = torch.tensor([-kv, ku], dtype=dtype)
    polarization = polarization / torch.linalg.vector_norm(polarization)
    field = torch.exp(1j * (ku * xx + kv * yy))[..., None] * polarization.to(
        torch.complex128
    )
    electric_vector = torch.zeros(3, dtype=dtype)
    wave_transverse = torch.zeros(3, dtype=dtype)
    electric_vector[transverse_axes] = polarization
    wave_transverse[transverse_axes[0]] = ku
    wave_transverse[transverse_axes[1]] = kv
    return (
        field,
        f,
        du,
        dv,
        omega,
        k,
        ku,
        kv,
        kz,
        polarization,
        electric_vector,
        wave_transverse,
    )


@pytest.mark.parametrize("normal_axis", [0, 1, 2])
@pytest.mark.parametrize("normal_sign", [-1, 1])
def test_angular_spectrum_plane_wave_phase_polarization_and_maxwell_h(
    normal_axis, normal_sign
):
    (
        field,
        f,
        du,
        dv,
        omega,
        k,
        ku,
        kv,
        kz,
        polarization,
        electric_vector,
        wave_transverse,
    ) = _plane_wave_grid(normal_axis)
    distance = 0.37
    e_out, h_out = angular_spectrum_torch(
        field,
        f,
        du,
        dv,
        distance,
        normal_axis=normal_axis,
        normal_sign=normal_sign,
    )
    phase = complex(math.cos(kz * distance), math.sin(kz * distance))
    wave = wave_transverse.clone()
    wave[normal_axis] = normal_sign * kz
    x = torch.arange(field.shape[1], dtype=torch.float64) * du
    y = torch.arange(field.shape[0], dtype=torch.float64) * dv
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    expected_e = (
        torch.exp(1j * (ku * xx + kv * yy))[..., None]
        * electric_vector.to(torch.complex128)
        * phase
    )
    expected_h_vec = torch.linalg.cross(wave, electric_vector) / (
        omega * 1.25663706212e-6
    )
    spatial = torch.exp(1j * (ku * xx + kv * yy))[..., None]
    expected_h = spatial * expected_h_vec.to(torch.complex128) * phase
    torch.testing.assert_close(e_out, expected_e, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(h_out, expected_h, rtol=2e-11, atol=2e-11)
    assert torch.max(torch.abs(torch.sum(e_out * wave, dim=-1))) < 1e-10


@pytest.mark.parametrize("normal_sign", [-1, 1])
def test_angular_spectrum_evanescent_modes_decay_for_both_directions(normal_sign):
    dtype = torch.float64
    nu, nv = 32, 8
    du = dv = 0.1
    f = 1e8
    k = 2 * math.pi * f / C_0
    ix = 4
    ku = 2 * math.pi * ix / (nu * du)
    alpha = math.sqrt(ku * ku - k * k)
    x = torch.arange(nu, dtype=dtype) * du
    phase_x = torch.exp(1j * ku * x)
    field = torch.zeros((nv, nu, 2), dtype=torch.complex128)
    field[..., 0] = phase_x[None, :]
    distance = 0.06
    propagated, _ = angular_spectrum_torch(
        field,
        f,
        du,
        dv,
        distance,
        normal_sign=normal_sign,
    )
    expected = math.exp(-alpha * distance)
    assert torch.max(torch.abs(propagated[..., 0])).item() == pytest.approx(
        expected, rel=2e-10
    )
    discarded, _ = angular_spectrum_torch(
        field,
        f,
        du,
        dv,
        distance,
        normal_sign=normal_sign,
        evanescent="discard",
    )
    assert torch.max(torch.abs(discarded)) < 1e-12


def test_angular_spectrum_batches_frequencies_and_preserves_eh_shapes():
    field, _, du, dv, _, _, _, _, _, _, _, _ = _plane_wave_grid(normal_axis=2)
    fields = torch.stack((field, field * (0.7 - 0.2j)))
    frequencies = torch.tensor([1.1e9, 1.3e9], dtype=torch.float64)
    electric, magnetic = angular_spectrum_torch(fields, frequencies, du, dv, 0.2)
    assert electric.shape == magnetic.shape == (2, *field.shape[:-1], 3)
    assert torch.isfinite(electric).all() and torch.isfinite(magnetic).all()


def test_angular_spectrum_rejects_nonzero_grazing_mode():
    nu, nv = 8, 4
    du = dv = 0.5
    ku = 2 * math.pi / (nu * du)
    frequency = C_0 * ku / (2 * math.pi)
    x = torch.arange(nu, dtype=torch.float64) * du
    field = torch.zeros((nv, nu, 2), dtype=torch.complex128)
    field[..., 0] = torch.exp(1j * ku * x)[None, :]
    with pytest.raises(ValueError, match="grazing"):
        angular_spectrum_torch(field, frequency, du, dv, 0.1)


def test_accelerated_paths_reject_complex_material_parameters():
    with pytest.raises(TypeError, match="real"):
        angular_spectrum_torch(
            torch.ones((2, 2, 2), dtype=torch.complex128),
            1e9,
            0.1,
            0.1,
            0.0,
            eps_r=1 + 0.1j,
        )
    dtype = torch.float64
    coordinates = torch.tensor([-1e-6, 1e-6], dtype=dtype)
    axis_u = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    axis_v = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    field = torch.ones((1, 2, 2, 3), dtype=torch.complex128)
    surface = make_planar_surface(
        torch.zeros(3, dtype=dtype),
        axis_u,
        axis_v,
        coordinates,
        coordinates,
        field,
        field,
        torch.tensor([1e9], dtype=dtype),
    )
    with pytest.raises(TypeError, match="real"):
        project_farfield_separable(
            surface,
            axis_u,
            axis_v,
            coordinates,
            coordinates,
            torch.tensor([0.0], dtype=dtype),
            torch.tensor([0.0], dtype=dtype),
            torch.zeros(3, dtype=dtype),
            mu_r=torch.tensor(1 + 0j),
        )


@pytest.mark.slow
def test_angular_spectrum_propagates_a_paraxial_gaussian_beam():
    dtype = torch.float64
    wavelength = 1.0
    waist = 8 * wavelength
    nu = nv = 256
    spacing = 11.7 * waist / nu
    z_rayleigh = math.pi * waist**2 / wavelength
    x = (torch.arange(nu, dtype=dtype) - (nu - 1) / 2) * spacing
    y = (torch.arange(nv, dtype=dtype) - (nv - 1) / 2) * spacing
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    input_e = torch.exp(-(xx.square() + yy.square()) / waist**2).to(torch.complex128)
    field = torch.zeros((nv, nu, 2), dtype=torch.complex128)
    field[..., 0] = input_e
    f = C_0 / wavelength
    result, _ = angular_spectrum_torch(field, f, spacing, spacing, z_rayleigh)
    beam_radius = math.sqrt(2) * waist
    radius_of_curvature = 2 * z_rayleigh
    gouy = math.pi / 4
    expected = (waist / beam_radius) * torch.exp(
        -(xx.square() + yy.square()) / beam_radius**2
        + 1j
        * (
            2 * math.pi / wavelength * z_rayleigh
            + 2
            * math.pi
            / wavelength
            * (xx.square() + yy.square())
            / (2 * radius_of_curvature)
            - gouy
        )
    )
    mask = torch.abs(expected) > 0.3
    relative = torch.abs(result[..., 0][mask] - expected[mask]) / torch.abs(
        expected[mask]
    )
    assert relative.max() < 0.01
