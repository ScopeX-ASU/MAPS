"""Gradient checks for native separable and FFT near-field paths."""

# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")

from core.nearfield import make_planar_surface
from core.nearfield.accelerated import (
    angular_spectrum_torch,
    project_farfield_separable,
)

pytestmark = pytest.mark.gradient


def test_separable_farfield_complex_gradient_matches_finite_difference():
    dtype = torch.float64
    u = torch.tensor([-0.7e-6, 0.0, 0.8e-6], dtype=dtype)
    v = torch.tensor([-0.6e-6, 0.2e-6, 0.9e-6], dtype=dtype)
    u_axis = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    v_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    origin = torch.tensor([0.1e-6, 0.2e-6, -0.3e-6], dtype=dtype)
    frequency = torch.tensor([1.8e14], dtype=dtype)
    generator = torch.Generator().manual_seed(31)
    electric = torch.randn(
        (1, 3, 3, 3), dtype=torch.complex128, generator=generator, requires_grad=True
    )
    magnetic = torch.randn(
        (1, 3, 3, 3), dtype=torch.complex128, generator=generator, requires_grad=True
    )
    u_dir = torch.tensor([-0.1, 0.18], dtype=dtype)
    v_dir = torch.tensor([-0.14, 0.08], dtype=dtype)

    def outputs(e, h):
        surface = make_planar_surface(origin, u_axis, v_axis, u, v, e, h, frequency)
        return project_farfield_separable(
            surface, u_axis, v_axis, u, v, u_dir, v_dir, origin
        )

    assert torch.autograd.gradcheck(
        outputs, (electric, magnetic), eps=1e-7, atol=2e-7, rtol=2e-5
    )


def test_fft_angular_spectrum_complex_gradient_matches_finite_difference():
    generator = torch.Generator().manual_seed(32)
    electric = torch.randn(
        (4, 5, 2), dtype=torch.complex128, generator=generator, requires_grad=True
    )

    def outputs(field):
        return angular_spectrum_torch(
            field,
            1e7,
            0.2,
            0.23,
            0.03,
            normal_axis=2,
            normal_sign=-1,
        )

    assert torch.autograd.gradcheck(
        outputs, (electric,), eps=1e-7, atol=2e-7, rtol=2e-5
    )


def test_fft_angular_spectrum_spacing_gradient_matches_finite_difference():
    dtype = torch.float64
    nu, nv = 8, 4
    spacing_u = torch.tensor(0.21, dtype=dtype, requires_grad=True)
    spacing_v = torch.tensor(0.27, dtype=dtype)
    x = torch.arange(nu, dtype=dtype) * spacing_u.detach()
    mode = torch.exp(1j * (2 * torch.pi / (nu * spacing_u.detach())) * x)
    field = torch.zeros((nv, nu, 2), dtype=torch.complex128)
    field[..., 0] = mode[None]

    def outputs(du):
        return angular_spectrum_torch(field, 2e8, du, spacing_v, 0.13)

    assert torch.autograd.gradcheck(
        outputs, (spacing_u,), eps=1e-7, atol=2e-6, rtol=2e-4
    )
