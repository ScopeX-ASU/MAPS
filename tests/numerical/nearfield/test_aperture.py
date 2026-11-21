"""Vector rectangular-aperture diffraction with an independent sinc integral."""

import math

import pytest

torch = pytest.importorskip("torch")

from core.nearfield import C_0, EPSILON_0, MU_0, make_planar_surface, project_farfield

pytestmark = pytest.mark.numerical


def test_uniform_aperture_complex_vector_pattern_and_phase():
    """A uniformly illuminated rectangle has the analytic vector sinc pattern."""
    dtype = torch.float64
    eta = math.sqrt(MU_0 / EPSILON_0)
    frequency = 1.2e8
    width, height = 0.73, 0.61
    origin = torch.tensor([0.11, -0.17, 0.23], dtype=dtype)
    u = torch.linspace(-width / 2, width / 2, 65, dtype=dtype)
    v = torch.linspace(-height / 2, height / 2, 65, dtype=dtype)
    electric = torch.zeros((1, 65, 65, 3), dtype=torch.complex128)
    magnetic = torch.zeros_like(electric)
    electric[..., 0] = 1 + 0.3j
    magnetic[..., 1] = (1 + 0.3j) / eta
    surface = make_planar_surface(
        origin,
        torch.tensor([1.0, 0.0, 0.0], dtype=dtype),
        torch.tensor([0.0, 1.0, 0.0], dtype=dtype),
        u,
        v,
        electric,
        magnetic,
        torch.tensor([frequency], dtype=dtype),
    )
    sx = torch.linspace(-0.71, 0.71, 17, dtype=dtype)
    sy = torch.full_like(sx, 0.23)
    directions = torch.stack((sx, sy, (1 - sx.square() - sy.square()).sqrt()), dim=-1)
    actual_e, actual_h = project_farfield(surface, directions)
    k = 2 * math.pi * frequency / C_0
    integral = (
        width
        * height
        * torch.sinc(k * directions[:, 0] * width / (2 * math.pi))
        * torch.sinc(k * directions[:, 1] * height / (2 * math.pi))
    )
    integral = integral * torch.exp(-1j * k * (directions @ origin)) * (1 + 0.3j)
    s = directions.to(torch.complex128)
    # For n=+z, unit Ex illumination gives J=-x/eta and M=-y.
    transverse_j_eta = (
        -torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128) + s * s[:, :1]
    )
    minus_cross_m = torch.linalg.cross(
        s, torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex128).expand_as(s)
    )
    expected_e = (1j * k * integral / (4 * math.pi))[:, None] * (
        transverse_j_eta + minus_cross_m
    )
    expected_h = torch.linalg.cross(s, expected_e) / eta
    for actual, expected in ((actual_e[0], expected_e), (actual_h[0], expected_h)):
        error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
            expected
        )
        assert error < 1e-4, error.item()
