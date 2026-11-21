"""Independent coordinate, polarization, and angular integration contracts."""

import math

import pytest

torch = pytest.importorskip("torch")

from core.nearfield.observations import (
    cartesian_observation_plane,
    direction_cosines,
    integrated_directional_power,
    spherical_components,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("normal_axis", [0, 1, 2])
@pytest.mark.parametrize("normal_sign", [-1, 1])
def test_direction_cosines_preserve_transverse_order(normal_axis, normal_sign):
    """Each axis orientation has the analytic 3-4-5 direction and unit norm."""
    directions = direction_cosines(
        torch.tensor(0.3, dtype=torch.float64),
        torch.tensor([0.4, 0.0], dtype=torch.float64),
        normal_axis=normal_axis,
        normal_sign=normal_sign,
    )
    transverse = [a for a in range(3) if a != normal_axis]
    torch.testing.assert_close(
        directions[:, transverse[0]], torch.full((2,), 0.3, dtype=torch.float64)
    )
    torch.testing.assert_close(
        directions[:, transverse[1]], torch.tensor([0.4, 0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        directions[0, normal_axis],
        torch.tensor(normal_sign * math.sqrt(0.75), dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(directions, dim=-1), torch.ones(2, dtype=torch.float64)
    )


def test_direction_cosines_reject_nonpropagating_and_nonfinite_values():
    """Evanescent transverse coordinates must not produce real directions."""
    for value in (1.01, float("nan")):
        with pytest.raises(ValueError):
            direction_cosines(torch.tensor(value), torch.tensor(0.0))


@pytest.mark.parametrize("normal_axis", [0, 1, 2])
def test_cartesian_plane_ij_order_and_gradient(normal_axis):
    """The v coordinate varies fastest and physical positions retain gradients."""
    u = torch.tensor([1e-6, 3e-6], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([-2e-6, 0.0, 4e-6], dtype=torch.float64)
    position = torch.tensor(5e-6, dtype=torch.float64, requires_grad=True)
    points = cartesian_observation_plane(
        u, v, normal_axis=normal_axis, position_m=position
    )
    transverse = [a for a in range(3) if a != normal_axis]
    torch.testing.assert_close(points[:, transverse[0]], u.repeat_interleave(3))
    torch.testing.assert_close(points[:, transverse[1]], v.repeat(2))
    torch.testing.assert_close(points[:, normal_axis], position.expand(6))
    points.sum().backward()
    torch.testing.assert_close(u.grad, torch.full_like(u, 3))
    torch.testing.assert_close(position.grad, torch.tensor(6.0, dtype=torch.float64))


def test_spherical_components_known_axes_and_poles():
    """At +x and either z pole, independent basis vectors resolve exactly."""
    directions = torch.tensor([[1.0, 0, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float64)
    # Analytic theta/phi bases: +x -> (-z,+y), +z -> (+x,+y), -z -> (-x,+y).
    theta = torch.tensor([[0.0, 0, -1], [1, 0, 0], [-1, 0, 0]], dtype=torch.float64)
    phi = torch.tensor([[0.0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float64)
    vectors = (2 * directions + (3 + 2j) * theta + (-1 + 4j) * phi)[None].repeat(
        2, 1, 1
    )
    result = spherical_components(vectors, directions)
    expected = torch.tensor([2, 3 + 2j, -1 + 4j], dtype=torch.complex128).expand(
        2, 3, 3
    )
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(
        result.abs().square().sum(-1), vectors.abs().square().sum(-1)
    )


def test_spherical_components_general_angles_and_gradient():
    """Projection keeps complex field gradients through an orthonormal basis."""
    direction = torch.tensor([[0.3, 0.4, math.sqrt(0.75)]], dtype=torch.float64)
    vectors = torch.tensor(
        [[[1 + 2j, 3 - 1j, 2j]]], dtype=torch.complex128, requires_grad=True
    )
    transformed = spherical_components(vectors, direction)
    transformed.abs().square().sum().backward()
    torch.testing.assert_close(vectors.grad, 2 * vectors)


def test_integrated_isotropic_power_uses_explicit_solid_angle_and_gradients():
    """Constant unit intensity integrates to 4 pi watts on a full-sphere rule."""
    intensity = torch.ones((2, 6), dtype=torch.float64, requires_grad=True)
    weights = torch.full((6,), 4 * math.pi / 6, dtype=torch.float64)
    power = integrated_directional_power(intensity, weights)
    torch.testing.assert_close(
        power, torch.full((2,), 4 * math.pi, dtype=torch.float64)
    )
    power.sum().backward()
    torch.testing.assert_close(intensity.grad, weights.expand_as(intensity))


def test_angular_quadrature_rejects_negative_weights():
    """Negative quadrature weights are outside the positive-area contract."""
    with pytest.raises(ValueError, match="positive"):
        integrated_directional_power(torch.ones((1, 2)), torch.tensor([1.0, -1.0]))


def test_angular_quadrature_accepts_zero_jacobian_at_poles():
    """Polar endpoint weights vanish without invalidating angular integration."""
    intensity = torch.tensor([[10.0, 2.0, 10.0]], dtype=torch.float64)
    weights = torch.tensor([0.0, 4 * math.pi, 0.0], dtype=torch.float64)
    torch.testing.assert_close(
        integrated_directional_power(intensity, weights),
        torch.tensor([8 * math.pi], dtype=torch.float64),
    )
