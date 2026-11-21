"""Unit contracts for native PyTorch near-field projection."""

import math

import pytest

torch = pytest.importorskip("torch")

from core.nearfield import Surface  # noqa: E402
from core.nearfield import (
    C_0,
    EPSILON_0,
    MU_0,
    concatenate_surfaces,
    direction_vectors_spherical,
    directional_power,
    make_planar_surface,
    project_farfield,
    project_nearfield,
)

pytestmark = pytest.mark.unit


def _single_surface(dtype: torch.dtype = torch.complex128) -> Surface:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    return Surface(
        positions_m=torch.tensor([[0.1, -0.2, 0.05]], dtype=real_dtype),
        normals=torch.tensor([[0.0, 0.0, 1.0]], dtype=real_dtype),
        weights_m2=torch.tensor([0.3], dtype=real_dtype),
        electric_fields=torch.tensor(
            [
                [
                    [1.0 + 0.5j, -0.2 + 0.1j, 0.3 - 0.4j],
                    [0.1 + 0.7j, 0.6 - 0.3j, -0.4 + 0.2j],
                ]
            ],
            dtype=dtype,
        )[:, :1],
        magnetic_fields=torch.tensor(
            [[[0.2 - 0.1j, 0.3 + 0.8j, -0.5 + 0.4j]] * 2], dtype=dtype
        )[:, :1],
        frequencies_hz=torch.tensor([1.3e8], dtype=real_dtype),
    )


def test_farfield_matches_single_sample_closed_form_and_outgoing_phase():
    surface = _single_surface()
    directions = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    electric, magnetic = project_farfield(
        surface, directions, source_chunk_size=1, observation_chunk_size=1
    )

    frequency = surface.frequencies_hz[0]
    k = 2 * math.pi * frequency / C_0
    eta = math.sqrt(MU_0 / EPSILON_0)
    normal = surface.normals[0].to(torch.complex128)
    current_j = torch.linalg.cross(normal, surface.magnetic_fields[0, 0])
    current_m = -torch.linalg.cross(normal, surface.electric_fields[0, 0])
    expected = []
    for direction in directions.to(torch.complex128):
        transverse_j = current_j - direction * torch.dot(direction, current_j)
        phase = torch.exp(
            -1j * k * torch.dot(direction, surface.positions_m[0].to(torch.complex128))
        )
        expected.append(
            phase
            * 1j
            * k
            * surface.weights_m2[0]
            / (4 * math.pi)
            * (eta * transverse_j - torch.linalg.cross(direction, current_m))
        )
    expected_e = torch.stack(expected)

    assert electric.shape == magnetic.shape == (1, 2, 3)
    torch.testing.assert_close(electric[0], expected_e, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        magnetic[0],
        torch.linalg.cross(directions.to(torch.complex128), expected_e) / eta,
    )
    assert torch.allclose(
        torch.sum(electric[0] * directions, dim=-1),
        torch.zeros(2, dtype=torch.complex128),
        atol=1e-12,
    )


def test_nearfield_matches_single_source_closed_form_and_chunking():
    surface = _single_surface()
    observation = torch.tensor(
        [[0.25, 0.15, 0.4], [-0.3, 0.1, 0.2]], dtype=torch.float64
    )
    result = project_nearfield(
        surface, observation, source_chunk_size=1, observation_chunk_size=1
    )
    result_large_chunks = project_nearfield(
        surface, observation, source_chunk_size=8, observation_chunk_size=8
    )
    torch.testing.assert_close(result[0], result_large_chunks[0])
    torch.testing.assert_close(result[1], result_large_chunks[1])

    source = surface.positions_m[0]
    normal = surface.normals[0].to(torch.complex128)
    current_j = torch.linalg.cross(normal, surface.magnetic_fields[0, 0])
    current_m = -torch.linalg.cross(normal, surface.electric_fields[0, 0])
    omega = 2 * math.pi * surface.frequencies_hz[0]
    k = omega / C_0
    expected_e, expected_h = [], []
    for point in observation:
        displacement = point - source
        radius = torch.linalg.vector_norm(displacement)
        radial = displacement / radius
        green = torch.exp(1j * k * radius) / (4 * math.pi * radius)
        inv_kr = 1 / (k * radius)
        a = 1 + 1j * inv_kr - inv_kr**2
        b = -1 - 3j * inv_kr + 3 * inv_kr**2
        radial_complex = radial.to(torch.complex128)
        dyadic_j = green * (
            a * current_j + b * radial_complex * torch.dot(radial_complex, current_j)
        )
        dyadic_m = green * (
            a * current_m + b * radial_complex * torch.dot(radial_complex, current_m)
        )
        grad_green = green * (1j * k - 1 / radius) * radial
        expected_e.append(
            surface.weights_m2[0]
            * (1j * omega * MU_0 * dyadic_j - torch.linalg.cross(grad_green, current_m))
        )
        expected_h.append(
            surface.weights_m2[0]
            * (
                torch.linalg.cross(grad_green, current_j)
                + 1j * omega * EPSILON_0 * dyadic_m
            )
        )
    torch.testing.assert_close(
        result[0][0], torch.stack(expected_e), rtol=2e-12, atol=1e-12
    )
    torch.testing.assert_close(
        result[1][0], torch.stack(expected_h), rtol=2e-12, atol=1e-12
    )


def test_planar_surface_uses_tensor_product_trapezoid_weights_and_concatenates():
    coords = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    fields = torch.ones((1, 3, 3, 3), dtype=torch.complex128)
    plane = make_planar_surface(
        torch.zeros(3, dtype=torch.float64),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        coords,
        coords,
        fields,
        fields,
        torch.tensor([1e9], dtype=torch.float64),
    )
    assert plane.positions_m.shape == (9, 3)
    assert plane.weights_m2.sum().item() == pytest.approx(9.0)
    assert torch.allclose(
        plane.normals, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand(9, 3)
    )
    joined = concatenate_surfaces([plane, plane])
    assert joined.positions_m.shape == (18, 3)
    assert joined.electric_fields.shape == (1, 18, 3)


def test_spherical_direction_and_directional_power_contracts():
    directions = direction_vectors_spherical(
        torch.tensor([0.0, math.pi / 2]), torch.tensor([0.3, 0.0])
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(directions, dim=-1), torch.ones(2)
    )
    assert torch.allclose(directions[0], torch.tensor([0.0, 0.0, 1.0]))
    e_far = torch.tensor([[2.0 + 0j, 0.0, 0.0]], dtype=torch.complex128)
    intensity = directional_power(e_far)
    assert intensity.shape == (1,)
    assert intensity[0].item() == pytest.approx(4 / (2 * math.sqrt(MU_0 / EPSILON_0)))


@pytest.mark.parametrize("bad_shape", [(3,), (2, 2)])
def test_observation_shape_is_validated(bad_shape):
    with pytest.raises(ValueError, match="shape"):
        project_nearfield(
            _single_surface(), torch.zeros(bad_shape, dtype=torch.float64)
        )


def test_projection_rejects_observation_on_source_sample():
    surface = _single_surface()
    with pytest.raises(ValueError, match="coincide"):
        project_nearfield(surface, surface.positions_m[:1])


def test_complex_material_is_rejected_without_discarding_absorption():
    """Lossless-medium APIs must not silently cast away an imaginary part."""
    medium = torch.tensor(2.0 + 0.1j)
    with pytest.raises(ValueError, match="real lossless"):
        project_nearfield(
            _single_surface(),
            torch.tensor([[0.25, 0.15, 0.4]], dtype=torch.float64),
            eps_r=medium,
        )
    with pytest.raises(ValueError, match="real lossless"):
        directional_power(torch.ones((1, 3), dtype=torch.complex128), eps_r=medium)
