"""Maxwell dipole fields provide independent surface projection baselines."""

import math

import pytest

torch = pytest.importorskip("torch")

from core.nearfield import Surface  # noqa: E402
from core.nearfield import (
    EPSILON_0,
    MU_0,
    directional_power,
    project_farfield,
    project_nearfield,
)

pytestmark = pytest.mark.numerical


def _closed_box(
    dipole: str, nodes_per_edge: int, frequency_hz: float = 8.0e8
) -> tuple[Surface, torch.Tensor, torch.Tensor]:
    """Sample closed-form Maxwell fields on cell-centered box patches."""
    dtype = torch.float64
    complex_dtype = torch.complex128
    half_width = 0.35
    spacing = 2 * half_width / nodes_per_edge
    coordinates = (
        -half_width + (torch.arange(nodes_per_edge, dtype=dtype) + 0.5) * spacing
    )
    first, second = torch.meshgrid(coordinates, coordinates, indexing="ij")
    patches, normals = [], []
    for axis in range(3):
        tangents = [idx for idx in range(3) if idx != axis]
        for sign in (-1.0, 1.0):
            patch = torch.zeros((nodes_per_edge, nodes_per_edge, 3), dtype=dtype)
            patch[..., axis] = sign * half_width
            patch[..., tangents[0]] = first
            patch[..., tangents[1]] = second
            patches.append(patch.reshape(-1, 3))
            normal = torch.zeros(3, dtype=dtype)
            normal[axis] = sign
            normals.append(normal.expand(nodes_per_edge * nodes_per_edge, 3))
    positions = torch.cat(patches)
    normal_vectors = torch.cat(normals)
    weights = torch.full((positions.shape[0],), spacing**2, dtype=dtype)
    frequency = torch.tensor([frequency_hz], dtype=dtype)
    omega = 2 * math.pi * frequency[0]
    k = omega * math.sqrt(EPSILON_0 * MU_0)
    displacement = positions
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    radial = displacement / radius[:, None]
    green = torch.exp(1j * k * radius) / (4 * math.pi * radius)
    kr_inverse = 1 / (k * radius)

    if dipole == "electric":
        # Electric dipole moment p, using the standard closed-form E and H fields.
        moment = torch.tensor(
            [1.0e-12 + 0.2e-12j, -0.3e-12j, 0.4e-12], dtype=complex_dtype
        )
        p_radial = torch.sum(radial.to(complex_dtype) * moment, dim=-1)
        transverse = moment - radial.to(complex_dtype) * p_radial[:, None]
        longitudinal = 3 * radial.to(complex_dtype) * p_radial[:, None] - moment
        electric = (
            (k**2 / EPSILON_0)
            * green[:, None]
            * (
                transverse
                + (kr_inverse.square() - 1j * kr_inverse)[:, None] * longitudinal
            )
        )
        magnetic = (
            (omega * k)
            * green[:, None]
            * (1 + 1j * kr_inverse)[:, None]
            * torch.linalg.cross(
                radial.to(complex_dtype),
                moment.expand_as(radial.to(complex_dtype)),
                dim=-1,
            )
        )
    elif dipole == "magnetic":
        # Magnetic dipole moment m; dual Maxwell fields with the same outgoing convention.
        moment = torch.tensor([0.7e-6 + 0.1e-6j, 0.2e-6, -0.4e-6j], dtype=complex_dtype)
        m_radial = torch.sum(radial.to(complex_dtype) * moment, dim=-1)
        transverse = moment - radial.to(complex_dtype) * m_radial[:, None]
        longitudinal = 3 * radial.to(complex_dtype) * m_radial[:, None] - moment
        magnetic = (
            k**2
            * green[:, None]
            * (
                transverse
                + (kr_inverse.square() - 1j * kr_inverse)[:, None] * longitudinal
            )
        )
        electric = (
            -(omega * MU_0 * k)
            * green[:, None]
            * (1 + 1j * kr_inverse)[:, None]
            * torch.linalg.cross(
                radial.to(complex_dtype),
                moment.expand_as(radial.to(complex_dtype)),
                dim=-1,
            )
        )
    else:
        raise AssertionError(dipole)

    surface = Surface(
        positions, normal_vectors, weights, electric[None], magnetic[None], frequency
    )
    return surface, moment, frequency


def _exact_dipole_field(
    dipole: str, moment: torch.Tensor, frequency: torch.Tensor, points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate independent closed-form Maxwell fields at selected locations."""
    omega = 2 * math.pi * frequency[0]
    k = omega * math.sqrt(EPSILON_0 * MU_0)
    displacement = points
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    radial = displacement / radius[:, None]
    green = torch.exp(1j * k * radius) / (4 * math.pi * radius)
    kr_inverse = 1 / (k * radius)
    radial_c = radial.to(torch.complex128)
    radial_moment = torch.sum(radial_c * moment, dim=-1)
    transverse = moment - radial_c * radial_moment[:, None]
    longitudinal = 3 * radial_c * radial_moment[:, None] - moment
    if dipole == "electric":
        electric = (
            (k**2 / EPSILON_0)
            * green[:, None]
            * (
                transverse
                + (kr_inverse.square() - 1j * kr_inverse)[:, None] * longitudinal
            )
        )
        magnetic = (
            omega
            * k
            * green[:, None]
            * (1 + 1j * kr_inverse)[:, None]
            * torch.linalg.cross(radial_c, moment.expand_as(radial_c), dim=-1)
        )
    else:
        magnetic = (
            k**2
            * green[:, None]
            * (
                transverse
                + (kr_inverse.square() - 1j * kr_inverse)[:, None] * longitudinal
            )
        )
        electric = (
            -omega
            * MU_0
            * k
            * green[:, None]
            * (1 + 1j * kr_inverse)[:, None]
            * torch.linalg.cross(radial_c, moment.expand_as(radial_c), dim=-1)
        )
    return electric, magnetic


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
def test_box_equivalence_reconstructs_dipole_outside_and_cancels_inside(dipole: str):
    surface, moment, frequency = _closed_box(dipole, nodes_per_edge=32)
    points = torch.tensor([[0.04, -0.03, 0.06], [0.9, 0.12, -0.1]], dtype=torch.float64)
    projected_e, projected_h = project_nearfield(
        surface,
        points,
        source_chunk_size=384,
        observation_chunk_size=2,
        checkpoint_chunks=False,
    )
    exact_e, exact_h = _exact_dipole_field(dipole, moment, frequency, points[1:])
    outside_e, outside_h = projected_e[0, 1:], projected_h[0, 1:]
    fine_e_error = torch.linalg.vector_norm(
        outside_e - exact_e
    ) / torch.linalg.vector_norm(exact_e)
    fine_h_error = torch.linalg.vector_norm(
        outside_h - exact_h
    ) / torch.linalg.vector_norm(exact_h)
    coarse_surface, _, _ = _closed_box(dipole, nodes_per_edge=16)
    coarse_e, coarse_h = project_nearfield(
        coarse_surface,
        points[1:],
        source_chunk_size=384,
        observation_chunk_size=1,
        checkpoint_chunks=False,
    )
    coarse_e_error = torch.linalg.vector_norm(
        coarse_e[0] - exact_e
    ) / torch.linalg.vector_norm(exact_e)
    coarse_h_error = torch.linalg.vector_norm(
        coarse_h[0] - exact_h
    ) / torch.linalg.vector_norm(exact_h)
    assert fine_e_error < coarse_e_error
    assert fine_h_error < coarse_h_error
    assert fine_e_error < 0.01
    assert fine_h_error < 0.01
    # With outward normals the Love currents reproduce the exterior and cancel inside.
    assert torch.linalg.vector_norm(
        projected_e[0, :1]
    ) < 0.01 * torch.linalg.vector_norm(exact_e)
    assert torch.linalg.vector_norm(
        projected_h[0, :1]
    ) < 0.01 * torch.linalg.vector_norm(exact_h)


@pytest.mark.parametrize("dipole", ["electric", "magnetic"])
def test_box_dipole_integrated_farfield_power_converges_to_analytic_value(dipole: str):
    surface, moment, frequency = _closed_box(dipole, nodes_per_edge=32)
    count = 2048
    index = torch.arange(count, dtype=torch.float64)
    z = 1 - 2 * (index + 0.5) / count
    phi = math.pi * (3 - math.sqrt(5)) * index
    radial = torch.sqrt(1 - z.square())
    directions = torch.stack(
        (radial * torch.cos(phi), radial * torch.sin(phi), z), dim=-1
    )
    electric_far, _ = project_farfield(
        surface,
        directions,
        source_chunk_size=384,
        observation_chunk_size=128,
        checkpoint_chunks=False,
    )
    integrated_power = directional_power(electric_far[0]).sum() * (4 * math.pi / count)
    omega = 2 * math.pi * frequency[0]
    k = omega * math.sqrt(EPSILON_0 * MU_0)
    moment_squared = torch.sum(moment.abs().square())
    if dipole == "electric":
        expected = (
            k**4
            * moment_squared
            / (12 * math.pi * EPSILON_0**2 * math.sqrt(MU_0 / EPSILON_0))
        )
    else:
        expected = math.sqrt(MU_0 / EPSILON_0) * k**4 * moment_squared / (12 * math.pi)
    assert integrated_power.item() == pytest.approx(expected.item(), rel=0.01)
