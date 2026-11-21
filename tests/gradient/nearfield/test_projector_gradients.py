"""Finite-difference checks for field gradients through the projectors."""

import pytest

torch = pytest.importorskip("torch")

from core.nearfield import project_farfield  # noqa: E402
from core.nearfield import Surface, project_nearfield, projection_adjoint

pytestmark = pytest.mark.gradient


def _surface(electric: torch.Tensor, magnetic: torch.Tensor) -> Surface:
    return Surface(
        positions_m=torch.tensor([[0.1, -0.2, 0.05]], dtype=torch.float64),
        normals=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        weights_m2=torch.tensor([0.3], dtype=torch.float64),
        electric_fields=electric,
        magnetic_fields=magnetic,
        frequencies_hz=torch.tensor([1.3e8], dtype=torch.float64),
    )


@pytest.mark.gradient
def test_nearfield_electric_and_magnetic_field_gradients_match_finite_difference():
    electric = torch.tensor(
        [[[1.0 + 0.5j, -0.2 + 0.1j, 0.3 - 0.4j]]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    magnetic = torch.tensor(
        [[[0.2 - 0.1j, 0.3 + 0.8j, -0.5 + 0.4j]]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    observation = torch.tensor([[0.25, 0.15, 0.4]], dtype=torch.float64)

    def objective(electric_input, magnetic_input):
        e, h = project_nearfield(
            _surface(electric_input, magnetic_input),
            observation,
            checkpoint_chunks=True,
        )
        return e.abs().square().sum() + 0.4 * h.abs().square().sum()

    assert torch.autograd.gradcheck(
        objective, (electric, magnetic), eps=1e-6, atol=1e-5, rtol=2e-4
    )


@pytest.mark.gradient
def test_farfield_electric_and_magnetic_field_gradients_match_finite_difference():
    electric = torch.tensor(
        [[[1.0 + 0.5j, -0.2 + 0.1j, 0.3 - 0.4j]]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    magnetic = torch.tensor(
        [[[0.2 - 0.1j, 0.3 + 0.8j, -0.5 + 0.4j]]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    directions = torch.tensor([[0.2, 0.1, (0.95**0.5)]], dtype=torch.float64)
    directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)

    def objective(electric_input, magnetic_input):
        e, h = project_farfield(
            _surface(electric_input, magnetic_input), directions, checkpoint_chunks=True
        )
        return e.abs().square().sum() + 0.4 * h.abs().square().sum()

    assert torch.autograd.gradcheck(
        objective, (electric, magnetic), eps=1e-6, atol=1e-8, rtol=2e-4
    )


@pytest.mark.parametrize("mode", ["near", "far"])
def test_fixed_geometry_adjoint_satisfies_complex_inner_product_identity(mode):
    torch.manual_seed(21)
    dtype = torch.float64
    positions = torch.tensor(
        [[0.1, -0.2, 0.05], [-0.15, 0.08, 0.12], [0.2, 0.1, -0.04]], dtype=dtype
    )
    normals = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype
    )
    weights = torch.tensor([0.12, 0.09, 0.15], dtype=dtype)
    frequencies = torch.tensor([1.1e8, 1.4e8], dtype=dtype)
    base_e = torch.randn((2, 3, 3), dtype=torch.complex128)
    base_h = torch.randn((2, 3, 3), dtype=torch.complex128)
    delta_e = torch.randn((2, 3, 3), dtype=torch.complex128)
    delta_h = torch.randn((2, 3, 3), dtype=torch.complex128)
    surface = Surface(positions, normals, weights, base_e, base_h, frequencies)
    perturbed = Surface(
        positions, normals, weights, base_e + delta_e, base_h + delta_h, frequencies
    )
    if mode == "near":
        observations = torch.tensor([[0.4, -0.1, 0.3], [-0.3, 0.25, 0.22]], dtype=dtype)
        forward = project_nearfield
    else:
        observations = torch.tensor(
            [[0.0, 0.0, 1.0], [0.3, -0.4, 0.866025403784]], dtype=dtype
        )
        observations = observations / torch.linalg.vector_norm(
            observations, dim=-1, keepdim=True
        )
        forward = project_farfield

    base_out = forward(
        surface, observations, source_chunk_size=2, observation_chunk_size=1
    )
    perturbed_out = forward(
        perturbed, observations, source_chunk_size=2, observation_chunk_size=1
    )
    delta_out = (perturbed_out[0] - base_out[0], perturbed_out[1] - base_out[1])
    cotangent_e = torch.randn(delta_out[0].shape, dtype=torch.complex128)
    cotangent_h = torch.randn(delta_out[1].shape, dtype=torch.complex128)
    adjoint_e, adjoint_h = projection_adjoint(
        surface,
        observations,
        cotangent_e,
        cotangent_h,
        mode=mode,
        source_chunk_size=2,
        observation_chunk_size=1,
    )
    forward_inner = torch.vdot(
        cotangent_e.flatten(), delta_out[0].flatten()
    ) + torch.vdot(cotangent_h.flatten(), delta_out[1].flatten())
    adjoint_inner = torch.vdot(adjoint_e.flatten(), delta_e.flatten()) + torch.vdot(
        adjoint_h.flatten(), delta_h.flatten()
    )
    torch.testing.assert_close(forward_inner, adjoint_inner, rtol=2e-11, atol=1e-11)
