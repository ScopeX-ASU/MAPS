"""CUDA agreement checks for the optional exact Triton backend."""

# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")

from core.nearfield import Surface, project_nearfield
from core.nearfield.triton_backend import project_nearfield_triton, triton

pytestmark = pytest.mark.numerical


def _require_backend():
    if triton is None or not torch.cuda.is_available():
        pytest.skip("Triton CUDA backend is unavailable")


def _make_surface(electric, magnetic):
    device = electric.device
    count = electric.shape[1]
    coordinate = torch.linspace(-0.2, 0.2, count, dtype=torch.float32, device=device)
    return Surface(
        positions_m=torch.stack(
            (
                coordinate,
                0.11 * torch.sin(11 * coordinate),
                0.06 * torch.cos(7 * coordinate),
            ),
            dim=-1,
        ),
        normals=torch.nn.functional.normalize(
            torch.tensor([[0.0, 0.0, 1.0]] * count, dtype=torch.float32, device=device),
            dim=-1,
        ),
        weights_m2=torch.full((count,), 0.01, dtype=torch.float32, device=device),
        electric_fields=electric,
        magnetic_fields=magnetic,
        frequencies_hz=torch.linspace(
            0.9e8, 1.2e8, electric.shape[0], dtype=torch.float32, device=device
        ),
    )


def test_triton_exact_nearfield_matches_torch_for_e_and_h_multiple_frequencies():
    _require_backend()
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(17)
    e = torch.randn((2, 37, 3), generator=gen, device=device) + 1j * torch.randn(
        (2, 37, 3), generator=gen, device=device
    )
    h = torch.randn((2, 37, 3), generator=gen, device=device) + 1j * torch.randn(
        (2, 37, 3), generator=gen, device=device
    )
    surface = _make_surface(e.to(torch.complex64), h.to(torch.complex64))
    observations = torch.stack(
        (
            torch.linspace(0.45, -0.45, 35, device=device),
            torch.linspace(0.25, -0.25, 35, device=device),
            torch.linspace(0.4, 0.55, 35, device=device),
        ),
        dim=-1,
    )
    actual = project_nearfield_triton(surface, observations, observation_block_size=8)
    expected = project_nearfield(
        surface, observations, source_chunk_size=8, observation_chunk_size=8
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=4e-4, atol=3e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=4e-4, atol=3e-5)


def test_triton_adjoint_matches_native_torch_for_complex_fields():
    _require_backend()
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(22)
    e = (
        (
            torch.randn((2, 37, 3), generator=gen, device=device)
            + 1j * torch.randn((2, 37, 3), generator=gen, device=device)
        )
        .to(torch.complex64)
        .requires_grad_()
    )
    h = (
        (
            torch.randn((2, 37, 3), generator=gen, device=device)
            + 1j * torch.randn((2, 37, 3), generator=gen, device=device)
        )
        .to(torch.complex64)
        .requires_grad_()
    )
    surface = _make_surface(e, h)
    observations = torch.tensor(
        [[0.5, 0.2, 0.4], [-0.3, 0.4, 0.6], [0.35, -0.45, 0.31]], device=device
    )
    probe_e = torch.randn((2, 3, 3), dtype=torch.complex64, device=device)
    probe_h = torch.randn((2, 3, 3), dtype=torch.complex64, device=device)
    actual = project_nearfield_triton(surface, observations, observation_block_size=4)
    loss = torch.real(
        torch.sum(torch.conj(probe_e) * actual[0])
        + torch.sum(torch.conj(probe_h) * actual[1])
    )
    grad_actual = torch.autograd.grad(loss, (e, h))
    expected = project_nearfield(surface, observations, checkpoint_chunks=True)
    loss_ref = torch.real(
        torch.sum(torch.conj(probe_e) * expected[0])
        + torch.sum(torch.conj(probe_h) * expected[1])
    )
    grad_ref = torch.autograd.grad(loss_ref, (e, h))
    torch.testing.assert_close(grad_actual[0], grad_ref[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(grad_actual[1], grad_ref[1], rtol=2e-5, atol=2e-6)


def test_triton_complex_intensity_gradient_matches_centered_directional_difference():
    _require_backend()
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(41)
    e = (
        (
            torch.randn((2, 37, 3), generator=gen, device=device)
            + 1j * torch.randn((2, 37, 3), generator=gen, device=device)
        )
        .to(torch.complex64)
        .requires_grad_()
    )
    h = (
        (
            torch.randn((2, 37, 3), generator=gen, device=device)
            + 1j * torch.randn((2, 37, 3), generator=gen, device=device)
        )
        .to(torch.complex64)
        .requires_grad_()
    )
    observations = torch.tensor(
        [[0.5, 0.2, 0.4], [-0.3, 0.4, 0.6], [0.35, -0.45, 0.31]], device=device
    )
    direction_e = (
        torch.randn(e.shape, generator=gen, device=device)
        + 1j * torch.randn(e.shape, generator=gen, device=device)
    ).to(torch.complex64)
    direction_h = (
        torch.randn(h.shape, generator=gen, device=device)
        + 1j * torch.randn(h.shape, generator=gen, device=device)
    ).to(torch.complex64)

    def objective(electric, magnetic):
        fields = _make_surface(electric, magnetic)
        e_out, h_out = project_nearfield_triton(
            fields, observations, observation_block_size=4
        )
        return e_out.abs().square().sum() + 0.3 * h_out.abs().square().sum()

    value = objective(e, h)
    grad_e, grad_h = torch.autograd.grad(value, (e, h))
    predicted = torch.real(
        torch.sum(torch.conj(grad_e) * direction_e)
        + torch.sum(torch.conj(grad_h) * direction_h)
    )
    step = 1e-3
    with torch.no_grad():
        plus = objective(e + step * direction_e, h + step * direction_h)
        minus = objective(e - step * direction_e, h - step * direction_h)
    measured = (plus - minus) / (2 * step)
    assert torch.abs(predicted - measured) <= 0.01 * torch.maximum(
        torch.abs(measured), torch.tensor(1e-8, device=device)
    )


def test_triton_rejects_observation_coincident_with_source_sample():
    _require_backend()
    device = torch.device("cuda")
    e = torch.ones((1, 37, 3), dtype=torch.complex64, device=device)
    h = torch.ones_like(e)
    surface = _make_surface(e, h)
    with pytest.raises(ValueError, match="coincide"):
        project_nearfield_triton(
            surface, surface.positions_m[:1].contiguous(), observation_block_size=4
        )
