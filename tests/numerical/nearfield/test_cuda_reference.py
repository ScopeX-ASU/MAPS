"""Executed CUDA precision and adjoint checks against CPU complex128."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from core.nearfield import Surface, project_farfield, project_nearfield

pytestmark = [pytest.mark.gpu, pytest.mark.numerical, pytest.mark.gradient]


def _case():
    generator = torch.Generator().manual_seed(147)
    positions = torch.randn(15, 3, generator=generator, dtype=torch.float64) * 1e-7
    normals = torch.zeros_like(positions)
    normals[:, 2] = 1
    electric = torch.randn(2, 15, 3, generator=generator, dtype=torch.complex128)
    magnetic = torch.randn(2, 15, 3, generator=generator, dtype=torch.complex128) / 377
    surface = Surface(
        positions,
        normals,
        torch.full((15,), 1e-14, dtype=torch.float64),
        electric,
        magnetic,
        torch.tensor([1.7e14, 2.1e14], dtype=torch.float64),
    )
    points = torch.randn(7, 3, generator=generator, dtype=torch.float64) * 1e-7
    points[:, 2] += 1.3e-6
    return surface, points


def _loss(electric, magnetic, objective):
    if objective == "focus":
        return electric[:, 2].abs().square().sum()
    if objective == "directional":
        return (
            electric.abs().square().sum(-1)
            * torch.linspace(0.1, 1, electric.shape[1], device=electric.device)
        ).sum()
    if objective == "matching":
        return (electric - (0.02 + 0.03j)).abs().square().sum() + 377**2 * (
            magnetic - (0.0001 - 0.0002j)
        ).abs().square().sum()
    return (electric[..., 0] + 1j * electric[..., 1]).abs().square().sum()


@pytest.mark.parametrize("method", ["exact", "far"])
@pytest.mark.parametrize(
    "objective", ["focus", "directional", "matching", "polarization"]
)
def test_cuda_complex64_fields_and_gradients_match_cpu_complex128(
    require_cuda, method, objective
):
    surface, points = _case()
    operation = project_nearfield if method == "exact" else project_farfield
    if method == "far":
        points = points / torch.linalg.vector_norm(points, dim=-1, keepdim=True)
    cpu = replace(
        surface,
        electric_fields=surface.electric_fields.requires_grad_(),
        magnetic_fields=surface.magnetic_fields.requires_grad_(),
    )
    reference = operation(cpu, points, source_chunk_size=4, observation_chunk_size=3)
    reference_grad = torch.autograd.grad(
        _loss(*reference, objective), (cpu.electric_fields, cpu.magnetic_fields)
    )
    cuda_values = {}
    for name in surface.__dataclass_fields__:
        tensor = (
            getattr(surface, name)
            .detach()
            .to(
                "cuda",
                dtype=(
                    torch.complex64
                    if getattr(surface, name).is_complex()
                    else torch.float32
                ),
            )
        )
        if name in ("electric_fields", "magnetic_fields"):
            tensor.requires_grad_()
        cuda_values[name] = tensor
    gpu = Surface(**cuda_values)
    actual = operation(
        gpu,
        points.to("cuda", dtype=torch.float32),
        source_chunk_size=6,
        observation_chunk_size=2,
    )
    actual_grad = torch.autograd.grad(
        _loss(*actual, objective), (gpu.electric_fields, gpu.magnetic_fields)
    )
    for candidate, expected in zip(
        (*actual, *actual_grad), (*reference, *reference_grad)
    ):
        error = torch.linalg.vector_norm(
            candidate.detach().cpu().to(expected.dtype) - expected.detach()
        ) / torch.linalg.vector_norm(expected.detach())
        assert error < 2e-5, (method, objective, error.item())
        assert candidate.device.type == "cuda" and candidate.dtype == torch.complex64
