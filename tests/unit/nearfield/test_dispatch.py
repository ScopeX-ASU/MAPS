"""Contracts for exact-shape benchmark based near-field dispatch."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from core.nearfield.dispatch import project, select_backend  # noqa: E402
from core.nearfield.projector import make_planar_surface  # noqa: E402
from core.nearfield.projector import project_nearfield

pytestmark = pytest.mark.unit


def _fake_inputs():
    device = SimpleNamespace(type="cuda")

    def real(shape):
        return SimpleNamespace(
            shape=shape, dtype=torch.float32, device=device, requires_grad=False
        )

    surface = SimpleNamespace(
        positions_m=real((1024, 3)),
        normals=real((1024, 3)),
        weights_m2=real((1024,)),
        electric_fields=SimpleNamespace(
            shape=(1, 1024, 3),
            dtype=torch.complex64,
            device=device,
            requires_grad=False,
        ),
        magnetic_fields=SimpleNamespace(
            shape=(1, 1024, 3),
            dtype=torch.complex64,
            device=device,
            requires_grad=False,
        ),
        frequencies_hz=real((1,)),
    )
    observations = real((100, 3))
    observations.ndim = 2
    return surface, observations


def _record(
    backend="torch", *, first=10.0, warm=1.0, first_backward=30.0, warm_backward=4.0
):
    surface, observations = _fake_inputs()
    del surface, observations
    return {
        "benchmark": "core.nearfield.project_nearfield",
        "backend": backend,
        "hardware": {
            "gpu_name": "Test GPU",
            "torch_version": "test-torch",
            "cuda_version": "test-cuda",
        },
        "configuration": {
            "real_dtype": "float32",
            "source_chunk_size": 256,
            "observation_chunk_size": 128,
            "source_chunk_applies": backend == "torch",
            "checkpoint_chunks": backend == "torch",
            "checkpoint_chunks_applies": backend == "torch",
            "triton_observation_block_size": 32 if backend == "triton" else None,
            "triton_source_block_size": 16 if backend == "triton" else None,
            "triton_backward_source_chunk_size": 256 if backend == "triton" else None,
            "triton_backward_observation_chunk_size": (
                128 if backend == "triton" else None
            ),
            "measured_warm_repeats": 3,
        },
        "measurements": {
            "surface_points": 1024,
            "observation_points": 100,
            "frequencies": 1,
            "field_dtype": "torch.complex64",
            "first_forward_seconds": first,
            "warm_forward_median_seconds": warm,
            "first_gradient_forward_seconds": first_backward - 1,
            "first_backward_seconds": 1.0,
            "first_forward_backward_seconds": first_backward,
            "warm_forward_backward_median_seconds": warm_backward,
        },
        "precision_reference": {
            "relative_l2_E_vs_complex128": 0.001,
            "relative_l2_H_vs_complex128": 0.001,
        },
    }


def _enable_fake_cuda_identity(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "Test GPU")
    monkeypatch.setattr(torch, "__version__", "test-torch")
    monkeypatch.setattr(torch.version, "cuda", "test-cuda")


def test_auto_uses_first_and_warm_forward_cost(monkeypatch):
    _enable_fake_cuda_identity(monkeypatch)
    monkeypatch.setattr("core.nearfield.dispatch.find_spec", lambda _name: object())
    surface, observations = _fake_inputs()
    torch_record = _record("torch", first=10.0, warm=1.0)
    triton_record = _record("triton", first=3.0, warm=2.0)

    decision = select_backend(
        surface,
        observations,
        calibration_records=(torch_record, triton_record),
        expected_calls=5,
    )

    assert decision.backend == "direct_triton"
    assert decision.estimated_seconds == pytest.approx(11.0)


def test_auto_uses_backward_cost_when_requested(monkeypatch):
    _enable_fake_cuda_identity(monkeypatch)
    monkeypatch.setattr("core.nearfield.dispatch.find_spec", lambda _name: object())
    surface, observations = _fake_inputs()
    torch_record = _record("torch", first_backward=4.0, warm_backward=1.0)
    triton_record = _record("triton", first_backward=20.0, warm_backward=0.5)

    decision = select_backend(
        surface,
        observations,
        calibration_records=(torch_record, triton_record),
        expected_calls=4,
        requires_backward=True,
    )

    assert decision.backend == "direct_torch"
    assert decision.estimated_seconds == pytest.approx(16.0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["hardware"].update(gpu_name="Different GPU"),
        lambda r: r["measurements"].update(field_dtype="torch.complex128"),
        lambda r: r["configuration"].update(measured_warm_repeats=2),
        lambda r: r["precision_reference"].update(relative_l2_E_vs_complex128=0.02),
        lambda r: r["measurements"].update(first_forward_seconds=float("nan")),
        lambda r: r["configuration"].update(observation_chunk_size=64),
    ],
)
def test_invalid_or_incompatible_record_is_ignored(monkeypatch, mutation):
    _enable_fake_cuda_identity(monkeypatch)
    surface, observations = _fake_inputs()
    invalid = deepcopy(_record())
    mutation(invalid)

    decision = select_backend(surface, observations, calibration_records=(invalid,))

    assert decision.backend == "direct_torch"
    assert decision.estimated_seconds is None
    assert "no applicable" in decision.reason


@pytest.mark.parametrize("calls", [0, -1, 1.5, True])
def test_expected_calls_must_be_positive_integer(calls):
    surface, observations = _fake_inputs()

    with pytest.raises(ValueError, match="positive integer"):
        select_backend(surface, observations, expected_calls=calls)


def test_auto_falls_back_on_cpu_and_matches_direct_torch():
    dtype = torch.float64
    coords = torch.tensor([-1e-6, 0.0, 1e-6], dtype=dtype)
    electric = torch.ones((1, 3, 3, 3), dtype=torch.complex128)
    magnetic = 0.2j * electric
    surface = make_planar_surface(
        torch.zeros(3, dtype=dtype),
        torch.tensor([1.0, 0.0, 0.0], dtype=dtype),
        torch.tensor([0.0, 1.0, 0.0], dtype=dtype),
        coords,
        coords,
        electric,
        magnetic,
        torch.tensor([2.0e14], dtype=dtype),
    )
    observations = torch.tensor([[0.0, 0.0, 3e-6]], dtype=dtype)

    expected = project_nearfield(
        surface, observations, source_chunk_size=256, observation_chunk_size=128
    )
    actual = project(surface, observations, "auto")

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
