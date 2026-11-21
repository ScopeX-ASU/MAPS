"""Deterministic optimizer integration checks."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.invdes.optimizer.adam import Adam


@pytest.mark.integration
@pytest.mark.gradient
def test_custom_adam_reduces_quadratic_and_tracks_state() -> None:
    """Custom Adam should reduce a convex quadratic and initialize optimizer state."""
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.5], dtype=torch.float64))
    target = torch.tensor([0.25, -0.5], dtype=torch.float64)
    optimizer = Adam([parameter], lr=0.08, init_v=1e-3)

    initial_loss = torch.sum((parameter.detach() - target) ** 2).item()

    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.sum((parameter - target) ** 2)
        loss.backward()
        optimizer.step()

    final_loss = torch.sum((parameter.detach() - target) ** 2).item()

    assert final_loss < initial_loss * 1e-3
    assert torch.allclose(parameter.detach(), target, atol=5e-2, rtol=0.0)
    assert parameter in optimizer.state

    state = optimizer.state[parameter]
    assert state["step"].item() == 80
    assert "exp_avg" in state
    assert "exp_avg_sq" in state


@pytest.mark.integration
def test_custom_adam_state_dict_round_trip_preserves_step() -> None:
    """Optimizer state serialization should preserve the accumulated step count."""
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = Adam([parameter], lr=0.05, init_v=1e-3)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (parameter**2).sum()
        loss.backward()
        optimizer.step()

    state_dict = optimizer.state_dict()

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = Adam([restored_parameter], lr=0.05, init_v=1e-3)
    restored.load_state_dict(state_dict)

    restored_state = restored.state[restored_parameter]
    assert restored_state["step"].item() == 3
    torch.testing.assert_close(
        restored_state["exp_avg"], optimizer.state[parameter]["exp_avg"]
    )
    torch.testing.assert_close(
        restored_state["exp_avg_sq"], optimizer.state[parameter]["exp_avg_sq"]
    )
