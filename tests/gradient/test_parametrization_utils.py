"""Gradient and autograd contract tests for inverse-design parametrization utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


_UTILS_PATH = (
    Path(__file__).resolve().parents[2]
    / "core"
    / "invdes"
    / "models"
    / "layers"
    / "parametrization"
    / "utils.py"
)
_spec = importlib.util.spec_from_file_location(
    "maps_invdes_parametrization_utils", _UTILS_PATH
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load parametrization utilities from {_UTILS_PATH}")
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

BinaryProjectionLayer = _utils.BinaryProjectionLayer
HeavisideProjection = _utils.HeavisideProjection
_gradient_2d_by_index = _utils._gradient_2d_by_index
tanh_projection = _utils.tanh_projection


@pytest.mark.gradient
def test_tanh_projection_zero_beta_is_clipped_identity() -> None:
    """Zero sharpness reduces to clipping the input density into ``[0, 1]``."""
    x = torch.tensor([-0.4, 0.2, 1.4], dtype=torch.float64)
    result = tanh_projection(x, beta=0.0, eta=0.5)
    expected = torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64)
    torch.testing.assert_close(result, expected)


@pytest.mark.gradient
def test_tanh_projection_infinite_beta_is_hard_threshold() -> None:
    """Infinite sharpness produces the documented hard threshold at ``eta``."""
    x = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    result = tanh_projection(x, beta=float("inf"), eta=0.5)
    expected = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(result, expected)


@pytest.mark.gradient
def test_gradient_2d_by_index_recovers_linear_field() -> None:
    """The unit-spacing stencil recovers constant slopes on a linear 2-D field."""
    i = torch.arange(4, dtype=torch.float64)[:, None]
    j = torch.arange(5, dtype=torch.float64)[None, :]
    x = 2.0 * i + 3.0 * j

    grad0, grad1 = _gradient_2d_by_index(x)

    torch.testing.assert_close(grad0, torch.full_like(x, 2.0))
    torch.testing.assert_close(grad1, torch.full_like(x, 3.0))


@pytest.mark.gradient
def test_gradient_2d_by_index_rejects_non_2d_input() -> None:
    """The gradient helper rejects tensors outside its documented 2-D contract."""
    with pytest.raises(ValueError, match="Expected a 2D tensor"):
        _gradient_2d_by_index(torch.ones(2, 3, 4))


@pytest.mark.gradient
def test_binary_projection_backward_is_finite_and_negative() -> None:
    """The custom binary-projection backward pass produces finite negative slopes."""
    x = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True)
    layer = BinaryProjectionLayer(threshold=0.05)

    loss = layer(x, t_bny=0.2).sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.all(x.grad < 0)


@pytest.mark.gradient
def test_heaviside_regular_and_ste_modes_preserve_shape_and_gradient() -> None:
    """Both supported projection modes keep tensor shape and expose finite gradients."""
    x_regular = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True)
    regular = HeavisideProjection(mode="regular")
    y_regular = regular(x_regular, beta=torch.tensor(8.0), eta=torch.tensor(0.5))
    y_regular.sum().backward()

    x_ste = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True)
    ste = HeavisideProjection(mode="ste")
    y_ste = ste(x_ste, beta=torch.tensor(8.0), eta=torch.tensor(0.5))
    y_ste.sum().backward()

    assert y_regular.shape == x_regular.shape
    assert y_ste.shape == x_ste.shape
    assert torch.all((y_regular >= 0) & (y_regular <= 1))
    assert torch.all((y_ste >= 0) & (y_ste <= 1))
    assert x_regular.grad is not None and torch.isfinite(x_regular.grad).all()
    assert x_ste.grad is not None and torch.isfinite(x_ste.grad).all()
