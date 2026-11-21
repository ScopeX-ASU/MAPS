"""Integration contracts for the legacy 2-D near-to-far objective path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import core.invdes.models.layers.objective as objective_module
from core.invdes.models.layers.objective import ShapeSimilarityNear2FarObjective

pytestmark = pytest.mark.integration


def _objective(return_field: str | None) -> ShapeSimilarityNear2FarObjective:
    """Build one selected 2-D objective with simple monitor metadata."""
    near_slice = SimpleNamespace(x=np.asarray([[1]]), y=np.arange(3)[None, :])
    far_slice = SimpleNamespace(x=np.arange(2)[:, None], y=np.arange(3)[None, :])
    return ShapeSimilarityNear2FarObjective(
        sims={(1.55, "Ez", 300): object()},
        port_slices={"nearfield_right": near_slice, "farfield": far_slice},
        port_slices_info={
            "nearfield_right": {
                "direction": "x+",
                "xs": np.asarray(0.0),
                "ys": np.arange(3),
            },
            "farfield": {
                "direction": "x+",
                "xs": np.arange(2),
                "ys": np.arange(3),
            },
        },
        in_slice_name="source",
        out_slice_name="farfield",
        in_mode="Ez1",
        out_modes=("Ez1",),
        name="raw_2d",
        target_wls=(1.55,),
        target_temps=(300,),
        shape_type=None,
        shape_cfg=None,
        grid_step=0.1,
        eps_bg=1.0,
        return_field=return_field,
    )


def _fields() -> dict:
    """Return differentiable 2-D fields in the legacy component layout."""
    key = ("source", 1.55, "Ez1", 300)
    base = torch.arange(12, dtype=torch.float64).reshape(3, 4).to(torch.complex128)
    base.requires_grad_(True)
    return {key: {"Ez": base, "Hx": 2 * base, "Hy": 3 * base}}


def test_return_nearfield_bypasses_projection_and_similarity(monkeypatch):
    """Raw near fields preserve complex tensors and their autograd graph."""
    monkeypatch.setattr(
        objective_module,
        "get_farfields_GreenFunction",
        lambda **_: pytest.fail("projection must not run for return_field='nearfield'"),
    )
    fields = _fields()
    result = _objective("nearfield")(fields)
    key = ("source", 1.55, "Ez1", 300)

    assert set(result[key]) == {"Ez", "Hx", "Hy"}
    assert result[key]["Ez"] is fields[key]["Ez"]
    result[key]["Ez"].real.sum().backward()
    assert fields[key]["Ez"].grad is not None


def test_return_farfield_returns_all_legacy_projector_components(monkeypatch):
    """Raw far fields bypass shape reduction without dropping polarization."""
    projected = {
        name: torch.full((1, 2, 3, 1), index + 1j, dtype=torch.complex128)
        for index, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
    }
    monkeypatch.setattr(
        objective_module, "get_farfields_GreenFunction", lambda **_: projected
    )
    monkeypatch.setattr(
        objective_module,
        "get_shape_similarity",
        lambda *_, **__: pytest.fail("similarity must not run for raw fields"),
    )

    result = _objective("farfield")(_fields())
    key = ("source", 1.55, "Ez1", 300)
    assert result[key] is projected
    assert set(result[key]) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}


def test_default_2d_mode_preserves_scalar_similarity(monkeypatch):
    """Unset ``return_field`` keeps the legacy projected-shape reduction."""
    projected = {
        name: torch.ones((1, 2, 3, 1), dtype=torch.complex128)
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    captured = {}
    monkeypatch.setattr(
        objective_module, "get_farfields_GreenFunction", lambda **_: projected
    )

    def _similarity(field, **kwargs):
        captured["field"] = field
        captured["kwargs"] = kwargs
        return field.abs().mean()

    monkeypatch.setattr(objective_module, "get_shape_similarity", _similarity)
    result = _objective(None)(_fields())

    assert result.ndim == 0
    assert captured["field"].shape == (2, 3)
    assert captured["kwargs"]["intensity"] is True
