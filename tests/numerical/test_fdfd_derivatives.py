"""CPU numerical regression checks for FDFD finite-difference operators."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CEVICHE_DIR = _REPO_ROOT / "thirdparty" / "ceviche"

_constants_spec = importlib.util.spec_from_file_location(
    "thirdparty.ceviche.constants", _CEVICHE_DIR / "constants.py"
)
assert _constants_spec is not None and _constants_spec.loader is not None
_constants = importlib.util.module_from_spec(_constants_spec)
_constants_spec.loader.exec_module(_constants)

_ceviche_stub = types.ModuleType("thirdparty.ceviche")
_ceviche_stub.__path__ = [str(_CEVICHE_DIR)]
sys.modules.setdefault("thirdparty.ceviche", _ceviche_stub)
sys.modules["thirdparty.ceviche.constants"] = _constants

_derivatives_spec = importlib.util.spec_from_file_location(
    "maps_fdfd_derivatives", _REPO_ROOT / "core" / "fdfd" / "derivatives.py"
)
assert _derivatives_spec is not None and _derivatives_spec.loader is not None
_derivatives = importlib.util.module_from_spec(_derivatives_spec)
_derivatives_spec.loader.exec_module(_derivatives)

make_Dxb = _derivatives.make_Dxb
make_Dxf = _derivatives.make_Dxf
make_Dyb = _derivatives.make_Dyb
make_Dyf = _derivatives.make_Dyf
createDws = _derivatives.createDws


@pytest.mark.numerical
def test_x_forward_derivative_matches_periodic_reference() -> None:
    """Forward x differences reproduce the periodic finite-difference stencil."""
    field = torch.tensor([0.0, 1.0, 2.0], dtype=torch.complex128)
    operator = make_Dxf(dL=1.0, shape=(3, 1), device="cpu")
    result = torch.sparse.mm(operator, field[:, None]).squeeze(1)

    expected = torch.tensor([1.0, 1.0, -2.0], dtype=torch.complex128)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.numerical
def test_x_backward_derivative_matches_periodic_reference() -> None:
    """Backward x differences reproduce the periodic backward stencil."""
    field = torch.tensor([0.0, 1.0, 2.0], dtype=torch.complex128)
    operator = make_Dxb(dL=1.0, shape=(3, 1), device="cpu")
    result = torch.sparse.mm(operator, field[:, None]).squeeze(1)

    expected = torch.tensor([-2.0, 1.0, 1.0], dtype=torch.complex128)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.numerical
def test_y_forward_and_backward_derivatives_match_periodic_reference() -> None:
    """Y-direction operators reproduce the analogous periodic stencils."""
    field = torch.tensor([0.0, 1.0, 2.0], dtype=torch.complex128)
    forward = make_Dyf(dL=1.0, shape=(1, 3), device="cpu")
    backward = make_Dyb(dL=1.0, shape=(1, 3), device="cpu")

    forward_result = torch.sparse.mm(forward, field[:, None]).squeeze(1)
    backward_result = torch.sparse.mm(backward, field[:, None]).squeeze(1)

    torch.testing.assert_close(
        forward_result,
        torch.tensor([1.0, 1.0, -2.0], dtype=torch.complex128),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        backward_result,
        torch.tensor([-2.0, 1.0, 1.0], dtype=torch.complex128),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.numerical
def test_invalid_derivative_component_and_direction_raise() -> None:
    """Derivative construction rejects unsupported component/direction pairs."""
    with pytest.raises(ValueError, match="not recognized"):
        createDws("z", "f", shape=(3, 3), dL=1.0, device="cpu")
