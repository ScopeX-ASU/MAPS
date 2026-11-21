"""Fast source-level contracts for MAPS FDTD-X simulation configuration."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.fdtdx, pytest.mark.unit]


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_EXAMPLE = _REPO_ROOT / "core/invdes/examples/waveguide_3d.py"
_WAVEGUIDE_REGRESSION = _REPO_ROOT / "tests/fdtdx/test_maps_waveguide3d_forward.py"

_REQUIRED_SIM_CFG_KEYS = {
    "solver",
    "resolution",
    "subpixel",
    "wl_cen",
    "wl_width",
    "n_wl",
    "max_time",
    "border_width",
    "PML",
    "cell_size",
    "plot_root",
    "optical_grid",
}
_REQUIRED_OPTICAL_GRID_KEYS = {
    "mode",
    "wavelength",
    "grid_x",
    "grid_y",
    "grid_z",
}


def _find_sim_cfg_dict(source_path: Path, function_name: str) -> ast.Dict:
    """Return the ``sim_cfg = dict(...)`` literal from a source function."""
    module = ast.parse(
        source_path.read_text(encoding="utf-8"), filename=str(source_path)
    )
    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "sim_cfg"
                for target in child.targets
            ):
                continue
            if isinstance(child.value, ast.Call) and isinstance(
                child.value.func, ast.Name
            ):
                if child.value.func.id == "dict":
                    return child.value  # type: ignore[return-value]
    raise AssertionError(f"Could not find sim_cfg=dict(...) in {source_path}")


def _keyword_names(config_call: ast.Call) -> set[str]:
    return {kw.arg for kw in config_call.keywords if kw.arg is not None}


def _nested_dict_keyword_names(config_call: ast.Call, key: str) -> set[str]:
    for keyword in config_call.keywords:
        if keyword.arg != key:
            continue
        value = keyword.value
        if isinstance(value, ast.Dict):
            names = set()
            for item_key in value.keys:
                if isinstance(item_key, ast.Constant) and isinstance(
                    item_key.value, str
                ):
                    names.add(item_key.value)
            return names
        raise AssertionError(f"{key!r} must be a dict literal")
    raise AssertionError(f"Missing {key!r} from sim_cfg")


def test_canonical_waveguide_fdtdx_sim_cfg_contract() -> None:
    """Protect the canonical example's required FDTD-X configuration fields."""
    sim_cfg = _find_sim_cfg_dict(_CANONICAL_EXAMPLE, "main")
    assert _REQUIRED_SIM_CFG_KEYS <= _keyword_names(sim_cfg)
    assert _REQUIRED_OPTICAL_GRID_KEYS <= _nested_dict_keyword_names(
        sim_cfg, "optical_grid"
    )


def test_waveguide_regression_copies_required_fdtdx_sim_cfg_contract() -> None:
    """Prevent the MAPS regression fixture from dropping required canonical fields."""
    sim_cfg = _find_sim_cfg_dict(_WAVEGUIDE_REGRESSION, "_build_waveguide")
    assert _REQUIRED_SIM_CFG_KEYS <= _keyword_names(sim_cfg)
    assert _REQUIRED_OPTICAL_GRID_KEYS <= _nested_dict_keyword_names(
        sim_cfg, "optical_grid"
    )
