"""Focused tests for 2D rectilinear monitor weights and observables."""

import importlib
import logging
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


def _install_import_stubs():
    """Provide optional symbols unused by these focused numerical tests."""
    pyutils = types.ModuleType("pyutils")
    pyutils.__path__ = []

    config = types.ModuleType("pyutils.config")

    class Config(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    config.Config = Config

    general = types.ModuleType("pyutils.general")
    general.ensure_dir = lambda path: path
    general.logger = logging.getLogger("tests.fdfd")

    torch_sparse = types.ModuleType("torch_sparse")

    def _unused_spmm(*args, **kwargs):
        raise AssertionError("torch_sparse.spmm is outside this test's scope")

    torch_sparse.spmm = _unused_spmm

    sys.modules.setdefault("pyutils", pyutils)
    sys.modules.setdefault("pyutils.config", config)
    sys.modules.setdefault("pyutils.general", general)
    sys.modules.setdefault("torch_sparse", torch_sparse)


def _load_n_ports():
    """Load device_base without executing the heavy models package initializer."""
    root = Path(__file__).parents[2]
    for package_name, package_path in (
        ("core.invdes.models", root / "core" / "invdes" / "models"),
        (
            "core.invdes.models.layers",
            root / "core" / "invdes" / "models" / "layers",
        ),
    ):
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package

    fdfd = types.ModuleType("core.fdfd")
    fdfd.__path__ = [str(root / "core" / "fdfd")]

    def _unused_solver(*args, **kwargs):
        raise AssertionError("FDFD solvers are outside this test's scope")

    fdfd.fdfd_ez = _unused_solver
    fdfd.fdfd_hz = _unused_solver
    sys.modules["core.fdfd"] = fdfd

    return importlib.import_module("core.invdes.models.layers.device_base").N_Ports


_install_import_stubs()
N_Ports = _load_n_ports()

from core.utils import (
    Slice,
    build_rectilinear_grid_metadata,
    get_eigenmode_coefficients,
    get_flux,
    grid_plane_weights,
)
from thirdparty.ceviche.constants import MICRON_UNIT

pytestmark = pytest.mark.numerical


def test_grid_plane_weights_use_transverse_2d_cell_widths():
    metadata = build_rectilinear_grid_metadata(
        coords=(
            np.array([-0.5, 1.0]),
            np.array([-0.5, 1.25, 2.75]),
        ),
        boundaries=(
            np.array([-1.0, 0.0, 2.0]),
            np.array([-2.0, 1.0, 1.5, 4.0]),
        ),
    )

    torch.testing.assert_close(
        grid_plane_weights(metadata, axis=0),
        torch.tensor([3.0, 0.5, 2.5]) * MICRON_UNIT,
    )
    torch.testing.assert_close(
        grid_plane_weights(metadata, axis=1),
        torch.tensor([1.0, 2.0]) * MICRON_UNIT,
    )


def test_weighted_2d_flux_matches_legacy_uniform_flux():
    hx = torch.zeros((1, 3), dtype=torch.complex128)
    hy = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.complex128)
    ez = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.complex128)
    monitor = Slice(0, slice(None))
    grid_step = 0.25

    legacy = get_flux(
        hx,
        hy,
        ez,
        monitor=monitor,
        grid_step=grid_step,
        direction="x",
    )
    weighted = get_flux(
        hx,
        hy,
        ez,
        monitor=monitor,
        direction="x",
        cell_weights=(
            np.full(3, grid_step * MICRON_UNIT),
            np.array([grid_step * MICRON_UNIT]),
        ),
    )

    torch.testing.assert_close(legacy, weighted)


def test_weighted_2d_eigenmode_coefficients_match_legacy_uniform_case():
    hx = torch.zeros((1, 3), dtype=torch.complex128)
    hy = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.complex128)
    ez = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.complex128)
    ht_m = torch.tensor([1.0, 1.0, 1.0], dtype=torch.complex128)
    et_m = torch.tensor([1.0, 1.0, 1.0], dtype=torch.complex128)
    monitor = Slice(0, slice(None))
    grid_step = 0.25

    legacy = get_eigenmode_coefficients(
        hx,
        hy,
        ez,
        ht_m,
        et_m,
        monitor,
        grid_step=grid_step,
        direction="x",
    )
    weighted = get_eigenmode_coefficients(
        hx,
        hy,
        ez,
        ht_m,
        et_m,
        monitor,
        grid_step=grid_step,
        direction="x",
        cell_weights=(
            np.full(3, grid_step * MICRON_UNIT),
            np.array([grid_step * MICRON_UNIT]),
        ),
    )

    for legacy_value, weighted_value in zip(legacy, weighted):
        torch.testing.assert_close(legacy_value, weighted_value)


def test_2d_monitor_slice_uses_distinct_native_and_export_grids():
    device = object.__new__(N_Ports)
    device.dim = 2
    device.axes = ("x", "y")
    device.grid_info_dict = {
        "epsilon_map": {
            "coords": (
                np.array([-1.0, 0.0, 1.0]),
                np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
            ),
        },
        "export_epsilon_map": {
            "coords": (
                np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
                np.array([-2.0, 0.0, 2.0]),
            ),
        },
    }
    device.port_monitor_slices = {}
    device.port_monitor_slices_info = {}
    device.port_monitor_slices_native = {}
    device.port_monitor_slices_export = {}

    device.add_monitor_slice(
        "port",
        center=(0.0, 0.0),
        size=(0.0, 3.0),
        direction="x+",
    )

    native = device.port_monitor_slices_native["port"]
    export = device.port_monitor_slices_export["port"]
    assert native[0] == 1
    assert export[0] == 2
    assert native[1] == slice(1, 4)
    assert export[1] == slice(0, 3)
