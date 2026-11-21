"""Deterministic regression tests for nonuniform FDFD symmetrization."""

import importlib
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.sparse")


def _load_submodule(package_name: str, module_name: str):
    """Load a single package module without executing its heavy package initializer."""
    root = Path(__file__).parents[2]
    package_path = root.joinpath(*package_name.split("."))
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.{module_name}")


preconditioner = _load_submodule("core.fdfd", "preconditioner")
ceviche_constants = _load_submodule("thirdparty.ceviche", "constants")
ceviche_derivatives = _load_submodule("thirdparty.ceviche", "derivatives")

create_symmetrizer = preconditioner.create_symmetrizer
MU_0 = ceviche_constants.MU_0
_stretched_widths = ceviche_derivatives._stretched_widths
compute_derivative_matrices = ceviche_derivatives.compute_derivative_matrices


def _operators(pml):
    shape = (4, 5)
    dxs = np.array([1.0, 1.3, 0.8, 1.1])
    dys = np.array([0.9, 1.2, 0.7, 1.4, 1.0])
    omega = 2.3
    derivatives = compute_derivative_matrices(omega, shape, pml, dxs=dxs, dys=dys)
    Dxf, Dxb, Dyf, Dyb, _, _ = derivatives
    return shape, dxs, dys, omega, Dxf, Dxb, Dyf, Dyb


def _transpose_error(matrix):
    difference = (matrix - matrix.T).tocoo()
    return np.max(np.abs(difference.data)) if difference.nnz else 0.0


@pytest.mark.numerical
def test_nonuniform_ez_is_complex_symmetric():
    """Symmetrized nonuniform-Ez operators should be complex symmetric."""
    shape, dxs, dys, omega, Dxf, Dxb, Dyf, Dyb = _operators(((1, 1), (2, 1)))
    A = -(Dxf @ Dxb + Dyf @ Dyb) / MU_0
    Pl, Pr = create_symmetrizer(
        _stretched_widths(dxs, (1, 1), omega, dual=False),
        _stretched_widths(dys, (2, 1), omega, dual=False),
    )

    assert _transpose_error(Pl @ A @ Pr) < 1e-8


@pytest.mark.numerical
def test_nonuniform_hz_is_complex_symmetric_with_asymmetric_pml():
    """Asymmetric PML stretching should preserve the Hz symmetry contract."""
    shape, dxs, dys, omega, Dxf, Dxb, Dyf, Dyb = _operators(((1, 1), (2, 1)))
    inverse_eps = sp.diags(np.linspace(0.2, 1.0, np.prod(shape)))
    A = Dxb @ inverse_eps @ Dxf + Dyb @ inverse_eps @ Dyf
    Pl, Pr = create_symmetrizer(
        _stretched_widths(dxs, (1, 1), omega, dual=True),
        _stretched_widths(dys, (2, 1), omega, dual=True),
    )

    assert _transpose_error(Pl @ A @ Pr) < 1e-8


@pytest.mark.numerical
def test_nonuniform_preconditioned_solution_matches_original():
    """Preconditioning and recovery must preserve the original sparse solution."""
    shape, dxs, dys, omega, Dxf, Dxb, Dyf, Dyb = _operators(((0, 0), (0, 0)))
    A = -(Dxf @ Dxb + Dyf @ Dyb) / MU_0 + sp.eye(np.prod(shape))
    Pl, Pr = create_symmetrizer(dxs, dys)
    source = np.arange(np.prod(shape), dtype=np.complex128) + 1j

    reference = sp.linalg.spsolve(A, source)
    transformed = sp.linalg.spsolve(Pl @ A @ Pr, Pl @ source)
    recovered = Pr @ transformed

    np.testing.assert_allclose(recovered, reference, rtol=1e-9, atol=1e-9)
