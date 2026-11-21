"""Deterministic regression tests for rectilinear FDFD derivatives."""

import importlib
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


def _fdfd_module(name: str):
    """Import one FDFD module without executing the heavy package initializer."""
    package = types.ModuleType("core.fdfd")
    package.__path__ = [str(Path(__file__).parents[2] / "core" / "fdfd")]
    sys.modules["core.fdfd"] = package
    return importlib.import_module(f"core.fdfd.{name}")


_derivatives = _fdfd_module("derivatives")
_rectilinear = _fdfd_module("derivatives_rectilinear")
make_uniform_Dxb = _derivatives.make_Dxb
make_uniform_Dxf = _derivatives.make_Dxf
make_uniform_Dyb = _derivatives.make_Dyb
make_uniform_Dyf = _derivatives.make_Dyf
compute_derivative_matrices = _rectilinear.compute_derivative_matrices
spacing_from_boundaries = _rectilinear.spacing_from_boundaries

DEVICE = "cpu"
DTYPE = torch.complex128


def dense(matrix):
    return matrix.to_dense().to(dtype=DTYPE, device=DEVICE)


@pytest.mark.numerical
def test_uniform_rectilinear_operators_match_uniform_implementation():
    """Uniform spacings must reproduce the existing derivative matrices."""
    shape = (4, 3)
    dL = 0.37
    dxs = np.full(shape[0], dL)
    dys = np.full(shape[1], dL)
    bloch_x = 0.23
    bloch_y = -0.41

    rectilinear = compute_derivative_matrices(
        omega=2.0,
        shape=shape,
        npml=(0, 0),
        dxs=dxs,
        dys=dys,
        bloch_x=bloch_x,
        bloch_y=bloch_y,
        device=DEVICE,
    )
    uniform = (
        make_uniform_Dxf(dL, shape, bloch_x=bloch_x, device=DEVICE),
        make_uniform_Dxb(dL, shape, bloch_x=bloch_x, device=DEVICE),
        make_uniform_Dyf(dL, shape, bloch_y=bloch_y, device=DEVICE),
        make_uniform_Dyb(dL, shape, bloch_y=bloch_y, device=DEVICE),
    )

    for rectilinear_matrix, uniform_matrix in zip(rectilinear, uniform):
        torch.testing.assert_close(dense(rectilinear_matrix), dense(uniform_matrix))


@pytest.mark.numerical
def test_nonuniform_coefficients_and_flattening():
    """Local spacing and x-major flattening must be preserved."""
    shape = (3, 2)
    dxs = np.array([1.0, 2.0, 4.0])
    dys = np.array([3.0, 5.0])
    dxf, dxb, dyf, dyb = compute_derivative_matrices(
        omega=1.0,
        shape=shape,
        npml=(0, 0),
        dxs=dxs,
        dys=dys,
        device=DEVICE,
    )

    expected_dxf = torch.tensor(
        [
            [-1, 0, 1, 0, 0, 0],
            [0, -1, 0, 1, 0, 0],
            [0, 0, -0.5, 0, 0.5, 0],
            [0, 0, 0, -0.5, 0, 0.5],
            [0.25, 0, 0, 0, -0.25, 0],
            [0, 0.25, 0, 0, 0, -0.25],
        ],
        dtype=DTYPE,
    )
    expected_dyf = torch.tensor(
        [
            [-1 / 3, 1 / 3, 0, 0, 0, 0],
            [1 / 5, -1 / 5, 0, 0, 0, 0],
            [0, 0, -1 / 3, 1 / 3, 0, 0],
            [0, 0, 1 / 5, -1 / 5, 0, 0],
            [0, 0, 0, 0, -1 / 3, 1 / 3],
            [0, 0, 0, 0, 1 / 5, -1 / 5],
        ],
        dtype=DTYPE,
    )
    torch.testing.assert_close(dense(dxf), expected_dxf)
    torch.testing.assert_close(dense(dyf), expected_dyf)

    assert dense(dxb)[0, 0].item() == pytest.approx(1 / 4)
    assert dense(dxb)[0, 4].item() == pytest.approx(-1 / 4)
    assert dense(dyb)[0, 0].item() == pytest.approx(1 / 5)
    assert dense(dyb)[0, 1].item() == pytest.approx(-1 / 5)


@pytest.mark.numerical
def test_boundaries_are_converted_to_cell_widths():
    """Boundary coordinates and explicit cell widths must produce identical operators."""
    boundaries = (
        np.array([-2.0, -1.0, 1.5, 5.0]),
        np.array([3.0, 4.25, 8.0]),
        np.array([0.0, 1.0]),
    )
    dxs, dys = spacing_from_boundaries(boundaries)
    np.testing.assert_allclose(dxs, [1.0, 2.5, 3.5])
    np.testing.assert_allclose(dys, [1.25, 3.75])

    from_boundaries = compute_derivative_matrices(
        omega=1.0,
        shape=(3, 2),
        npml=(0, 0),
        boundaries=boundaries,
        device=DEVICE,
    )
    from_spacing = compute_derivative_matrices(
        omega=1.0,
        shape=(3, 2),
        npml=(0, 0),
        dxs=dxs,
        dys=dys,
        device=DEVICE,
    )
    for boundaries_matrix, spacing_matrix in zip(from_boundaries, from_spacing):
        torch.testing.assert_close(dense(boundaries_matrix), dense(spacing_matrix))


@pytest.mark.numerical
def test_invalid_spacing_and_boundary_arguments_are_rejected():
    with pytest.raises(ValueError, match="dxs"):
        compute_derivative_matrices(
            1.0, (3, 2), (0, 0), dxs=[1.0, 2.0], dys=[1.0, 1.0], device=DEVICE
        )

    with pytest.raises(ValueError, match="either boundaries"):
        compute_derivative_matrices(
            1.0,
            (3, 2),
            (0, 0),
            dxs=[1.0, 1.0, 1.0],
            dys=[1.0, 1.0],
            boundaries=(np.arange(4), np.arange(3)),
            device=DEVICE,
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        spacing_from_boundaries((np.array([0.0, 1.0, 1.0]), np.arange(3)))


@pytest.mark.numerical
def test_rectilinear_pml_has_expected_shape_and_finite_values():
    """Rectilinear PML operators have finite values and expected dimensions."""
    derivatives = compute_derivative_matrices(
        omega=2.0,
        shape=(5, 4),
        npml=(1, 1),
        dxs=np.array([0.5, 0.75, 1.0, 1.25, 1.5]),
        dys=np.array([0.4, 0.6, 0.8, 1.0]),
        device=DEVICE,
    )
    for matrix in derivatives:
        assert matrix.shape == (20, 20)
        assert torch.isfinite(matrix.values()).all()
