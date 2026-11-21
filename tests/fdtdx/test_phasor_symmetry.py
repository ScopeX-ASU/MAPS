"""Regression tests for dense FDTD-X phasor symmetry unfolding."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from core.utils import unfold_phasor

pytestmark = [pytest.mark.fdtdx, pytest.mark.unit]


@pytest.mark.parametrize(
    ("wall", "component_parity"),
    [
        (-1, (-1, 1, -1, 1, -1, 1)),
        (1, (1, -1, 1, -1, 1, -1)),
    ],
    ids=("pec_y", "pmc_y"),
)
def test_dense_y_unfold_and_pullback_match_component_parity(
    wall: int, component_parity: tuple[int, ...]
) -> None:
    """PEC/PMC y unfolding and its Torch pullback use component parity."""
    components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    monitor = SimpleNamespace(
        _grid_slice_tuple=((0, 2), (0, 3), (0, 4)),
        components=components,
        reduce_volume=False,
    )
    config = SimpleNamespace(has_symmetry=True, symmetry=(0, wall, 0))
    reduced = torch.arange(1 * 6 * 2 * 3 * 4, dtype=torch.float64).reshape(
        1, 6, 2, 3, 4
    )
    reduced = torch.complex(reduced, reduced / 7).requires_grad_(True)

    dense = unfold_phasor(
        reduced,
        monitor=monitor,
        components=components,
        config=config,
    )

    assert dense.shape == (1, 6, 2, 6, 4)
    parity = torch.tensor(component_parity, dtype=dense.dtype).reshape(1, 6, 1, 1, 1)
    torch.testing.assert_close(dense[:, :, :, 3:, :], reduced)
    torch.testing.assert_close(
        dense[:, :, :, :3, :],
        torch.flip(reduced, dims=(3,)) * parity,
    )

    cotangent = torch.complex(
        torch.arange(dense.numel(), dtype=torch.float64).reshape(dense.shape),
        torch.ones(dense.shape, dtype=torch.float64),
    )
    (actual_pullback,) = torch.autograd.grad(dense, reduced, cotangent)
    expected_pullback = (
        cotangent[:, :, :, 3:, :]
        + torch.flip(cotangent[:, :, :, :3, :], dims=(3,)) * parity
    )
    torch.testing.assert_close(actual_pullback, expected_pullback)
