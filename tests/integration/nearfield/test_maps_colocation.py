"""Independent affine baseline for MAPS' documented Yee observation position."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from core.utils import yee_to_colocate_interpolate

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("is_electric", "native_offsets"),
    [
        (True, ((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5))),
        (False, ((0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0))),
    ],
    ids=["electric", "magnetic"],
)
def test_affine_yee_fields_are_observed_at_ez_position(is_electric, native_offsets):
    """An affine physical field reproduces f(i,j,k+1/2) in the interior."""
    x, y, z = torch.meshgrid(
        *(torch.arange(6, dtype=torch.float64) for _ in range(3)), indexing="ij"
    )
    # These samples follow the offsets documented by MAPS interpolate_fields.
    # Analytic affine interpolation provides an independent expected result.
    fields = torch.stack(
        [x + dx + 2 * (y + dy) + 3 * (z + dz) for dx, dy, dz in native_offsets]
    ).to(torch.complex128) * (1 + 0.25j)
    observed = yee_to_colocate_interpolate(
        fields, objects=SimpleNamespace(boundary_objects=()), is_E=is_electric
    )
    expected = ((x + 2 * y + 3 * (z + 0.5)) * (1 + 0.25j))[None].expand_as(fields)
    # Exclude boundary padding, which intentionally uses extrapolation rules.
    torch.testing.assert_close(
        observed[:, 1:-1, 1:-1, 1:-1],
        expected[:, 1:-1, 1:-1, 1:-1],
        atol=0,
        rtol=0,
    )
