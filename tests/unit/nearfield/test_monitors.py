"""Physical plane extraction and gradient contracts for MAPS observations."""

import pytest

torch = pytest.importorskip("torch")

from core.nearfield.monitors import from_maps_fields

pytestmark = pytest.mark.unit


def _case():
    """Return an affine six-component observation with nonuniform SI edges."""
    edges = tuple(
        torch.tensor(a, dtype=torch.float64) * 1e-6
        for a in ([0, 1, 3], [0, 2, 5, 9], [0, 1, 4, 8, 13])
    )
    axes = tuple((a[1:] + a[:-1]) / 2 for a in edges)
    x, y, z = torch.meshgrid(*axes, indexing="ij")
    affine = (x + 2 * y + 3 * z) * 1e6
    fields = torch.stack([affine + 10 * c for c in range(6)], dim=0)
    fields = torch.stack((fields, 2 * fields), dim=0).to(torch.complex128)
    fields = fields * (1 + 0.3j)
    return fields, axes, edges


def _adapt(fields, axes, edges, **kwargs):
    """Apply the verified representation declarations shared by fixtures."""
    return from_maps_fields(
        fields,
        coordinates_m=axes,
        frequencies_hz=[1e14, 2e14],
        boundaries_m=edges,
        colocated=True,
        temporally_aligned=True,
        unfolded=True,
        **kwargs,
    )


@pytest.mark.parametrize("normal_axis", [0, 1, 2])
@pytest.mark.parametrize("normal_sign", [-1, 1])
def test_affine_plane_positions_fields_and_si_area(normal_axis, normal_sign):
    """Flattening retains xyz order, outward signs, and physical cell areas."""
    fields, axes, edges = _case()
    result = _adapt(
        fields, axes, edges, normal_axis=normal_axis, normal_sign=normal_sign, index=1
    )
    transverse = [a for a in range(3) if a != normal_axis]
    points = torch.cartesian_prod(*(axes[a] for a in transverse))
    expected_positions = torch.empty((len(points), 3), dtype=torch.float64)
    expected_positions[:, transverse] = points
    expected_positions[:, normal_axis] = axes[normal_axis][1]
    torch.testing.assert_close(result.positions_m, expected_positions)
    torch.testing.assert_close(
        result.normals[:, normal_axis],
        torch.full((len(points),), normal_sign, dtype=torch.float64),
    )
    assert torch.count_nonzero(result.normals[:, transverse]) == 0
    value = (expected_positions * torch.tensor([1, 2, 3])).sum(-1) * 1e6
    expected = value[None, :, None] + 10 * torch.arange(6)[None, None, :]
    expected = expected * torch.tensor([1, 2])[:, None, None] * (1 + 0.3j)
    torch.testing.assert_close(result.electric_fields, expected[..., :3])
    torch.testing.assert_close(result.magnetic_fields, expected[..., 3:])
    area = (
        torch.diff(edges[transverse[0]])[:, None]
        * torch.diff(edges[transverse[1]])[None, :]
    )
    torch.testing.assert_close(result.weights_m2, area.flatten(), atol=0, rtol=1e-15)
    assert result.electric_fields.dtype == fields.dtype
    assert result.electric_fields.device == fields.device


def test_tuple_inputs_do_not_apply_timing_or_normalization_again():
    """Complex amplitudes pass through exactly, including magnetic phase."""
    fields, axes, edges = _case()
    result = _adapt(
        tuple(fields.unbind(1)), axes, edges, normal_axis=2, normal_sign=-1, index=0
    )
    expected = fields[..., 0].flatten(2).transpose(1, 2)
    torch.testing.assert_close(
        result.electric_fields, expected[..., :3], atol=0, rtol=0
    )
    torch.testing.assert_close(
        result.magnetic_fields, expected[..., 3:], atol=0, rtol=0
    )


def test_sparse_requires_tangential_coverage_but_not_normal_components():
    """Missing zero-filled tangential samples cannot become measured zeros."""
    fields, axes, edges = _case()
    arguments = dict(normal_axis=2, normal_sign=1, index=1, sparse=True)
    with pytest.raises(ValueError, match="explicit coverage"):
        _adapt(fields, axes, edges, **arguments)
    coverage = torch.ones_like(fields, dtype=torch.bool)
    coverage[:, [2, 5]] = False
    _adapt(fields, axes, edges, coverage=coverage, **arguments)
    coverage[0, 0, 1, 1, 1] = False
    with pytest.raises(ValueError, match="tangential"):
        _adapt(fields, axes, edges, coverage=coverage, **arguments)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradient_scatter_only_changes_selected_plane(device):
    """The adapter's VJP scatters each complex cotangent without rescaling."""
    fields, axes, edges = _case()
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    fields = fields.to(device)
    fields.requires_grad_(True)
    result = _adapt(fields, axes, edges, normal_axis=1, normal_sign=-1, index=1)
    loss = result.electric_fields.real.sum() + 2 * result.magnetic_fields.imag.sum()
    loss.backward()
    expected = torch.zeros_like(fields)
    expected[:, :3, :, 1, :] = 1
    expected[:, 3:, :, 1, :] = 2j
    torch.testing.assert_close(fields.grad, expected, atol=0, rtol=0)


@pytest.mark.parametrize("declaration", ["colocated", "temporally_aligned", "unfolded"])
def test_rejects_raw_or_unverified_representations(declaration):
    """Callers must establish the observation convention before projection."""
    fields, axes, edges = _case()
    declarations = dict(colocated=True, temporally_aligned=True, unfolded=True)
    declarations[declaration] = False
    with pytest.raises(ValueError, match="must be colocated"):
        from_maps_fields(
            fields,
            frequencies_hz=[1e14, 2e14],
            coordinates_m=axes,
            boundaries_m=edges,
            normal_axis=0,
            normal_sign=1,
            index=0,
            **declarations,
        )


def test_explicit_area_preserves_boundary_independent_quadrature():
    """Explicit physical weights support verified non-cell-centered samples."""
    fields, axes, _ = _case()
    weights = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4) * 1e-12
    result = from_maps_fields(
        fields,
        frequencies_hz=[1e14, 2e14],
        coordinates_m=axes,
        weights_m2=weights,
        normal_axis=0,
        normal_sign=1,
        index=0,
        colocated=True,
        temporally_aligned=True,
        unfolded=True,
    )
    torch.testing.assert_close(result.weights_m2, weights.flatten(), atol=0, rtol=0)
