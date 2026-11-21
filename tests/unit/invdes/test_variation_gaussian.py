"""Gaussian source profile contracts on the native FDTDx source implementation."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def source_profile(offset=(0.0, 0.0), waist_radius=1e-6, edges=None):
    """Evaluate a waist-plane profile on uniform or rectilinear coordinates.

    Args:
        offset: Transverse beam offsets in meters.
        waist_radius: Radius at 1/e field amplitude in meters.
        edges: Optional pair of transverse edge arrays in meters.

    Returns:
        Complex profile with shape ``(5, 5, 1)``.
    """
    import jax.numpy as jnp

    from thirdparty.fdtdx.objects.sources.linear_polarization import (
        GaussianPlaneSourceTidy3d,
    )

    source = SimpleNamespace(
        wave_character=SimpleNamespace(get_frequency=lambda: 299792458 / 1.31e-6),
        _config=SimpleNamespace(
            dtype=jnp.float32, has_nonuniform_grid=edges is not None
        ),
        grid_shape=(5, 5, 1),
        horizontal_axis=0,
        vertical_axis=1,
        propagation_axis=2,
        waist_radius=waist_radius,
        waist_distance=0.0,
        direction="-",
        beam_center_offset=offset,
        _source_resolution=lambda: 0.5e-6,
        _local_edge_coordinates=lambda: (*[jnp.asarray(axis) for axis in edges], None),
        _gaussian_beam_on_source_plane=GaussianPlaneSourceTidy3d._gaussian_beam_on_source_plane,
    )
    center = [2.0, 2.0] if edges is None else [axis[-1] / 2 for axis in edges]
    return np.asarray(
        GaussianPlaneSourceTidy3d._get_amplitude_raw(source, jnp.asarray(center))
    )


def test_nominal_gaussian_matches_analytical_profile():
    x = (np.arange(5) - 2) * 0.5
    expected = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2))
    np.testing.assert_allclose(source_profile()[..., 0], expected, rtol=2e-7, atol=1e-8)


def test_gaussian_offset_moves_beam_without_moving_plane():
    x = (np.arange(5) - 2) * 0.5
    expected = np.exp(-((x[:, None] - 0.25) ** 2 + (x[None, :] + 0.5) ** 2))
    np.testing.assert_allclose(
        source_profile((0.25e-6, -0.5e-6))[..., 0], expected, rtol=2e-7, atol=1e-8
    )


@pytest.mark.parametrize("waist_radius_um", [10.0, 100.0])
def test_broad_gaussian_is_nearly_uniform_inside_fixed_design_aperture(waist_radius_um):
    """Increasing the waist flattens a fixed four-micrometer source rectangle."""
    edges = (
        np.array([0.0, 0.2, 1.0, 2.0, 3.7, 4.0]) * 1e-6,
        np.array([0.0, 0.4, 1.2, 2.7, 3.8, 4.0]) * 1e-6,
    )
    profile = np.abs(
        source_profile(waist_radius=waist_radius_um * 1e-6, edges=edges)[..., 0]
    )
    coordinates = [0.5 * (axis[:-1] + axis[1:]) - 2e-6 for axis in edges]
    expected = np.exp(
        -(coordinates[0][:, None] ** 2 + coordinates[1][None, :] ** 2)
        / (waist_radius_um * 1e-6) ** 2
    )
    assert profile.shape == (5, 5)
    np.testing.assert_allclose(profile, expected, rtol=2e-7, atol=1e-8)
    # Cell centers lie inside the physical rectangle; its corners are the lower bound.
    assert profile.min() / profile.max() >= np.exp(-8.0 / waist_radius_um**2)


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("waist_radius_um", [2.0, 8.0])
def test_waist_radius_changes_profile_after_power_normalization(
    nonuniform, waist_radius_um
):
    """Power normalization preserves the physical Gaussian width on both grids."""
    if nonuniform:
        edges = (
            np.array([0.0, 0.3, 0.9, 1.4, 2.1, 2.5]) * 1e-6,
            np.array([0.0, 0.4, 0.8, 1.5, 1.9, 2.5]) * 1e-6,
        )
        coords = [0.5 * (axis[:-1] + axis[1:]) - axis[-1] / 2 for axis in edges]
        area = np.diff(edges[0])[:, None] * np.diff(edges[1])[None, :]
    else:
        edges = None
        coords = [(np.arange(5) - 2) * 0.5e-6] * 2
        area = np.full((5, 5), (0.5e-6) ** 2)
    radius = waist_radius_um * 1e-6
    profile = source_profile(waist_radius=radius, edges=edges)[..., 0]
    expected = np.exp(-(coords[0][:, None] ** 2 + coords[1][None, :] ** 2) / radius**2)
    np.testing.assert_allclose(profile, expected, rtol=2e-7, atol=1e-8)
    # The common impedance factor cancels for this homogeneous source plane.
    normalized = profile / np.sqrt(np.sum(np.abs(profile) ** 2 * area))
    expected /= np.sqrt(np.sum(expected**2 * area))
    np.testing.assert_allclose(normalized, expected, rtol=3e-7, atol=1e-8)
    other_radius = (8.0 if waist_radius_um == 2.0 else 2.0) * 1e-6
    other = source_profile(waist_radius=other_radius, edges=edges)[..., 0]
    other = other / np.sqrt(np.sum(np.abs(other) ** 2 * area))
    assert np.linalg.norm(normalized - other) / np.linalg.norm(normalized) > 0.02
