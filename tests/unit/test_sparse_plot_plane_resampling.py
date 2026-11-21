"""Regression tests for memory-bounded sparse 3-D plot preparation."""

import numpy as np
import pytest
import torch

from core.invdes.models.layers.utils import plot_eps_field_3d
from core.utils import resample_rectilinear_plane, resample_rectilinear_tensor

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("fixed_axis", (0, 1, 2))
@pytest.mark.parametrize("mode", ("cartesian", "polar"))
@pytest.mark.parametrize("interpolation", ("linear", "nearest"))
def test_plane_resampling_matches_full_volume(
    fixed_axis: int, mode: str, interpolation: str
) -> None:
    """Thin-slab interpolation matches the corresponding dense output plane."""
    generator = torch.Generator().manual_seed(17)
    real = torch.randn((2, 4, 5, 6), generator=generator)
    imag = torch.randn((2, 4, 5, 6), generator=generator)
    values = torch.complex(real, imag)
    src_coords = (
        np.asarray([-2.0, -0.3, 0.4, 2.2]),
        np.asarray([-1.7, -0.8, 0.0, 0.9, 1.8]),
        np.asarray([-3.0, -1.1, -0.2, 0.5, 1.4, 3.2]),
    )
    dst_coords = (
        np.linspace(-2.0, 2.2, 7),
        np.linspace(-1.7, 1.8, 8),
        np.linspace(-3.0, 3.2, 9),
    )
    dst_index = (2, 4, 6)[fixed_axis]

    dense = resample_rectilinear_tensor(
        values,
        src_coords=src_coords,
        dst_coords=dst_coords,
        axes=(-3, -2, -1),
        mode=mode,
        interpolation=interpolation,
    )
    actual = resample_rectilinear_plane(
        values,
        src_coords=src_coords,
        dst_coords=dst_coords,
        fixed_axis=fixed_axis,
        dst_index=dst_index,
        axes=(-3, -2, -1),
        mode=mode,
        interpolation=interpolation,
    )
    expected = dense.select(1 + fixed_axis, dst_index)

    torch.testing.assert_close(actual, expected)


def test_nearest_resampling_preserves_material_step() -> None:
    """Nearest plot sampling does not invent a mixed interface cell."""
    values = torch.tensor([2.0, 2.0, 1.0, 1.0])
    source = (np.asarray([-0.75, -0.25, 0.25, 0.75]),)
    destination = (np.asarray([-0.6, -0.1, 0.1, 0.6]),)

    linear = resample_rectilinear_tensor(
        values,
        src_coords=source,
        dst_coords=destination,
        mode="cartesian",
    )
    nearest = resample_rectilinear_tensor(
        values,
        src_coords=source,
        dst_coords=destination,
        mode="cartesian",
        interpolation="nearest",
    )

    assert torch.any((linear > 1.0) & (linear < 2.0))
    torch.testing.assert_close(nearest, torch.tensor([2.0, 2.0, 1.0, 1.0]))


def test_plot_eps_field_3d_accepts_precomputed_planes(tmp_path) -> None:
    """The sparse plotting interface renders without dense 3-D inputs."""
    shape = (4, 5, 6)
    field = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    field = torch.complex(field, torch.zeros_like(field))
    eps = torch.linspace(2.0, 12.0, np.prod(shape)).reshape(shape)
    fixed_indices = (3, 2, 1)
    plane_slices = (
        (slice(None), slice(None), fixed_indices[0]),
        (slice(None), fixed_indices[1], slice(None)),
        (fixed_indices[2], slice(None), slice(None)),
    )
    planes = []
    for fixed_idx, plane_slice in zip(fixed_indices, plane_slices):
        field_plane = field[plane_slice]
        planes.append(
            {
                "field": field_plane,
                "total_field_abs": field_plane.abs(),
                "eps": eps[plane_slice],
                "base_eps": torch.full_like(eps[plane_slice], 2.0),
                "eps_grad": torch.ones_like(eps[plane_slice]),
                "thermal": None,
                "heat_source": None,
                "fixed_idx": fixed_idx,
            }
        )

    output = tmp_path / "sparse_planes.png"
    plot_eps_field_3d(
        None,
        "Ex",
        None,
        precomputed_planes=planes,
        grid_shape=shape,
        filepath=str(output),
        x_width=4.0,
        y_height=5.0,
        z_depth=6.0,
        NPML=(0, 0, 0),
    )

    assert output.is_file()
    assert output.stat().st_size > 0
