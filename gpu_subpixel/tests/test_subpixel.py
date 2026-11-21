import numpy as np
import pytest

cp = pytest.importorskip("cupy")
td = pytest.importorskip("tidy3d")


def _available_gpu():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _available_gpu(), reason="CUDA device unavailable")


def _grid(x=(-1, 0, 1), y=(-1, 0, 1), z=(-1, 0, 1)):
    return td.Grid(boundaries=td.Coords(x=x, y=y, z=z))


def test_empty_and_box_overlap():
    from gpu_subpixel import gpu_subpixel_rasterize

    grid = _grid()
    empty = gpu_subpixel_rasterize([], grid, background_eps=2.0)
    assert empty.shape == (2, 2, 2)
    assert cp.all(empty == 2.0)
    box = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)),
        medium=td.Medium(permittivity=10),
    )
    result = gpu_subpixel_rasterize([box], grid, dtype=cp.float64)
    np.testing.assert_allclose(cp.asnumpy(result), 2.125)


def test_nonuniform_polyslab_and_fraction():
    from gpu_subpixel import gpu_subpixel_rasterize

    grid = _grid(x=(-1, -0.2, 0.4, 1), y=(-1, 0.1, 1), z=(-1, -0.25, 0.75, 1))
    triangle = td.Structure(
        geometry=td.PolySlab(
            vertices=[(-0.8, -0.8), (0.8, -0.8), (0, 0.8)], slab_bounds=(-0.25, 0.75)
        ),
        medium=td.Medium(permittivity=4),
    )
    result, fraction = gpu_subpixel_rasterize([triangle], grid, return_fraction=True)
    assert result.shape == (3, 2, 3)
    assert cp.all(fraction >= 0) and cp.all(fraction <= 1)
    assert float(cp.asnumpy(fraction).max()) > 0


def test_polygon_with_arbitrary_vertex_count():
    from gpu_subpixel import gpu_subpixel_rasterize

    angles = np.linspace(0.0, 2.0 * np.pi, 257, endpoint=False)
    vertices = np.column_stack((0.8 * np.cos(angles), 0.8 * np.sin(angles)))
    grid = _grid(x=np.linspace(-1, 1, 9), y=np.linspace(-1, 1, 9), z=(-0.5, 0.5))
    polygon = td.Structure(
        geometry=td.PolySlab(vertices=vertices, slab_bounds=(-0.5, 0.5)),
        medium=td.Medium(permittivity=4),
    )
    result = gpu_subpixel_rasterize([polygon], grid)
    assert result.shape == (8, 8, 1)
    assert float(cp.asnumpy(result).max()) == pytest.approx(4.0)


def test_sidewall_polyslab_uses_tapered_endpoint_polygons():
    from gpu_subpixel import gpu_subpixel_rasterize

    grid = _grid(x=(-2, 2), y=(-2, 2), z=(-1, 1))
    geometry = td.PolySlab(
        vertices=[(-1, -0.5), (1, -0.5), (1, 0.5), (-1, 0.5)],
        slab_bounds=(-1, 1),
        sidewall_angle=np.deg2rad(12),
        reference_plane="middle",
    )
    structure = td.Structure(geometry=geometry, medium=td.Medium(permittivity=5))
    result, fraction = gpu_subpixel_rasterize(
        [structure], grid, background_eps=1.0, dtype=cp.float64, return_fraction=True
    )

    def polygon_area(vertices):
        vertices = np.asarray(vertices)
        return 0.5 * abs(
            np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1))
            - np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1))
        )

    samples = np.linspace(-1, 1, 2001)
    areas = np.asarray(
        [
            polygon_area(
                np.asarray(geometry.base_polygon)
                + (z + 1)
                / 2
                * (np.asarray(geometry.top_polygon) - np.asarray(geometry.base_polygon))
            )
            for z in samples
        ]
    )
    expected_fraction = np.trapezoid(areas, samples) / 32.0
    actual_fraction = float(cp.asnumpy(fraction)[0, 0, 0])
    actual_value = float(cp.asnumpy(result)[0, 0, 0])
    assert actual_fraction == pytest.approx(expected_fraction, rel=1e-7)
    assert actual_value == pytest.approx(1.0 + expected_fraction * 4.0, rel=1e-7)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cylinder_conserves_volume_on_nonuniform_grid(axis):
    """Integrate the exact cylinder volume for every Cartesian extrusion axis."""
    from gpu_subpixel import gpu_subpixel_rasterize

    boundaries = (
        np.asarray([-1.0, -0.65, -0.1, 0.4, 1.0]),
        np.asarray([-1.0, -0.4, 0.2, 0.75, 1.0]),
        np.asarray([-1.0, -0.5, 0.15, 0.6, 1.0]),
    )
    grid = _grid(*boundaries)
    center = (0.08, -0.07, 0.03)
    radius = 0.61
    length = 1.2
    cylinder = td.Structure(
        geometry=td.Cylinder(center=center, axis=axis, radius=radius, length=length),
        medium=td.Medium(permittivity=5),
    )
    result, fraction = gpu_subpixel_rasterize(
        [cylinder], grid, dtype=cp.float64, return_fraction=True
    )
    cell_volumes = (
        np.diff(boundaries[0])[:, None, None]
        * np.diff(boundaries[1])[None, :, None]
        * np.diff(boundaries[2])[None, None, :]
    )
    occupied_volume = float(np.sum(cp.asnumpy(fraction) * cell_volumes))

    assert result.shape == cell_volumes.shape
    assert occupied_volume == pytest.approx(np.pi * radius**2 * length, rel=2e-12)


def test_nested_cylinder_order_overrides_inner_material():
    """Preserve Tidy3D ordering for a smaller cylinder inside a larger one."""
    from gpu_subpixel import gpu_subpixel_rasterize

    grid = _grid(
        x=(-1.0, -0.25, 0.25, 1.0),
        y=(-1.0, -0.25, 0.25, 1.0),
        z=(-0.5, 0.5),
    )
    outer = td.Structure(
        geometry=td.Cylinder(center=(0, 0, 0), radius=0.9, length=1.0),
        medium=td.Medium(permittivity=5),
    )
    inner = td.Structure(
        geometry=td.Cylinder(center=(0, 0, 0), radius=0.4, length=1.0),
        medium=td.Medium(permittivity=2),
    )
    result = cp.asnumpy(gpu_subpixel_rasterize([outer, inner], grid, dtype=cp.float64))

    assert result[1, 1, 0] == pytest.approx(2.0)
    assert result[0, 1, 0] > 1.0


def test_priority_composites_volume_fraction():
    from gpu_subpixel import gpu_subpixel_rasterize

    grid = _grid(x=(-1, 0, 1), y=(-1, 0, 1), z=(-1, 1))
    low = td.Structure(
        geometry=td.Box(center=(-0.25, 0, 0), size=(1.5, 2, 2)),
        medium=td.Medium(permittivity=4),
        priority=1,
    )
    high = td.Structure(
        geometry=td.Box(center=(0.25, 0, 0), size=(1.5, 2, 2)),
        medium=td.Medium(permittivity=9),
        priority=2,
    )
    result = cp.asnumpy(gpu_subpixel_rasterize([low, high], grid))
    assert result[0, 0, 0] == pytest.approx(6.5)
    assert result[1, 0, 0] == pytest.approx(9.0)
