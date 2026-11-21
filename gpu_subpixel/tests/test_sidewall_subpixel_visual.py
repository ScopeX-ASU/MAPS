from os import environ
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
td = pytest.importorskip("tidy3d")

"""pytest -q gpu_subpixel/tests/test_sidewall_subpixel_visual.py
"""


def _available_gpu():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _available_gpu(), reason="CUDA device unavailable")


def _build_sidewall_simulation(*, subpixel, grid, structure, monitor):
    return td.Simulation(
        center=(0, 0, 0),
        size=(4, 4, 2),
        grid_spec=td.GridSpec.from_grid(grid),
        medium=td.Medium(permittivity=1.0),
        structures=[structure],
        sources=[],
        monitors=[monitor],
        subpixel=subpixel,
        run_time=1e-15,
        boundary_spec=td.BoundarySpec.pml(x=False, y=False, z=False),
    )


def test_sidewall_subpixel_cross_section_plot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib is required for the visualization test")

    frequency = td.C_0 / 1.55
    grid = td.Grid(
        boundaries=td.Coords(
            x=np.linspace(-2.0, 2.0, 81),
            y=np.linspace(-2.0, 2.0, 81),
            z=np.linspace(-1.0, 1.0, 20),
        )
    )
    geometry = td.PolySlab(
        vertices=[(-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)],
        slab_bounds=(-1.0, 1.0),
        sidewall_angle=np.deg2rad(7.0),
        reference_plane="middle",
        axis=2,
    )
    structure = td.Structure(
        geometry=geometry,
        medium=td.Medium(permittivity=12.0),
    )
    monitor = td.PermittivityMonitor(
        center=(0, 0, 0),
        size=(4, 4, 2),
        freqs=[frequency],
        name="sidewall_epsilon",
    )

    td.config.simulation.use_local_subpixel = False
    sim_without_subpixel = _build_sidewall_simulation(
        subpixel=False, grid=grid, structure=structure, monitor=monitor
    )
    epsilon_without = np.real(sim_without_subpixel.epsilon(monitor).to_numpy())

    try:
        td.config.simulation.use_local_subpixel = True
        sim_with_subpixel = _build_sidewall_simulation(
            subpixel=True, grid=grid, structure=structure, monitor=monitor
        )
        epsilon_with = np.real(sim_with_subpixel.epsilon(monitor).to_numpy())
        subpixel_title = "subpixel=True (Tidy3D)"
        subpixel_grid = sim_with_subpixel.grid
    except Exception:
        from gpu_subpixel import gpu_subpixel_rasterize

        epsilon_with = cp.asnumpy(
            gpu_subpixel_rasterize(
                [structure],
                grid,
                background_eps=1.0,
                dtype=cp.float64,
            )
        )
        subpixel_title = "subpixel=True (local CuPy fallback)"
        subpixel_grid = grid

    difference = np.abs(epsilon_with - epsilon_without)

    assert epsilon_with.shape == epsilon_without.shape
    assert np.max(difference) > 0.0

    x_centers = np.asarray(subpixel_grid.centers.x)
    y_centers = np.asarray(subpixel_grid.centers.y)
    z_centers = np.asarray(subpixel_grid.centers.z)
    x_boundaries = np.asarray(subpixel_grid.boundaries.x)
    y_boundaries = np.asarray(subpixel_grid.boundaries.y)
    z_boundaries = np.asarray(subpixel_grid.boundaries.z)
    output_dir = Path(environ.get("GPU_SUBPIXEL_PLOT_DIR", "figs/gpu_subpixel"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sidewall_subpixel_cross_sections.png"

    figure, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    images = []
    for row, z_index in enumerate((0, len(z_centers) // 2, len(z_centers) - 1)):
        images.append(
            axes[row, 0].imshow(
                epsilon_without[:, :, z_index].T,
                origin="lower",
                extent=(x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]),
                cmap="viridis",
                vmin=1.0,
                vmax=12.0,
                aspect="equal",
            )
        )
        images.append(
            axes[row, 1].imshow(
                epsilon_with[:, :, z_index].T,
                origin="lower",
                extent=(x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]),
                cmap="viridis",
                vmin=1.0,
                vmax=12.0,
                aspect="equal",
            )
        )
        images.append(
            axes[row, 2].imshow(
                difference[:, :, z_index].T,
                origin="lower",
                extent=(x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]),
                cmap="magma",
                vmin=0.0,
                vmax=max(float(difference.max()), 1e-12),
                aspect="equal",
            )
        )
        for axis in axes[row, :]:
            axis.vlines(
                x_boundaries,
                y_boundaries[0],
                y_boundaries[-1],
                colors="gray",
                linewidth=0.35,
                alpha=0.6,
                zorder=3,
            )
            axis.hlines(
                y_boundaries,
                x_boundaries[0],
                x_boundaries[-1],
                colors="gray",
                linewidth=0.35,
                alpha=0.6,
                zorder=3,
            )
        axes[row, 0].set_ylabel(f"z = {z_centers[z_index]:.3f} um")

    axes[0, 0].set_title("subpixel=False")
    axes[0, 1].set_title(subpixel_title)
    axes[0, 2].set_title("absolute difference")
    for axis in axes.flat:
        axis.set_xlabel("x (um)")
        axis.set_ylabel(axis.get_ylabel() or "y (um)")
    figure.colorbar(
        images[0], ax=axes[:, :2], location="right", shrink=0.8, label="permittivity"
    )
    figure.colorbar(
        images[2], ax=axes[:, 2], location="right", shrink=0.8, label="|difference|"
    )
    figure.suptitle("Tapered waveguide cross-sections: sidewall subpixel comparison")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0

    y_index = int(np.argmin(np.abs(y_centers)))
    xz_output_path = output_dir / "sidewall_subpixel_xz_cross_section.png"
    xz_figure, xz_axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    xz_maps = (
        epsilon_without[:, y_index, :].T,
        epsilon_with[:, y_index, :].T,
        difference[:, y_index, :].T,
    )
    base_polygon = np.asarray(geometry.base_polygon)
    top_polygon = np.asarray(geometry.top_polygon)
    ideal_base_half_width = 0.5 * (base_polygon[:, 0].max() - base_polygon[:, 0].min())
    ideal_top_half_width = 0.5 * (top_polygon[:, 0].max() - top_polygon[:, 0].min())
    ideal_x = np.array(
        [
            -ideal_base_half_width,
            ideal_base_half_width,
            ideal_top_half_width,
            -ideal_top_half_width,
            -ideal_base_half_width,
        ]
    )
    ideal_z = np.array(
        [
            z_boundaries[0],
            z_boundaries[0],
            z_boundaries[-1],
            z_boundaries[-1],
            z_boundaries[0],
        ]
    )
    xz_titles = ("subpixel=False", subpixel_title, "absolute difference")
    for axis, values, title in zip(xz_axes, xz_maps, xz_titles):
        image = axis.imshow(
            values,
            origin="lower",
            extent=(
                x_boundaries[0],
                x_boundaries[-1],
                z_boundaries[0],
                z_boundaries[-1],
            ),
            cmap="magma" if title == "absolute difference" else "Grays",
            vmin=0.0 if title == "absolute difference" else 1.0,
            vmax=(
                max(float(difference.max()), 1e-12)
                if title == "absolute difference"
                else 12.0
            ),
            aspect="equal",
        )
        axis.vlines(
            x_boundaries,
            z_boundaries[0],
            z_boundaries[-1],
            colors="gray",
            linewidth=0.15,
            alpha=0.6,
            zorder=3,
        )
        axis.hlines(
            z_boundaries,
            x_boundaries[0],
            x_boundaries[-1],
            colors="gray",
            linewidth=0.15,
            alpha=0.6,
            zorder=3,
        )
        axis.plot(
            ideal_x,
            ideal_z,
            color="black",
            linestyle="-",
            linewidth=0.5,
            label="ideal sidewall geometry",
            zorder=4,
        )
        axis.set_title(title)
        axis.set_xlabel("x (um)")
        axis.set_ylabel("z (um)")
        axis.legend(loc="upper right", fontsize=8, framealpha=0.8)
        xz_figure.colorbar(
            image,
            ax=axis,
            shrink=0.8,
            label="|difference|" if title == "absolute difference" else "permittivity",
        )
    xz_figure.suptitle(
        f"Vertical tapered waveguide cross-section at y = {y_centers[y_index]:.3f} um"
    )
    xz_figure.savefig(xz_output_path, dpi=300)
    plt.close(xz_figure)

    assert xz_output_path.is_file()
    assert xz_output_path.stat().st_size > 0
    print(f"Saved sidewall comparison plot to {output_path}")
    print(f"Saved vertical trapezoid plot to {xz_output_path}")
