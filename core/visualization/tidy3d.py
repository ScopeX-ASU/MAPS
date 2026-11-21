"""Tidy3D data adapters and an inline 3D field-plane viewer.

Tidy3D's Python ``Simulation.plot_3d`` currently displays geometry in a hosted
viewer and does not expose field overlays. This module uses Tidy3D's native
``FieldData``/``SimulationData`` models, then provides a small Matplotlib and
ipywidgets fallback for notebook geometry and field-plane visualization.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_VECTOR_NAMES = {"E": ("Ex", "Ey", "Ez"), "H": ("Hx", "Hy", "Hz")}
logger = logging.getLogger(__name__)


def _profile_requires_simulation(profiles: Any) -> bool:
    """Return whether a MAPS port-profile collection contains a source."""
    if not isinstance(profiles, Mapping):
        return False
    for profile in profiles.values():
        if isinstance(profile, Sequence) and profile and profile[-1] is True:
            return True
    return False


def _profile_mode_names(profiles: Any) -> tuple[str, ...]:
    """Extract distinct MAPS modal labels from normalization-profile keys."""
    if not isinstance(profiles, Mapping):
        return ()
    names = []
    for key in profiles:
        if not isinstance(key, tuple):
            continue
        for value in reversed(key):
            if isinstance(value, str):
                names.append(value)
                break
    return tuple(dict.fromkeys(names))


def _mode_index(mode_name: str) -> int:
    """Convert MAPS' one-based modal suffix to a zero-based mode index."""
    match = re.search(r"(\d+)$", mode_name)
    return max(int(match.group(1)) - 1, 0) if match else 0


def _mode_spec(td: Any, mode_names: Sequence[str]):
    """Build a Tidy3D mode specification matching MAPS mode numbering."""
    num_modes = max((_mode_index(name) for name in mode_names), default=0) + 1
    polarizations = {
        "tm" if name.startswith("E") else "te"
        for name in mode_names
        if name.startswith(("E", "H"))
    }
    kwargs = {"num_modes": num_modes}
    if len(polarizations) == 1:
        polarization = polarizations.pop().upper()
        kwargs["sort_spec"] = td.ModeSortSpec(
            filter_key=f"{polarization}_fraction",
            filter_reference=0.5,
            filter_order="over",
            keep_modes="all",
            sort_key="n_eff",
            sort_order="descending",
        )
    return td.ModeSpec(**kwargs)


def _source_direction(direction: Any) -> str:
    """Convert a MAPS axis/direction label to Tidy3D's propagation sign.

    MAPS port metadata may use an unsigned normal axis such as ``"x"`` or a
    signed direction such as ``"x+"``. Unsigned port sources follow MAPS'
    positive-direction launch convention.
    """
    direction = str(direction)
    if direction.endswith(("+", "-")):
        return direction[-1]
    if direction in {"x", "y", "z"}:
        return "+"
    raise ValueError(
        f"Unsupported MAPS source direction {direction!r}; expected x/y/z "
        "with an optional '+' or '-' suffix"
    )


def materialize_port_visualization_objects(
    device: Any,
    *,
    frequency: float | None = None,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Convert physical MAPS source and modal-monitor planes to Tidy3D objects.

    Args:
        device: MAPS device after ``init_monitors`` and ``norm_run``. Its port
            slice metadata supplies physical centers and sizes, while its
            normalized profile registry distinguishes sources from monitors.
        frequency: Visualization frequency in Hz. By default it is derived
            from ``device.sim_cfg['wl_cen']``, whose unit is micrometers.

    Returns:
        ``(sources, monitors)`` tuples containing native ``td.ModeSource`` and
        ``td.ModeMonitor`` objects.

    Notes:
        Only names shared by ``port_monitor_slices_info`` and
        ``port_sources_dict`` are exported. Auxiliary ``*_field_monitor``
        profiles, design-region detectors, and full-domain phasor detectors
        are therefore intentionally omitted. The objects preserve visualization
        metadata and are not intended to turn this artifact into a runnable
        replacement for the MAPS simulation.
    """
    import tidy3d as td

    if frequency is None:
        wavelength_um = _real_scalar(device.sim_cfg["wl_cen"], "wl_cen")
        if wavelength_um <= 0:
            raise ValueError("wl_cen must be positive")
        frequency = float(td.C_0 / wavelength_um)
    frequency = _real_scalar(frequency, "frequency")
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    slice_info = getattr(device, "port_monitor_slices_info", {})
    profile_registry = getattr(device, "port_sources_dict", {})
    if not isinstance(slice_info, Mapping) or not isinstance(profile_registry, Mapping):
        return (), ()

    sources = []
    monitors = []
    for plane_name in slice_info:
        if plane_name not in profile_registry:
            continue
        if plane_name.endswith("_field_monitor"):
            continue
        info = slice_info[plane_name]
        if not isinstance(info, Mapping):
            continue
        center = tuple(float(value) for value in info.get("center", ()))
        size = tuple(float(value) for value in info.get("size", ()))
        if len(center) != 3 or len(size) != 3:
            continue
        zero_axes = sum(np.isclose(value, 0.0) for value in size)
        if zero_axes != 1 or any(value < 0 for value in size):
            logger.warning(
                "Skipping port visualization plane %r because size %r is not a "
                "3-D codimension-one plane",
                plane_name,
                size,
            )
            continue

        profiles = profile_registry[plane_name]
        mode_names = _profile_mode_names(profiles)
        if _profile_requires_simulation(profiles):
            # A separate native source represents each requested MAPS mode.
            # Repeated wavelength entries share one broadband source frame.
            for mode_name in mode_names or ("mode1",):
                sources.append(
                    td.ModeSource(
                        center=center,
                        size=size,
                        source_time=td.GaussianPulse(
                            freq0=frequency,
                            fwidth=frequency / 10,
                        ),
                        direction=_source_direction(info.get("direction", "+")),
                        mode_spec=_mode_spec(td, (mode_name,)),
                        mode_index=_mode_index(mode_name),
                        name=f"{plane_name}_{mode_name}_visualization_source",
                    )
                )
        else:
            monitors.append(
                td.ModeMonitor(
                    center=center,
                    size=size,
                    freqs=(frequency,),
                    mode_spec=_mode_spec(td, mode_names),
                    name=f"{plane_name}_visualization_monitor",
                )
            )
    return tuple(sources), tuple(monitors)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _field_mapping(fields: Mapping[str, Any] | Sequence[Any]) -> dict[str, np.ndarray]:
    if isinstance(fields, Mapping):
        missing = set(_FIELD_NAMES) - set(fields)
        if missing:
            raise ValueError(f"Missing field components: {sorted(missing)}")
        return {name: _numpy(fields[name]) for name in _FIELD_NAMES}
    if len(fields) != len(_FIELD_NAMES):
        raise ValueError("fields must contain Ex, Ey, Ez, Hx, Hy, and Hz")
    return {name: _numpy(value) for name, value in zip(_FIELD_NAMES, fields)}


def _real_scalar(value: Any, name: str) -> float:
    """Return one finite real scalar from tensor- or array-like input."""
    array = _numpy(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    scalar = float(np.real(array.reshape(())))
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _region_mask_slice(mask: Any, axis: int, axis_name: str) -> slice:
    """Resolve one contiguous design-region slice from MAPS mask metadata."""
    value = getattr(mask, axis_name, None)
    if value is None:
        try:
            value = mask[axis]
        except (IndexError, TypeError):
            value = None
    if not isinstance(value, slice):
        raise TypeError(
            f"Design-region mask axis {axis_name!r} must be a slice, got {value!r}"
        )
    if value.start is None or value.stop is None or value.step not in (None, 1):
        raise ValueError(
            f"Design-region mask axis {axis_name!r} must be a contiguous bounded slice"
        )
    return slice(int(value.start), int(value.stop))


def materialize_design_region_polyslabs(
    device: Any,
    eps_map: Any,
    *,
    region_name: str | None = None,
    threshold: float | None = None,
    frequency: float | None = None,
    sidewall_angle: float = 0.0,
    pixel_exact: bool = True,
) -> tuple[Any, ...]:
    """Convert one filled planar MAPS design region to Tidy3D geometry.

    Args:
        device: MAPS device providing design-region configuration, masks, and
            optical-grid metadata. For an inverse-design model, pass
            ``optimization.hr_device`` together with its ``hr_eps_map``.
        eps_map: Filled 2-D or 3-D relative-permittivity map returned by
            ``build_device``. Torch tensors are detached for visualization.
        region_name: Design-region key. It may be omitted only when the device
            contains exactly one design region.
        threshold: Relative-permittivity threshold. By default, the physical
            midpoint between the region's ``eps_bg`` and ``eps`` is used.
        frequency: Tidy3D sampling frequency in Hz. By default it is derived
            from the device's center wavelength, stored in micrometers.
        sidewall_angle: Extrusion sidewall angle in degrees. This matches MAPS
            device and extrusion configuration; Tidy3D receives radians.
        pixel_exact: Whether Tidy3D should retain exact thresholded grid cells
            instead of interpolating the contour between cell centers.

    Returns:
        Tuple containing a visualization-only ``td.Structure``. Its geometry
        is a ``td.PolySlab`` for a simple polygon and may be a Tidy3D
        ``GeometryGroup`` or ``ClipOperation`` for disconnected shapes or
        polygons with holes.

    Raises:
        ValueError: If the design region, grid, map shape, threshold, or planar
            slab definition is invalid.
        RuntimeError: If required MAPS grid or mask metadata is unavailable.

    Notes:
        This function does not mutate the device or the FDTD material map. It
        uses Tidy3D's public GDS geometry conversion so holes and islands are
        represented as composite geometry instead of invalid self-intersecting
        ``PolySlab`` outlines. The function name is retained for API
        compatibility with the original simple-polygon implementation.
    """
    import tidy3d as td

    region_cfgs = getattr(device, "design_region_cfgs", {})
    if region_name is None:
        if len(region_cfgs) != 1:
            raise ValueError(
                "region_name is required unless the device has exactly one design region"
            )
        region_name = next(iter(region_cfgs))
    if region_name not in region_cfgs:
        raise ValueError(f"Unknown design region {region_name!r}")
    region = region_cfgs[region_name]
    center = np.asarray(region["center"], dtype=float)
    size = np.asarray(region["size"], dtype=float)
    if center.size != 3 or size.size != 3 or np.any(size <= 0):
        raise ValueError("PolySlab materialization requires a finite 3-D design region")

    grid = getattr(device, "optical_grid_metadata", None)
    if grid is None:
        grid = getattr(device, "grid_info_dict", {}).get("epsilon_map")
    if grid is None or "coords" not in grid or "boundaries" not in grid:
        raise RuntimeError("Design-region optical grid coordinates are unavailable")
    coords = tuple(np.asarray(axis, dtype=float) for axis in grid["coords"])
    boundaries = tuple(np.asarray(axis, dtype=float) for axis in grid["boundaries"])
    if len(coords) != 3 or len(boundaries) != 3:
        raise ValueError("PolySlab materialization requires a 3-D optical grid")

    values = _numpy(eps_map)
    if np.iscomplexobj(values):
        values = values.real
    if values.ndim not in (2, 3):
        raise ValueError(f"eps_map must be 2-D or 3-D, got shape {values.shape}")
    expected_shape = tuple(len(axis) for axis in coords[: values.ndim])
    if values.shape != expected_shape:
        raise ValueError(
            f"eps_map shape {values.shape} does not match optical grid {expected_shape}"
        )

    masks = getattr(device, "design_region_masks", {})
    mask = masks.get(region_name)
    if mask is None:
        raise RuntimeError(f"Design-region mask is unavailable for {region_name!r}")
    x_slice = _region_mask_slice(mask, 0, "x")
    y_slice = _region_mask_slice(mask, 1, "y")
    if x_slice.stop > len(coords[0]) or y_slice.stop > len(coords[1]):
        raise ValueError("Design-region mask lies outside the optical grid")

    if values.ndim == 3:
        z_value = getattr(mask, "z", None)
        if isinstance(z_value, slice):
            z_slice = _region_mask_slice(mask, 2, "z")
            z_index = (z_slice.start + z_slice.stop - 1) // 2
        else:
            z_index = int(np.argmin(np.abs(coords[2] - center[2])))
        design_values = values[x_slice, y_slice, z_index]
    else:
        design_values = values[x_slice, y_slice]

    x = coords[0][x_slice]
    y = coords[1][y_slice]
    x_bounds = boundaries[0][x_slice.start : x_slice.stop + 1]
    y_bounds = boundaries[1][y_slice.start : y_slice.stop + 1]
    if len(x_bounds) != len(x) + 1 or len(y_bounds) != len(y) + 1:
        raise ValueError("Design-region boundary arrays do not match its cell centers")

    eps_bg = _real_scalar(region.get("eps_bg", device.eps_bg), "eps_bg")
    if "eps" not in region:
        raise ValueError(f"Design region {region_name!r} has no 'eps' material value")
    eps_high = _real_scalar(region["eps"], "eps")
    threshold = (
        0.5 * (eps_bg + eps_high)
        if threshold is None
        else _real_scalar(threshold, "threshold")
    )
    if not min(eps_bg, eps_high) < threshold < max(eps_bg, eps_high):
        raise ValueError("threshold must lie strictly between eps_bg and eps")
    if not np.isfinite(sidewall_angle) or not -90.0 <= sidewall_angle <= 90.0:
        raise ValueError("sidewall_angle must be finite and within [-90, 90] degrees")

    if frequency is None:
        wavelength_um = _real_scalar(device.sim_cfg["wl_cen"], "wl_cen")
        if wavelength_um <= 0:
            raise ValueError("wl_cen must be positive")
        frequency = float(td.C_0 / wavelength_um)
    frequency = _real_scalar(frequency, "frequency")
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    middle_z = float(center[2])
    scalar = td.ScalarFieldDataArray(
        np.asarray(design_values, dtype=float)[:, :, None, None],
        coords={"x": x, "y": y, "z": [middle_z], "f": [frequency]},
    )
    dataset = td.PermittivityDataset(eps_xx=scalar, eps_yy=scalar, eps_zz=scalar)
    custom_medium = td.CustomMedium(eps_dataset=dataset)
    region_box = td.Structure(
        geometry=td.Box(
            center=(
                float((x_bounds[0] + x_bounds[-1]) / 2),
                float((y_bounds[0] + y_bounds[-1]) / 2),
                middle_z,
            ),
            size=(
                float(x_bounds[-1] - x_bounds[0]),
                float(y_bounds[-1] - y_bounds[0]),
                float(size[2]),
            ),
        ),
        medium=custom_medium,
    )
    contour_simulation = td.Simulation(
        center=region_box.geometry.center,
        size=region_box.geometry.size,
        grid_spec=td.GridSpec.uniform(
            dl=float(
                min(
                    np.min(np.diff(x_bounds)),
                    np.min(np.diff(y_bounds)),
                    size[2],
                )
            )
        ),
        medium=td.Medium(permittivity=eps_bg),
        structures=(region_box,),
        sources=(),
        monitors=(),
        run_time=1e-15,
        boundary_spec=td.BoundarySpec.pml(x=False, y=False, z=False),
    )
    if pixel_exact:
        # Tidy3D's pixel-exact conversion reconstructs pixel edges from sample
        # centers. That loses authoritative MAPS boundaries on a nonuniform
        # grid, so union the thresholded physical cells using Tidy3D's gdstk
        # dependency instead.
        import gdstk

        selected = (
            design_values >= threshold
            if eps_high > eps_bg
            else design_values <= threshold
        )
        cells = [
            gdstk.rectangle(
                (float(x_bounds[ix]), float(y_bounds[iy])),
                (float(x_bounds[ix + 1]), float(y_bounds[iy + 1])),
            )
            for ix, iy in np.argwhere(selected)
        ]
        polygons = [] if not cells else gdstk.boolean(cells, [], "or", precision=1e-9)
    else:
        polygons = contour_simulation.to_gdstk(
            z=middle_z,
            frequency=frequency,
            permittivity_threshold=threshold,
            pixel_exact=False,
        )
    slab_bounds = (middle_z - float(size[2]) / 2, middle_z + float(size[2]) / 2)
    valid_polygons = [
        polygon
        for polygon in polygons
        if np.asarray(polygon.points).ndim == 2
        and np.asarray(polygon.points).shape[0] >= 3
        and np.asarray(polygon.points).shape[1] == 2
    ]
    if not valid_polygons:
        return ()

    # A GDS polygon containing a hole is encoded using a zero-width bridge,
    # which looks self-intersecting to PolySlab's validator. Tidy3D's public
    # GDS importer repairs that topology with Shapely and expresses holes as
    # ClipOperation differences. It also applies the opposite sidewall angle
    # to an inner boundary, preserving the intended extruded solid.
    import gdstk

    gds_cell = gdstk.Cell(f"maps_{region_name}_visualization")
    gds_cell.add(*valid_polygons)
    geometry_kwargs = dict(
        gds_cell=gds_cell,
        axis=2,
        slab_bounds=slab_bounds,
        gds_layer=0,
        gds_dtype=0,
        reference_plane="middle",
        merge_adjacent=True,
    )
    sidewall_angle_rad = np.deg2rad(float(sidewall_angle))
    used_sidewall_angle = float(sidewall_angle)
    try:
        geometry = td.Geometry.from_gds(
            **geometry_kwargs,
            sidewall_angle=sidewall_angle_rad,
        )
    except (td.exceptions.Tidy3dError, ValueError) as error:
        if np.isclose(sidewall_angle_rad, 0.0):
            raise
        # Tidy3D ComplexPolySlab can reject valid pixelated contours when a
        # sidewall offset causes narrow features to split or disappear. A
        # vertical extrusion preserves the authoritative thresholded footprint
        # and its holes. This artifact is for visualization only, so prefer an
        # explicit, logged approximation over losing the completed run's
        # geometry artifact or inventing per-pixel angled walls.
        logger.warning(
            "Tidy3D could not extrude design region %r at %.6g degrees; "
            "exporting the same footprint with vertical sidewalls instead: %s",
            region_name,
            sidewall_angle,
            error,
        )
        geometry = td.Geometry.from_gds(
            **geometry_kwargs,
            sidewall_angle=0.0,
        )
        used_sidewall_angle = 0.0
    return (
        td.Structure(
            geometry=geometry,
            medium=td.Medium(permittivity=eps_high),
            name=(
                f"{region_name}_visualization"
                if used_sidewall_angle == float(sidewall_angle)
                else f"{region_name}_visualization_vertical_sidewall_fallback"
            ),
        ),
    )


def build_device_visualization_simulation(
    device: Any,
    eps_map: Any | None = None,
    *,
    port_device: Any | None = None,
    region_name: str | None = None,
    threshold: float | None = None,
    frequency: float | None = None,
    sidewall_angle: float = 0.0,
    pixel_exact: bool = True,
):
    """Build a Tidy3D visualization simulation for a MAPS device.

    Args:
        device: MAPS device containing fixed Tidy3D structures.
        eps_map: Optional filled permittivity map. When supplied, its selected
            design region is converted to explicit visualization geometry.
        port_device: Optional MAPS device containing initialized physical port
            source/monitor metadata. Defaults to ``device``. This supports a
            high-resolution geometry device paired with simulation-resolution
            port normalization metadata.
        region_name: Optional design-region key. When omitted, every configured
            design region is materialized.
        threshold: Optional physical permittivity threshold.
        frequency: Optional contour sampling frequency in Hz.
        sidewall_angle: Design-region sidewall angle in degrees.
        pixel_exact: Whether extracted polygons retain exact thresholded cells.

    Returns:
        A lightweight ``td.Simulation`` suitable for ``simulation.plot_3d()``.

    Notes:
        The returned simulation is visualization-only and does not replace the
        FDTD simulation or mutate its geometry/material state.
    """
    import tidy3d as td

    structures = [
        structure
        for structure in getattr(device, "geometry", {}).values()
        if isinstance(structure, td.Structure)
    ]
    if eps_map is not None:
        region_names = (
            (region_name,)
            if region_name is not None
            else tuple(getattr(device, "design_region_cfgs", {}))
        )
        for selected_region in region_names:
            structures.extend(
                materialize_design_region_polyslabs(
                    device,
                    eps_map,
                    region_name=selected_region,
                    threshold=threshold,
                    frequency=frequency,
                    sidewall_angle=sidewall_angle,
                    pixel_exact=pixel_exact,
                )
            )

    center = tuple(float(value) for value in device.cell_center)
    size = tuple(float(value) for value in device.cell_size)
    if len(center) != 3 or len(size) != 3:
        raise ValueError("3-D visualization requires a 3-D MAPS device")
    sources, monitors = materialize_port_visualization_objects(
        device if port_device is None else port_device,
        frequency=frequency,
    )
    return td.Simulation(
        center=center,
        size=size,
        grid_spec=td.GridSpec.uniform(dl=float(device.grid_step)),
        medium=td.Medium(permittivity=_real_scalar(device.eps_bg, "eps_bg")),
        structures=tuple(structures),
        sources=sources,
        monitors=monitors,
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.pml(x=False, y=False, z=False),
    )


def load_device_visualization_simulation(path: Any):
    """Load a saved Tidy3D device-visualization artifact.

    Args:
        path: HDF5 file previously written by
            ``BaseOptimization.save_tidy3d_geometry`` or ``Simulation.to_file``.

    Returns:
        A native ``tidy3d.Simulation`` ready for ``plot_3d()``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    from pathlib import Path

    import tidy3d as td

    artifact = Path(path).expanduser()
    if not artifact.is_file():
        raise FileNotFoundError(f"Tidy3D visualization artifact not found: {artifact}")
    return td.Simulation.from_file(artifact)


def plot_tidy3d_slices(
    simulation: Any,
    *,
    positions: Sequence[float] | None = None,
    num_steps: int = 200,
    source_alpha: float | None = None,
    monitor_alpha: float | None = None,
    fill_structures: bool = True,
):
    """Display three independently movable orthogonal Tidy3D slices.

    Args:
        simulation: Tidy3D ``Simulation`` to slice. Geometry, sources, and
            monitors are rendered through its native :meth:`plot` method.
        positions: Optional initial ``(x, y, z)`` positions in micrometers.
            The simulation center is used by default.
        num_steps: Number of slider intervals across each simulation axis.
        source_alpha: Optional opacity forwarded to Tidy3D source rendering.
        monitor_alpha: Optional opacity forwarded to Tidy3D monitor rendering.
        fill_structures: Whether Tidy3D should fill structure cross-sections.

    Returns:
        An ``ipywidgets.VBox`` containing x/y/z sliders, per-structure and
        source/monitor visibility controls, and three Matplotlib
        cross-sections: YZ at x, XZ at y, and XY at z.

    Raises:
        ImportError: If Matplotlib or ipywidgets is unavailable.
        ValueError: If the simulation bounds, positions, or ``num_steps`` are
            invalid.

    Notes:
        Slider redraws occur on release rather than continuously while
        dragging. This helper controls local 2-D plots; Tidy3D's hosted
        ``plot_3d`` iframe does not expose Python-controlled clipping planes.
    """
    try:
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "Interactive slice plotting requires matplotlib and ipywidgets; "
            "install MAPS[notebook-visualization]."
        ) from exc

    center = np.asarray(simulation.center, dtype=float)
    size = np.asarray(simulation.size, dtype=float)
    if center.shape != (3,) or size.shape != (3,):
        raise ValueError("simulation center and size must each contain x, y, z")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(size)):
        raise ValueError("simulation center and size must be finite")
    if np.any(size <= 0):
        raise ValueError("simulation size must be positive along x, y, and z")
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")

    initial = center if positions is None else np.asarray(positions, dtype=float)
    if initial.shape != (3,) or not np.all(np.isfinite(initial)):
        raise ValueError("positions must contain three finite x, y, z values")
    lower = center - size / 2
    upper = center + size / 2
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("positions must lie within the simulation bounds")

    sliders = tuple(
        widgets.FloatSlider(
            description=f"{axis} (µm)",
            min=float(axis_lower),
            max=float(axis_upper),
            step=float(axis_size / num_steps),
            value=float(value),
            continuous_update=False,
            readout_format=".3f",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="31%"),
        )
        for axis, axis_lower, axis_upper, axis_size, value in zip(
            "xyz", lower, upper, size, initial
        )
    )

    structures = tuple(getattr(simulation, "structures", ()))
    sources = tuple(getattr(simulation, "sources", ()))
    monitors = tuple(getattr(simulation, "monitors", ()))
    structure_toggles = tuple(
        widgets.Checkbox(
            value=True,
            description=(
                f"{index}: {structure.name}"
                if getattr(structure, "name", None)
                else f"{index}: {structure.geometry.__class__.__name__}"
            ),
            indent=False,
            layout=widgets.Layout(width="auto"),
        )
        for index, structure in enumerate(structures)
    )
    sources_toggle = widgets.Checkbox(
        value=True,
        description=f"Sources ({len(sources)})",
        disabled=not sources,
        indent=False,
    )
    monitors_toggle = widgets.Checkbox(
        value=True,
        description=f"Monitors ({len(monitors)})",
        disabled=not monitors,
        indent=False,
    )
    structure_grid = widgets.GridBox(
        structure_toggles,
        layout=widgets.Layout(
            grid_template_columns="repeat(3, minmax(180px, 1fr))",
            grid_gap="2px 12px",
            width="100%",
        ),
    )
    visibility_controls = widgets.VBox(
        (
            widgets.HBox((sources_toggle, monitors_toggle)),
            structure_grid,
        )
    )
    visibility = widgets.Accordion(children=(visibility_controls,))
    visibility.set_title(0, "Visibility")

    # Create one canvas for the lifetime of the widget. Reusing its axes avoids
    # both ipympl's automatic display of new canvases and accumulation in the
    # notebook output when a slider changes repeatedly.
    with plt.ioff():
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    output = widgets.Output()
    widget_canvas = isinstance(fig.canvas, widgets.Widget)
    canvas_displayed = False

    def _draw() -> None:
        nonlocal canvas_displayed
        x, y, z = (float(slider.value) for slider in sliders)
        for axis in axes:
            axis.clear()
        plot_kwargs = {
            "source_alpha": source_alpha,
            "monitor_alpha": monitor_alpha,
            "fill_structures": fill_structures,
        }
        visible_structures = tuple(
            structure
            for structure, toggle in zip(structures, structure_toggles)
            if toggle.value
        )
        visible_sources = sources if sources_toggle.value else ()
        visible_monitors = monitors if monitors_toggle.value else ()
        if (
            visible_structures == structures
            and visible_sources == sources
            and visible_monitors == monitors
        ):
            visible_simulation = simulation
        else:
            visible_simulation = simulation.updated_copy(
                structures=visible_structures,
                sources=visible_sources,
                monitors=visible_monitors,
            )
        visible_simulation.plot(x=x, ax=axes[0], **plot_kwargs)
        visible_simulation.plot(y=y, ax=axes[1], **plot_kwargs)
        visible_simulation.plot(z=z, ax=axes[2], **plot_kwargs)
        axes[0].set_title(f"YZ at x={x:.3f} µm")
        axes[1].set_title(f"XZ at y={y:.3f} µm")
        axes[2].set_title(f"XY at z={z:.3f} µm")
        fig.canvas.draw_idle()
        with output:
            if widget_canvas:
                if not canvas_displayed:
                    display(fig.canvas)
                    canvas_displayed = True
            else:
                output.clear_output(wait=True)
                display(fig)

    def _control_changed(change: Mapping[str, Any]) -> None:
        if change.get("name") == "value":
            _draw()

    for control in (*sliders, *structure_toggles, sources_toggle, monitors_toggle):
        control.observe(_control_changed, names="value")
    _draw()
    return widgets.VBox(
        (
            widgets.HBox(sliders, layout=widgets.Layout(width="100%")),
            visibility,
            output,
        )
    )


def maps_fields_to_tidy3d_field_data(
    fields: Mapping[str, Any] | Sequence[Any],
    *,
    coords: Sequence[np.ndarray],
    frequencies: Sequence[float],
    boundaries: Sequence[np.ndarray] | None = None,
    name: str = "maps_fields",
):
    """Convert MAPS phasors on cell centers into Tidy3D ``FieldData``.

    Args:
        fields: Mapping of six component names to arrays or the six-tensor tuple
            returned by ``fdtd3d.solve``. Each array has shape ``(f, x, y, z)``.
        coords: Physical cell-center coordinates in x, y, z order, in micrometers.
        frequencies: Frequency samples in Hz, one per leading field dimension.
        boundaries: Optional physical grid boundaries in micrometers. When supplied,
            they are stored as the Tidy3D expanded grid for accurate cell bounds.
        name: Name of the metadata-only Tidy3D field monitor.

    Returns:
        Tidy3D ``FieldData`` with each component stored in ``(x, y, z, f)`` order.

    Notes:
        ``fdtd3d.solve`` already applies temporal correction, Yee colocation,
        symmetry unfolding, and optional export-grid resampling. No offsets or
        interpolation are applied here. The Tidy3D monitor is metadata; its
        flux/integration semantics are not validated for these cell-centered
        MAPS samples.
    """
    import tidy3d as td

    arrays = _field_mapping(fields)
    axes = tuple(np.asarray(axis, dtype=float) for axis in coords)
    freq = np.asarray(frequencies, dtype=float)
    if len(axes) != 3 or any(axis.ndim != 1 for axis in axes):
        raise ValueError("coords must contain three one-dimensional axes")
    expected = (len(freq), *(len(axis) for axis in axes))
    for component, values in arrays.items():
        if values.shape != expected:
            raise ValueError(
                f"{component} has shape {values.shape}; expected {expected}"
            )
    if (
        freq.ndim != 1
        or freq.size == 0
        or not np.all(np.isfinite(freq))
        or np.any(freq <= 0)
    ):
        raise ValueError("frequencies must be a non-empty finite positive 1D sequence")
    if any(
        axis.size == 0
        or not np.all(np.isfinite(axis))
        or (axis.size > 1 and np.any(np.diff(axis) <= 0))
        for axis in axes
    ):
        raise ValueError("coordinate axes must be non-empty, finite, and increasing")

    edge_axes = None
    if boundaries is not None:
        edge_axes = tuple(np.asarray(axis, dtype=float) for axis in boundaries)
        if len(edge_axes) != 3 or any(
            edges.ndim != 1
            or len(edges) != len(centers) + 1
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0)
            or centers[0] < edges[0]
            or centers[-1] > edges[-1]
            for edges, centers in zip(edge_axes, axes)
        ):
            raise ValueError(
                "each boundaries axis must be finite, increasing, and one value longer than coords"
            )

    coords_dict = {axis: values for axis, values in zip("xyz", axes)}
    coords_dict["f"] = freq
    scalar_arrays = {
        name: td.ScalarFieldDataArray(
            np.transpose(values, (1, 2, 3, 0)), coords=coords_dict
        )
        for name, values in arrays.items()
    }
    monitor_axes = edge_axes if edge_axes is not None else axes
    center = tuple(float((axis[0] + axis[-1]) / 2) for axis in monitor_axes)
    size = tuple(float(axis[-1] - axis[0]) for axis in monitor_axes)
    monitor = td.FieldMonitor(
        center=center,
        size=size,
        freqs=freq.tolist(),
        fields=list(_FIELD_NAMES),
        name=name,
        colocate=True,
    )
    kwargs: dict[str, Any] = {**scalar_arrays, "monitor": monitor}
    if edge_axes is not None:
        kwargs["grid_expanded"] = td.Grid(
            boundaries=td.Coords(x=edge_axes[0], y=edge_axes[1], z=edge_axes[2])
        )
    return td.FieldData(**kwargs)


def maps_fields_to_tidy3d_data(
    simulation: Any,
    fields: Mapping[str, Any] | Sequence[Any],
    *,
    coords: Sequence[np.ndarray],
    frequencies: Sequence[float],
    boundaries: Sequence[np.ndarray] | None = None,
):
    """Build native Tidy3D ``SimulationData`` around MAPS field phasors.

    Args:
        simulation: Existing Tidy3D ``Simulation`` carrying the desired geometry.
        fields: MAPS component mapping or six-tensor ``fdtd3d.solve`` result.
        coords: Cell-center coordinates in micrometers, in x, y, z order.
        frequencies: Frequencies in Hz.
        boundaries: Optional cell boundaries in micrometers.

    Returns:
        Tidy3D ``SimulationData`` containing the supplied simulation and fields.
    """
    import tidy3d as td

    field_data = maps_fields_to_tidy3d_field_data(
        fields, coords=coords, frequencies=frequencies, boundaries=boundaries
    )
    monitor_names = {monitor.name for monitor in simulation.monitors}
    if field_data.monitor.name not in monitor_names:
        simulation = simulation.updated_copy(
            monitors=(*simulation.monitors, field_data.monitor)
        )
    return td.SimulationData(simulation=simulation, data=(field_data,))


def select_field_quantity(
    field_data: Any, field: str, quantity: str, frequency: float
) -> np.ndarray:
    """Select a scalar field quantity at one stored frequency.

    Args:
        field_data: Tidy3D ``FieldData`` created by this adapter.
        field: ``Ex``/``Ey``/``Ez``/``Hx``/``Hy``/``Hz``, ``E``, or ``H``.
        quantity: One of ``real``, ``imag``, ``abs``, ``abs2``, or ``phase``.
        frequency: Frequency in Hz; the nearest stored sample is selected.

    Returns:
        Real-valued NumPy array in x, y, z axis order.
    """
    name = field if field in _FIELD_NAMES else field.upper()
    if name in _VECTOR_NAMES:
        parts = [
            np.asarray(
                getattr(field_data, component).sel(f=frequency, method="nearest")
            )
            for component in _VECTOR_NAMES[name]
        ]
        scalar = np.sqrt(sum(np.abs(part) ** 2 for part in parts))
        if quantity in ("abs", "abs2"):
            return scalar**2 if quantity == "abs2" else scalar
        raise ValueError("vector fields support only 'abs' and 'abs2' quantities")
    if name not in _FIELD_NAMES:
        raise ValueError(f"Unsupported field {field!r}")
    values = np.asarray(getattr(field_data, name).sel(f=frequency, method="nearest"))
    choices = {
        "real": np.real,
        "imag": np.imag,
        "abs": np.abs,
        "abs2": lambda value: np.abs(value) ** 2,
        "phase": np.angle,
    }
    if quantity not in choices:
        raise ValueError(f"Unsupported quantity {quantity!r}")
    return choices[quantity](values)


def _colorize_field(layer: np.ndarray, colormap: Any) -> np.ndarray:
    """Map real field values to RGBA while handling constant or invalid slices."""
    from matplotlib.colors import Normalize

    finite = layer[np.isfinite(layer)]
    if finite.size:
        low = float(finite.min())
        high = float(finite.max())
        if low == high:
            high = low + max(abs(low), 1.0) * 1e-12
        color_scale = Normalize(low, high)
        color_layer = np.where(np.isfinite(layer), layer, low)
    else:
        color_scale = Normalize(0.0, 1.0)
        color_layer = np.zeros_like(layer, dtype=float)
    return colormap(color_scale(color_layer))


def _structure_vertices(structure: Any):
    """Yield simple display meshes for Tidy3D boxes and polygonal slabs."""
    import tidy3d as td
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    geometry = structure.geometry
    if isinstance(geometry, td.Box):
        low = np.asarray(geometry.center) - np.asarray(geometry.size) / 2
        high = np.asarray(geometry.center) + np.asarray(geometry.size) / 2
        x0, y0, z0 = low
        x1, y1, z1 = high
        faces = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]
        yield Poly3DCollection(faces)
    elif isinstance(geometry, td.PolySlab):
        if (
            geometry.axis != 2
            or geometry.sidewall_angle != 0
            or (
                geometry.bulges is not None and np.any(np.asarray(geometry.bulges) != 0)
            )
        ):
            return
        vertices = np.asarray(geometry.vertices, dtype=float)
        z0, z1 = (float(value) for value in geometry.slab_bounds)
        lower = [(x, y, z0) for x, y in vertices]
        upper = [(x, y, z1) for x, y in vertices]
        faces = [lower, upper] + [
            [
                lower[index],
                lower[(index + 1) % len(lower)],
                upper[(index + 1) % len(upper)],
                upper[index],
            ]
            for index in range(len(vertices))
        ]
        yield Poly3DCollection(faces)


def plot_tidy3d_fields_3d(
    simulation_data: Any,
    *,
    field: str | None = "Ey",
    quantity: str = "abs",
    wavelength: float | None = None,
    plane: tuple[str, float] = ("z", 0.0),
    planes: Sequence[tuple[str, float]] | None = None,
    geometry_opacity: float = 0.25,
):
    """Create a widget-backed inline 3D view of geometry and field planes.

    Args:
        simulation_data: Tidy3D ``SimulationData`` from the adapter.
        field: Scalar component or ``E``/``H`` vector magnitude; ``None`` hides fields.
        quantity: Field transform selected by :func:`select_field_quantity`.
        wavelength: Wavelength in meters. The nearest frequency sample is shown.
        plane: Fixed plane ``('x'|'y'|'z', coordinate_in_micrometers)``.
        planes: Optional planes to show together. The first plane is controlled
            by the movable slice controls; remaining planes stay at fixed positions.
        geometry_opacity: Initial structure opacity from 0 to 1.

    Returns:
        An ipywidgets widget that displays an interactive Matplotlib 3D scene.

    Notes:
        Notebook camera rotate, pan, and zoom require ``ipympl`` and
        ``ipywidgets``. The view draws Box and straight z-extruded PolySlab
        geometry; CustomMedium structures and unsupported geometry are skipped.
        This display does not change simulation or integration data.
    """
    try:
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from ipympl.backend_nbagg import Canvas
    except ImportError as exc:
        raise ImportError(
            "Interactive 3D plotting requires the optional dependencies "
            "'ipympl' and 'ipywidgets'; install MAPS[notebook-visualization]."
        ) from exc

    data = next(item for item in simulation_data.data if hasattr(item, "Ex"))
    arrays = {name: getattr(data, name) for name in _FIELD_NAMES}
    x, y, z = (np.asarray(arrays["Ex"][axis], dtype=float) for axis in "xyz")
    frequencies = np.asarray(arrays["Ex"].f, dtype=float)
    if (
        frequencies.ndim != 1
        or frequencies.size == 0
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies <= 0)
    ):
        raise ValueError("SimulationData must contain finite positive frequencies")
    if wavelength is not None and (not np.isfinite(wavelength) or wavelength <= 0):
        raise ValueError("wavelength must be a finite positive value in meters")
    initial_freq = (
        float(frequencies[np.argmin(np.abs(frequencies - 299792458.0 / wavelength))])
        if wavelength is not None
        else float(frequencies[0])
    )
    plane_specs = list(planes) if planes is not None else [plane]
    if not plane_specs or any(
        axis not in "xyz" or not np.isfinite(position) for axis, position in plane_specs
    ):
        raise ValueError("planes must contain ('x'|'y'|'z', finite position) pairs")
    plane = plane_specs[0]
    if not 0 <= geometry_opacity <= 1:
        raise ValueError("geometry_opacity must be between 0 and 1")
    if quantity not in {"real", "imag", "abs", "abs2", "phase"}:
        raise ValueError("quantity must be real, imag, abs, abs2, or phase")
    field_control = widgets.Dropdown(
        options=["off", *_FIELD_NAMES, "E", "H"],
        value=field or "off",
        description="Field",
    )
    if field in _VECTOR_NAMES and quantity == "phase":
        quantity = "abs"
    quantity_control = widgets.Dropdown(
        options=["real", "imag", "abs", "abs2", "phase"],
        value=quantity,
        description="Quantity",
    )
    frequency_control = widgets.SelectionSlider(
        options=[(f"{299792458.0 / f:.6g} m", float(f)) for f in frequencies],
        value=initial_freq,
        description="Wavelength",
    )
    axis_control = widgets.Dropdown(
        options=["x", "y", "z"], value=plane[0], description="Plane"
    )
    center_axis = {"x": x, "y": y, "z": z}[plane[0]]
    coordinate_control = widgets.FloatSlider(
        value=float(np.clip(plane[1], center_axis[0], center_axis[-1])),
        min=float(center_axis[0]),
        max=float(center_axis[-1]),
        step=float(np.min(np.diff(center_axis))) if len(center_axis) > 1 else 1e-9,
        description="Position (µm)",
    )
    opacity_control = widgets.FloatSlider(
        value=geometry_opacity, min=0, max=1, step=0.05, description="Geometry α"
    )
    geometry_control = widgets.Checkbox(value=True, description="Show geometry")
    fig = plt.figure(figsize=(8, 7))
    fig.canvas = Canvas(fig)
    ax = fig.add_subplot(111, projection="3d")
    canvas = fig.canvas

    def render(
        field_name,
        selected_quantity,
        frequency,
        plane_axis,
        position,
        opacity,
        show_geometry,
    ):
        ax.clear()
        sim = simulation_data.simulation
        if show_geometry:
            import tidy3d as td

            for structure in sim.structures:
                # A CustomMedium may encode a rasterized free-form pattern. Its
                # bounding box is not a faithful depiction of that pattern.
                if isinstance(structure.medium, td.CustomMedium) or hasattr(
                    structure.medium, "eps_dataset"
                ):
                    continue
                try:
                    mesh = next(_structure_vertices(structure))
                except (StopIteration, TypeError, ValueError):
                    continue
                mesh.set_alpha(opacity)
                mesh.set_facecolor("#7896ad")
                mesh.set_edgecolor("#314b60")
                ax.add_collection3d(mesh)
        if field_name != "off":
            values = select_field_quantity(
                data, field_name, selected_quantity, frequency
            )
            for current_axis, current_position in [
                (plane_axis, position),
                *plane_specs[1:],
            ]:
                coordinates = {"x": x, "y": y, "z": z}[current_axis]
                idx = int(np.argmin(np.abs(coordinates - current_position)))
                if current_axis == "z":
                    xx, yy = np.meshgrid(x, y, indexing="ij")
                    layer = values[:, :, idx]
                    surface = (xx, yy, np.full_like(xx, z[idx]))
                elif current_axis == "y":
                    xx, zz = np.meshgrid(x, z, indexing="ij")
                    layer = values[:, idx, :]
                    surface = (xx, np.full_like(xx, y[idx]), zz)
                else:
                    yy, zz = np.meshgrid(y, z, indexing="ij")
                    layer = values[idx, :, :]
                    surface = (np.full_like(yy, x[idx]), yy, zz)
                ax.plot_surface(
                    *surface,
                    facecolors=_colorize_field(layer, plt.cm.viridis),
                    shade=False,
                    alpha=0.85,
                )
        bounds = sim.bounds
        ax.set_xlim(bounds[0][0], bounds[1][0])
        ax.set_ylim(bounds[0][1], bounds[1][1])
        ax.set_zlim(bounds[0][2], bounds[1][2])
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_zlabel("z (µm)")
        ax.set_box_aspect(np.maximum(np.asarray(sim.size), 1e-12))
        fig.canvas.draw_idle()

    controls = widgets.VBox(
        [
            widgets.HBox([field_control, quantity_control, frequency_control]),
            widgets.HBox(
                [axis_control, coordinate_control, opacity_control, geometry_control]
            ),
        ]
    )

    def keep_vector_quantity_valid(change):
        if change["new"] in _VECTOR_NAMES and quantity_control.value == "phase":
            quantity_control.value = "abs"

    field_control.observe(keep_vector_quantity_valid, names="value")
    interactive = widgets.interactive_output(
        render,
        {
            "field_name": field_control,
            "selected_quantity": quantity_control,
            "frequency": frequency_control,
            "plane_axis": axis_control,
            "position": coordinate_control,
            "opacity": opacity_control,
            "show_geometry": geometry_control,
        },
    )

    def update_coordinate_range(change):
        axis = {"x": x, "y": y, "z": z}[change["new"]]
        coordinate_control.min = float(axis[0])
        coordinate_control.max = float(axis[-1])
        coordinate_control.step = (
            float(np.min(np.diff(axis))) if len(axis) > 1 else 1e-9
        )
        coordinate_control.value = float(
            np.clip(coordinate_control.value, axis[0], axis[-1])
        )

    axis_control.observe(update_coordinate_range, names="value")
    viewer = widgets.VBox([controls, canvas, interactive])
    return viewer
