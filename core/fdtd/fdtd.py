"""
MAPS-facing FDTD simulation classes backed by fdtdx.

This mirrors the public shape of ``core.fdfd.fdfd``: the class owns an external
``eps_r`` grid, accepts fdtdx-style source objects/specs in ``solve()``, and returns field component
grids. fdtdx is a 3D FDTD engine, so the concrete convenience class here is
``fdtd3d``. fdtdx-specific scene construction and adjoint-source construction
live behind the injected runtime object from ``core.fdtd.solver``.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Any, Dict, Sequence

try:
    import fdtdx
except ImportError:
    fdtdx = None
import jax
import jax.numpy as jnp
import numpy as np
import torch
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from fdtdx.core.physics.metrics import normalize_by_poynting_flux
from torch import Tensor, nn

from core.utils import (
    _torch_to_jax,
    build_rectilinear_grid_metadata,
    convert_to_fdtdx_grid_shape_loc_slice,
    resample_rectilinear_tensor,
    unfold_phasor,
    yee_to_colocate_interpolate,
)

from .solver import (
    FDTDXSolveTorch,
    _derive_magnetic_phasor_from_electric,
    clone_phasor_detector,
    native_h_slab_cells,
    phasor_detector_components,
)

__all__ = ["fdtd3d"]
_MAPS_ENABLE_PHASOR_H = os.environ.get("MAPS_ENABLE_PHASOR_H", "0") == "1"


def _real_bounds_constraints(obj, real_bounds):
    """
    real_bounds:
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) in meters.
    """
    constraints = []

    for axis, (lo, hi) in enumerate(real_bounds):
        constraints.append(
            fdtdx.RealCoordinateConstraint(
                object=obj.name,
                axes=(axis,),
                sides=("-",),
                coordinates=(float(lo),),
            )
        )
        constraints.append(
            fdtdx.RealCoordinateConstraint(
                object=obj.name,
                axes=(axis,),
                sides=("+",),
                coordinates=(float(hi),),
            )
        )

    return constraints


def _bounds_indices_to_real_bounds(bounds, grid_boundaries):
    """
    bounds:
        ((ix0, ix1), (iy0, iy1), (iz0, iz1))
    grid_boundaries:
        (x_edges, y_edges, z_edges)
    """
    return tuple(
        (
            float(grid_boundaries[axis][start]),
            float(grid_boundaries[axis][stop]),
        )
        for axis, (start, stop) in enumerate(bounds)
    )


def boundary_objects_from_config_real_coordinates(
    config,
    volume,
    grid_boundaries,
    full_domain_grid_shape,
):
    """
    Creates fdtdx boundary objects and places them using RealCoordinateConstraint
    instead of place_relative_to / PositionConstraint.

    Args:
        config:
            fdtdx.BoundaryConfig.
        volume:
            SimulationVolume. Kept for API symmetry; not used for placement.
        grid_boundaries:
            Tuple/list of edge arrays: (x_edges, y_edges, z_edges), in meters.
            Each must have length full_domain_grid_shape[axis] + 1.
        full_domain_grid_shape:
            Tuple (Nx, Ny, Nz), number of Yee cells in the simulation volume.

    Returns:
        boundaries:
            dict mapping 'min_x', 'max_x', etc. to boundary objects.
        constraints:
            list of RealCoordinateConstraint.
    """
    del volume

    grid_boundaries = tuple(
        np.asarray(edges, dtype=np.float64) for edges in grid_boundaries
    )
    full_domain_grid_shape = tuple(int(v) for v in full_domain_grid_shape)

    if len(grid_boundaries) != 3:
        raise ValueError(
            "grid_boundaries must be a tuple/list of (x_edges, y_edges, z_edges)."
        )

    if len(full_domain_grid_shape) != 3:
        raise ValueError("full_domain_grid_shape must be a 3-tuple: (Nx, Ny, Nz).")

    for axis, edges in enumerate(grid_boundaries):
        expected_len = full_domain_grid_shape[axis] + 1
        if edges.ndim != 1:
            raise ValueError(f"grid_boundaries[{axis}] must be 1D.")
        if len(edges) != expected_len:
            raise ValueError(
                f"grid_boundaries[{axis}] has length {len(edges)}, "
                f"but expected {expected_len} for shape {full_domain_grid_shape[axis]}."
            )
        if np.any(np.diff(edges) <= 0):
            raise ValueError(f"grid_boundaries[{axis}] must be strictly increasing.")

    boundaries: dict[str, Any] = {}
    constraints = []

    thickness_dict = config.get_dict()
    type_dict = config.get_type_dict()
    kappa_start_dict = config.get_kappa_dict("kappa_start")
    kappa_end_dict = config.get_kappa_dict("kappa_end")
    kappa_order_dict = config.get_order_dict("kappa_order")
    alpha_start_dict = config.get_alpha_dict("alpha_start")
    alpha_end_dict = config.get_alpha_dict("alpha_end")
    alpha_order_dict = config.get_order_dict("alpha_order")
    sigma_start_dict = config.get_sigma_dict("sigma_start")
    sigma_end_dict = config.get_sigma_dict("sigma_end")
    sigma_order_dict = config.get_order_dict("sigma_order")

    for kind, thickness in thickness_dict.items():
        axis, direction = fdtdx.objects.boundaries.utils.axis_direction_from_kind(kind)
        boundary_type = type_dict[kind]

        kappa_start = kappa_start_dict[kind]
        kappa_end = kappa_end_dict[kind]
        kappa_order = kappa_order_dict[kind]

        alpha_start = alpha_start_dict[kind]
        alpha_end = alpha_end_dict[kind]
        alpha_order = alpha_order_dict[kind]

        sigma_start = sigma_start_dict[kind]
        sigma_end = sigma_end_dict[kind]
        sigma_order = sigma_order_dict[kind]

        grid_shape_list: list[int | None] = [None, None, None]
        grid_shape_list[axis] = int(thickness) if boundary_type == "pml" else 1
        grid_shape = tuple(grid_shape_list)

        if boundary_type == "pml":
            cur_boundary = fdtdx.PerfectlyMatchedLayer(
                axis=axis,
                partial_grid_shape=grid_shape,
                kappa_start=kappa_start,
                kappa_end=kappa_end,
                kappa_order=kappa_order,
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                alpha_order=alpha_order,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                sigma_order=sigma_order,
                direction=direction,
            )
        elif boundary_type == "periodic":
            cur_boundary = fdtdx.BlochBoundary(
                axis=axis,
                partial_grid_shape=grid_shape,
                direction=direction,
                bloch_vector=(0.0, 0.0, 0.0),
            )
        elif boundary_type == "pec":
            cur_boundary = fdtdx.PerfectElectricConductor(
                axis=axis,
                partial_grid_shape=grid_shape,
                direction=direction,
            )
        elif boundary_type == "pmc":
            cur_boundary = fdtdx.PerfectMagneticConductor(
                axis=axis,
                partial_grid_shape=grid_shape,
                direction=direction,
            )
        elif boundary_type == "bloch":
            cur_boundary = fdtdx.BlochBoundary(
                axis=axis,
                partial_grid_shape=grid_shape,
                direction=direction,
                bloch_vector=config.bloch_vector,
            )
        else:
            raise ValueError(
                f"Unknown boundary type '{boundary_type}' for '{kind}'. "
                "Supported types: 'pml', 'periodic', 'pec', 'pmc', 'bloch'."
            )

        # Build the grid-index bounds of this boundary object.
        #
        # Example:
        #   min_x PML of thickness t:
        #       x: [0, t]
        #       y: [0, Ny]
        #       z: [0, Nz]
        #
        #   max_x PML of thickness t:
        #       x: [Nx - t, Nx]
        #       y: [0, Ny]
        #       z: [0, Nz]
        bounds = []
        for ax in range(3):
            if ax == axis:
                t = int(grid_shape[axis])
                n = full_domain_grid_shape[axis]

                if t <= 0:
                    raise ValueError(
                        f"Boundary '{kind}' has non-positive thickness {t}."
                    )
                if t > n:
                    raise ValueError(
                        f"Boundary '{kind}' thickness {t} exceeds axis size {n}."
                    )

                if direction == "-":
                    lo, hi = 0, t
                elif direction == "+":
                    lo, hi = n - t, n
                else:
                    raise ValueError(
                        f"Boundary '{kind}' has invalid direction {direction!r}."
                    )
            else:
                lo, hi = 0, full_domain_grid_shape[ax]

            bounds.append((lo, hi))

        real_bounds = _bounds_indices_to_real_bounds(
            tuple(bounds),
            grid_boundaries,
        )

        boundaries[kind] = cur_boundary
        constraints.extend(_real_bounds_constraints(cur_boundary, real_bounds))

    return boundaries, constraints


def real_box_to_grid_slices(
    boundaries,
    center,
    size,
    direction,
    *,
    transverse_rule="cover",
    clip=True,
):
    """
    Convert a physical source/monitor box to native grid slices.

    Parameters
    ----------
    boundaries:
        Tuple/list of 3 monotonic 1D arrays: (x_edges, y_edges, z_edges).
        Length of each edge array is N_axis + 1.
    center:
        Physical center coordinate, same unit as boundaries.
    size:
        Physical box size, same unit as boundaries.
    direction:
        Propagation direction, e.g. "x+", "x-", "y+", "z-".
        The propagation axis is forced to one cell.
    transverse_rule:
        "cover":
            Include all cells whose intervals overlap the requested aperture.
            This guarantees the physical aperture is covered.
        "center":
            Include cells whose centers lie inside the requested aperture.
            This gives a closer effective center/size but may under-cover boundaries.
    clip:
        Whether to clip slices to the simulation domain.

    Returns
    -------
    tuple[slice, slice, slice]
        Native grid slice tuple.
    """

    if len(boundaries) != 3:
        raise ValueError("boundaries must be a tuple/list of 3 edge arrays.")

    if len(center) != 3 or len(size) != 3:
        raise ValueError("center and size must be length-3 sequences.")

    if direction[0] not in "xyz":
        raise ValueError(f"Invalid direction {direction!r}. Expected e.g. 'x+'.")

    prop_axis = "xyz".index(direction[0])

    grid_slices = []

    for ax in range(3):
        edges = np.asarray(boundaries[ax], dtype=np.float64)

        if edges.ndim != 1:
            raise ValueError(f"boundaries[{ax}] must be 1D.")

        if len(edges) < 2:
            raise ValueError(f"boundaries[{ax}] must have at least 2 entries.")

        if np.any(np.diff(edges) <= 0):
            raise ValueError(f"boundaries[{ax}] must be strictly increasing.")

        n_cells = len(edges) - 1

        c = float(center[ax])
        L = float(size[ax])

        if ax == prop_axis:
            # Plane source: choose the one cell whose center is closest
            # to the requested physical plane center.
            cell_centers = 0.5 * (edges[:-1] + edges[1:])
            start = int(np.argmin(np.abs(cell_centers - c)))
            stop = start + 1

        else:
            lo = c - L / 2
            hi = c + L / 2

            if hi <= lo:
                raise ValueError(
                    f"Invalid transverse size on axis {ax}: "
                    f"center={c}, size={L}, lo={lo}, hi={hi}"
                )

            if transverse_rule == "cover":
                # Include every cell whose interval [edge_i, edge_{i+1}]
                # overlaps [lo, hi].
                #
                # First overlapping cell:
                #   edge[i+1] > lo
                # Last overlapping cell:
                #   edge[i] < hi
                start = int(np.searchsorted(edges, lo, side="right") - 1)
                stop = int(np.searchsorted(edges, hi, side="left"))

                # If lo lies exactly on an edge, start should be that cell,
                # not the previous cell.
                if start + 1 < len(edges) and np.isclose(edges[start + 1], lo):
                    start = start + 1

            elif transverse_rule == "center":
                # Include cells whose centers are inside [lo, hi].
                cell_centers = 0.5 * (edges[:-1] + edges[1:])
                idx = np.nonzero((cell_centers >= lo) & (cell_centers <= hi))[0]

                if len(idx) == 0:
                    # Aperture is smaller than the local cell width.
                    # Use nearest cell to aperture center.
                    nearest = int(np.argmin(np.abs(cell_centers - c)))
                    start = nearest
                    stop = nearest + 1
                else:
                    start = int(idx[0])
                    stop = int(idx[-1]) + 1

            else:
                raise ValueError(
                    f"Unknown transverse_rule={transverse_rule!r}. "
                    "Use 'cover' or 'center'."
                )

            if clip:
                start = max(0, start)
                stop = min(n_cells, stop)

            if stop <= start:
                raise ValueError(
                    f"Empty slice on axis {ax}: "
                    f"lo={lo}, hi={hi}, start={start}, stop={stop}, "
                    f"domain=({edges[0]}, {edges[-1]})"
                )

        grid_slices.append(slice(start, stop))

    return tuple(grid_slices)


class fdtd3d(nn.Module):
    """Base class for fdtdx-backed differentiable 3D FDTD simulations."""

    def __init__(
        self,
        ## each wavelength group, this simulation instance is uniquely created.
        wl_cen: float,
        wl_width: float,
        n_wl: int,
        dL: float,
        eps_r: np.ndarray | Tensor,
        port_sources_dict: Dict,
        port_monitor_slices: Dict,
        port_monitor_slices_info: Dict,
        npml: Sequence[int],
        electrical_conductivity_map: np.ndarray | Tensor | None = None,
        optical_grid_metadata: dict[str, Any] | None = None,
        max_time: float = 1e-10,
        symmetry: tuple[int, int, int] = (0, 0, 0),  # [0: no symmetry, -1: PEC, 1: PMC]
        *,
        device: str | torch.device = "cpu",
        **runtime_kwargs: Any,
    ) -> None:
        super().__init__()
        self.wl_cen = wl_cen
        self.wl_width = wl_width
        self.n_wl = n_wl
        self.dL = dL  # m unit
        self.npml = tuple(
            npml
        )  # [(xmin_thickness, xmax_thickness), (ymin_thickness, ymax_thickness), (zmin_thickness, zmax_thickness)]
        self.device = torch.device(device)
        self.port_sources_dict = port_sources_dict
        self.port_monitor_slices = port_monitor_slices
        self.port_monitor_slices_info = port_monitor_slices_info
        self.solver = FDTDXSolveTorch()
        self.optical_grid_metadata = optical_grid_metadata
        self.field_grid_metadata = optical_grid_metadata
        self.eps_r = eps_r
        self.electrical_conductivity_map = electrical_conductivity_map
        self.max_time = max_time
        self.symmetry = symmetry

    def _grid_metadata(self) -> dict[str, Any]:
        if self.optical_grid_metadata is not None:
            return self.optical_grid_metadata
        uniform_grid = build_rectilinear_grid_metadata(
            shape=self.shape,
            spacing=self.dL,
            label="optical_uniform",
        ).to_dict()
        return {
            "native": uniform_grid,
            "export": uniform_grid,
            "export_policy": "identity",
            "native_equals_export": True,
        }

    def _metadata_length_scale(self) -> float:
        # MAPS device-layer optical-grid metadata is expressed in microns,
        # while fdtdx expects SI metres. The internal fallback metadata path
        # already uses ``self.dL`` in metres.
        return 1e-6 if self.optical_grid_metadata is not None else 1.0

    def _simulation_grid(self):
        metadata = self._grid_metadata()
        native = metadata["native"]
        scale = self._metadata_length_scale()
        boundaries = tuple(axis * scale for axis in native["boundaries"])
        # print(metadata)
        if native.get("is_uniform", False):
            spacing = float(native["spacing"][0]) * scale
            origin = tuple(float(axis[0]) for axis in boundaries)
            # return UniformGrid(spacing=spacing, origin=origin)
            return RectilinearGrid.uniform(
                shape=tuple(len(axis) - 1 for axis in boundaries),
                spacing=spacing,
                origin=origin,
            )
        edge_arrays = tuple(jnp.asarray(axis, dtype=jnp.float32) for axis in boundaries)
        # print("Using nonuniform grid with edges:", edge_arrays)
        ## uniform or nonuniform grid, we use RectilinearGrid to support both cases.
        return RectilinearGrid.custom(
            x_edges=edge_arrays[0],
            y_edges=edge_arrays[1],
            z_edges=edge_arrays[2],
        )

    def _native_boundaries(self):
        native = self._grid_metadata()["native"]
        scale = self._metadata_length_scale()
        return tuple(axis * scale for axis in native["boundaries"])

    def _native_extent(self) -> tuple[float, float, float]:
        boundaries = self._native_boundaries()
        extents = []
        for axis in boundaries:
            extent = float(axis[-1] - axis[0])
            # Keep the simulation-volume metric extent infinitesimally inside
            # the last edge so fdtdx's nonuniform length->cell-count snapping
            # resolves to the final valid cell interval rather than one past it.
            extents.append(float(np.nextafter(extent, 0.0)))
        return tuple(extents)

    def _uses_nonuniform_grid(self) -> bool:
        return not self._grid_metadata().get("native", {}).get("is_uniform", False)

    def _sanitize_constraints_for_nonuniform(self, constraints):
        if not self._uses_nonuniform_grid():
            return constraints
        sanitized = []
        for constraint in constraints:
            if hasattr(constraint, "grid_margins"):
                margins = tuple(
                    None if value in (None, 0) else value
                    for value in constraint.grid_margins
                )
                constraint = replace(constraint, grid_margins=margins)
            if hasattr(constraint, "grid_offsets"):
                offsets = tuple(
                    None if value in (None, 0) else value
                    for value in constraint.grid_offsets
                )
                constraint = replace(constraint, grid_offsets=offsets)
            sanitized.append(constraint)
        return sanitized

    def _resolve_slice_bounds(
        self,
        monitor_slice,
        direction: str,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        _, native_bounds = convert_to_fdtdx_grid_shape_loc_slice(
            monitor_slice, direction
        )
        validated_bounds = []
        for axis, axis_bounds in enumerate(native_bounds):
            start, stop = int(axis_bounds[0]), int(axis_bounds[1])
            axis_size = int(self.shape[axis])
            if start < 0 or stop > axis_size or start >= stop:
                raise ValueError(
                    "Monitor slice bounds must live on the native optical grid. "
                    f"axis={axis}, bounds={(start, stop)}, native_shape={self.shape}"
                )
            validated_bounds.append(slice(start, stop, None))
        return tuple(validated_bounds)

    def _object_with_native_shape(self, obj: Any, bounds) -> Any:
        native_shape = tuple(
            int(axis_bounds[1] - axis_bounds[0]) for axis_bounds in bounds
        )
        if tuple(getattr(obj, "partial_grid_shape", ())) == native_shape:
            return obj
        if hasattr(obj, "aset"):
            return obj.aset("partial_grid_shape", native_shape)
        return obj

    def _placement_kwargs(self, bounds):
        if self._grid_metadata().get("native_equals_export", False):
            return {
                "grid_margins": tuple(int(axis_bounds[0]) for axis_bounds in bounds)
            }
        native_edges = self._native_boundaries()
        margins = tuple(
            float(native_edges[axis][axis_bounds[0]] - native_edges[axis][0])
            for axis, axis_bounds in enumerate(bounds)
        )
        return {"margins": margins}

    # def _nonuniform_object_constraints(self, obj, volume, bounds):
    #     full_bounds = self._full_volume_bounds()
    #     native_edges = self._native_boundaries()
    #     constraints = []
    #     full_axes = tuple(
    #         axis
    #         for axis, axis_bounds in enumerate(bounds)
    #         if tuple(axis_bounds) == tuple(full_bounds[axis])
    #     )
    #     if full_axes:
    #         constraints.append(obj.same_size(volume, axes=full_axes))
    #     all_axes = tuple(range(3))
    #     constraints.append(
    #         fdtdx.RealCoordinateConstraint(
    #             object=obj.name,
    #             axes=all_axes,
    #             sides=tuple("-" for _ in all_axes),
    #             coordinates=tuple(
    #                 float(native_edges[axis][bounds[axis][0]]) for axis in all_axes
    #             ),
    #         )
    #     )
    #     return constraints

    def _nonuniform_object_constraints(self, obj, volume, bounds):
        full_bounds = self._full_volume_bounds()
        native_edges = self._native_boundaries()
        constraints = []
        full_axes = tuple(
            axis
            for axis, axis_bounds in enumerate(bounds)
            if tuple(axis_bounds) == tuple(full_bounds[axis])
        )
        if full_axes:
            # constraints.append(obj.same_size(volume, axes=full_axes))
            # Position full axes at origin
            constraints.append(
                fdtdx.RealCoordinateConstraint(
                    object=obj.name,
                    axes=full_axes,
                    sides=tuple("-" for _ in full_axes),
                    coordinates=tuple(
                        float(native_edges[axis][0]) for axis in full_axes
                    ),
                )
            )
        remaining_axes = tuple(axis for axis in range(3) if axis not in full_axes)
        if remaining_axes:
            constraints.append(
                fdtdx.RealCoordinateConstraint(
                    object=obj.name,
                    axes=remaining_axes,
                    sides=tuple("-" for _ in remaining_axes),
                    coordinates=tuple(
                        float(native_edges[axis][bounds[axis][0]])
                        for axis in remaining_axes
                    ),
                )
            )
        return constraints

    def _area_weights_for_bounds(self, bounds, axis: int):
        boundaries = self._native_boundaries()
        transverse = []
        for axis_index, axis_bounds in enumerate(bounds):
            if axis_index == axis:
                continue
            widths = np.diff(
                # boundaries[axis_index][axis_bounds[0] : axis_bounds[1] + 1]
                boundaries[axis_index][axis_bounds.start : axis_bounds.stop + 1]
            )
            transverse.append(jnp.asarray(widths, dtype=jnp.float32))
        if len(transverse) == 0:
            return None
        area = transverse[0]
        for axis_width in transverse[1:]:
            area = area.reshape(area.shape + (1,)) * axis_width.reshape(
                (1,) * area.ndim + axis_width.shape
            )
        return area

    def _full_volume_bounds(self):
        return tuple((0, int(size)) for size in self.shape)

    def prepare_arrays(
        self,
        source,
        source_grid_bounds,
        source_center,
        source_size,
        monitor,
        extra_monitors=None,
        normalize_source_power=False,
        direction: str | None = None,
        symmetry: tuple[int, int, int] = (0, 0, 0),
    ):
        eps_r = self.eps_r
        electrical_conductivity_map = self.electrical_conductivity_map

        if torch.is_complex(eps_r):
            # if torch.any(eps_r.imag != 0) and electrical_conductivity_map is None:
            # raise NotImplementedError(
            #     "fdtd3d.prepare_arrays() received complex eps_r with a nonzero imaginary part, "
            #     "but no electrical_conductivity_map was provided for fdtdx loss modeling."
            # )
            eps_r = eps_r.real

        if electrical_conductivity_map is not None and torch.is_complex(
            electrical_conductivity_map
        ):
            electrical_conductivity_map = electrical_conductivity_map.real

        # every preparation is called before solve fdtd, and need one source
        objects = []
        constraints = []
        # print(f"SimulationVolume eps_r.shape: {eps_r.shape}")
        volume = fdtdx.SimulationVolume(
            partial_grid_shape=eps_r.shape,
            material=fdtdx.Material(permittivity=1.0),  # Dummy material
        )
        objects.append(volume)

        ## non-zero means PML, zero means periodic
        # nonzero_npml = [n for n in self.npml if n > 0]
        # assert all(
        #     n == nonzero_npml[0] for n in nonzero_npml
        # ), f"All direction's n_PML must be the same thickness, but got {self.npml}"
        # pml_thickness = max(nonzero_npml) if nonzero_npml else 0
        # override_types = {}
        # if self.npml[0] > 0:
        #     override_types.update(
        #         {
        #             "min_x": "pml",
        #             "max_x": "pml",
        #         }
        #     )
        # if self.npml[1] > 0:
        #     override_types.update(
        #         {
        #             "min_y": "pml",
        #             "max_y": "pml",
        #         }
        #     )
        # if self.npml[2] > 0:
        #     override_types.update(
        #         {
        #             "min_z": "pml",
        #             "max_z": "pml",
        #         }
        #     )
        # bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        #     boundary_type="periodic",
        #     thickness=int(pml_thickness),
        #     override_types=override_types,
        # )
        bound_cfg = fdtdx.BoundaryConfig(
            thickness_grid_minx=max(1, self.npml[0][0]),
            thickness_grid_maxx=max(1, self.npml[0][1]),
            thickness_grid_miny=max(1, self.npml[1][0]),
            thickness_grid_maxy=max(1, self.npml[1][1]),
            thickness_grid_minz=max(1, self.npml[2][0]),
            thickness_grid_maxz=max(1, self.npml[2][1]),
            boundary_type_minx="pml" if self.npml[0][0] > 0 else "periodic",
            boundary_type_maxx="pml" if self.npml[0][1] > 0 else "periodic",
            boundary_type_miny="pml" if self.npml[1][0] > 0 else "periodic",
            boundary_type_maxy="pml" if self.npml[1][1] > 0 else "periodic",
            boundary_type_minz="pml" if self.npml[2][0] > 0 else "periodic",
            boundary_type_maxz="pml" if self.npml[2][1] > 0 else "periodic",
        )
        # bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=self.npml[0])
        if self._uses_nonuniform_grid():
            bound_dict, bound_constraints = (
                boundary_objects_from_config_real_coordinates(
                    bound_cfg,
                    volume,
                    grid_boundaries=self._native_boundaries(),  # already meters
                    full_domain_grid_shape=self.shape,
                )
            )
        else:
            bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(
                bound_cfg, volume
            )
        objects.extend(bound_dict.values())
        constraints.extend(bound_constraints)

        config = fdtdx.SimulationConfig(
            grid=self._simulation_grid(),
            # time=1e-10,  # this is the max simulation time, can set to a larger number if needed, but this is already very large
            time=self.max_time,  # this is the max simulation time, can set to a larger number if needed, but this is already very large
            dtype=jnp.float32,
            courant_factor=0.99,
            symmetry=symmetry,
        )
        key = jax.random.PRNGKey(0)

        # place source
        # source_profile = self.port_sources_dict[input_slice_name]
        # source, *_ = source_profile[((wl_cen, wl_width, n_wl), mode)]
        # source = self._object_with_native_shape(source, source_grid_bounds)

        objects.append(source)  # source[0] is the source objective
        # input_slice = self.port_monitor_slices[input_slice_name]
        # direction = self.port_monitor_slices_info[input_slice_name]["direction"]
        # partial_grid_shape, grid_slice_tuple = convert_to_fdtdx_grid_shape_loc_slice(
        #     input_slice, direction
        # )

        # constraints.append(
        #         fdtdx.RealCoordinateConstraint(
        #             object=source.name,
        #             axes=(0, 1, 2),
        #             sides=("-", "-", "-"),
        #             coordinates=tuple(
        #                 source_center[i] - source_size[i] / 2 for i in range(3)
        #             ),
        #         )
        #     )

        # if direction is not None:
        #     ## this is forward mode source with direction
        #     ## we use gridcoordinateconstraint
        #     source_grid_slice = real_box_to_grid_slices(
        #         boundaries=self._native_boundaries(),
        #         center=source_center,
        #         size=source_size,
        #         direction=direction,
        #         transverse_rule="cover",
        #         clip=True,
        #     )
        #     constraints.append(
        #         fdtdx.GridCoordinateConstraint(
        #             object=source.name,
        #             axes=(0, 1, 2),
        #             sides=("-", "-", "-"),
        #             coordinates=tuple(int(source_grid_slice[i].start) for i in range(3)),
        #         )
        #     )
        # else:
        #     ## no direction is defined, it will be adjoint source
        #     constraints.append(
        #         fdtdx.RealCoordinateConstraint(
        #             object=source.name,
        #             axes=(0, 1, 2),
        #             sides=("-", "-", "-"),
        #             coordinates=tuple(
        #                 source_center[i] - source_size[i] / 2 for i in range(3)
        #             ),
        #         )
        #     )

        # print(source.partial_grid_shape)
        # print(source.partial_real_shape)
        # print(constraints[-1])

        # if self._uses_nonuniform_grid():
        #     # constraints.extend(
        #     #     self._nonuniform_object_constraints(source, volume, source_grid_bounds)
        #     # )

        #     constraints.append(
        #         fdtdx.RealCoordinateConstraint(
        #             object=source.name,
        #             axes=(0, 1, 2),
        #             sides=("-", "-", "-"),
        #             coordinates=tuple(
        #                 source_center[i] - source_size[i] / 2 for i in range(3)
        #             ),
        #         ),
        #     )
        # else:
        #     constraints.append(
        #         source.place_relative_to(
        #             volume,
        #             axes=(0, 1, 2),
        #             other_positions=(-1, -1, -1),
        #             own_positions=(-1, -1, -1),
        #             **self._placement_kwargs(source_grid_bounds),
        #         )
        #     )
        # place broadband phasor detectors attached to this source by key ((wl_cen, wl_width, n_wl), mode)
        # monitor_profile = self.port_sources_dict[input_slice_name + "_field_monitor"]
        # monitor, *_ = monitor_profile[((wl_cen, wl_width, n_wl), mode)]
        if isinstance(monitor, fdtdx.PhasorDetector):
            # constraints.extend(monitor.same_position_and_size(volume))
            if self._uses_nonuniform_grid():
                full_bounds = self._full_volume_bounds()
                monitor = self._object_with_native_shape(monitor, full_bounds)
                constraints.extend(
                    self._nonuniform_object_constraints(monitor, volume, full_bounds)
                )
            else:
                constraints.extend(monitor.same_position_and_size(volume))
            objects.append(monitor)
        else:
            raise NotImplementedError(
                f"Only PhasorDetector is supported as monitor for source, but got {type(monitor)}"
            )
        if extra_monitors is not None:
            for extra_monitor, extra_grid_margins in extra_monitors:
                extra_monitor = self._object_with_native_shape(
                    extra_monitor, extra_grid_margins
                )
                if not isinstance(extra_monitor, fdtdx.PhasorDetector):
                    raise NotImplementedError(
                        "Only PhasorDetector is supported as auxiliary monitor"
                    )
                constraints.extend(
                    self._nonuniform_object_constraints(
                        extra_monitor, volume, extra_grid_margins
                    )
                )
                objects.append(extra_monitor)

        ## place Energy Detector to monitor when to stop FDTD
        decay_detector = fdtdx.EnergyDetector(
            name="decay",
            reduce_volume=True,
            partial_grid_shape=eps_r.shape,
            plot=False,
            exact_interpolation=False,  # no need to interpolate here.
            switch=fdtdx.OnOffSwitch(interval=1000),
        )
        # constraints.extend(decay_detector.same_position_and_size(volume))
        if self._uses_nonuniform_grid():
            constraints.extend(
                self._nonuniform_object_constraints(
                    decay_detector, volume, self._full_volume_bounds()
                )
            )
        else:
            constraints.extend(decay_detector.same_position_and_size(volume))
        objects.append(decay_detector)

        # start = time.time()
        # constraints = self._sanitize_constraints_for_nonuniform(constraints)
        objects, arrays, params, config, info = fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            key=key,
        )
        # print(f"fdtdx.place_objects takes {time.time() - start:.2f} seconds")
        # start = time.time()
        inv_permittivities = _torch_to_jax(1 / eps_r[None, ...].data)
        inv_permeabilities = 1.0

        if config.has_symmetry:
            symmetry_mid_abs = info["symmetry_mid_abs"]
            inv_permittivities = inv_permittivities[
                :, symmetry_mid_abs[0] :, symmetry_mid_abs[1] :, symmetry_mid_abs[2] :
            ]
        arrays = arrays.at["inv_permittivities"].set(inv_permittivities)
        arrays = arrays.at["inv_permeabilities"].set(inv_permeabilities)
        ## [WARNING] in fdtdx, electric_conductivity which is designed to use as constant loss for DC/low-freq simulation
        ## for lossy material, e.g., Metal:
        # we cannot use complex permittivity in freq-domain, e.g., -10 + 20j, this is valid in FDFD, but not causal in time-domain FDTD with engative real(epsilon)
        # to approximate metal in fdtd, Tidy3D treats metal as PEC medium with electric_conductivity
        # since metal is far away from optical region, sometimes even in PML, we don't need to model it..
        # if electrical_conductivity_map is not None:
        # electric_conductivity = _torch_to_jax(
        #     electrical_conductivity_map[None, ...].data
        # ).astype(config.dtype)
        # if getattr(arrays, "electric_conductivity", None) is None:
        #     arrays = arrays.aset(
        #         "electric_conductivity",
        #         electric_conductivity,
        #         create_new_ok=True,
        #     )
        # else:
        #     arrays = arrays.at["electric_conductivity"].set(
        #         electric_conductivity,
        #         create_new_ok=True,
        #     )
        # print(f"Setting inv_permittivities takes {time.time() - start:.2f} seconds")
        # start = time.time()
        arrays, objects, _ = fdtdx.apply_params(
            arrays=arrays,
            objects=objects,
            params=params,
            key=key,
        )
        # print(f"fdtdx.apply_params takes {time.time() - start:.2f} seconds")

        if normalize_source_power:
            source = objects[source.name]
            # After apply_params, manually normalize by Poynting flux

            # In your simulation setup after source.apply():
            E_norm, H_norm = normalize_by_poynting_flux(
                source._E,
                source._H,
                axis=source.propagation_axis,
                area_weights=self._area_weights_for_bounds(
                    source_grid_bounds,
                    axis=source.propagation_axis,
                ),
            )

            source = source.aset("_E", E_norm)
            source = source.aset("_H", H_norm)
            # print(f"normalize_source_power takes {time.time() - start:.2f} seconds")
        # self.arrays = arrays
        # self.objects = objects
        # self.config = config
        # self.key = key
        # self.field_monitor = objects[monitor.name]
        resolved_extra_monitors = {}
        if extra_monitors is not None:
            for extra_monitor, _ in extra_monitors:
                resolved_extra_monitors[extra_monitor.name] = objects[
                    extra_monitor.name
                ]
        # for obj in objects.objects:
        #     print(obj.name, obj.grid_slice, obj.partial_grid_shape, obj.partial_real_shape, obj.partial_real_position)
        return (
            arrays,
            objects,
            config,
            key,
            objects[monitor.name],
            resolved_extra_monitors,
        )

    @property
    def eps_r(self) -> Tensor:
        """Relative permittivity grid."""
        return self._eps_r

    @eps_r.setter
    def eps_r(self, new_eps: np.ndarray | Tensor) -> None:
        if isinstance(new_eps, np.ndarray):
            new_eps = torch.from_numpy(new_eps)
        new_eps = new_eps.to(self.device)
        self._save_shape(new_eps)
        self._eps_r = new_eps

    @property
    def electrical_conductivity_map(self) -> Tensor | None:
        return self._electrical_conductivity_map

    @electrical_conductivity_map.setter
    def electrical_conductivity_map(
        self, new_sigma: np.ndarray | Tensor | None
    ) -> None:
        if new_sigma is None:
            self._electrical_conductivity_map = None
            return
        if isinstance(new_sigma, np.ndarray):
            new_sigma = torch.from_numpy(new_sigma)
        self._electrical_conductivity_map = new_sigma.to(
            device=self.device,
            dtype=torch.float32,
        )

    def obtain_source_native_grid_slice(self, source, source_center, source_size):
        objects = []
        constraints = []
        # print(f"SimulationVolume eps_r.shape: {self.eps_r.shape}")
        # print(self._native_boundaries())
        # print(self._native_extent())
        # print("source_center:", source_center)
        # print("source_size:", source_size)
        volume = fdtdx.SimulationVolume(
            partial_grid_shape=self.eps_r.shape,
            material=fdtdx.Material(permittivity=1.0),  # Dummy material
        )
        objects.append(volume)

        config = fdtdx.SimulationConfig(
            grid=self._simulation_grid(),
            # time=1e-10,  # this is the max simulation time, can set to a larger number if needed, but this is already very large
            time=2e-13,  # this is the max simulation time, can set to a larger number if needed, but this is already very large
            dtype=jnp.float32,
            courant_factor=0.99,
        )
        key = jax.random.PRNGKey(0)
        objects.append(source)  # source[0] is the source objective

        # constraints.append(
        #     fdtdx.RealCoordinateConstraint(
        #         object=source.name,
        #         axes=(0, 1, 2),
        #         sides=("-", "-", "-"),
        #         coordinates=tuple(
        #             source_center[i] - source_size[i] / 2 for i in range(3)
        #         ),
        #     ),
        # )

        objects, arrays, params, config, _ = fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            key=key,
        )

        source_grid_indices = objects[source.name].grid_slice

        # print(source_center)
        # print(source_size)
        # print(self._native_boundaries())
        # print(source_grid_indices)

        return source_grid_indices

    def prepare_forward_simulation(
        self,
        input_slice_name: str,
        wl_cen: float,
        wl_width: float,
        n_wl: int,
        mode: str,
        verbose: bool = False,
    ):
        # place source
        start = time.time()
        source_profile = self.port_sources_dict[input_slice_name]
        source, *_ = source_profile[((wl_cen, wl_width, n_wl), mode)]
        input_slice = self.port_monitor_slices[input_slice_name]
        direction = self.port_monitor_slices_info[input_slice_name]["direction"]
        source_center = self.port_monitor_slices_info[input_slice_name]["center"]
        source_size = self.port_monitor_slices_info[input_slice_name]["size"]
        scale = self._metadata_length_scale()
        source_center = tuple(s * scale for s in source_center)
        source_size = tuple(s * scale for s in source_size)

        if 1 or self._uses_nonuniform_grid():
            # source needs to be placed with RealCoordinateConstraint, so we need to obtain the native grid slice for this source first, then place it at the right position with RealCoordinateConstraint in prepare_arrays.
            source_grid_indices = self.obtain_source_native_grid_slice(
                source, source_center, source_size
            )
            # print(source_grid_indices, input_slice)
            grid_slice_tuple = source_grid_indices
        else:
            grid_slice_tuple = self._resolve_slice_bounds(input_slice, direction)
        # place broadband phasor detectors attached to this source by key ((wl_cen, wl_width, n_wl), mode)
        monitor_profile = self.port_sources_dict[input_slice_name + "_field_monitor"]
        monitor, *_ = monitor_profile[((wl_cen, wl_width, n_wl), mode)]
        desired_components = phasor_detector_components()
        if tuple(getattr(monitor, "components", ())) != desired_components:
            monitor = clone_phasor_detector(monitor, components=desired_components)

        native_h_monitor = None
        native_h_slice = None
        extra_monitors = None
        # slab_cells = native_h_slab_cells()
        # if desired_components == ("Ex", "Ey", "Ez") and slab_cells > 0:
        #     axis = "xyz".index(direction[0])
        #     slab_starts = [int(grid_slice_tuple[i].start) for i in range(3)]
        #     slab_stops = [int(grid_slice_tuple[i].stop) for i in range(3)]
        #     slab_starts[axis] = max(0, slab_starts[axis] - slab_cells)
        #     slab_stops[axis] = min(int(self.shape[axis]), slab_stops[axis] + slab_cells)
        #     slab_shape = tuple(slab_stops[i] - slab_starts[i] for i in range(3))
        #     native_h_monitor = clone_phasor_detector(
        #         monitor,
        #         components=("Hx", "Hy", "Hz"),
        #         name=f"{monitor.name}_native_h_slab",
        #         partial_grid_shape=slab_shape,
        #     )
        #     native_h_slice = tuple(
        #         slice(slab_starts[i], slab_stops[i]) for i in range(3)
        #     )
        #     extra_monitors = [
        #         (
        #             native_h_monitor,
        #             tuple((slab_starts[i], slab_stops[i]) for i in range(3)),
        #         )
        #     ]

        arrays, objects, config, key, field_monitor, resolved_extra_monitors = (
            self.prepare_arrays(
                source=source,
                source_grid_bounds=grid_slice_tuple,
                source_center=source_center,
                source_size=source_size,
                monitor=monitor,
                extra_monitors=extra_monitors,
                normalize_source_power=False,
                direction=direction,
                symmetry=self.symmetry,
            )
        )
        if native_h_monitor is not None:
            native_h_monitor = resolved_extra_monitors[native_h_monitor.name]
        if verbose:
            print(f"prepare_forward_simulation takes {time.time() - start:.2f} seconds")
        return (
            arrays,
            objects,
            config,
            key,
            field_monitor,
            native_h_monitor,
            native_h_slice,
        )

    def solve(
        self,
        input_slice_name: Any,
        wl_cen: Any,
        wl_width: Any,
        n_wl: Any,
        mode: Any,
        *,
        eps_r: np.ndarray | Tensor | None = None,
        adjoint_group_info: Any = None,
        interpolate_to_export_grid: bool = True,
        **runtime_kwargs: Any,
    ) -> tuple[Tensor, ...]:
        """Run a differentiable fdtdx forward solve and return field grids.

        Args:
            input_slice_name: Name of the input slice to use as the source.
            wl_cen: Center wavelength for the source.
            wl_width: Wavelength width for the source.
            n_wl: Number of wavelength points for the source.
            mode: Mode index for the source e.g., "Ez1" for TM0 mode
            eps_r: Optional external permittivity for this call.  When omitted,
                ``self.eps_r`` is used.
            interpolate_to_export_grid: Whether to interpolate the fields to the export grid.
            runtime_kwargs: Extra options forwarded to the concrete fdtdx runtime.
        """

        eps = self.eps_r if eps_r is None else self._as_tensor(eps_r)
        self._save_shape(eps)
        self.eps_r = eps
        (
            fwd_arrays,
            fwd_objects,
            fwd_config,
            fwd_key,
            fwd_field_monitor,
            fwd_native_h_monitor,
            fwd_native_h_slice,
        ) = self.prepare_forward_simulation(
            input_slice_name=input_slice_name,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
            mode=mode,
        )
        ## those are needed for objective to do yee grid interpolation
        self.objects = fwd_objects
        self.config = fwd_config

        def prepare_adjoint_simulation(source, source_grid_margins, monitor):

            arrays, objects, config, key, field_monitor, _ = self.prepare_arrays(
                source=source,
                source_grid_bounds=source_grid_margins,
                source_center=(0, 0, 0),
                source_size=self._native_extent(),
                monitor=monitor,
                symmetry=self.symmetry,
            )
            return arrays, objects, config, key, field_monitor

        fields, fwd_arrays = self.solver(
            eps,
            fwd_arrays,
            fwd_objects,
            fwd_config,
            fwd_key,
            monitor=fwd_field_monitor,
            native_h_monitor=fwd_native_h_monitor,
            native_h_slice=fwd_native_h_slice,
            field_grid_metadata=self.field_grid_metadata,
            prepare_adjoint_simulation=prepare_adjoint_simulation,
            adjoint_group_info=adjoint_group_info,
            **runtime_kwargs,
        )
        if _MAPS_ENABLE_PHASOR_H and fields.shape[-4] == 6:
            native_e, native_h = fields.chunk(2, dim=-4)
        else:
            native_e = fields
            native_h = _derive_magnetic_phasor_from_electric(
                native_e,
                arrays=fwd_arrays,
                objects=fwd_objects,
                config=fwd_config,
                wave_characters=fwd_field_monitor.wave_characters,
            )
        # print("native_e shape:", native_e.shape)
        # print("native_h shape:", native_h.shape)
        native_e = yee_to_colocate_interpolate(
            native_e, fwd_objects, fwd_config, is_E=True
        )
        native_h = yee_to_colocate_interpolate(
            native_h, fwd_objects, fwd_config, is_E=False
        )

        fields = torch.cat((native_e, native_h), dim=1)

        fields = unfold_phasor(
            fields,
            monitor=fwd_field_monitor,
            components=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
            config=fwd_config,
        )

        metadata = self.field_grid_metadata or {}
        native_grid = metadata.get("native")
        export_grid = metadata.get("export")
        if (
            interpolate_to_export_grid
            and native_grid is not None
            and export_grid is not None
            and not metadata.get("native_equals_export", False)
        ):
            fields = resample_rectilinear_tensor(
                fields,
                src_coords=native_grid.get("coords"),
                dst_coords=export_grid.get("coords"),
                axes=(-3, -2, -1),
            )
        # fields [nfreqs, 6, X, Y, Z]
        # print(fields.shape)
        Ex, Ey, Ez, Hx, Hy, Hz = fields.chunk(6, dim=1)
        (Ex, Ey, Ez, Hx, Hy, Hz) = (
            Ex.squeeze(1),
            Ey.squeeze(1),
            Ez.squeeze(1),
            Hx.squeeze(1),
            Hy.squeeze(1),
            Hz.squeeze(1),
        )
        self.fields_are_colocated = True
        return Ex, Ey, Ez, Hx, Hy, Hz

    def forward(self, source: Any, **kwargs: Any) -> tuple[Tensor, ...]:
        return self.solve(source, **kwargs)

    def clear_solver_cache(self) -> None:
        if hasattr(self.runtime, "clear_solver_cache"):
            self.runtime.clear_solver_cache()

    def set_cache_mode(self, mode: bool) -> None:
        if hasattr(self.runtime, "set_cache_mode"):
            self.runtime.set_cache_mode(mode)

    def _as_tensor(self, value: np.ndarray | Tensor) -> Tensor:
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        return value.to(self.device)

    def _as_source(self, source: Any) -> Any:
        if isinstance(source, (np.ndarray, Tensor)):
            return self._as_tensor(source)
        return source

    def _vec_to_grid(self, vec: Tensor) -> Tensor:
        return vec.reshape(self.shape)

    def _grid_to_vec(self, grid: Tensor) -> Tensor:
        return grid.flatten()

    def _save_shape(self, grid: Tensor) -> None:
        self.shape = tuple(grid.shape)
        if len(self.shape) != 3:
            raise ValueError(f"fdtdx eps_r must be 3D, got shape {self.shape}")
        self.Nx, self.Ny, self.Nz = self.shape
        self.N = int(np.prod(self.shape))
