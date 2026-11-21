"""
MAPS-facing FDTD simulation classes backed by fdtdx.

This mirrors the public shape of ``core.fdfd.fdfd``: the class owns an external
``eps_r`` grid, accepts fdtdx-style source objects/specs in ``solve()``, and returns field component
grids. fdtdx is a 3D FDTD engine, so the concrete convenience class here is
``fdtd3d``. fdtdx-specific scene construction and adjoint-source construction
live behind the injected runtime object from ``core.fdtd.solver``.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import replace
from typing import Any, Dict, Sequence

try:
    from thirdparty.fdtdx import fdtdx
except ImportError:
    fdtdx = None
import jax
import jax.numpy as jnp
import numpy as np
import torch
from fdtdx import constants as fdtdx_constants
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from fdtdx.core.physics.metrics import normalize_by_poynting_flux
from torch import Tensor, nn

from core.env_args import get_env_arg
from core.utils import (
    _jax_to_torch,
    _torch_to_jax,
    build_rectilinear_grid_metadata,
    convert_to_fdtdx_grid_shape_loc_slice,
    resample_rectilinear_tensor,
    unfold_phasor,
    yee_to_colocate_interpolate,
)

from .solver import (
    _MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM,
    FDTDXSolveTorch,
    _derive_magnetic_phasor_from_electric,
    _normalize_phasor_by_source_spectrum,
    clone_phasor_detector,
    clone_phasor_detector_for_waves,
    native_h_slab_cells,
    phasor_detector_components,
)

__all__ = ["fdtd3d"]


def _split_complex_permittivity_for_fdtdx(
    eps_r: Tensor | np.ndarray,
    *,
    wavelength_m: float,
    time_step_duration: float,
    courant_number: float,
    electrical_conductivity_map: Tensor | np.ndarray | None = None,
) -> tuple[Tensor, Tensor | None]:
    r"""Convert MAPS complex permittivity to FDTDX material arrays.

    FDTDX evolves a real relative permittivity and represents nondispersive
    electric loss with its dimensionless conductivity update coefficient. For
    ``epsilon = epsilon' + 1j * epsilon''`` at the reference wavelength, this
    uses the same mapping as ``Material.from_complex_permittivity``:

    .. math::
        \sigma = \omega_0 \epsilon_0 \epsilon'',\qquad
        \sigma_\mathrm{fdtdx} = \sigma c_0 \Delta t / C.

    The returned conductivity is therefore ready for
    ``arrays.electric_conductivity``. An explicitly supplied conductivity map
    is interpreted in MAPS' conductivity-map units (S/um), converted to S/m,
    and added before this normalization.

    Args:
        eps_r: Relative permittivity map, real or complex, with spatial shape
            ``(Nx, Ny, Nz)``.
        wavelength_m: Reference free-space wavelength in metres.
        time_step_duration: FDTDX time step in seconds.
        courant_number: FDTDX Courant number ``C``.
        electrical_conductivity_map: Optional MAPS conductivity map in S/um.
            It is converted to S/m and combined with the conductivity derived
            from the imaginary permittivity.

    Returns:
        A pair ``(real_permittivity, dimensionless_conductivity)``. The second
        item is ``None`` when both the input permittivity and explicit map are
        lossless.

    Raises:
        ValueError: If the reference wavelength or Courant number is invalid.
    """
    eps_r = torch.as_tensor(eps_r)
    if wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be positive, got {wavelength_m}")
    if courant_number <= 0:
        raise ValueError(f"courant_number must be positive, got {courant_number}")

    if torch.is_complex(eps_r):
        real_permittivity = eps_r.real
        omega = 2.0 * np.pi * fdtdx_constants.c / wavelength_m
        conductivity = omega * fdtdx_constants.eps0 * eps_r.imag
    else:
        real_permittivity = eps_r
        conductivity = None

    if electrical_conductivity_map is not None:
        explicit_conductivity = torch.as_tensor(
            electrical_conductivity_map,
            device=real_permittivity.device,
            dtype=real_permittivity.dtype,
        )
        if torch.is_complex(explicit_conductivity):
            explicit_conductivity = explicit_conductivity.real
        # Device-base electrical conductivity maps use S/um, whereas the
        # FDTDX material conversion and SI constants use S/m.
        explicit_conductivity = explicit_conductivity * 1e6
        conductivity = (
            explicit_conductivity
            if conductivity is None
            else conductivity + explicit_conductivity
        )

    if conductivity is not None:
        conductivity = conductivity * (
            fdtdx_constants.c * time_step_duration / courant_number
        )
        conductivity = conductivity.to(dtype=real_permittivity.dtype)
    return real_permittivity, conductivity


_MAPS_ENABLE_PHASOR_H = get_env_arg("MAPS_ENABLE_PHASOR_H")
_MAPS_COMPARE_EXACT_INTERPOLATION = get_env_arg("MAPS_COMPARE_EXACT_INTERPOLATION")
_MAPS_SPARSE_PHASOR = get_env_arg("MAPS_SPARSE_PHASOR")
_MAPS_COMPARE_SPARSE_DENSE = get_env_arg("MAPS_COMPARE_SPARSE_DENSE")
_SPARSE_DENSE_REFERENCE_NAME = "sparse_dense_reference"


def _report_exact_interpolation_errors(
    *,
    fields: Tensor,
    arrays: Any,
    objects: Any,
    config: Any,
    source_template: Any,
) -> None:
    """Compare Torch colocation with native exact detector phasors."""
    component_names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    reference_name = None
    reference = None
    for name, state in arrays.detector_states.items():
        detector = objects[name]
        if not getattr(detector, "exact_interpolation", False):
            continue
        if not hasattr(detector, "components"):
            continue
        reference_name = name
        reference = _jax_to_torch(state["phasor"])[0]
        if _MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM:
            reference, _ = _normalize_phasor_by_source_spectrum(
                reference,
                source_template=source_template,
                monitor=detector,
                config=config,
            )
        reference = unfold_phasor(
            reference,
            monitor=detector,
            components=component_names,
            config=config,
        )
        break

    if reference is None:
        return

    comparison_slices = [(reference_name, (slice(None),) * 5)]
    for detector in objects.objects:
        name = detector.name
        if "mode_overlap_detector" not in name:
            continue
        grid_slice = getattr(detector, "grid_slice", None)
        if grid_slice is None:
            continue
        comparison_slices.append((name, (slice(None), slice(None), *grid_slice)))

    for name, indexer in comparison_slices:
        reference_view = reference[indexer]
        fields_view = fields[indexer]
        if reference_view.shape != fields_view.shape:
            print(
                f"[fdtdx exact-compare] {name}: shape mismatch "
                f"native={tuple(reference_view.shape)} torch={tuple(fields_view.shape)}",
                flush=True,
            )
            continue
        denominator = torch.linalg.vector_norm(
            reference_view.reshape(
                reference_view.shape[0], reference_view.shape[1], -1
            ),
            dim=2,
        )
        numerator = torch.linalg.vector_norm(
            (fields_view - reference_view).reshape(
                fields_view.shape[0], fields_view.shape[1], -1
            ),
            dim=2,
        )
        errors = numerator / denominator.clamp_min(torch.finfo(denominator.dtype).eps)
        print(
            f"[fdtdx exact-compare] {name}: "
            + ", ".join(
                f"{component}={errors[:, index].item():.6e}"
                for index, component in enumerate(component_names)
            ),
            flush=True,
        )


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


def _grid_bounds_lower_constraints(obj, bounds):
    """Anchor a known cell-shaped object at exact grid-edge indices.

    Args:
        obj: FDTD-X simulation object whose ``partial_grid_shape`` matches
            ``bounds``.
        bounds: Half-open cell bounds ``((start, stop), ...)`` for each axis.

    Returns:
        A list containing an index-space constraint for the lower edge on all
        three axes. FDTD-X derives the upper edges from the object's known cell
        shape, avoiding a lossy grid-index-to-real-coordinate round trip.
    """
    axes = tuple(range(len(bounds)))
    return [
        fdtdx.GridCoordinateConstraint(
            object=obj.name,
            axes=axes,
            sides=tuple("-" for _ in axes),
            coordinates=tuple(int(axis_bounds[0]) for axis_bounds in bounds),
        )
    ]


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
    """Convert a physical source or monitor rectangle to native cell slices.

    Args:
        boundaries: Three strictly increasing edge arrays in x, y, z order.
            Each axis contains one more edge than cells.
        center: Three physical center coordinates, in the same units as boundaries.
        size: Three physical extents, in the same units as boundaries.
        direction: Normal-axis direction, such as ``"z-"``. The normal slice
            selects the one cell center nearest the requested plane. ``None``
            selects a volume, covering the physical extent on every axis.
        transverse_rule: ``"cover"`` includes every cell overlapping the
            transverse rectangle. ``"center"`` selects cell centers inside it
            and may under-cover the requested physical boundary.
        clip: Whether to restrict slices to the simulation domain.

    Returns:
        Three half-open cell slices in x, y, z order.

    Raises:
        ValueError: If dimensions, edges, direction, transverse sizes, or the
            transverse selection rule are invalid.

    Notes:
        Edge-alignment tolerance is relative to physical coordinates rather
        than an absolute length, so nanometer overlaps are retained in SI units.
    """

    if len(boundaries) != 3:
        raise ValueError("boundaries must be a tuple/list of 3 edge arrays.")

    if len(center) != 3 or len(size) != 3:
        raise ValueError("center and size must be length-3 sequences.")

    if direction is not None and direction[0] not in "xyz":
        raise ValueError(f"Invalid direction {direction!r}. Expected e.g. 'x+'.")

    prop_axis = None if direction is None else "xyz".index(direction[0])

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
                if start + 1 < len(edges) and np.isclose(
                    edges[start + 1], lo, rtol=1e-12, atol=0.0
                ):
                    start = start + 1
                # An aligned upper edge can round infinitesimally below hi.
                # Retain the exact edge rather than adding the following cell.
                if stop > 0 and np.isclose(edges[stop - 1], hi, rtol=1e-12, atol=0.0):
                    stop = stop - 1

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


def normalize_fields_phase(E: Tensor, H: Tensor) -> tuple[Tensor, Tensor]:
    """
    Normalize the global phase of source fields.

    Args:
        E: complex jax array of shape [..., 3, X, Y, Z]
        H: complex jax array of shape [..., 3, X, Y, Z]

    Returns:
        E_norm, H_norm
    """
    assert (
        len(E.shape) >= 3 and E.shape == H.shape
    ), "E and H must have the same shape with at least 3 dimensions."

    if isinstance(E, torch.Tensor):
        E_s = E.squeeze()  # [3, H, W]
        H_s = H.squeeze()  # [3, H, W]
        # Total field magnitude
        mag = torch.sqrt(torch.sum(torch.abs(E_s) ** 2 + torch.abs(H_s) ** 2, dim=-3))

        # Location of maximum magnitude
        idx = torch.argmax(mag)
        h, w = mag.shape[-2:]
        y = idx // w
        x = idx % w

        # Strongest vector component
        field = torch.cat([E_s[..., y, x], H_s[..., y, x]])
        phase = torch.angle(field[torch.argmax(torch.abs(field))])

        phase_factor = torch.exp(-1j * phase)

        return E * phase_factor, H * phase_factor
    elif isinstance(E, jnp.ndarray):
        # Total field magnitude
        E_s = jnp.squeeze(E)  # [3, H, W]
        H_s = jnp.squeeze(H)  # [3, H, W]
        mag = jnp.sqrt(jnp.sum(jnp.abs(E_s) ** 2 + jnp.abs(H_s) ** 2, axis=-3))

        # Location of maximum magnitude
        idx = jnp.argmax(mag)
        y, x = jnp.unravel_index(idx, mag.shape[-2:])

        # Find the strongest vector component at that location
        field = jnp.concatenate([E_s[..., y, x], H_s[..., y, x]])
        phase = jnp.angle(field[jnp.argmax(jnp.abs(field))])

        phase_factor = jnp.exp(-1j * phase)

        return E * phase_factor, H * phase_factor


def _source_has_physical_size(source: Any) -> bool:
    """Check whether any source dimension still needs physical-size resolution.

    Args:
        source: Unplaced FDTD-X source with partial grid and physical shapes.

    Returns:
        Whether a dimension has a physical extent without an explicit cell
        count. Explicit cell counts take precedence over physical extents,
        including for adjoint sources carrying both representations.
    """
    return any(
        cells is None and extent is not None
        for cells, extent in zip(source.partial_grid_shape, source.partial_real_shape)
    )


def detector_native_grid_bounds(detector, boundaries, grid):
    """Resolve a detector aperture against edges at its physical location.

    Args:
        detector: Unplaced FDTD-X detector with partial shapes and position.
        boundaries: Three unreduced native edge arrays in meters.
        grid: Corresponding FDTD-X rectilinear grid, used to center axes with
            explicit cell counts.

    Returns:
        Three half-open ``(start, stop)`` cell bounds. Explicit cell counts
        take precedence over physical sizes. Physical intervals include all
        overlapping cells. Unspecified axes span the domain; unspecified
        centers default to the domain midpoint.

    Raises:
        ValueError: If a physical extent is invalid or a cell count cannot fit.
    """
    center = []
    size = []
    for axis, edges in enumerate(boundaries):
        midpoint = float((edges[0] + edges[-1]) / 2)
        position = detector.partial_real_position[axis]
        cells = detector.partial_grid_shape[axis]
        extent = detector.partial_real_shape[axis]
        center.append(midpoint if position is None else float(position))
        size.append(float(edges[-1] - edges[0]) if extent is None else float(extent))
        if cells is not None or extent is None:
            # These axes are resolved independently after physical coverage.
            center[axis] = midpoint
            size[axis] = float(edges[-1] - edges[0])
    slices = list(real_box_to_grid_slices(boundaries, center, size, None))
    for axis, cells in enumerate(detector.partial_grid_shape):
        if cells is not None:
            position = detector.partial_real_position[axis]
            if position is None:
                position = (boundaries[axis][0] + boundaries[axis][-1]) / 2
            lo, hi = grid.bounds_for_center(axis, float(position), int(cells))
            slices[axis] = slice(lo, hi)
    return tuple((index.start, index.stop) for index in slices)


def detector_native_placement(detector, boundaries, grid, *, bounds=None):
    """Create a detector clone and constraints fixing its native aperture.

    Args:
        detector: Unplaced FDTD-X detector prototype; it is not mutated.
        boundaries: Three unreduced native edge arrays in meters.
        grid: Corresponding FDTD-X rectilinear grid.
        bounds: Optional explicit half-open cell bounds overriding physical
            metadata, for example a padded design-region recorder support.

    Returns:
        Pair of detector clone and lower/upper physical-edge constraints. The
        clone has explicit native cell counts and no physical-center constraint,
        since a snapped aperture need not retain the requested exact center.

    Raises:
        ValueError: If bounds are empty or outside the native domain, or the
            requested physical aperture cannot be resolved.

    Notes:
        This helper does not initialize arrays or solve modes. Pass its result
        to ``fdtdx.place_objects``. Bounds are supplied before symmetry folding.
    """
    if bounds is None:
        bounds = detector_native_grid_bounds(detector, boundaries, grid)
    if len(bounds) != 3:
        raise ValueError("Detector bounds must contain three native axis intervals.")
    bounds = tuple((int(lo), int(hi)) for lo, hi in bounds)
    for axis, (lo, hi) in enumerate(bounds):
        if not 0 <= lo < hi <= len(boundaries[axis]) - 1:
            raise ValueError(f"Invalid detector bounds on axis {axis}: {(lo, hi)}")
    clone = detector.aset("partial_grid_shape", tuple(hi - lo for lo, hi in bounds))
    clone = clone.aset("partial_real_position", (None, None, None))
    constraints = _real_bounds_constraints(
        clone, _bounds_indices_to_real_bounds(bounds, boundaries)
    )
    return clone, constraints


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
        self.last_solve_field_grid: str | None = None
        self.last_solve_frequencies_hz: tuple[float, ...] | None = None
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

    def _monitor_real_bounds(self, monitor):
        """Return snapped native edge coordinates for a detector aperture.

        Args:
            monitor: Unplaced FDTD-X detector prototype.

        Returns:
            Three lower/upper coordinates in meters, honoring stored support
            bounds and explicit cell counts before physical extents.
        """
        return _bounds_indices_to_real_bounds(
            self._monitor_grid_bounds(monitor), self._native_boundaries()
        )

    def _monitor_grid_bounds(self, monitor):
        """Resolve native detector bounds, giving registered supports priority.

        Args:
            monitor: Unplaced FDTD-X detector prototype.

        Returns:
            Three half-open cell intervals on the unreduced native grid.
        """
        explicit_bounds = getattr(self, "_sparse_monitor_bounds", {}).get(monitor.name)
        if explicit_bounds is not None:
            return explicit_bounds
        return detector_native_grid_bounds(
            monitor, self._native_boundaries(), self._simulation_grid()
        )

    def _monitor_constraints(self, monitor, volume):
        explicit_bounds = getattr(self, "_sparse_monitor_bounds", {}).get(monitor.name)
        if not self._uses_nonuniform_grid():
            bounds = (
                explicit_bounds
                if explicit_bounds is not None
                else self._monitor_grid_bounds(monitor)
            )
            return _grid_bounds_lower_constraints(monitor, bounds)
        if explicit_bounds is not None:
            return _real_bounds_constraints(
                monitor,
                _bounds_indices_to_real_bounds(
                    explicit_bounds, self._native_boundaries()
                ),
            )
        return _real_bounds_constraints(monitor, self._monitor_real_bounds(monitor))

    def _profile_detectors(self, profile, *, mode, monitor_name):
        detectors = []
        for key, value in profile.items():
            if len(key) < 2 or key[1] != mode:
                continue
            candidate = value[0]
            if not isinstance(candidate, fdtdx.PhasorDetector):
                continue
            if len(value) > 4 and value[4] is not None:
                if not hasattr(self, "_sparse_monitor_bounds"):
                    self._sparse_monitor_bounds = {}
                self._sparse_monitor_bounds[candidate.name] = value[4]
            if candidate.name == monitor_name:
                continue
            detectors.append(candidate)
        return detectors

    def _sparse_monitor_extras(
        self,
        input_slice_name,
        mode,
        monitor_name=None,
        *,
        include_port_monitors=True,
        include_visualization_monitors=True,
    ):
        """Return localized phasor detectors not already in ``monitor_name``.

        Args:
            input_slice_name: Source-port name owning the sparse monitor profile.
            mode: Source mode used to select detector profile entries.
            monitor_name: Detector or detector collection already registered.
            include_port_monitors: Whether objective-port detectors are included.
            include_visualization_monitors: Whether forward-only cross-section
                detectors used by 3-D plotting are included.

        Returns:
            ``(detector, bounds)`` entries accepted by :meth:`prepare_arrays`.
            Bounds are resolved from the detector metadata during placement.
        """
        monitor_profile = self.port_sources_dict[input_slice_name + "_field_monitor"]
        if monitor_name is None:
            sparse_monitor_names = set()
        elif isinstance(monitor_name, (tuple, list)):
            sparse_monitor_names = {monitor.name for monitor in monitor_name}
        else:
            sparse_monitor_names = {getattr(monitor_name, "name", monitor_name)}
        sparse_monitors = []
        for candidate in self._profile_detectors(
            monitor_profile,
            mode=mode,
            monitor_name=monitor_name,
        ):
            if not include_port_monitors and candidate.name.startswith(
                "field_detector_port_"
            ):
                continue
            if not include_visualization_monitors and candidate.name.startswith(
                "field_detector_visualization_"
            ):
                continue
            if candidate.name not in sparse_monitor_names:
                sparse_monitor_names.add(candidate.name)
                sparse_monitors.append(candidate)
        return [(candidate, None) for candidate in sparse_monitors]

    def prepare_arrays(
        self,
        source,
        source_grid_bounds,
        source_center,
        source_size,
        monitor,
        extra_monitors=None,
        normalize_source_power=False,
        normalize_source_phase=False,
        direction: str | None = None,
        symmetry: tuple[int, int, int] = (0, 0, 0),
    ):
        eps_r = self.eps_r
        electrical_conductivity_map = self.electrical_conductivity_map
        if torch.is_complex(eps_r) and (self.n_wl != 1 or self.wl_width != 0.0):
            warnings.warn(
                "Complex FDTDX permittivity is being used with multiple or "
                "broadband wavelengths. The imaginary permittivity is represented "
                "by a static conductivity referenced to wl_cen, so the loss is "
                "exact only at that wavelength.",
                UserWarning,
                stacklevel=2,
            )
        if _MAPS_SPARSE_PHASOR:
            source_grid_bounds = tuple(
                (
                    (
                        int(axis_bounds.start),
                        int(axis_bounds.stop),
                    )
                    if isinstance(axis_bounds, slice)
                    else (int(axis_bounds[0]), int(axis_bounds[1]))
                )
                for axis_bounds in source_grid_bounds
            )

        if torch.is_complex(eps_r):
            eps_r = eps_r.real

        # every preparation is called before solve fdtd, and need one source
        objects = []
        constraints = []
        # print(f"SimulationVolume eps_r.shape: {eps_r.shape}")
        # Benchmark-only tensor probes need FDTDX to allocate the requested
        # 3- or 9-component permittivity layout before the material map
        # is installed. The tiny placeholder anisotropy is fully replaced by
        # override_inv_permittivities below; it is never part of the solved map.
        tensor_components = getattr(self, "_maps_force_permittivity_components", 1)
        if tensor_components not in {1, 3, 9}:
            raise ValueError(
                "FDTDX permittivity layout must have 1, 3, or 9 components, "
                f"got {tensor_components}"
            )
        if tensor_components == 3:
            volume_material = fdtdx.Material(permittivity=(1.0, 1.000001, 1.0))
        elif tensor_components == 9:
            volume_material = fdtdx.Material(
                permittivity=(1.0, 1e-6, 0.0, 1e-6, 1.0, 0.0, 0.0, 0.0, 1.0)
            )
        else:
            volume_material = fdtdx.Material(permittivity=1.0)
        volume = fdtdx.SimulationVolume(
            partial_grid_shape=eps_r.shape,
            material=volume_material,  # Dummy material, replaced by the map below
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

        physical_source = self._uses_nonuniform_grid() and _source_has_physical_size(
            source
        )
        if physical_source:
            bounds = tuple(
                (
                    (axis_bounds.start, axis_bounds.stop)
                    if isinstance(axis_bounds, slice)
                    else tuple(axis_bounds)
                )
                for axis_bounds in source_grid_bounds
            )
            source = self._object_with_native_shape(source, bounds)
            # The snapped physical aperture is now fully specified by native
            # cells. Re-resolving its center/length would use origin cell widths.
            source = source.aset("partial_real_position", (None, None, None))
            constraints.extend(
                _real_bounds_constraints(
                    source,
                    _bounds_indices_to_real_bounds(bounds, self._native_boundaries()),
                )
            )
        objects.append(source)  # source[0] is the source objective
        if _MAPS_SPARSE_PHASOR and not physical_source:
            constraints.append(
                source.place_relative_to(
                    volume,
                    axes=(0, 1, 2),
                    other_positions=(-1, -1, -1),
                    own_positions=(-1, -1, -1),
                    **self._placement_kwargs(source_grid_bounds),
                )
            )
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
        if not _MAPS_SPARSE_PHASOR:
            if isinstance(monitor, fdtdx.PhasorDetector):
                if self._uses_nonuniform_grid():
                    full_bounds = self._full_volume_bounds()
                    monitor, detector_constraints = detector_native_placement(
                        monitor,
                        self._native_boundaries(),
                        self._simulation_grid(),
                        bounds=full_bounds,
                    )
                    constraints.extend(detector_constraints)
                else:
                    constraints.extend(monitor.same_position_and_size(volume))
                objects.append(monitor)
            else:
                raise NotImplementedError(
                    "Only PhasorDetector is supported as monitor for source, "
                    f"but got {type(monitor)}"
                )
            if extra_monitors is not None:
                for extra_monitor, extra_grid_margins in extra_monitors:
                    if self._uses_nonuniform_grid():
                        extra_monitor, detector_constraints = detector_native_placement(
                            extra_monitor,
                            self._native_boundaries(),
                            self._simulation_grid(),
                            bounds=extra_grid_margins,
                        )
                        constraints.extend(detector_constraints)
                    elif extra_grid_margins is not None:
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
            resolved_monitors = [monitor]
        else:
            monitors = monitor if isinstance(monitor, (tuple, list)) else (monitor,)
            if not all(
                isinstance(candidate, fdtdx.PhasorDetector) for candidate in monitors
            ):
                raise NotImplementedError(
                    "Only PhasorDetector objects are supported as source monitors"
                )
            resolved_monitors = []
            for candidate in monitors:
                monitor_bounds = self._monitor_grid_bounds(candidate)
                if self._uses_nonuniform_grid():
                    candidate, detector_constraints = detector_native_placement(
                        candidate,
                        self._native_boundaries(),
                        self._simulation_grid(),
                        bounds=monitor_bounds,
                    )
                    constraints.extend(detector_constraints)
                elif monitor_bounds != self._full_volume_bounds():
                    candidate = self._object_with_native_shape(
                        candidate, monitor_bounds
                    )
                    constraints.extend(self._monitor_constraints(candidate, volume))
                else:
                    constraints.extend(candidate.same_position_and_size(volume))
                objects.append(candidate)
                resolved_monitors.append(candidate)
            if extra_monitors is not None:
                for extra_monitor, extra_grid_margins in extra_monitors:
                    if not isinstance(extra_monitor, fdtdx.PhasorDetector):
                        raise NotImplementedError(
                            "Only PhasorDetector is supported as auxiliary monitor"
                        )
                    monitor_bounds = (
                        extra_grid_margins
                        if extra_grid_margins is not None
                        else self._monitor_grid_bounds(extra_monitor)
                    )
                    if self._uses_nonuniform_grid():
                        extra_monitor, detector_constraints = detector_native_placement(
                            extra_monitor,
                            self._native_boundaries(),
                            self._simulation_grid(),
                            bounds=monitor_bounds,
                        )
                        constraints.extend(detector_constraints)
                    elif monitor_bounds == self._full_volume_bounds():
                        extra_monitor = self._object_with_native_shape(
                            extra_monitor, monitor_bounds
                        )
                        constraints.extend(extra_monitor.same_position_and_size(volume))
                    else:
                        extra_monitor = self._object_with_native_shape(
                            extra_monitor, monitor_bounds
                        )
                        constraints.extend(
                            self._monitor_constraints(extra_monitor, volume)
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

        eps_r, electrical_conductivity_map = _split_complex_permittivity_for_fdtdx(
            self.eps_r,
            wavelength_m=self.wl_cen * 1e-6,
            time_step_duration=config.time_step_duration,
            courant_number=config.courant_number,
            electrical_conductivity_map=electrical_conductivity_map,
        )

        inv_permittivities = _torch_to_jax(1 / eps_r[None, ...].data)
        if tensor_components == 3:
            inv_permittivities = jnp.broadcast_to(
                inv_permittivities, (3, *inv_permittivities.shape[1:])
            )
        elif tensor_components == 9:
            diagonal = inv_permittivities[0]
            inv_permittivities = jnp.zeros((9, *diagonal.shape), dtype=diagonal.dtype)
            inv_permittivities = inv_permittivities.at[0].set(diagonal)
            inv_permittivities = inv_permittivities.at[4].set(diagonal)
            inv_permittivities = inv_permittivities.at[8].set(diagonal)
        inv_permeabilities = 1.0

        # start = time.time()
        # constraints = self._sanitize_constraints_for_nonuniform(constraints)
        objects, arrays, params, config, info = fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            key=key,
            override_inv_permittivities=inv_permittivities,
            override_electric_conductivity=(
                None
                if electrical_conductivity_map is None
                else _torch_to_jax(electrical_conductivity_map[None, ...].data).astype(
                    config.dtype
                )
            ),
        )
        # print(f"fdtdx.place_objects takes {time.time() - start:.2f} seconds")
        # start = time.time()

        # arrays = arrays.at["inv_permittivities"].set(inv_permittivities)
        # arrays = arrays.at["inv_permeabilities"].set(inv_permeabilities)
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
            objects = objects.replace_sources([source])
        if normalize_source_phase:
            source = objects[source.name]
            # After apply_params, manually normalize by Poynting flux

            # find phase in the max magnitude point
            E_norm, H_norm = normalize_fields_phase(
                source._E,
                source._H,
            )
            source = source.aset("_E", E_norm)
            source = source.aset("_H", H_norm)
            objects = objects.replace_sources([source])
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
        placed_monitors = tuple(
            objects[candidate.name] for candidate in resolved_monitors
        )
        return (
            arrays,
            objects,
            config,
            key,
            placed_monitors[0] if len(placed_monitors) == 1 else placed_monitors,
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
        """Locate a source on the unreduced native optical grid.

        Args:
            source: FDTD-X source object before placement.
            source_center: Physical source center in meters.
            source_size: Physical source size in meters. Plane sources have
                zero extent along their one-cell normal axis. Explicit cell
                counts take precedence on grid-sized axes.

        Returns:
            Three half-open native slices. On nonuniform grids, physical axes
            cover their requested intervals at the actual source location;
            grid-sized axes retain their cell counts and select the interval
            nearest the physical center. Unspecified axes span the domain.
            Fully grid-sized sources and uniform grids retain the existing
            FDTD-X placement path. The source prototype is not modified.

        Raises:
            ValueError: If a physical extent is invalid or a grid-sized axis
                does not fit the native domain.
        """
        if self._uses_nonuniform_grid() and _source_has_physical_size(source):
            edges = self._native_boundaries()
            center = list(source_center)
            size = list(source_size)
            for axis, (cells, extent) in enumerate(
                zip(source.partial_grid_shape, source.partial_real_shape)
            ):
                if cells is not None or extent is None:
                    # Resolve physical axes only. Other axes temporarily span
                    # the domain, then explicit cell counts are restored below.
                    center[axis] = (edges[axis][0] + edges[axis][-1]) / 2
                    size[axis] = edges[axis][-1] - edges[axis][0]
            bounds = list(
                real_box_to_grid_slices(
                    edges,
                    center,
                    size,
                    None,
                    transverse_rule="cover",
                )
            )
            grid = self._simulation_grid()
            for axis, cells in enumerate(source.partial_grid_shape):
                if cells is not None:
                    lower, upper = grid.bounds_for_center(
                        axis, float(source_center[axis]), int(cells)
                    )
                    bounds[axis] = slice(lower, upper)
            return tuple(bounds)
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
            apply_objects=False,
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
        monitor_key = ((wl_cen, wl_width, n_wl), mode)
        if _MAPS_SPARSE_PHASOR:
            sparse_keys = [
                key
                for key in monitor_profile
                if len(key) == 3 and key[:2] == monitor_key
            ]
            monitors = [monitor_profile[key][0] for key in sparse_keys]
            if not monitors:
                raise KeyError(f"No sparse phasor monitors found for {monitor_key!r}")
            monitor = tuple(monitors)
        else:
            monitor, *_ = monitor_profile[monitor_key]
        desired_components = phasor_detector_components()
        if _MAPS_SPARSE_PHASOR:
            monitor = tuple(
                (
                    clone_phasor_detector(
                        candidate,
                        components=desired_components,
                        exact_interpolation=True,
                    )
                    if tuple(getattr(candidate, "components", ())) != desired_components
                    else candidate
                )
                for candidate in monitor
            )
        elif tuple(getattr(monitor, "components", ())) != desired_components:
            monitor = clone_phasor_detector(monitor, components=desired_components)

        native_h_monitor = None
        native_h_slice = None
        extra_monitors = []

        if _MAPS_COMPARE_EXACT_INTERPOLATION and not _MAPS_SPARSE_PHASOR:
            exact_monitor = clone_phasor_detector(
                monitor,
                name=f"{monitor.name}_exact_interpolation",
                components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                exact_interpolation=True,
            )
            extra_monitors.append((exact_monitor, None))

        if _MAPS_SPARSE_PHASOR:
            extra_monitors.extend(
                self._sparse_monitor_extras(
                    input_slice_name=input_slice_name,
                    mode=mode,
                    monitor_name=monitor,
                )
            )
            if self._uses_nonuniform_grid() and _source_has_physical_size(source):
                source_port_key = monitor_key + (f"port:{input_slice_name}",)
                if source_port_key in monitor_profile:
                    source_monitor_name = monitor_profile[source_port_key][0].name
                    source_bounds = tuple(
                        (axis.start, axis.stop) for axis in grid_slice_tuple
                    )
                    if not hasattr(self, "_sparse_monitor_bounds"):
                        self._sparse_monitor_bounds = {}
                    self._sparse_monitor_bounds[source_monitor_name] = source_bounds
                    monitor = tuple(
                        (
                            self._object_with_native_shape(
                                candidate, source_bounds
                            ).aset("partial_real_position", (None, None, None))
                            if candidate.name == source_monitor_name
                            else candidate
                        )
                        for candidate in monitor
                    )
        ## for debug purpose, we can add native fdtdx modeoverlapmonitors here
        # for name, extra_monitor_profile in self.port_sources_dict.items():
        #     if name == input_slice_name or name == input_slice_name + "_field_monitor":
        #         continue
        #     for wl in np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl):
        #         extra_monitor, *_ = extra_monitor_profile[(wl, mode)]
        #         print(
        #             f"Adding extra monitor {extra_monitor.name} for source {input_slice_name}"
        #         )
        #         extra_monitors.append((extra_monitor, None))

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
                normalize_source_phase=True,
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
    ) -> tuple[Tensor, ...] | dict[str, Tensor]:
        """Run an FDTD-X forward solve and return dense or sparse phasors.

        Both modes return the legacy six-component field tuple. Sparse mode
        records localized detector phasors internally, then scatters them into
        the same dense output contract.

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
        primary_forward_monitor = (
            fwd_field_monitor[0]
            if isinstance(fwd_field_monitor, (tuple, list))
            else fwd_field_monitor
        )

        def prepare_adjoint_simulation(source, source_grid_margins, monitor):
            adjoint_extras = []
            if _MAPS_SPARSE_PHASOR and hasattr(source, "aset"):
                source = source.aset(
                    "partial_grid_shape",
                    tuple(
                        int(axis_bounds[1] - axis_bounds[0])
                        for axis_bounds in source_grid_margins
                    ),
                )
            if _MAPS_SPARSE_PHASOR:
                if not hasattr(self, "_sparse_monitor_bounds"):
                    self._sparse_monitor_bounds = {}
                primary_bounds = self._sparse_monitor_bounds.get(
                    primary_forward_monitor.name
                )
                if primary_bounds is None:
                    primary_bounds = self._monitor_grid_bounds(primary_forward_monitor)
                self._sparse_monitor_bounds[monitor.name] = primary_bounds
            if _MAPS_SPARSE_PHASOR:
                placement_monitor = monitor
                adjoint_extras = self._sparse_monitor_extras(
                    input_slice_name=input_slice_name,
                    mode=mode,
                    monitor_name=(monitor, primary_forward_monitor),
                    include_port_monitors=False,
                    include_visualization_monitors=False,
                )
                adjoint_extras = [
                    (
                        clone_phasor_detector_for_waves(
                            candidate,
                            wave_characters=tuple(monitor.wave_characters),
                            components=tuple(monitor.components),
                            exact_interpolation=True,
                        ),
                        bounds,
                    )
                    for candidate, bounds in adjoint_extras
                ]
            else:
                placement_monitor = monitor
            arrays, objects, config, key, field_monitor, _ = self.prepare_arrays(
                source=source,
                source_grid_bounds=source_grid_margins,
                source_center=(0, 0, 0),
                source_size=self._native_extent(),
                monitor=placement_monitor,
                extra_monitors=adjoint_extras,
                symmetry=self.symmetry,
            )
            if _MAPS_SPARSE_PHASOR:
                field_monitor = objects[monitor.name]
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
        if _MAPS_SPARSE_PHASOR:
            self.phasor_states = {}
            self.fields_are_colocated = True
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
            self.last_solve_field_grid = (
                "export"
                if interpolate_to_export_grid
                and export_grid is not None
                and not metadata.get("native_equals_export", False)
                else "native"
            )
            self.last_solve_frequencies_hz = tuple(
                float(wave.get_frequency())
                for wave in primary_forward_monitor.wave_characters
            )
            Ex, Ey, Ez, Hx, Hy, Hz = fields.chunk(6, dim=1)
            return tuple(component.squeeze(1) for component in (Ex, Ey, Ez, Hx, Hy, Hz))
        self.phasor_states = {}
        native_exact = getattr(primary_forward_monitor, "exact_interpolation", False)
        if native_exact:
            native_e = fields[:, :3]
            if fields.shape[-4] == 6:
                native_h = fields[:, 3:]
            else:
                native_h = _derive_magnetic_phasor_from_electric(
                    native_e,
                    arrays=fwd_arrays,
                    objects=fwd_objects,
                    config=fwd_config,
                    wave_characters=primary_forward_monitor.wave_characters,
                )
            fields = torch.cat((native_e, native_h), dim=1)
            fields = unfold_phasor(
                fields,
                monitor=primary_forward_monitor,
                components=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
                config=fwd_config,
            )
        elif _MAPS_ENABLE_PHASOR_H and fields.shape[-4] == 6:
            native_e, native_h = fields.chunk(2, dim=-4)
        else:
            native_e = fields
            native_h = _derive_magnetic_phasor_from_electric(
                native_e,
                arrays=fwd_arrays,
                objects=fwd_objects,
                config=fwd_config,
                wave_characters=primary_forward_monitor.wave_characters,
            )
        # print("native_e shape:", native_e.shape)
        # print("native_h shape:", native_h.shape)

        if not native_exact:
            ## time-average of H and spatial colocation from the Yee grid.
            freq = torch.as_tensor(
                [wc.get_frequency() for wc in primary_forward_monitor.wave_characters],
                device=native_h.device,
                dtype=native_h.real.dtype,
            )

            omega = 2 * torch.pi * freq
            dt = fwd_config.time_step_duration
            theta = omega * dt / 2
            h_time_factor = (torch.exp(1j * theta) * torch.cos(theta)).reshape(
                -1, 1, 1, 1, 1
            )
            native_h = native_h * h_time_factor

            native_e = yee_to_colocate_interpolate(
                native_e, fwd_objects, fwd_config, is_E=True
            )
            native_h = yee_to_colocate_interpolate(
                native_h, fwd_objects, fwd_config, is_E=False
            )
            fields = torch.cat((native_e, native_h), dim=1)

            fields = unfold_phasor(
                fields,
                monitor=primary_forward_monitor,
                components=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
                config=fwd_config,
            )

        if _MAPS_COMPARE_EXACT_INTERPOLATION:
            _report_exact_interpolation_errors(
                fields=fields,
                arrays=fwd_arrays,
                objects=fwd_objects,
                config=fwd_config,
                source_template=fwd_objects.sources[0],
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
        self.last_solve_field_grid = (
            "export"
            if interpolate_to_export_grid
            and export_grid is not None
            and not metadata.get("native_equals_export", False)
            else "native"
        )
        self.last_solve_frequencies_hz = tuple(
            float(wave.get_frequency())
            for wave in primary_forward_monitor.wave_characters
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

    def to_tidy3d_data(
        self,
        fields: tuple[Tensor, ...] | dict[str, Tensor],
        simulation: Any,
        *,
        wavelengths: Sequence[float] | None = None,
        grid: str = "auto",
    ) -> Any:
        """Wrap solved MAPS phasors in Tidy3D's native data model.

        Args:
            fields: The six-component result returned by :meth:`solve`, or a
                mapping keyed by ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``.
            simulation: Existing Tidy3D ``Simulation`` whose structures define
                the visualization geometry.
            wavelengths: Optional sampled wavelengths in meters. By default,
                the exact frequencies from the most recent solve are used; if
                unavailable, the solver's source band is converted from micrometers.
            grid: ``"auto"``, ``"native"``, or ``"export"``. ``"auto"`` uses
                the grid recorded by the most recent :meth:`solve` call. For
                externally supplied fields it infers the grid only when exactly
                one coordinate set matches the field shape; ambiguous grids need
                an explicit choice.

        Returns:
            Tidy3D ``SimulationData`` with the MAPS fields attached.

        Notes:
            The solve output is already temporally corrected, colocated, unfolded,
            and optionally resampled. This method only transfers physical
            coordinates and values; it does not reapply Yee offsets. Tidy3D
            monitor flux and integration operations are not validated for these
            visualization data.
        """
        from core.visualization.tidy3d import maps_fields_to_tidy3d_data

        metadata = self._grid_metadata()
        if grid not in {"auto", "native", "export"}:
            raise ValueError("grid must be 'auto', 'native', or 'export'")
        field_values = list(fields.values()) if isinstance(fields, dict) else fields
        if not field_values:
            raise ValueError("fields cannot be empty")
        shape = tuple(int(v) for v in field_values[0].shape[-3:])
        candidates = {
            name: metadata[name]
            for name in ("native", "export")
            if name in metadata
            and tuple(shape) == tuple(len(axis) for axis in metadata[name]["coords"])
        }
        if grid == "auto":
            previous_grid = getattr(self, "last_solve_field_grid", None)
            if previous_grid in candidates:
                grid = previous_grid
            elif len(candidates) == 1:
                grid = next(iter(candidates))
            elif len(candidates) == 2 and all(
                np.array_equal(
                    candidates["native"]["coords"][axis],
                    candidates["export"]["coords"][axis],
                )
                for axis in range(3)
            ):
                grid = "native"
            elif len(candidates) > 1:
                raise ValueError(
                    "Field shape matches both native and export grids, whose coordinates differ; "
                    "pass grid='native' or grid='export' explicitly"
                )
            else:
                raise ValueError(
                    "Field grid shape does not match MAPS native or export coordinates"
                )
        if grid not in candidates:
            raise ValueError(
                f"Field shape {shape} does not match the requested {grid} grid"
            )
        selected_grid = candidates[grid]
        # Tidy3D geometry/data in the MAPS workflows use micrometers. Optical
        # metadata is already in micrometers; the fallback grid is built from
        # fdtd3d.dL in meters and needs the corresponding conversion.
        scale = 1.0 if self.optical_grid_metadata is not None else 1e6
        coords = tuple(np.asarray(axis) * scale for axis in selected_grid["coords"])
        boundaries = tuple(
            np.asarray(axis) * scale for axis in selected_grid["boundaries"]
        )
        count = int(field_values[0].shape[0])
        if wavelengths is None:
            last_frequencies = getattr(self, "last_solve_frequencies_hz", None)
            if last_frequencies is not None and len(last_frequencies) == count:
                frequencies = np.asarray(last_frequencies, dtype=float)
            else:
                wavelength_samples = 1e-6 * np.linspace(
                    self.wl_cen - self.wl_width / 2,
                    self.wl_cen + self.wl_width / 2,
                    count,
                )
                if not np.all(np.isfinite(wavelength_samples)) or np.any(
                    wavelength_samples <= 0
                ):
                    raise ValueError(
                        "Configured wavelength band must be finite and positive"
                    )
                frequencies = 299792458.0 / wavelength_samples
        else:
            wavelength_samples = np.asarray(wavelengths, dtype=float)
            if wavelength_samples.shape != (count,) or not np.all(
                np.isfinite(wavelength_samples)
            ):
                raise ValueError(
                    f"Expected {count} finite wavelengths for the supplied phasors"
                )
            if np.any(wavelength_samples <= 0):
                raise ValueError("Wavelengths must be positive meters")
            frequencies = 299792458.0 / wavelength_samples
        return maps_fields_to_tidy3d_data(
            simulation,
            fields,
            coords=coords,
            frequencies=frequencies,
            boundaries=boundaries,
        )

    def plot_3d(
        self,
        fields: tuple[Tensor, ...] | dict[str, Tensor],
        simulation: Any,
        *,
        field: str | None = "Ey",
        wavelength: float | None = None,
        quantity: str = "abs",
        plane: tuple[str, float] = ("z", 0.0),
        planes: Sequence[tuple[str, float]] | None = None,
        geometry_opacity: float = 0.25,
        wavelengths: Sequence[float] | None = None,
        grid: str = "auto",
    ) -> Any:
        """Display geometry and a MAPS field slice in an interactive notebook.

        Args:
            fields: Six-component result from :meth:`solve`, or a component map.
            simulation: Existing Tidy3D ``Simulation`` carrying the geometry.
            field: Component ``Ex``...``Hz``, vector magnitude ``E``/``H``, or
                ``None`` to show geometry only.
            wavelength: Wavelength in meters to display. The nearest stored
                frequency is selected; defaults to the first sample.
            quantity: ``real``, ``imag``, ``abs``, ``abs2``, or ``phase``.
            plane: Field slice as ``('x'|'y'|'z', position_in_meters)``. The
                position is converted to the micrometer coordinates used by MAPS.
            planes: Optional multiple field planes. The first plane is movable;
                later planes remain fixed.
            geometry_opacity: Initial opacity between zero and one.
            wavelengths: Optional wavelengths in meters corresponding to the
                frequency dimension. By default frequencies from the latest solve
                are used, with the configured source band as fallback.
            grid: ``"auto"``, ``"native"``, or ``"export"`` field coordinates.

        Returns:
            An ipywidgets container with camera and field controls.

        Notes:
            Rotate, pan, and zoom are provided by the Matplotlib ``ipympl``
            notebook backend. The Tidy3D-native 3D viewer only renders geometry
            and does not expose field overlays to Python, so this method uses a
            small inline Matplotlib fallback. Geometry is visualized from the
            supplied Tidy3D structures; only Box and straight, z-extruded
            PolySlab meshes are drawn. Rasterized CustomMedium structures are
            skipped instead of being shown as misleading bounding boxes.
        """
        from core.visualization.tidy3d import plot_tidy3d_fields_3d

        if wavelength is not None and (not np.isfinite(wavelength) or wavelength <= 0):
            raise ValueError("wavelength must be a finite positive value in meters")
        data = self.to_tidy3d_data(
            fields, simulation, wavelengths=wavelengths, grid=grid
        )
        plane_um = (plane[0], float(plane[1]) * 1e6)
        planes_um = (
            None
            if planes is None
            else [(axis, float(position) * 1e6) for axis, position in planes]
        )
        return plot_tidy3d_fields_3d(
            data,
            field=field,
            quantity=quantity,
            wavelength=wavelength,
            plane=plane_um,
            planes=planes_um,
            geometry_opacity=geometry_opacity,
        )

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
