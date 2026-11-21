"""
Date: 2024-10-02 20:59:04
LastEditors: Jiaqi Gu && jqgu@utexas.edu
LastEditTime: 2026-09-02 23:48:52
FilePath: /MAPS/core/invdes/models/layers/device_base.py
"""

import copy
import inspect
import logging
import os
from functools import lru_cache
from typing import Any, List, Sequence, Tuple

try:
    from thirdparty.fdtdx import fdtdx
except ImportError:
    fdtdx = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None
import numpy as np
import torch
from pyutils.config import Config
from pyutils.general import ensure_dir
from torch.nn import functional as F

from core.env_args import get_env_arg
from core.fdfd import fdfd_ez as fdfd_ez_torch
from core.fdfd import fdfd_hz as fdfd_hz_torch
from core.heat.transfer import conservative_rectilinear_remap
from core.invdes.models.layers.utils import (
    apply_line_edge_roughness,
    modulation_fn_dict,
    slice_to_indices,
)
from core.utils import (
    Si_eps,
    SiO2_eps,
    Slice,
    Slice3D,
    _design_region_visualization_bounds,
    _jax_to_torch,
    _origin_grid_cell_bounds,
    build_rectilinear_grid_metadata,
    convert_to_fdtdx_grid_shape_loc_slice,
    get_electrical_conductivity_fn,
    get_flux,
    get_heat_capacity_fn,
    get_thermal_conductivity_fn,
    get_thermo_optic_coeff_fn,
    grid_plane_weights,
    resample_rectilinear_tensor,
)
from thirdparty.ceviche import fdfd_ez, fdfd_hz
from thirdparty.ceviche.constants import C_0, MICRON_UNIT

try:
    import tidy3d as td

    td.config.logging.level = "ERROR"

    TD_SUPPORTED = True
except ImportError:
    TD_SUPPORTED = False

try:
    import meshio

    MESHIO_SUPPORTED = True
except ImportError:
    meshio = None
    MESHIO_SUPPORTED = False

try:
    from core.heat import HeatSolver

    HEAT_SUPPORTED = True
except Exception:
    HeatSolver = None
    HEAT_SUPPORTED = False

from .utils import insert_mode, plot_eps_field

__all__ = ["BaseDevice", "N_Ports"]

_AXES = ("x", "y", "z")
_MAPS_SPARSE_PHASOR = get_env_arg("MAPS_SPARSE_PHASOR", context="device")
_MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES = get_env_arg(
    "MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES", context="device"
)
_MAPS_ENABLE_PHASOR_H = get_env_arg("MAPS_ENABLE_PHASOR_H", context="device")
_PHASOR_COMPONENTS = (
    ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    if _MAPS_ENABLE_PHASOR_H
    else ("Ex", "Ey", "Ez")
)
_HEAT_PROPERTY_KEYS = ("thermal_conductivity", "k")
_HEAT_PROPERTY_BG_KEYS = ("thermal_conductivity_bg", "k_bg")
_ELECTRICAL_CONDUCTIVITY_KEYS = ("electrical_conductivity", "sigma")
_ELECTRICAL_CONDUCTIVITY_BG_KEYS = ("electrical_conductivity_bg", "sigma_bg")
_HEAT_CAPACITY_KEYS = ("heat_capacity", "capacity")
_HEAT_CAPACITY_BG_KEYS = ("heat_capacity_bg", "capacity_bg")
_THERMO_OPTIC_KEYS = ("thermo_optic_coeff", "dn_dT")
_THERMO_OPTIC_BG_KEYS = ("thermo_optic_coeff_bg", "dn_dT_bg")
_PROPERTY_KEY_SPECS = {
    "conductivity": (_HEAT_PROPERTY_KEYS, _HEAT_PROPERTY_BG_KEYS),
    "electrical_conductivity": (
        _ELECTRICAL_CONDUCTIVITY_KEYS,
        _ELECTRICAL_CONDUCTIVITY_BG_KEYS,
    ),
    "heat_capacity": (_HEAT_CAPACITY_KEYS, _HEAT_CAPACITY_BG_KEYS),
    "thermo_optic_coeff": (_THERMO_OPTIC_KEYS, _THERMO_OPTIC_BG_KEYS),
}
_HEAT_SOURCE_VALUE_KEYS = ("heat_density", "q", "q_density", "source")
_HEAT_SOURCE_DUMMY_CFG_KEY = "_maps_dummy_heat_source"
_HEAT_SOURCE_DUMMY_ENCODED_OFFSET = 1e-2
_MISSING = object()


def _infer_dim_from_cfgs(*cfg_groups, cell_size=None) -> int:
    def _iter_cfgs(cfg):
        yield cfg
        for key in ("geometries",):
            for child in cfg.get(key, []) or []:
                if isinstance(child, dict):
                    yield from _iter_cfgs(child)
        for key in ("geometry", "geometry_a", "geometry_b"):
            child = cfg.get(key)
            if isinstance(child, dict):
                yield from _iter_cfgs(child)

    if cell_size not in (None, "None"):
        return 3 if len(cell_size) >= 3 and cell_size[2] != 0 else 2
    for cfgs in cfg_groups:
        for cfg in cfgs.values():
            for child_cfg in _iter_cfgs(cfg):
                for key in ("center", "size"):
                    value = child_cfg.get(key)
                    if (
                        isinstance(value, Sequence)
                        and not isinstance(value, str)
                        and len(value) >= 3
                    ):
                        return 3
                if "vertices" in child_cfg and any(
                    len(v) >= 3 for v in child_cfg["vertices"]
                ):
                    return 3
                if child_cfg.get("height", 0) not in (
                    0,
                    None,
                    float("inf"),
                ) and np.isfinite(child_cfg.get("height", 0)):
                    return 3
    return 2


def _as_3(values, fill: float = 0.0) -> tuple[float, float, float]:
    values = tuple(values)
    if len(values) > 3:
        raise ValueError(f"Expected at most 3 coordinates, got {values}")
    return tuple(values + (fill,) * (3 - len(values)))


def _vector(values, fill: float = 0.0):
    mp = _import_meep()
    return mp.Vector3(*_as_3(values, fill=fill))


def _axis_index(direction: str) -> int:
    axis = direction[0].lower()
    if axis not in _AXES:
        raise ValueError(f"Direction {direction} not supported")
    return _AXES.index(axis)


def _tidy3d_medium_from_permittivity(permittivity, *, freq: float):
    eps = complex(permittivity)
    if np.isclose(eps.imag, 0.0, atol=1e-12, rtol=1e-12) and eps.real >= 1.0:
        return td.Medium(permittivity=float(eps.real))

    n_complex = np.lib.scimath.sqrt(eps)
    if n_complex.real < 0:
        n_complex = -n_complex
    if n_complex.imag < 0:
        n_complex = np.conj(n_complex)

    return td.medium_from_nk(
        n=float(n_complex.real),
        k=float(n_complex.imag),
        freq=float(freq),
    )


def _axis_pairs(values, dim: int, default: float = 0.0) -> list[float]:
    values = list(values or [])
    if len(values) == 2 * dim:
        return values
    if len(values) == dim:
        return [v for value in values for v in (value, value)]
    if dim == 3 and len(values) == 4:
        return values + [default, default]
    if dim == 2 and len(values) == 6:
        return values[:4]
    if not values:
        return [default] * (2 * dim)
    raise ValueError(f"Expected {dim} or {2 * dim} values, got {values}")


def _axis_values(values, dim: int, default: float = 0.0) -> list[float]:
    values = list(values or [])
    if len(values) == dim:
        return values
    if dim == 3 and len(values) == 2:
        return values + [default]
    if dim == 2 and len(values) >= 2:
        return values[:2]
    if not values:
        return [default] * dim
    raise ValueError(f"Expected {dim} values, got {values}")


def _spacing_values(
    values, dim: int, default: float | None = None
) -> tuple[float, ...]:
    if values is None:
        if default is None:
            raise ValueError("Spacing values are required.")
        return tuple([float(default)] * dim)
    if isinstance(values, (int, float)):
        return tuple([float(values)] * dim)
    values = tuple(float(v) for v in values)
    if len(values) == dim:
        return values
    if dim == 3 and len(values) == 2:
        fill = float(default if default is not None else values[-1])
        return values + (fill,)
    if dim == 2 and len(values) >= 2:
        return values[:2]
    raise ValueError(f"Expected {dim} spacing values, got {values}")


def _slice_region(slices: Sequence[slice]):
    return (
        Slice(x=slices[0], y=slices[1])
        if len(slices) == 2
        else Slice3D(x=slices[0], y=slices[1], z=slices[2])
    )


def _pixel_coverage_slice_from_centers(
    start: float,
    stop: float,
    centers: np.ndarray,
    atol: float = 1e-9,
) -> tuple[slice, np.ndarray]:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1:
        raise ValueError(f"Expected 1D grid centers, got shape {centers.shape}")
    if centers.size == 0:
        return slice(0, 0), np.zeros((0,), dtype=np.float32)
    if centers.size == 1:
        step = 1.0
        edges = np.asarray([centers[0] - step / 2, centers[0] + step / 2])
    else:
        midpoints = (centers[:-1] + centers[1:]) / 2
        left_edge = centers[0] - (centers[1] - centers[0]) / 2
        right_edge = centers[-1] + (centers[-1] - centers[-2]) / 2
        edges = np.concatenate([[left_edge], midpoints, [right_edge]])

    coverage = (np.minimum(edges[1:], stop) - np.maximum(edges[:-1], start)) / (
        edges[1:] - edges[:-1]
    )
    coverage = np.clip(coverage, 0.0, 1.0)
    covered = coverage > atol
    if not np.any(covered):
        insert_at = int(np.searchsorted(centers, (start + stop) / 2))
        return slice(insert_at, insert_at), np.zeros((0,), dtype=np.float32)
    first = int(np.argmax(covered))
    last = int(len(covered) - np.argmax(covered[::-1]))
    return slice(first, last), coverage[first:last].astype(np.float32)


def _outer_coverage(axis_weights: Sequence[np.ndarray]) -> np.ndarray:
    weight = np.ones(tuple(len(w) for w in axis_weights), dtype=np.float32)
    for axis, axis_weight in enumerate(axis_weights):
        shape = [1] * len(axis_weights)
        shape[axis] = axis_weight.size
        weight *= axis_weight.reshape(shape)
    return weight


def _pixel_coverage_slice_from_boundaries(
    start: float,
    stop: float,
    boundaries: np.ndarray,
    atol: float = 1e-9,
) -> tuple[slice, np.ndarray]:
    boundaries = np.asarray(boundaries, dtype=float)
    if boundaries.ndim != 1 or boundaries.size < 2:
        raise ValueError(
            f"Expected 1D grid boundaries with at least 2 entries, got shape {boundaries.shape}"
        )
    cell_lengths = boundaries[1:] - boundaries[:-1]
    coverage = (
        np.minimum(boundaries[1:], stop) - np.maximum(boundaries[:-1], start)
    ) / (cell_lengths)
    coverage = np.clip(coverage, 0.0, 1.0)
    covered = coverage > atol
    if not np.any(covered):
        cell_centers = 0.5 * (boundaries[:-1] + boundaries[1:])
        insert_at = int(np.searchsorted(cell_centers, (start + stop) / 2))
        return slice(insert_at, insert_at), np.zeros((0,), dtype=np.float32)
    first = int(np.argmax(covered))
    last = int(len(covered) - np.argmax(covered[::-1]))
    return slice(first, last), coverage[first:last].astype(np.float32)


def _centers_to_boundaries(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 1:
        raise ValueError("centers must be a non-empty 1D array.")
    if centers.size == 1:
        half = 0.5
        return np.asarray([centers[0] - half, centers[0] + half], dtype=np.float64)
    mid = 0.5 * (centers[:-1] + centers[1:])
    left = centers[0] - 0.5 * (centers[1] - centers[0])
    right = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate([[left], mid, [right]]).astype(np.float64)


def _slice_from_dense_weight(
    weight: np.ndarray,
    atol: float = 1e-6,
) -> tuple[tuple[slice, ...], np.ndarray]:
    covered = weight > atol
    if not np.any(covered):
        empty = tuple(slice(0, 0) for _ in range(weight.ndim))
        return empty, weight[empty].astype(np.float32)

    slices = []
    for axis in range(weight.ndim):
        reduce_axes = tuple(i for i in range(weight.ndim) if i != axis)
        axis_covered = np.any(covered, axis=reduce_axes)
        first = int(np.argmax(axis_covered))
        last = int(len(axis_covered) - np.argmax(axis_covered[::-1]))
        slices.append(slice(first, last))
    slices = tuple(slices)
    return slices, weight[slices].astype(np.float32)


def _cfg_scalar(
    cfg: dict,
    keys: Sequence[str],
    *,
    default=_MISSING,
):
    for key in keys:
        if key in cfg:
            return cfg[key]
    if default is _MISSING:
        raise KeyError(f"None of keys {tuple(keys)} found in cfg")
    return default


def _freeze_structure(value):
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_structure(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_structure(v) for v in value)
    return value


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_heat_mesh_type(mesh_type) -> str:
    normalized = str(mesh_type or "rectangular").strip().lower().replace("-", "_")
    aliases = {
        "rectangle": "rectangular",
        "structured": "rectangular",
        "fixed_distance_unstructured_tidy3d": "fixed_distance_unstructured_tidy3d",
        "distance_unstructured_tidy3d": "fixed_distance_unstructured_tidy3d",
        "tidy3d_distance_unstructured": "fixed_distance_unstructured_tidy3d",
        "fixed_rectilinear_nonuniform": "fixed_rectilinear_nonuniform",
        "rectilinear_nonuniform": "fixed_rectilinear_nonuniform",
        "nonuniform_rectilinear": "fixed_rectilinear_nonuniform",
    }
    return aliases.get(normalized, normalized)


def _geometry_cfg_is_compound(cfg: dict) -> bool:
    geo_type = cfg.get("type")
    if geo_type in {
        "group",
        "geometry_group",
        "clip",
        "clip_operation",
        "boolean",
        "compound",
    }:
        return True
    if "geometries" in cfg:
        return True
    return "geometry_a" in cfg and "geometry_b" in cfg


def _import_fdtdx_runtime():
    import fdtdx

    from core.fdtd import fdtd3d

    return fdtdx, fdtd3d


def _import_meep():
    import meep as mp

    return mp


class SimulationConfig(Config):
    def __init__(self):
        super().__init__()
        self.update(
            dict(
                device=dict(
                    type="",
                    cfg=dict(),
                ),
                sources=[],
                simulation=dict(),
            )
        )


class BaseDevice(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = SimulationConfig()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.sources = []
        self.heat_source_cfgs = {}
        self.heat_sources_dict = {}
        self.heat_source_map = None
        self._runtime_heat_source_map_cache = []
        self.geometry = {}
        self.sim = None

    def build_ports(self):
        ### build geometry for input/output ports
        """
        Hook for building input/output port geometry on a base device.

        Returns:
            None in the base implementation. Concrete device subclasses override this method when port geometry must be constructed from higher-level device configuration.
        """
        pass

    def update_device_config(self, device_type, device_cfg):
        """
        Replace the active device type and device-configuration payload.

        Args:
            device_type: String identifying the concrete device representation.
            device_cfg: Serializable configuration dictionary associated with the device.

        Returns:
            None. ``self.config.device.type`` and ``self.config.device.cfg`` are updated in-place.
        """
        self.config.device.type = device_type
        self.config.device.update(dict(cfg=device_cfg))

    def reset_device_config(self):
        """
        Reset the stored device type and device configuration to empty defaults.

        Returns:
            None. ``self.config.device.type`` becomes an empty string and ``self.config.device.cfg`` becomes an empty dictionary.
        """
        self.config.device.type = ""
        self.config.device.update(dict(cfg=dict()))

    def add_source_config(self, source_config):
        """
        Append one optical source configuration to the device configuration.

        Args:
            source_config: Serializable source-configuration object/dictionary consumed by the active solver backend.

        Returns:
            None. The object is appended to ``self.config.sources`` without solver construction.
        """
        self.config.sources.append(source_config)

    def reset_source_config(self):
        """
        Remove all stored optical source configurations.

        Returns:
            None. ``self.config.sources`` becomes an empty list.
        """
        self.config.sources = []

    def add_heat_source_config(self, source_name, source_config):
        """
        Register or replace one named thermal heat-source configuration.

        Args:
            source_name: Unique key used to identify the heat source.
            source_config: Serializable configuration for the heat source geometry/intensity/material coupling.

        Returns:
            None. The source is stored in ``self.heat_source_cfgs`` and the runtime heat-source cache is invalidated.
        """
        self.heat_source_cfgs[source_name] = source_config
        self._runtime_heat_source_map_cache = []

    def reset_heat_source_config(self):
        """
        Reset all thermal heat-source configuration and cached runtime maps.

        Returns:
            None. Clears ``heat_source_cfgs``, ``heat_sources_dict``, ``heat_source_map``, and the runtime heat-source map cache.
        """
        self.heat_source_cfgs = {}
        self.heat_sources_dict = {}
        self.heat_source_map = None
        self._runtime_heat_source_map_cache = []

    def update_simulation_config(self, simulation_config):
        """
        Replace the active solver/simulation configuration payload.

        Args:
            simulation_config: Serializable dictionary containing simulation settings such as resolution, wavelength, cell size, PML, and backend-specific options.

        Returns:
            None. ``self.config.simulation`` is replaced with the supplied mapping.
        """
        self.config.update(dict(simulation=simulation_config))

    def reset_simulation_config(self):
        """
        Reset the stored simulation configuration to an empty dictionary.

        Returns:
            None.
        """
        self.config.update(dict(simulation=dict()))

    def dump_config(self, filepath, verbose=False):
        """
        Serialize the current device configuration to a YAML file.

        Args:
            filepath: Destination YAML path. Parent directories are created when needed.
            verbose: If true, print a confirmation message after writing the file.

        Returns:
            None. The active ``self.config`` is serialized; simulation/device runtime tensors are not exported by this method.
        """
        ensure_dir(os.path.dirname(filepath))
        self.config.dump_to_yml(filepath)
        if verbose:
            print(f"Dumped device config to {filepath}")

    def trim_pml(self, resolution, PML, x):
        """
        Remove PML cells from the boundary of a dense field/material tensor.

        Args:
            resolution: Grid resolution in pixels per physical length unit, used to convert the requested physical PML thickness into integer cell counts.
            PML: Scalar or per-axis physical PML thickness specification.
            x: Tensor/array whose trailing spatial dimensions correspond to the simulation grid. The implementation supports 2-D and 3-D spatial arrays.

        Returns:
            Tensor/array with the same leading dimensions as ``x`` and with the PML cells removed from every spatial axis.
        """
        dim = x.ndim if x.ndim in (2, 3) else min(3, x.ndim)
        pml = [int(round(i * resolution)) for i in _axis_values(PML, dim)]
        slices = tuple(slice(v, -v if v > 0 else None) for v in pml)
        return x[(..., *slices)]


def _port_cross_section_box_cfgs(
    device,
    *,
    direction: str,
    plane_coordinate: float,
):
    """Extract box-material cross sections for an auxiliary port device.

    Args:
        device: Source device containing recentered geometry configurations.
        direction: Port propagation direction, such as ``"x"`` or ``"z-"``.
        plane_coordinate: Physical coordinate of the mode plane along the
            propagation axis.

    Returns:
        Geometry configurations for boxes intersecting the mode plane. Each
        selected box is extruded across the auxiliary cell along the
        propagation axis, producing a translation-invariant mode cross section.

    Notes:
        Non-box geometries are intentionally omitted. Copying a finite curved
        or compound object into the normalization device would introduce a
        scatterer rather than its local cross section. Port backgrounds such as
        buried and substrate layers are represented as boxes.
    """
    axis = _axis_index(direction)
    dim = int(device.dim)
    cell_size = tuple(float(value) for value in device.cell_size)
    cross_section_cfgs = {}
    tolerance = 1e-9

    for name, cfg in device.geometry_cfgs.items():
        if cfg.get("_exclude_from_port_mode", False):
            continue
        if str(cfg.get("type", "box")).lower() != "box":
            continue
        center = list(cfg.get("center", ()))
        size = list(cfg.get("size", ()))
        if len(center) < dim or len(size) < dim:
            continue

        axis_size = size[axis]
        intersects_plane = axis_size is None
        if axis_size is not None:
            axis_size = float(axis_size)
            intersects_plane = not np.isfinite(axis_size) or (
                abs(float(plane_coordinate) - float(center[axis]))
                <= axis_size / 2 + tolerance
            )
        if not intersects_plane:
            continue

        cross_section_cfg = copy.deepcopy(cfg)
        cross_section_center = list(cross_section_cfg["center"])
        cross_section_size = list(cross_section_cfg["size"])
        cross_section_center[axis] = 0.0
        cross_section_size[axis] = cell_size[axis]
        cross_section_cfg["center"] = cross_section_center
        cross_section_cfg["size"] = cross_section_size
        cross_section_cfg.pop("roughness", None)
        cross_section_cfgs[f"port_cross_section_{name}"] = cross_section_cfg

    return cross_section_cfgs


def get_two_ports(device, port_name, slice_name=None):
    """
    Construct an auxiliary ``N_Ports`` device representing one optical port and its associated monitor slice.

    Args:
        device: Existing MAPS device supplying port configuration, simulation configuration, native grid metadata, epsilon background, and monitor-slice information.
        port_name: Key into ``device.port_cfgs`` selecting the port to materialize.
        slice_name: Optional monitor-slice name. When provided, only that compatible slice is copied.

    Returns:
        A new ``N_Ports`` object whose cell extent/grid is aligned with the source device and whose port geometry is oriented along the configured port direction.

    Notes:
        The auxiliary device is not allowed to reuse the source ``Slice`` object directly because slice indices can depend on the target grid. Instead, compatible monitor slices are reconstructed from their physical center/size/direction coordinates. Symmetry settings are removed when they are incompatible with the requested port location.

        Box geometries intersecting the requested mode plane are copied and
        extruded along the propagation axis. Consequently, buried layers,
        substrates, and other box-based cross-section materials participate in
        mode solving without copying finite downstream structures into the
        normalization simulation.

    Example:
        ``port_device = get_two_ports(device, "in_port")``
        creates the single-port normalization device used by modal source/reference construction.
    """
    port = device.port_cfgs[port_name]
    center = list(port["center"])
    size = list(port["size"])
    direction = port["direction"]
    direction_axis = direction[0].lower()
    eps = port["eps"]
    sidewall_angle = port.get("sidewall_angle", 0.0)
    cell_size = device.cell_size
    normal_axis = _axis_index(direction)
    plane_coordinate = float(center[normal_axis])
    if slice_name is not None:
        slice_info = device.port_monitor_slices_info.get(slice_name)
        if slice_info is not None:
            plane_coordinate = float(slice_info["center"][normal_axis])
    cross_section_geometry_cfgs = _port_cross_section_box_cfgs(
        device,
        direction=direction,
        plane_coordinate=plane_coordinate,
    )
    if len(center) == 3:
        if direction_axis == "x":
            center = [0, center[1], center[2]]
            size = [cell_size[0], size[1], size[2]]
        elif direction_axis == "y":
            center = [center[0], 0, center[2]]
            size = [size[0], cell_size[1], size[2]]
        elif direction_axis == "z":
            center = [center[0], center[1], 0]
            size = [size[0], size[1], cell_size[2]]
        else:
            raise ValueError(f"Direction {direction} not supported")
    elif len(center) == 2:
        if direction_axis == "x":
            center = [0, center[1]]
            size = [cell_size[0], size[1]]
        elif direction_axis == "y":
            center = [center[0], 0]
            size = [size[0], cell_size[1]]
        else:
            raise ValueError(f"Direction {direction} not supported")
    else:
        raise ValueError(f"Center must be 2D or 3D, got {center}")
    sim_cfg = copy.deepcopy(device.sim_cfg)
    sim_cfg["cell_size"] = device.cell_size
    if "optical_grid" in device.sim_cfg:
        sim_cfg["optical_grid"] = copy.deepcopy(device.sim_cfg["optical_grid"])
    native_grid = device.grid_info_dict.get("epsilon_map", {}).get("grid")

    if "symmetry" in sim_cfg:
        remove_symmetry = False
        symmetry = sim_cfg["symmetry"]
        if symmetry[0] != 0:
            if center[0] != 0:
                remove_symmetry = True
        if symmetry[1] != 0:
            if center[1] != 0:
                remove_symmetry = True
        if remove_symmetry:
            sim_cfg.pop("symmetry")

    two_ports = N_Ports(
        eps_bg=device.eps_bg,
        port_cfgs={
            port_name: dict(
                type="box",
                direction=direction,
                center=center,
                size=size,
                eps=eps,
                sidewall_angle=sidewall_angle,
            ),
        },
        geometry_cfgs=cross_section_geometry_cfgs,
        design_region_cfgs=dict(),
        sim_cfg=sim_cfg,
        grid=native_grid,
        grid_info=device.grid_info_dict.get("epsilon_map"),
        device=device.device,
    )
    # Recreate the source monitor on the auxiliary device.  Reusing the
    # source Slice object is unsafe because it may be indexed for a different
    # grid; add_monitor_slice derives fresh native/export indices from the
    # shared physical grid coordinates.
    requested_slice_name = slice_name
    for monitor_name, slice_info in device.port_monitor_slices_info.items():
        if requested_slice_name is not None and monitor_name != requested_slice_name:
            continue
        if slice_info.get("direction", "")[0] != direction[0]:
            continue
        two_ports.add_monitor_slice(
            monitor_name,
            slice_info["center"],
            slice_info["size"],
            slice_info["direction"],
        )
    return two_ports


def calculate_pml_thickness_for_face(
    edges,
    desired_physical_thickness,
    direction: str = "+",
    *,
    min_cells: int = 0,
    return_actual: bool = False,
):
    """
    Calculate the number of boundary cells whose accumulated physical thickness best matches a target PML thickness.

    Args:
        edges: Strictly increasing 1-D cell-boundary coordinates with shape ``[N + 1]``. Units must match ``desired_physical_thickness``.
        desired_physical_thickness: Target physical PML thickness in the same units as ``edges``.
        direction: Boundary side to accumulate from: ``"-"`` starts at the minimum coordinate and ``"+"`` starts at the maximum coordinate.
        min_cells: Minimum number of boundary cells allowed.
        return_actual: If true, return both the selected cell count and its actual accumulated physical thickness.

    Returns:
        Integer number of PML cells when ``return_actual=False``; otherwise ``(num_cells, actual_thickness)``.

    Raises:
        ValueError: If ``edges`` are not strictly increasing, ``direction`` is invalid, or ``min_cells`` is negative.

    Notes:
        The selection minimizes ``abs(cumulative_thickness - desired_physical_thickness)`` over all admissible cell counts. This is the nonuniform-grid analogue of rounding ``desired_thickness / dx`` on a uniform grid.
    """

    edges = np.asarray(edges)
    cell_sizes = np.diff(edges)

    if np.any(cell_sizes <= 0):
        raise ValueError("Grid edges must be strictly increasing.")

    if direction == "-":
        boundary_cell_sizes = cell_sizes
    elif direction == "+":
        boundary_cell_sizes = cell_sizes[::-1]
    else:
        raise ValueError("direction must be '-' or '+'.")

    cumulative = np.concatenate([[0.0], np.cumsum(boundary_cell_sizes)])
    # cumulative[N] = physical thickness using N cells

    # Enforce min_cells.
    min_cells = int(min_cells)
    if min_cells < 0:
        raise ValueError("min_cells must be non-negative.")

    valid_counts = np.arange(min_cells, len(cumulative))

    if valid_counts.size == 0:
        num_cells = len(boundary_cell_sizes)
    else:
        errors = np.abs(cumulative[valid_counts] - desired_physical_thickness)
        num_cells = int(valid_counts[np.argmin(errors)])

    actual_thickness = float(cumulative[num_cells])

    if return_actual:
        return num_cells, actual_thickness

    return num_cells


def calculate_pml_grid_thickness_from_physical(
    pml,
    grid_boundaries,
    *,
    min_cells: int = 0,
    return_actual: bool = False,
) -> List[Tuple[int, int]]:
    """
    Calculate minimum/maximum PML cell counts independently for every simulation axis.

    Args:
        pml: Per-axis desired physical PML thickness. The sequence length must match the number of axes in ``grid_boundaries``.
        grid_boundaries: Sequence of 1-D boundary-coordinate arrays; each array has shape ``[N_axis + 1]`` and is strictly increasing.
        min_cells: Minimum number of cells allowed on each face.
        return_actual: If true, each face result is returned as ``(num_cells, actual_thickness)``.

    Returns:
        List of ``(min_face, max_face)`` pairs, one pair per axis. With ``return_actual=False`` each entry is an integer cell count; with ``True`` each entry is a tuple returned by ``calculate_pml_thickness_for_face``.
    """
    NPML = []

    for axis, desired in enumerate(pml):
        min_face = calculate_pml_thickness_for_face(
            grid_boundaries[axis],
            desired,
            direction="-",
            min_cells=min_cells,
            return_actual=return_actual,
        )
        max_face = calculate_pml_thickness_for_face(
            grid_boundaries[axis],
            desired,
            direction="+",
            min_cells=min_cells,
            return_actual=return_actual,
        )
        NPML.append((min_face, max_face))

    return NPML


def recenter_geometries(geometry_cfgs, extends):
    ### extends: [(xmin, xmax), (ymin, ymax)] or [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
    ## geometry_cfgs contains the geometry configurations.
    ## we need to adjust the geometry center by shifting the structures such that extends are centered at (0,0,0)
    """
    Translate configured geometries so their enclosing extent is centered at the origin.

    Args:
        geometry_cfgs: Mapping from geometry name to mutable geometry configuration dictionaries.
        extends: Axis-aligned physical extents in the form ``[(xmin, xmax), (ymin, ymax)]`` or ``[(xmin, xmax), (ymin, ymax), (zmin, zmax)]``.

    Returns:
        None. The configuration dictionaries are modified in-place.

    Notes:
        For ``sine_bend`` geometries, the offsets are applied according to the configured propagation axis so transverse start/end coordinates and the slab bounds remain physically aligned. Other supported geometry types use their center/position representation. This helper is intended to establish the device-coordinate convention before rasterization.
    """
    offset_x = -(extends[0][0] + extends[0][1]) / 2
    offset_y = -(extends[1][0] + extends[1][1]) / 2
    offset_z = -(extends[2][0] + extends[2][1]) / 2 if len(extends) == 3 else 0

    for name, geometry_cfg in geometry_cfgs.items():
        if geometry_cfg["type"] == "sine_bend":
            if geometry_cfg["axis"] == 0:
                geometry_cfg["start"] = (
                    geometry_cfg["start"][0] + offset_y,
                    geometry_cfg["start"][1] + offset_z,
                )
                geometry_cfg["end"] = (
                    geometry_cfg["end"][0] + offset_y,
                    geometry_cfg["end"][1] + offset_z,
                )
                geometry_cfg["slab_bounds"] = (
                    geometry_cfg["slab_bounds"][0] + offset_x,
                    geometry_cfg["slab_bounds"][1] + offset_x,
                )

            elif geometry_cfg["axis"] == 1:
                geometry_cfg["start"] = (
                    geometry_cfg["start"][0] + offset_x,
                    geometry_cfg["start"][1] + offset_z,
                )
                geometry_cfg["end"] = (
                    geometry_cfg["end"][0] + offset_x,
                    geometry_cfg["end"][1] + offset_z,
                )
                geometry_cfg["slab_bounds"] = (
                    geometry_cfg["slab_bounds"][0] + offset_y,
                    geometry_cfg["slab_bounds"][1] + offset_y,
                )
            elif geometry_cfg["axis"] == 2:
                geometry_cfg["start"] = (
                    geometry_cfg["start"][0] + offset_x,
                    geometry_cfg["start"][1] + offset_y,
                )
                geometry_cfg["end"] = (
                    geometry_cfg["end"][0] + offset_x,
                    geometry_cfg["end"][1] + offset_y,
                )
                geometry_cfg["slab_bounds"] = (
                    geometry_cfg["slab_bounds"][0] + offset_z,
                    geometry_cfg["slab_bounds"][1] + offset_z,
                )
            else:
                raise ValueError(
                    f"Invalid axis {geometry_cfg['axis']} for sine_bend geometry."
                )
        else:
            center = geometry_cfg["center"]
            new_center_x = center[0] + offset_x
            new_center_y = center[1] + offset_y
            if len(center) == 3:
                new_center_z = center[2] + offset_z
                geometry_cfg["center"] = [new_center_x, new_center_y, new_center_z]
            else:
                geometry_cfg["center"] = [new_center_x, new_center_y]

    new_extends = [
        (extends[0][0] + offset_x, extends[0][1] + offset_x),
        (extends[1][0] + offset_y, extends[1][1] + offset_y),
    ]
    if len(extends) == 3:
        new_extends.append((extends[2][0] + offset_z, extends[2][1] + offset_z))
    return geometry_cfgs, new_extends


def build_sine_bend_cell(cfg):
    import gdstk

    start = np.array(cfg["start"], dtype=float)
    end = np.array(cfg["end"], dtype=float)
    direction = cfg.get("direction", "x")  # "x" for x-faced, "y" for y-faced
    n = int(cfg.get("num_samples", 100))
    width = float(cfg["width"])

    if n < 3:
        raise ValueError("num_samples must be >= 3")
    if direction not in ("x", "y"):
        raise ValueError(f"direction must be 'x' or 'y', got {direction!r}")

    if direction == "x":
        x_start = start[0]
        l_bend = end[0] - start[0]
        x_bend = np.linspace(
            x_start, x_start + l_bend, 100
        )  # x coordinates of the top edge vertices
        h_bend = abs(end[1] - start[1])
        y_bend = (
            (x_bend - x_start) * h_bend / l_bend
            - h_bend * np.sin(2 * np.pi * (x_bend - x_start) / l_bend) / (np.pi * 2)
        ) + (
            start[1] if end[1] > start[1] else -start[1]
        )  # y coordinates of the top edge vertices

        # add path to the cell
        cell = gdstk.Cell("bends")
        # print(start, end)
        # print(y_bend)
        cell.add(
            gdstk.FlexPath(
                x_bend + (1j if end[1] > start[1] else -1j) * y_bend,
                width,
                layer=1,
                datatype=0,
            )
        )
    elif direction == "y":
        y_start = start[1]
        l_bend = end[1] - start[1]
        y_bend = np.linspace(
            y_start, y_start + l_bend, 100
        )  # y coordinates of the top edge vertices
        h_bend = end[0] - start[0]
        x_bend = (
            (y_bend - y_start) * h_bend / l_bend
            - h_bend * np.sin(2 * np.pi * (y_bend - y_start) / l_bend) / (np.pi * 2)
            + start[0]
            if end[0] > start[0]
            else -start[0]
        )  # x coordinates of the top edge vertices

        # add path to the cell
        cell = gdstk.Cell("bends")
        cell.add(
            gdstk.FlexPath(
                (1 if end[0] > start[0] else -1) * x_bend + 1j * y_bend,
                width,
                layer=1,
                datatype=0,
            )
        )
    else:
        raise ValueError(f"direction must be 'x' or 'y', got {direction!r}")

    return cell


class N_Ports(BaseDevice):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        orig_init = cls.__init__

        # Avoid double-wrapping if the file is reloaded.
        if getattr(orig_init, "_captures_constructor_kwargs", False):
            return

        sig = inspect.signature(orig_init)

        def wrapped_init(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()

            # Save kwargs accepted by the real subclass constructor.
            constructor_kwargs = {
                k: copy.deepcopy(v) for k, v in bound.arguments.items() if k != "self"
            }

            orig_init(self, *args, **kwargs)

            # Store after init so N_Ports.__init__ cannot overwrite it.
            self._constructor_kwargs = constructor_kwargs

        wrapped_init._captures_constructor_kwargs = True
        wrapped_init.__name__ = orig_init.__name__
        wrapped_init.__qualname__ = orig_init.__qualname__
        wrapped_init.__doc__ = orig_init.__doc__

        cls.__init__ = wrapped_init

    def __init__(
        self,
        eps_bg: float = SiO2_eps(1.55),
        k_bg: float | None = None,
        electrical_conductivity_bg: float | None = None,
        heat_capacity_bg: float | None = None,
        thermo_optic_coeff_bg: float | None = None,
        port_cfgs=dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-1.5, 0],
                size=[3, 0.48],
                eps=Si_eps(1.55),
            ),
            out_port_1=dict(
                type="box",
                direction="x",
                center=[1.5, 0],
                size=[3, 0.48],
                eps=Si_eps(1.55),
            ),
        ),
        geometry_cfgs=dict(),
        design_region_cfgs=dict(
            region_1=dict(
                type="box",
                center=[0, 0],
                size=[1, 1],
                eps_bg=SiO2_eps(1.55),
                eps=Si_eps(1.55),
            )
        ),
        heat_source_cfgs=dict(),
        active_region_cfgs=dict(),
        sim_cfg: dict = {
            "border_width": [
                0,
                0,
                1.5,
                1.5,
            ],  # left, right, lower, upper, containing PML
            "PML": [1, 1],  # left/right, lower/upper
            "cell_size": None,
            "resolution": 50,
            "wl_cen": 1.55,
            "wl_width": 0,
            "n_wl": 1,
            "plot_root": "./figs",
        },
        grid=None,
        grid_info=None,
        device="cuda:0",
        verbose: bool = True,
    ):
        super().__init__()

        self.eps_bg = eps_bg
        self.k_bg = k_bg
        self.electrical_conductivity_bg = electrical_conductivity_bg
        self.heat_capacity_bg = heat_capacity_bg
        self.thermo_optic_coeff_bg = thermo_optic_coeff_bg
        self.port_cfgs = port_cfgs
        self.geometry_cfgs = geometry_cfgs

        self.design_region_cfgs = design_region_cfgs
        self.heat_source_cfgs = heat_source_cfgs
        self.active_region_cfgs = active_region_cfgs

        self.resolution = sim_cfg["resolution"]
        self.grid_step = 1 / self.resolution
        self.dim = _infer_dim_from_cfgs(
            port_cfgs,
            geometry_cfgs,
            design_region_cfgs,
            active_region_cfgs,
            cell_size=sim_cfg.get("cell_size"),
        )
        self.axes = _AXES[: self.dim]

        device_cfg = dict(
            port_cfgs=port_cfgs,
            geometry_cfgs=geometry_cfgs,
            eps_bg=eps_bg,
            k_bg=k_bg,
            electrical_conductivity_bg=electrical_conductivity_bg,
            heat_capacity_bg=heat_capacity_bg,
            thermo_optic_coeff_bg=thermo_optic_coeff_bg,
            heat_source_cfgs=heat_source_cfgs,
            resolution=self.resolution,
            grid_step=self.grid_step,
            dim=self.dim,
        )
        self.device = device
        self.verbose = verbose
        super().__init__(**device_cfg)
        self.update_device_config(self.__class__.__name__, device_cfg)
        self.update_simulation_config(sim_cfg)
        self.sim_cfg = sim_cfg
        # Optional native raster grid supplied by an existing device.  This
        # is primarily used by normalization's extended two-port device so
        # that both devices have identical native coordinates and cell widths.
        self.native_grid = grid
        self.native_grid_info = grid_info
        self.material_property_maps = {}
        self.conductivity_map = None
        self.electrical_conductivity_map = None
        self.heat_capacity_map = None
        self.thermal_capacity_map = None
        self.thermo_optic_coeff_map = None
        self.thermal_coefficient_map = None
        self.thermal_grid_spacing = None
        self.thermal_grid_shape = None
        self.thermal_coords = None
        self.thermal_design_region_masks = {}
        self.thermal_design_region_mask_weights = {}
        self.thermal_design_region_axis_weights = {}
        self.native_epsilon_map = None
        self.export_epsilon_map = None
        self.export_grid_metadata = None
        self.fdtdx_native_grid_metadata = None
        self.fdtdx_field_grid_metadata = None
        self.fdtdx_native_cell_weights = None
        self.fdtdx_native_design_region_masks = {}
        self.fdtdx_native_design_region_mask_weights = {}
        self.fdtdx_native_design_region_axis_weights = {}
        self.material_property_backend = None
        self._electrical_conductivity_map_built = False
        self._heat_solver = None
        self._heat_solver_signature = None
        self._heat_solution_cache = None
        self._heat_solution_cache_interval = None
        self._heat_solution_query_count = 0
        self._heat_fixed_mesh = None
        self._heat_fixed_mesh_signature = None
        self.grid_info_dict = {}
        self._thermal_raster_grid_spec = None
        self.heat_source_cfgs = copy.deepcopy(heat_source_cfgs)
        self.heat_sources_dict = {}
        self.heat_source_map = None
        self._runtime_heat_source_map_cache = []

        explicit_use_tidy3d = self.sim_cfg.get("use_tidy3d")
        if explicit_use_tidy3d is None:
            explicit_use_tidy3d = self._heat_sim_cfg().get("use_tidy3d")
        use_tidy3d = (
            bool(explicit_use_tidy3d)
            if explicit_use_tidy3d is not None
            else self.resolution >= 1
        ) and TD_SUPPORTED
        if self._cfgs_require_tidy3d(port_cfgs) or self._cfgs_require_tidy3d(
            geometry_cfgs
        ):
            if not TD_SUPPORTED:
                raise ImportError(
                    "Compound geometry cfgs require tidy3d, but tidy3d is not installed."
                )
            use_tidy3d = True
        self.override_structures = {}
        self.add_geometries(port_cfgs, use_tidy3d=use_tidy3d, apply_roughness=False)
        self.add_geometries(geometry_cfgs, use_tidy3d=use_tidy3d, apply_roughness=False)
        ## do not add design region to geometry, otherwise meep will have subpixel smoothing on the border
        ## but need to consider this in bounding box
        # self.add_geometries(design_region_cfgs)
        self.add_override_structures(design_region_cfgs, use_tidy3d=use_tidy3d)
        if self.sim_cfg["cell_size"] is None or self.sim_cfg["cell_size"] == "None":
            self.cell_size, self.cell_extend = self.get_geometry_box(
                border_width=sim_cfg["border_width"], PML=sim_cfg["PML"]
            )
        else:
            self.cell_size = tuple(_axis_values(sim_cfg["cell_size"], self.dim))
            self.cell_extend = tuple((-s / 2, s / 2) for s in self.cell_size)
        if self.native_grid_info is not None:
            boundaries = self.native_grid_info["boundaries"]
            self.cell_extend = tuple(
                (float(axis[0]), float(axis[-1])) for axis in boundaries[: self.dim]
            )
            self.cell_size = tuple(
                float(axis[-1] - axis[0]) for axis in boundaries[: self.dim]
            )
        self.cell_center = tuple((s[0] + s[1]) / 2 for s in self.cell_extend)
        ### here we use ceil to match meep

        ## we need to recenter the geometry to centered at (0,0,0)
        recenter_geometries(self.design_region_cfgs, self.cell_extend)
        recenter_geometries(self.port_cfgs, self.cell_extend)
        recenter_geometries(self.geometry_cfgs, self.cell_extend)
        self.geometry = {}
        self.override_structures = {}

        self.add_geometries(port_cfgs, use_tidy3d=use_tidy3d, apply_roughness=False)
        self.add_geometries(geometry_cfgs, use_tidy3d=use_tidy3d, apply_roughness=False)
        ## do not add design region to geometry, otherwise meep will have subpixel smoothing on the border
        ## but need to consider this in bounding box
        # self.add_geometries(design_region_cfgs)
        self.add_override_structures(design_region_cfgs, use_tidy3d=use_tidy3d)
        if self.sim_cfg["cell_size"] is None or self.sim_cfg["cell_size"] == "None":
            self.cell_size, self.cell_extend = self.get_geometry_box(
                border_width=sim_cfg["border_width"], PML=sim_cfg["PML"]
            )
        else:
            self.cell_size = tuple(_axis_values(sim_cfg["cell_size"], self.dim))
            self.cell_extend = tuple((-s / 2, s / 2) for s in self.cell_size)
        if self.native_grid_info is not None:
            boundaries = self.native_grid_info["boundaries"]
            self.cell_extend = tuple(
                (float(axis[0]), float(axis[-1])) for axis in boundaries[: self.dim]
            )
            self.cell_size = tuple(
                float(axis[-1] - axis[0]) for axis in boundaries[: self.dim]
            )
        self.cell_center = tuple((s[0] + s[1]) / 2 for s in self.cell_extend)
        if not np.allclose(self.cell_center, (0,) * self.dim, atol=1e-9):
            logging.warning(
                f"Cell center: {self.cell_center} is not at origin (0,0,0)."
            )

        ## we rebuild geometry with roughness
        ## because we do not want roughness to change cell_extend and center.
        self.geometry = {}
        self.override_structures = {}
        self.add_geometries(port_cfgs, use_tidy3d=use_tidy3d, apply_roughness=True)
        self.add_geometries(geometry_cfgs, use_tidy3d=use_tidy3d, apply_roughness=True)
        self.add_override_structures(design_region_cfgs, use_tidy3d=use_tidy3d)

        self.build_epsilon_map(use_tidy3d=use_tidy3d)
        # self.epsilon_map_unique_count = int(np.unique(self.epsilon_map).size)
        # print(
        #     f"Initial epsilon_map unique values: {self.epsilon_map_unique_count}"
        # )

        self.grid_shape = tuple(self.epsilon_map.shape)
        if len(self.grid_shape) != self.dim:
            self.dim = len(self.grid_shape)
            self.axes = _AXES[: self.dim]
        self.Nx = self.grid_shape[0]
        self.Ny = self.grid_shape[1]
        self.Nz = self.grid_shape[2] if self.dim == 3 else 1
        ## PML should be calculated on optical native grid, not on export grid
        self.NPML_export = [
            int(round(i * self.resolution))
            for i in _axis_values(sim_cfg["PML"], self.dim)
        ]
        self.NPML = calculate_pml_grid_thickness_from_physical(
            pml=sim_cfg["PML"],
            grid_boundaries=self.grid_info_dict.get("epsilon_map")["boundaries"],
        )
        epsilon_grid_info = self.grid_info_dict.get("epsilon_map")
        if epsilon_grid_info is not None:
            self.coords = tuple(
                np.asarray(axis, dtype=np.float64)
                for axis in epsilon_grid_info["coords"]
            )
        else:
            self.coords = tuple(
                np.linspace(
                    -(n - 1) / 2 * self.grid_step,
                    (n - 1) / 2 * self.grid_step,
                    n,
                )
                for n in self.grid_shape
            )
        self._sync_optical_grid_metadata()
        self.meshes = np.meshgrid(*self.coords, indexing="ij")
        self.xs, self.ys = self.meshes[:2]

        self.design_region_masks = self.build_design_region_mask(design_region_cfgs)
        self._sync_optical_grid_metadata()
        ## active region must within the design region
        self.active_region_masks = self.build_active_region_mask(active_region_cfgs)
        self.ports_regions = self.build_port_region(port_cfgs)

        self.port_monitor_slices = {}  # {port_name: Slice or mask} # on export grid
        self.port_monitor_slices_native = (
            {}
        )  # {port_name: Slice or mask} # on native grid
        self.port_monitor_slices_native_symmetry = (
            {}
        )  # {port_name: Slice or mask} # on native grid, but reduced due to symmetry
        self.port_monitor_slices_export = (
            {}
        )  # {port_name: Slice or mask} # on export grid
        self.port_monitor_slices_info = {}  # {port_name: dict of slice info}
        self.port_sources_dict = (
            {}
        )  # {slice_name: {(wl, mode): (profile, ht_m, et_m, norm_power)}}

    def add_geometries(
        self,
        cfgs,
        use_tidy3d: bool = False,
        apply_roughness: bool = True,
    ):
        """
        Add configured geometries to the appropriate device dictionary.

        Args:
            cfgs: Mapping from geometry name to geometry configuration dictionary.
                Configurations with ``override_structure=True`` are stored in
                ``self.override_structures`` instead of ``self.geometry``.
            use_tidy3d: Whether geometry construction should use the Tidy3D geometry representation when supported.
            apply_roughness: If true, apply the configured line-edge roughness transform before geometry construction.

        Returns:
            None. Each geometry is constructed and stored under its configuration
            name. Configurations with a false or absent ``override_structure``
            flag remain ordinary physical geometry.
        """
        for name, cfg in cfgs.items():
            geometry_cfg = (
                apply_line_edge_roughness(cfg)
                if apply_roughness and cfg.get("roughness") is not None
                else cfg
            )
            if cfg.get("override_structure", False):
                self.add_override_structures(
                    {name: geometry_cfg}, use_tidy3d=use_tidy3d
                )
            else:
                self.add_geometry(name, geometry_cfg, use_tidy3d=use_tidy3d)

    def add_geometry(self, name, cfg, use_tidy3d: bool = False):
        """
        Build one geometry object from a configuration dictionary and register it in ``self.geometry``.

        Args:
            name: Geometry key used in the device geometry mapping.
            cfg: Geometry configuration. Supported types include the concrete geometry primitives handled by ``_build_geometry_from_cfg`` such as boxes, prisms, sine bends, cylinders, and compound forms.
            use_tidy3d: Whether to request the Tidy3D representation when that backend is available.

        Returns:
            None. The resulting geometry object is stored as ``self.geometry[name]``.

        Notes:
            ``cfg`` controls geometry only; the design-region mask itself is handled separately so that solver-specific subpixel/rasterization behavior is not inadvertently applied to the optimization boundary.
        """
        geometry = self._build_geometry_from_cfg(
            cfg,
            value_key="eps",
            bg_key="eps_bg",
            use_tidy3d=use_tidy3d,
        )
        self.geometry[name] = geometry

    def add_override_structures(self, cfgs, use_tidy3d: bool = False):
        """
        Build geometry objects that override the material/background map without adding them to the primary geometry collection.

        Args:
            cfgs: Mapping from override-structure name to geometry configuration.
            use_tidy3d: Whether to construct backend-specific Tidy3D geometry when supported.

        Returns:
            None. Each constructed object is stored in ``self.override_structures``.

        Notes:
            Override structures are used for regions such as design-region material overrides where the geometry must affect the material representation without becoming an ordinary geometry primitive in the main device collection.
        """
        for name, cfg in cfgs.items():
            geometry = self._build_geometry_from_cfg(
                cfg,
                value_key="eps",
                bg_key="eps_bg",
                use_tidy3d=use_tidy3d,
            )
            self.override_structures[name] = geometry

    def _tidy3d_geometry_from_cfg(self, cfg):
        geo_type = cfg["type"]
        ## convert None to td.inf
        size = []
        for i in range(len(cfg.get("size", []))):
            if cfg["size"][i] is None:
                size.append(td.inf)
            else:
                size.append(cfg["size"][i])
        if geo_type == "box":
            sidewall_angle = cfg.get("sidewall_angle")
            if sidewall_angle is not None and len(cfg.get("size", [])) == 3:
                # Public APIs use degrees; Tidy3D uses the same signed angle in radians.
                angle_radians = np.deg2rad(float(sidewall_angle))
                axis = int(cfg.get("axis", 2))
                if axis not in (0, 1, 2):
                    raise ValueError("PolySlab axis must be 0, 1, or 2.")
                size = [float(value) for value in cfg["size"]]
                center = [float(value) for value in cfg["center"]]
                normal_origin = center[axis] - size[axis] / 2.0
                normal_top = center[axis] + size[axis] / 2.0
                transverse_axes = [index for index in range(3) if index != axis]
                u, v = transverse_axes
                half_u = size[u] / 2.0
                half_v = size[v] / 2.0
                vertices = [
                    (center[u] - half_u, center[v] - half_v),
                    (center[u] + half_u, center[v] - half_v),
                    (center[u] + half_u, center[v] + half_v),
                    (center[u] - half_u, center[v] + half_v),
                ]

                return td.PolySlab(
                    vertices=vertices,
                    slab_bounds=(normal_origin, normal_top),
                    sidewall_angle=angle_radians,
                    reference_plane="middle",
                    axis=axis,
                )
            return td.Box(
                size=_as_3(size, fill=td.inf if self.dim == 2 else 0),
                center=_as_3(cfg["center"], fill=0),
            )
        if geo_type == "prism":
            sidewall_angle = cfg.get("sidewall_angle")
            slab_bounds = cfg.get("slab_bounds", (-td.inf, td.inf))
            angle_radians = (
                None if sidewall_angle is None else np.deg2rad(float(sidewall_angle))
            )

            return td.PolySlab(
                vertices=cfg["vertices"],
                slab_bounds=slab_bounds,
                **(
                    {
                        "sidewall_angle": angle_radians,
                        "reference_plane": "top",
                        "axis": int(cfg.get("axis", 2)),
                    }
                    if sidewall_angle is not None
                    else {}
                ),
            )
        if geo_type == "sine_bend":
            cell = build_sine_bend_cell(cfg)
            sidewall_angle = cfg.get("sidewall_angle")
            angle_radians = (
                0.0 if sidewall_angle is None else np.deg2rad(float(sidewall_angle))
            )
            return td.PolySlab.from_gds(
                cell,
                gds_layer=1,
                axis=cfg.get("axis", 2),
                slab_bounds=cfg.get("slab_bounds", None),
                sidewall_angle=angle_radians,
                reference_plane="middle" if sidewall_angle is not None else "middle",
            )[0]
        if geo_type == "cylinder":
            cylinder_kwargs = dict(
                radius=cfg["radius"],
                length=cfg.get("height", td.inf),
                center=_as_3(cfg.get("center", (0, 0, 0)), fill=0),
            )
            if "axis" in cfg:
                cylinder_kwargs["axis"] = int(cfg["axis"])
            return td.Cylinder(**cylinder_kwargs)
        if geo_type in {"group", "geometry_group"} or "geometries" in cfg:
            geometries = tuple(
                self._tidy3d_geometry_from_cfg(child)
                for child in cfg.get("geometries", [])
            )
            if not geometries:
                raise ValueError(
                    "Compound geometry group requires non-empty geometries."
                )
            return td.GeometryGroup(geometries=geometries)
        if geo_type in {"clip", "clip_operation", "boolean"} or (
            "geometry_a" in cfg and "geometry_b" in cfg
        ):
            operation = cfg.get("operation")
            if operation is None:
                raise KeyError("Compound clip geometry requires an 'operation' field.")
            return td.ClipOperation(
                geometry_a=self._tidy3d_geometry_from_cfg(cfg["geometry_a"]),
                geometry_b=self._tidy3d_geometry_from_cfg(cfg["geometry_b"]),
                operation=operation,
            )
        if geo_type == "compound":
            operation = cfg.get("operation", "union")
            geometries = cfg.get("geometries", [])
            if len(geometries) < 2:
                raise ValueError(
                    "Compound geometry requires at least two child geometries."
                )
            geometry = self._tidy3d_geometry_from_cfg(geometries[0])
            for child_cfg in geometries[1:]:
                geometry = td.ClipOperation(
                    geometry_a=geometry,
                    geometry_b=self._tidy3d_geometry_from_cfg(child_cfg),
                    operation=operation,
                )
            return geometry
        raise ValueError(f"Geometry type {geo_type} not supported")

    def _build_geometry_from_cfg(
        self,
        cfg,
        *,
        value_key,
        bg_key=None,
        use_tidy3d: bool = False,
        average_value_with_bg: bool = False,
    ):
        geo_type = cfg["type"]
        scalar_value = _cfg_scalar(cfg, (value_key,))
        tidy3d_freq = C_0 / (self.sim_cfg["wl_cen"] * MICRON_UNIT)
        if bg_key is None:
            scalar_bg = scalar_value
        else:
            scalar_bg = _cfg_scalar(cfg, (bg_key,), default=scalar_value)
        if average_value_with_bg:
            scalar_value = (scalar_value + scalar_bg) / 2

        if _geometry_cfg_is_compound(cfg):
            if not use_tidy3d:
                raise ValueError(
                    "Compound/nested geometry cfgs currently require use_tidy3d=True."
                )
            return td.Structure(
                geometry=self._tidy3d_geometry_from_cfg(cfg),
                medium=_tidy3d_medium_from_permittivity(
                    scalar_value,
                    freq=tidy3d_freq,
                ),
            )

        match geo_type:
            case "box":
                if use_tidy3d:
                    geometry = td.Structure(
                        geometry=self._tidy3d_geometry_from_cfg(cfg),
                        medium=_tidy3d_medium_from_permittivity(
                            scalar_value,
                            freq=tidy3d_freq,
                        ),
                    )
                else:
                    mp = _import_meep()
                    geometry = mp.Block(
                        _vector(cfg["size"]),
                        center=_vector(cfg["center"]),
                        material=mp.Medium(epsilon=scalar_value),
                    )

            case "prism":
                if use_tidy3d:
                    geometry = td.Structure(
                        geometry=self._tidy3d_geometry_from_cfg(cfg),
                        medium=_tidy3d_medium_from_permittivity(
                            scalar_value,
                            freq=tidy3d_freq,
                        ),
                    )
                else:
                    mp = _import_meep()
                    geometry = mp.Prism(
                        [_vector(v) for v in cfg["vertices"]],
                        height=cfg.get("height", mp.inf),
                        material=mp.Medium(epsilon=scalar_value),
                    )
            case "sine_bend":
                if use_tidy3d:
                    geometry = td.Structure(
                        geometry=self._tidy3d_geometry_from_cfg(cfg),
                        medium=_tidy3d_medium_from_permittivity(
                            scalar_value,
                            freq=tidy3d_freq,
                        ),
                    )
                else:
                    raise ValueError(
                        "sine bend geometry currently requires use_tidy3d=True."
                    )
            case "cylinder":
                if use_tidy3d:
                    geometry = td.Structure(
                        geometry=self._tidy3d_geometry_from_cfg(cfg),
                        medium=_tidy3d_medium_from_permittivity(
                            scalar_value,
                            freq=tidy3d_freq,
                        ),
                    )
                else:
                    mp = _import_meep()
                    geometry = mp.Cylinder(
                        radius=cfg["radius"],
                        height=cfg.get("height", mp.inf),
                        center=_vector(cfg["center"]),
                        material=mp.Medium(epsilon=scalar_value),
                    )
            case _:
                raise ValueError(f"Geometry type {geo_type} not supported")

        return geometry

    def get_geometry_box(self, border_width=[0, 0], PML=[0, 0]):
        del PML
        mins = np.full(self.dim, float("inf"), dtype=float)
        maxs = np.full(self.dim, float("-inf"), dtype=float)
        mp = None
        for design_region in self.design_region_cfgs.values():
            center = np.asarray(_as_3(design_region["center"])[: self.dim], dtype=float)
            size = np.asarray(_as_3(design_region["size"])[: self.dim], dtype=float)
            mins = np.minimum(mins, center - size / 2)
            maxs = np.maximum(maxs, center + size / 2)

        bounded_geometries = dict(self.geometry)
        bounded_geometries.update(
            {
                name: self.override_structures[name]
                for name, cfg in self.geometry_cfgs.items()
                if cfg.get("override_structure", False)
                and name in self.override_structures
            }
        )
        for name, geometry in bounded_geometries.items():
            # Some background-partition geometries must fill the already
            # inferred cell without enlarging that cell themselves.
            if self.geometry_cfgs.get(name, {}).get("_exclude_from_bounds", False):
                continue
            if TD_SUPPORTED and isinstance(geometry, td.Structure):
                min_bounds, max_bounds = geometry.geometry.bounds
                geo_min = np.asarray(min_bounds[: self.dim], dtype=float)
                geo_max = np.asarray(max_bounds[: self.dim], dtype=float)
                ## if infinite, we set to 0
                geo_min = np.where(np.isfinite(geo_min), geo_min, 0)
                geo_max = np.where(np.isfinite(geo_max), geo_max, 0)
            else:
                if mp is None:
                    mp = _import_meep()
                if isinstance(geometry, mp.Block):
                    center = np.asarray(
                        [geometry.center.x, geometry.center.y, geometry.center.z][
                            : self.dim
                        ],
                        dtype=float,
                    )
                    size = np.asarray(
                        [geometry.size.x, geometry.size.y, geometry.size.z][: self.dim],
                        dtype=float,
                    )
                    geo_min = center - size / 2
                    geo_max = center + size / 2
                elif isinstance(geometry, mp.Prism):
                    vertices = np.asarray(
                        [[v.x, v.y, v.z][: self.dim] for v in geometry.vertices],
                        dtype=float,
                    )
                    geo_min = vertices.min(axis=0)
                    geo_max = vertices.max(axis=0)
                    if self.dim == 3 and np.isfinite(geometry.height):
                        geo_min[2] = min(geo_min[2], -geometry.height / 2)
                        geo_max[2] = max(geo_max[2], geometry.height / 2)
                elif isinstance(geometry, mp.Cylinder):
                    center = np.asarray(
                        [geometry.center.x, geometry.center.y, geometry.center.z][
                            : self.dim
                        ],
                        dtype=float,
                    )
                    radius = np.full(self.dim, geometry.radius, dtype=float)
                    if self.dim == 3:
                        height = geometry.height if np.isfinite(geometry.height) else 0
                        radius[2] = height / 2
                    geo_min = center - radius
                    geo_max = center + radius
                else:
                    raise ValueError(f"Geometry type {type(geometry)} not supported")
            mins = np.minimum(mins, geo_min)
            maxs = np.maximum(maxs, geo_max)
        if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
            raise ValueError(
                "Cannot infer cell_size from empty geometry/design regions"
            )
        border = np.asarray(_axis_pairs(border_width, self.dim), dtype=float).reshape(
            self.dim, 2
        )
        ## extend, ((xmin, xmax), (ymin, ymax)) or ((xmin, xmax), (ymin, ymax), (zmin, zmax)) including border
        extend = [
            (mins[0] - border[0][0], maxs[0] + border[0][1]),
            (mins[1] - border[1][0], maxs[1] + border[1][1]),
        ]
        if self.dim == 3:
            extend.append((mins[2] - border[2][0], maxs[2] + border[2][1]))
        return tuple((maxs - mins + border.sum(axis=1)).tolist()), tuple(extend)

    def get_epsilon_map(
        self,
        cell_size,
        geometry,
        PML,
        resolution,
        eps_bg,
        use_tidy3d: bool = False,
        override_structures: dict | None = None,
        spacing: Sequence[float] | None = None,
        grid_spec_cfg=None,
        map_name: str = "epsilon_map",
        is_complex: bool = True,
        grid=None,  # can pass grid to reuse
    ):
        raster_map, _ = self.rasterize_geometry_map(
            cell_size=cell_size,
            geometry=geometry,
            override_structures=override_structures,
            PML=PML,
            resolution=resolution,
            background_value=eps_bg,
            use_tidy3d=use_tidy3d,
            spacing=spacing,
            grid_spec_cfg=grid_spec_cfg,
            map_name=map_name,
            is_complex=is_complex,
            grid=grid,
            subpixel=self.sim_cfg.get("subpixel", True),
        )
        return raster_map

    def _crop_raster_to_cell_bounds(
        self,
        values,
        coords,
        *,
        cell_size,
        boundaries=None,
        atol: float = 1e-6,
    ):
        cropped_values = values
        cropped_coords = []
        cropped_boundaries = [] if boundaries is not None else None
        crop_slices = []
        for axis, coord in enumerate(coords):
            coord = np.asarray(coord, dtype=np.float64)
            lo = -float(cell_size[axis]) / 2 - atol
            hi = float(cell_size[axis]) / 2 + atol
            mask = (coord >= lo) & (coord <= hi)
            if not np.any(mask):
                raise ValueError(
                    f"No raster cells remain along axis {axis}. "
                    f"coord range=({coord.min()}, {coord.max()}), target=({lo}, {hi})"
                )
            idx = np.where(mask)[0]
            axis_slice = slice(int(idx[0]), int(idx[-1]) + 1)
            crop_slices.append(axis_slice)
            cropped_coords.append(coord[axis_slice])
            if boundaries is not None:
                axis_boundaries = np.asarray(boundaries[axis], dtype=np.float64)
                if len(axis_boundaries) != len(coord) + 1:
                    raise ValueError(
                        "Expected raster boundaries to have length coord+1, got "
                        f"{len(axis_boundaries)} vs {len(coord)} on axis {axis}"
                    )
                cropped_boundaries.append(
                    axis_boundaries[axis_slice.start : axis_slice.stop + 1]
                )
        cropped_values = (
            cropped_values[tuple(crop_slices)] if values is not None else None
        )
        return (
            cropped_values,
            tuple(cropped_coords),
            (None if cropped_boundaries is None else tuple(cropped_boundaries)),
        )

    def _build_raster_grid_info(
        self,
        *,
        map_name: str,
        backend: str,
        coords,
        boundaries=None,
        grid=None,
    ):
        coords = tuple(np.asarray(axis, dtype=np.float64) for axis in coords)
        if boundaries is None:
            boundaries = tuple(_centers_to_boundaries(axis) for axis in coords)
        else:
            boundaries = tuple(
                np.asarray(axis, dtype=np.float64) for axis in boundaries
            )
        return {
            "name": str(map_name),
            "backend": str(backend),
            "grid": grid,
            "coords": coords,
            "boundaries": boundaries,
            "shape": tuple(int(len(axis)) for axis in coords),
        }

    def _store_raster_grid_info(
        self,
        map_name: str,
        grid_info: dict,
    ):
        copied = {
            "name": str(grid_info["name"]),
            "backend": str(grid_info["backend"]),
            "grid": grid_info.get("grid"),
            "coords": tuple(
                np.asarray(axis, dtype=np.float64) for axis in grid_info["coords"]
            ),
            "boundaries": tuple(
                np.asarray(axis, dtype=np.float64) for axis in grid_info["boundaries"]
            ),
            "shape": tuple(int(v) for v in grid_info["shape"]),
        }
        self.grid_info_dict[str(map_name)] = copied

    def _get_raster_grid_info(self, map_name: str) -> dict | None:
        return self.grid_info_dict.get(str(map_name))

    def rasterize_geometry_map(
        self,
        *,
        cell_size,
        geometry,
        PML,
        resolution,
        background_value,
        use_tidy3d: bool = False,
        override_structures: dict | None = None,
        spacing: Sequence[float] | None = None,
        grid_spec_cfg=None,
        map_name: str = "raster_map",
        is_complex: bool = True,
        grid=None,
        grid_info: dict | None = None,
        preprocess_map=None,
        postprocess_map=None,
        record_grid: bool = True,
        subpixel: bool = True,
        return_grid_only: bool = False,
    ):
        import time

        def _cast_epsilon_map_dtype(array):
            if np.iscomplexobj(array):
                return np.asarray(array, dtype=np.complex64)
            return np.asarray(array, dtype=np.float32)

        start = time.time()
        pml = _axis_values(PML, self.dim)
        cell_size_3d = _as_3(cell_size)
        if spacing is None:
            spacing_values = tuple([1.0 / float(resolution)] * self.dim)
            tidy3d_label = f"uniform resolution {resolution} px/um"
        else:
            spacing_values = _spacing_values(spacing, self.dim)
            tidy3d_label = f"nominal spacing {spacing_values} um"
        if use_tidy3d:
            if isinstance(grid_spec_cfg, td.GridSpec):
                tidy3d_label = f"explicit GridSpec ({type(grid_spec_cfg).__name__})"
            elif grid_spec_cfg is not None:
                tidy3d_label = f"grid_spec override with {tidy3d_label}"
            print(
                f"Using Tidy3d to generate {map_name} with {tidy3d_label} and cell size {cell_size}"
            )

            freq = C_0 / (self.sim_cfg["wl_cen"] * MICRON_UNIT)

            # center = (0, 0, 0)
            center = list((xmin + xmax) / 2 for xmin, xmax in self.cell_extend)

            if self.dim == 2 and len(center) == 2:
                center.append(0)
            monitor = td.PermittivityMonitor(
                center=center,
                freqs=[freq],
                size=cell_size_3d if self.dim == 3 else _as_3(cell_size, fill=td.inf),
                name="eps_monitor",
            )
            # monitor = td.Box(
            #     center=(0, 0, 0),
            #     size=cell_size_3d if self.dim == 3 else _as_3(cell_size, fill=td.inf),
            # )
            td.config.simulation.use_local_subpixel = subpixel
            # print(
            #     center,
            #     cell_size_3d,
            #     list(geometry.values()),
            #     (
            #         self._tidy3d_grid_spec_for_raster(
            #             spacing_values,
            #             grid_spec_cfg=grid_spec_cfg,
            #             override_structures=override_structures,
            #         )
            #         if grid is None and grid_info is None
            #         else td.GridSpec.from_grid(
            #             grid if grid is not None else grid_info["grid"]
            #         )
            #     ),
            # )
            simulation_kwargs = dict(
                center=center,
                symmetry=self.sim_cfg.get("symmetry", (0, 0, 0)),
                size=cell_size_3d,
                grid_spec=(
                    self._tidy3d_grid_spec_for_raster(
                        spacing_values,
                        grid_spec_cfg=grid_spec_cfg,
                        override_structures=override_structures,
                    )
                    if grid is None and grid_info is None
                    else td.GridSpec.from_grid(
                        grid if grid is not None else grid_info["grid"]
                    )
                ),
                medium=_tidy3d_medium_from_permittivity(
                    background_value,
                    freq=freq,
                ),
                structures=list(geometry.values()) if not return_grid_only else [],
                sources=[],
                monitors=[monitor] if not return_grid_only else [],
                subpixel=subpixel,
                run_time=1e-15,
                boundary_spec=td.BoundarySpec.pml(x=False, y=False, z=False),
            )
            local_subpixel = False
            try:
                sim = td.Simulation(**simulation_kwargs)
            except Exception as tidy3d_error:
                if not subpixel:
                    raise
                logging.warning(
                    "Tidy3D subpixel rasterization failed; using the local CuPy "
                    "geometric volume-averaging implementation instead. "
                    f"Original error: {tidy3d_error}",
                )
                local_subpixel = True
                simulation_kwargs["subpixel"] = False
                sim = td.Simulation(**simulation_kwargs)
            coords_xyz = sim.grid.boundaries.to_list
            for i, sym in enumerate(self.sim_cfg.get("symmetry", (0, 0, 0))):
                if sym != 0:
                    assert (
                        len(coords_xyz[i]) % 2 != 0
                    ), f"Expected odd number of grid boundaries along axis {i} for symmetry {sym}, got {len(coords_xyz[i])}"
            # Assert coords size is odd
            # print(len(coords_xyz[0]), len(coords_xyz[1]), len(coords_xyz[2]))
            # print(self._tidy3d_grid_spec_for_raster(
            #             spacing_values,
            #             grid_spec_cfg=grid_spec_cfg,
            #             override_structures=override_structures,
            #         ))
            # print(sim.grid.boundaries.y)
            # exit(0)
            ## this ensures Tidy3d ignores metal (with negative permittivity) and set it to Air.
            ## we approximate metal with PEC with very high electric_conductivity in fdtdx
            ## do not use the correct Box+freq usage, otherwise it fills the metal permittivity
            ## which causes fdtd to diverge.
            if return_grid_only:
                raster_data = raster_map = None
                raster_coords = tuple(
                    np.asarray(getattr(sim.grid.centers, axis))
                    for axis in _AXES[: self.dim]
                )
                raster_boundaries = tuple(
                    np.asarray(getattr(sim.grid.boundaries, axis))
                    for axis in _AXES[: self.dim]
                )
            else:
                if local_subpixel:
                    import cupy as cp

                    from gpu_subpixel import gpu_subpixel_rasterize

                    local_structures = list(geometry.values())
                    local_map = gpu_subpixel_rasterize(
                        local_structures,
                        sim.grid,
                        background_eps=background_value,
                        dtype=cp.float64 if is_complex else cp.float32,
                    )
                    raster_map = np.asarray(local_map.get())
                    raster_data = None
                else:
                    try:
                        raster_data = sim.epsilon(monitor)
                    except Exception as tidy3d_error:
                        if not subpixel:
                            raise
                        logging.warning(
                            "Tidy3D subpixel rasterization failed while extracting "
                            "epsilon; using the local CuPy geometric "
                            "volume-averaging implementation instead. "
                            f"Original error: {tidy3d_error}"
                        )
                        local_subpixel = True
                        simulation_kwargs["subpixel"] = False
                        sim = td.Simulation(**simulation_kwargs)
                        raster_data = None
                    if raster_data is not None:
                        raster_map = np.asarray(raster_data.to_numpy())
                    else:
                        import cupy as cp

                        from gpu_subpixel import gpu_subpixel_rasterize

                        local_structures = list(geometry.values())
                        local_map = gpu_subpixel_rasterize(
                            local_structures,
                            sim.grid,
                            background_eps=background_value,
                            dtype=cp.float64 if is_complex else cp.float32,
                        )
                        raster_map = np.asarray(local_map.get())
                raster_coords = (
                    tuple(
                        np.asarray(raster_data.coords[axis])
                        for axis in _AXES[: self.dim]
                    )
                    if raster_data is not None
                    else tuple(
                        np.asarray(getattr(sim.grid.centers, axis))
                        for axis in _AXES[: self.dim]
                    )
                )

                raster_inds = sim.grid.discretize_inds(monitor)
                raster_boundaries = tuple(
                    np.asarray(getattr(sim.grid.boundaries, axis), dtype=np.float64)[
                        int(start) : int(stop) + 1
                    ]
                    for axis, (start, stop) in zip(
                        _AXES[: self.dim], raster_inds[: self.dim]
                    )
                )
                # print(raster_map.shape, raster_coords, raster_boundaries)
            if self.dim == 2:
                raster_map = raster_map[..., 0] if raster_map is not None else None
                raster_coords = raster_coords[:2]
                raster_boundaries = raster_boundaries[:2]
            # raster_map, raster_coords, raster_boundaries = (
            #     self._crop_raster_to_cell_bounds(
            #         raster_map,
            #         raster_coords,
            #         cell_size=cell_size[: self.dim],
            #         boundaries=raster_boundaries,
            #     )
            # )
            raster_map = (
                _cast_epsilon_map_dtype(raster_map) if raster_map is not None else None
            )
            if not is_complex and raster_map is not None:
                raster_map = raster_map.real
            backend_name = "local_cupy" if local_subpixel else "tidy3d"
            raster_grid_info = self._build_raster_grid_info(
                map_name=map_name,
                backend=backend_name,
                coords=raster_coords,
                boundaries=raster_boundaries,
                grid=sim.grid,
            )
            # print(raster_grid_info)
        else:
            print(
                f"Using Meep to generate {map_name} with resolution {resolution} and cell size {cell_size}"
            )
            mp = _import_meep()
            if spacing is not None:
                if not np.allclose(
                    spacing_values,
                    [spacing_values[0]] * self.dim,
                    atol=1e-12,
                    rtol=1e-12,
                ):
                    raise ValueError(
                        "Anisotropic heat mesh spacing requires use_tidy3d=True; "
                        "Meep rasterization only supports isotropic resolution."
                    )
                resolution = 1.0 / spacing_values[0]
            boundary = [
                mp.PML(width, direction=direction)
                for width, direction in zip(pml, (mp.X, mp.Y, mp.Z))
                if width > 0
            ]
            sim = mp.Simulation(
                resolution=resolution,
                cell_size=mp.Vector3(*cell_size_3d),
                boundary_layers=boundary,
                geometry=list(geometry.values()),
                sources=None,
                default_material=mp.Medium(epsilon=background_value),
                eps_averaging=True,
            )
            sim.run(until=0)
            raster_map = _cast_epsilon_map_dtype(sim.get_epsilon())
            raster_coords = tuple(
                np.asarray(coord, dtype=np.float64) for coord in self.coords
            )
            backend_name = "meep"
            raster_grid_info = self._build_raster_grid_info(
                map_name=map_name,
                backend=backend_name,
                coords=raster_coords,
                grid=None,
            )

        if preprocess_map is not None and raster_map is not None:
            raster_map = preprocess_map(raster_map)
        if postprocess_map is not None and raster_map is not None:
            raster_map = postprocess_map(raster_map)
        if record_grid:
            self._store_raster_grid_info(map_name, raster_grid_info)
        self._epsilon_backend = backend_name
        end = time.time()
        print(
            f"{map_name.capitalize()} (use_tidy3d={use_tidy3d}) generated in {end - start:.2f} seconds (shape: {raster_map.shape if raster_map is not None else None}, dtype: {raster_map.dtype if raster_map is not None else None})"
        )
        return raster_map, raster_grid_info

    def _epsilon_inputs_are_complex(self):
        if np.iscomplexobj(self.eps_bg):
            return True

        for cfg in (
            list(self.port_cfgs.values())
            + list(self.geometry_cfgs.values())
            + list(self.design_region_cfgs.values())
        ):
            for key in ("eps", "eps_bg"):
                if key in cfg and np.iscomplexobj(cfg[key]):
                    return True
        return False

    def _cfgs_require_tidy3d(self, cfgs: dict) -> bool:
        return any(_geometry_cfg_is_compound(cfg) for cfg in cfgs.values())

    def _combined_geometry_cfgs(self):
        """Return physical port and non-port geometry configurations.

        Returns:
            A new mapping containing port configurations and ordinary geometry
            configurations. Mesh-only entries marked with
            ``override_structure=True`` are excluded.
        """
        combined_cfgs = {}
        combined_cfgs.update(self.port_cfgs)
        combined_cfgs.update(
            {
                name: cfg
                for name, cfg in self.geometry_cfgs.items()
                if not cfg.get("override_structure", False)
            }
        )
        return combined_cfgs

    def _has_cfg_property(self, value_keys, bg_value):
        if bg_value is None:
            return False
        all_cfgs = list(self._combined_geometry_cfgs().values())
        return all(
            any(key in cfg for key in value_keys) or cfg.get("material") is not None
            for cfg in all_cfgs
        )

    def _heat_value_from_cfg(self, cfg):
        if any(key in cfg for key in _HEAT_PROPERTY_KEYS):
            return float(_cfg_scalar(cfg, _HEAT_PROPERTY_KEYS))
        material = cfg.get("material")
        if material is None:
            raise KeyError("Missing thermal conductivity and material in cfg.")
        return float(get_thermal_conductivity_fn(material)(self.sim_cfg["wl_cen"]))

    def _heat_bg_from_cfg(self, cfg, default):
        if any(key in cfg for key in _HEAT_PROPERTY_BG_KEYS):
            return float(_cfg_scalar(cfg, _HEAT_PROPERTY_BG_KEYS))
        material_bg = cfg.get("material_bg")
        if material_bg is not None:
            return float(
                get_thermal_conductivity_fn(material_bg)(self.sim_cfg["wl_cen"])
            )
        return default

    def _heat_capacity_value_from_cfg(self, cfg):
        if any(key in cfg for key in _HEAT_CAPACITY_KEYS):
            return float(_cfg_scalar(cfg, _HEAT_CAPACITY_KEYS))
        material = cfg.get("material")
        if material is None:
            raise KeyError("Missing heat capacity and material in cfg.")
        return float(get_heat_capacity_fn(material)(self.sim_cfg["wl_cen"]))

    def _heat_capacity_bg_from_cfg(self, cfg, default):
        if any(key in cfg for key in _HEAT_CAPACITY_BG_KEYS):
            return float(_cfg_scalar(cfg, _HEAT_CAPACITY_BG_KEYS))
        material_bg = cfg.get("material_bg")
        if material_bg is not None:
            return float(get_heat_capacity_fn(material_bg)(self.sim_cfg["wl_cen"]))
        return default

    def _thermo_optic_value_from_cfg(self, cfg):
        if any(key in cfg for key in _THERMO_OPTIC_KEYS):
            return float(_cfg_scalar(cfg, _THERMO_OPTIC_KEYS))
        material = cfg.get("material")
        if material is None:
            raise KeyError("Missing thermo-optic coefficient and material in cfg.")
        return float(get_thermo_optic_coeff_fn(material)(self.sim_cfg["wl_cen"]))

    def _thermo_optic_bg_from_cfg(self, cfg, default):
        if any(key in cfg for key in _THERMO_OPTIC_BG_KEYS):
            return float(_cfg_scalar(cfg, _THERMO_OPTIC_BG_KEYS))
        material_bg = cfg.get("material_bg")
        if material_bg is not None:
            return float(get_thermo_optic_coeff_fn(material_bg)(self.sim_cfg["wl_cen"]))
        return default

    def _electrical_conductivity_value_from_cfg(self, cfg):
        sigma = _cfg_scalar(cfg, _ELECTRICAL_CONDUCTIVITY_KEYS, default=None)
        if sigma is not None:
            return float(sigma)
        material = cfg.get("material")
        if material is None:
            return 0.0
        try:
            return float(
                get_electrical_conductivity_fn(material)(self.sim_cfg["wl_cen"])
            )
        except Exception:
            return 0.0

    def _electrical_conductivity_bg_from_cfg(self, cfg, default):
        sigma_bg = _cfg_scalar(
            cfg,
            _ELECTRICAL_CONDUCTIVITY_BG_KEYS,
            default=None,
        )
        if sigma_bg is not None:
            return float(sigma_bg)
        material_bg = cfg.get("material_bg")
        if material_bg is not None:
            try:
                return float(
                    get_electrical_conductivity_fn(material_bg)(self.sim_cfg["wl_cen"])
                )
            except Exception:
                return 0.0
        return default

    def _geometry_measure_from_cfg(self, cfg):
        if _geometry_cfg_is_compound(cfg):
            raise ValueError(
                "Geometry measure for compound cfgs is ambiguous; provide explicit "
                "cross_section_area or total_power normalization instead."
            )
        geo_type = cfg["type"]
        if geo_type == "box":
            size = np.asarray(_as_3(cfg["size"])[: self.dim], dtype=float)
            return float(np.prod(size))
        if geo_type == "cylinder":
            radius = float(cfg["radius"])
            height = float(cfg.get("height", 1.0 if self.dim == 2 else 0.0))
            if self.dim == 2:
                return float(np.pi * radius**2)
            return float(np.pi * radius**2 * height)
        if geo_type == "prism":
            vertices = np.asarray([_as_3(v)[:2] for v in cfg["vertices"]], dtype=float)
            x = vertices[:, 0]
            y = vertices[:, 1]
            base_area = 0.5 * np.abs(
                np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
            )
            if self.dim == 2:
                return float(base_area)
            return float(base_area * float(cfg.get("height", 0.0)))
        raise ValueError(
            f"Geometry type {geo_type} not supported for heat source measure"
        )

    def _heat_source_cross_section_area_from_cfg(self, cfg):
        cross_section_area = cfg.get("cross_section_area")
        if cross_section_area is not None:
            return float(cross_section_area)
        if cfg.get("type") != "box" or self.dim != 3:
            return None
        direction = cfg.get(
            "current_direction", cfg.get("source_direction", cfg.get("direction"))
        )
        if direction is None:
            return None
        axis = _axis_index(str(direction))
        size = np.asarray(_as_3(cfg["size"])[: self.dim], dtype=float)
        area_terms = [size[i] for i in range(self.dim) if i != axis]
        if not area_terms:
            return None
        return float(np.prod(area_terms))

    def _heat_source_sigma_from_cfg(self, cfg):
        sigma = _cfg_scalar(
            cfg,
            ("electrical_conductivity", "sigma"),
            default=None,
        )
        if sigma is not None:
            return float(sigma)
        material = cfg.get(
            "electrical_material", cfg.get("heater_material", cfg.get("material"))
        )
        if material is None:
            return None
        return float(get_electrical_conductivity_fn(material)(self.sim_cfg["wl_cen"]))

    def _heat_source_value_from_cfg(self, cfg):
        if any(key in cfg for key in _HEAT_SOURCE_VALUE_KEYS):
            return float(_cfg_scalar(cfg, _HEAT_SOURCE_VALUE_KEYS))
        if "total_power" in cfg:
            measure = self._geometry_measure_from_cfg(cfg)
            if measure <= 0:
                raise ValueError("Heat source geometry measure must be positive.")
            return float(cfg["total_power"]) / measure
        if "current" in cfg:
            current = float(cfg["current"])
            sigma = self._heat_source_sigma_from_cfg(cfg)
            if sigma is None:
                raise ValueError(
                    "Heat source config with current requires electrical_conductivity/sigma "
                    "or a material with known electrical conductivity."
                )
            cross_section_area = self._heat_source_cross_section_area_from_cfg(cfg)
            if cross_section_area is None:
                raise ValueError(
                    "Heat source config with current requires cross_section_area or a "
                    "3D box source with current_direction/source_direction/direction."
                )
            current_density = current / float(cross_section_area)
            volumetric_heat = current_density**2 / sigma
            print(
                f"[HEAT]: current: {current:.3e} A, cross-section area: {cross_section_area:.3e} um^2, sigma: {sigma:.3e} S/um, "
                f"volumetric heat rate {volumetric_heat:.3e} W/um^3"
            )
            return volumetric_heat
        raise KeyError(
            "Heat source config requires one of heat_density/q/source, total_power, "
            "or current with electrical_conductivity and cross_section_area."
        )

    def _heat_source_bg_from_cfg(self, cfg, default):
        return float(cfg.get("background_heat_density", 0.0))

    def _heat_source_raster_cfgs(self, heat_source_cfgs: dict | None):
        if not heat_source_cfgs:
            return {} if heat_source_cfgs is None else heat_source_cfgs

        raster_cfgs = {}
        for name, cfg in self._combined_geometry_cfgs().items():
            dummy_cfg = copy.deepcopy(cfg)
            dummy_cfg[_HEAT_SOURCE_DUMMY_CFG_KEY] = True
            dummy_cfg.setdefault("q", 0.0)
            dummy_cfg.setdefault("background_heat_density", 0.0)
            raster_cfgs[f"__heat_source_dummy__{name}"] = dummy_cfg

        for name, cfg in heat_source_cfgs.items():
            raster_cfgs[name] = cfg
        return raster_cfgs

    def _heat_source_encoded_value_override(
        self,
        cfg,
        *,
        encoded_bg_value: float,
        encode_value,
    ):
        del encode_value
        if not cfg.get(_HEAT_SOURCE_DUMMY_CFG_KEY):
            return None
        dummy_value = float(encoded_bg_value) - _HEAT_SOURCE_DUMMY_ENCODED_OFFSET
        return dummy_value, dummy_value

    def _postprocess_heat_source_raster_map(self, array):
        array = np.asarray(array, dtype=np.float32)
        array[array < 0.0] = 0.0
        return array

    def _build_scalar_property_map(
        self,
        *,
        cfgs=None,
        map_name: str = "scalar_property_map",
        value_keys,
        bg_value,
        value_from_cfg,
        bg_from_cfg,
        cell_size,
        PML,
        resolution,
        use_tidy3d: bool = False,
        spacing: Sequence[float] | None = None,
        grid_spec_cfg=None,
        preprocess_value=None,
        postprocess_map=None,
        encoded_value_override_from_cfg=None,
        is_complex: bool = True,
        reuse_grid_info: dict | None = None,
        include_override_structures: bool = True,
    ):
        geometry_cfgs = self._combined_geometry_cfgs() if cfgs is None else cfgs
        # cfgs = geometry_cfgs | self.override_structures
        if include_override_structures:
            cfgs = geometry_cfgs | self.design_region_cfgs
        else:
            cfgs = geometry_cfgs
        if bg_value is None:
            return None
        if self._cfgs_require_tidy3d(cfgs):
            if not TD_SUPPORTED:
                raise ImportError(
                    "Compound geometry cfgs require tidy3d, but tidy3d is not installed."
                )
            use_tidy3d = True

        def _resolve_property_values(cfg):
            raw_prop_value = float(value_from_cfg(cfg))
            raw_prop_bg = float(bg_from_cfg(cfg, raw_prop_value))
            prop_value = (
                float(preprocess_value(raw_prop_value))
                if preprocess_value is not None
                else raw_prop_value
            )
            prop_bg = (
                float(preprocess_value(raw_prop_bg))
                if preprocess_value is not None
                else raw_prop_bg
            )
            return raw_prop_value, raw_prop_bg, prop_value, prop_bg

        property_values = []
        for cfg in cfgs.values():
            if not (
                any(key in cfg for key in value_keys) or cfg.get("material") is not None
            ):
                return None
            _, _, prop_value, prop_bg = _resolve_property_values(cfg)
            property_values.extend((prop_value, prop_bg))
        encoded_bg_value = (
            float(preprocess_value(float(bg_value)))
            if preprocess_value is not None
            else float(bg_value)
        )
        property_values.append(encoded_bg_value)
        if not property_values:
            return None

        encode_value = lambda value: float(value)
        decode_map = lambda array: array
        if use_tidy3d:
            prop_min = min(property_values)
            prop_max = max(property_values)
            if np.isclose(prop_max, prop_min):
                reference_shape = None
                if spacing is not None:
                    spacing_values = _spacing_values(spacing, self.dim)
                    reference_shape = tuple(
                        int(round(axis_size / dl))
                        for axis_size, dl in zip(cell_size[: self.dim], spacing_values)
                    )
                else:
                    reference = getattr(self, "epsilon_map", None)
                    reference_shape = None if reference is None else reference.shape
                if reference_shape is None:
                    reference_shape = tuple(
                        int(round(axis_size * resolution)) for axis_size in cell_size
                    )[: self.dim]
                result = np.full(reference_shape, prop_min, dtype=np.float32)
                if postprocess_map is not None:
                    result = np.asarray(postprocess_map(result), dtype=np.float32)
                return result

            prop_scale = 1.0 / (prop_max - prop_min)

            def encode_value(value):
                return 1.0 + (float(value) - prop_min) * prop_scale

            def decode_map(array):
                decoded = np.real((array - 1.0) / prop_scale + prop_min).astype(
                    np.float32
                )
                if postprocess_map is not None:
                    decoded = np.asarray(postprocess_map(decoded), dtype=np.float32)
                return decoded

        property_geometry = {}
        property_override_structures = {}
        property_cfgs = {}
        for name, cfg in cfgs.items():
            property_cfg = copy.deepcopy(cfg)
            _, _, prop_value, prop_bg = _resolve_property_values(property_cfg)
            encoded_value = encode_value(prop_value)
            encoded_bg = encode_value(prop_bg)
            if encoded_value_override_from_cfg is not None:
                override = encoded_value_override_from_cfg(
                    property_cfg,
                    encoded_bg_value=encoded_bg_value,
                    encode_value=encode_value,
                )
                if override is not None:
                    encoded_value, encoded_bg = override
            property_cfg["eps"] = encoded_value
            property_cfg["eps_bg"] = encoded_bg
            geo = self._build_geometry_from_cfg(
                property_cfg,
                value_key="eps",
                bg_key="eps_bg",
                use_tidy3d=use_tidy3d,
                average_value_with_bg=False,
            )

            if name in self.override_structures:
                property_override_structures[name] = geo
            else:
                property_geometry[name] = geo

            property_cfgs[name] = property_cfg

        # print(
        #     map_name, prop_min, prop_max, property_cfgs, encode_value(encoded_bg_value)
        # )
        encoded_map, _ = self.rasterize_geometry_map(
            cell_size=cell_size,
            geometry=property_geometry,
            PML=PML,
            resolution=resolution,
            background_value=encode_value(encoded_bg_value),
            use_tidy3d=use_tidy3d,
            override_structures=property_override_structures,
            spacing=spacing,
            grid_spec_cfg=grid_spec_cfg,
            map_name=map_name,
            is_complex=is_complex,
            grid_info=reuse_grid_info,
            subpixel=self.sim_cfg.get("subpixel", True),
        )
        decoded_map = decode_map(encoded_map)
        # print(np.min(decoded_map), np.max(decoded_map), np.mean(decoded_map))
        if not use_tidy3d and postprocess_map is not None:
            decoded_map = np.asarray(postprocess_map(decoded_map), dtype=np.float32)
        return decoded_map

    def _build_uniform_export_grid_info(self):
        ### consider we filter out cells whose edges are outside the cell_size region
        ### this corresponds to floor not round or ceil.
        _, export_grid_info = self.rasterize_geometry_map(
            cell_size=self.cell_size,
            geometry={},
            PML=self.sim_cfg["PML"],
            resolution=self.resolution,
            background_value=self.eps_bg,
            use_tidy3d=True,
            grid_spec_cfg=None,
            map_name="export_epsilon_map",
            subpixel=False,
            return_grid_only=True,
        )

        # export_shape = tuple(
        #     max(1, int(np.floor(float(axis_size) * float(self.resolution))))
        #     for axis_size in self.cell_size[: self.dim]
        # )
        # export_grid = build_rectilinear_grid_metadata(
        #     shape=export_shape,
        #     spacing=self.grid_step,
        #     label="optical_export",
        # ).to_dict()

        export_grid = build_rectilinear_grid_metadata(
            coords=export_grid_info["coords"],
            boundaries=export_grid_info["boundaries"],
            label="optical_export",
        ).to_dict()
        return {
            "name": "export_epsilon_map",
            "backend": "uniform_export",
            "grid": None,
            "coords": export_grid["coords"],
            "boundaries": export_grid["boundaries"],
            "shape": tuple(int(v) for v in export_grid["shape"]),
        }, export_grid

    def build_epsilon_map(self, use_tidy3d: bool | None = None):
        if use_tidy3d is None:
            use_tidy3d = self.resolution >= 1 and TD_SUPPORTED

        optical_grid_spec_cfg = self._optical_tidy3d_grid_spec_cfg()

        if self.native_grid is not None and not use_tidy3d:
            raise ValueError("A supplied native grid requires use_tidy3d=True")

        native_epsilon_map, native_grid_info = self.rasterize_geometry_map(
            cell_size=self.cell_size,
            geometry=self.geometry,
            override_structures=self.override_structures,
            PML=self.sim_cfg["PML"],
            resolution=self.resolution,
            background_value=self.eps_bg,
            use_tidy3d=use_tidy3d,
            grid_spec_cfg=optical_grid_spec_cfg,
            grid=self.native_grid,
            map_name="epsilon_map",
            subpixel=self.sim_cfg.get("subpixel", True),
        )
        self.native_epsilon_map = native_epsilon_map
        self.optical_grid_metadata = build_rectilinear_grid_metadata(
            coords=tuple(
                np.asarray(axis, dtype=np.float64)
                for axis in native_grid_info["coords"]
            ),
            boundaries=tuple(
                np.asarray(axis, dtype=np.float64)
                for axis in native_grid_info["boundaries"]
            ),
            label="optical_native",
        ).to_dict()

        export_grid_info, export_grid_metadata = self._build_uniform_export_grid_info()
        self.export_grid_metadata = export_grid_metadata
        self._store_raster_grid_info("export_epsilon_map", export_grid_info)
        if tuple(int(v) for v in self.optical_grid_metadata["shape"]) == tuple(
            int(v) for v in self.export_grid_metadata["shape"]
        ) and all(
            np.allclose(
                np.asarray(
                    self.optical_grid_metadata["coords"][axis], dtype=np.float64
                ),
                np.asarray(self.export_grid_metadata["coords"][axis], dtype=np.float64),
                atol=1e-12,
                rtol=1e-9,
            )
            for axis in range(self.dim)
        ):
            export_epsilon_map = np.asarray(native_epsilon_map)
        else:
            export_epsilon_map = (
                self.resample_map_between_coords(
                    native_epsilon_map,
                    src_coords=self.optical_grid_metadata["coords"],
                    dst_coords=self.export_grid_metadata["coords"],
                )
                .detach()
                .cpu()
                .numpy()
            )
        self.export_epsilon_map = export_epsilon_map
        self.epsilon_map = self.native_epsilon_map
        # import matplotlib.pyplot as plt
        # plt.imshow(np.real(self.export_epsilon_map)[self.export_epsilon_map.shape[0] // 2, :, :])
        # plt.savefig("debug_build_epsilon_map.png")
        self.has_complex_epsilon = bool(np.iscomplexobj(self.epsilon_map))
        self.material_property_maps["epsilon"] = self.epsilon_map
        self.material_property_backend = native_grid_info["backend"]
        self._sync_optical_grid_metadata()
        return self.epsilon_map

    def build_electrical_conductivity_map(self, use_tidy3d: bool | None = None):
        if use_tidy3d is None:
            if getattr(self, "epsilon_map", None) is None:
                self.build_epsilon_map()
            use_tidy3d = getattr(self, "material_property_backend", None) == "tidy3d"

        cfgs = self._combined_geometry_cfgs()
        has_region_sigma = any(
            any(key in cfg for key in _ELECTRICAL_CONDUCTIVITY_KEYS)
            or cfg.get("material") is not None
            for cfg in cfgs.values()
        )
        if not has_region_sigma and self.electrical_conductivity_bg is None:
            self.electrical_conductivity_map = None
            self.material_property_maps.pop("electrical_conductivity", None)
            self._electrical_conductivity_map_built = True
            return None

        bg_value = (
            self.electrical_conductivity_bg
            if self.electrical_conductivity_bg is not None
            else 0.0
        )
        heat_spacing = self._heat_mesh_spacing()
        heat_resolution = self._heat_mesh_resolution_for_raster() or self.resolution
        heat_grid_spec_cfg = self._heat_tidy3d_grid_spec_cfg()
        ## this electrical_conductivity, e.g., TiN=2.3 S/um, if only for DC/low-freq simulation, e.g., Joule heating
        ## do not use this in optical freq simulation, need to use complex permittivity.
        electrical_conductivity_map = self._build_scalar_property_map(
            map_name="electrical_conductivity_map",
            value_keys=_ELECTRICAL_CONDUCTIVITY_KEYS,
            bg_value=bg_value,
            value_from_cfg=self._electrical_conductivity_value_from_cfg,
            bg_from_cfg=self._electrical_conductivity_bg_from_cfg,
            cell_size=self.cell_size,
            PML=self.sim_cfg["PML"],
            resolution=heat_resolution,
            use_tidy3d=use_tidy3d,
            spacing=heat_spacing,
            grid_spec_cfg=heat_grid_spec_cfg,
            is_complex=False,
        )
        if electrical_conductivity_map is not None and np.allclose(
            electrical_conductivity_map,
            0.0,
            atol=1e-12,
            rtol=1e-12,
        ):
            electrical_conductivity_map = None

        self.electrical_conductivity_map = electrical_conductivity_map
        if electrical_conductivity_map is None:
            self.material_property_maps.pop("electrical_conductivity", None)
        else:
            self.material_property_maps["electrical_conductivity"] = (
                electrical_conductivity_map
            )
        self._electrical_conductivity_map_built = True
        return self.electrical_conductivity_map

    def build_thermal_property_maps(self, use_tidy3d: bool | None = None):
        if use_tidy3d is None:
            use_tidy3d = self._heat_use_tidy3d_default()
        mesh_type = _normalize_heat_mesh_type(self._heat_mesh_type_default())
        if mesh_type not in {
            "rectangular",
            "fixed_distance_unstructured_tidy3d",
            "fixed_rectilinear_nonuniform",
        }:
            raise ValueError(f"Unsupported heat mesh_type: {mesh_type}")

        if getattr(self, "epsilon_map", None) is None:
            self.build_epsilon_map(use_tidy3d=use_tidy3d)

        heat_spacing = self._heat_mesh_spacing()
        heat_resolution = self._heat_mesh_resolution_for_raster() or self.resolution
        heat_grid_spec_cfg = self._heat_tidy3d_grid_spec_cfg()

        conductivity_map = self.get_conductivity_map(
            self.cell_size,
            self.sim_cfg["PML"],
            heat_resolution,
            use_tidy3d=use_tidy3d,
            spacing=heat_spacing,
            grid_spec_cfg=heat_grid_spec_cfg,
        )
        conductivity_grid_info = self._get_raster_grid_info("conductivity_map")
        if use_tidy3d and TD_SUPPORTED and conductivity_grid_info is not None:
            self._thermal_raster_grid_spec = td.GridSpec.from_grid(
                conductivity_grid_info["grid"]
            )
        else:
            self._thermal_raster_grid_spec = None
        heat_capacity_map = None
        if self._heat_include_capacity_default():
            heat_capacity_map = self.get_heat_capacity_map(
                self.cell_size,
                self.sim_cfg["PML"],
                heat_resolution,
                use_tidy3d=use_tidy3d,
                spacing=heat_spacing,
                grid_spec_cfg=heat_grid_spec_cfg,
                reuse_grid_info=conductivity_grid_info,
            )

        ## thermo_optic_coeff is always on optical native grid
        ## it is used in optical simulation, not in thermal simulation
        optical_grid_spec_cfg = self._optical_tidy3d_grid_spec_cfg()
        optical_grid_info = self._get_raster_grid_info("epsilon_map")
        thermo_optic_coeff_map = self.get_thermo_optic_coeff_map(
            self.cell_size,
            self.sim_cfg["PML"],
            heat_resolution,
            use_tidy3d=use_tidy3d,
            spacing=heat_spacing,
            grid_spec_cfg=optical_grid_spec_cfg,
            reuse_grid_info=optical_grid_info,
        )

        self.conductivity_map = conductivity_map
        self.heat_capacity_map = heat_capacity_map
        self.thermal_capacity_map = heat_capacity_map
        self.thermo_optic_coeff_map = thermo_optic_coeff_map
        self.thermal_coefficient_map = thermo_optic_coeff_map
        thermal_shape = None
        for thermal_map in (
            conductivity_map,
            heat_capacity_map,
            thermo_optic_coeff_map,
        ):
            if thermal_map is not None:
                thermal_shape = tuple(int(v) for v in np.asarray(thermal_map).shape)
                break
        if thermal_shape is None:
            thermal_shape = self._heat_grid_shape_from_spacing(
                cell_size=self.cell_size,
                spacing=heat_spacing,
            )
        if (
            use_tidy3d
            and conductivity_grid_info is not None
            and tuple(int(v) for v in conductivity_map.shape[: self.dim])
            == tuple(len(c) for c in conductivity_grid_info["coords"])
        ):
            self._sync_thermal_grid_metadata_from_coords(
                conductivity_grid_info["coords"]
            )
        else:
            self._sync_thermal_grid_metadata_from_shape(thermal_shape)
        self.material_property_maps.update(
            {
                "conductivity": self.conductivity_map,
                "heat_capacity": self.heat_capacity_map,
                "thermal_capacity": self.thermal_capacity_map,
                "thermo_optic_coeff": self.thermo_optic_coeff_map,
                "thermal_coefficient": self.thermal_coefficient_map,
            }
        )
        self.material_property_backend = "tidy3d" if use_tidy3d else "meep"
        return {
            "conductivity": self.conductivity_map,
            "heat_capacity": self.heat_capacity_map,
            "thermal_capacity": self.thermal_capacity_map,
            "thermo_optic_coeff": self.thermo_optic_coeff_map,
            "thermal_coefficient": self.thermal_coefficient_map,
        }

    def build_material_property_maps(self, use_tidy3d: bool | None = None):
        self.build_epsilon_map(use_tidy3d=use_tidy3d)
        self.build_electrical_conductivity_map(use_tidy3d=use_tidy3d)
        self.build_thermal_property_maps(use_tidy3d=use_tidy3d)
        return self.material_property_maps

    def rebuild_thermal_property_maps(self, use_tidy3d: bool | None = None):
        return self.build_thermal_property_maps(use_tidy3d=use_tidy3d)

    def rebuild_material_property_maps(self, use_tidy3d: bool | None = None):
        return self.build_material_property_maps(use_tidy3d=use_tidy3d)

    def get_material_property_map(self, property_name: str):
        if not self.material_property_maps:
            raise RuntimeError("Material property maps have not been built yet.")
        if property_name not in self.material_property_maps:
            # raise KeyError(
            #     f"Property map '{property_name}' is not available. "
            #     "If this is a thermal property, call build_thermal_property_maps() first."
            # )
            return None
        return self.material_property_maps[property_name]

    def _sync_thermal_grid_metadata_from_shape(self, shape):
        shape = tuple(int(v) for v in shape)
        if not shape:
            return
        self.thermal_grid_spacing = tuple(float(v) for v in self._heat_mesh_spacing())
        self.thermal_grid_shape = shape
        self.thermal_coords = self._heat_coords_from_shape_spacing(
            shape=self.thermal_grid_shape,
            spacing=self.thermal_grid_spacing,
        )
        (
            self.thermal_design_region_masks,
            self.thermal_design_region_mask_weights,
            self.thermal_design_region_axis_weights,
        ) = self._build_region_masks_from_coords(
            self.design_region_cfgs,
            self.thermal_coords,
        )

    def _sync_thermal_grid_metadata_from_coords(self, coords):
        coords = tuple(np.asarray(axis, dtype=np.float64) for axis in coords)
        if not coords:
            return
        self.thermal_coords = coords
        self.thermal_grid_shape = tuple(int(axis.size) for axis in coords)
        self.thermal_grid_spacing = tuple(
            float(np.mean(np.diff(axis))) if axis.size > 1 else float(self.grid_step)
            for axis in coords
        )
        (
            self.thermal_design_region_masks,
            self.thermal_design_region_mask_weights,
            self.thermal_design_region_axis_weights,
        ) = self._build_region_masks_from_coords(
            self.design_region_cfgs,
            self.thermal_coords,
        )

    def _normalize_property_name(self, property_name: str | None) -> str:
        if property_name is None:
            return "epsilon"
        normalized = str(property_name).strip().lower()
        aliases = {
            "eps": "epsilon",
            "permittivity": "epsilon",
            "heat_conductivity": "conductivity",
            "thermal_conductivity": "conductivity",
            "k": "conductivity",
            "capacity": "heat_capacity",
            "thermal_capacity": "heat_capacity",
            "dn_dt": "thermo_optic_coeff",
            "thermo_optic": "thermo_optic_coeff",
            "sigma": "electrical_conductivity",
            "q": "heat_source",
            "heat_density": "heat_source",
            "heat_source_map": "heat_source",
        }
        return aliases.get(normalized, normalized)

    def _resolve_property_plot_data(
        self,
        property_name: str | None = None,
        property_map=None,
    ):
        if property_map is not None:
            resolved_name = (
                "property"
                if property_name is None
                else self._normalize_property_name(property_name)
            )
            plot_map = (
                property_map.detach().cpu().numpy()
                if isinstance(property_map, torch.Tensor)
                else np.asarray(property_map)
            )
            shape = tuple(int(v) for v in plot_map.shape)
            if shape == tuple(int(v) for v in self.grid_shape):
                coords = self.coords
                grid_kind = "optical"
                grid_info = self._get_raster_grid_info("epsilon_map")
            else:
                candidate_shapes = {
                    tuple(int(v) for v in np.asarray(arr).shape)
                    for arr in (
                        self.conductivity_map,
                        self.heat_capacity_map,
                        self.thermo_optic_coeff_map,
                        self.heat_source_map,
                    )
                    if arr is not None
                }
                if shape in candidate_shapes and (
                    self.thermal_grid_shape is None
                    or shape != tuple(int(v) for v in self.thermal_grid_shape)
                ):
                    self._sync_thermal_grid_metadata_from_shape(shape)
                grid_info = None
            if self.thermal_grid_shape is not None and shape == tuple(
                int(v) for v in self.thermal_grid_shape
            ):
                if self.thermal_coords is None:
                    self._sync_thermal_grid_metadata_from_shape(shape)
                coords = self.thermal_coords
                grid_kind = "thermal"
                if grid_info is None:
                    grid_info = self._get_raster_grid_info("conductivity_map")
            elif shape != tuple(int(v) for v in self.grid_shape):
                raise ValueError(
                    f"property_map shape {shape} does not match optical grid "
                    f"{tuple(int(v) for v in self.grid_shape)} or thermal grid "
                    f"{None if self.thermal_grid_shape is None else tuple(int(v) for v in self.thermal_grid_shape)}."
                )
            return (
                plot_map,
                coords,
                resolved_name,
                grid_kind,
                grid_info,
            )

        normalized_name = self._normalize_property_name(property_name)
        if normalized_name == "epsilon":
            return (
                np.asarray(self.epsilon_map),
                self.coords,
                normalized_name,
                "optical",
                self._get_raster_grid_info("epsilon_map"),
            )

        if normalized_name in {
            "conductivity",
            "heat_capacity",
            "thermo_optic_coeff",
            "electrical_conductivity",
        }:
            # self.build_thermal_property_maps()
            plot_map = self.get_material_property_map(normalized_name)
            if plot_map is None:
                self.build_thermal_property_maps()
            return (
                np.asarray(plot_map),
                self.thermal_coords,
                normalized_name,
                "thermal",
                self._get_raster_grid_info(f"{normalized_name}_map"),
            )

        if normalized_name == "heat_source":
            if self.heat_source_map is None:
                if self.heat_source_cfgs and self._heat_build_sources_default():
                    self.build_heat_source_maps()
            if self.heat_source_map is None:
                raise KeyError("Heat source map is not available.")
            return (
                np.asarray(self.heat_source_map),
                self.thermal_coords,
                normalized_name,
                "thermal",
                self._get_raster_grid_info("conductivity_map"),
            )

        raise KeyError(f"Unsupported property for plotting: {property_name}")

    def _plot_property_requires_harmonic_resample(self, property_name: str | None):
        normalized_name = self._normalize_property_name(property_name)
        return normalized_name in {"conductivity", "electrical_conductivity"}

    def _plot_target_grid_info(self):
        export_grid_info = self._get_raster_grid_info("export_epsilon_map")
        if export_grid_info is not None:
            return export_grid_info
        return self._build_raster_grid_info(
            map_name="plot_optical_grid",
            backend="uniform",
            coords=tuple(
                np.asarray(axis, dtype=np.float64)
                for axis in (
                    self.export_grid_metadata["coords"]
                    if self.export_grid_metadata is not None
                    else self.coords
                )
            ),
        )

    def _resample_plot_map_to_uniform_grid(
        self,
        values,
        *,
        src_grid_info: dict | None,
        property_name: str | None,
    ):
        target_grid_info = self._plot_target_grid_info()
        target_coords = target_grid_info["coords"]

        if src_grid_info is None:
            return values, target_coords

        src_coords = src_grid_info["coords"]
        if (
            tuple(values.shape) == tuple(int(v) for v in target_grid_info["shape"])
            and len(src_coords) == len(target_coords)
            and all(
                src.shape == dst.shape and np.allclose(src, dst, atol=1e-9, rtol=1e-6)
                for src, dst in zip(src_coords, target_coords)
            )
        ):
            return values, target_coords

        # print(np.min(values), np.max(values), np.mean(values))
        if isinstance(values, np.ndarray):
            src_tensor = torch.as_tensor(values, device=self.device)
        else:
            src_tensor = values.data.to(device=self.device)
        if self._plot_property_requires_harmonic_resample(property_name):
            # print(f"Using harmonic resampling for property '{property_name}'")
            # print(src_tensor.min().item(), src_tensor.max().item())
            eps = torch.finfo(torch.float32).tiny
            src_tensor = 1.0 / torch.clamp(src_tensor.to(dtype=torch.float32), min=eps)
            dst_tensor = resample_rectilinear_tensor(
                src_tensor,
                src_coords=src_coords,
                dst_coords=target_coords,
            )
            dst_tensor = 1.0 / torch.clamp(dst_tensor, min=eps)
            # print(dst_tensor.min(), dst_tensor.max())
        else:
            dst_tensor = resample_rectilinear_tensor(
                src_tensor,
                src_coords=src_coords,
                dst_coords=target_coords,
            )
        return dst_tensor.detach().cpu().numpy(), target_coords

    def _align_overlay_map_to_grid(
        self,
        overlay_map,
        target_grid_kind: str,
        *,
        overlay_grid_info: dict | None = None,
        property_name: str | None = "heat_source",
    ):
        overlay_arr = (
            overlay_map.detach().cpu().numpy()
            if isinstance(overlay_map, torch.Tensor)
            else np.asarray(overlay_map)
        )
        if target_grid_kind == "optical":
            aligned, _ = self._resample_plot_map_to_uniform_grid(
                overlay_arr,
                src_grid_info=overlay_grid_info,
                property_name=property_name,
            )
            return aligned

        thermal_grid_info = overlay_grid_info or self._get_raster_grid_info(
            "conductivity_map"
        )
        if thermal_grid_info is None:
            return overlay_arr
        target_thermal_coords = tuple(
            np.asarray(axis, dtype=np.float64) for axis in thermal_grid_info["coords"]
        )
        if (
            tuple(overlay_arr.shape)
            == tuple(int(v) for v in thermal_grid_info["shape"])
            and len(target_thermal_coords) == overlay_arr.ndim
        ):
            return overlay_arr
        src_grid_info = overlay_grid_info
        if src_grid_info is None:
            return overlay_arr
        src_coords = tuple(
            np.asarray(axis, dtype=np.float64) for axis in src_grid_info["coords"]
        )
        aligned = resample_rectilinear_tensor(
            torch.as_tensor(overlay_arr, dtype=torch.float32, device=self.device),
            src_coords=src_coords,
            dst_coords=target_thermal_coords,
        )
        return aligned.detach().cpu().numpy()

    def _ensure_plot_monitors_initialized(self):
        if getattr(self, "port_monitor_slices", None):
            return
        init_monitors = getattr(self, "init_monitors", None)
        if not callable(init_monitors):
            return
        try:
            init_monitors(verbose=False)
        except TypeError:
            init_monitors()

    @staticmethod
    def _cross_section_index_from_coord(coord_values, target_value: float) -> int:
        coord_values = np.asarray(coord_values, dtype=float)
        return int(np.argmin(np.abs(coord_values - float(target_value))))

    @staticmethod
    def _coord_extent_from_centers(coord_values):
        coord_values = np.asarray(coord_values, dtype=float)
        if coord_values.ndim != 1 or coord_values.size == 0:
            raise ValueError("coord_values must be a non-empty 1D array")
        if coord_values.size == 1:
            half_step = 0.5
        else:
            diffs = np.diff(coord_values)
            left_step = diffs[0]
            right_step = diffs[-1]
            return (
                float(coord_values[0] - 0.5 * left_step),
                float(coord_values[-1] + 0.5 * right_step),
            )
        return (
            float(coord_values[0] - half_step),
            float(coord_values[0] + half_step),
        )

    @staticmethod
    def _coords_are_uniform(coord_values, *, atol=1e-9, rtol=1e-6):
        coord_values = np.asarray(coord_values, dtype=float)
        if coord_values.size <= 2:
            return True
        diffs = np.diff(coord_values)
        return np.allclose(diffs, diffs[0], atol=atol, rtol=rtol)

    @staticmethod
    def _idx_to_phys(idx_values, coord_values):
        coord_values = np.asarray(coord_values, dtype=float)
        idx_values = np.asarray(idx_values, dtype=float)
        if coord_values.size == 0:
            return np.zeros_like(idx_values, dtype=float)
        clipped = np.clip(np.round(idx_values).astype(int), 0, coord_values.size - 1)
        result = coord_values[clipped]
        result = np.asarray(result, dtype=float)
        result[np.isnan(idx_values)] = np.nan
        return result

    @staticmethod
    def _insert_nan_gaps(vals):
        vals = np.asarray(vals, dtype=float)
        if vals.size <= 1:
            return vals
        vals = vals[np.isfinite(vals)]
        vals = np.unique(vals)
        vals.sort()
        if vals.size <= 1:
            return vals
        diffs = np.diff(vals)
        positive_diffs = diffs[diffs > 1e-12]
        step = np.median(positive_diffs) if positive_diffs.size > 0 else 1.0
        gap_threshold = max(1.5 * step, step + 1e-9)
        out = [vals[0]]
        for a, b in zip(vals[:-1], vals[1:]):
            if b - a > gap_threshold:
                out.append(np.nan)
            out.append(b)
        return np.asarray(out, dtype=float)

    def _plot_index_line(self, ax, idx_a, idx_b, coords_a, coords_b, color):
        phys_a = self._idx_to_phys(idx_a, coords_a)
        phys_b = self._idx_to_phys(idx_b, coords_b)
        ax.plot(phys_a, phys_b, color=color, alpha=0.85, linewidth=1.0, zorder=20)

    def _scatter_index_points(self, ax, idx_a, idx_b, coords_a, coords_b, color):
        phys_a = self._idx_to_phys(idx_a, coords_a)
        phys_b = self._idx_to_phys(idx_b, coords_b)
        ax.scatter(phys_a, phys_b, c=color, s=4.0, alpha=0.65, linewidths=0, zorder=21)

    def _draw_slice_overlay_2d(self, ax, slice_obj, coords_a, coords_b, color):
        if len(slice_obj.x.shape) == 0:
            ys = self._insert_nan_gaps(slice_obj.y.astype(float))
            xs = np.full_like(ys, float(slice_obj.x), dtype=float)
            xs[np.isnan(ys)] = np.nan
            self._plot_index_line(ax, xs, ys, coords_a, coords_b, color)
        elif len(slice_obj.y.shape) == 0:
            xs = self._insert_nan_gaps(slice_obj.x.astype(float))
            ys = np.full_like(xs, float(slice_obj.y), dtype=float)
            ys[np.isnan(xs)] = np.nan
            self._plot_index_line(ax, xs, ys, coords_a, coords_b, color)
        else:
            xs = slice_obj.x[:, 0].astype(float)
            ys = slice_obj.y[0].astype(float)
            x_line = self._insert_nan_gaps(xs)
            y_line = self._insert_nan_gaps(ys)
            x_min = np.nanmin(xs)
            x_max = np.nanmax(xs)
            y_min = np.nanmin(ys)
            y_max = np.nanmax(ys)
            self._plot_index_line(
                ax,
                x_line,
                np.full_like(x_line, y_min, dtype=float),
                coords_a,
                coords_b,
                color,
            )
            self._plot_index_line(
                ax,
                x_line,
                np.full_like(x_line, y_max, dtype=float),
                coords_a,
                coords_b,
                color,
            )
            self._plot_index_line(
                ax,
                np.full_like(y_line, x_min, dtype=float),
                y_line,
                coords_a,
                coords_b,
                color,
            )
            self._plot_index_line(
                ax,
                np.full_like(y_line, x_max, dtype=float),
                y_line,
                coords_a,
                coords_b,
                color,
            )

    def _draw_slice3d_overlay(
        self, ax, slice_obj, axis_index, slice_idx, coords, plane_axes, color
    ):
        coords_map = {0: slice_obj.x, 1: slice_obj.y, 2: slice_obj.z}

        # Convert slice/int/array indexers into explicit index arrays.
        # index_values = {
        #     axis: self._slice_indexer_to_indices(
        #         coords_map[axis],
        #         axis_size=len(coords[axis]),
        #     )
        #     for axis in range(3)
        # }
        index_values = {
            axis: slice_to_indices(
                coords_map[axis],
            )
            for axis in range(3)
        }

        # Only draw this monitor if it intersects the current cross-section.
        if not np.any(
            np.isclose(
                index_values[axis_index],
                float(slice_idx),
                atol=1e-6,
                rtol=0.0,
            )
        ):
            return

        h_dim, v_dim = plane_axes
        h_vals = index_values[h_dim]
        v_vals = index_values[v_dim]

        h_vals = h_vals[np.isfinite(h_vals)]
        v_vals = v_vals[np.isfinite(v_vals)]

        if h_vals.size == 0 or v_vals.size == 0:
            return

        h_vals = np.unique(h_vals)
        v_vals = np.unique(v_vals)

        h_scalar = h_vals.size <= 1
        v_scalar = v_vals.size <= 1

        coords_a = coords[h_dim]
        coords_b = coords[v_dim]

        if h_scalar and v_scalar:
            self._scatter_index_points(
                ax, [h_vals[0]], [v_vals[0]], coords_a, coords_b, color
            )
        elif h_scalar:
            v_line = self._insert_nan_gaps(v_vals)
            h_line = np.full_like(v_line, h_vals[0], dtype=float)
            h_line[np.isnan(v_line)] = np.nan
            self._plot_index_line(ax, h_line, v_line, coords_a, coords_b, color)
        elif v_scalar:
            h_line = self._insert_nan_gaps(h_vals)
            v_line = np.full_like(h_line, v_vals[0], dtype=float)
            v_line[np.isnan(h_line)] = np.nan
            self._plot_index_line(ax, h_line, v_line, coords_a, coords_b, color)
        else:
            h_line = self._insert_nan_gaps(h_vals)
            v_line = self._insert_nan_gaps(v_vals)

            h_min, h_max = np.nanmin(h_vals), np.nanmax(h_vals)
            v_min, v_max = np.nanmin(v_vals), np.nanmax(v_vals)

            self._plot_index_line(
                ax,
                h_line,
                np.full_like(h_line, v_min, dtype=float),
                coords_a,
                coords_b,
                color,
            )
            self._plot_index_line(
                ax,
                h_line,
                np.full_like(h_line, v_max, dtype=float),
                coords_a,
                coords_b,
                color,
            )
            self._plot_index_line(
                ax,
                np.full_like(v_line, h_min, dtype=float),
                v_line,
                coords_a,
                coords_b,
                color,
            )
            self._plot_index_line(
                ax,
                np.full_like(v_line, h_max, dtype=float),
                v_line,
                coords_a,
                coords_b,
                color,
            )

    def _draw_mask_overlay(self, ax, mask, coords_a, coords_b):
        idx_a, idx_b = np.nonzero(np.asarray(mask))
        if len(idx_a) > 0:
            self._scatter_index_points(ax, idx_a, idx_b, coords_a, coords_b, "purple")

    def _draw_npml_overlay(self, ax, plane_axes, map_shape, coords):
        import matplotlib.pyplot as plt

        if len(map_shape) != 2:
            return
        x_axis, y_axis = plane_axes
        npml_pairs = {
            axis: (
                self.NPML_export[_AXES.index(axis)]
                if _AXES.index(axis) < len(self.NPML_export)
                else 0
            )
            for axis in plane_axes
        }
        n0, n1 = map_shape
        c0 = coords[_AXES.index(x_axis)]
        c1 = coords[_AXES.index(y_axis)]
        if c0.size == 0 or c1.size == 0:
            return
        dx = abs(float(c0[1] - c0[0])) if c0.size > 1 else self.grid_step
        dy = abs(float(c1[1] - c1[0])) if c1.size > 1 else self.grid_step
        x_min, x_max = float(c0[0] - dx / 2), float(c0[-1] + dx / 2)
        y_min, y_max = float(c1[0] - dy / 2), float(c1[-1] + dy / 2)
        pml0 = min(max(int(npml_pairs[x_axis]), 0), n0 // 2)
        pml1 = min(max(int(npml_pairs[y_axis]), 0), n1 // 2)
        if pml0 == 0 and pml1 == 0:
            return
        pml_x = pml0 * dx
        pml_y = pml1 * dy
        rect_kw = dict(facecolor="gray", alpha=0.25, edgecolor="none", zorder=5)
        if pml_x > 0:
            ax.add_patch(plt.Rectangle((x_min, y_min), pml_x, y_max - y_min, **rect_kw))
            ax.add_patch(
                plt.Rectangle((x_max - pml_x, y_min), pml_x, y_max - y_min, **rect_kw)
            )
        if pml_y > 0:
            inner_x0 = x_min + pml_x
            inner_w = max((x_max - x_min) - 2 * pml_x, 0.0)
            ax.add_patch(plt.Rectangle((inner_x0, y_min), inner_w, pml_y, **rect_kw))
            ax.add_patch(
                plt.Rectangle((inner_x0, y_max - pml_y), inner_w, pml_y, **rect_kw)
            )

    def _draw_heat_source_overlay(
        self,
        ax,
        *,
        plane_axes,
        axis_index=None,
        slice_coord=None,
    ):
        import matplotlib.pyplot as plt

        if not self.heat_source_cfgs:
            return

        def _intersects_slice(center_val, size_val, coord_val):
            half = float(size_val) / 2.0
            return (
                (float(center_val) - half)
                <= float(coord_val)
                <= (float(center_val) + half)
            )

        def _draw_box(cfg, color):
            center = np.asarray(_as_3(cfg["center"], fill=0.0), dtype=float)
            size = np.asarray(
                _as_3(cfg["size"], fill=td.inf if self.dim == 2 else 0.0), dtype=float
            )
            if axis_index is not None and not _intersects_slice(
                center[axis_index], size[axis_index], slice_coord
            ):
                return
            h_axis, v_axis = plane_axes
            x0 = center[h_axis] - size[h_axis] / 2.0
            y0 = center[v_axis] - size[v_axis] / 2.0
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0),
                    float(size[h_axis]),
                    float(size[v_axis]),
                    fill=False,
                    edgecolor=color,
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=22,
                )
            )

        def _draw_prism(cfg, color):
            vertices = np.asarray(
                [_as_3(v, fill=0.0) for v in cfg["vertices"]], dtype=float
            )
            if axis_index is None:
                poly = vertices[:, list(plane_axes)]
            else:
                height = cfg.get("height", 0.0)
                if not np.isfinite(height):
                    height = float("inf")
                center_axis = 0.0
                if axis_index == 2:
                    if not _intersects_slice(center_axis, height, slice_coord):
                        return
                    poly = vertices[:, list(plane_axes)]
                else:
                    coord_fixed = vertices[:, axis_index]
                    if not np.allclose(
                        coord_fixed, coord_fixed[0], atol=1e-9, rtol=1e-6
                    ):
                        poly = vertices[:, list(plane_axes)]
                    else:
                        if not np.isclose(
                            coord_fixed[0], slice_coord, atol=1e-9, rtol=1e-6
                        ):
                            return
                        other_axis = plane_axes[0]
                        z_half = float(height) / 2.0 if np.isfinite(height) else 1.0
                        other_vals = vertices[:, other_axis]
                        poly = np.array(
                            [
                                [other_vals.min(), -z_half],
                                [other_vals.max(), -z_half],
                                [other_vals.max(), z_half],
                                [other_vals.min(), z_half],
                            ],
                            dtype=float,
                        )
            ax.add_patch(
                plt.Polygon(
                    poly,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=22,
                )
            )

        def _draw_cylinder(cfg, color):
            center = np.asarray(
                _as_3(cfg.get("center", (0, 0, 0)), fill=0.0), dtype=float
            )
            radius = float(cfg["radius"])
            axis = int(cfg.get("axis", 2))
            length = cfg.get("height", cfg.get("length", td.inf))
            h_axis, v_axis = plane_axes
            if axis_index is None:
                if axis == 2:
                    ax.add_patch(
                        plt.Circle(
                            (center[h_axis], center[v_axis]),
                            radius,
                            fill=False,
                            edgecolor=color,
                            linewidth=1.2,
                            alpha=0.9,
                            zorder=22,
                        )
                    )
                    return
                x0 = center[h_axis] - radius
                y0 = center[v_axis] - radius
                ax.add_patch(
                    plt.Rectangle(
                        (x0, y0),
                        2 * radius,
                        2 * radius,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.2,
                        alpha=0.9,
                        zorder=22,
                    )
                )
                return
            if axis_index == axis:
                if np.isfinite(length) and not _intersects_slice(
                    center[axis], length, slice_coord
                ):
                    return
                ax.add_patch(
                    plt.Circle(
                        (center[h_axis], center[v_axis]),
                        radius,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.2,
                        alpha=0.9,
                        zorder=22,
                    )
                )
                return
            if not _intersects_slice(center[axis_index], 2 * radius, slice_coord):
                return
            lengths = np.array([2 * radius, 2 * radius, 2 * radius], dtype=float)
            if np.isfinite(length):
                lengths[axis] = float(length)
            x0 = center[h_axis] - lengths[h_axis] / 2.0
            y0 = center[v_axis] - lengths[v_axis] / 2.0
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0),
                    float(lengths[h_axis]),
                    float(lengths[v_axis]),
                    fill=False,
                    edgecolor=color,
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=22,
                )
            )

        for name, cfg in self.heat_source_cfgs.items():
            color = "orange" if "heater" in name.lower() else "gold"
            geo_type = str(cfg.get("type", "box")).lower()
            try:
                if geo_type == "box":
                    _draw_box(cfg, color)
                elif geo_type == "prism":
                    _draw_prism(cfg, color)
                elif geo_type == "cylinder":
                    _draw_cylinder(cfg, color)
            except Exception:
                continue

    def _overlay_device_plot_annotations(
        self,
        ax,
        *,
        plot_coords,
        plane_axes,
        axis_index=None,
        slice_idx=None,
        overlay_monitors: bool = True,
        overlay_pml: bool = True,
        overlay_heat_sources: bool = False,
        map_shape=None,
    ):
        coords_a = plot_coords[plane_axes[0]]
        coords_b = plot_coords[plane_axes[1]]

        if overlay_pml:
            self._draw_npml_overlay(
                ax, tuple(_AXES[idx] for idx in plane_axes), map_shape, plot_coords
            )

        if overlay_monitors:
            for name, monitor in self.port_monitor_slices.items():
                if name.startswith("rad_"):
                    color = "g"
                elif name.startswith("in_"):
                    color = "r"
                else:
                    color = "b"
                if hasattr(monitor, "z") and axis_index is not None:
                    self._draw_slice3d_overlay(
                        ax,
                        monitor,
                        axis_index,
                        slice_idx,
                        plot_coords,
                        plane_axes,
                        color,
                    )
                elif (
                    hasattr(monitor, "x")
                    and hasattr(monitor, "y")
                    and axis_index is None
                ):
                    self._draw_slice_overlay_2d(ax, monitor, coords_a, coords_b, color)
                elif isinstance(monitor, np.ndarray):
                    if monitor.ndim == 2 and axis_index is None:
                        self._draw_mask_overlay(ax, monitor, coords_a, coords_b)
                    elif monitor.ndim == 3 and axis_index is not None:
                        mask_view = np.take(monitor, indices=slice_idx, axis=axis_index)
                        self._draw_mask_overlay(ax, mask_view, coords_a, coords_b)

        if overlay_heat_sources:
            slice_coord = (
                None
                if axis_index is None
                else float(plot_coords[axis_index][slice_idx])
            )
            self._draw_heat_source_overlay(
                ax,
                plane_axes=plane_axes,
                axis_index=axis_index,
                slice_coord=slice_coord,
            )

    @staticmethod
    def _plot_value_mode(
        values,
        *,
        value_mode: str = "auto",
        property_name: str | None = None,
    ):
        values = np.asarray(values)
        mode = str(value_mode).strip().lower()
        if mode == "auto":
            if np.iscomplexobj(values):
                if property_name in {"epsilon", "permittivity", "eps"}:
                    mode = "real"
                else:
                    mode = "abs"
            else:
                mode = "real"

        if mode == "real":
            return np.real(values), mode
        if mode == "imag":
            return np.imag(values), mode
        if mode == "abs":
            return np.abs(values), mode
        if mode in {"abs_sq", "abs2", "magnitude_sq"}:
            return np.abs(values) ** 2, "abs_sq"
        if mode == "phase":
            return np.angle(values), mode
        raise ValueError(
            "value_mode must be one of auto, real, imag, abs, abs_sq, or phase"
        )

    def _slice_indexer_to_indices(self, indexer, axis_size: int):
        if isinstance(indexer, slice):
            start, stop, step = indexer.indices(int(axis_size))
            return np.arange(start, stop, step, dtype=float)

        arr = np.asarray(indexer)
        if arr.ndim == 0:
            return np.asarray([float(arr)], dtype=float)

        return arr.astype(float).reshape(-1)

    def plot_property(
        self,
        property_name: str | None = None,
        *,
        property_map=None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        cmap: str = "viridis",
        vmin=None,
        vmax=None,
        value_mode: str = "auto",
        overlay_monitors: bool = True,
        overlay_pml: bool = True,
        overlay_heat_sources: bool = False,
        heat_source_map=None,
        ax=None,
        colorbar: bool = True,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        interpolation: str = "nearest",
        aspect: str = "equal",
    ):
        import matplotlib.pyplot as plt

        if overlay_monitors:
            self._ensure_plot_monitors_initialized()
        if overlay_heat_sources and heat_source_map is None and self.heat_source_cfgs:
            heat_source_map = None

        (
            plot_map,
            coords,
            normalized_name,
            grid_kind,
            plot_grid_info,
        ) = self._resolve_property_plot_data(
            property_name=property_name,
            property_map=property_map,
        )
        plot_map, plot_coords = self._resample_plot_map_to_uniform_grid(
            plot_map,
            src_grid_info=plot_grid_info,
            property_name=normalized_name,
        )
        coords = tuple(
            np.asarray(axis_coords, dtype=float) for axis_coords in plot_coords
        )
        grid_kind = "optical"
        if len(coords) != plot_map.ndim:
            raise ValueError(
                f"Coordinate dimension mismatch: map has ndim={plot_map.ndim}, coords={len(coords)}"
            )

        requested = {
            axis: value
            for axis, value in {"x": x, "y": y, "z": z}.items()
            if value is not None
        }
        if plot_map.ndim == 2:
            if requested:
                raise ValueError(
                    "2D property maps do not support x/y/z cross-section selection; "
                    "plotting uses the full x-y plane."
                )
            view = plot_map
            x0, x1 = self._coord_extent_from_centers(coords[0])
            y0, y1 = self._coord_extent_from_centers(coords[1])
            extent = (x0, x1, y0, y1)
            plane_axes = (0, 1)
            axis_index = None
            slice_idx = None
        elif plot_map.ndim == 3:
            if len(requested) != 1:
                raise ValueError(
                    "3D property plots require exactly one of x=..., y=..., or z=... "
                    "to specify the cross-section location in um."
                )
            slice_axis, slice_coord = next(iter(requested.items()))
            axis_index = _AXES.index(slice_axis)
            slice_idx = self._cross_section_index_from_coord(
                coords[axis_index], slice_coord
            )
            view = np.take(plot_map, indices=slice_idx, axis=axis_index)
            remaining_axes = [axis for axis in range(3) if axis != axis_index]
            plane_axes = tuple(remaining_axes)
            x0, x1 = self._coord_extent_from_centers(coords[remaining_axes[0]])
            y0, y1 = self._coord_extent_from_centers(coords[remaining_axes[1]])
            extent = (x0, x1, y0, y1)
        else:
            raise ValueError(
                f"Only 2D and 3D property maps are supported, got ndim={plot_map.ndim}"
            )

        view_to_plot, resolved_value_mode = self._plot_value_mode(
            view,
            value_mode=value_mode,
            property_name=normalized_name,
        )

        created_fig = False
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
            created_fig = True

        image = ax.imshow(
            np.asarray(view_to_plot).T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            aspect=aspect,
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        if xlabel is None:
            xlabel = f"{_AXES[plane_axes[0]]} (um)"
        if ylabel is None:
            ylabel = f"{_AXES[plane_axes[1]]} (um)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        self._overlay_device_plot_annotations(
            ax,
            plot_coords=coords,
            plane_axes=plane_axes,
            axis_index=axis_index,
            slice_idx=slice_idx,
            overlay_monitors=overlay_monitors and grid_kind == "optical",
            overlay_pml=overlay_pml and grid_kind == "optical",
            overlay_heat_sources=overlay_heat_sources,
            map_shape=view.shape,
        )

        if title is None:
            display_name = normalized_name
            if np.iscomplexobj(view):
                display_name = f"{display_name} ({resolved_value_mode})"
            if plot_map.ndim == 3:
                title = (
                    f"{display_name} at "
                    f"{next(iter(requested.keys()))}={float(next(iter(requested.values()))):.4g} um"
                )
            else:
                title = display_name
        ax.set_title(title)

        if colorbar:
            ax.figure.colorbar(image, ax=ax)

        if created_fig:
            ax.figure.tight_layout()
        return ax, image

    def plot_eps(self, *args, **kwargs):
        return self.plot_property("epsilon", *args, **kwargs)

    def get_conductivity_map(
        self,
        cell_size,
        PML,
        resolution,
        use_tidy3d: bool = False,
        spacing: Sequence[float] | None = None,
        grid_spec_cfg=None,
        reuse_grid_info: dict | None = None,
    ):
        """
        Build the thermal-conductivity map on the configured simulation/raster grid.

        Args:
            cell_size (Sequence[float]): Physical simulation-cell size in micrometers. The first ``self.dim`` entries are used.
            PML (Sequence[float]): PML thickness configuration in the device coordinate system.
            resolution (float): Optical/thermal raster resolution in pixels per micrometer when a uniform raster is requested.
            use_tidy3d (bool): Whether to build the property map using Tidy3D rasterization and harmonic subpixel averaging. Defaults to False.
            spacing (Sequence[float] | None): Explicit thermal-grid spacing in micrometers. Defaults to None.
            grid_spec_cfg (dict | Any | None): Optional Tidy3D grid specification or configuration mapping. Defaults to None.
            reuse_grid_info (dict | None): Existing raster-grid metadata to reuse when rebuilding a map. Defaults to None.

        Returns:
            NumPy array containing thermal conductivity values on the generated or reused raster grid.
        """

        def _reciprocal_conductivity(value: float) -> float:
            value = float(value)
            if value <= 0:
                raise ValueError(
                    "Thermal conductivity must be strictly positive to use harmonic subpixel averaging."
                )
            return 1.0 / value

        def _invert_resistivity_map(array):
            array = np.asarray(array, dtype=np.float32)
            eps = np.finfo(np.float32).tiny
            return 1.0 / np.clip(array, eps, None)

        return self._build_scalar_property_map(
            map_name="conductivity_map",
            value_keys=_HEAT_PROPERTY_KEYS,
            bg_value=self.k_bg,
            value_from_cfg=self._heat_value_from_cfg,
            bg_from_cfg=self._heat_bg_from_cfg,
            cell_size=cell_size,
            PML=PML,
            resolution=resolution,
            use_tidy3d=use_tidy3d,
            spacing=spacing,
            grid_spec_cfg=grid_spec_cfg,
            preprocess_value=_reciprocal_conductivity if use_tidy3d else None,
            postprocess_map=_invert_resistivity_map if use_tidy3d else None,
            is_complex=False,
            reuse_grid_info=reuse_grid_info,
        )

    def get_electrical_conductivity_map(
        self,
        cell_size,
        PML,
        resolution,
        use_tidy3d: bool = False,
        spacing: Sequence[float] | None = None,
        grid_spec_cfg=None,
        reuse_grid_info: dict | None = None,
    ):
        """
        Build the electrical-conductivity map on the configured simulation/raster grid.

        Args:
            cell_size (Sequence[float]): Physical simulation-cell size in micrometers. The first ``self.dim`` entries are used.
            PML (Sequence[float]): PML thickness configuration in the device coordinate system.
            resolution (float): Raster resolution in pixels per micrometer when a uniform raster is requested.
            use_tidy3d (bool): Whether to use Tidy3D rasterization for the property map. Defaults to False.
            spacing (Sequence[float] | None): Explicit thermal-grid spacing in micrometers. Defaults to None.
            grid_spec_cfg (dict | Any | None): Optional Tidy3D grid specification or configuration mapping. Defaults to None.
            reuse_grid_info (dict | None): Existing raster-grid metadata to reuse when rebuilding a map. Defaults to None.

        Returns:
            NumPy array containing electrical conductivity values on the generated or reused raster grid.
        """
        return self._build_scalar_property_map(
            map_name="electrical_conductivity_map",
            value_keys=_ELECTRICAL_CONDUCTIVITY_KEYS,
            bg_value=(
                self.electrical_conductivity_bg
                if self.electrical_conductivity_bg is not None
                else 0.0
            ),
            value_from_cfg=self._electrical_conductivity_value_from_cfg,
            bg_from_cfg=self._electrical_conductivity_bg_from_cfg,
            cell_size=cell_size,
            PML=PML,
            resolution=resolution,
            use_tidy3d=use_tidy3d,
            spacing=spacing,
            grid_spec_cfg=grid_spec_cfg,
            is_complex=False,
            reuse_grid_info=reuse_grid_info,
        )

    def get_heat_capacity_map(
        self,
        cell_size,
        PML,
        resolution,
        use_tidy3d: bool = False,
        spacing: Sequence[float] | None = None,
        grid_spec_cfg=None,
        reuse_grid_info: dict | None = None,
    ):
        """
        Build the heat-capacity map on the configured simulation/raster grid.

        Args:
            cell_size (Sequence[float]): Physical simulation-cell size in micrometers. The first ``self.dim`` entries are used.
            PML (Sequence[float]): PML thickness configuration in the device coordinate system.
            resolution (float): Raster resolution in pixels per micrometer when a uniform raster is requested.
            use_tidy3d (bool): Whether to use Tidy3D rasterization for the property map. Defaults to False.
            spacing (Sequence[float] | None): Explicit thermal-grid spacing in micrometers. Defaults to None.
            grid_spec_cfg (dict | Any | None): Optional Tidy3D grid specification or configuration mapping. Defaults to None.
            reuse_grid_info (dict | None): Existing raster-grid metadata to reuse when rebuilding a map. Defaults to None.

        Returns:
            NumPy array containing volumetric heat-capacity values on the generated or reused raster grid.
        """
        return self._build_scalar_property_map(
            map_name="heat_capacity_map",
            value_keys=_HEAT_CAPACITY_KEYS,
            bg_value=self.heat_capacity_bg,
            value_from_cfg=self._heat_capacity_value_from_cfg,
            bg_from_cfg=self._heat_capacity_bg_from_cfg,
            cell_size=cell_size,
            PML=PML,
            resolution=resolution,
            use_tidy3d=use_tidy3d,
            spacing=spacing,
            grid_spec_cfg=grid_spec_cfg,
            is_complex=False,
            reuse_grid_info=reuse_grid_info,
        )

    def get_thermo_optic_coeff_map(
        self,
        cell_size,
        PML,
        resolution,
        use_tidy3d: bool = False,
        spacing: Sequence[float] | None = None,
        grid_spec_cfg=None,
        reuse_grid_info: dict | None = None,
    ):
        """
        Build the thermo-optic coefficient map on the configured simulation/raster grid.

        Args:
            cell_size (Sequence[float]): Physical simulation-cell size in micrometers. The first ``self.dim`` entries are used.
            PML (Sequence[float]): PML thickness configuration in the device coordinate system.
            resolution (float): Raster resolution in pixels per micrometer when a uniform raster is requested.
            use_tidy3d (bool): Whether to use Tidy3D rasterization for the property map. Defaults to False.
            spacing (Sequence[float] | None): Explicit thermal-grid spacing in micrometers. Defaults to None.
            grid_spec_cfg (dict | Any | None): Optional Tidy3D grid specification or configuration mapping. Defaults to None.
            reuse_grid_info (dict | None): Existing raster-grid metadata to reuse when rebuilding a map. Defaults to None.

        Returns:
            NumPy array containing thermo-optic coefficient values on the generated or reused raster grid.
        """
        return self._build_scalar_property_map(
            map_name="thermo_optic_coeff_map",
            value_keys=_THERMO_OPTIC_KEYS,
            bg_value=self.thermo_optic_coeff_bg,
            value_from_cfg=self._thermo_optic_value_from_cfg,
            bg_from_cfg=self._thermo_optic_bg_from_cfg,
            cell_size=cell_size,
            PML=PML,
            resolution=resolution,
            use_tidy3d=use_tidy3d,
            spacing=spacing,
            grid_spec_cfg=grid_spec_cfg,
            is_complex=False,
            reuse_grid_info=reuse_grid_info,
        )

    def build_heat_source_maps(
        self,
        heat_source_cfgs: dict | None = None,
        *,
        use_tidy3d: bool | None = None,
        combine: bool | None = None,
    ):
        """
        Rasterize configured heat sources into per-source maps and optionally combine them.

        Args:
            heat_source_cfgs (dict | None): Mapping from source name to heat-source configuration. Defaults to None.
            use_tidy3d (bool | None): Whether to use the Tidy3D thermal rasterization path. Defaults to None.
            combine (bool | None): Whether to also store the elementwise sum of all generated source maps in ``self.heat_source_map``. Defaults to None.

        Returns:
            dict[str, np.ndarray]: Mapping from source name to its rasterized volumetric heat-source map.

        Side Effects:
            Updates ``self.heat_sources_dict`` and, when ``combine`` is enabled, ``self.heat_source_map``.
        """
        if use_tidy3d is None:
            use_tidy3d = self._heat_use_tidy3d_default()
        if combine is None:
            combine = self._heat_combine_sources_default()
        cfgs = self.heat_source_cfgs if heat_source_cfgs is None else heat_source_cfgs
        heat_spacing = self._heat_mesh_spacing()
        heat_resolution = self._heat_mesh_resolution_for_raster() or self.resolution
        heat_grid_spec_cfg = self._heat_tidy3d_grid_spec_cfg()
        if use_tidy3d and self._thermal_raster_grid_spec is not None:
            heat_grid_spec_cfg = self._thermal_raster_grid_spec
        raster_cfgs = self._heat_source_raster_cfgs(cfgs) if use_tidy3d else cfgs
        thermal_grid_info = self._get_raster_grid_info("conductivity_map")
        built_sources = {}
        for name, cfg in cfgs.items():
            source_map = self._build_scalar_property_map(
                cfgs=(
                    (
                        {
                            dummy_name: dummy_cfg
                            for dummy_name, dummy_cfg in raster_cfgs.items()
                            if dummy_name != name
                        }
                        | {name: raster_cfgs[name]}
                    )
                    if use_tidy3d
                    else {name: cfg}
                ),
                map_name=f"heat_source_map_'{name}'",
                value_keys=_HEAT_SOURCE_VALUE_KEYS + ("total_power", "current"),
                bg_value=0.0,
                value_from_cfg=self._heat_source_value_from_cfg,
                bg_from_cfg=self._heat_source_bg_from_cfg,
                cell_size=self.cell_size,
                PML=self.sim_cfg["PML"],
                resolution=heat_resolution,
                use_tidy3d=use_tidy3d,
                spacing=heat_spacing,
                grid_spec_cfg=heat_grid_spec_cfg,
                postprocess_map=(
                    self._postprocess_heat_source_raster_map if use_tidy3d else None
                ),
                encoded_value_override_from_cfg=(
                    self._heat_source_encoded_value_override if use_tidy3d else None
                ),
                is_complex=False,
                reuse_grid_info=thermal_grid_info,
                include_override_structures=False,
            )
            if source_map is not None:
                built_sources[name] = source_map

        self.heat_sources_dict = built_sources
        if combine and built_sources:
            combined = np.zeros_like(next(iter(built_sources.values())))
            for source_map in built_sources.values():
                combined = combined + source_map
            self.heat_source_map = combined
        elif combine:
            self.heat_source_map = None
        return self.heat_sources_dict

    def rebuild_heat_source_maps(
        self,
        heat_source_cfgs: dict | None = None,
        *,
        use_tidy3d: bool | None = None,
        combine: bool | None = None,
    ):
        """
        Rebuild the configured heat-source maps using the same rasterization path as ``build_heat_source_maps``.

        Args:
            heat_source_cfgs (dict | None): Mapping from source name to heat-source configuration. Defaults to None.
            use_tidy3d (bool | None): Whether to use the Tidy3D thermal rasterization path. Defaults to None.
            combine (bool | None): Whether to also rebuild the combined heat-source map. Defaults to None.

        Returns:
            dict[str, np.ndarray]: Mapping from source name to its rebuilt rasterized heat-source map.
        """
        return self.build_heat_source_maps(
            heat_source_cfgs=heat_source_cfgs,
            use_tidy3d=use_tidy3d,
            combine=combine,
        )

    def get_heat_source_map(self, source_name: str | None = None):
        """
        Return the combined heat-source map or one named source map.

        Args:
            source_name (str | None): Name of a source in ``self.heat_sources_dict``. When omitted, the combined map is returned. Defaults to None.

        Returns:
            np.ndarray | None: The requested source map, or ``self.heat_source_map`` when ``source_name`` is None.

        Raises:
            KeyError: If ``source_name`` is provided but is not present in ``self.heat_sources_dict``.
        """
        if source_name is None:
            return self.heat_source_map
        if source_name not in self.heat_sources_dict:
            raise KeyError(f"Heat source '{source_name}' is not available.")
        return self.heat_sources_dict[source_name]

    def _heat_sim_cfg(self):
        heat_cfg = self.sim_cfg.get("heat", {})
        return heat_cfg if isinstance(heat_cfg, dict) else {}

    def _heat_use_tidy3d_default(self):
        heat_cfg = self._heat_sim_cfg()
        use_tidy3d = heat_cfg.get("use_tidy3d")
        if use_tidy3d is not None:
            return bool(use_tidy3d)
        spacing = self._heat_mesh_spacing()
        if not np.allclose(spacing, [spacing[0]] * self.dim, atol=1e-12, rtol=1e-12):
            return TD_SUPPORTED
        return (1.0 / spacing[0]) >= 25 and TD_SUPPORTED

    def _heat_backend_default(self):
        return self._heat_sim_cfg().get("backend", "jax")

    def _heat_solve_interval(self):
        interval = self._heat_sim_cfg().get("solve_interval", 1)
        try:
            interval = int(interval)
        except (TypeError, ValueError):
            interval = 1
        return max(interval, 1)

    def _heat_mesh_type_default(self):
        return _normalize_heat_mesh_type(
            self._heat_sim_cfg().get("mesh_type", "rectangular")
        )

    def _heat_fixed_mesh_cfg(self):
        heat_cfg = self._heat_sim_cfg()
        mesh_cfg = heat_cfg.get("fixed_mesh")
        if isinstance(mesh_cfg, dict):
            return mesh_cfg
        mesh_keys = {
            "mesh_path",
            "vtu_path",
            "mesh_file",
            "points",
            "cells",
            "xs",
            "ys",
            "zs",
            "x",
            "y",
            "z",
            "axes",
            "coords",
            "ele_type",
            "cell_type",
            "meshio_cell_type",
            "normalize_to_origin",
            "transfer_neighbors",
            "grid_shape",
            "distance_unstructured_grid",
            "dl_interface",
            "dl_bulk",
            "distance_interface",
            "distance_bulk",
            "sampling",
            "uniform_grid_mediums",
            "non_refined_structures",
            "mesh_refinements",
        }
        derived = {key: heat_cfg[key] for key in mesh_keys if key in heat_cfg}
        return derived

    def _heat_mesh_spacing(self):
        heat_cfg = self._heat_sim_cfg()
        spacing = _coalesce(
            heat_cfg.get("spacing"),
            heat_cfg.get("grid_step"),
            heat_cfg.get("resolution"),
        )
        if spacing is None:
            return tuple([self.grid_step] * self.dim)
        return _spacing_values(spacing, self.dim, default=self.grid_step)

    def _heat_mesh_resolution_for_raster(self):
        spacing = self._heat_mesh_spacing()
        if np.allclose(spacing, [spacing[0]] * self.dim, atol=1e-12, rtol=1e-12):
            return float(1.0 / spacing[0])
        return None

    def _build_tidy3d_grid_axis(self, axis_spec, fallback_dl):
        if axis_spec is None:
            return td.UniformGrid(dl=float(fallback_dl))
        if isinstance(
            axis_spec,
            (
                td.UniformGrid,
                td.AutoGrid,
                td.QuasiUniformGrid,
                td.CustomGrid,
                td.CustomGridBoundaries,
            ),
        ):
            return axis_spec
        if isinstance(axis_spec, (int, float)):
            return td.UniformGrid(dl=float(axis_spec))
        if not isinstance(axis_spec, dict):
            raise TypeError(f"Unsupported tidy3d grid axis spec: {type(axis_spec)!r}")
        axis_type = (
            str(axis_spec.get("type", "uniform")).strip().lower().replace("-", "_")
        )
        cfg = {k: v for k, v in axis_spec.items() if k != "type"}
        if axis_type == "uniform":
            cfg.setdefault("dl", float(fallback_dl))
            return td.UniformGrid(**cfg)
        if axis_type == "auto":
            cfg.setdefault("min_steps_per_wvl", max(6.0, 1.55 / float(fallback_dl)))
            return td.AutoGrid(**cfg)
        if axis_type in {"quasi_uniform", "quasiuniform"}:
            cfg.setdefault("dl", float(fallback_dl))
            return td.QuasiUniformGrid(**cfg)
        if axis_type == "custom":
            return td.CustomGrid(**cfg)
        if axis_type in {"custom_boundaries", "customgridboundaries"}:
            return td.CustomGridBoundaries(**cfg)
        raise ValueError(f"Unsupported tidy3d grid axis type: {axis_type!r}")

    def _tidy3d_grid_spec_for_raster(
        self, spacing_values, *, grid_spec_cfg=None, override_structures=None
    ):
        if not TD_SUPPORTED:
            raise ImportError("tidy3d is required for tidy3d raster grid construction.")
        cfg = grid_spec_cfg
        if cfg is None:
            grid_kwargs = {
                "wavelength": 1.55,
                "grid_x": td.UniformGrid(dl=spacing_values[0]),
                "grid_y": td.UniformGrid(dl=spacing_values[1]),
            }
            if self.dim == 3:
                grid_kwargs["grid_z"] = td.UniformGrid(dl=spacing_values[2])
            return td.GridSpec(**grid_kwargs)
        if isinstance(cfg, td.GridSpec):
            return cfg
        if not isinstance(cfg, dict):
            raise TypeError(f"Unsupported tidy3d grid_spec config type: {type(cfg)!r}")
        wavelength = float(cfg.get("wavelength", 1.55))
        grid_kwargs = {"wavelength": wavelength}
        axis_map = {
            "x": cfg.get("grid_x", cfg.get("x")),
            "y": cfg.get("grid_y", cfg.get("y")),
        }
        if self.dim == 3:
            axis_map["z"] = cfg.get("grid_z", cfg.get("z"))
        for axis_index, axis_name in enumerate(self.axes):
            grid_kwargs[f"grid_{axis_name}"] = self._build_tidy3d_grid_axis(
                axis_map.get(axis_name),
                spacing_values[axis_index],
            )
        for key in ("override_structures", "snapping_points", "layer_refinement_specs"):
            if key in cfg:
                grid_kwargs[key] = cfg[key]
        if override_structures is not None:
            grid_kwargs["override_structures"] = override_structures.values()
            print(
                f"Applied {len(override_structures)} override structures to tidy3d grid spec. {override_structures}"
            )
        return td.GridSpec(**grid_kwargs)

    def _cfg_bounds_3d(self, cfg):
        geo_type = str(cfg.get("type", "box")).lower()
        if geo_type == "box":
            center = np.asarray(_as_3(cfg["center"], fill=0.0), dtype=float)
            size = np.asarray(_as_3(cfg["size"], fill=0.0), dtype=float)
            return center - size / 2.0, center + size / 2.0
        if geo_type == "prism":
            if self.dim == 3:
                # PolySlab vertices occupy the two axes transverse to its
                # extrusion axis; slab_bounds occupy the extrusion axis.
                # The Tidy3D config builder only forwards a non-default axis
                # when sidewall settings are present.
                axis = (
                    int(cfg.get("axis", 2))
                    if cfg.get("sidewall_angle") is not None
                    else 2
                )
                if axis not in (0, 1, 2):
                    raise ValueError("Prism axis must be 0, 1, or 2.")
                vertices = np.asarray(cfg["vertices"], dtype=float)
                if vertices.ndim != 2 or vertices.shape[1] != 2:
                    raise ValueError("Prism vertices must have shape (N, 2).")
                transverse_axes = [idx for idx in range(3) if idx != axis]
                mins = np.full(3, np.inf)
                maxs = np.full(3, -np.inf)
                mins[transverse_axes] = vertices.min(axis=0)
                maxs[transverse_axes] = vertices.max(axis=0)
                slab_bounds = cfg.get("slab_bounds")
                if slab_bounds is not None:
                    slab_bounds = np.asarray(slab_bounds, dtype=float)
                    if (
                        slab_bounds.shape != (2,)
                        or np.any(np.isnan(slab_bounds))
                        or slab_bounds[1] < slab_bounds[0]
                    ):
                        raise ValueError(
                            "Prism slab_bounds must be an ordered pair of values."
                        )
                    mins[axis], maxs[axis] = slab_bounds
                elif axis == 2:
                    # Preserve the legacy height convention when the prism
                    # does not provide its Tidy3D slab_bounds explicitly.
                    mins[axis] = maxs[axis] = 0.0
                    height = cfg.get("height", 0.0)
                    if np.isfinite(height):
                        mins[axis] = min(mins[axis], -float(height) / 2.0)
                        maxs[axis] = max(maxs[axis], float(height) / 2.0)
                else:
                    # Tidy3D uses unbounded slab limits when omitted.
                    mins[axis], maxs[axis] = -np.inf, np.inf
            else:
                vertices = np.asarray(
                    [_as_3(v, fill=0.0) for v in cfg["vertices"]], dtype=float
                )
                mins = vertices.min(axis=0)
                maxs = vertices.max(axis=0)
                mins[2] = 0.0
                maxs[2] = 0.0
            return mins, maxs
        if geo_type == "cylinder":
            center = np.asarray(
                _as_3(cfg.get("center", (0, 0, 0)), fill=0.0), dtype=float
            )
            radius = float(cfg["radius"])
            mins = center.copy()
            maxs = center.copy()
            mins[:2] -= radius
            maxs[:2] += radius
            axis = int(cfg.get("axis", 2))
            if self.dim == 3:
                length = cfg.get("height", cfg.get("length", 0.0))
                if np.isfinite(length):
                    mins[axis] = center[axis] - float(length) / 2.0
                    maxs[axis] = center[axis] + float(length) / 2.0
                else:
                    mins[axis] = center[axis]
                    maxs[axis] = center[axis]
                transverse_axes = [idx for idx in range(3) if idx != axis]
                for transverse_axis in transverse_axes:
                    mins[transverse_axis] = center[transverse_axis] - radius
                    maxs[transverse_axis] = center[transverse_axis] + radius
            else:
                mins[2] = 0.0
                maxs[2] = 0.0
            return mins, maxs
        if geo_type in {"group", "geometry_group"} or "geometries" in cfg:
            children = cfg.get("geometries", [])
            if not children:
                raise ValueError("Geometry group requires non-empty geometries.")
            child_bounds = [self._cfg_bounds_3d(child) for child in children]
            mins = np.min([bounds[0] for bounds in child_bounds], axis=0)
            maxs = np.max([bounds[1] for bounds in child_bounds], axis=0)
            return mins, maxs
        if geo_type in {"clip", "clip_operation", "boolean"} or (
            "geometry_a" in cfg and "geometry_b" in cfg
        ):
            child_bounds = [
                self._cfg_bounds_3d(cfg["geometry_a"]),
                self._cfg_bounds_3d(cfg["geometry_b"]),
            ]
            mins = np.min([bounds[0] for bounds in child_bounds], axis=0)
            maxs = np.max([bounds[1] for bounds in child_bounds], axis=0)
            return mins, maxs
        if geo_type == "compound":
            children = cfg.get("geometries", [])
            if not children:
                raise ValueError("Compound geometry requires non-empty geometries.")
            child_bounds = [self._cfg_bounds_3d(child) for child in children]
            mins = np.min([bounds[0] for bounds in child_bounds], axis=0)
            maxs = np.max([bounds[1] for bounds in child_bounds], axis=0)
            return mins, maxs
        raise ValueError(f"Geometry type {geo_type} not supported for snapping.")

    def _dedupe_snapping_points(self, snapping_points):
        deduped = []
        seen = set((None, None, None))
        for point in snapping_points:
            normalized = []
            for axis_i, value in enumerate(point):
                if value is None or not np.isfinite(value):
                    normalized.append(None)
                else:
                    ## borders are not added as snapping point.
                    if (
                        value == self.cell_extend[axis_i][0]
                        or value == self.cell_extend[axis_i][1]
                    ):
                        normalized.append(None)
                    else:
                        normalized.append(round(float(value), 9))
            key = tuple(normalized)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(tuple(normalized))
        return deduped

    def _derived_tidy3d_snapping_points(self, *, include_heat_sources: bool = False):
        cfg_groups = [self.port_cfgs, self.geometry_cfgs, self.design_region_cfgs]
        if include_heat_sources:
            cfg_groups.append(self.heat_source_cfgs)
        points = []
        for cfg_group in cfg_groups:
            for cfg in cfg_group.values():
                try:
                    mins, maxs = self._cfg_bounds_3d(cfg)
                except Exception:
                    continue
                for axis_index in range(self.dim):
                    for value in (mins[axis_index], maxs[axis_index]):
                        if not np.isfinite(value):
                            continue
                        point = [None, None, None]
                        point[axis_index] = float(value)
                        points.append(tuple(point))
        return self._dedupe_snapping_points(points)

    def _merge_tidy3d_snapping_points(
        self, base_cfg, *, include_heat_sources: bool = False
    ):
        if not TD_SUPPORTED:
            return base_cfg
        auto_points = self._derived_tidy3d_snapping_points(
            include_heat_sources=include_heat_sources
        )
        if not auto_points:
            return base_cfg
        if base_cfg is None:
            return {"snapping_points": auto_points}
        if isinstance(base_cfg, td.GridSpec):
            existing = list(getattr(base_cfg, "snapping_points", ()) or ())
            return base_cfg.updated_copy(
                snapping_points=self._dedupe_snapping_points(existing + auto_points)
            )
        if not isinstance(base_cfg, dict):
            raise TypeError(
                f"Unsupported tidy3d grid_spec config type: {type(base_cfg)!r}"
            )
        merged_cfg = dict(base_cfg)
        existing = list(merged_cfg.get("snapping_points", []) or [])
        merged_cfg["snapping_points"] = self._dedupe_snapping_points(
            existing + auto_points
        )
        return merged_cfg

    def _tidy3d_custom_grid_spec_from_coords(self, coords, *, base_cfg=None):
        if not TD_SUPPORTED:
            raise ImportError("tidy3d is required for tidy3d raster grid construction.")

        coords = tuple(np.asarray(axis, dtype=np.float64) for axis in coords)
        if len(coords) != self.dim:
            raise ValueError(f"Expected {self.dim} coordinate axes, got {len(coords)}.")

        wavelength = 1.55
        extra_cfg = {}
        if isinstance(base_cfg, td.GridSpec):
            wavelength = float(getattr(base_cfg, "wavelength", wavelength))
        elif isinstance(base_cfg, dict):
            wavelength = float(base_cfg.get("wavelength", wavelength))
            for key in (
                "override_structures",
                "snapping_points",
                "layer_refinement_specs",
            ):
                if key in base_cfg:
                    extra_cfg[key] = base_cfg[key]

        grid_spec_cfg = {"wavelength": wavelength, **extra_cfg}
        for axis_name, axis in zip(self.axes, coords):
            boundaries = _centers_to_boundaries(axis)
            grid_spec_cfg[f"grid_{axis_name}"] = {
                "type": "custom_boundaries",
                "coords": boundaries.astype(float).tolist(),
            }
        return grid_spec_cfg

    def _heat_tidy3d_grid_spec_cfg(self):
        heat_cfg = self._heat_sim_cfg()
        base_cfg = heat_cfg.get("grid_spec", heat_cfg.get("tidy3d_grid_spec"))
        base_cfg = self._merge_tidy3d_snapping_points(
            base_cfg,
            include_heat_sources=True,
        )
        return self._merge_heat_tidy3d_override_structures(base_cfg)

    def _heat_mesh_override_cfg(self):
        heat_cfg = self._heat_sim_cfg()
        override_cfg = heat_cfg.get("mesh_override")
        if override_cfg is None:
            override_cfg = heat_cfg.get("mesh_overrides")
        if override_cfg is None:
            override_cfg = heat_cfg.get("heater_mesh_override")
        return override_cfg if isinstance(override_cfg, dict) else {}

    def _heat_source_mesh_override_cfg(self, cfg: dict):
        for key in (
            "mesh_override",
            "mesh_override_cfg",
            "thermal_mesh_override",
            "thermal_mesh_override_cfg",
        ):
            value = cfg.get(key)
            if isinstance(value, dict):
                return value
        return {}

    def _heat_source_override_dl(self, cfg: dict):
        for key in (
            "mesh_override_dl",
            "thermal_mesh_override_dl",
            "override_dl",
            "dl",
        ):
            value = cfg.get(key)
            if value is not None:
                return value
        return None

    def _build_heat_source_mesh_override_structures(self):
        if not TD_SUPPORTED:
            return []
        if not self.heat_source_cfgs:
            return []

        global_cfg = self._heat_mesh_override_cfg()
        if global_cfg.get("enabled", True) is False:
            return []

        default_dl = _coalesce(
            global_cfg.get("dl"),
            global_cfg.get("spacing"),
            global_cfg.get("grid_step"),
        )
        if default_dl is None:
            default_dl = self._heat_mesh_spacing()
        default_dl = _axis_values(default_dl, self.dim, default=self.grid_step)
        default_enforce = bool(global_cfg.get("enforce", True))
        default_shadow = bool(global_cfg.get("shadow", default_enforce))
        default_drop_outside = bool(global_cfg.get("drop_outside_sim", True))
        default_enabled = bool(global_cfg.get("from_heat_sources", True))

        if not default_enabled:
            return []

        override_structures = []
        for _, cfg in self.heat_source_cfgs.items():
            source_override_cfg = self._heat_source_mesh_override_cfg(cfg)
            if source_override_cfg.get("enabled", True) is False:
                continue

            dl_value = self._heat_source_override_dl(source_override_cfg)
            if dl_value is None:
                dl_value = self._heat_source_override_dl(cfg)
            if dl_value is None:
                dl_value = default_dl
            dl = tuple(
                None if value is None else float(value)
                for value in _axis_values(dl_value, self.dim, default=self.grid_step)
            )

            enforce = bool(
                source_override_cfg.get(
                    "enforce", cfg.get("mesh_override_enforce", default_enforce)
                )
            )
            shadow = bool(
                source_override_cfg.get(
                    "shadow", cfg.get("mesh_override_shadow", default_shadow)
                )
            )
            drop_outside_sim = bool(
                source_override_cfg.get(
                    "drop_outside_sim",
                    cfg.get("mesh_override_drop_outside_sim", default_drop_outside),
                )
            )
            if enforce and not shadow:
                shadow = True

            geometry_cfg = copy.deepcopy(cfg)
            geometry_cfg.pop("mesh_override", None)
            geometry_cfg.pop("mesh_override_cfg", None)
            geometry_cfg.pop("thermal_mesh_override", None)
            geometry_cfg.pop("thermal_mesh_override_cfg", None)
            geometry_cfg.pop("mesh_override_dl", None)
            geometry_cfg.pop("thermal_mesh_override_dl", None)
            geometry_cfg.pop("mesh_override_enforce", None)
            geometry_cfg.pop("mesh_override_shadow", None)
            geometry_cfg.pop("mesh_override_drop_outside_sim", None)
            geometry = self._tidy3d_geometry_from_cfg(geometry_cfg)
            override_structures.append(
                td.MeshOverrideStructure(
                    geometry=geometry,
                    dl=_as_3(dl, fill=None),
                    enforce=enforce,
                    shadow=shadow,
                    drop_outside_sim=drop_outside_sim,
                )
            )
        return override_structures

    def _merge_heat_tidy3d_override_structures(self, base_cfg):
        if not TD_SUPPORTED:
            return base_cfg

        auto_overrides = self._build_heat_source_mesh_override_structures()
        if not auto_overrides:
            return base_cfg

        if base_cfg is None:
            return {"override_structures": auto_overrides}
        if isinstance(base_cfg, td.GridSpec):
            return base_cfg.updated_copy(
                override_structures=[
                    *(tuple(getattr(base_cfg, "override_structures", ()) or ())),
                    *auto_overrides,
                ]
            )
        if not isinstance(base_cfg, dict):
            raise TypeError(
                f"Unsupported tidy3d thermal grid_spec config type: {type(base_cfg)!r}"
            )

        merged_cfg = dict(base_cfg)
        merged_cfg["override_structures"] = [
            *(list(merged_cfg.get("override_structures", []) or [])),
            *auto_overrides,
        ]
        return merged_cfg

    def _heat_mesh_pml(self):
        return _axis_values(
            self._heat_sim_cfg().get("PML", self.sim_cfg["PML"]), self.dim
        )

    def _heat_mesh_cell_size(self):
        cell_size = self._heat_sim_cfg().get("cell_size")
        if cell_size is None or cell_size == "None":
            return tuple(self.cell_size)
        return tuple(_axis_values(cell_size, self.dim))

    def _heat_padding_default(self):
        return self._heat_sim_cfg().get("padding")

    def _heat_padding_spec(self):
        padding = self._heat_padding_default()
        if padding in (None, 0, 0.0, (), [], {}):
            return tuple([0.0] * (2 * self.dim)), False

        use_cells = False
        padding_values = padding
        if isinstance(padding, dict):
            if padding.get("cells") is not None:
                padding_values = padding["cells"]
                use_cells = True
            else:
                padding_values = _coalesce(
                    padding.get("distance"),
                    padding.get("size"),
                    padding.get("physical"),
                    padding.get("um"),
                    0,
                )
        elif isinstance(padding, Sequence) and not isinstance(padding, str):
            use_cells = all(float(v).is_integer() for v in padding)
        elif isinstance(padding, (int, np.integer)):
            use_cells = True

        pad_values = _axis_pairs(padding_values, self.dim, default=0.0)
        return tuple(float(v) for v in pad_values), use_cells

    def _heat_padding_mode_default(self):
        padding = self._heat_padding_default()
        if isinstance(padding, dict):
            return str(padding.get("mode", "replicate")).lower()
        return "replicate"

    def _heat_source_padding_mode_default(self):
        padding = self._heat_padding_default()
        if isinstance(padding, dict):
            return str(
                padding.get("source_mode", padding.get("q_mode", "constant"))
            ).lower()
        return "constant"

    def _heat_padding_axis_widths(
        self,
        base_step: float,
        *,
        distance: float | None = None,
        cell_count: int | None = None,
    ):
        base_step = float(base_step)
        if base_step <= 0:
            raise ValueError(
                f"Heat padding requires a positive edge cell size, got {base_step}."
            )

        padding = self._heat_padding_default()
        padding_cfg = padding if isinstance(padding, dict) else {}
        scheme = str(padding_cfg.get("scheme", "constant")).strip().lower()
        scheme = scheme.replace("-", "_")
        if scheme in {"uniform", "replicate"}:
            scheme = "constant"

        growth_rate = max(float(padding_cfg.get("growth_rate", 1.0)), 1.0)
        max_scale_value = padding_cfg.get("max_scale", 1.0)
        max_scale = 1.0 if max_scale_value is None else max(float(max_scale_value), 1.0)
        max_width = base_step * max_scale

        def _next_width(prev_width: float) -> float:
            if scheme != "graded":
                return base_step
            return min(max_width, prev_width * growth_rate)

        if cell_count is not None:
            count = max(0, int(round(cell_count)))
            widths = []
            current_width = base_step
            for _ in range(count):
                widths.append(float(current_width))
                current_width = _next_width(current_width)
            return widths

        target_distance = 0.0 if distance is None else max(0.0, float(distance))
        widths = []
        current_width = base_step
        covered = 0.0
        while covered + 1e-6 < target_distance:
            widths.append(min(target_distance - covered, float(current_width)))
            covered += float(current_width)
            current_width = _next_width(current_width)
        return widths

    def _heat_recorded_grid_info(self, *, map_shape=None):
        grid_info = self._get_raster_grid_info("conductivity_map")
        if grid_info is not None:
            shape = tuple(int(v) for v in grid_info["shape"])
            if map_shape is None or tuple(int(v) for v in map_shape) == shape:
                return grid_info

        if self.thermal_coords is None:
            return None

        coords = tuple(
            np.asarray(axis, dtype=np.float64) for axis in self.thermal_coords
        )
        shape = tuple(int(axis.size) for axis in coords)
        if map_shape is not None and tuple(int(v) for v in map_shape) != shape:
            return None
        return {
            "coords": coords,
            "boundaries": tuple(_centers_to_boundaries(axis) for axis in coords),
            "shape": shape,
            "backend": "recorded",
        }

    def _heat_padding_cells(self, map_shape=None, spacing=None):
        padding = self._heat_padding_default()
        if padding in (None, 0, 0.0, (), [], {}):
            return tuple([0] * (2 * self.dim))

        spacing = _spacing_values(spacing or self._heat_solver_spacing(), self.dim)
        pad_values, use_cells = self._heat_padding_spec()
        if (
            self._heat_mesh_type_default() == "fixed_rectilinear_nonuniform"
            and self.thermal_coords is not None
        ):
            padding_cells = []
            for axis_index, center_axis in enumerate(self.thermal_coords):
                center_axis = np.asarray(center_axis, dtype=np.float64)
                if center_axis.size <= 1:
                    left_step = right_step = float(spacing[axis_index])
                else:
                    boundaries = _centers_to_boundaries(center_axis)
                    left_step = float(boundaries[1] - boundaries[0])
                    right_step = float(boundaries[-1] - boundaries[-2])

                if use_cells:
                    padding_cells.extend(
                        [
                            max(0, int(round(pad_values[2 * axis_index]))),
                            max(0, int(round(pad_values[2 * axis_index + 1]))),
                        ]
                    )
                else:
                    left_widths = self._heat_padding_axis_widths(
                        left_step,
                        distance=pad_values[2 * axis_index],
                    )
                    right_widths = self._heat_padding_axis_widths(
                        right_step,
                        distance=pad_values[2 * axis_index + 1],
                    )
                    padding_cells.extend([len(left_widths), len(right_widths)])
            return tuple(padding_cells)

        if use_cells:
            return tuple(max(0, int(round(v))) for v in pad_values)
        return tuple(
            max(0, int(np.ceil(float(v) / dl)))
            for v, dl in zip(
                pad_values, [value for dl in spacing for value in (dl, dl)]
            )
        )

    def _heat_padding_pad_tuple(self, padding_cells):
        padding_cells = tuple(int(v) for v in padding_cells)
        if self.dim == 2:
            return (
                padding_cells[2],
                padding_cells[3],
                padding_cells[0],
                padding_cells[1],
            )
        if self.dim == 3:
            return (
                padding_cells[4],
                padding_cells[5],
                padding_cells[2],
                padding_cells[3],
                padding_cells[0],
                padding_cells[1],
            )
        raise ValueError(f"Unsupported heat padding dimension {self.dim}")

    def _pad_heat_tensor(self, tensor, padding_cells, *, mode: str, value: float = 0.0):
        padding_cells = tuple(int(v) for v in padding_cells)
        if not any(padding_cells):
            return tensor
        pad_tuple = self._heat_padding_pad_tuple(padding_cells)
        padded = tensor[None, None]
        if mode == "constant":
            padded = F.pad(padded, pad_tuple, mode="constant", value=float(value))
        elif mode in {"replicate", "reflect"}:
            padded = F.pad(padded, pad_tuple, mode=mode)
        else:
            raise ValueError(f"Unsupported heat padding mode: {mode}")
        return padded[0, 0]

    def _crop_heat_tensor(self, tensor, padding_cells):
        padding_cells = tuple(int(v) for v in padding_cells)
        if not any(padding_cells):
            return tensor
        slices = []
        for axis in range(self.dim):
            start = padding_cells[2 * axis]
            stop_pad = padding_cells[2 * axis + 1]
            stop = -stop_pad if stop_pad > 0 else None
            slices.append(slice(start, stop))
        return tensor[tuple(slices)]

    def _heat_grid_shape_from_spacing(self, cell_size=None, spacing=None):
        cell_size = tuple(cell_size or self.cell_size)
        spacing = tuple(spacing or self._heat_mesh_spacing())
        return tuple(
            max(1, int(round(axis_size / dl)))
            for axis_size, dl in zip(cell_size[: self.dim], spacing[: self.dim])
        )

    def _heat_coords_from_shape_spacing(self, shape=None, spacing=None):
        shape = tuple(shape or self._heat_grid_shape_from_spacing())
        spacing = tuple(spacing or self._heat_mesh_spacing())
        return tuple(
            np.linspace(
                -(n - 1) / 2 * dl,
                (n - 1) / 2 * dl,
                n,
            )
            for n, dl in zip(shape, spacing)
        )

    def _heat_dirichlet_bc_default(self):
        return self._heat_sim_cfg().get("dirichlet_bc")

    def _heat_neumann_bc_default(self):
        return self._heat_sim_cfg().get("neumann_bc")

    def _heat_solver_options_default(self):
        return self._heat_sim_cfg().get("solver_options")

    def _heat_adjoint_solver_options_default(self):
        return self._heat_sim_cfg().get("adjoint_solver_options")

    def _heat_include_capacity_default(self):
        return bool(self._heat_sim_cfg().get("include_heat_capacity", False))

    def _heat_requires_temp_grad_default(self):
        return bool(self._heat_sim_cfg().get("requires_temp_grad", True))

    def _heat_build_sources_default(self):
        return bool(self._heat_sim_cfg().get("build_source_maps", False))

    def _heat_combine_sources_default(self):
        return bool(self._heat_sim_cfg().get("combine_sources", True))

    def _heat_solver_spacing(self):
        return tuple(self._heat_mesh_spacing())

    def _heat_solver_grid_shape(self):
        return self._heat_grid_shape_from_spacing(cell_size=self.cell_size)

    def _uniform_heat_remap_grid(self, source_boundaries):
        """Construct a uniform thermal grid with the same physical endpoints.

        Args:
            source_boundaries: Per-axis boundaries of a cell-centered thermal
                material/source raster.

        Returns:
            Tuple ``(boundaries, centers, spacing)`` for a nearly requested-
            spacing uniform grid. Cell counts are rounded from the physical
            lengths, then spacing is adjusted slightly so the source and target
            domains have exactly identical endpoints.
        """
        requested_spacing = self._heat_solver_spacing()
        target_boundaries = []
        target_centers = []
        actual_spacing = []
        for axis, requested_dl in zip(source_boundaries, requested_spacing):
            axis = np.asarray(axis, dtype=np.float64)
            if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0.0):
                raise ValueError(
                    "Thermal raster boundaries must be strictly increasing 1D arrays."
                )
            length = float(axis[-1] - axis[0])
            count = max(1, int(round(length / float(requested_dl))))
            boundaries = np.linspace(axis[0], axis[-1], count + 1)
            target_boundaries.append(boundaries)
            target_centers.append(0.5 * (boundaries[:-1] + boundaries[1:]))
            actual_spacing.append(length / count)
        return (
            tuple(target_boundaries),
            tuple(target_centers),
            tuple(actual_spacing),
        )

    def _remap_nonuniform_heat_inputs_to_uniform(
        self, conductivity, source, source_boundaries
    ):
        """Conservatively map nonuniform thermal inputs to a uniform grid.

        Args:
            conductivity: Positive cell-wise thermal conductivity tensor.
            source: Cell-wise volumetric heat-density tensor.
            source_boundaries: Physical source cell boundaries for every axis.

        Returns:
            Tuple containing remapped conductivity, remapped source, target
            boundaries, target centers, and actual target spacing.

        Notes:
            Source density uses volume-overlap averaging, preserving total heat.
            Conductivity uses the same overlap operator on resistivity followed
            by inversion, retaining the previous resistance-aware coarse-grid
            convention without assuming equal source-cell volumes.
        """
        target_boundaries, target_centers, target_spacing = (
            self._uniform_heat_remap_grid(source_boundaries)
        )
        source_uniform = conservative_rectilinear_remap(
            source, source_boundaries, target_boundaries
        )
        tiny = torch.finfo(conductivity.dtype).tiny
        target_resistivity = conservative_rectilinear_remap(
            1.0 / torch.clamp(conductivity, min=tiny),
            source_boundaries,
            target_boundaries,
        )
        conductivity_uniform = 1.0 / torch.clamp(target_resistivity, min=tiny)
        return (
            conductivity_uniform,
            source_uniform,
            target_boundaries,
            target_centers,
            target_spacing,
        )

    def _heat_fixed_mesh_transfer_neighbors(self):
        mesh_cfg = self._heat_fixed_mesh_cfg()
        return int(mesh_cfg.get("transfer_neighbors", 8))

    def _heat_fixed_mesh_grid_shape(self):
        mesh_cfg = self._heat_fixed_mesh_cfg()
        grid_shape = mesh_cfg.get("grid_shape")
        if grid_shape is None:
            if (
                self._heat_mesh_type_default() == "fixed_rectilinear_nonuniform"
                and self.thermal_grid_shape is not None
            ):
                return tuple(int(v) for v in self.thermal_grid_shape)
            return tuple(int(v) for v in self._heat_solver_grid_shape())
        return tuple(int(v) for v in grid_shape)

    def _normalize_heat_boundary_dict(self, bc):
        if bc is None:
            return None
        if self.dim == 2:
            return {
                k: v for k, v in bc.items() if k in {"xmin", "xmax", "ymin", "ymax"}
            }
        return {
            k: v
            for k, v in bc.items()
            if k in {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
        }

    def _heat_tidy3d_distance_grid_spec(self):
        mesh_cfg = self._heat_fixed_mesh_cfg()
        spec_cfg = mesh_cfg.get("distance_unstructured_grid")
        if TD_SUPPORTED and isinstance(spec_cfg, td.DistanceUnstructuredGrid):
            return spec_cfg
        if spec_cfg is None:
            spec_cfg = {
                key: mesh_cfg[key]
                for key in (
                    "dl_interface",
                    "dl_bulk",
                    "distance_interface",
                    "distance_bulk",
                    "sampling",
                    "uniform_grid_mediums",
                    "non_refined_structures",
                    "mesh_refinements",
                )
                if key in mesh_cfg
            }
        if not spec_cfg:
            return None
        if not TD_SUPPORTED:
            raise ImportError(
                "tidy3d is required to build DistanceUnstructuredGrid metadata."
            )
        return td.DistanceUnstructuredGrid(**spec_cfg)

    def _supported_meshio_cell_types(self):
        if self.dim == 2:
            return {
                "triangle": "TRI3",
                "triangle6": "TRI6",
                "quad": "QUAD4",
                "quad8": "QUAD8",
            }
        return {
            "tetra": "TET4",
            "tetra10": "TET10",
            "hexahedron": "HEX8",
        }

    def _load_meshio_fixed_mesh(self, mesh_path, *, cell_type=None, ele_type=None):
        if not MESHIO_SUPPORTED:
            raise ImportError(
                "meshio is required to load a fixed_distance_unstructured_tidy3d mesh file."
            )
        mesh = meshio.read(mesh_path)
        supported = self._supported_meshio_cell_types()
        chosen_cell_type = cell_type
        if chosen_cell_type is None:
            for candidate in supported:
                if candidate in mesh.cells_dict:
                    chosen_cell_type = candidate
                    break
        if chosen_cell_type is None:
            raise ValueError(
                f"No supported cell block found in mesh file {mesh_path!r}. "
                f"Supported types: {tuple(supported)}"
            )
        if chosen_cell_type not in mesh.cells_dict:
            raise KeyError(
                f"Mesh file {mesh_path!r} does not contain cell block {chosen_cell_type!r}."
            )
        points = np.asarray(mesh.points[:, : self.dim], dtype=np.float64)
        cells = np.asarray(mesh.cells_dict[chosen_cell_type], dtype=np.int64)
        return points, cells, ele_type or supported[chosen_cell_type]

    def _coerce_fixed_mesh_arrays(self, points, cells, *, ele_type=None):
        points = np.asarray(points, dtype=np.float64)
        cells = np.asarray(cells, dtype=np.int64)
        if points.ndim != 2:
            raise ValueError(
                f"Fixed heat mesh points must have shape [num_points, dim], got {points.shape}."
            )
        if cells.ndim != 2:
            raise ValueError(
                f"Fixed heat mesh cells must have shape [num_cells, nodes_per_cell], got {cells.shape}."
            )
        if points.shape[1] < self.dim:
            raise ValueError(
                f"Fixed heat mesh points only provide {points.shape[1]} coordinates for a {self.dim}D device."
            )
        points = points[:, : self.dim]
        if ele_type is None:
            if self.dim == 2:
                infer_map = {3: "TRI3", 4: "QUAD4", 6: "TRI6", 8: "QUAD8"}
            else:
                infer_map = {4: "TET4", 8: "HEX8", 10: "TET10"}
            ele_type = infer_map.get(cells.shape[1])
            if ele_type is None:
                raise ValueError(
                    f"Could not infer finite-element type from cell connectivity width {cells.shape[1]}."
                )
        return points, cells, str(ele_type)

    def _build_rectilinear_nonuniform_points_cells(self, axes):
        axes = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
        if len(axes) != self.dim:
            raise ValueError(
                f"Expected {self.dim} coordinate axes for a {self.dim}D rectilinear mesh, got {len(axes)}."
            )
        for axis_name, axis in zip(self.axes, axes):
            if axis.ndim != 1:
                raise ValueError(
                    f"Axis {axis_name!r} must be 1D, got shape {axis.shape}."
                )
            if axis.size < 2:
                raise ValueError(
                    f"Axis {axis_name!r} must contain at least two coordinates."
                )
            if not np.all(np.diff(axis) > 0):
                raise ValueError(
                    f"Axis {axis_name!r} coordinates must be strictly increasing."
                )

        if self.dim == 2:
            xs, ys = axes
            xv, yv = np.meshgrid(xs, ys, indexing="ij")
            points = np.column_stack([xv.ravel(), yv.ravel()])
            nx, ny = len(xs), len(ys)
            point_ids = np.arange(points.shape[0], dtype=np.int64).reshape(nx, ny)
            c0 = point_ids[:-1, :-1]
            c1 = point_ids[1:, :-1]
            c2 = point_ids[1:, 1:]
            c3 = point_ids[:-1, 1:]
            cells = np.stack((c0, c1, c2, c3), axis=2).reshape(-1, 4)
            ele_type = "QUAD4"
        else:
            xs, ys, zs = axes
            xv, yv, zv = np.meshgrid(xs, ys, zs, indexing="ij")
            points = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
            nx, ny, nz = len(xs), len(ys), len(zs)
            point_ids = np.arange(points.shape[0], dtype=np.int64).reshape(nx, ny, nz)
            c0 = point_ids[:-1, :-1, :-1]
            c1 = point_ids[1:, :-1, :-1]
            c2 = point_ids[1:, 1:, :-1]
            c3 = point_ids[:-1, 1:, :-1]
            c4 = point_ids[:-1, :-1, 1:]
            c5 = point_ids[1:, :-1, 1:]
            c6 = point_ids[1:, 1:, 1:]
            c7 = point_ids[:-1, 1:, 1:]
            cells = np.stack((c0, c1, c2, c3, c4, c5, c6, c7), axis=3).reshape(-1, 8)
            ele_type = "HEX8"
        return points, cells, ele_type

    def _rectilinear_mesh_axes_from_cfg(self, mesh_cfg):
        recorded_grid_info = self._heat_recorded_grid_info()
        if recorded_grid_info is not None:
            return tuple(
                np.asarray(axis, dtype=np.float64)
                for axis in recorded_grid_info["boundaries"]
            )

        axes = mesh_cfg.get("axes")
        if axes is not None:
            axes = tuple(axes)
        elif mesh_cfg.get("coords") is not None:
            coords = mesh_cfg["coords"]
            if isinstance(coords, dict):
                axes = tuple(coords[axis] for axis in self.axes)
            else:
                axes = tuple(coords)
        else:
            axes = []
            for axis_name in self.axes:
                axis_values = _coalesce(
                    mesh_cfg.get(axis_name), mesh_cfg.get(f"{axis_name}s")
                )
                if axis_values is None:
                    if self.thermal_coords is not None:
                        return tuple(
                            _centers_to_boundaries(center_axis)
                            for center_axis in self.thermal_coords
                        )
                    raise ValueError(
                        "Heat mesh_type 'fixed_rectilinear_nonuniform' requires "
                        f"coordinate arrays for every axis; missing {axis_name!r}."
                    )
                axes.append(axis_values)
            axes = tuple(axes)
        return tuple(np.asarray(axis, dtype=np.float64) for axis in axes)

    def _prepare_fixed_distance_unstructured_tidy3d_mesh(self):
        mesh_cfg = self._heat_fixed_mesh_cfg()
        if not isinstance(mesh_cfg, dict):
            raise ValueError(
                "Heat mesh_type 'fixed_distance_unstructured_tidy3d' requires a dict-like heat mesh config."
            )

        mesh_path = _coalesce(
            mesh_cfg.get("mesh_path"),
            mesh_cfg.get("vtu_path"),
            mesh_cfg.get("mesh_file"),
        )
        points = mesh_cfg.get("points")
        cells = mesh_cfg.get("cells")
        ele_type = mesh_cfg.get("ele_type")
        cell_type = _coalesce(
            mesh_cfg.get("cell_type"),
            mesh_cfg.get("meshio_cell_type"),
        )

        if mesh_path is not None:
            points, cells, ele_type = self._load_meshio_fixed_mesh(
                mesh_path,
                cell_type=cell_type,
                ele_type=ele_type,
            )
        elif points is None or cells is None:
            grid_spec = self._heat_tidy3d_distance_grid_spec()
            grid_spec_msg = (
                ""
                if grid_spec is None
                else " A DistanceUnstructuredGrid spec was found, but local mesh generation "
                "is not available from the installed tidy3d API; provide an exported VTU/mesh "
                "file via heat.fixed_mesh.mesh_path (or heat.mesh_path) or explicit points/cells."
            )
            raise ValueError(
                "Heat mesh_type 'fixed_distance_unstructured_tidy3d' requires either "
                "a mesh file path or explicit points/cells." + grid_spec_msg
            )

        points, cells, ele_type = self._coerce_fixed_mesh_arrays(
            points,
            cells,
            ele_type=ele_type,
        )
        normalize_to_origin = bool(mesh_cfg.get("normalize_to_origin", True))
        points_min = points.min(axis=0)
        if normalize_to_origin:
            points = points - points_min
        signature_cfg = {
            key: value
            for key, value in mesh_cfg.items()
            if key not in {"points", "cells"}
        }
        signature = (
            _freeze_structure(signature_cfg),
            tuple(int(v) for v in points.shape),
            tuple(int(v) for v in cells.shape),
            (
                float(np.min(points)),
                float(np.max(points)),
                float(np.sum(points)),
                float(np.sum(cells)),
            ),
            ele_type,
        )
        metadata = {
            "mesh_type": "fixed_distance_unstructured_tidy3d",
            "source": (
                os.fspath(mesh_path)
                if mesh_path is not None
                else "in_memory_points_cells"
            ),
            "original_points_min": tuple(float(v) for v in points_min.tolist()),
            "normalized_to_origin": normalize_to_origin,
            "distance_unstructured_grid": self._heat_tidy3d_distance_grid_spec(),
        }
        return {
            "points": points,
            "cells": cells,
            "ele_type": ele_type,
            "signature": signature,
            "metadata": metadata,
        }

    def _pad_rectilinear_mesh_axes(self, axes, padding_cells):
        padding_cells = tuple(int(v) for v in padding_cells)
        if not any(padding_cells):
            return tuple(np.asarray(axis, dtype=np.float64) for axis in axes)

        pad_values, use_cells = self._heat_padding_spec()

        padded_axes = []
        for axis_index, axis in enumerate(axes):
            axis = np.asarray(axis, dtype=np.float64)
            left_cells = padding_cells[2 * axis_index]
            right_cells = padding_cells[2 * axis_index + 1]
            if axis.size < 2:
                raise ValueError(
                    f"Rectilinear axis {axis_index} must contain at least two boundary coordinates."
                )
            left_step = float(axis[1] - axis[0])
            right_step = float(axis[-1] - axis[-2])

            pieces = []
            if left_cells > 0:
                left_widths = (
                    self._heat_padding_axis_widths(
                        left_step,
                        cell_count=left_cells,
                    )
                    if use_cells
                    else self._heat_padding_axis_widths(
                        left_step,
                        distance=pad_values[2 * axis_index],
                    )
                )
                left_offsets = np.cumsum(np.asarray(left_widths, dtype=np.float64))[
                    ::-1
                ]
                pieces.append(axis[0] - left_offsets)
            pieces.append(axis)
            if right_cells > 0:
                right_widths = (
                    self._heat_padding_axis_widths(
                        right_step,
                        cell_count=right_cells,
                    )
                    if use_cells
                    else self._heat_padding_axis_widths(
                        right_step,
                        distance=pad_values[2 * axis_index + 1],
                    )
                )
                right_offsets = np.cumsum(np.asarray(right_widths, dtype=np.float64))
                pieces.append(axis[-1] + right_offsets)
            padded_axes.append(np.concatenate(pieces))
        return tuple(padded_axes)

    def _prepare_fixed_rectilinear_nonuniform_mesh(self, padding_cells=None):
        mesh_cfg = self._heat_fixed_mesh_cfg()
        if not isinstance(mesh_cfg, dict):
            raise ValueError(
                "Heat mesh_type 'fixed_rectilinear_nonuniform' requires a dict-like heat mesh config."
            )
        padding_cells = tuple(int(v) for v in (padding_cells or [0] * (2 * self.dim)))
        axes = self._rectilinear_mesh_axes_from_cfg(mesh_cfg)
        axes = self._pad_rectilinear_mesh_axes(axes, padding_cells)
        points, cells, ele_type = self._build_rectilinear_nonuniform_points_cells(axes)
        center_axes = tuple(0.5 * (axis[:-1] + axis[1:]) for axis in axes)
        center_mesh = np.meshgrid(*center_axes, indexing="ij")
        grid_points = np.stack(center_mesh, axis=len(center_axes)).reshape(
            -1, len(center_axes)
        )
        normalize_to_origin = bool(mesh_cfg.get("normalize_to_origin", True))
        points_min = points.min(axis=0)
        if normalize_to_origin:
            points = points - points_min
            axes = tuple(axis - axis[0] for axis in axes)
            grid_points = grid_points - points_min
        signature_cfg = {
            key: value
            for key, value in mesh_cfg.items()
            if key
            not in {
                "points",
                "cells",
                "x",
                "xs",
                "y",
                "ys",
                "z",
                "zs",
                "axes",
                "coords",
            }
        }
        signature = (
            _freeze_structure(signature_cfg),
            tuple(tuple(float(v) for v in axis.tolist()) for axis in axes),
            ele_type,
        )
        metadata = {
            "mesh_type": "fixed_rectilinear_nonuniform",
            "normalized_to_origin": normalize_to_origin,
            "original_points_min": tuple(float(v) for v in points_min.tolist()),
            "axis_lengths": tuple(int(axis.size) for axis in axes),
            "cell_counts": tuple(int(axis.size - 1) for axis in axes),
            "padding_cells": padding_cells,
        }
        return {
            "points": points,
            "cells": cells,
            "grid_points": grid_points,
            "grid_shape": tuple(int(axis.size - 1) for axis in axes),
            "direct_cell_mapping": True,
            "ele_type": ele_type,
            "signature": signature,
            "metadata": metadata,
        }

    def _get_or_create_heat_fixed_mesh(self, *, padding_cells=None):
        mesh_type = self._heat_mesh_type_default()
        if mesh_type not in {
            "fixed_distance_unstructured_tidy3d",
            "fixed_rectilinear_nonuniform",
        }:
            return None
        if mesh_type == "fixed_distance_unstructured_tidy3d":
            fixed_mesh = self._prepare_fixed_distance_unstructured_tidy3d_mesh()
        else:
            fixed_mesh = self._prepare_fixed_rectilinear_nonuniform_mesh(
                padding_cells=padding_cells,
            )
        if (
            self._heat_fixed_mesh is None
            or self._heat_fixed_mesh_signature != fixed_mesh["signature"]
        ):
            self._heat_fixed_mesh = fixed_mesh
            self._heat_fixed_mesh_signature = fixed_mesh["signature"]
        return self._heat_fixed_mesh

    def _get_or_create_heat_solver(
        self,
        *,
        map_shape=None,
        padding_cells=None,
        grid_step=None,
        backend=None,
        dirichlet_bc=None,
        neumann_bc=None,
        solver_options=None,
        adjoint_solver_options=None,
    ):
        if not HEAT_SUPPORTED:
            raise ImportError("core.heat is not available in this environment.")
        backend = _coalesce(backend, self._heat_backend_default(), "jax")
        dirichlet_bc = self._normalize_heat_boundary_dict(
            _coalesce(dirichlet_bc, self._heat_dirichlet_bc_default())
        )
        neumann_bc = self._normalize_heat_boundary_dict(
            _coalesce(neumann_bc, self._heat_neumann_bc_default())
        )
        solver_options = _coalesce(solver_options, self._heat_solver_options_default())
        adjoint_solver_options = _coalesce(
            adjoint_solver_options,
            self._heat_adjoint_solver_options_default(),
        )
        solver_grid_step = tuple(
            float(value)
            for value in _spacing_values(
                grid_step or self._heat_solver_spacing(), self.dim
            )
        )
        mesh_type = self._heat_mesh_type_default()
        fixed_mesh = None
        if mesh_type in {
            "fixed_distance_unstructured_tidy3d",
            "fixed_rectilinear_nonuniform",
        }:
            fixed_mesh = self._get_or_create_heat_fixed_mesh(
                padding_cells=padding_cells,
            )
        map_shape = tuple(
            int(v)
            for v in (map_shape or self._heat_solver_grid_shape() or self.grid_shape)
        )
        signature = (
            self.dim,
            mesh_type,
            map_shape,
            solver_grid_step,
            backend,
            tuple(sorted((dirichlet_bc or {}).items())),
            tuple(sorted((neumann_bc or {}).items())),
            (fixed_mesh["signature"] if fixed_mesh is not None else None),
            (
                None
                if fixed_mesh is None or fixed_mesh.get("grid_points") is None
                else (
                    tuple(int(v) for v in fixed_mesh["grid_points"].shape),
                    float(np.sum(fixed_mesh["grid_points"])),
                )
            ),
            (
                self._heat_fixed_mesh_transfer_neighbors()
                if fixed_mesh is not None
                else None
            ),
            _freeze_structure(solver_options) if solver_options else None,
            (
                _freeze_structure(adjoint_solver_options)
                if adjoint_solver_options
                else None
            ),
        )
        if self._heat_solver is None or self._heat_solver_signature != signature:
            solver_kwargs = dict(
                grid_step=solver_grid_step,
                dimension=f"{self.dim}d",
                backend=backend,
                dirichlet_bc=dirichlet_bc,
                neumann_bc=neumann_bc,
                solver_options=solver_options,
                adjoint_solver_options=adjoint_solver_options,
            )
            if mesh_type in {
                "fixed_distance_unstructured_tidy3d",
                "fixed_rectilinear_nonuniform",
            }:
                solver_kwargs.update(
                    mesh_type="fixed",
                    fixed_mesh_points=fixed_mesh["points"],
                    fixed_mesh_cells=fixed_mesh["cells"],
                    fixed_mesh_ele_type=fixed_mesh["ele_type"],
                    fixed_mesh_grid_shape=fixed_mesh.get(
                        "grid_shape", self._heat_fixed_mesh_grid_shape()
                    ),
                    fixed_mesh_grid_points=fixed_mesh.get("grid_points"),
                    fixed_mesh_direct_cell_mapping=bool(
                        fixed_mesh.get("direct_cell_mapping", False)
                    ),
                    fixed_mesh_transfer_neighbors=self._heat_fixed_mesh_transfer_neighbors(),
                )
            self._heat_solver = HeatSolver(**solver_kwargs)
            self._heat_solver_signature = signature
        return self._heat_solver

    def solve_heat(
        self,
        *,
        k_map=None,
        q_map=None,
        backend=None,
        dirichlet_bc=None,
        neumann_bc=None,
        solver_options=None,
        adjoint_solver_options=None,
        return_metadata: bool = False,
    ):
        solve_interval = self._heat_solve_interval()
        if self._heat_solution_cache_interval != solve_interval:
            self._heat_solution_cache = None
            self._heat_solution_query_count = 0
            self._heat_solution_cache_interval = solve_interval
        query_index = self._heat_solution_query_count
        self._heat_solution_query_count += 1
        if (
            solve_interval > 1
            and query_index % solve_interval != 0
            and self._heat_solution_cache is not None
            and self._heat_solution_cache[0] == return_metadata
        ):
            print(f"HEAT solver takes 0 seconds. Use cached solution", flush=True)
            cached_result = self._heat_solution_cache[1]
            if return_metadata:
                return cached_result[0], copy.deepcopy(cached_result[1])
            return cached_result

        backend = _coalesce(backend, self._heat_backend_default(), "jax")
        dirichlet_bc = _coalesce(dirichlet_bc, self._heat_dirichlet_bc_default())
        neumann_bc = _coalesce(neumann_bc, self._heat_neumann_bc_default())
        solver_options = _coalesce(solver_options, self._heat_solver_options_default())
        adjoint_solver_options = _coalesce(
            adjoint_solver_options,
            self._heat_adjoint_solver_options_default(),
        )
        if k_map is None:
            if self.conductivity_map is None:
                self.build_thermal_property_maps()
            k_map = self.conductivity_map
        if q_map is None:
            q_map = self.heat_source_map
            if q_map is None:
                if self.heat_source_cfgs and self._heat_build_sources_default():
                    self.build_heat_source_maps()
                    q_map = self.heat_source_map
            if q_map is None:
                q_map = np.zeros_like(k_map)

        if isinstance(k_map, np.ndarray):
            k_map = torch.as_tensor(k_map, dtype=torch.float32, device=self.device)
        if isinstance(q_map, np.ndarray):
            q_map = torch.as_tensor(q_map, dtype=torch.float32, device=self.device)

        k_shape = tuple(int(v) for v in k_map.shape)
        thermal_grid_info = self._heat_recorded_grid_info(map_shape=k_shape)
        mesh_type = self._heat_mesh_type_default()
        if mesh_type == "fixed_rectilinear_nonuniform" and thermal_grid_info is None:
            raise RuntimeError(
                "solve_heat() requires recorded thermal grid metadata for mesh_type='fixed_rectilinear_nonuniform'. "
                "Build the thermal property maps first so the conductivity grid info is available."
            )

        remap_temperature_coords = None
        solver_temperature_coords = None
        solver_grid_step = self._heat_solver_spacing()
        if mesh_type == "rectangular" and thermal_grid_info is not None:
            source_boundaries = tuple(
                np.asarray(axis, dtype=np.float64)
                for axis in thermal_grid_info["boundaries"]
            )
            remap_temperature_coords = tuple(
                np.asarray(axis, dtype=np.float64)
                for axis in thermal_grid_info["coords"]
            )
            (
                k_map,
                q_map,
                _,
                solver_temperature_coords,
                solver_grid_step,
            ) = self._remap_nonuniform_heat_inputs_to_uniform(
                k_map,
                q_map,
                source_boundaries,
            )
            k_shape = tuple(int(v) for v in k_map.shape)
            input_grid_step = solver_grid_step
        else:
            input_grid_step = self.thermal_grid_spacing or tuple(
                [self.grid_step] * self.dim
            )
        padding_cells = self._heat_padding_cells(
            map_shape=k_shape,
            spacing=input_grid_step,
        )

        if mesh_type in {
            "fixed_distance_unstructured_tidy3d",
            "fixed_rectilinear_nonuniform",
        }:
            fixed_mesh = self._get_or_create_heat_fixed_mesh(
                padding_cells=padding_cells,
            )
            solver_shape = tuple(int(v) for v in fixed_mesh["grid_shape"])
            solver_input_grid_step = None
        else:
            solver_shape = self._heat_grid_shape_from_spacing(
                cell_size=tuple(
                    float(n * dl)
                    for n, dl in zip(
                        (
                            int(k_map.shape[axis])
                            + padding_cells[2 * axis]
                            + padding_cells[2 * axis + 1]
                            for axis in range(self.dim)
                        ),
                        _spacing_values(input_grid_step, self.dim),
                    )
                ),
                spacing=input_grid_step,
            )
            solver_input_grid_step = input_grid_step

        solver = self._get_or_create_heat_solver(
            map_shape=solver_shape,
            padding_cells=padding_cells,
            backend=backend,
            dirichlet_bc=dirichlet_bc,
            neumann_bc=neumann_bc,
            solver_options=solver_options,
            adjoint_solver_options=adjoint_solver_options,
            grid_step=solver_grid_step,
        )
        result = solver(
            k_map,
            q_map,
            dirichlet_bc=dirichlet_bc,
            neumann_bc=neumann_bc,
            input_grid_step=solver_input_grid_step,
            padding_cells=padding_cells,
            padding_mode=self._heat_padding_mode_default(),
            source_padding_mode=self._heat_source_padding_mode_default(),
            return_metadata=return_metadata,
        )
        if remap_temperature_coords is not None:
            if return_metadata:
                temperature_map, metadata = result
            else:
                temperature_map = result
                metadata = None
            temperature_map = resample_rectilinear_tensor(
                temperature_map,
                src_coords=solver_temperature_coords,
                dst_coords=remap_temperature_coords,
                axes=tuple(range(self.dim)),
                mode="cartesian",
            )
            if metadata is not None:
                metadata["conservative_input_remap"] = True
                metadata["solver_unpadded_grid_shape"] = tuple(
                    int(axis.size) for axis in solver_temperature_coords
                )
                metadata["solver_grid_spacing"] = tuple(
                    float(value) for value in solver_grid_step
                )
                metadata["cropped_output_grid_shape"] = tuple(
                    int(v) for v in temperature_map.shape
                )
                result = (temperature_map, metadata)
            else:
                result = temperature_map
        if return_metadata:
            temperature_map, metadata = result
            fixed_mesh = (
                self._get_or_create_heat_fixed_mesh(padding_cells=padding_cells)
                if self._heat_mesh_type_default()
                in {
                    "fixed_distance_unstructured_tidy3d",
                    "fixed_rectilinear_nonuniform",
                }
                else None
            )
            metadata["padding_faces"] = {
                face: int(padding_cells[idx])
                for idx, face in enumerate(
                    ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")[: 2 * self.dim]
                )
            }
            metadata["cropped_output_grid_shape"] = tuple(
                int(v) for v in temperature_map.shape
            )
            if thermal_grid_info is not None:
                metadata["thermal_grid_shape"] = tuple(
                    int(v) for v in thermal_grid_info["shape"]
                )
                metadata["thermal_grid_coords"] = thermal_grid_info["coords"]
                metadata["thermal_grid_boundaries"] = thermal_grid_info["boundaries"]
            if fixed_mesh is not None:
                metadata["fixed_mesh"] = fixed_mesh["metadata"]
            result = (temperature_map, metadata)

        if solve_interval > 1:
            if return_metadata:
                cached_result = (result[0].detach(), copy.deepcopy(result[1]))
            else:
                cached_result = result.detach()
            self._heat_solution_cache = (return_metadata, cached_result)
        return result

    def _tidy3d_structure_from_cfg(
        self,
        cfg,
        permittivity: float,
    ):
        medium = _tidy3d_medium_from_permittivity(
            permittivity,
            freq=C_0 / (self.sim_cfg["wl_cen"] * MICRON_UNIT),
        )
        geometry = self._tidy3d_geometry_from_cfg(cfg)
        return td.Structure(geometry=geometry, medium=medium)

    def _build_tidy3d_design_region_weight(
        self,
        name,
        cfg,
        *,
        coords=None,
        grid_spec_cfg=None,
    ):
        del name
        bg_eps = 1.0
        region_eps = 2.0
        pml = _axis_values(self.sim_cfg["PML"], self.dim)
        cell_size_3d = _as_3(self.cell_size)
        monitor = td.PermittivityMonitor(
            center=(0, 0, 0),
            freqs=[250e12],
            size=cell_size_3d if self.dim == 3 else _as_3(self.cell_size, fill=td.inf),
            name="design_region_indicator",
        )
        if coords is None:
            grid_kwargs = {
                "wavelength": 1.55,
                "grid_x": td.UniformGrid(dl=self.grid_step),
                "grid_y": td.UniformGrid(dl=self.grid_step),
            }
            if self.dim == 3:
                grid_kwargs["grid_z"] = td.UniformGrid(dl=self.grid_step)
            grid_spec = td.GridSpec(**grid_kwargs)
        else:
            coords = tuple(np.asarray(axis, dtype=np.float64) for axis in coords)
            grid_spec = self._tidy3d_grid_spec_for_raster(
                tuple(
                    (
                        float(np.mean(np.diff(axis)))
                        if axis.size > 1
                        else float(self.grid_step)
                    )
                    for axis in coords
                ),
                grid_spec_cfg=self._tidy3d_custom_grid_spec_from_coords(
                    coords,
                    base_cfg=grid_spec_cfg,
                ),
            )
        # center = (0, 0, 0)
        center = list((xmin + xmax) / 2 for xmin, xmax in self.cell_extend)

        if self.dim == 2:
            center.append(0.0)
        sim = td.Simulation(
            center=center,
            size=cell_size_3d,
            grid_spec=grid_spec,
            symmetry=self.sim_cfg.get("symmetry", (0, 0, 0)),
            medium=_tidy3d_medium_from_permittivity(
                bg_eps,
                freq=C_0 / (self.sim_cfg["wl_cen"] * MICRON_UNIT),
            ),
            structures=[self._tidy3d_structure_from_cfg(cfg, region_eps)],
            sources=[],
            subpixel=True,
            monitors=[monitor],
            run_time=1e-15,
            boundary_spec=td.BoundarySpec.pml(
                x=pml[0] > 0,
                y=pml[1] > 0,
                z=self.dim == 3 and pml[2] > 0,
            ),
        )
        weight_data = sim.epsilon(monitor)
        weight = (weight_data.to_numpy().real - bg_eps) / (region_eps - bg_eps)
        if self.dim == 2:
            weight = weight[..., 0]

        target_coords = self.coords if coords is None else coords
        target_shape = tuple(len(axis) for axis in target_coords)
        crop_slices = []
        for axis_name, coord in zip(self.axes, target_coords):
            indicator_coord = np.asarray(
                weight_data.coords[axis_name], dtype=np.float64
            )
            if indicator_coord.shape == coord.shape and np.allclose(
                indicator_coord, coord
            ):
                crop_slices.append(slice(None))
                continue

            target_boundaries = _centers_to_boundaries(coord)
            atol = max(1e-9, 1e-6 * max(1.0, np.max(np.abs(target_boundaries))))
            mask = (indicator_coord >= target_boundaries[0] - atol) & (
                indicator_coord <= target_boundaries[-1] + atol
            )
            if not np.any(mask):
                raise RuntimeError(
                    f"Tidy3D design-region indicator {axis_name}-grid does not overlap target coords"
                )
            index = np.where(mask)[0]
            axis_slice = slice(int(index[0]), int(index[-1]) + 1)
            cropped_coord = indicator_coord[axis_slice]
            if cropped_coord.shape != coord.shape or not np.allclose(
                cropped_coord, coord
            ):
                raise RuntimeError(
                    "Tidy3D design-region indicator grid mismatch after cropping: "
                    f"axis={axis_name}, got={cropped_coord.shape}, expected={coord.shape}"
                )
            crop_slices.append(axis_slice)

        if tuple(weight.shape) != tuple(target_shape):
            weight = weight[tuple(crop_slices)]
        if tuple(weight.shape) != tuple(target_shape):
            raise RuntimeError(
                "Tidy3D design-region indicator grid shape mismatch after cropping: "
                f"{tuple(weight.shape)} vs {tuple(target_shape)}"
            )

        return np.clip(weight, 0.0, 1.0).astype(np.float32)

    def _build_region_masks_from_coords(self, region_cfgs, coords):
        region_masks = {}
        region_mask_weights = {}
        region_axis_weights = {}
        for name, cfg in region_cfgs.items():
            center = np.asarray(_as_3(cfg["center"])[: self.dim], dtype=float)
            size = np.asarray(_as_3(cfg["size"])[: self.dim], dtype=float)
            starts = center - size / 2
            stops = starts + size
            slices, axis_weights = zip(
                *[
                    _pixel_coverage_slice_from_centers(start, stop, centers)
                    for start, stop, centers in zip(starts, stops, coords)
                ]
            )
            region_masks[name] = _slice_region(slices)
            region_axis_weights[name] = axis_weights
            region_mask_weights[name] = _outer_coverage(axis_weights)
        return region_masks, region_mask_weights, region_axis_weights

    def _build_region_masks_from_boundaries(self, region_cfgs, boundaries):
        region_masks = {}
        region_mask_weights = {}
        region_axis_weights = {}
        for name, cfg in region_cfgs.items():
            center = np.asarray(_as_3(cfg["center"])[: self.dim], dtype=float)
            size = np.asarray(_as_3(cfg["size"])[: self.dim], dtype=float)
            starts = center - size / 2
            stops = starts + size
            slices, axis_weights = zip(
                *[
                    _pixel_coverage_slice_from_boundaries(start, stop, axis_boundaries)
                    for start, stop, axis_boundaries in zip(starts, stops, boundaries)
                ]
            )
            region_masks[name] = _slice_region(slices)
            region_axis_weights[name] = axis_weights
            region_mask_weights[name] = _outer_coverage(axis_weights)
        return region_masks, region_mask_weights, region_axis_weights

    def build_design_region_mask(self, design_region_cfgs):
        """
        Build spatial design-region selectors and fractional pixel-coverage weights on the optical grid.

        Args:
            design_region_cfgs: Mapping from region name to geometry configuration. Each region requires ``center`` and ``size`` in the device physical coordinate system; coordinates are interpreted in the first ``self.dim`` axes.

        Returns:
            Dictionary mapping each region name to a ``Slice`` (2-D) or ``Slice3D`` (3-D) covering the intersected optical-grid cells. The selected span is the bounding index range of all cells with nonzero geometric coverage.

        Side Effects:
            Populates ``self.design_region_mask_weights`` with an array of fractional cell-coverage weights and ``self.design_region_axis_weights`` with one 1-D coverage vector per spatial axis. When raster-grid boundary metadata is available, coverage is computed from cell boundaries; otherwise it is computed from grid centers.

        Scientific contract:
            The mask is an indexer, while the weight array contains values in ``[0, 1]`` that represent the fraction of each selected cell covered by the physical design-region rectangle/box. This distinction is important for nonuniform grids and FDTD-X native-grid blending.

        Example:
            For a 2-D region centered at ``(0, 0)`` with ``size=(1.0, 0.5)``, the returned mask selects the minimal rectangular index range intersecting that physical box; ``design_region_mask_weights[name]`` contains the corresponding per-cell area fractions.
        """
        design_region_masks = {}
        self.design_region_mask_weights = {}
        self.design_region_axis_weights = {}
        epsilon_grid_info = self._get_raster_grid_info("epsilon_map")
        boundaries = (
            None if epsilon_grid_info is None else epsilon_grid_info["boundaries"]
        )
        for name, cfg in design_region_cfgs.items():
            if boundaries is not None:
                masks, weights, axis_weights = self._build_region_masks_from_boundaries(
                    {name: cfg},
                    tuple(np.asarray(axis, dtype=np.float64) for axis in boundaries),
                )
            else:
                masks, weights, axis_weights = self._build_region_masks_from_coords(
                    {name: cfg},
                    self.coords,
                )
            design_region_masks[name] = masks[name]
            self.design_region_axis_weights[name] = axis_weights[name]
            self.design_region_mask_weights[name] = weights[name]

        # print(design_region_cfgs, design_region_masks)
        return design_region_masks

    def _optical_grid_cfg(self):
        cfg = self.sim_cfg.get("optical_grid")
        if isinstance(cfg, dict):
            return cfg
        return {}

    def _optical_grid_mode_default(self) -> str:
        cfg = self._optical_grid_cfg()
        mode = cfg.get("mode", cfg.get("type", "uniform"))
        return str(mode).strip().lower().replace("-", "_")

    def _optical_tidy3d_grid_spec_cfg(self):
        cfg = self._optical_grid_cfg()
        grid_spec = cfg.get("grid_spec", cfg.get("tidy3d_grid_spec"))
        if grid_spec is not None:
            return self._merge_tidy3d_snapping_points(grid_spec)
        if any(key in cfg for key in ("grid_x", "grid_y", "grid_z", "x", "y", "z")):
            return self._merge_tidy3d_snapping_points(cfg)
        return self._merge_tidy3d_snapping_points(None)

    def _optical_grid_uses_fdtdx_nonuniform(self) -> bool:
        return self._optical_grid_mode_default() in {
            "fdtdx_rectilinear_nonuniform",
            "rectilinear_nonuniform",
            "nonuniform_rectilinear",
        }

    def _sync_optical_grid_metadata(self):
        native_grid_info = self._get_raster_grid_info("epsilon_map")
        export_grid_info = self._get_raster_grid_info("export_epsilon_map")
        if native_grid_info is None or export_grid_info is None:
            return
        self.optical_grid_metadata = build_rectilinear_grid_metadata(
            coords=tuple(
                np.asarray(axis, dtype=np.float64)
                for axis in native_grid_info["coords"]
            ),
            boundaries=tuple(
                np.asarray(axis, dtype=np.float64)
                for axis in native_grid_info["boundaries"]
            ),
            label="optical_native",
        ).to_dict()
        self.export_grid_metadata = build_rectilinear_grid_metadata(
            coords=tuple(
                np.asarray(axis, dtype=np.float64)
                for axis in export_grid_info["coords"]
            ),
            boundaries=tuple(
                np.asarray(axis, dtype=np.float64)
                for axis in export_grid_info["boundaries"]
            ),
            label="optical_export",
        ).to_dict()
        self.fdtdx_native_grid_metadata = self.optical_grid_metadata
        self.fdtdx_native_cell_weights = tuple(
            grid_plane_weights(self.fdtdx_native_grid_metadata, axis)
            for axis in range(self.dim)
        )
        self.fdtdx_field_grid_metadata = {
            "native": self.fdtdx_native_grid_metadata,
            "export": self.export_grid_metadata,
            "export_policy": "uniform_yee_resample",
            "native_equals_export": tuple(self.fdtdx_native_grid_metadata["shape"])
            == tuple(self.export_grid_metadata["shape"])
            and all(
                np.allclose(
                    np.asarray(
                        self.fdtdx_native_grid_metadata["coords"][axis],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        self.export_grid_metadata["coords"][axis], dtype=np.float64
                    ),
                    atol=1e-12,
                    rtol=1e-9,
                )
                for axis in range(self.dim)
            ),
        }
        design_region_masks = getattr(self, "design_region_masks", None)
        if design_region_masks is None:
            self.fdtdx_native_design_region_masks = {}
            self.fdtdx_native_design_region_mask_weights = {}
            self.fdtdx_native_design_region_axis_weights = {}
            return
        self.fdtdx_native_design_region_masks = dict(design_region_masks)
        self.fdtdx_native_design_region_mask_weights = dict(
            getattr(self, "design_region_mask_weights", {})
        )
        self.fdtdx_native_design_region_axis_weights = dict(
            getattr(self, "design_region_axis_weights", {})
        )

    def _build_fdtdx_native_design_region_metadata(self, native_grid):
        (
            self.fdtdx_native_design_region_masks,
            self.fdtdx_native_design_region_mask_weights,
            self.fdtdx_native_design_region_axis_weights,
        ) = self._build_region_masks_from_boundaries(
            self.design_region_cfgs,
            tuple(
                np.asarray(axis, dtype=np.float64) for axis in native_grid["boundaries"]
            ),
        )

    def _build_fdtdx_native_grid_metadata(self):
        self._sync_optical_grid_metadata()
        native_grid = self.fdtdx_native_grid_metadata
        if native_grid is None:
            raise RuntimeError(
                "Native optical grid metadata is unavailable. build_epsilon_map() must run first."
            )
        self._build_fdtdx_native_design_region_metadata(native_grid)
        return self.fdtdx_field_grid_metadata

    def resample_map_between_coords(
        self,
        values,
        *,
        src_coords,
        dst_coords,
        mode: str = "cartesian",
        interpolation: str = "linear",
    ):
        """Resample a material or field tensor between physical grids.

        Args:
            values: Tensor or array with trailing spatial dimensions.
            src_coords: Source cell-center coordinates in micrometers.
            dst_coords: Destination cell-center coordinates in micrometers.
            mode: Complex interpolation convention.
            interpolation: ``"linear"`` or ``"nearest"`` spatial sampling.

        Returns:
            Resampled PyTorch tensor on the device array placement.
        """
        if not isinstance(values, torch.Tensor):
            values_np = np.asarray(values)
            dtype = torch.complex64 if np.iscomplexobj(values_np) else torch.float32
            values = torch.as_tensor(values_np, dtype=dtype, device=self.device)
        return resample_rectilinear_tensor(
            values,
            src_coords=src_coords,
            dst_coords=dst_coords,
            axes=tuple(range(values.ndim - len(src_coords), values.ndim)),
            mode=mode,
            interpolation=interpolation,
        )

    def _resample_map_to_fdtdx_native(
        self,
        values,
        *,
        map_name: str,
        native_grid: dict,
    ):
        """Resample a material map from its recorded grid to the optical grid.

        Thermal and optical material maps may use different rectilinear grids.
        FDTD-X, however, combines permittivity and electric conductivity cell by
        cell, so an auxiliary map must be represented on the native optical grid
        before it is passed to the solver.

        Args:
            values: Material map tensor or NumPy array. Spatial dimensions must
                be the final dimensions of the value.
            map_name (str): Raster-grid metadata key associated with ``values``.
            native_grid (dict): FDTD-X native optical grid metadata.

        Returns:
            The original map when its coordinates already match ``native_grid``,
            otherwise a differentiable map resampled onto the native grid.

        Raises:
            ValueError: If the map shape differs from the native grid and no
                source coordinates are available for a physically meaningful
                resampling.
        """
        if values is None:
            return None

        target_coords = tuple(
            np.asarray(axis, dtype=np.float64) for axis in native_grid["coords"]
        )
        value_shape = tuple(int(v) for v in values.shape)
        target_shape = tuple(int(axis.size) for axis in target_coords)
        if len(value_shape) < len(target_shape):
            raise ValueError(
                f"{map_name} has shape {value_shape}, but the native FDTD-X "
                f"grid requires {len(target_shape)} spatial dimensions."
            )
        spatial_shape = value_shape[-len(target_shape) :]

        source_grid = self._get_raster_grid_info(map_name)
        source_coords = None if source_grid is None else source_grid.get("coords")
        if source_coords is None and map_name == "electrical_conductivity_map":
            source_coords = self.thermal_coords
        if source_coords is not None:
            source_coords = tuple(
                np.asarray(axis, dtype=np.float64) for axis in source_coords
            )
            if len(source_coords) != len(target_coords):
                raise ValueError(
                    f"{map_name} has {len(source_coords)} source coordinate axes, "
                    f"but the native FDTD-X grid has {len(target_coords)}."
                )
            if spatial_shape == target_shape and all(
                np.allclose(source_axis, target_axis, atol=1e-12, rtol=1e-9)
                for source_axis, target_axis in zip(source_coords, target_coords)
            ):
                return values
            return self.resample_map_between_coords(
                values,
                src_coords=source_coords,
                dst_coords=target_coords,
            )

        if spatial_shape == target_shape:
            return values
        raise ValueError(
            f"Cannot align {map_name} with shape {value_shape} to the native "
            f"FDTD-X shape {target_shape}: source grid coordinates are unavailable."
        )

    def _region_mask_to_index(self, region_mask):
        if isinstance(region_mask, np.ndarray):
            return tuple(region_mask.tolist())
        return tuple(region_mask)

    def _blend_export_map_onto_fdtdx_native_design_regions(
        self,
        export_map,
        *,
        base_export_map,
        field_grid_metadata: dict | None = None,
        region_masks: dict | None = None,
        region_weights: dict | None = None,
    ):
        field_grid_metadata = (
            field_grid_metadata
            if field_grid_metadata is not None
            else self._build_fdtdx_native_grid_metadata()
        )
        if field_grid_metadata.get("native_equals_export", False):
            if isinstance(export_map, torch.Tensor):
                return export_map
            return torch.as_tensor(export_map, device=self.device)

        export_grid = field_grid_metadata["export"]
        native_grid = field_grid_metadata["native"]
        export_map = (
            export_map
            if isinstance(export_map, torch.Tensor)
            else torch.as_tensor(export_map, device=self.device)
        )
        export_dtype = export_map.dtype
        base_export_values = base_export_map
        if not torch.is_complex(export_map) and np.iscomplexobj(base_export_values):
            base_export_values = np.asarray(base_export_values).real
        native_base_map = self.resample_map_between_coords(
            torch.as_tensor(base_export_values, dtype=export_dtype, device=self.device),
            src_coords=export_grid["coords"],
            dst_coords=native_grid["coords"],
        )
        native_export_map = self.resample_map_between_coords(
            export_map,
            src_coords=export_grid["coords"],
            dst_coords=native_grid["coords"],
        )
        native_map = native_base_map.clone()
        region_masks = region_masks or self.fdtdx_native_design_region_masks
        region_weights = region_weights or self.fdtdx_native_design_region_mask_weights
        for region_name, region_mask in region_masks.items():
            region_index = self._region_mask_to_index(region_mask)
            if len(region_index) == 0:
                continue
            weight = region_weights.get(region_name)
            if weight is None:
                native_map[region_index] = native_export_map[region_index]
                continue
            weight_tensor = torch.as_tensor(
                np.array(weight, copy=True),
                dtype=(
                    native_map.real.dtype
                    if torch.is_complex(native_map)
                    else native_map.dtype
                ),
                device=native_map.device,
            )
            native_map[region_index] = native_base_map[region_index] + weight_tensor * (
                native_export_map[region_index] - native_base_map[region_index]
            )
        return native_map

    def build_fdtdx_native_permittivity(
        self,
        permittivity,
        *,
        field_grid_metadata: dict | None = None,
    ):
        return self._blend_export_map_onto_fdtdx_native_design_regions(
            permittivity,
            base_export_map=self.epsilon_map,
            field_grid_metadata=field_grid_metadata,
        )

    def build_active_region_mask(self, active_region_cfgs):
        """
        Build active-region selectors and enforce that every active region is contained within its corresponding design region.

        Args:
            active_region_cfgs: Mapping from active-region name to ``center``/``size`` configuration. Each key must also exist in ``self.design_region_cfgs``.

        Returns:
            Mapping from active-region name to a ``Slice`` or ``Slice3D`` selecting the intersected optical-grid cells.

        Raises:
            AssertionError: If an active-region name is not present in the design regions, or if its physical bounding box extends outside the parent design-region box.

        Notes:
            Active-region bounds are checked in physical coordinates before conversion to grid indices. The returned mask is generated from grid centers and does not carry fractional coverage weights.
        """
        active_region_masks = {}
        for name, cfg in active_region_cfgs.items():
            assert (
                name in self.design_region_masks
            ), f"Active region {name} not found in design region"
            design_region_cfg = self.design_region_cfgs[name]
            center = np.asarray(_as_3(cfg["center"])[: self.dim], dtype=float)
            size = np.asarray(_as_3(cfg["size"])[: self.dim], dtype=float)
            design_center = np.asarray(
                _as_3(design_region_cfg["center"])[: self.dim], dtype=float
            )
            design_size = np.asarray(
                _as_3(design_region_cfg["size"])[: self.dim], dtype=float
            )

            active_min = center - size / 2
            active_max = center + size / 2
            design_min = design_center - design_size / 2
            design_max = design_center + design_size / 2
            if np.any(active_min < design_min) or np.any(active_max > design_max):
                raise AssertionError(
                    f"Active region {name} bounds {active_min.tolist()}..{active_max.tolist()} "
                    f"out of design region {design_min.tolist()}..{design_max.tolist()}"
                )

            active_slices = [
                _pixel_coverage_slice_from_centers(start, stop, centers)[0]
                for start, stop, centers in zip(active_min, active_max, self.coords)
            ]
            active_region_masks[name] = _slice_region(active_slices)

        return active_region_masks

    def apply_active_modulation(self, eps, control_cfgs):
        ## eps_r: permittivity tensor, denormalized
        ## control_cfgs, include control signals for (multiple) active region(s).
        if isinstance(eps, torch.Tensor):
            eps_copy = eps.clone()
        else:
            eps_copy = eps.copy()
        for name, control_cfg in control_cfgs.items():
            design_region_cfg = self.design_region_cfgs[name]
            eps_bg, eps_r = design_region_cfg["eps_bg"], design_region_cfg["eps"]
            active_region_cfg = self.active_region_cfgs[name]
            method = active_region_cfg["method"]
            eps_r_cfg = active_region_cfg["eps_r"]
            eps_bg_cfg = active_region_cfg["eps_bg"]
            mod_fn = modulation_fn_dict[method]

            eps_r_new = mod_fn(eps_r, **eps_r_cfg, **control_cfg)
            eps_bg_new = mod_fn(eps_bg, **eps_bg_cfg, **control_cfg)

            active_region = self.active_region_masks[name]
            eps_region = (eps[active_region] - eps_bg) / (eps_r - eps_bg) * (
                eps_r_new - eps_bg_new
            ) + eps_bg_new
            eps_copy[active_region] = eps_region
        return eps_copy

    def build_port_region(self, port_cfgs, rel_width=2):
        """
        Rasterize all configured optical ports into one boolean device mask.

        Args:
            port_cfgs: Mapping from port name to port geometry configuration. Each port uses ``center``, ``size``, and ``direction``.
            rel_width: Transverse width multiplier. The propagation/normal axis uses a multiplier of ``1`` while transverse axes use ``rel_width``.

        Returns:
            Boolean NumPy array with shape ``self.grid_shape``. ``True`` marks cells belonging to at least one port-region mask; the individual port masks are combined with logical OR.

        Notes:
            The returned mask is based on the optical grid coordinates ``self.meshes`` and is intended to exclude port areas from radiation monitoring rather than to represent a physical geometry boundary.
        """
        port_regions = []
        coord_tensors = [torch.tensor(mesh, device=self.device) for mesh in self.meshes]
        for name, cfg in port_cfgs.items():
            center = torch.tensor(_as_3(cfg["center"])[: self.dim], device=self.device)
            size = torch.tensor(_as_3(cfg["size"])[: self.dim], device=self.device)
            normal_axis = _axis_index(cfg["direction"])
            width_scale = torch.full((self.dim,), float(rel_width), device=self.device)
            width_scale[normal_axis] = 1.0
            region = torch.ones(self.grid_shape, dtype=torch.bool, device=self.device)
            for axis, coords in enumerate(coord_tensors):
                region &= (
                    torch.abs(coords - center[axis])
                    < size[axis] / 2 * width_scale[axis]
                )
            port_regions.append(region)
        if not port_regions:
            return np.zeros(self.grid_shape, dtype=np.bool_)
        return torch.stack(port_regions, dim=0).any(dim=0).cpu().numpy()

    def add_monitor_slice(
        self,
        slice_name: str,
        center: Tuple[int, ...],
        size: Tuple[int, ...],
        direction: str | None = None,
    ):
        """
        Register a codimension-1 optical monitor slice and derive native/export-grid indexers from physical coordinates.

        Args:
            slice_name: Unique monitor identifier.
            center: Physical center coordinates in micrometers, in the device coordinate system.
            size: Physical extents in micrometers. Exactly one spatial dimension must have size zero, defining the monitor plane.
            direction: Propagation/normal direction label such as ``"x"``, ``"x+"``, ``"x-"``, ``"y"``, or ``"z"``. When omitted, the zero-size axis determines the normal axis.

        Returns:
            A ``Slice`` (2-D) or ``Slice3D`` (3-D) indexer on the export grid, stored in ``self.port_monitor_slices[slice_name]``.

        Side Effects:
            Stores physical/grid metadata in ``self.port_monitor_slices_info`` and creates separate indexers for the native optical grid, symmetry-reduced native grid, and export grid in ``self.port_monitor_slices_native``, ``self.port_monitor_slices_native_symmetry``, and ``self.port_monitor_slices_export``.

        Grid contract:
            The public ``port_monitor_slices`` entry is export-grid indexed for plotting/visualization. Solver-facing workflows should use the native entry when running on the native optical/FDTD-X grid. In 3-D, native and export slices are constructed independently from physical coordinates because their grid spacings may differ.

        Notes:
            Nonuniform-grid detector apertures cover their physical transverse
            intervals using edges at the requested center. The normal axis
            retains one cell. Explicit native shape and edge constraints are
            applied before symmetry reduction; plotted slices continue to use
            the export coordinate grid.

        Raises:
            AssertionError: If ``size`` does not define exactly one zero-width axis.
            ValueError: If the requested direction is incompatible with the device dimension.

        Example:
            ``add_monitor_slice("in_port_1", center=(-2.0, 0), size=(0, 1.0), direction="x-")`` creates a vertical 2-D port plane at ``x=-2 um`` spanning ``1 um`` in ``y``.
        """
        center = np.asarray(_as_3(center)[: self.dim], dtype=float)
        size = np.asarray(_as_3(size)[: self.dim], dtype=float)
        ## we build monitor slices on export slice
        coords = self.grid_info_dict["export_epsilon_map"]["coords"]
        zero_axes = np.flatnonzero(size == 0)
        assert zero_axes.size == 1, "Only codimension-1 monitor slices are supported"
        if direction is None:
            direction = self.axes[int(zero_axes[0])]
        normal_axis = _axis_index(direction)
        if normal_axis >= self.dim:
            raise ValueError(f"Direction {direction} not supported for {self.dim}D")
        monitor_slice = self._monitor_slice_on_coords(
            center=center,
            size=size,
            direction=direction,
            coords=coords,
        )
        indexers = []
        coord_values = coords
        raw_indexers = tuple(getattr(monitor_slice, axis) for axis in self.axes)
        for indexer in raw_indexers:
            if isinstance(indexer, slice):
                indexers.append(np.arange(indexer.start, indexer.stop))
            else:
                indexers.append(np.array(int(indexer)))

        coords = [
            coord_axis[idx] if np.ndim(idx) > 0 else coord_axis[int(idx)]
            for coord_axis, idx in zip(coord_values, indexers)
        ]
        self.port_monitor_slices[slice_name] = monitor_slice
        info = dict(center=center.tolist(), size=size.tolist(), direction=direction)
        info.update({f"{axis}s": coord for axis, coord in zip(self.axes, coords)})
        self.port_monitor_slices_info[slice_name] = info

        if self.dim == 3:
            ##### for 3D uniform/nonuniform grids, we always need to build monitor slice on optical native grid
            ##### because adjoint group need to check the slices on native grid
            objects = []
            constraints = []
            volume = fdtdx.SimulationVolume(
                partial_grid_shape=self.epsilon_map.shape,
                material=fdtdx.Material(permittivity=1.0),  # Dummy material
            )
            objects.append(volume)

            native_grid = self.grid_info_dict["epsilon_map"]
            boundaries = tuple(axis * MICRON_UNIT for axis in native_grid["boundaries"])

            edge_arrays = tuple(
                jnp.asarray(axis, dtype=jnp.float32) for axis in boundaries
            )
            grid = fdtdx.RectilinearGrid.custom(
                x_edges=edge_arrays[0],
                y_edges=edge_arrays[1],
                z_edges=edge_arrays[2],
            )

            config = fdtdx.SimulationConfig(
                grid=grid,
                time=1e-15,  # this is the max simulation time, can set to a larger number if needed, but this is already very large
                dtype=jnp.float32,
                courant_factor=0.99,
                symmetry=self.sim_cfg.get("symmetry", (0, 0, 0)),
            )
            key = jax.random.PRNGKey(0)

            source_slice_size = size
            partial_real_shape = [s * MICRON_UNIT for s in source_slice_size]
            partial_real_position = [c * MICRON_UNIT for c in center]
            direction_axis = "xyz".index(direction[0])
            partial_grid_shape = [None, None, None]
            partial_grid_shape[direction_axis] = 1
            partial_real_shape[direction_axis] = None

            wave = fdtdx.WaveCharacter(wavelength=1.55 * MICRON_UNIT)
            pulse = fdtdx.GaussianPulseProfile(
                center_wave=wave,
                spectral_width=fdtdx.WaveCharacter(wavelength=1.55 * MICRON_UNIT * 10),
            )

            # source = fdtdx.ModePlaneSource(
            #     name="dummy_source",
            #     partial_grid_shape=partial_grid_shape,
            #     partial_real_shape=partial_real_shape,
            #     partial_real_position=partial_real_position,
            #     wave_character=wave,
            #     direction=direction[-1],
            #     mode_index=0,
            #     temporal_profile=pulse,
            #     static_amplitude_factor=1,
            #     filter_pol="te",
            # )
            source = fdtdx.ModeOverlapDetector(
                name="dummy_source",
                partial_grid_shape=partial_grid_shape,
                partial_real_shape=partial_real_shape,
                partial_real_position=partial_real_position,
                wave_characters=(wave,),
                direction=direction[-1],
                mode_index=0,
                scaling_mode="pulse",
                filter_pol="te",
            )

            export_source = fdtdx.ModeOverlapDetector(
                name="dummy_source",
                partial_grid_shape=partial_grid_shape,
                partial_real_shape=partial_real_shape,
                partial_real_position=partial_real_position,
                wave_characters=(wave,),
                direction=direction[-1],
                mode_index=0,
                scaling_mode="pulse",
                filter_pol="te",
            )
            objects.append(source)

            if config.has_nonuniform_grid:
                from core.fdtd.fdtd import detector_native_placement

                source, detector_constraints = detector_native_placement(
                    source, boundaries, grid
                )
                objects[-1] = source
                constraints.extend(detector_constraints)

            objects, arrays, params, config, info = fdtdx.place_objects(
                object_list=objects,
                config=config,
                constraints=constraints,
                key=key,
                apply_objects=False,
            )
            if config.has_symmetry:
                # print(objects[source.name].grid_slice)
                ## we expand grid_slice only if the grid_slice is symmetric to the symmetry axis
                if self._check_object_on_symmetry_axis(
                    input_slice_name=slice_name,
                    symmetry=self.sim_cfg.get("symmetry", (0, 0, 0)),
                ):
                    native_grid_indices = fdtdx.unfold_grid_slice(
                        objects[source.name], config, info["symmetry_mid_abs"]
                    )
                else:
                    # if the monitor is on the top half of the symmetry axis, we just use it. do not mirror it.
                    ## simply because our slice has to be a continuous slice(start, stop), cannot have gap..
                    native_grid_indices = tuple(
                        slice(s.start + offset, s.stop + offset)
                        for s, offset in zip(
                            objects[source.name].grid_slice, info["symmetry_mid_abs"]
                        )
                    )
                # print(native_grid_indices)
                ## we record the reduced grid_slice for adjoint group check
                self.port_monitor_slices_native_symmetry[slice_name] = objects[
                    source.name
                ].grid_slice
            else:
                native_grid_indices = objects[source.name].grid_slice

            ##### for 3D uniform/nonuniform grids, we always need to build monitor slice on optical export grid
            ##### because plot_eps_fields need export grid monitor slices.
            objects = []
            constraints = []
            volume = fdtdx.SimulationVolume(
                partial_grid_shape=self.grid_info_dict["export_epsilon_map"]["shape"],
                material=fdtdx.Material(permittivity=1.0),  # Dummy material
            )
            objects.append(volume)

            export_grid = self.grid_info_dict["export_epsilon_map"]
            boundaries = tuple(axis * MICRON_UNIT for axis in export_grid["boundaries"])

            edge_arrays = tuple(
                jnp.asarray(axis, dtype=jnp.float32) for axis in boundaries
            )
            grid = fdtdx.RectilinearGrid.custom(
                x_edges=edge_arrays[0],
                y_edges=edge_arrays[1],
                z_edges=edge_arrays[2],
            )

            config = fdtdx.SimulationConfig(
                grid=grid,
                time=1e-15,  # this is the max simulation time, can set to a larger number if needed, but this is already very large
                dtype=jnp.float32,
                courant_factor=0.99,
                # symmetry=self.sim_cfg.get("symmetry", (0, 0, 0)),
            )

            objects.append(export_source)

            if config.has_nonuniform_grid:
                from core.fdtd.fdtd import detector_native_placement

                export_source, detector_constraints = detector_native_placement(
                    export_source, boundaries, grid
                )
                objects[-1] = export_source
                constraints.extend(detector_constraints)

            objects, arrays, params, config, info = fdtdx.place_objects(
                object_list=objects,
                config=config,
                constraints=constraints,
                key=key,
                apply_objects=False,
            )

            if config.has_symmetry:
                if self._check_object_on_symmetry_axis(
                    input_slice_name=slice_name,
                    symmetry=self.sim_cfg.get("symmetry", (0, 0, 0)),
                ):
                    export_grid_indices = fdtdx.unfold_grid_slice(
                        objects[export_source.name], config, info["symmetry_mid_abs"]
                    )
                else:
                    export_grid_indices = tuple(
                        slice(s.start + offset, s.stop + offset)
                        for s, offset in zip(
                            objects[export_source.name].grid_slice,
                            info["symmetry_mid_abs"],
                        )
                    )
            else:
                export_grid_indices = objects[export_source.name].grid_slice

        else:
            native_grid = self.grid_info_dict.get("epsilon_map")
            export_grid = self.grid_info_dict.get("export_epsilon_map")
            if native_grid is None or export_grid is None:
                native_grid_indices = monitor_slice
                export_grid_indices = monitor_slice
            else:
                native_grid_indices = self._monitor_slice_on_coords(
                    center=center,
                    size=size,
                    direction=direction,
                    coords=native_grid["coords"],
                )
                export_grid_indices = self._monitor_slice_on_coords(
                    center=center,
                    size=size,
                    direction=direction,
                    coords=export_grid["coords"],
                )

        self.port_monitor_slices_native[slice_name] = native_grid_indices
        self.port_monitor_slices_export[slice_name] = export_grid_indices
        self.port_monitor_slices[slice_name] = monitor_slice

        return monitor_slice

    def _monitor_slice_on_coords(
        self,
        *,
        center,
        size,
        direction: str,
        coords,
    ):
        """
        Construct a ``Slice``/``Slice3D`` indexer for a physical monitor definition on a specified coordinate grid.

        Args:
            center: Physical monitor center in micrometers.
            size: Physical monitor extent in micrometers; the coordinate along the normal axis is ignored for width selection and becomes a single nearest grid index.
            direction: Monitor normal/propagation direction label such as ``"x+"`` or ``"y-"``.
            coords: Tuple of 1-D coordinate arrays, one per spatial axis.

        Returns:
            A ``Slice`` or ``Slice3D`` whose normal axis is a single integer index nearest to ``center[normal_axis]`` and whose transverse axes are contiguous ``slice`` objects spanning the physical interval ``center +/- size/2``.

        Notes:
            The transverse span uses pixel-center coverage selection. The returned indexer is therefore grid-specific; it must be rebuilt when moving between native, export, or auxiliary two-port grids.
        """
        center = np.asarray(_as_3(center)[: self.dim], dtype=float)
        size = np.asarray(_as_3(size)[: self.dim], dtype=float)
        normal_axis = _axis_index(direction)
        coord_values = tuple(np.asarray(axis, dtype=np.float64) for axis in coords)
        slices = []
        for axis, (coord_axis, axis_center, width) in enumerate(
            zip(coord_values, center, size)
        ):
            if axis == normal_axis:
                slices.append(int(np.argmin(np.abs(coord_axis - axis_center))))
            else:
                start = float(axis_center) - float(width) / 2.0
                stop = float(axis_center) + float(width) / 2.0
                axis_slice, _ = _pixel_coverage_slice_from_centers(
                    start, stop, coord_axis
                )
                slices.append(axis_slice)
        return _slice_region(slices)

    def build_port_monitor_slice(
        self,
        port_name: str = "in_port_1",
        slice_name: str = "in_port_1",
        rel_loc=0.2,
        rel_width=2,
        rel_height=None,
        direction: str = None,
    ):
        """
        Create a monitor slice from a configured port using relative longitudinal location and transverse width.

        Args:
            port_name: Key into ``self.port_cfgs``.
            slice_name: Name stored in the monitor dictionaries.
            rel_loc: Relative position along the port's propagation axis, measured from the low side of the port box.
            rel_width: Transverse size multiplier. ``float("inf")`` expands the transverse monitor to the cell extent.
            rel_height: Optional third-axis multiplier for 3-D ports; defaults to ``rel_width`` and is clamped to at least ``1``.
            direction: Optional override of the port direction. If omitted, uses ``port_cfg["direction"]``.

        Returns:
            The ``Slice``/``Slice3D`` returned by ``add_monitor_slice``.

        Notes:
            The monitor is placed at ``center_normal - size_normal/2 + rel_loc*size_normal`` along the propagation axis, with that axis forced to zero width. In 3-D, ``rel_height`` controls the dimension orthogonal to the port plane.
        """
        port_cfg = self.port_cfgs[port_name]
        direction = port_cfg["direction"] if direction is None else direction
        center = np.asarray(_as_3(port_cfg["center"])[: self.dim], dtype=float)
        size = np.asarray(_as_3(port_cfg["size"])[: self.dim], dtype=float)
        normal_axis = _axis_index(direction)

        if rel_width == float("inf"):
            transverse = [axis for axis in range(self.dim) if axis != normal_axis]
            rel_width = min(self.cell_size[axis] / size[axis] for axis in transverse)

        if self.dim == 3:
            if rel_height is None:
                rel_height = float(rel_width)
            rel_height = max(rel_height, 1.0)

        monitor_center = center.copy()
        monitor_center[normal_axis] = (
            center[normal_axis] - size[normal_axis] / 2 + rel_loc * size[normal_axis]
        )
        ## z direction use rel_height
        if self.dim == 3 and normal_axis in (0, 1):
            monitor_size = size * rel_width
            monitor_size[2] = size[2] * rel_height
            monitor_size[normal_axis] = 0
        else:
            ## if point to x or y direction, the z-axis uses rel_height
            monitor_size = size * rel_width
            monitor_size[normal_axis] = 0
        return self.add_monitor_slice(
            slice_name, monitor_center.tolist(), monitor_size.tolist(), direction
        )

    def build_farfield_region(
        self,
        region_name: str = "farfield",
        center: Tuple[float, float] = (3, 0),
        size: Tuple[float, float] = (1, 1),
        direction: str = "x+",
    ):
        ## extend the farfield from range[0] to range[1] um along the direction

        """
        Construct a rectangular 2-D far-field integration region and register it as a monitor slice.

        Args:
            region_name: Name used in ``port_monitor_slices`` and ``port_monitor_slices_info``.
            center: Physical center of the rectangular region in micrometers.
            size: Physical ``(x, y)`` extent in micrometers.
            direction: Propagation direction metadata, for example ``"x+"``.

        Returns:
            A ``Slice`` whose ``x`` and ``y`` index arrays span the requested rectangular region on the uniform optical grid.

        Side Effects:
            Stores the slice and physical coordinate arrays ``xs``/``ys`` in ``self.port_monitor_slices_info[region_name]``.

        Notes:
            The implementation assumes the 2-D uniform grid convention ``x=(index-(Nx-1)/2)*grid_step`` and ``y=(index-(Ny-1)/2)*grid_step`` when constructing physical coordinates.
        """
        region_center = [
            int(round((c + offset / 2) / self.grid_step))
            for c, offset in zip(center, self.cell_size)
        ]
        half_width_x = int(round(size[0] / 2 / self.grid_step))
        half_width_y = int(round(size[1] / 2 / self.grid_step))
        xs = np.arange(region_center[0] - half_width_x, region_center[0] + half_width_x)
        ys = np.arange(
            region_center[1] - half_width_y,
            region_center[1] + half_width_y,
        )

        region = Slice(
            x=xs[:, None],
            y=ys[None, :],
        )

        # center of pixel's physical locations (um)
        xs = (-(self.Nx - 1) / 2 + region.x) * self.grid_step
        ys = (-(self.Ny - 1) / 2 + region.y) * self.grid_step
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        self.port_monitor_slices[region_name] = region
        self.port_monitor_slices_info[region_name] = dict(
            center=center,
            size=size,
            xs=xs,
            ys=ys,
            direction=direction,
        )

        return region

    def build_farfield_region_ext(
        self,
        region_name: str = "farfield",
        direction: str = "x+",
        extension_range: Tuple[float, float] = (3, 6),
    ):
        ## extend the farfield from range[0] to range[1] um along the direction
        """
        Construct an extended 2-D far-field monitor spanning a specified physical distance range along ``x`` or ``y``.

        Args:
            region_name: Monitor identifier.
            direction: Axis along which the extension range is defined; currently ``"x"`` or ``"y"``.
            extension_range: Two physical coordinates in micrometers defining the near/far limits along the selected direction.

        Returns:
            A ``Slice`` covering the requested extension and the full transverse simulation span.

        Raises:
            ValueError: If ``direction`` is not ``"x"`` or ``"y"``.

        Notes:
            The generated metadata records physical ``xs``/``ys`` coordinates in micrometers and is later used by far-field radiation-monitor construction.
        """
        if direction == "x":
            center = (sum(extension_range) / 2, 0)
            size = (
                extension_range[1] - extension_range[0],
                (self.Ny - 0.5) * self.grid_step,
            )
            region_center = [
                int(round((c + offset / 2) / self.grid_step))
                for c, offset in zip(center, self.cell_size)
            ]
            half_width_x = int(round(size[0] / 2 / self.grid_step))
            half_width_y = int(round(size[1] / 2 / self.grid_step))
            xs = np.arange(
                region_center[0] - half_width_x, region_center[0] + half_width_x
            )
            ys = np.arange(self.Ny)

        elif direction == "y":
            center = (0, sum(extension_range) / 2)
            size = (
                (self.Nx - 0.5) * self.grid_step,
                extension_range[1] - extension_range[0],
            )
            region_center = [
                int(round((c + offset / 2) / self.grid_step))
                for c, offset in zip(center, self.cell_size)
            ]

            half_width_x = int(round(size[0] / 2 / self.grid_step))
            half_width_y = int(round(size[1] / 2 / self.grid_step))
            xs = np.arange(self.Nx)
            ys = np.arange(
                region_center[1] - half_width_y,
                region_center[1] + half_width_y,
            )
        else:
            raise ValueError(f"Direction {direction} not supported")

        region = Slice(
            x=xs[:, None],
            y=ys[None, :],
        )

        # center of pixel's physical locations (um)
        xs = (-(self.Nx - 1) / 2 + region.x) * self.grid_step
        ys = (-(self.Ny - 1) / 2 + region.y) * self.grid_step
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        self.port_monitor_slices[region_name] = region
        self.port_monitor_slices_info[region_name] = dict(
            center=center,
            size=size,
            xs=xs,
            ys=ys,
            direction=direction,
        )

        return region

    def build_near2far_slice(
        self,
        slice_name: str = "nearfield_1",
        center: Tuple[float, float] = (0, 0),
        size: Tuple[float, float] = (0, 1),
        direction="x+",
    ):
        """
        Build and register a near-to-far-field monitor slice and check that it lies in a homogeneous medium.

        Args:
            slice_name: Monitor identifier.
            center: Physical monitor center in micrometers.
            size: Physical monitor extent in micrometers with one zero-width normal axis.
            direction: Propagation/normal direction, for example ``"x+"``.

        Returns:
            The ``Slice``/``Slice3D`` monitor created by ``add_monitor_slice``.

        Notes:
            After construction, the method samples ``self.epsilon_map`` with
            the native-grid slice and reports a warning when more than one
            unique permittivity value is present. The returned slice remains
            export-grid indexed for the public monitor API.
        """
        monitor_slice = self.add_monitor_slice(slice_name, center, size, direction)
        ## need to check the slice of eps is homogeneous medium
        native_slice = self.port_monitor_slices_native[slice_name]
        if isinstance(native_slice, (tuple, list)):
            monitor_index = tuple(native_slice)
        else:
            monitor_index = tuple(
                getattr(native_slice, axis) for axis in self.axes[: self.dim]
            )
        eps_slice = self.epsilon_map[monitor_index]
        if not (np.unique(eps_slice).size == 1):
            print(
                f"Near2far slice {slice_name} is not in a homogeneous medium",
                flush=True,
            )
        return monitor_slice

    def build_radiation_monitor(
        self, monitor_name: str = "rad_slice", distance_to_PML=[0.2, 0.2]
    ):
        """
        Build four side radiation monitors around the non-PML simulation region and exclude cells belonging to optical ports.

        Args:
            monitor_name: Prefix used for the four side monitors: ``_xp``, ``_xm``, ``_yp``, and ``_ym``.
            distance_to_PML: Two physical offsets in micrometers controlling the gap between the radiation monitors and the PML on the horizontal/vertical transverse directions.

        Returns:
            Tuple ``(xp, xm, yp, ym)`` of export-grid ``Slice`` objects after port exclusion.

        Side Effects:
            Creates native and export monitor slices, then removes indices that intersect the boolean port-region mask. The final public ``self.port_monitor_slices`` entries are the export-grid slices.

        Notes:
            The radiation monitors are codimension-1 slices located at the four cell faces inside the PML. The monitor width is reduced by both the PML thickness and ``distance_to_PML``. Port exclusion is recomputed on each recorded grid (native vs export) so the returned slices remain grid-consistent.
        """
        xp_slice_name = monitor_name + "_xp"
        xp_center = (
            self.cell_size[0] / 2 - self.sim_cfg["PML"][0] - distance_to_PML[0],
            0,
        )
        monitor_size_x = [
            0,
            self.cell_size[1] - 2 * distance_to_PML[1] - 2 * self.sim_cfg["PML"][1],
        ]
        radiation_monitor_xp = self.add_monitor_slice(
            xp_slice_name,
            xp_center,
            monitor_size_x,
            "x",
        )
        xm_slice_name = monitor_name + "_xm"
        xm_center = (
            -self.cell_size[0] / 2 + self.sim_cfg["PML"][0] + distance_to_PML[0],
            0,
        )
        radiation_monitor_xm = self.add_monitor_slice(
            xm_slice_name,
            xm_center,
            monitor_size_x,
            "x",
        )
        yp_slice_name = monitor_name + "_yp"
        yp_center = (
            0,
            self.cell_size[1] / 2 - self.sim_cfg["PML"][1] - distance_to_PML[1],
        )
        monitor_size_y = [
            self.cell_size[0] - 2 * distance_to_PML[0] - 2 * self.sim_cfg["PML"][0],
            0,
        ]
        radiation_monitor_yp = self.add_monitor_slice(
            yp_slice_name,
            yp_center,
            monitor_size_y,
            "y",
        )
        ym_slice_name = monitor_name + "_ym"
        ym_center = (
            0,
            -self.cell_size[1] / 2 + self.sim_cfg["PML"][1] + distance_to_PML[1],
        )
        radiation_monitor_ym = self.add_monitor_slice(
            ym_slice_name,
            ym_center,
            monitor_size_y,
            "y",
        )

        def port_mask_for_grid(grid_name):
            grid_info = self.grid_info_dict.get(grid_name)
            if grid_info is None:
                return self.ports_regions
            shape = tuple(len(axis) for axis in grid_info["coords"])
            if grid_name == "epsilon_map" and self.ports_regions.shape == shape:
                return self.ports_regions

            meshes = np.meshgrid(*grid_info["coords"], indexing="ij")
            mask = np.zeros(shape, dtype=bool)
            for port_cfg in self.port_cfgs.values():
                center = np.asarray(_as_3(port_cfg["center"])[: self.dim])
                size = np.asarray(_as_3(port_cfg["size"])[: self.dim])
                normal_axis = _axis_index(port_cfg["direction"])
                width_scale = np.full(self.dim, 2.0)
                width_scale[normal_axis] = 1.0
                port_mask = np.ones(shape, dtype=bool)
                for axis in range(self.dim):
                    port_mask &= np.abs(meshes[axis] - center[axis]) < (
                        size[axis] / 2 * width_scale[axis]
                    )
                mask |= port_mask
            return mask

        def index_array(indexer, axis_length):
            if isinstance(indexer, slice):
                return np.arange(*indexer.indices(axis_length), dtype=int)
            return np.asarray(indexer, dtype=int).reshape(-1)

        def exclude_ports(slice_obj, port_mask):
            x_indices = index_array(slice_obj.x, port_mask.shape[0])
            y_indices = index_array(slice_obj.y, port_mask.shape[1])
            if x_indices.size == 1:
                keep = ~port_mask[x_indices[0], y_indices]
                return Slice(x=x_indices, y=y_indices[keep])
            if y_indices.size == 1:
                keep = ~port_mask[x_indices, y_indices[0]]
                return Slice(x=x_indices[keep], y=y_indices)
            raise ValueError("Radiation monitor must be a codimension-1 slice")

        monitor_names = (
            xp_slice_name,
            xm_slice_name,
            yp_slice_name,
            ym_slice_name,
        )
        native_mask = port_mask_for_grid("epsilon_map")
        export_mask = port_mask_for_grid("export_epsilon_map")
        for slice_name in monitor_names:
            native_slice = exclude_ports(
                self.port_monitor_slices_native[slice_name], native_mask
            )
            export_slice = exclude_ports(
                self.port_monitor_slices_export[slice_name], export_mask
            )
            self.port_monitor_slices_native[slice_name] = native_slice
            self.port_monitor_slices_export[slice_name] = export_slice
            self.port_monitor_slices[slice_name] = export_slice
        return (
            self.port_monitor_slices_export[xp_slice_name],
            self.port_monitor_slices_export[xm_slice_name],
            self.port_monitor_slices_export[yp_slice_name],
            self.port_monitor_slices_export[ym_slice_name],
        )

    def build_farfield_radiation_monitor(
        self, monitor_name: str = "farfield_rad_monitor"
    ):
        """
        for now, only xp_plus, xp_minus, yp and ym will be initialized
        """
        # self.port_monitor_slices[slice_name] = monitor_slice # index to refer in the np array or torch tensor
        # self.port_monitor_slices_info[slice_name] = dict(
        #     center=center, # coordinates of the center of the slice (um)
        #     size=size,     # size of the slice (um)
        #     xs=xs,         # x coordinates of the slice (um)
        #     ys=ys,         # y coordinates of the slice (um)
        #     direction=direction, # direction of the slice
        # )
        nearfield_vertices_x_coords = []
        nearfield_vertices_y_coords = []
        farfield_vertices_x_coords = []
        farfield_vertices_y_coords = []
        yp_info = {}
        ym_info = {}
        xp_plus_info = {}
        xp_minus_info = {}
        for key in list(self.port_monitor_slices_info.keys()):
            if "nearfield" in key:
                nearfield_vertices_x_coords.append(
                    self.port_monitor_slices_info[key]["xs"]
                )
                nearfield_vertices_y_coords.append(
                    self.port_monitor_slices_info[key]["ys"]
                )
            elif "farfield" in key:
                print(self.port_monitor_slices_info[key])
                farfield_vertices_x_coords.append(
                    self.port_monitor_slices_info[key]["xs"]
                )
                farfield_vertices_y_coords.append(
                    self.port_monitor_slices_info[key]["ys"]
                )

        def find_abs_max(input_list):
            max_abs_value = float("-inf")
            # Traverse the list
            for item in input_list:
                if isinstance(item, (int, float)):  # If it's a float or int
                    max_abs_value = max(max_abs_value, abs(item))
                elif isinstance(item, np.ndarray):  # If it's an array
                    max_abs_value = max(max_abs_value, np.max(np.abs(item)))
            return max_abs_value

        def find_max(input_list):
            max_value = float("-inf")
            # Traverse the list
            for item in input_list:
                if isinstance(item, (int, float)):
                    max_value = max(max_value, item)
                elif isinstance(item, np.ndarray):
                    max_value = max(max_value, np.max(item))
            return max_value

        nearfield_x_max = find_max(nearfield_vertices_x_coords) + 1
        nearfield_y_max_abs = find_abs_max(nearfield_vertices_y_coords)
        farfield_x_max = find_max(farfield_vertices_x_coords)
        farfield_y_max_abs = find_abs_max(farfield_vertices_y_coords)
        yp_info["center"] = [
            (nearfield_x_max + farfield_x_max) / 2,
            max(nearfield_y_max_abs, farfield_y_max_abs) + 1,
        ]
        yp_info["size"] = [farfield_x_max - nearfield_x_max, 0]
        yp_info["direction"] = "y"
        ym_info["center"] = [
            (nearfield_x_max + farfield_x_max) / 2,
            -max(nearfield_y_max_abs, farfield_y_max_abs) - 1,
        ]
        ym_info["size"] = [farfield_x_max - nearfield_x_max, 0]
        ym_info["direction"] = "y"
        yp_info["xs"] = (
            np.arange(
                int(
                    round(
                        (yp_info["center"][0] - yp_info["size"][0] / 2) / self.grid_step
                    )
                ),
                int(
                    round(
                        (yp_info["center"][0] + yp_info["size"][0] / 2) / self.grid_step
                    )
                ),
            )
            * self.grid_step
        )
        yp_info["ys"] = np.float32(yp_info["center"][1])
        ym_info["xs"] = (
            np.arange(
                int(
                    round(
                        (ym_info["center"][0] - ym_info["size"][0] / 2) / self.grid_step
                    )
                ),
                int(
                    round(
                        (ym_info["center"][0] + ym_info["size"][0] / 2) / self.grid_step
                    )
                ),
            )
            * self.grid_step
        )
        ym_info["ys"] = np.float32(ym_info["center"][1])

        xp_plus_info["center"] = [
            farfield_x_max,
            (farfield_y_max_abs + yp_info["center"][1]) / 2,
        ]
        xp_plus_info["size"] = [0, yp_info["center"][1] - farfield_y_max_abs]
        xp_plus_info["direction"] = "x"
        xp_minus_info["center"] = [
            nearfield_x_max,
            -(farfield_y_max_abs + yp_info["center"][1]) / 2,
        ]
        xp_minus_info["size"] = [0, -farfield_y_max_abs - ym_info["center"][1]]
        xp_minus_info["direction"] = "x"
        xp_plus_info["xs"] = np.float32(xp_plus_info["center"][0])
        xp_plus_info["ys"] = (
            np.arange(
                int(
                    round(
                        (xp_plus_info["center"][1] - xp_plus_info["size"][1] / 2)
                        / self.grid_step
                    )
                ),
                int(
                    round(
                        (xp_plus_info["center"][1] + xp_plus_info["size"][1] / 2)
                        / self.grid_step
                    )
                ),
            )
            * self.grid_step
        )
        xp_minus_info["xs"] = np.float32(xp_minus_info["center"][0])
        xp_minus_info["ys"] = (
            np.arange(
                int(
                    round(
                        (xp_minus_info["center"][1] - xp_minus_info["size"][1] / 2)
                        / self.grid_step
                    )
                ),
                int(
                    round(
                        (xp_minus_info["center"][1] + xp_minus_info["size"][1] / 2)
                        / self.grid_step
                    )
                ),
            )
            * self.grid_step
        )

        self.port_monitor_slices_info[monitor_name + "_yp"] = yp_info
        self.port_monitor_slices_info[monitor_name + "_ym"] = ym_info
        self.port_monitor_slices_info[monitor_name + "_xp_plus"] = xp_plus_info
        self.port_monitor_slices_info[monitor_name + "_xp_minus"] = xp_minus_info
        print(yp_info)
        print(ym_info)
        print(xp_plus_info)
        print(xp_minus_info)

    def _check_object_on_symmetry_axis(self, input_slice_name, symmetry=(0, 0, 0)):
        # source_slice_size = self.port_monitor_slices_info[input_slice_name]["size"]
        """
        Check whether a configured source/monitor slice is centered on every active simulation symmetry axis.

        Args:
            input_slice_name: Key into ``self.port_monitor_slices_info``.
            symmetry: Per-axis symmetry flags, conventionally a 3-tuple aligned with ``x/y/z``. Zero means no symmetry constraint on that axis.

        Returns:
            ``True`` when the slice center lies on every axis with nonzero symmetry; otherwise ``False``.

        Notes:
            The check uses the geometric cell center derived from ``self.cell_extend`` and a tolerance of ``1e-3`` in the device physical coordinate units. It is used when deciding whether fdtdx grid slices/fields can be unfolded across symmetry planes.
        """
        source_slice_center = self.port_monitor_slices_info[input_slice_name]["center"]
        cell_extend = self.cell_extend
        if symmetry is not None:
            ## either the source is symmetric along the axis, or the source is not across the symmetry axis
            for axis, sym in enumerate(symmetry):
                if sym != 0:
                    xmin, xmax = cell_extend[axis]
                    cell_center = (xmin + xmax) / 2
                    if np.abs(source_slice_center[axis] - cell_center) > 1e-3:
                        return False
            return True
        else:
            return False

    def _check_object_symmetry(self, input_slice_name, symmetry=(0, 0, 0)):
        """
        Validate that a source slice is compatible with the active FDTD-X symmetry convention.

        Args:
            input_slice_name: Key into ``self.port_monitor_slices_info``.
            symmetry: Per-axis symmetry flags, conventionally ``(sx, sy, sz)``.

        Raises:
            ValueError: If, on a symmetric axis, the source slice lies entirely on the lower half of the cell or crosses the symmetry plane without being centered on it.

        Notes:
            The current FDTD-X convention allows a slice centered on the symmetry plane or entirely on the upper/non-folded side. A slice on the lower side is rejected because the reduced-domain source representation cannot be unfolded consistently.
        """
        source_slice_size = self.port_monitor_slices_info[input_slice_name]["size"]
        source_slice_center = self.port_monitor_slices_info[input_slice_name]["center"]
        cell_extend = self.cell_extend
        if symmetry is not None:
            ## either the source is symmetric along the axis, or the source is not across the symmetry axis
            for axis, sym in enumerate(symmetry):
                if sym != 0:
                    xmin, xmax = cell_extend[axis]
                    slice_min = source_slice_center[axis] - source_slice_size[axis] / 2
                    slice_max = source_slice_center[axis] + source_slice_size[axis] / 2
                    cell_center = (xmin + xmax) / 2
                    if (
                        np.abs(source_slice_center[axis] - cell_center) < 1e-3
                        or slice_min >= cell_center
                    ):  # threshold 1nm
                        pass
                    elif slice_max <= cell_center:
                        raise ValueError(
                            f"Source slice {input_slice_name} is on the lower half of the symmetric domain (axis={axis}), which is not allowed in fdtdx"
                        )
                    else:
                        raise ValueError(
                            f"Source slice {input_slice_name} is not symmetric along axis {axis}, cell_center={cell_center}, slice_min={slice_min}, slice_max={slice_max}"
                        )

    def insert_monitors_fdtd3d(
        self,
        input_slice_name: str,
        eps,
        type,
        monitor_slice: Slice,
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        source_modes: Tuple[int] = ("Ez1",),
        direction: str = "x+",
        field_grid_metadata: dict | None = None,
        slice_info: dict | None = None,
        on_native_grid: bool = False,
        field_components: tuple = _PHASOR_COMPONENTS,
    ):
        """
        Construct FDTD-X mode, field, or flux monitor profiles for a port slice.

        Args:
            input_slice_name: Port/monitor identifier used to resolve physical slice metadata for mode monitors.
            eps: Permittivity map used when solving mode profiles; it may be native- or export-grid data depending on ``on_native_grid``.
            type: Monitor kind. Supported values are ``\"mode\"``, ``\"field\"``,
                ``\"flux\"``, ``\"near2far\"``, ``\"design_region_field\"``,
                and ``\"sparse_field\"``.
            monitor_slice: Slice indexer defining the monitor plane. For mode monitors with ``slice_info``, the slice is rebuilt from physical coordinates on the selected grid.
            wl_cen: Center wavelength in micrometers.
            wl_width: Broadband wavelength span in micrometers.
            n_wl: Number of wavelength samples.
            source_modes: Modal identifiers such as ``\"Ez1\"`` or ``\"Hz1\"``.
            direction: Propagation direction such as ``\"x+\"`` or ``\"y-\"``; the final sign character controls FDTD-X propagation direction.
            field_grid_metadata: Optional native/export grid metadata used for epsilon resampling.
            slice_info: Optional physical center/size/direction metadata used to rebuild the slice.
            on_native_grid: Selects native optical grid mode solving instead of the uniform export grid.
            field_components: Field components recorded by field detectors.
                Sparse phasor mode uses the configured electric-only or
                six-component convention.

        Returns:
            Mapping of detector keys to monitor metadata. Mode entries use
            ``(wl, mode)`` keys and tuples
            ``(detector, H_mode, E_mode, power_scale)``. Dense field-monitor
            entries use ``((wl_cen, wl_width, n_wl), mode)``. Sparse field
            entries add a support identifier as the third key element and a
            native-grid bounds tuple as the fifth value:
            ``(detector, None, None, power_scale, bounds)``. Sparse forward
            supports include every design region, objective port, and enabled
            design-centered visualization plane.

        Notes:
            Mode profiles are phase-normalized before conversion to complex
            Torch tensors. FDTD-X ``Ez*`` modes use ``tm`` filtering and
            ``Hz*`` modes use ``te`` filtering. Native-grid mode profiles use
            recorded rectilinear boundaries, while export-grid mode profiles
            use a uniform grid with ``self.grid_step`` spacing. On nonuniform
            grids, physical modal apertures are converted to native cell bounds
            at their actual location before the mode solve. Unplaced prototypes
            retain the original physical metadata. During sparse
            adjoint simulations, only the design-region detectors are placed;
            objective-port and visualization detectors are forward-only.
            ``MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES=1`` (the default) adds
            thin full-span ``xy``, ``xz``, and ``yz`` slabs through each design
            region center. Identical slabs shared by multiple regions are
            recorded once. If the device has no design regions, one set of
            three slabs is centered at physical ``(0, 0, 0)`` using the native
            rectilinear coordinates. Set the variable to ``0`` before importing
            MAPS to minimize forward detector work when cross-section plots are
            not needed.
        """
        assert type in (
            "mode",
            "field",
            "flux",
            "near2far",
            "design_region_field",
            "sparse_field",
        ), f"Monitor type {type} not supported"
        # fdtdx, _ = _import_fdtdx_runtime()
        monitor_profile = {}
        # monitor_slice = slice
        if type == "mode" and slice_info is not None:
            grid_metadata = (
                self.grid_info_dict.get("epsilon_map")
                if on_native_grid
                else self.export_grid_metadata
            )
            if grid_metadata is None:
                raise ValueError(
                    "Mode monitor coordinate metadata is unavailable for the "
                    f"{'native' if on_native_grid else 'export'} grid."
                )
            monitor_slice = self._monitor_slice_on_coords(
                center=slice_info["center"],
                size=slice_info["size"],
                direction=direction,
                coords=grid_metadata["coords"],
            )
        partial_grid_shape, grid_slice_tuple = convert_to_fdtdx_grid_shape_loc_slice(
            monitor_slice, direction
        )
        wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)

        def _resolve_export_grid_eps():
            export_grid_info = self._get_raster_grid_info("export_epsilon_map")
            if export_grid_info is None:
                if self.export_epsilon_map is not None:
                    return np.asarray(self.export_epsilon_map)
                return np.asarray(self.epsilon_map)
            export_coords = tuple(
                np.asarray(axis, dtype=np.float64)
                for axis in export_grid_info["coords"]
            )
            export_shape = tuple(int(v) for v in export_grid_info["shape"])

            if eps is None:
                if self.export_epsilon_map is not None:
                    return np.asarray(self.export_epsilon_map)
                return np.asarray(self.epsilon_map)

            eps_arr = (
                eps.detach().cpu().numpy()
                if isinstance(eps, torch.Tensor)
                else np.asarray(eps)
            )
            if tuple(eps_arr.shape) == export_shape:
                return eps_arr

            metadata = (
                field_grid_metadata
                if field_grid_metadata is not None
                else self.fdtdx_field_grid_metadata
            )
            if metadata is None:
                metadata = self._build_fdtdx_native_grid_metadata()
            native_grid = None if metadata is None else metadata.get("native")
            export_grid = None if metadata is None else metadata.get("export")
            if (
                native_grid is not None
                and export_grid is not None
                and tuple(int(v) for v in native_grid["shape"]) == tuple(eps_arr.shape)
            ):
                resampled = resample_rectilinear_tensor(
                    (
                        torch.as_tensor(
                            eps_arr, dtype=torch.complex64, device=self.device
                        )
                        if np.iscomplexobj(eps_arr)
                        else torch.as_tensor(
                            eps_arr, dtype=torch.float32, device=self.device
                        )
                    ),
                    src_coords=native_grid["coords"],
                    dst_coords=export_coords,
                )
                return resampled.detach().cpu().numpy()

            raise ValueError(
                "Mode-monitor epsilon must be defined on the uniform export grid or "
                "be resample-able from the native fdtdx grid. "
                f"Got eps shape {tuple(eps_arr.shape)}, export shape {export_shape}."
            )

        if type == "mode":
            if on_native_grid:
                eps = self.epsilon_map
                # grid = self.grid_info_dict["epsilon_map"]["grid"]

                # native = self.fdtdx_field_grid_metadata["native"]
                # boundaries = tuple(axis * MICRON_UNIT for axis in native["boundaries"])
                native_grid = self.grid_info_dict["epsilon_map"]
                boundaries = tuple(
                    axis * MICRON_UNIT for axis in native_grid["boundaries"]
                )

                edge_arrays = tuple(
                    jnp.asarray(axis, dtype=jnp.float32) for axis in boundaries
                )
                grid_shape = self.epsilon_map.shape
                # print("Using nonuniform grid with edges:", edge_arrays)
                grid = fdtdx.RectilinearGrid.custom(
                    x_edges=edge_arrays[0],
                    y_edges=edge_arrays[1],
                    z_edges=edge_arrays[2],
                )
            else:
                eps = _resolve_export_grid_eps()
                grid_shape = tuple(int(v) for v in np.asarray(eps).shape)
                grid = fdtdx.UniformGrid(
                    spacing=self.grid_step * MICRON_UNIT,
                    origin=(0.0, 0.0, 0.0),
                )

            config = fdtdx.SimulationConfig(
                grid=grid,
                time=1e-15,
                dtype=jnp.float32,
                courant_factor=0.99,
                symmetry=self.sim_cfg.get("symmetry", (0, 0, 0)),
            )
            key = jax.random.PRNGKey(0)

            source_slice_size = self.port_monitor_slices_info[input_slice_name]["size"]
            source_slice_center = self.port_monitor_slices_info[input_slice_name][
                "center"
            ]
            partial_real_position = tuple(s * MICRON_UNIT for s in source_slice_center)
            partial_real_shape = [s * MICRON_UNIT for s in source_slice_size]
            direction_axis = "xyz".index(direction[0])
            partial_grid_shape = [None, None, None]
            partial_grid_shape[direction_axis] = 1
            partial_real_shape[direction_axis] = None

            objects = []
            volume = fdtdx.SimulationVolume(
                partial_grid_shape=grid_shape,
                material=fdtdx.Material(permittivity=1.0),  # Dummy material
            )
            objects.append(volume)

            inv_permittivities = 1 / jnp.asarray(eps[np.newaxis, ...]).astype(
                config.dtype
            )
            # print(inv_permittivities.shape, jnp.min(inv_permittivities), jnp.max(inv_permittivities))
            # inv_permeabilities = 1.0

            unplaced_objects = {}
            constraints = []
            for wl in wls:
                for source_mode in source_modes:
                    wave = fdtdx.WaveCharacter(wavelength=float(wl * MICRON_UNIT))
                    name = (
                        f"{input_slice_name}_mode_overlap_detector_{wl}_{source_mode}"
                    )
                    monitor = fdtdx.ModeOverlapDetector(
                        name=name,
                        partial_grid_shape=partial_grid_shape,
                        partial_real_shape=partial_real_shape,
                        partial_real_position=partial_real_position,
                        wave_characters=(wave,),
                        direction=direction[-1],
                        mode_index=int(source_mode[2:]) - 1,
                        scaling_mode="pulse",
                        filter_pol="tm" if source_mode.startswith("Ez") else "te",
                    )
                    unplaced_objects[name] = monitor
                    if config.has_nonuniform_grid:
                        from core.fdtd.fdtd import detector_native_placement

                        monitor, detector_constraints = detector_native_placement(
                            monitor, boundaries, grid
                        )
                        constraints.extend(detector_constraints)
                    objects.append(monitor)

            # constraints.append(constraint)
            objects, arrays, params, config, info = fdtdx.place_objects(
                object_list=objects,
                config=config,
                constraints=constraints,
                key=key,
                override_inv_permittivities=inv_permittivities,
            )
            self._check_object_symmetry(
                input_slice_name,
                symmetry=self.sim_cfg.get("symmetry", (0, 0, 0)),
            )

            # if config.has_symmetry:
            #     symmetry_mid_abs = info["symmetry_mid_abs"]
            #     inv_permittivities = inv_permittivities[
            #         :,
            #         symmetry_mid_abs[0] :,
            #         symmetry_mid_abs[1] :,
            #         symmetry_mid_abs[2] :,
            #     ]
            # arrays = arrays.at["inv_permittivities"].set(inv_permittivities)
            # arrays = arrays.at["inv_permeabilities"].set(inv_permeabilities)

            # arrays, objects, _ = fdtdx.apply_params(
            #     arrays=arrays,
            #     objects=objects,
            #     params=params,
            #     key=key,
            # )
            # print(objects[name]._mode_H.shape)
            from core.fdtd.fdtd import normalize_fields_phase

            for wl in wls:
                for source_mode in source_modes:
                    name = (
                        f"{input_slice_name}_mode_overlap_detector_{wl}_{source_mode}"
                    )
                    if config.has_symmetry and self._check_object_on_symmetry_axis(
                        input_slice_name=input_slice_name,
                        symmetry=self.sim_cfg.get("symmetry", (0, 0, 0)),
                    ):
                        _mode_E = fdtdx.unfold_fields(
                            objects[name]._mode_E[0], self.sim_cfg["symmetry"], "E"
                        )
                        _mode_H = fdtdx.unfold_fields(
                            objects[name]._mode_H[0], self.sim_cfg["symmetry"], "H"
                        )

                        symmetry_factor = 1
                        for sym in self.sim_cfg["symmetry"]:
                            if sym != 0:
                                symmetry_factor *= 2**0.5
                        _mode_E /= symmetry_factor
                        _mode_H /= symmetry_factor
                    else:
                        _mode_E = objects[name]._mode_E[0]
                        _mode_H = objects[name]._mode_H[0]
                    #

                    _mode_E, _mode_H = normalize_fields_phase(_mode_E, _mode_H)
                    ht_m = _jax_to_torch(_mode_H).to(torch.complex64)[None, ...]
                    et_m = _jax_to_torch(_mode_E).to(torch.complex64)[None, ...]
                    monitor_profile[(wl, source_mode)] = (
                        unplaced_objects[name],  # source objective
                        ht_m,  # h_m
                        et_m,  # e_m
                        1,  # power scale is not necessary for fdtd
                    )

            ## we use grid_slice here to override the add_monitor_slice's grid_slice
            if config.has_symmetry:
                # print(objects[monitor.name].grid_slice)
                ## we expand grid_slice only if the grid_slice is symmetric to the symmetry axis
                if self._check_object_on_symmetry_axis(
                    input_slice_name=input_slice_name,
                    symmetry=self.sim_cfg.get("symmetry", (0, 0, 0)),
                ):
                    native_grid_indices = fdtdx.unfold_grid_slice(
                        objects[monitor.name], config, info["symmetry_mid_abs"]
                    )
                else:
                    # if the monitor is on the top half of the symmetry axis, we just use it. do not mirror it.
                    ## simply because our slice has to be a continuous slice(start, stop), cannot have gap..
                    native_grid_indices = tuple(
                        slice(s.start + offset, s.stop + offset)
                        for s, offset in zip(
                            objects[monitor.name].grid_slice,
                            info["symmetry_mid_abs"],
                        )
                    )
                # print(native_grid_indices)
                ## we record the reduced grid_slice for adjoint group check
                self.port_monitor_slices_native_symmetry[input_slice_name] = objects[
                    monitor.name
                ].grid_slice
            else:
                native_grid_indices = objects[monitor.name].grid_slice

            self.port_monitor_slices_native[input_slice_name] = native_grid_indices
            # print(input_slice_name, _mode_E.shape, _mode_H.shape, native_grid_indices)
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(2, 3)
            # ## plot ht_m and et_m's abs
            # print(ht_m.shape, et_m.shape)
            # axes[0, 0].imshow(ht_m[0, 0].abs().squeeze().cpu().numpy(), origin="lower")
            # axes[0, 1].imshow(ht_m[0, 1].abs().squeeze().cpu().numpy(), origin="lower")
            # axes[0, 2].imshow(ht_m[0, 2].abs().squeeze().cpu().numpy(), origin="lower")
            # axes[1, 0].imshow(et_m[0, 0].abs().squeeze().cpu().numpy(), origin="lower")
            # axes[1, 1].imshow(et_m[0, 1].abs().squeeze().cpu().numpy(), origin="lower")
            # axes[1, 2].imshow(et_m[0, 2].abs().squeeze().cpu().numpy(), origin="lower")
            # plt.savefig(f"mode_{wl}_{source_mode}.png")
            # exit(0)

        elif type == "flux":
            for wl in wls:
                monitor = fdtdx.PoyntingFluxDetector(
                    name=f"flux_detector_{direction}_{wl}",
                    partial_grid_shape=partial_grid_shape,
                    direction=direction[-1],
                )
                monitor_profile[(wl, None)] = (
                    monitor,  # source objective
                    None,  # not used
                    None,  # not used
                    1,  # power scale is not necessary for fdtd
                )
        elif type == "field":
            for source_mode in source_modes:
                monitor = fdtdx.PhasorDetector(
                    name=f"field_detector_{source_mode}",
                    partial_grid_shape=self.epsilon_map.shape,
                    wave_characters=tuple(
                        fdtdx.WaveCharacter(wavelength=wl * MICRON_UNIT) for wl in wls
                    ),
                    components=field_components,
                    scaling_mode="pulse",
                    plot=False,
                    switch=fdtdx.OnOffSwitch(
                        interval=8
                    ),  # we downsample by 8x, note that the field amplitude also reduce proprtionally
                    exact_interpolation=True,
                    # exact_interpolation=False,
                )
                ## this broadband phasor detector saves (num_frequencies, num_components, X, Y, Z) tensor for fields.
                monitor_profile[((wl_cen, wl_width, n_wl), source_mode)] = (
                    monitor,  # source objective
                    None,  # not used
                    None,  # not used
                    1,  # power scale is not necessary for fdtd
                )
        elif type == "near2far":
            if self.dim != 3:
                raise ValueError(
                    "near2far PhasorDetector monitors require a 3-D device"
                )
            if input_slice_name not in self.port_monitor_slices_native:
                raise KeyError(
                    f"Near2far slice {input_slice_name!r} is not registered on the native grid"
                )
            native_slice = self.port_monitor_slices_native[input_slice_name]
            full_shape = tuple(int(size) for size in self.epsilon_map.shape)
            native_bounds = []
            for axis, axis_slice in enumerate(native_slice):
                if isinstance(axis_slice, slice):
                    start = 0 if axis_slice.start is None else int(axis_slice.start)
                    stop = (
                        full_shape[axis]
                        if axis_slice.stop is None
                        else int(axis_slice.stop)
                    )
                else:
                    start = int(axis_slice)
                    stop = start + 1
                if start < 0 or stop > full_shape[axis] or stop <= start:
                    raise ValueError(
                        f"Near2far slice {input_slice_name!r} has invalid native "
                        f"bounds on axis {axis}: {(start, stop)}"
                    )
                native_bounds.append((start, stop))
            native_bounds = tuple(native_bounds)
            detector_shape = tuple(stop - start for start, stop in native_bounds)
            if sum(size == 1 for size in detector_shape) != 1:
                raise ValueError(
                    "near2far monitor must be a single-cell-thick planar surface, "
                    f"got native shape {detector_shape}"
                )
            normal_axis = next(
                axis for axis, size in enumerate(detector_shape) if size == 1
            )
            components = (
                ("Ey", "Ez", "Hy", "Hz"),
                ("Ex", "Ez", "Hx", "Hz"),
                ("Ex", "Ey", "Hx", "Hy"),
            )[normal_axis]
            for source_mode in source_modes:
                monitor = fdtdx.PhasorDetector(
                    name=f"near2far_detector_{input_slice_name}_{source_mode}",
                    partial_grid_shape=detector_shape,
                    wave_characters=tuple(
                        fdtdx.WaveCharacter(wavelength=wl * MICRON_UNIT) for wl in wls
                    ),
                    components=components,
                    scaling_mode="pulse",
                    switch=fdtdx.OnOffSwitch(interval=8),
                    exact_interpolation=True,
                )
                monitor_profile[
                    ((wl_cen, wl_width, n_wl), source_mode, input_slice_name)
                ] = (monitor, None, None, 1, native_bounds)
        elif type in {"design_region_field", "sparse_field"}:
            visualization_supports = set()

            def _register_visualization_monitors(support_name, center_bounds):
                """Register unique forward-only slabs around one plot center.

                Args:
                    support_name: Design-region name or ``"origin"`` fallback.
                    center_bounds: Three native half-open bounds whose midpoint
                        defines the cross-section center.
                """
                if (
                    type != "sparse_field"
                    or not _MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES
                ):
                    return
                visualization_bounds = _design_region_visualization_bounds(
                    center_bounds,
                    self.epsilon_map.shape,
                    self.sim_cfg.get("symmetry", (0, 0, 0)),
                )
                for plane_name, plane_bounds in visualization_bounds.items():
                    plane_shape = tuple(stop - start for start, stop in plane_bounds)
                    for source_mode in source_modes:
                        support_signature = (source_mode, plane_bounds)
                        if support_signature in visualization_supports:
                            continue
                        visualization_supports.add(support_signature)
                        monitor = fdtdx.PhasorDetector(
                            name=(
                                f"field_detector_visualization_{support_name}_"
                                f"{plane_name}_{source_mode}"
                            ),
                            partial_grid_shape=plane_shape,
                            wave_characters=tuple(
                                fdtdx.WaveCharacter(wavelength=wl * MICRON_UNIT)
                                for wl in wls
                            ),
                            components=field_components,
                            scaling_mode="pulse",
                            switch=fdtdx.OnOffSwitch(interval=8),
                            exact_interpolation=True,
                        )
                        monitor_profile[
                            (
                                (wl_cen, wl_width, n_wl),
                                source_mode,
                                f"visualization:{support_name}:{plane_name}",
                            )
                        ] = (
                            monitor,
                            None,
                            None,
                            1,
                            plane_bounds,
                        )

            for region_name, region_mask in self.design_region_masks.items():
                region_index = self._region_mask_to_index(region_mask)
                if len(region_index) != 3:
                    raise ValueError(
                        "design_region_field requires 3-D design-region masks, "
                        f"got {region_name!r} with {len(region_index)} axes"
                    )
                region_shape = tuple(
                    int(axis_slice.stop) - int(axis_slice.start)
                    for axis_slice in region_index
                )
                region_bounds = tuple(
                    (int(axis_slice.start), int(axis_slice.stop))
                    for axis_slice in region_index
                )
                if any(size <= 0 for size in region_shape):
                    raise ValueError(
                        f"Design region {region_name!r} has an empty grid slice: "
                        f"{region_index}"
                    )
                for source_mode in source_modes:
                    monitor = fdtdx.PhasorDetector(
                        name=f"field_detector_{region_name}_{source_mode}",
                        partial_grid_shape=region_shape,
                        wave_characters=tuple(
                            fdtdx.WaveCharacter(wavelength=wl * MICRON_UNIT)
                            for wl in wls
                        ),
                        components=field_components,
                        scaling_mode="pulse",
                        switch=fdtdx.OnOffSwitch(interval=8),
                        exact_interpolation=True,
                    )
                    monitor_profile[
                        ((wl_cen, wl_width, n_wl), source_mode, region_name)
                    ] = (
                        monitor,  # source objective
                        None,  # not used
                        None,  # not used
                        1,  # power scale is not necessary for fdtd
                        region_bounds,
                    )
                _register_visualization_monitors(region_name, region_bounds)

            if not self.design_region_masks:
                origin_bounds = _origin_grid_cell_bounds(self.coords)
                _register_visualization_monitors("origin", origin_bounds)
            if type == "sparse_field":
                full_shape = tuple(int(size) for size in self.epsilon_map.shape)
                for port_name, port_slice in self.port_monitor_slices_native.items():
                    if len(port_slice) != 3:
                        raise ValueError(
                            "sparse_field requires 3-D native port slices, "
                            f"got {port_name!r} with {len(port_slice)} axes"
                        )
                    port_bounds = []
                    for axis, axis_slice in enumerate(port_slice):
                        if not isinstance(axis_slice, slice):
                            start = int(axis_slice)
                            stop = start + 1
                        else:
                            start = (
                                0 if axis_slice.start is None else int(axis_slice.start)
                            )
                            stop = (
                                full_shape[axis]
                                if axis_slice.stop is None
                                else int(axis_slice.stop)
                            )
                        if start < 0 or stop > full_shape[axis] or stop <= start:
                            raise ValueError(
                                f"Port {port_name!r} has invalid native bounds on "
                                f"axis {axis}: {(start, stop)}, full_shape={full_shape}"
                            )
                        port_bounds.append((start, stop))
                    port_bounds = tuple(port_bounds)
                    port_shape = tuple(stop - start for start, stop in port_bounds)
                    for source_mode in source_modes:
                        monitor = fdtdx.PhasorDetector(
                            name=f"field_detector_port_{port_name}_{source_mode}",
                            partial_grid_shape=port_shape,
                            wave_characters=tuple(
                                fdtdx.WaveCharacter(wavelength=wl * MICRON_UNIT)
                                for wl in wls
                            ),
                            components=field_components,
                            scaling_mode="pulse",
                            switch=fdtdx.OnOffSwitch(interval=8),
                            exact_interpolation=True,
                        )
                        support_name = f"port:{port_name}"
                        monitor_profile[
                            ((wl_cen, wl_width, n_wl), source_mode, support_name)
                        ] = (
                            monitor,
                            None,
                            None,
                            1,
                            port_bounds,
                        )
        else:
            raise ValueError(f"Monitor type {type} not supported")
        return monitor_profile

    def insert_plane_wave_fdtd3d(
        self,
        source_slice_name: str,
        eps,
        slice: Slice,
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        grid_step=None,
        power_scales: dict = None,
        source_modes: Tuple[int] = ("Ez1",),
        direction: str = "x+",
    ):
        """Create a native FDTD-X uniform TFSF plane source.

        Args:
            source_slice_name: Registered source-plane slice name.
            eps: Retained for source-builder API compatibility.
            slice: Retained for source-builder API compatibility.
            wl_cen: Center wavelength in micrometers.
            wl_width: Broadband wavelength span recorded in the profile key.
            n_wl: Number of monitor wavelengths recorded in the profile key.
            grid_step: Retained for source-builder API compatibility.
            power_scales: Retained for source-builder API compatibility.
            source_modes: Labels whose one-based suffix selects one of the two
                transverse electric polarizations.
            direction: Propagation direction such as ``"z+"``.

        Returns:
            Source profiles keyed by ``((wl_cen, wl_width, n_wl), mode)``.

        Notes:
            ``fdtdx.UniformPlaneSource`` is the concrete uniform-amplitude
            implementation of FDTD-X's TFSF plane-source formulation. Its
            ``amplitude`` and ``static_amplitude_factor`` retain the native
            FDTD-X field-amplitude convention.
        """
        del eps, slice, grid_step, power_scales
        if len(direction) != 2 or direction[0] not in "xyz" or direction[1] not in "+-":
            raise ValueError(f"Invalid plane-wave direction {direction!r}")
        source_info = self.port_monitor_slices_info[source_slice_name]
        source_size = source_info["size"]
        source_center = source_info["center"]
        normal_axis = "xyz".index(direction[0])
        transverse_axes = [axis for axis in range(3) if axis != normal_axis]
        partial_grid_shape = [None, None, None]
        partial_grid_shape[normal_axis] = 1
        partial_real_shape = [size * MICRON_UNIT for size in source_size]
        partial_real_shape[normal_axis] = None
        wave = fdtdx.WaveCharacter(wavelength=wl_cen * MICRON_UNIT)
        pulse = fdtdx.GaussianPulseProfile(
            center_wave=wave,
            spectral_width=fdtdx.WaveCharacter(wavelength=wl_cen * MICRON_UNIT * 10),
        )
        profiles = {}
        for source_mode in source_modes:
            try:
                polarization_index = int(str(source_mode)[-1]) - 1
            except ValueError as error:
                raise ValueError(
                    f"Plane-wave mode {source_mode!r} must end in 1 or 2"
                ) from error
            if polarization_index not in (0, 1):
                raise ValueError(f"Plane-wave mode {source_mode!r} must end in 1 or 2")
            electric_polarization = [0.0, 0.0, 0.0]
            electric_polarization[transverse_axes[polarization_index]] = 1.0
            source = fdtdx.UniformPlaneSource(
                partial_real_position=tuple(
                    coordinate * MICRON_UNIT for coordinate in source_center
                ),
                amplitude=1.0,
                wave_character=wave,
                direction=direction[-1],
                fixed_E_polarization_vector=tuple(electric_polarization),
                fixed_H_polarization_vector=None,
                partial_grid_shape=tuple(partial_grid_shape),
                partial_real_shape=tuple(partial_real_shape),
                temporal_profile=pulse,
                normalize_by_energy=False,
            )
            profiles[((wl_cen, wl_width, n_wl), source_mode)] = (
                source,
                None,
                None,
                1,
            )
        return profiles

    def insert_gaussian_beam_fdtd3d(
        self,
        source_slice_name: str,
        eps,
        slice: Slice,
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        grid_step=None,
        power_scales: dict = None,
        source_modes: Tuple[int] = ("Ez1",),
        # radii: Tuple[float] = (1e-6,),  # m
        waist_radii: Tuple[float] = (1e-6,),  # m
        waist_distances: Tuple[float] = (0.0,),  # m
        direction: str = "x+",
    ):
        # for 3D FDTDX, the mode is not solved until the apply_params is called on source object every optimization iteration.
        # fdtdx, _ = _import_fdtdx_runtime()
        """
        Create normalized broadband Gaussian-plane source profiles for FDTD-X.

        Args:
            source_slice_name: Port slice name used to obtain physical source center and extent.
            eps: Retained for API compatibility; the current Gaussian-plane implementation does not directly use the epsilon array.
            slice: Port monitor slice used to determine the source-plane grid location.
            wl_cen: Center wavelength in micrometers.
            wl_width: Broadband wavelength span recorded in the source-profile key.
            n_wl: Number of wavelength samples recorded in source metadata.
            grid_step: Optional optical grid step, retained for compatibility.
            power_scales: Optional mapping from ``(wavelength, mode)`` to an amplitude scale applied to the generated Gaussian source.
            source_modes: Modal labels; for supported ``z`` propagation the final digit selects fixed transverse E polarization.
            waist_radii: Gaussian waist radii. ``None`` derives a radius from the transverse source extent.
            waist_distances: Beam-waist distances from the source plane.
            direction: Propagation direction. The current Gaussian implementation supports ``z`` propagation.

        Returns:
            Dictionary keyed by ``((wl_cen, wl_width, n_wl), source_mode)``. Each value is ``(source_object, None, None, 1)``.

        Normalization contract:
            The FDTD-X source uses ``normalize_by_poynting_flux=True`` and ``normalize_by_energy=False`` with unit ``static_amplitude_factor`` by default so the launched full-domain optical power is one watt after discrete source-spectrum normalization. ``GaussianPlaneSourceTidy3d`` accounts for active symmetry multiplicity internally.
        """
        mode_profiles = {}
        wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)

        partial_grid_shape, _ = convert_to_fdtdx_grid_shape_loc_slice(slice, direction)

        source_slice_size = self.port_monitor_slices_info[source_slice_name]["size"]
        source_slice_center = self.port_monitor_slices_info[source_slice_name]["center"]
        partial_real_shape = [s * MICRON_UNIT for s in source_slice_size]
        direction_axis = "xyz".index(direction[0])
        partial_grid_shape = [None, None, None]
        partial_grid_shape[direction_axis] = 1
        partial_real_shape[direction_axis] = None

        for source_mode, waist_radius, waist_distance in zip(
            source_modes, waist_radii, waist_distances
        ):
            wave = fdtdx.WaveCharacter(wavelength=wl_cen * MICRON_UNIT)
            pulse = fdtdx.GaussianPulseProfile(
                center_wave=wave,
                spectral_width=fdtdx.WaveCharacter(
                    wavelength=wl_cen * MICRON_UNIT * 10
                ),
            )
            if waist_radius is None:
                if direction[0] == "x":
                    waist_radius = min(partial_real_shape[1:3]) / 2
                elif direction[0] == "y":
                    waist_radius = min(partial_real_shape[0], partial_real_shape[2]) / 2
                elif direction[0] == "z":
                    waist_radius = min(partial_real_shape[0:2]) / 2
                else:
                    raise ValueError(f"Direction {direction} not supported")
            # std = 1/3 by default, it means the Gaussian drops to ~exp(-4.5) ≈ 1% at the radius edge
            ## Hz1 means: X-polarized E field.
            ## Hz2 means: Y-polarized E field.
            if direction[0] == "z":
                if source_mode[-1] == "1":
                    # X-polarized E field.
                    fixed_E_polarization_vector = (1.0, 0.0, 0.0)
                    fixed_H_polarization_vector = None
                elif source_mode[-1] == "2":
                    # Y-polarized E field.
                    fixed_E_polarization_vector = (0.0, 1.0, 0.0)
                    fixed_H_polarization_vector = None
                else:
                    raise ValueError(
                        f"Source mode {source_mode} not supported for direction {direction}"
                    )
            else:
                raise ValueError(
                    f"Direction {direction} not supported for Gaussian beam source in fdtdx"
                )

            # Compute scaling factor based on symmetry
            # [09/01/2026] comment out this: fdtdx internally handled symmetry multiplicity, so we don't need to scale the source amplitude by symmetry factor.
            # symmetry_factor = 1.0
            # for axis, sym in enumerate(self.sim_cfg.get("symmetry", [0, 0, 0])):
            #     if sym != 0:
            #         symmetry_factor *= 2**0.5  # Each symmetric axis halves the domain

            ## for original fdtdx.GaussianPlaneSource
            ## in tidy3d, waist_radius = √2 × σ
            ## in fdtdx, radius = 3 × σ
            ## so to have the same gaussian source, we need to set gaussian_radius = √2/3 * waist_radius

            ## [06/25/2026] We support GaussianPlaneSourceTidy3d with same waist_radius and waist_distance as Tidy3d
            source = fdtdx.GaussianPlaneSourceTidy3d(
                partial_real_position=tuple(
                    source_slice_center[i] * MICRON_UNIT for i in range(3)
                ),
                waist_radius=waist_radius,  # Beam radius at waist
                waist_distance=waist_distance,  # Distance from the source plane to the beam waist
                wave_character=wave,
                direction=direction[-1],  # Propagating downward (negative z)
                fixed_E_polarization_vector=fixed_E_polarization_vector,
                fixed_H_polarization_vector=fixed_H_polarization_vector,
                partial_grid_shape=partial_grid_shape,
                partial_real_shape=partial_real_shape,
                static_amplitude_factor=(
                    1.0
                    if power_scales is None
                    else power_scales.get((wls[0], source_mode), 1.0)
                ),
                temporal_profile=pulse,
                normalize_by_energy=False,  # launch unit power not energy.
                normalize_by_poynting_flux=True,  # launch unit power not energy.
            )

            ## We have broadband source, use wl_cen, wl_width, n_wl as key
            mode_profiles[((wl_cen, wl_width, n_wl), source_mode)] = (
                source,  # source objective
                None,  # not used
                None,  # not used
                1,  # power scale is not necessary for fdtd
            )
        return mode_profiles

    def insert_modes_fdtd3d(
        self,
        source_slice_name: str,
        eps,
        slice: Slice,
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        grid_step=None,
        power_scales: dict = None,
        source_modes: Tuple[int] = ("Ez1",),
        direction: str = "x+",
    ):
        # for 3D FDTDX, the mode is not solved until the apply_params is called on source object every optimization iteration.
        # fdtdx, _ = _import_fdtdx_runtime()
        """
        Create broadband FDTD-X eigenmode plane sources from a port slice.

        Args:
            source_slice_name: Port slice name used to obtain physical source center and extent.
            eps: Retained for API compatibility; the current implementation builds the FDTD-X mode source without directly reading the epsilon tensor here.
            slice: Port monitor slice used to determine the source-plane grid location.
            wl_cen: Center wavelength in micrometers.
            wl_width: Broadband wavelength span.
            n_wl: Number of wavelength samples recorded in source metadata.
            grid_step: Optional optical grid step, retained for compatibility.
            power_scales: Optional mapping from ``(wavelength, mode)`` to an amplitude scale.
            source_modes: Modal labels such as ``\"Ez1\"`` and ``\"Hz1\"``. The numeric suffix is converted from one-based user numbering to zero-based FDTD-X ``mode_index``.
            direction: Propagation direction such as ``\"x+\"`` or ``\"y-\"``.

        Returns:
            Dictionary keyed by ``((wl_cen, wl_width, n_wl), source_mode)``. Each value is ``(ModePlaneSource, None, None, 1)``.

        Symmetry contract:
            The source is checked against device symmetry and its static amplitude is divided by ``sqrt(2)`` for each active symmetry axis.

        Polarization contract:
            ``Ez*`` sources use ``filter_pol=\"tm\"``; ``Hz*`` sources use ``filter_pol=\"te\"``.
        """
        mode_profiles = {}
        wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)

        # partial_grid_shape, _ = convert_to_fdtdx_grid_shape_loc_slice(slice, direction)
        source_slice_size = self.port_monitor_slices_info[source_slice_name]["size"]
        source_slice_center = self.port_monitor_slices_info[source_slice_name]["center"]
        partial_real_shape = [s * MICRON_UNIT for s in source_slice_size]
        direction_axis = "xyz".index(direction[0])
        partial_grid_shape = [None, None, None]
        partial_grid_shape[direction_axis] = 1
        partial_real_shape[direction_axis] = None

        for source_mode in source_modes:
            wave = fdtdx.WaveCharacter(wavelength=wl_cen * MICRON_UNIT)
            pulse = fdtdx.GaussianPulseProfile(
                center_wave=wave,
                spectral_width=fdtdx.WaveCharacter(
                    wavelength=wl_cen * MICRON_UNIT * 20
                ),
            )

            # Compute scaling factor based on symmetry
            self._check_object_symmetry(
                source_slice_name, symmetry=self.sim_cfg.get("symmetry", (0, 0, 0))
            )
            symmetry_factor = 1.0
            for axis, sym in enumerate(self.sim_cfg.get("symmetry", [0, 0, 0])):
                if sym != 0:
                    symmetry_factor *= 2**0.5  # Each symmetric axis halves the domain

            broadband_mode_method = self.sim_cfg.get(
                "fdtdx_mode_source_method", "chebyshev_3"
            )
            chebyshev_orders = {
                "chebyshev_3": 3,
                "chebyshev_5": 5,
            }
            if broadband_mode_method not in (
                "center",
                "exact",
                *chebyshev_orders,
            ):
                raise ValueError(
                    "sim_cfg['fdtdx_mode_source_method'] must be 'center', "
                    "'chebyshev_3', 'chebyshev_5', or 'exact', got "
                    f"{broadband_mode_method!r}."
                )
            use_frequency_dependent_profiles = (
                broadband_mode_method == "exact" and n_wl > 1 and wl_width > 0
            )
            use_chebyshev_profiles = (
                broadband_mode_method in chebyshev_orders and n_wl > 1 and wl_width > 0
            )
            if use_frequency_dependent_profiles:
                source_class = fdtdx.BroadbandModePlaneSource
                broadband_source_kwargs = {
                    "wave_characters": tuple(
                        fdtdx.WaveCharacter(wavelength=float(wl) * MICRON_UNIT)
                        for wl in wls
                    )
                }
            elif use_chebyshev_profiles:
                source_class = fdtdx.ChebyshevModePlaneSource
                broadband_source_kwargs = {
                    "num_freqs": chebyshev_orders[broadband_mode_method]
                }
            else:
                source_class = fdtdx.ModePlaneSource
                broadband_source_kwargs = {}
            source = source_class(
                name=f"input_mode_source_{source_mode}",
                partial_grid_shape=partial_grid_shape,
                partial_real_shape=partial_real_shape,
                partial_real_position=tuple(
                    source_slice_center[i] * MICRON_UNIT for i in range(3)
                ),
                wave_character=wave,
                direction=direction[-1],
                mode_index=int(source_mode[2:]) - 1,
                temporal_profile=pulse,
                static_amplitude_factor=(
                    1.0
                    if power_scales is None
                    else power_scales.get((wls[0], source_mode), 1.0)
                )
                / symmetry_factor,
                filter_pol="tm" if source_mode.startswith("Ez") else "te",
                **broadband_source_kwargs,
            )
            ## We have broadband source, use wl_cen, wl_width, n_wl as key
            mode_profiles[((wl_cen, wl_width, n_wl), source_mode)] = (
                source,  # source objective
                None,  # not used
                None,  # not used
                1,  # power scale is not necessary for fdtd
            )
        return mode_profiles

    def insert_modes(
        self,
        eps,
        slice: Slice,
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        grid_step=None,
        power_scales: dict = None,
        source_modes: Tuple[int] = ("Ez1",),
        direction: str = "x+",
        dxs=None,
        dys=None,
    ):
        """
        Solve 2-D eigenmodes on the configured monitor slice and build per-wavelength source/field profiles.

        Args:
            eps: Permittivity map on the optical native grid used by the modal solver.
            slice: ``Slice`` defining the transverse mode-solving plane.
            wl_cen: Center wavelength in micrometers.
            wl_width: Wavelength span in micrometers.
            n_wl: Number of wavelength samples.
            grid_step: Optional physical grid step in micrometers; defaults to ``self.grid_step``.
            power_scales: Optional mapping from ``(wavelength, mode)`` to an amplitude scale applied to returned mode fields.
            source_modes: Modal identifiers such as ``\"Ez1\"`` or ``\"Hz1\"`` passed to ``insert_mode``.
            direction: Propagation direction such as ``\"x+\"`` or ``\"y-\"``.
            dxs: Optional nonuniform x-cell widths in physical solver units.
            dys: Optional nonuniform y-cell widths in physical solver units; ``dxs`` and ``dys`` must be supplied together when overriding spacing.

        Returns:
            Dictionary keyed by ``(wl, source_mode)`` with values ``[mode, H_mode, E_mode, power_scale]``.

        Notes:
            When ``dxs``/``dys`` are omitted, native-grid boundary metadata is used to derive the cell widths. This keeps modal solving consistent with nonuniform FDFD grids.
        """
        grid_step = grid_step or self.grid_step
        dl = grid_step * MICRON_UNIT
        if dxs is None and dys is None:
            epsilon_grid_info = self.grid_info_dict.get("epsilon_map")
            if epsilon_grid_info is not None:
                boundaries = epsilon_grid_info.get("boundaries")
                if boundaries is not None and len(boundaries) >= 2:
                    dxs = np.diff(np.asarray(boundaries[0], dtype=float))
                    dys = np.diff(np.asarray(boundaries[1], dtype=float))
        elif dxs is None or dys is None:
            raise ValueError("dxs and dys must be provided together")
        mode_profiles = {}

        for wl in np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl):
            for source_mode in source_modes:
                # there is no need to calculate the modes for different temperatures
                # since the eps is only modulated at active region
                # current_eps = get_temp_related_eps(eps, wl, temp)
                omega = 2 * np.pi * C_0 / (wl * MICRON_UNIT)

                ht_m, et_m, _, mode = insert_mode(
                    omega,
                    dl,
                    slice.x,
                    slice.y,
                    eps,
                    m=source_mode,
                    direction=direction,
                    dxs=dxs,
                    dys=dys,
                )
                # print(ht_m)
                # ht_m, et_m, _, mode = insert_mode_spins(
                #     omega, dl, slice.x, slice.y, eps, m=source_mode
                # )
                # print(ht_m)
                # exit(0)
                if power_scales is not None:
                    power_scale = power_scales[(wl, source_mode)]
                    ht_m = ht_m * power_scale
                    et_m = et_m * power_scale
                    mode = mode * power_scale
                else:
                    power_scale = 1
                mode_profiles[(wl, source_mode)] = [mode, ht_m, et_m, power_scale]
        return mode_profiles

    def insert_plane_wave(
        self,
        eps,
        slice: Slice,
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        source_modes: Tuple[str] = ("Ez1",),
        grid_step=None,
        power_scales: dict = None,
        direction: str = "x+",
        custom_source: np.ndarray | torch.Tensor = None,
    ):
        """
        Build a complex plane-wave source profile directly on a 2-D monitor slice.

        Args:
            eps: Permittivity array defining the source-map canvas; Torch is selected when ``custom_source`` is a Torch tensor.
            slice: ``Slice`` defining the source plane.
            wl_cen: Center wavelength in micrometers used for the phase increment between adjacent source planes.
            wl_width: Wavelength span in micrometers.
            n_wl: Number of wavelength samples.
            source_modes: Mode labels used as dictionary keys; the plane wave is not eigenmode-solved.
            grid_step: Physical grid step in micrometers.
            power_scales: Optional mapping from ``(wavelength, mode)`` to source amplitude scaling.
            direction: Propagation direction using ``x+``/``x-`` or ``y+``/``y-``.
            custom_source: Optional NumPy or Torch source pattern replacing unit-amplitude values on the source slice.

        Returns:
            Dictionary keyed by ``(wl, source_mode)`` with values ``[source, H_profile, E_profile, power_scale]``.

        Notes:
            The source is written on the selected slice and one adjacent plane with a phase factor based on ``wl_cen`` and ``grid_step``; the direction sign selects which adjacent plane receives that phase-shifted copy.
        """
        if isinstance(custom_source, torch.Tensor):
            lib = torch
            eps = torch.tensor(eps, dtype=torch.float32, device=custom_source.device)
        elif isinstance(custom_source, np.ndarray) or custom_source is None:
            lib = np
        else:
            raise ValueError("custom_source must be either np.ndarray or torch.Tensor")
        grid_step = grid_step or self.grid_step
        source_profiles = {}
        offset = -1 if direction[1] == "+" else 1
        for wl in lib.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl):
            for source_mode in source_modes:
                source = lib.zeros_like(eps, dtype=lib.complex64)
                if lib == torch:
                    source = source.to(custom_source.device)
                if direction[0] == "y":  # horizontal slice
                    source[:, slice.y] = 1 if custom_source is None else custom_source
                    if lib == torch:
                        source[:, slice.y + offset] = lib.exp(
                            torch.tensor(
                                [
                                    -1j * 2 * lib.pi / wl_cen * grid_step - 1j * lib.pi,
                                ],
                                device=source.device,
                            )
                        ) * (1 if custom_source is None else custom_source)
                    else:
                        source[:, slice.y + offset] = lib.exp(
                            -1j * 2 * lib.pi / wl_cen * grid_step - 1j * lib.pi
                        ) * (1 if custom_source is None else custom_source)
                elif direction[0] == "x":  # vertical slice
                    source[slice.x, :] = 1 if custom_source is None else custom_source
                    if lib == torch:
                        source[slice.x + offset, :] = lib.exp(
                            torch.tensor(
                                [
                                    -1j * 2 * lib.pi / wl_cen * grid_step - 1j * lib.pi,
                                ],
                                device=source.device,
                            )
                        ) * (1 if custom_source is None else custom_source)
                    else:
                        source[slice.x + offset, :] = lib.exp(
                            -1j * 2 * lib.pi / wl_cen * grid_step - 1j * lib.pi
                        ) * (1 if custom_source is None else custom_source)

                ht_m = et_m = source.reshape(-1)
                if power_scales is not None:
                    power_scale = power_scales[
                        (wl, source_mode)
                    ]  # use direction as a placeholder for mode
                    ht_m = et_m = et_m * power_scale
                    source = source * power_scale
                else:
                    power_scale = 1
                if isinstance(wl, torch.Tensor):
                    wl = round(wl.item(), 2)
                source_profiles[(wl, source_mode)] = [source, ht_m, et_m, power_scale]
        return source_profiles

    def create_simulation(
        self,
        omega,
        dl,
        eps,
        NPML,
        solver="ceviche",
        pol: str = "Ez",
        dxs=None,
        dys=None,
    ):
        """
        Construct a 2-D frequency-domain simulation for the selected polarization and solver backend.

        Args:
            omega: Angular frequency in radians per second.
            dl: Nominal grid spacing in physical solver units.
            eps: Permittivity map.
            NPML: PML thickness in cell counts. For nonuniform ``dxs``/``dys`` grids, native-grid counts are used.
            solver: ``\"ceviche\"`` or ``\"ceviche_torch\"``.
            pol: ``\"Ez\"`` or ``\"Hz\"``.
            dxs: Optional native x-cell widths.
            dys: Optional native y-cell widths.

        Returns:
            Solver object configured for the selected backend and polarization.

        Raises:
            ValueError: If only one of ``dxs``/``dys`` is provided or the polarization/backend is unsupported.

        Notes:
            ``Ez`` and ``Hz`` select the corresponding Ceviche formulation. With rectilinear spacing, PML cell counts are interpreted on the same native grid as ``dxs``/``dys``.
        """
        if dxs is not None or dys is not None:
            if dxs is None or dys is None:
                raise ValueError("dxs and dys must be provided together")
            # NPML_export is defined for the uniform export grid.  A
            # rectilinear FDFD simulation, however, is assembled on the grid
            # represented by dxs/dys and therefore needs the native-grid PML
            # cell counts.  These counts are based on the requested physical
            # PML thickness and are not interchangeable with export counts.
            if all(np.isscalar(axis_count) for axis_count in NPML):
                NPML = self.NPML
        if solver == "ceviche":
            if pol == "Ez":
                return fdfd_ez(omega, dl, eps, NPML, dxs=dxs, dys=dys)
            elif pol == "Hz":
                return fdfd_hz(omega, dl, eps, NPML, dxs=dxs, dys=dys)
            else:
                raise ValueError(f"Pol {pol} not supported")
        elif solver == "ceviche_torch":
            if pol == "Ez":
                fdfd_fn = fdfd_ez_torch
            elif pol == "Hz":
                fdfd_fn = fdfd_hz_torch
            return fdfd_fn(
                omega,
                dl,
                eps,
                NPML,
                neural_solver=self.sim_cfg.get("neural_solver", None),
                numerical_solver=self.sim_cfg.get("numerical_solver", "solve_direct"),
                use_autodiff=self.sim_cfg.get("use_autodiff", False),
                dxs=dxs,
                dys=dys,
            )
        else:
            raise ValueError(f"Solver {solver} not supported")

    def create_simulation_fdtdx(self, eps, wl_cen, wl_width, n_wl, NPML):
        _, fdtd3d = _import_fdtdx_runtime()
        if not self._electrical_conductivity_map_built:
            self.build_electrical_conductivity_map()
        field_grid_metadata = self._build_fdtdx_native_grid_metadata()
        sim_eps = eps
        sim_sigma = self.electrical_conductivity_map
        native_grid = field_grid_metadata["native"]
        export_grid = field_grid_metadata["export"]
        sim_sigma = self._resample_map_to_fdtdx_native(
            sim_sigma,
            map_name="electrical_conductivity_map",
            native_grid=native_grid,
        )
        # if not field_grid_metadata.get("native_equals_export", False):
        #     sim_eps = self.build_fdtdx_native_permittivity(
        #         eps,
        #         field_grid_metadata=field_grid_metadata,
        #     )
        #     if sim_sigma is not None:
        #         sim_sigma = self.resample_map_between_coords(
        #             sim_sigma,
        #             src_coords=export_grid["coords"],
        #             dst_coords=native_grid["coords"],
        #         )
        return fdtd3d(
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
            dL=self.grid_step * MICRON_UNIT,
            eps_r=sim_eps,
            electrical_conductivity_map=sim_sigma,
            optical_grid_metadata=field_grid_metadata,
            port_sources_dict=self.port_sources_dict,
            port_monitor_slices=self.port_monitor_slices,
            port_monitor_slices_info=self.port_monitor_slices_info,
            npml=NPML,
            max_time=self.sim_cfg.get("max_time", 1e-10),
            symmetry=self.sim_cfg.get("symmetry", (0, 0, 0)),
            device=self.device,
        )

    def solve_fdtdx(
        self,
        eps,
        input_slice_name=None,  # every simulation, only one source
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        mode: str = "Ez1",
    ):
        simulation = self.create_simulation_fdtdx(
            eps,
            wl_cen,
            wl_width,
            n_wl,
            self.NPML,
        )
        fields = simulation.solve(
            input_slice_name=input_slice_name,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
            mode=mode,
            eps_r=eps,
        )
        if _MAPS_SPARSE_PHASOR:
            return fields
        Ex, Ey, Ez, Hx, Hy, Hz = fields
        return {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}

    def solve_ceviche(
        self,
        eps,
        source,
        wl: float = 1.55,
        grid_step=None,
        solver: str = "ceviche",
        pol: str = "Ez",
        dxs=None,
        dys=None,
    ):
        """
        Solve one 2-D frequency-domain source and return polarization-specific field components.

        Args:
            eps: Permittivity map supplied to the selected FDFD backend.
            source: Complex source profile.
            wl: Wavelength in micrometers.
            grid_step: Optional nominal grid step in micrometers.
            solver: ``\"ceviche\"`` or ``\"ceviche_torch\"``.
            pol: ``\"Ez\"`` or ``\"Hz\"``.
            dxs: Optional native x-cell widths.
            dys: Optional native y-cell widths.

        Returns:
            For ``Ez`` polarization, ``{\"Hx\": Fx, \"Hy\": Fy, \"Ez\": Fz}``; for ``Hz``, ``{\"Ex\": Fx, \"Ey\": Fy, \"Hz\": Fz}``.

        Notes:
            Angular frequency is computed from wavelength using ``MICRON_UNIT``. The Torch Ceviche normalization/reference path executes under ``torch.no_grad()``.
        """
        omega = 2 * np.pi * C_0 / (wl * MICRON_UNIT)
        grid_step = grid_step or self.grid_step
        dl = grid_step * MICRON_UNIT
        # simulation = fdfd_ez(omega, dl, eps, [self.NPML[0], self.NPML[1]])
        simulation = self.create_simulation(
            omega,
            dl,
            eps,
            self.NPML,
            solver=solver,
            pol=pol,
            dxs=dxs,
            dys=dys,
        )

        if hasattr(simulation, "solver"):  # which means that it is a torch simulation
            with torch.no_grad():
                Fx, Fy, Fz = simulation.solve(
                    source, slice_name="Norm", mode="Norm", temp="Norm"
                )
        else:
            Fx, Fy, Fz = simulation.solve(source)

        if pol == "Ez":
            return {"Hx": Fx, "Hy": Fy, "Ez": Fz}
        elif pol == "Hz":
            return {"Ex": Fx, "Ey": Fy, "Hz": Fz}
        else:
            raise ValueError(f"Unknown simulation {type(simulation)} type")

    def solve(
        self,
        eps,
        source_profiles,
        solver="ceviche",
        grid_step=None,
        dxs=None,
        dys=None,
    ):
        """
        Run the configured 2-D FDFD solver for every source profile and collect fields by wavelength and mode.

        Args:
            eps: Device permittivity map.
            source_profiles: Mapping keyed by ``(wl, mode)`` whose first tuple/list element is the source field.
            solver: Backend identifier. The active implementation supports ``\"ceviche\"`` and ``\"ceviche_torch\"``; ``\"fdtdx\"`` currently raises ``NotImplementedError`` in this high-level method.
            grid_step: Optional nominal grid step.
            dxs: Optional native x-cell widths.
            dys: Optional native y-cell widths.

        Returns:
            Mapping keyed by ``(wl, mode)`` to polarization-specific field dictionaries. ``Ez*`` yields ``Hx``, ``Hy``, ``Ez``; ``Hz*`` yields ``Ex``, ``Ey``, ``Hz``.

        Notes:
            The supplied ``eps`` is reused for each source profile; this method does not perform a wavelength/temperature material-update loop itself.
        """
        grid_step = grid_step or self.grid_step
        fields = {}
        if solver in {"ceviche", "ceviche_torch"}:
            for (wl, mode), (source, _, _, _) in source_profiles.items():
                # current_eps = get_temp_related_eps(eps, wl, temp)
                current_eps = eps
                pol = mode[:2]  # "Ez1" -> "Ez"
                field_sol = self.solve_ceviche(
                    current_eps,
                    source,
                    wl=wl,
                    grid_step=grid_step,
                    solver=solver,
                    pol=pol,
                    dxs=dxs,
                    dys=dys,
                )
                fields[(wl, mode)] = field_sol
            return fields
        elif solver == "fdtdx":
            raise NotImplementedError(
                "The current implementation of solve() does not support fdtdx solver"
            )
            ## we cluster multiple wavelengths for each mode as FDTDX is broadband
            # mode_to_wavelengths = {}
            # for wl, mode in source_profiles:
            #     if mode not in mode_to_wavelengths:
            #         mode_to_wavelengths[mode] = []
            #     mode_to_wavelengths[mode].append(wl)
            # for mode, wls in mode_to_wavelengths.items():
            #     field_sol = self.solve_fdtdx(
            #         eps,
            #         source,
            #         wls=wls,
            #         grid_step=grid_step,
            #         mode=mode,
            #     )  # {wl: {"Hx": Hx, "Hy": Hy, "Ez": Ez}, ...} multi-wavelength solutions
            #     for wl in wls:
            #         fields[(wl, mode)] = field_sol[wl]

        else:
            raise ValueError(f"Solver {solver} not supported")

    @lru_cache(maxsize=128)
    def build_norm_sources(
        self,
        source_modes: Tuple[str] = ("Ez1",),
        input_port_name: str = "in_port_1",
        input_slice_name: str = "in_slice_1",
        wl_cen=1.55,
        wl_width=0,
        n_wl=1,
        solver="ceviche",
        power: float = 1e-8,
        source_type: str = "mode",
        waist_radii=(1e-6,),
        waist_distances=(0.0,),
        plot=False,
        require_sim: bool = False,
    ):
        """
        Build and normalize auxiliary two-port source profiles used to establish target-power optical excitation.

        Args:
            source_modes: Modal identifiers such as ``\"Ez1\"`` or ``\"Hz1\"``.
            input_port_name: Port key used by ``get_two_ports`` for the auxiliary normalization device.
            input_slice_name: Launch-plane monitor name.
            wl_cen: Center wavelength in micrometers.
            wl_width: Wavelength span in micrometers.
            n_wl: Number of wavelength samples.
            solver: Optical backend, currently Ceviche variants or FDTD-X.
            power: Target launched optical power.
            source_type: ``\"mode\"``, ``\"plane_wave\"``, or ``\"gaussian_beam\"``.
            waist_radii: Gaussian waist radii for the FDTD-X Gaussian source path.
            waist_distances: Gaussian waist distances.
            plot: Whether to export a diagnostic source/mode plot.
            require_sim: Whether the profile is intended to be inserted as a simulation source.

        Returns:
            Source-profile dictionary keyed by wavelength/mode. The common tuple layout is ``[source_profile, H_mode, E_mode, power_or_scale, require_sim]``.

        Normalization procedure:
            In the FDFD path, an auxiliary two-port device runs an unscaled reference excitation, measures reference flux, and scales the source by ``sqrt(power / measured_flux)`` independently for each wavelength/mode. The normalized profiles are stored in ``self.port_sources_dict[input_slice_name]``.

        Grid contract:
            The auxiliary two-port raster grid can differ from the original device grid. Full-field source profiles and transverse mode fields are resampled back onto the original native grid before they are returned.
        """
        assert source_type in {
            "mode",
            "plane_wave",
            "gaussian_beam",
        }, f"Source type {source_type} not supported"

        input_slice = self.port_monitor_slices[input_slice_name]
        direction = self.port_monitor_slices_info[input_slice_name]["direction"]
        two_ports = get_two_ports(self, input_port_name, input_slice_name)

        if solver == "fdtdx":
            if require_sim:  # it is a light source
                if source_type == "mode":
                    source_profiles = two_ports.insert_modes_fdtd3d(
                        input_slice_name,
                        None,
                        input_slice,
                        wl_cen=wl_cen,
                        wl_width=wl_width,
                        n_wl=n_wl,
                        power_scales=None,
                        source_modes=source_modes,
                        direction=direction,
                    )
                elif source_type == "plane_wave":
                    source_profiles = two_ports.insert_plane_wave_fdtd3d(
                        input_slice_name,
                        None,
                        input_slice,
                        wl_cen=wl_cen,
                        wl_width=wl_width,
                        n_wl=n_wl,
                        power_scales=None,
                        source_modes=source_modes,
                        direction=direction,
                        grid_step=self.grid_step,
                    )
                elif source_type == "gaussian_beam":
                    source_profiles = two_ports.insert_gaussian_beam_fdtd3d(
                        input_slice_name,
                        None,
                        input_slice,
                        wl_cen=wl_cen,
                        wl_width=wl_width,
                        n_wl=n_wl,
                        power_scales=None,
                        source_modes=source_modes,
                        waist_radii=waist_radii,
                        waist_distances=waist_distances,
                        direction=direction,
                        grid_step=self.grid_step,
                    )
                ## every FDTDX simulation need PhasorDetector(s) to record fields for this source (that requires simulation).
                if _MAPS_SPARSE_PHASOR:
                    ### we need to insert phasor detectors for each design region
                    phasor_profiles = self.insert_monitors_fdtd3d(
                        input_slice_name,
                        self.epsilon_map,
                        "sparse_field",
                        input_slice,
                        wl_cen=wl_cen,
                        wl_width=wl_width,
                        n_wl=n_wl,
                        source_modes=source_modes,
                        direction=direction,
                    )
                else:
                    phasor_profiles = two_ports.insert_monitors_fdtd3d(
                        input_slice_name,
                        two_ports.epsilon_map,
                        "field",
                        input_slice,
                        wl_cen=wl_cen,
                        wl_width=wl_width,
                        n_wl=n_wl,
                        source_modes=source_modes,
                        direction=direction,
                    )
                # add requires_sim
                phasor_profiles = {
                    k: list(v) + [False] for k, v in phasor_profiles.items()
                }
                ## create another auxiliary slice with the same slice prefix but with "_field_monitor" suffix to place the phasor detector for recording fields for this source, and store the monitor profiles in port_sources_dict with the same key as source_profiles but with "_field_monitor" suffix
                self.port_sources_dict[input_slice_name + "_field_monitor"] = (
                    phasor_profiles
                )
            # elif: # it is a monitor/detector, we do not place them in simulation, just need its mode_H and mode_E
            else:
                source_profiles = two_ports.insert_monitors_fdtd3d(
                    input_slice_name,
                    two_ports.epsilon_map,
                    "mode",
                    input_slice,
                    wl_cen=wl_cen,
                    wl_width=wl_width,
                    n_wl=n_wl,
                    source_modes=source_modes,
                    direction=direction,
                    slice_info=self.port_monitor_slices_info[input_slice_name],
                    on_native_grid=True,
                )
            source_profiles = {
                k: list(v) + [require_sim] for k, v in source_profiles.items()
            }
            self.port_sources_dict[input_slice_name] = source_profiles

            if plot:
                for key, (_, ht_m, et_m, *_) in source_profiles.items():
                    if ht_m is None or et_m is None:
                        break
                    import matplotlib.pyplot as plt

                    fig, axes = plt.subplots(3, 6, figsize=(20, 6))
                    ## row 0: abs; row 1: real; row 2: imag
                    ## col 0 Ex; col 1 Ey; col 2 Ez; col 3 Hx; col 4 Hy; col 5 Hz
                    epsilon_map = two_ports.epsilon_map
                    if isinstance(epsilon_map, torch.Tensor):
                        epsilon_map = epsilon_map.detach().cpu().numpy()
                    else:
                        epsilon_map = np.asarray(epsilon_map)
                    if np.iscomplexobj(epsilon_map):
                        epsilon_map = np.abs(epsilon_map)
                    grid_slice = self.port_monitor_slices_native[input_slice_name]
                    epsilon_overlay = epsilon_map[grid_slice].T.squeeze()
                    epsilon_overlay = np.flip(epsilon_overlay, axis=0)
                    # if epsilon_map.ndim == 3:
                    #     direction_axis = "xyz".index(direction[0])
                    #     slice_axis = getattr(input_slice, "xyz"[direction_axis])
                    #     if not isinstance(slice_axis, (int, np.integer)):
                    #         slice_axis = epsilon_map.shape[direction_axis] // 2
                    #     epsilon_overlay = np.take(
                    #         epsilon_map,
                    #         int(slice_axis),
                    #         axis=direction_axis,
                    #     ).T
                    # elif epsilon_map.ndim == 2:
                    #     epsilon_overlay = epsilon_map.T
                    # else:
                    #     epsilon_overlay = None

                    for i, component in enumerate(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]):
                        ax = axes[0, i]
                        idx = i % 3
                        mode = ht_m if component[0] == "H" else et_m

                        data = mode[0, idx].data.cpu().squeeze().numpy().T
                        data = np.flip(data, axis=0)

                        im = ax.imshow(np.abs(data), cmap="magma")
                        if (
                            epsilon_overlay is not None
                            and epsilon_overlay.shape == data.shape
                        ):
                            ax.imshow(
                                epsilon_overlay,
                                cmap="Grays",
                                alpha=0.3,
                                interpolation="nearest",
                            )
                        ax.set_title(f"|{component}|")
                        fig.colorbar(im, ax=ax)

                        ax = axes[1, i]
                        im = ax.imshow(np.real(data), cmap="RdBu")
                        if (
                            epsilon_overlay is not None
                            and epsilon_overlay.shape == data.shape
                        ):
                            ax.imshow(
                                epsilon_overlay,
                                cmap="Grays",
                                alpha=0.3,
                                interpolation="nearest",
                            )
                        ax.set_title(f"Re({component})")
                        fig.colorbar(im, ax=ax)

                        ax = axes[2, i]
                        im = ax.imshow(np.imag(data), cmap="RdBu")
                        if (
                            epsilon_overlay is not None
                            and epsilon_overlay.shape == data.shape
                        ):
                            ax.imshow(
                                epsilon_overlay,
                                cmap="Grays",
                                alpha=0.3,
                                interpolation="nearest",
                            )
                        ax.set_title(f"Im({component})")
                        fig.colorbar(im, ax=ax)
                    filepath = os.path.join(
                        self.sim_cfg["plot_root"],
                        f"{self.config.device.type}_mode-{input_slice_name}.png",
                    )
                    dirname = os.path.dirname(filepath)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    plt.tight_layout()
                    fig.savefig(filepath, dpi=300)
                    plt.close()
                    break

        else:
            direction = self.port_monitor_slices_info[input_slice_name]["direction"]
            # Mode profiles are inserted into the native full-device field and
            # must therefore be generated with the native device shape.  The
            # public monitor slice remains export-grid indexed for plotting.
            native_grid_info = two_ports.grid_info_dict.get("epsilon_map")
            input_slice_info = self.port_monitor_slices_info[input_slice_name]
            native_input_slice = two_ports._monitor_slice_on_coords(
                center=input_slice_info["center"],
                size=input_slice_info["size"],
                direction=input_slice_info["direction"],
                coords=native_grid_info["coords"],
            )
            native_boundaries = native_grid_info.get("boundaries")
            native_dxs = (
                np.diff(np.asarray(native_boundaries[0], dtype=float)) * MICRON_UNIT
            )
            native_dys = (
                np.diff(np.asarray(native_boundaries[1], dtype=float)) * MICRON_UNIT
            )

            if direction[0] == "x":
                output_slice = Slice(
                    x=two_ports.Nx - native_input_slice.x, y=native_input_slice.y
                )
                if two_ports.Nx - native_input_slice.x > native_input_slice.x:
                    direction = "x+"
                else:
                    direction = "x-"
            elif direction[0] == "y":
                output_slice = Slice(
                    x=native_input_slice.x, y=two_ports.Ny - native_input_slice.y
                )
                if two_ports.Ny - native_input_slice.y > native_input_slice.y:
                    direction = "y+"
                else:
                    direction = "y-"

            def _norm_run(power_scales=None):
                if source_type == "mode":
                    source_profiles = two_ports.insert_modes(
                        two_ports.epsilon_map,
                        native_input_slice,
                        wl_cen=wl_cen,
                        wl_width=wl_width,
                        n_wl=n_wl,
                        power_scales=power_scales,
                        source_modes=source_modes,
                        direction=direction,
                        dxs=native_dxs,
                        dys=native_dys,
                    )  # {(wl, mode): [source, ht_m, et_m, scale], ...}
                    # print_stat(source_profiles[(1.55, 1)][0])
                    # monitor_profiles = self.insert_modes(
                    #     in_port_eps,
                    #     output_slice,
                    #     wl_cen=wl_cen,
                    #     wl_width=wl_width,
                    #     n_wl=n_wl,
                    #     temp=temp,
                    #     power_scales=power_scales,
                    #     source_modes=source_modes,
                    # )  # {(wl, mode): [monitor, ht_m, et_m, scale], ...}
                elif source_type == "plane_wave":
                    source_profiles = self.insert_plane_wave(
                        two_ports.epsilon_map,
                        native_input_slice,
                        wl_cen=wl_cen,
                        wl_width=wl_width,
                        n_wl=n_wl,
                        source_modes=source_modes,
                        power_scales=power_scales,
                        direction=direction,
                    )

                # print_stat(monitor_profiles[(1.55, 1)][0])
                fields = two_ports.solve(
                    two_ports.epsilon_map,
                    source_profiles,
                    solver=solver,
                    grid_step=self.grid_step,
                    dxs=native_dxs,
                    dys=native_dys,
                )  # [(wl, mode, Hx), ...], [(wl, mode, Hy), ...], [(wl, mode, Ez), ...]
                # print_stat(fields[(1.55, 1)]["Ez"])

                input_SCALE = {}
                for k in source_profiles:
                    mode = k[1]
                    pol = mode[:2]
                    if pol == "Ez":
                        Fx, Fy, Fz = fields[k]["Hx"], fields[k]["Hy"], fields[k]["Ez"]
                    elif pol == "Hz":
                        Fx, Fy, Fz = fields[k]["Ex"], fields[k]["Ey"], fields[k]["Hz"]
                    else:
                        raise ValueError(f"Unknown polarization {pol}")

                    # _, ht_m, et_m, _ = source_profiles[k]
                    # print("this is the type of Hx:", type(Hx), flush=True)
                    # print("this is the type of Hy:", type(Hy), flush=True)
                    # print("this is the type of Ez:", type(Ez), flush=True)
                    # print("this is the type of ht_m:", type(ht_m), flush=True)
                    # print("this is the type of et_m:", type(et_m), flush=True)
                    # ht_m = torch.from_numpy(ht_m).to(Ez.device)
                    # et_m = torch.from_numpy(et_m).to(Ez.device)
                    # eigen_energy = get_eigenmode_coefficients(
                    #     Fx,
                    #     Fy,
                    #     Fz,
                    #     ht_m,
                    #     et_m,
                    #     output_slice,
                    #     grid_step=self.grid_step,
                    #     direction=direction,
                    #     energy=True,
                    #     pol=pol,
                    # )
                    # print("eigen_energy:", eigen_energy)
                    ## used to verify eigen mode coefficients, need to be the same as eigen energy
                    flux = get_flux(
                        Fx,
                        Fy,
                        Fz,
                        output_slice,
                        grid_step=self.grid_step,
                        direction=direction,
                        pol=pol,
                        cell_weights=(
                            two_ports.fdtdx_native_cell_weights
                            if two_ports.dim == 2
                            else None
                        ),
                    )
                    # print("norm flux:", flux)
                    if isinstance(flux, torch.Tensor):
                        flux = flux.item()
                    input_SCALE[k] = np.abs(flux)

                return input_SCALE, fields, source_profiles

            input_scale, fields, source_profiles = _norm_run()  # to get eigen energy
            input_scale = {
                k: (power / v) ** 0.5 for k, v in input_scale.items()
            }  # normalize the source power to target power for all wavelengths and modes

            pol = list(fields.keys())[0][1][:2]  # [wl, mode] get pol from mode
            Fz = list(fields.values())[0][pol]
            if isinstance(Fz, torch.Tensor):
                source_profiles = {
                    k: [torch.from_numpy(i).to(Fz.device) for i in v[:-1]] + [v[-1]]
                    for k, v in source_profiles.items()
                }
            source_profiles = {
                k: [e * input_scale[k] for e in v[:-1]] + [power] + [require_sim]
                for k, v in source_profiles.items()
            }
            # The auxiliary two-port rasterization can have a different
            # native shape because Tidy3D snapping depends on the structures
            # present in the simulation.  The normalized source is consumed
            # later by simulations of the original device, so transfer its
            # full-field source from the two-port native coordinates to the
            # original device native coordinates.
            source_grid = two_ports.grid_info_dict.get("epsilon_map")
            target_grid = self.grid_info_dict.get("epsilon_map")
            if source_grid is not None and target_grid is not None:
                source_shape = tuple(len(axis) for axis in source_grid["coords"])
                target_shape = tuple(len(axis) for axis in target_grid["coords"])
                grids_differ = source_shape != target_shape or not all(
                    src.shape == dst.shape
                    and np.allclose(src, dst, atol=1e-9, rtol=1e-6)
                    for src, dst in zip(source_grid["coords"], target_grid["coords"])
                )
                if grids_differ:
                    for key, profile in source_profiles.items():
                        source_field = profile[0]
                        if tuple(source_field.shape[-self.dim :]) == source_shape:
                            was_numpy = isinstance(source_field, np.ndarray)
                            resampled_source = two_ports.resample_map_between_coords(
                                source_field,
                                src_coords=source_grid["coords"],
                                dst_coords=target_grid["coords"],
                            )
                            if was_numpy:
                                resampled_source = (
                                    resampled_source.detach().cpu().numpy()
                                )
                            source_profiles[key][0] = resampled_source

                        transverse_axes = {
                            "x": (1, 2),
                            "y": (0, 2),
                            "z": (0, 1),
                        }[direction[0]]
                        target_input_slice = self._monitor_slice_on_coords(
                            center=input_slice_info["center"],
                            size=input_slice_info["size"],
                            direction=input_slice_info["direction"],
                            coords=target_grid["coords"],
                        )

                        def _slice_indices(value, length):
                            if isinstance(value, (int, np.integer)):
                                return np.asarray([int(value)])
                            if isinstance(value, slice):
                                return np.arange(length)[value]
                            return np.asarray(value).reshape(-1)

                        source_slice_indexers = tuple(
                            getattr(native_input_slice, axis) for axis in two_ports.axes
                        )
                        target_slice_indexers = tuple(
                            getattr(target_input_slice, axis) for axis in self.axes
                        )
                        src_transverse_coords = tuple(
                            source_grid["coords"][axis][
                                _slice_indices(
                                    source_slice_indexers[axis], source_shape[axis]
                                )
                            ]
                            for axis in transverse_axes
                        )
                        dst_transverse_coords = tuple(
                            target_grid["coords"][axis][
                                _slice_indices(
                                    target_slice_indexers[axis], target_shape[axis]
                                )
                            ]
                            for axis in transverse_axes
                        )
                        for profile_index in (1, 2):
                            mode_field = profile[profile_index]
                            if mode_field.ndim < 2 or tuple(
                                mode_field.shape[-2:]
                            ) != tuple(len(coords) for coords in src_transverse_coords):
                                continue
                            was_numpy = isinstance(mode_field, np.ndarray)
                            resampled_mode = two_ports.resample_map_between_coords(
                                mode_field,
                                src_coords=src_transverse_coords,
                                dst_coords=dst_transverse_coords,
                            )
                            if was_numpy:
                                resampled_mode = resampled_mode.detach().cpu().numpy()
                            source_profiles[key][profile_index] = resampled_mode
            # source_profiles["require_sim"] = require_sim
            # input_SCALE, fields, source_profiles = _norm_run(power_scales=input_scale)

            if plot:
                mode = list(fields.keys())[0][1]
                plot_field = Fz * list(input_scale.values())[0]
                plot_eps = two_ports.epsilon_map
                plot_input_slice = two_ports.port_monitor_slices.get(
                    input_slice_name, input_slice
                )
                plot_output_slice = two_ports.port_monitor_slices.get(
                    input_slice_name, output_slice
                )
                native_grid = two_ports.grid_info_dict.get("epsilon_map")
                export_grid = two_ports.grid_info_dict.get("export_epsilon_map")
                if native_grid is not None and export_grid is not None:
                    native_coords = native_grid["coords"]
                    export_coords = export_grid["coords"]
                    native_shape = tuple(len(axis) for axis in native_coords)
                    export_shape = tuple(len(axis) for axis in export_coords)
                    if (
                        tuple(plot_field.shape[-2:]) == native_shape
                        and native_shape != export_shape
                    ):
                        plot_field = two_ports.resample_map_between_coords(
                            plot_field,
                            src_coords=native_coords,
                            dst_coords=export_coords,
                        )
                        plot_eps = two_ports.resample_map_between_coords(
                            torch.as_tensor(
                                plot_eps,
                                device=(
                                    plot_field.device
                                    if isinstance(plot_field, torch.Tensor)
                                    else None
                                ),
                            ),
                            src_coords=native_coords,
                            dst_coords=export_coords,
                        )

                    def _export_slice(native_slice):
                        centers = []
                        sizes = []
                        for axis, axis_slice in enumerate(
                            (native_slice.x, native_slice.y)
                        ):
                            if isinstance(axis_slice, (int, np.integer)):
                                index = int(axis_slice)
                                centers.append(float(native_coords[axis][index]))
                                sizes.append(0.0)
                            else:
                                start = (
                                    0 if axis_slice.start is None else axis_slice.start
                                )
                                stop = (
                                    len(native_coords[axis])
                                    if axis_slice.stop is None
                                    else axis_slice.stop
                                )
                                boundaries = native_grid["boundaries"][axis]
                                centers.append(
                                    float((boundaries[start] + boundaries[stop]) / 2)
                                )
                                sizes.append(
                                    float(boundaries[stop] - boundaries[start])
                                )
                        return two_ports._monitor_slice_on_coords(
                            center=centers,
                            size=sizes,
                            direction=direction,
                            coords=export_coords,
                        )

                    plot_input_slice = _export_slice(native_input_slice)
                    plot_output_slice = _export_slice(output_slice)
                plot_eps_field(
                    plot_field,
                    mode,
                    plot_eps,
                    zoom_eps_factor=1,
                    filepath=os.path.join(
                        self.sim_cfg["plot_root"],
                        f"{self.config.device.type}_norm-{input_slice_name}.png",
                    ),
                    x_width=two_ports.cell_size[0],
                    y_height=two_ports.cell_size[1],
                    monitors=[(plot_input_slice, "r"), (plot_output_slice, "b")],
                    title=f"|{pol}|^2, Norm run at {input_slice_name}",
                    field_stat="intensity_real",
                )
            if self.port_sources_dict.get(input_slice_name) is not None:
                self.port_sources_dict[input_slice_name].update(source_profiles)
            else:
                self.port_sources_dict[input_slice_name] = source_profiles
        # print(source_profiles)
        # exit(0)
        return source_profiles  # {(wl, mode): [profile, ht_m, et_m, SCALE, require_sim], ...}

    def _fill_design_region_map(
        self, base_map, values, region_name=None, region_masks=None
    ):
        if region_name is None:
            if len(self.design_region_cfgs) != 1:
                raise ValueError(
                    "region_name is required when the device has multiple design regions."
                )
            region_name = next(iter(self.design_region_cfgs))

        region_masks = region_masks or self.design_region_masks
        region_mask = region_masks[region_name]
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        filled_map = copy.deepcopy(base_map)
        filled_map[region_mask] = np.asarray(values).reshape(-1)
        return filled_map

    def _fill_design_region_map_torch(
        self, base_map, values, region_name=None, region_masks=None
    ):
        if region_name is None:
            if len(self.design_region_cfgs) != 1:
                raise ValueError(
                    "region_name is required when the device has multiple design regions."
                )
            region_name = next(iter(self.design_region_cfgs))

        region_masks = region_masks or self.design_region_masks
        region_mask = region_masks[region_name]
        if not isinstance(base_map, torch.Tensor):
            base_map = torch.as_tensor(base_map, device=self.device)
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(
                values, dtype=base_map.dtype, device=base_map.device
            )
        filled_map = base_map.clone()
        filled_map[region_mask] = values.reshape(-1)
        return filled_map

    def _ambient_temperature(self) -> float:
        heat_cfg = self._heat_sim_cfg()
        dirichlet_bc = heat_cfg.get("dirichlet_bc", {})
        if isinstance(dirichlet_bc, dict) and dirichlet_bc:
            first_value = next(iter(dirichlet_bc.values()))
            if np.isscalar(first_value):
                return float(first_value)
        return 300.0

    def _uniform_heat_dirichlet_temperature(self) -> float | None:
        dirichlet_bc = self._normalize_heat_boundary_dict(
            self._heat_dirichlet_bc_default()
        )
        if not dirichlet_bc:
            return None

        values = [float(value) for value in dirichlet_bc.values()]
        first_value = values[0]
        if all(np.isclose(value, first_value) for value in values[1:]):
            return first_value
        return None

    def _heat_source_cfgs_with_currents(self, currents: dict[str, float] | None):
        runtime_cfgs = copy.deepcopy(self.heat_source_cfgs)
        currents = currents or {}
        for source_name, current in currents.items():
            if source_name not in runtime_cfgs:
                raise KeyError(f"Heat source '{source_name}' is not defined.")
            runtime_cfgs[source_name]["current"] = float(current)
        for source_name, cfg in runtime_cfgs.items():
            if source_name not in currents and "current" in cfg:
                cfg["current"] = 0.0
        return runtime_cfgs

    def _runtime_heat_source_cache_tolerances(self) -> tuple[float, float]:
        heat_cfg = self._heat_sim_cfg()
        atol = float(
            heat_cfg.get("q_map_cache_atol", heat_cfg.get("current_cache_atol", 1e-12))
        )
        rtol = float(
            heat_cfg.get("q_map_cache_rtol", heat_cfg.get("current_cache_rtol", 1e-6))
        )
        return atol, rtol

    def _runtime_heat_source_cache_max_entries(self) -> int:
        heat_cfg = self._heat_sim_cfg()
        return max(1, int(heat_cfg.get("q_map_cache_size", 8)))

    def _normalized_runtime_currents(
        self, currents: dict[str, float] | None
    ) -> dict[str, float]:
        requested_currents = currents or {}
        normalized_currents = {}
        for source_name, cfg in self.heat_source_cfgs.items():
            if source_name in requested_currents:
                normalized_currents[source_name] = float(
                    requested_currents[source_name]
                )
            elif "current" in cfg:
                normalized_currents[source_name] = 0.0
        return normalized_currents

    def _runtime_heat_source_currents_close(
        self,
        lhs: dict[str, float],
        rhs: dict[str, float],
        *,
        atol: float,
        rtol: float,
    ) -> bool:
        source_names = set(lhs) | set(rhs)
        for source_name in source_names:
            if not np.isclose(
                float(lhs.get(source_name, 0.0)),
                float(rhs.get(source_name, 0.0)),
                atol=atol,
                rtol=rtol,
            ):
                return False
        return True

    def _get_cached_runtime_heat_source_map(
        self, currents: dict[str, float]
    ) -> torch.Tensor | None:
        atol, rtol = self._runtime_heat_source_cache_tolerances()
        for entry in self._runtime_heat_source_map_cache:
            if self._runtime_heat_source_currents_close(
                currents,
                entry["currents"],
                atol=atol,
                rtol=rtol,
            ):
                return entry["q_map"]
        return None

    def _store_cached_runtime_heat_source_map(
        self,
        currents: dict[str, float],
        q_map: torch.Tensor,
    ) -> torch.Tensor:
        self._runtime_heat_source_map_cache = [
            entry
            for entry in self._runtime_heat_source_map_cache
            if entry["currents"] != currents
        ]
        self._runtime_heat_source_map_cache.append(
            {
                "currents": dict(currents),
                "q_map": q_map,
            }
        )
        max_entries = self._runtime_heat_source_cache_max_entries()
        if len(self._runtime_heat_source_map_cache) > max_entries:
            self._runtime_heat_source_map_cache = self._runtime_heat_source_map_cache[
                -max_entries:
            ]
        return q_map

    def build_runtime_heat_source_map(self, currents: dict[str, float] | None):
        normalized_currents = self._normalized_runtime_currents(currents)
        cached_q_map = self._get_cached_runtime_heat_source_map(normalized_currents)
        if cached_q_map is not None:
            return cached_q_map

        runtime_cfgs = self._heat_source_cfgs_with_currents(currents)
        built_sources = self.build_heat_source_maps(
            heat_source_cfgs=runtime_cfgs,
            combine=True,
        )
        if not built_sources:
            if self.conductivity_map is None:
                self.build_thermal_property_maps()
            q_map = torch.zeros(
                tuple(int(v) for v in self.thermal_grid_shape),
                dtype=torch.float32,
                device=self.device,
            )
            return self._store_cached_runtime_heat_source_map(
                normalized_currents, q_map
            )
        q_map = torch.as_tensor(
            self.heat_source_map,
            dtype=torch.float32,
            device=self.device,
        )
        return self._store_cached_runtime_heat_source_map(normalized_currents, q_map)

    def interpolate_map_between_grids(
        self,
        values,
        target_shape: Sequence[int],
    ):
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values, dtype=torch.float64, device=self.device)
        target_shape = tuple(int(v) for v in target_shape)
        if tuple(values.shape) == target_shape:
            return values
        if values.ndim == 2:
            resized = F.interpolate(
                values[None, None],
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )
            return resized[0, 0]
        if values.ndim == 3:
            resized = F.interpolate(
                values[None, None],
                size=target_shape,
                mode="trilinear",
                align_corners=False,
            )
            return resized[0, 0]
        raise ValueError(f"Unsupported map dimension {values.ndim}")

    def apply_thermo_optic_perturbation(
        self,
        permittivity,
        temperature_map,
        thermo_optic_coeff_map,
    ):
        if not isinstance(permittivity, torch.Tensor):
            permittivity = torch.as_tensor(permittivity, device=self.device)
        if not isinstance(temperature_map, torch.Tensor):
            temperature_map = torch.as_tensor(
                temperature_map, dtype=permittivity.dtype, device=permittivity.device
            )
        if not isinstance(thermo_optic_coeff_map, torch.Tensor):
            thermo_optic_coeff_map = torch.as_tensor(
                thermo_optic_coeff_map,
                dtype=permittivity.dtype,
                device=permittivity.device,
            )
        ambient_temperature = self._ambient_temperature()
        delta_t = temperature_map - ambient_temperature
        use_complex_branch = torch.is_complex(permittivity) or torch.any(
            permittivity < 0
        )
        if use_complex_branch:
            complex_dtype = torch.complex64
            permittivity_complex = permittivity.to(dtype=complex_dtype)
            refractive_index = torch.sqrt(permittivity_complex)
            delta_t = delta_t.to(dtype=complex_dtype)
            thermo_optic_coeff_map = thermo_optic_coeff_map.to(dtype=complex_dtype)
            return torch.square(refractive_index + thermo_optic_coeff_map * delta_t)

        refractive_index = torch.sqrt(torch.clamp(permittivity, min=0.0))
        return torch.square(refractive_index + thermo_optic_coeff_map * delta_t)

    def build_runtime_thermal_state(
        self,
        permittivity,
        currents: dict[str, float] | None,
        *,
        runtime_material_maps: dict | None = None,
    ):
        requires_temp_grad = self._heat_requires_temp_grad_default()
        currents = currents or {}
        current_values = [float(value) for value in currents.values()]
        all_zero = len(current_values) == 0 or all(
            abs(value) == 0.0 for value in current_values
        )
        zero_source_temperature = self._uniform_heat_dirichlet_temperature()

        if runtime_material_maps is None:
            if self.conductivity_map is None:
                self.build_thermal_property_maps()
            if not self._electrical_conductivity_map_built:
                self.build_electrical_conductivity_map()
            conductivity_map = torch.as_tensor(
                self.conductivity_map, dtype=torch.float32, device=self.device
            )
            electrical_conductivity_map = None
            if self.electrical_conductivity_map is not None:
                electrical_conductivity_map = torch.as_tensor(
                    self.electrical_conductivity_map,
                    dtype=torch.float32,
                    device=self.device,
                )
            heat_capacity_map = None
            if self.heat_capacity_map is not None:
                heat_capacity_map = torch.as_tensor(
                    self.heat_capacity_map,
                    dtype=torch.float32,
                    device=self.device,
                )
            thermo_optic_coeff_map = torch.as_tensor(
                self.thermo_optic_coeff_map, dtype=torch.float32, device=self.device
            )
        else:
            conductivity_map = runtime_material_maps["conductivity"].to(
                device=self.device,
                dtype=torch.float32,
            )
            electrical_conductivity_map = runtime_material_maps.get(
                "electrical_conductivity"
            )
            if electrical_conductivity_map is not None:
                electrical_conductivity_map = electrical_conductivity_map.to(
                    device=self.device,
                    dtype=torch.float32,
                )
            heat_capacity_map = runtime_material_maps["heat_capacity"]
            if heat_capacity_map is not None:
                heat_capacity_map = heat_capacity_map.to(
                    device=self.device,
                    dtype=torch.float32,
                )
            thermo_optic_coeff_map = runtime_material_maps["thermo_optic_coeff"].to(
                device=self.device,
                dtype=torch.float32,
            )

        if all_zero and zero_source_temperature is not None:
            q_map = torch.zeros_like(conductivity_map)
            temperature_map = torch.full_like(
                conductivity_map,
                zero_source_temperature,
            )
        else:
            q_map = self.build_runtime_heat_source_map(currents).to(
                conductivity_map.device
            )
            temperature_map = self.solve_heat(
                k_map=conductivity_map,
                q_map=q_map,
            )
            if not isinstance(temperature_map, torch.Tensor):
                temperature_map = torch.as_tensor(
                    temperature_map,
                    dtype=conductivity_map.dtype,
                    device=conductivity_map.device,
                )

        if not requires_temp_grad and isinstance(temperature_map, torch.Tensor):
            temperature_map = temperature_map.detach()

        optical_temperature = self.interpolate_map_between_grids(
            temperature_map,
            tuple(int(v) for v in permittivity.shape),
        )
        optical_temperature = self.resample_map_between_coords(
            values=temperature_map,
            src_coords=self.thermal_coords,
            dst_coords=self.grid_info_dict["epsilon_map"]["coords"],
        )
        if not isinstance(thermo_optic_coeff_map, torch.Tensor):
            thermo_optic_coeff_map = torch.as_tensor(
                thermo_optic_coeff_map,
                dtype=torch.float32,
                device=self.device,
            )

        # optical_thermo_optic_coeff = self.interpolate_map_between_grids(
        #     thermo_optic_coeff_map,
        #     tuple(int(v) for v in permittivity.shape),
        # )
        eps = self.apply_thermo_optic_perturbation(
            permittivity,
            optical_temperature,
            thermo_optic_coeff_map,
        )
        return {
            "q_map": q_map,
            "temperature": temperature_map,
            "optical_temperature": optical_temperature,
            "thermo_optic_coeff": thermo_optic_coeff_map,
            "optical_thermo_optic_coeff": thermo_optic_coeff_map,
            "heat_capacity": heat_capacity_map,
            "conductivity": conductivity_map,
            "electrical_conductivity": electrical_conductivity_map,
            "eps": eps,
        }

    def build_runtime_fdtdx_state(
        self,
        permittivity,
        currents: dict[str, float] | None,
        *,
        runtime_material_maps: dict | None = None,
    ):
        base_state = self.build_runtime_thermal_state(
            permittivity,
            currents,
            runtime_material_maps=runtime_material_maps,
        )
        field_grid_metadata = self._build_fdtdx_native_grid_metadata()
        native_grid = field_grid_metadata["native"]
        if field_grid_metadata.get("native_equals_export", False):
            base_state["electrical_conductivity"] = self._resample_map_to_fdtdx_native(
                base_state.get("electrical_conductivity"),
                map_name="electrical_conductivity_map",
                native_grid=native_grid,
            )
            base_state["field_grid_metadata"] = field_grid_metadata
            base_state["native_eps"] = base_state["eps"]
            base_state["native_design_region_masks"] = (
                self.fdtdx_native_design_region_masks
            )
            base_state["native_design_region_mask_weights"] = (
                self.fdtdx_native_design_region_mask_weights
            )
            return base_state

        export_grid = field_grid_metadata["export"]
        # print(permittivity.shape)
        native_base_permittivity = self.build_fdtdx_native_permittivity(
            permittivity,
            field_grid_metadata=field_grid_metadata,
        )
        native_optical_temperature = None
        if base_state.get("temperature") is not None:
            native_optical_temperature = self.resample_map_between_coords(
                base_state["temperature"],
                src_coords=self.thermal_coords,
                dst_coords=native_grid["coords"],
            )
        native_optical_thermo_optic_coeff = self.resample_map_between_coords(
            base_state["optical_thermo_optic_coeff"],
            src_coords=export_grid["coords"],
            dst_coords=native_grid["coords"],
        )
        if native_optical_temperature is not None:
            native_eps = self.apply_thermo_optic_perturbation(
                native_base_permittivity,
                native_optical_temperature,
                native_optical_thermo_optic_coeff,
            )
        else:
            native_eps = native_base_permittivity
        native_electrical_conductivity = None
        if base_state.get("electrical_conductivity") is not None:
            native_electrical_conductivity = self._resample_map_to_fdtdx_native(
                base_state["electrical_conductivity"],
                map_name="electrical_conductivity_map",
                native_grid=native_grid,
            )
        base_state.update(
            {
                "field_grid_metadata": field_grid_metadata,
                "native_eps": native_eps,
                "native_optical_temperature": native_optical_temperature,
                "native_optical_thermo_optic_coeff": native_optical_thermo_optic_coeff,
                "native_electrical_conductivity": native_electrical_conductivity,
                "native_design_region_masks": self.fdtdx_native_design_region_masks,
                "native_design_region_mask_weights": self.fdtdx_native_design_region_mask_weights,
                "eps": native_eps,
                "electrical_conductivity": native_electrical_conductivity,
            }
        )
        return base_state

    def get_design_region_property_bounds(
        self,
        region_name: str,
        *,
        value_keys: Sequence[str],
        bg_keys: Sequence[str],
    ) -> tuple[float, float]:
        cfg = self.design_region_cfgs[region_name]
        value = float(_cfg_scalar(cfg, value_keys))
        bg_value = float(_cfg_scalar(cfg, bg_keys, default=value))
        return value, bg_value

    def denormalize_design_region_conductivity(
        self,
        density,
        region_name: str | None = None,
    ):
        if region_name is None:
            if len(self.design_region_cfgs) != 1:
                raise ValueError(
                    "region_name is required when the device has multiple design regions."
                )
            region_name = next(iter(self.design_region_cfgs))
        k_hi, k_lo = self.get_design_region_property_bounds(
            region_name,
            value_keys=_HEAT_PROPERTY_KEYS,
            bg_keys=_HEAT_PROPERTY_BG_KEYS,
        )
        if isinstance(density, torch.Tensor):
            return density * (k_hi - k_lo) + k_lo
        density = np.asarray(density, dtype=np.float32)
        return density * (k_hi - k_lo) + k_lo

    def denormalize_design_region_electrical_conductivity(
        self,
        density,
        region_name: str | None = None,
    ):
        return self.denormalize_design_region_property(
            density,
            value_keys=_ELECTRICAL_CONDUCTIVITY_KEYS,
            bg_keys=_ELECTRICAL_CONDUCTIVITY_BG_KEYS,
            region_name=region_name,
        )

    def denormalize_design_region_property(
        self,
        density,
        *,
        value_keys: Sequence[str],
        bg_keys: Sequence[str],
        region_name: str | None = None,
    ):
        if region_name is None:
            if len(self.design_region_cfgs) != 1:
                raise ValueError(
                    "region_name is required when the device has multiple design regions."
                )
            region_name = next(iter(self.design_region_cfgs))
        prop_hi, prop_lo = self.get_design_region_property_bounds(
            region_name,
            value_keys=value_keys,
            bg_keys=bg_keys,
        )
        if isinstance(density, torch.Tensor):
            return density * (prop_hi - prop_lo) + prop_lo
        density = np.asarray(density, dtype=np.float32)
        return density * (prop_hi - prop_lo) + prop_lo

    def denormalize_design_region_heat_capacity(
        self,
        density,
        region_name: str | None = None,
    ):
        return self.denormalize_design_region_property(
            density,
            value_keys=_HEAT_CAPACITY_KEYS,
            bg_keys=_HEAT_CAPACITY_BG_KEYS,
            region_name=region_name,
        )

    def denormalize_design_region_thermo_optic_coeff(
        self,
        density,
        region_name: str | None = None,
    ):
        return self.denormalize_design_region_property(
            density,
            value_keys=_THERMO_OPTIC_KEYS,
            bg_keys=_THERMO_OPTIC_BG_KEYS,
            region_name=region_name,
        )

    def obtain_eps(self, permittivity: torch.Tensor, region_name: str | None = None):
        ## we need denormalized permittivity for the design region
        return self._fill_design_region_map(
            self.epsilon_map,
            permittivity,
            region_name=region_name,
            region_masks=self.design_region_masks,
        )

    def obtain_conductivity(
        self,
        conductivity: torch.Tensor,
        region_name: str | None = None,
    ):
        if self.conductivity_map is None:
            raise ValueError(
                "conductivity_map is unavailable. Add thermal conductivity values "
                "to the device configs and pass k_bg when constructing the device."
            )
        return self._fill_design_region_map(
            self.conductivity_map,
            conductivity,
            region_name=region_name,
            region_masks=self.thermal_design_region_masks or self.design_region_masks,
        )

    def obtain_electrical_conductivity(
        self,
        electrical_conductivity: torch.Tensor,
        region_name: str | None = None,
    ):
        if self.electrical_conductivity_map is None:
            raise ValueError(
                "electrical_conductivity_map is unavailable. Add electrical conductivity values "
                "to the device configs or use conductive materials in the geometry."
            )
        return self.obtain_material_property(
            "electrical_conductivity",
            electrical_conductivity,
            region_name=region_name,
        )

    def obtain_material_property(
        self,
        property_name: str,
        values: torch.Tensor,
        region_name: str | None = None,
    ):
        property_map = self.get_material_property_map(property_name)
        if property_map is None:
            raise ValueError(
                f"{property_name}_map is unavailable. Add the corresponding property "
                "values to the device configs and construct the device with a "
                "background value for that property."
            )
        return self._fill_design_region_map(
            property_map,
            values,
            region_name=region_name,
            region_masks=self.thermal_design_region_masks or self.design_region_masks,
        )

    def obtain_heat_capacity(
        self,
        heat_capacity: torch.Tensor,
        region_name: str | None = None,
    ):
        if self.heat_capacity_map is None:
            raise ValueError(
                "heat_capacity_map is unavailable. Add heat capacity values to the "
                "device configs and pass heat_capacity_bg when constructing the device."
            )
        return self.obtain_material_property(
            "heat_capacity",
            heat_capacity,
            region_name=region_name,
        )

    def obtain_thermo_optic_coeff(
        self,
        thermo_optic_coeff: torch.Tensor,
        region_name: str | None = None,
    ):
        if self.thermo_optic_coeff_map is None:
            raise ValueError(
                "thermo_optic_coeff_map is unavailable. Add thermo-optic values to "
                "the device configs and pass thermo_optic_coeff_bg when constructing "
                "the device."
            )
        return self.obtain_material_property(
            "thermo_optic_coeff",
            thermo_optic_coeff,
            region_name=region_name,
        )

    def copy(self, resolution: int | None = None, **overrides):
        if not hasattr(self, "_constructor_kwargs"):
            raise RuntimeError(
                f"{self.__class__.__name__} was not constructed through a captured "
                "constructor; cannot replay constructor kwargs."
            )

        kwargs = copy.deepcopy(self._constructor_kwargs)

        sim_cfg = copy.deepcopy(kwargs.get("sim_cfg", self.sim_cfg))
        if resolution is not None:
            sim_cfg["resolution"] = resolution
        for name, value in overrides.items():
            if name in kwargs:
                kwargs[name] = copy.deepcopy(value)
            else:
                sim_cfg[name] = copy.deepcopy(value)

        kwargs["sim_cfg"] = sim_cfg

        return self.__class__(**kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}(size={self.cell_size}, Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}, grid_step={self.grid_step}, eps_bg={self.eps_bg})"
