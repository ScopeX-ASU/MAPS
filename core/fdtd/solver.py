"""
Torch autodiff bridge for fdtdx-backed FDTD solves.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from itertools import count, product
from pathlib import Path
from typing import Any, Literal, Sequence, Tuple

try:
    import fdtdx
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
from fdtdx import constants
from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.objects.sources.profile import GaussianPulseProfile, SingleFrequencyProfile
from torch import Tensor, nn

try:
    from thirdparty.fdtdx import (
        CurrentSourceBlock,
        DistributedCurrentSource,
        FactoredChebyshevCurrentSource,
        discrete_source_spectrum,
    )
except:
    CurrentSourceBlock = None
    DistributedCurrentSource = None
    FactoredChebyshevCurrentSource = None
    discrete_source_spectrum = None
    logging.warning(
        "thirdparty.fdtdx.DistributedCurrentSource could not be imported. Adjoint fdtd may not work."
    )

from core.env_args import get_env_arg
from core.utils import (
    _field_component_parity,
    _jax_to_torch,
    _torch_to_jax,
    get_wrap_padding_axes,
    pad_fields_for_boundaries,
    unfold_phasor,
    yee_to_colocate_interpolate,
)
from thirdparty.fdtdx.fdtd.symmetry import unfold_grid

ArrayBackend = Literal["jax", "numpy", "torch"]


_DECAY_BY = 1e-3
_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_E_COMPONENTS = {"Ex": 0, "Ey": 1, "Ez": 2}
_H_COMPONENTS = {"Hx": 0, "Hy": 1, "Hz": 2}
# SCALE = 1e-11
## this scale=1e-16 will make dL/deps_r  on the order of 1e-4.
## after backprop to dL/dknots =eps_r * dL/deps_r = 10 * 1e-4 = 1e-3.
## the gradient is roughly 1/1000 of the parameter range (0, 1), so it is a healthy scale of gradients.
# SCALE = 1e-15
# SCALE = 3e-12
SCALE = 5e-11
_MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM = get_env_arg(
    "MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM"
)
_WRITE_ADJOINT_DEBUG_PLOTS = get_env_arg("MAPS_FDTDX_ADJOINT_DEBUG")
_MAPS_ENABLE_BROADBAND_ADJOINT = get_env_arg("MAPS_ENABLE_BROADBAND_ADJOINT")
_BROADBAND_ADJOINT_METHOD = get_env_arg("MAPS_BROADBAND_ADJOINT_METHOD")
_BROADBAND_ADJOINT_PROFILE_ORDER = get_env_arg("MAPS_BROADBAND_ADJOINT_PROFILE_ORDER")
_BROADBAND_ADJOINT_PROFILE_RTOL = get_env_arg("MAPS_BROADBAND_ADJOINT_PROFILE_RTOL")
_BROADBAND_ADJOINT_MIN_OVERLAP = get_env_arg("MAPS_BROADBAND_ADJOINT_MIN_OVERLAP")
_BROADBAND_ADJOINT_MAX_COEFFICIENT_NORM = get_env_arg(
    "MAPS_BROADBAND_ADJOINT_MAX_COEFFICIENT_NORM"
)
_MAPS_SPARSE_PHASOR = get_env_arg("MAPS_SPARSE_PHASOR")
_MAPS_COMPARE_SPARSE_DENSE = get_env_arg("MAPS_COMPARE_SPARSE_DENSE")
_MAPS_WRITE_SPARSE_DENSE_PLOTS = get_env_arg("MAPS_WRITE_SPARSE_DENSE_PLOTS")
_MAPS_SPARSE_ADJOINT_SOURCE = get_env_arg("MAPS_SPARSE_ADJOINT_SOURCE")
_ADJOINT_SOURCE_YEE_HALO = 1
_MAPS_ENABLE_PHASOR_H = get_env_arg("MAPS_ENABLE_PHASOR_H")
_MAPS_PHASOR_H_NATIVE_SLAB_CELLS = get_env_arg("MAPS_PHASOR_H_NATIVE_SLAB_CELLS")
if not _MAPS_ENABLE_PHASOR_H:
    logging.warning(
        "MAPS_ENABLE_PHASOR_H is disabled. H phasors are analytically derived from the E phasors."
        + " H fields on the source plane are still recorded by an extra PhasorDetector."
        + " This can potentially speedup FDTD and save memory."
    )
if _MAPS_ENABLE_BROADBAND_ADJOINT:
    logging.warning(
        "Experimental broadband adjoints are enabled. Smooth current-profile fits "
        "use discrete spectrum and amplification checks with per-frequency fallback. "
        "Validate new device/objective classes against independent adjoints."
    )
_ADJOINT_DEBUG_DIR = Path(get_env_arg("MAPS_FDTDX_ADJOINT_DEBUG_DIR"))
_ADJOINT_DEBUG_COUNTER = count()


def unfold_array(
    arr: torch.Tensor,
    symmetry: tuple[int, int, int],
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    signs: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Mirror-and-concatenate a spatial array along each symmetric axis.

    For each physical axis a with symmetry[a] != 0:
      1) flip arr along spatial_axes[a]
      2) optionally multiply the mirrored half by signs[a]
      3) concatenate [mirror, arr] along that axis

    This is differentiable as long as `arr` and `signs[a]` are torch tensors.
    """
    if symmetry == (0, 0, 0):
        return arr

    out = arr
    for a in range(3):
        if symmetry[a] == 0:
            continue

        ax = spatial_axes[a]
        sign = 1.0 if signs is None else signs.get(a, 1.0)

        mirror = torch.flip(out, dims=(ax,)) * sign
        out = torch.cat([mirror, out], dim=ax)

    return out


def _object_partial_grid_bounds(
    obj: Any,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    partial_shape_raw = getattr(obj, "partial_grid_shape", ())
    if (
        isinstance(partial_shape_raw, tuple)
        and len(partial_shape_raw) == 3
        and all(v is not None for v in partial_shape_raw)
    ):
        partial_shape = tuple(int(v) for v in partial_shape_raw)
    else:
        profile = getattr(obj, "electric_profiles", None)
        if profile is None:
            profile = getattr(obj, "magnetic_profiles", None)
        if profile is None:
            raise ValueError(
                f"Expected a concrete 3D partial_grid_shape, got {partial_shape_raw}"
            )
        profile_shape = tuple(int(v) for v in profile.shape)
        if len(profile_shape) < 5:
            raise ValueError(
                "Could not infer a 3D object shape from source profiles with shape "
                f"{profile_shape}"
            )
        partial_shape = profile_shape[-3:]
    return tuple((0, dim) for dim in partial_shape)


def _object_grid_slice_bounds(
    obj: Any,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    grid_slice_tuple = getattr(obj, "grid_slice_tuple", None)
    if grid_slice_tuple is None:
        raise ValueError(f"Object {obj!r} does not expose a grid_slice_tuple.")
    bounds = []
    for axis_bounds in grid_slice_tuple:
        if axis_bounds is None or len(axis_bounds) != 2:
            raise ValueError(
                f"Invalid grid_slice_tuple entry {axis_bounds!r} for {obj!r}"
            )
        start, stop = axis_bounds
        if start is None or stop is None:
            raise ValueError(
                f"Unresolved grid_slice_tuple entry {axis_bounds!r} for {obj!r}"
            )
        bounds.append((int(start), int(stop)))
    return tuple(bounds)


def _crop_cotangents_to_monitor(
    cotangents: Tensor,
    monitor: Any,
    config: Any,
) -> Tensor:
    symmetry = tuple(getattr(config, "symmetry", (0, 0, 0)))
    reduced_shape = tuple(
        stop - start for start, stop in _object_grid_slice_bounds(monitor)
    )
    unfolded_shape = tuple(
        length * (2 if symmetry[axis] else 1)
        for axis, length in enumerate(reduced_shape)
    )
    bounds = _unfolded_detector_spatial_slices(
        monitor,
        cotangents,
        cotangents.shape[2:],
        config,
        detector_spatial_shape=unfolded_shape,
    )
    return cotangents[
        (
            slice(None),
            slice(None),
            *bounds,
        )
    ]


def _original_grid_bounds(
    monitor: Any,
    full_shape: Sequence[int],
    config: Any,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    unreduced = getattr(monitor, "_unreduced_grid_slice_tuple", None)
    if unreduced is None:
        return _object_grid_slice_bounds(monitor)
    return tuple(
        (
            (
                int(full_shape[axis]) // 2 + int(bounds[0]),
                int(full_shape[axis]) // 2 + int(bounds[1]),
            )
            if config.symmetry[axis]
            else (int(bounds[0]), int(bounds[1]))
        )
        for axis, bounds in enumerate(unreduced)
    )  # type: ignore[return-value]


def _fold_symmetric_cotangents(
    cotangents: Tensor,
    components: Sequence[str],
    symmetry: Sequence[int],
) -> Tensor:
    folded = cotangents
    for axis, wall in enumerate(symmetry):
        if wall == 0:
            continue
        spatial_axis = axis + 2
        size = folded.shape[spatial_axis]
        if size % 2:
            continue
        half = size // 2
        lower = folded.narrow(spatial_axis, 0, half)
        upper = folded.narrow(spatial_axis, half, half)
        signs = torch.tensor(
            [
                _field_component_parity(
                    "E" if component.startswith("E") else "H",
                    component,
                    axis,
                    wall,
                )
                for component in components
            ],
            dtype=folded.real.dtype,
            device=folded.device,
        ).reshape((1, len(components)) + (1,) * 3)
        folded = upper + torch.flip(lower, dims=(spatial_axis,)) * signs
    return folded


def _embed_native_phasor(
    phasor: Tensor,
    monitor: Any,
    spatial_shape: Sequence[int],
    config: Any,
    placement_monitor: Any | None = None,
) -> Tensor:
    unfolded = unfold_phasor(
        phasor,
        monitor=monitor,
        components=tuple(getattr(monitor, "components", _COMPONENTS)),
        config=config,
    )
    spatial_slices = _unfolded_detector_spatial_slices(
        monitor if placement_monitor is None else placement_monitor,
        unfolded,
        spatial_shape,
        config,
    )
    embedded = torch.zeros(
        (phasor.shape[0], phasor.shape[1], *spatial_shape),
        dtype=phasor.dtype,
        device=phasor.device,
    )
    embedded[
        (
            slice(None),
            slice(None),
            *spatial_slices,
        )
    ] = unfolded
    return embedded


def phasor_detector_components() -> tuple[str, ...]:
    return _COMPONENTS if _MAPS_ENABLE_PHASOR_H else ("Ex", "Ey", "Ez")


def extract_phasor_detector_states(
    arrays: ArrayContainer,
    *,
    exclude_names: Sequence[str] = (),
) -> dict[str, Tensor]:
    """Extract recorded phasor states from an FDTD-X array container.

    Args:
        arrays: FDTD-X arrays returned by ``run_fdtd``.
        exclude_names: Detector names to omit from the returned mapping.

    Returns:
        Mapping from detector name to Torch phasor tensor with shape
        ``(frequency, component, x, y, z)``. Detector states without a
        ``"phasor"`` entry, such as energy decay detectors, are ignored.
    """
    excluded = set(exclude_names)
    phasors = {}
    for name, state in arrays.detector_states.items():
        if name in excluded or "phasor" not in state:
            continue
        phasor = state["phasor"]
        if getattr(phasor, "ndim", 0) == 6:
            phasor = phasor[0]
        phasors[name] = _jax_to_torch(phasor)
    return phasors


def _unfolded_detector_spatial_slices(
    detector: Any,
    detector_fields: Tensor,
    full_shape: Sequence[int],
    config: Any,
    *,
    detector_spatial_shape: Sequence[int] | None = None,
) -> tuple[slice, slice, slice]:
    """Map a detector's reduced-domain slice into unfolded coordinates."""
    grid_slice = getattr(detector, "grid_slice", None)
    if grid_slice is None:
        raise ValueError(f"Detector {detector!r} has no placed grid slice.")

    symmetry = tuple(getattr(config, "symmetry", (0, 0, 0)))
    spatial_slices = []
    for axis, axis_slice in enumerate(grid_slice):
        start = int(axis_slice.start)
        stop = int(axis_slice.stop)
        base_length = stop - start
        length = int((detector_spatial_shape or detector_fields.shape[2:])[axis])
        if symmetry[axis] == 0:
            unfolded_start = start
        else:
            midpoint = int(full_shape[axis]) // 2
            if start == 0 and length == 2 * base_length:
                unfolded_start = midpoint - base_length
            else:
                unfolded_start = midpoint + start
        unfolded_stop = unfolded_start + length
        if unfolded_start < 0 or unfolded_stop > int(full_shape[axis]):
            raise ValueError(
                f"Detector {detector.name!r} maps outside unfolded shape on axis {axis}: "
                f"{unfolded_start}:{unfolded_stop}, shape={tuple(full_shape)}"
            )
        spatial_slices.append(slice(unfolded_start, unfolded_stop))
    return tuple(spatial_slices)  # type: ignore[return-value]


def _sparse_phasors_to_dense(
    phasors: dict[str, Tensor],
    *,
    objects: ObjectContainer,
    monitor: Any,
    spatial_shape: Sequence[int],
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Scatter localized detector phasors into the dense autograd contract.

    Args:
        phasors: Mapping from placed detector name to a complex tensor shaped
            ``[Nfreq, Ncomponent_local, nx, ny, nz]``.
        objects: Placed FDTD-X objects containing detector component and grid
            metadata for every entry in ``phasors``.
        monitor: Primary detector or detector collection. Its wave metadata
            defines ``Nfreq``.
        spatial_shape: Full native simulation shape ``[Nx, Ny, Nz]``.
        config: FDTD-X configuration containing the active PEC/PMC symmetry.
        device: Compatibility argument retained by the scatter interface. The
            recorded detector state determines the actual output device.
        dtype: Compatibility argument retained by the scatter interface. The
            recorded detector state determines the actual output dtype.

    Returns:
        Complex tensor shaped ``[Nfreq, 6, Nx, Ny, Nz]`` in component order
        ``(Ex, Ey, Ez, Hx, Hy, Hz)``. Cells/components not covered by a sparse
        detector are zero. Symmetry-reduced detector values must be unfolded
        before this helper is called.

    Raises:
        ValueError: If no phasor was recorded, detector metadata and tensor
            shapes disagree, or overlapping detectors contain different values.
        KeyError: If a recorded detector is absent from ``objects``.

    Notes:
        The returned tensor intentionally preserves the dense forward/autograd
        interface; sparse mode changes FDTD-X recording cost, not downstream
        tensor rank, axis order, or device semantics.

        Overlapping detectors are allowed only when they record identical values
        for the same frequency, component, and cell. Values are not summed.
    """
    monitors = monitor if isinstance(monitor, (tuple, list)) else (monitor,)
    wave_characters = (
        tuple(getattr(monitors[0], "wave_characters", ())) if monitors else ()
    )
    n_freq = len(wave_characters) or max(
        (phasor.shape[0] for phasor in phasors.values()), default=1
    )
    first_phasor = next(iter(phasors.values()), None)
    if first_phasor is None:
        raise ValueError("No localized phasor states were recorded.")
    device = first_phasor.device
    dtype = first_phasor.dtype
    dense = torch.zeros(
        (n_freq, len(_COMPONENTS), *tuple(int(size) for size in spatial_shape)),
        dtype=dtype,
        device=device,
    )
    covered = torch.zeros(dense.shape, dtype=torch.bool, device=device)
    for detector_name, phasor in phasors.items():
        if phasor.ndim != 5:
            continue
        if detector_name not in objects:
            raise KeyError(f"Recorded detector {detector_name!r} is not in objects.")
        detector = objects[detector_name]
        components = tuple(getattr(detector, "components", ()))
        if not components:
            continue
        if phasor.shape[0] != n_freq:
            raise ValueError(
                f"Detector {detector_name!r} recorded {phasor.shape[0]} frequencies; "
                f"expected {n_freq}."
            )
        if phasor.shape[1] != len(components):
            raise ValueError(
                f"Detector {detector_name!r} component axis has size "
                f"{phasor.shape[1]}, but metadata lists {len(components)} components."
            )
        spatial_slices = _unfolded_detector_spatial_slices(
            detector, phasor, spatial_shape, config
        )
        mapped_shape = tuple(
            axis_slice.stop - axis_slice.start for axis_slice in spatial_slices
        )
        if mapped_shape != tuple(phasor.shape[2:]):
            raise ValueError(
                f"Detector {detector_name!r} maps to {mapped_shape}, but its phasor "
                f"has spatial shape {tuple(phasor.shape[2:])}."
            )
        for component_index, component in enumerate(components):
            if component not in _COMPONENTS:
                raise ValueError(
                    f"Detector {detector_name!r} has unsupported component {component!r}."
                )
            dense_index = (
                slice(None),
                _COMPONENTS.index(component),
                *spatial_slices,
            )
            target = dense[dense_index]
            target_covered = covered[dense_index]
            incoming = phasor[:, component_index]
            if bool(target_covered.any()) and not torch.equal(
                target[target_covered], incoming[target_covered]
            ):
                difference = (target[target_covered] - incoming[target_covered]).abs()
                raise ValueError(
                    "Overlapping sparse phasor detectors disagree: "
                    f"detector={detector_name!r}, component={component!r}, "
                    f"max_abs={float(difference.max().detach().cpu()):.6e}."
                )
            target[~target_covered] = incoming[~target_covered]
            target_covered.fill_(True)
    return dense


def _adjoint_phasors_to_dense(
    arrays: ArrayContainer,
    *,
    objects: ObjectContainer,
    monitor: Any,
    source_template: Any,
    spatial_shape: Sequence[int],
    config: Any,
    exclude_names: Sequence[str] = (),
) -> Tensor:
    """Assemble localized adjoint recordings into dense full-domain fields.

    Args:
        arrays: FDTD-X arrays after the adjoint time-domain simulation.
        objects: Placed adjoint objects. Sparse mode is expected to contain
            phasor detectors only for design regions, plus the decay detector.
        monitor: Primary localized adjoint detector defining frequency order.
        source_template: Adjoint source used for source-spectrum normalization.
        spatial_shape: Full native simulation shape ``[Nx, Ny, Nz]``.
        config: FDTD-X configuration containing the active symmetry.
        exclude_names: Optional detector names omitted from assembly.

    Returns:
        Source-normalized complex adjoint tensor shaped
        ``[Nfreq, 6, Nx, Ny, Nz]``. Symmetry is unfolded with component parity,
        localized design-region values are scattered to their full-domain
        coordinates, and uncovered cells/components are zero.
    """
    native_phasors = extract_phasor_detector_states(arrays, exclude_names=exclude_names)
    normalized_phasors = {}
    for detector_name, phasor in native_phasors.items():
        detector = objects[detector_name]
        if _MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM:
            phasor, _ = _post_normalize_adjoint_phasor_by_source_spectrum(
                phasor,
                source_template=source_template,
                monitor=detector,
                config=config,
            )
        normalized_phasors[detector_name] = unfold_phasor(
            phasor,
            monitor=detector,
            components=tuple(detector.components),
            config=config,
        )
    return _sparse_phasors_to_dense(
        normalized_phasors,
        objects=objects,
        monitor=monitor,
        spatial_shape=spatial_shape,
        config=config,
        device=next(iter(normalized_phasors.values())).device,
        dtype=next(iter(normalized_phasors.values())).dtype,
    )


def native_h_slab_cells() -> int:
    return _MAPS_PHASOR_H_NATIVE_SLAB_CELLS


def extract_group_grad_compact_torch(
    grad: torch.Tensor,
    group: dict,
):
    """
    Output shape:
        [ngroup_freq, 6, N]
    """

    freq_indices = group["freq_indices"].to(grad.device)
    spatial_indices = group["spatial_indices"].to(grad.device)

    xs = spatial_indices[:, 0]
    ys = spatial_indices[:, 1]
    zs = spatial_indices[:, 2]

    # group_grad = grad.index_select(0, freq_indices)
    # group_grad = group_grad[:, :, xs, ys, zs]
    out = torch.zeros_like(grad[freq_indices])
    out[:, :, xs, ys, zs] = grad[freq_indices][:, :, xs, ys, zs]

    return out


def _slice_to_spatial_indices(monitor_slice: Any) -> torch.Tensor:
    def _axis_indices(axis_value: Any) -> torch.Tensor:
        if isinstance(axis_value, slice):
            start = axis_value.start if axis_value.start is not None else 0
            stop = axis_value.stop
            if stop is None:
                raise ValueError(f"Slice {axis_value} must have a stop value.")
            return torch.arange(start, stop, dtype=torch.long)
        axis_array = np.asarray(axis_value)
        if axis_array.ndim == 0:
            return torch.tensor([int(axis_array.item())], dtype=torch.long)
        return torch.from_numpy(np.unique(axis_array.astype(np.int64).reshape(-1)))

    xs = _axis_indices(monitor_slice[0])
    ys = _axis_indices(monitor_slice[1])
    zs = _axis_indices(monitor_slice[2])
    return torch.cartesian_prod(xs, ys, zs)


def build_adjoint_groups_from_monitor_info(
    grad: torch.Tensor,
    adjoint_group_info: Sequence[dict[str, Any]] | None,
    threshold: float = 0.0,
) -> dict[str, dict[str, torch.Tensor]]:
    if not adjoint_group_info:
        return {}

    support = torch.amax(torch.abs(grad.detach()), dim=1) > threshold
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for entry in adjoint_group_info:
        group_id = entry.get("group_id")
        monitor_slice = entry.get("monitor_slice")
        freq_indices = entry.get("freq_indices", ())
        if group_id is None or monitor_slice is None or len(freq_indices) == 0:
            continue

        # Colocation/interpolation pullbacks spread planar cotangents to the
        # neighboring Yee cells. Retain that halo when grouping by port.
        source_slice = _expand_adjoint_source_slice(monitor_slice, grad.shape[-3:])
        spatial_indices = _slice_to_spatial_indices(source_slice).to(grad.device)
        freq_tensor = torch.as_tensor(
            sorted({int(i) for i in freq_indices}),
            dtype=torch.long,
            device=grad.device,
        )
        if freq_tensor.numel() == 0 or spatial_indices.numel() == 0:
            continue

        xs = spatial_indices[:, 0]
        ys = spatial_indices[:, 1]
        zs = spatial_indices[:, 2]
        active = support.index_select(0, freq_tensor)[:, xs, ys, zs].any(dim=0)
        if active.any():
            spatial_indices = spatial_indices[active]
        if spatial_indices.numel() == 0:
            continue

        groups[str(group_id)] = {
            "freq_indices": freq_tensor,
            "spatial_indices": spatial_indices,
        }
    return groups


def group_adjoint_spatial_support(
    grad: torch.Tensor,
    threshold: float = 0.0,
):
    """
    Group frequencies whose adjoint-gradient spatial supports overlap.

    Parameters
    ----------
    grad:
        Torch tensor with shape [nfreq, 6, X, Y, Z].
    threshold:
        A voxel is active for frequency f if
        max(abs(grad[f, :, x, y, z])) > threshold.

    Returns
    -------
    groups:
        {
            "group_1": {
                "freq_indices": LongTensor [ngroup_freq],
                "spatial_indices": LongTensor [N, 3],  # x, y, z
            },
            ...
        }
    """

    if grad.ndim != 5:
        raise ValueError("Expected grad shape [nfreq, ncomp, X, Y, Z].")

    nfreq, ncomp, X, Y, Z = grad.shape

    if ncomp not in (3, 6):
        raise ValueError("Expected grad.shape[1] in {3, 6}.")

    device = grad.device

    # support[f, x, y, z] = True if any recorded component is active
    support = torch.amax(torch.abs(grad.detach()), dim=1) > threshold
    # shape [nfreq, X, Y, Z]

    # Move only the boolean support to CPU for graph construction.
    # The returned indices are moved back to grad.device.
    support_cpu = support.cpu()

    coord_to_freqs = defaultdict(list)

    for f in range(nfreq):
        coords = torch.nonzero(support_cpu[f], as_tuple=False)
        for coord in coords:
            coord_to_freqs[tuple(coord.tolist())].append(f)

    # Build overlap graph between frequencies
    adjacency = [set() for _ in range(nfreq)]

    for freqs in coord_to_freqs.values():
        if len(freqs) <= 1:
            continue

        for i in freqs:
            for j in freqs:
                if i != j:
                    adjacency[i].add(j)

    # Connected components of frequency graph
    visited = [False] * nfreq
    freq_groups = []

    for start in range(nfreq):
        if visited[start]:
            continue

        queue = deque([start])
        visited[start] = True
        component = []

        while queue:
            f = queue.popleft()
            component.append(f)

            for nb in adjacency[f]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        freq_groups.append(sorted(component))

    # Build final group output
    groups = {}

    for group_id, freqs in enumerate(freq_groups, start=1):
        freq_indices_cpu = torch.tensor(freqs, dtype=torch.long)

        group_support = torch.any(support_cpu[freq_indices_cpu], dim=0)
        spatial_indices_cpu = torch.nonzero(group_support, as_tuple=False)

        groups[f"group_{group_id}"] = {
            "freq_indices": freq_indices_cpu.to(device),
            "spatial_indices": spatial_indices_cpu.to(device),
        }

    return groups


def gaussian_pulse_phasor_spectrum(
    *,
    source_or_profile: Any,
    wave_characters: Sequence[Any],
    config: SimulationConfig,
    detector: Any | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex64,
    include_static_amplitude: bool = False,
    normalize_to_max: bool = False,
) -> Tensor:
    """Discrete complex pulse coefficient seen by an FDTD-X phasor detector.

    The detector demodulates real fields with exp(+i omega t).  For an fdtdx
    Gaussian pulse g(t) cos(omega0 t + phi0), this evaluates the actual
    truncated source samples at exactly the time indices accumulated by the
    detector. It compensates pulse spectrum only; it does not de-mix
    finite-window cross-frequency leakage.
    """
    profile = getattr(source_or_profile, "temporal_profile", source_or_profile)
    if not isinstance(profile, GaussianPulseProfile):
        return torch.ones(
            len(wave_characters),
            dtype=dtype,
            device=device,
        )

    device = torch.device("cpu") if device is None else torch.device(device)
    if discrete_source_spectrum is None:
        raise RuntimeError("MAPS discrete source-spectrum support is unavailable.")
    spectrum = torch.as_tensor(
        discrete_source_spectrum(
            source=source_or_profile,
            wave_characters=wave_characters,
            config=config,
            detector=detector,
            include_static_amplitude=include_static_amplitude,
        ),
        dtype=dtype,
        device=device,
    )
    if normalize_to_max:
        spectrum = spectrum / spectrum.abs().max()
    return spectrum


def _normalize_phasor_by_source_spectrum(
    phasor: Tensor,
    *,
    source_template: Any,
    monitor: Any,
    config: SimulationConfig,
) -> tuple[Tensor, Tensor]:
    spectrum = gaussian_pulse_phasor_spectrum(
        source_or_profile=source_template,
        wave_characters=monitor.wave_characters,
        config=config,
        detector=monitor,
        device=phasor.device,
        dtype=phasor.dtype,
        normalize_to_max=False,
    )
    denom = spectrum.reshape((spectrum.shape[0],) + (1,) * (phasor.ndim - 1))
    return phasor / denom, spectrum


def _post_normalize_adjoint_phasor_by_source_spectrum(
    adjoint_phasor: Tensor,
    *,
    source_template: Any,
    monitor: Any,
    config: SimulationConfig,
) -> tuple[Tensor, Tensor]:
    return _normalize_phasor_by_source_spectrum(
        adjoint_phasor,
        source_template=source_template,
        monitor=monitor,
        config=config,
    )


def plot_adjoint_debug_multifreq(
    *,
    filepath: str | Path,
    phasor_gradients: Tensor | np.ndarray,
    forward_phasors: Tensor | np.ndarray | None = None,
    adjoint_sources: Tensor | np.ndarray | None = None,
    adjoint_phasors: Tensor | np.ndarray,
    grad_eps_by_freq: Tensor | np.ndarray,
    wave_characters: Sequence[Any] | None = None,
    gradient_component: str = "Ey",
    field_component: str = "Ey",
    plane_indices: tuple[int | None, int | None, int | None] = (None, None, None),
    max_freqs: int = 2,
) -> Path:
    """Plot per-wavelength adjoint debug tensors on x-y, x-z, and y-z planes.

    The figure is organized as rows of:
    - phasor-output gradient component
    - injected adjoint source component, when provided
    - adjoint field component
    - per-frequency dL/deps
    for each wavelength, all in one figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _component_index(component: str) -> int:
        if component not in _COMPONENTS:
            raise ValueError(f"Unsupported component {component!r}")
        return _COMPONENTS.index(component)

    def _plane_views(
        arr3: np.ndarray, z_idx: int, y_idx: int, x_idx: int
    ) -> list[np.ndarray]:
        return [
            np.rot90(arr3[:, :, z_idx]),
            np.rot90(arr3[:, y_idx, :]),
            np.rot90(arr3[x_idx, :, :]),
        ]

    def _freq_label(freq_idx: int) -> str:
        if wave_characters is None:
            return f"wl{freq_idx + 1}"
        wave = wave_characters[freq_idx]
        wl_um = wave.get_wavelength() / 1e-6
        return f"{wl_um:.4f} um"

    grad_np = _to_numpy(phasor_gradients)
    fwd_np = None if forward_phasors is None else _to_numpy(forward_phasors)
    src_np = None if adjoint_sources is None else _to_numpy(adjoint_sources)
    adj_np = _to_numpy(adjoint_phasors)
    eps_grad_np = _to_numpy(grad_eps_by_freq)

    if grad_np.ndim != 5:
        raise ValueError(
            f"Expected phasor_gradients shape (freq, comp, Nx, Ny, Nz), got {grad_np.shape}"
        )
    if fwd_np is not None and fwd_np.ndim != 5:
        raise ValueError(
            f"Expected forward_phasors shape (freq, comp, Nx, Ny, Nz), got {fwd_np.shape}"
        )
    if src_np is not None and src_np.ndim != 5:
        raise ValueError(
            f"Expected adjoint_sources shape (freq, comp, Nx, Ny, Nz), got {src_np.shape}"
        )
    if adj_np.ndim != 5:
        raise ValueError(
            f"Expected adjoint_phasors shape (freq, comp, Nx, Ny, Nz), got {adj_np.shape}"
        )
    if eps_grad_np.ndim != 4:
        raise ValueError(
            f"Expected grad_eps_by_freq shape (freq, Nx, Ny, Nz), got {eps_grad_np.shape}"
        )

    nfreq = min(max_freqs, grad_np.shape[0], adj_np.shape[0], eps_grad_np.shape[0])
    if fwd_np is not None:
        nfreq = min(nfreq, fwd_np.shape[0])
    if src_np is not None:
        nfreq = min(nfreq, src_np.shape[0])
    grad_comp_idx = _component_index(gradient_component)
    field_comp_idx = _component_index(field_component)

    quantity_specs = [
        ("phasor grad", grad_np, grad_comp_idx, np.real, "RdBu_r"),
    ]
    if fwd_np is not None:
        # forward_e_norm = np.linalg.norm(fwd_np[:, :3], axis=1, keepdims=True)
        quantity_specs.append(
            ("forward field", fwd_np, field_comp_idx, np.real, "RdBu_r")
        )
        # quantity_specs.append(
        #     ("|forward field|", fwd_np, field_comp_idx, np.abs, "magma")
        # )
        # quantity_specs.append(("|forward E|", forward_e_norm, 0, np.asarray, "magma"))
    if src_np is not None:
        quantity_specs.append(
            ("|adjoint source|", src_np, field_comp_idx, np.abs, "magma")
        )
        # quantity_specs.append(
        #     ("|adjoint source|", src_np, field_comp_idx, np.abs, "magma")
        # )
        # quantity_specs.append(
        #     ("|adjoint source E|", source_e_norm, 0, np.asarray, "magma")
        # )
    # adjoint_e_norm = np.linalg.norm(adj_np[:, :3], axis=1, keepdims=True)
    quantity_specs.extend(
        [
            ("adjoint field", adj_np, field_comp_idx, np.real, "RdBu_r"),
            # ("|adjoint field|", adj_np, field_comp_idx, np.abs, "magma"),
            # ("|adjoint E|", adjoint_e_norm, 0, np.asarray, "magma"),
            ("dL/deps", eps_grad_np[:, None, ...], 0, np.asarray, "RdBu_r"),
        ]
    )
    rows_per_freq = len(quantity_specs)

    fig, axes = plt.subplots(
        nrows=rows_per_freq * nfreq,
        ncols=3,
        figsize=(36, 3 * max(4 * nfreq, 6)),
        squeeze=False,
    )

    for freq_idx in range(nfreq):
        wl_label = _freq_label(freq_idx)
        for quantity_idx, (quantity_name, data, comp_idx, transform, cmap) in enumerate(
            quantity_specs
        ):
            row = rows_per_freq * freq_idx + quantity_idx
            vol = transform(data[freq_idx, comp_idx])
            nx, ny, nz = vol.shape
            z_idx = min(
                nz - 1,
                nz // 2 if plane_indices[0] is None else int(plane_indices[0]),
            )
            y_idx = min(
                ny - 1,
                ny // 2 if plane_indices[1] is None else int(plane_indices[1]),
            )
            x_idx = min(
                nx - 1,
                nx // 2 if plane_indices[2] is None else int(plane_indices[2]),
            )
            views = _plane_views(vol, z_idx=z_idx, y_idx=y_idx, x_idx=x_idx)
            plane_titles = (
                f"x-y, z={z_idx}",
                f"x-z, y={y_idx}",
                f"y-z, x={x_idx}",
            )
            vmax = (
                float(max(np.max(np.abs(view)) for view in views if view.size > 0))
                if any(view.size > 0 for view in views)
                else 1.0
            )
            # vmax = max(vmax, 1e-15)
            for col, (view, plane_title) in enumerate(zip(views, plane_titles)):
                ax = axes[row, col]
                if quantity_name.startswith("|"):
                    im = ax.imshow(view, cmap=cmap, vmin=0, vmax=vmax)
                else:
                    im = ax.imshow(view, cmap=cmap, vmin=-vmax, vmax=vmax)
                ax.set_title(f"{wl_label} | {quantity_name} | {plane_title}")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved adjoint debug plot to {out_path}")
    plt.close(fig)
    return out_path


def plot_sparse_dense_comparison(
    *,
    filepath: str | Path,
    reference: Tensor,
    sparse: Tensor,
    detector_name: str,
    component: str,
    max_freqs: int = 2,
) -> Path:
    """Plot reference and sparse detector phasors on their local support."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_np = reference.detach().cpu().numpy()
    sparse_np = sparse.detach().cpu().numpy()
    nfreq = min(max_freqs, reference_np.shape[0], sparse_np.shape[0])
    fig, axes = plt.subplots(
        nrows=2 * nfreq,
        ncols=3,
        figsize=(12, max(4, 4 * nfreq)),
        squeeze=False,
    )
    for freq_idx in range(nfreq):
        ref = reference_np[freq_idx]
        sparse_value = sparse_np[freq_idx]
        difference = ref - sparse_value
        z_idx = int(np.argmax(np.abs(ref).sum(axis=(0, 1))))
        panels = (
            ("reference |field|", np.abs(ref[:, :, z_idx]), "magma"),
            ("sparse |field|", np.abs(sparse_value[:, :, z_idx]), "magma"),
            ("|difference|", np.abs(difference[:, :, z_idx]), "magma"),
            ("reference phase", np.angle(ref[:, :, z_idx]), "twilight"),
            ("sparse phase", np.angle(sparse_value[:, :, z_idx]), "twilight"),
            ("difference phase", np.angle(difference[:, :, z_idx]), "twilight"),
        )
        for panel_idx, (title, panel, cmap) in enumerate(panels):
            row = 2 * freq_idx + panel_idx // 3
            axis = axes[row, panel_idx % 3]
            axis.imshow(panel.T, origin="lower", cmap=cmap)
            axis.set_title(f"f={freq_idx}, z={z_idx}: {title}")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(f"{detector_name} | {component}")
    fig.tight_layout()
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sparse-dense comparison plot to {out_path}", flush=True)
    return out_path


def _write_adjoint_debug_plot(
    *,
    monitor: Any,
    phasor_gradients: Tensor,
    forward_phasors: Tensor | None = None,
    adjoint_sources: Tensor | None = None,
    adjoint_phasors: Tensor,
    grad_eps_by_freq: Tensor,
    wave_characters: Sequence[Any] | None = None,
) -> Path | None:
    if not _WRITE_ADJOINT_DEBUG_PLOTS:
        return None
    plot_idx = next(_ADJOINT_DEBUG_COUNTER)
    out_path = _ADJOINT_DEBUG_DIR / f"{monitor.name}_backward_{plot_idx:04d}.png"
    component = "Ey"
    if "Ez" in monitor.name:
        component = "Ez"
    elif "Ex" in monitor.name:
        component = "Ex"
    return plot_adjoint_debug_multifreq(
        filepath=out_path,
        phasor_gradients=phasor_gradients,
        forward_phasors=forward_phasors,
        adjoint_sources=adjoint_sources,
        adjoint_phasors=adjoint_phasors,
        grad_eps_by_freq=grad_eps_by_freq,
        wave_characters=(
            monitor.wave_characters if wave_characters is None else wave_characters
        ),
        gradient_component=component,
        field_component=component,
        max_freqs=min(
            4,
            (
                len(monitor.wave_characters)
                if wave_characters is None
                else len(wave_characters)
            ),
        ),
    )


def _cotangents_to_current_profiles(
    cotangents: Any,
    components: Sequence[str],
) -> tuple[jax.Array | None, jax.Array | None]:
    """Convert phasor-output cotangents into distributed adjoint currents."""
    cot = jnp.asarray(cotangents)
    if cot.ndim < 3:
        raise ValueError(
            f"Expected cotangents shaped (freq, component, ...), got {cot.shape}"
        )

    profile_shape = (cot.shape[0], 3, *cot.shape[2:])
    electric = jnp.zeros(profile_shape, dtype=jnp.complex64)
    magnetic = jnp.zeros(profile_shape, dtype=jnp.complex64)
    has_e = False
    has_h = False
    for idx, component in enumerate(components):
        # Tidy3D-style convention: adjoint source amplitude is -i conj(dJ/doutput).
        current = -1j * jnp.conj(cot[:, idx])
        if component in _E_COMPONENTS:
            electric = electric.at[:, _E_COMPONENTS[component]].set(current)
            has_e = True
        elif component in _H_COMPONENTS:
            ## -1 here is to reverse the light direction
            magnetic = magnetic.at[:, _H_COMPONENTS[component]].set(-current)
            has_h = True
        else:
            raise ValueError(f"Unsupported phasor component {component!r}")

    return (electric if has_e else None), (magnetic if has_h else None)


def _expand_adjoint_source_slice(
    monitor_slice: tuple[slice, slice, slice],
    grid_shape: Sequence[int],
) -> tuple[slice, slice, slice]:
    expanded = []
    for axis_slice, axis_size in zip(monitor_slice, grid_shape):
        if not isinstance(axis_slice, slice):
            raise ValueError(
                "Sparse adjoint source metadata requires monitor_slice entries "
                "to be slice objects."
            )
        start = 0 if axis_slice.start is None else int(axis_slice.start)
        stop = axis_size if axis_slice.stop is None else int(axis_slice.stop)
        expanded.append(
            slice(
                max(0, start - _ADJOINT_SOURCE_YEE_HALO),
                min(axis_size, stop + _ADJOINT_SOURCE_YEE_HALO),
            )
        )
    return tuple(expanded)  # type: ignore[return-value]


def _build_sparse_current_blocks(
    *,
    cotangents: Any,
    components: Sequence[str],
    adjoint_group_info: Sequence[dict[str, Any]],
    global_freq_idx: int,
) -> tuple[CurrentSourceBlock, ...]:
    grid_shape = tuple(int(size) for size in cotangents.shape[-3:])
    blocks = []
    for entry in adjoint_group_info:
        freq_indices = {int(index) for index in entry.get("freq_indices", ())}
        if global_freq_idx not in freq_indices:
            continue
        monitor_slice = entry.get("monitor_slice")
        if monitor_slice is None or len(monitor_slice) != 3:
            continue
        grid_slice = _expand_adjoint_source_slice(monitor_slice, grid_shape)
        local_cotangents = cotangents[(slice(None), slice(None), *grid_slice)]
        if isinstance(local_cotangents, Tensor):
            # A port crop is a strided CUDA view; JAX's CUDA array interface
            # only accepts compact storage. The standard DLPack bridge also
            # resolves Torch's lazy complex conjugation without changing values.
            local_cotangents = _torch_to_jax(local_cotangents)
        electric_profiles, magnetic_profiles = _cotangents_to_current_profiles(
            local_cotangents, components
        )
        blocks.append(
            CurrentSourceBlock(
                grid_slice=grid_slice,
                electric_profiles=electric_profiles,
                magnetic_profiles=magnetic_profiles,
            )
        )
    return tuple(blocks)


def _source_profile_for_debug(
    source: Any,
    grid_shape: Sequence[int],
) -> Tensor:
    """Materialize all six source components for debug visualization."""
    source_shape = tuple(int(size) for size in grid_shape)
    electric = (
        None
        if source.electric_profiles is None
        else _jax_to_torch(source.electric_profiles)
    )
    magnetic = (
        None
        if source.magnetic_profiles is None
        else _jax_to_torch(source.magnetic_profiles)
    )
    device = (
        electric.device
        if electric is not None
        else (magnetic.device if magnetic is not None else torch.device("cpu"))
    )
    profile = torch.zeros(
        (
            len(getattr(source, "sample_wave_characters", source.wave_characters)),
            len(_COMPONENTS),
            *source_shape,
        ),
        dtype=torch.complex64,
        device=device,
    )
    if electric is not None:
        profile[
            :,
            :3,
            : source.electric_profiles.shape[-3],
            : source.electric_profiles.shape[-2],
            : source.electric_profiles.shape[-1],
        ] = electric
    if magnetic is not None:
        profile[
            :,
            3:,
            : source.magnetic_profiles.shape[-3],
            : source.magnetic_profiles.shape[-2],
            : source.magnetic_profiles.shape[-1],
        ] = magnetic
    if electric is not None or magnetic is not None:
        return profile

    for block in source.blocks or ():
        if block.electric_profiles is not None:
            profile[:, :3, *block.grid_slice] += _jax_to_torch(block.electric_profiles)
        if block.magnetic_profiles is not None:
            profile[:, 3:, *block.grid_slice] += _jax_to_torch(block.magnetic_profiles)
    return profile


def _pullback_colocated_cotangents_to_yee(
    cotangents: Tensor,
    components: Sequence[str],
    *,
    objects: ObjectContainer,
    config: SimulationConfig,
) -> Tensor:
    """Pull exact-detector cotangents back to the Yee grid.

    ``yee_to_colocate_interpolate`` is the forward field map.  Its reverse
    mode is needed here because ``DistributedCurrentSource`` consumes Yee-grid
    current profiles, while exact phasor detectors expose colocated fields.
    """
    if cotangents.ndim != 5:
        raise ValueError(
            "Expected cotangents shaped (freq, component, Nx, Ny, Nz), got "
            f"{tuple(cotangents.shape)}"
        )

    pulled_back = torch.zeros_like(cotangents)
    grouped_components = {
        True: [
            i for i, component in enumerate(components) if component.startswith("E")
        ],
        False: [
            i for i, component in enumerate(components) if component.startswith("H")
        ],
    }
    with torch.enable_grad():
        for is_electric, indices in grouped_components.items():
            if not indices:
                continue
            yee_field = torch.zeros(
                (cotangents.shape[0], 3, *cotangents.shape[2:]),
                dtype=cotangents.dtype,
                device=cotangents.device,
                requires_grad=True,
            )
            colocated_field = yee_to_colocate_interpolate(
                yee_field, objects, config, is_E=is_electric
            )
            component_cotangents = torch.zeros_like(colocated_field)
            for component_idx in indices:
                component_cotangents[
                    :, component_idx if is_electric else component_idx - 3
                ] = cotangents[:, component_idx]
            yee_cotangents = torch.autograd.grad(
                outputs=colocated_field,
                inputs=yee_field,
                grad_outputs=component_cotangents,
                retain_graph=False,
                create_graph=False,
            )[0]
            for component_idx in indices:
                source_idx = component_idx if is_electric else component_idx - 3
                pulled_back[:, component_idx] = yee_cotangents[:, source_idx]
    return pulled_back


def _fold_unfolded_phasor_cotangents(
    cotangents: Tensor,
    *,
    monitor: Any,
    config: SimulationConfig,
) -> Tensor:
    """Apply the transpose of symmetry unfolding to dense cotangents.

    For one touched symmetry axis, the forward map is
    ``U(x) = cat(flip(x) * parity, x)``. Therefore
    ``U*(g) = flip(g_left * parity) + g_right``. Multiple axes are folded in
    reverse order because the forward unfolds them in x, y, z order.

    Args:
        cotangents: Full-domain cotangents shaped ``(F, 6, Nx, Ny, Nz)``.
        monitor: Placed detector carrying its reduced-domain slice.
        config: Simulation configuration carrying symmetry wall types.

    Returns:
        Cotangents folded onto the detector's reduced symmetry domain.
    """
    if not config.has_symmetry or not any(config.symmetry):
        return cotangents
    if getattr(monitor, "reduce_volume", False):
        raise ValueError("Dense phasor pullback requires a spatial detector.")

    reduced_slice = monitor._grid_slice_tuple
    touched = tuple(
        (
            config.symmetry[axis]
            if config.symmetry[axis] != 0 and reduced_slice[axis][0] == 0
            else 0
        )
        for axis in range(3)
    )
    folded = cotangents
    component_spec = (("E", 0), ("E", 1), ("E", 2), ("H", 0), ("H", 1), ("H", 2))
    for physical_axis in reversed(range(3)):
        wall = touched[physical_axis]
        if wall == 0:
            continue
        tensor_axis = physical_axis + 2
        axis_size = folded.shape[tensor_axis]
        if axis_size % 2:
            raise ValueError(
                "Unfolded cotangent axis must have even length: "
                f"axis={physical_axis}, size={axis_size}."
            )
        half_size = axis_size // 2
        mirrored, retained = torch.split(
            folded, (half_size, half_size), dim=tensor_axis
        )
        parities = [
            _field_component_parity(field_type, component, physical_axis, wall)
            for field_type, component in component_spec
        ]
        parity_shape = [1] * folded.ndim
        parity_shape[1] = len(parities)
        parity = torch.as_tensor(
            parities, dtype=folded.dtype, device=folded.device
        ).reshape(parity_shape)
        folded = retained + torch.flip(mirrored * parity.conj(), dims=(tensor_axis,))
    return folded


def _edge_average_transpose_weights(
    reference: Tensor,
    *,
    config: SimulationConfig,
    axis: int,
) -> tuple[Tensor | float, Tensor | float]:
    """Return current/previous weights for a backward edge-average VJP.

    Args:
        reference: Output cotangent whose final three axes are spatial.
        config: Uniform or rectilinear-grid simulation configuration.
        axis: Physical spatial axis of the edge average.

    Returns:
        Weights multiplying the current and previous input samples.
    """
    if not getattr(config, "has_nonuniform_grid", False):
        return 0.5, 0.5
    grid = getattr(config, "resolved_grid", None)
    if grid is None:
        return 0.5, 0.5

    widths = _jax_to_torch(grid.cell_widths(axis)).to(
        device=reference.device, dtype=reference.real.dtype
    )
    if widths.numel() != reference.shape[-3 + axis]:
        raise ValueError(
            "Grid widths do not match cotangent shape: "
            f"axis={axis}, widths={widths.numel()}, "
            f"cotangents={reference.shape[-3 + axis]}."
        )
    previous_widths = torch.cat((widths[:1], widths[:-1]))
    denominator = widths + previous_widths
    weight_shape = [1] * reference.ndim
    weight_shape[-3 + axis] = widths.numel()
    current_weight = (previous_widths / denominator).reshape(weight_shape)
    previous_weight = (widths / denominator).reshape(weight_shape)
    return current_weight, previous_weight


def _scatter_colocation_component_transpose(
    padded_gradient: Tensor,
    gradient: Tensor,
    *,
    backward_axes: Sequence[int],
    average_forward_z: bool,
    config: SimulationConfig,
) -> None:
    """Scatter one colocated component cotangent onto its padded Yee grid.

    Args:
        padded_gradient: Destination shaped ``(F, Nx+2, Ny+2, Nz+2)``.
        gradient: Colocated component cotangent shaped ``(F, Nx, Ny, Nz)``.
        backward_axes: Axes using the grid-weighted backward edge average.
        average_forward_z: Whether the forward map also averages adjacent z
            samples with arithmetic weights.
        config: Simulation configuration defining rectilinear cell widths.
    """
    choices: list[tuple[int, tuple[tuple[slice, Tensor | float], ...]]] = []
    for axis in backward_axes:
        current_weight, previous_weight = _edge_average_transpose_weights(
            gradient, config=config, axis=axis
        )
        choices.append(
            (
                axis,
                (
                    (slice(1, -1), current_weight),
                    (slice(0, -2), previous_weight),
                ),
            )
        )
    if average_forward_z:
        choices.append((2, ((slice(1, -1), 0.5), (slice(2, None), 0.5))))

    if not choices:
        padded_gradient[..., 1:-1, 1:-1, 1:-1].add_(gradient)
        return

    for selected in product(*(axis_choices for _, axis_choices in choices)):
        spatial_slices = [slice(1, -1), slice(1, -1), slice(1, -1)]
        coefficient: Tensor | float = 1.0
        for (axis, _), (axis_slice, weight) in zip(choices, selected):
            spatial_slices[axis] = axis_slice
            coefficient = coefficient * weight
        padded_gradient[(..., *spatial_slices)].add_(gradient * coefficient)


def _transpose_boundary_padding(
    padded_gradient: Tensor,
    *,
    objects: ObjectContainer,
    config: SimulationConfig,
    unpadded_shape: Sequence[int],
) -> Tensor:
    """Apply the transpose of boundary correction and three-axis padding.

    Args:
        padded_gradient: Cotangent of the padded field tensor.
        objects: Object container carrying periodic and Bloch boundaries.
        config: Simulation configuration used to evaluate Bloch phases.
        unpadded_shape: Original three-dimensional Yee-grid shape.

    Returns:
        Cotangent on the unpadded Yee grid.
    """
    gradient = padded_gradient
    for boundary in reversed(tuple(getattr(objects, "boundary_objects", ()))):
        if not getattr(boundary, "uses_wrap_padding", False):
            continue
        if not getattr(boundary, "needs_complex_fields", False):
            continue
        if config.has_nonuniform_grid:
            if config.resolved_grid is None:
                raise ValueError("Nonuniform grid is missing resolved_grid.")
            spacing = float(config.resolved_grid.min_spacing)
        else:
            spacing = config.uniform_spacing()
        phase = torch.as_tensor(
            boundary.get_bloch_phase(tuple(int(v) for v in unpadded_shape), spacing),
            dtype=gradient.dtype,
            device=gradient.device,
        )
        multiplier = phase.conj() if boundary.direction == "-" else phase
        index = [slice(None)] * gradient.ndim
        index[-3 + boundary.axis] = 0 if boundary.direction == "-" else -1
        gradient[tuple(index)].mul_(multiplier.conj())

    periodic_axes = get_wrap_padding_axes(objects)
    for axis in reversed(range(3)):
        tensor_axis = gradient.ndim - 3 + axis
        interior_index = [slice(None)] * gradient.ndim
        interior_index[tensor_axis] = slice(1, -1)
        unpadded = gradient[tuple(interior_index)].clone()
        low = gradient.select(tensor_axis, 0)
        high = gradient.select(tensor_axis, gradient.shape[tensor_axis] - 1)
        low_target = -1 if periodic_axes[axis] else 0
        high_target = 0 if periodic_axes[axis] else -1
        unpadded.select(tensor_axis, low_target).add_(low)
        unpadded.select(tensor_axis, high_target).add_(high)
        gradient = unpadded
    return gradient


def _colocation_interpolation_transpose(
    cotangents: Tensor,
    *,
    objects: ObjectContainer,
    config: SimulationConfig,
    is_electric: bool,
) -> Tensor:
    """Apply the exact transpose of Yee-to-colocated interpolation.

    Args:
        cotangents: Colocated vector-field cotangents shaped
            ``(F, 3, Nx, Ny, Nz)``.
        objects: Object container defining boundary padding.
        config: Simulation configuration defining grid weights.
        is_electric: Select the electric or magnetic Yee staggering.

    Returns:
        Three-component cotangents on the unpadded Yee grid.
    """
    if cotangents.ndim != 5 or cotangents.shape[1] != 3:
        raise ValueError(
            "Expected colocated cotangents shaped (F, 3, Nx, Ny, Nz), got "
            f"{tuple(cotangents.shape)}."
        )
    padded_shape = tuple(int(size) + 2 for size in cotangents.shape[-3:])
    padded = torch.zeros(
        (cotangents.shape[0], 3, *padded_shape),
        dtype=cotangents.dtype,
        device=cotangents.device,
    )
    if is_electric:
        stencils = (((0,), True), ((1,), True), ((), False))
    else:
        stencils = (((1,), False), ((0,), False), ((0, 1), True))
    for component, (backward_axes, average_forward_z) in enumerate(stencils):
        _scatter_colocation_component_transpose(
            padded[:, component],
            cotangents[:, component],
            backward_axes=backward_axes,
            average_forward_z=average_forward_z,
            config=config,
        )
    return _transpose_boundary_padding(
        padded,
        objects=objects,
        config=config,
        unpadded_shape=cotangents.shape[-3:],
    )


def _pullback_dense_cotangents_autodiff_reference(
    cotangents: Tensor,
    *,
    monitor: Any,
    objects: ObjectContainer,
    config: SimulationConfig,
) -> Tensor:
    """Evaluate the dense observation-map VJP using PyTorch autodiff.

    The dense reference exposes fields after spatial Yee colocation, magnetic
    half-step phase correction, and symmetry unfolding. Sparse forward
    recording performs an equivalent localized observation inside FDTD-X, but
    its custom autograd boundary receives cotangents on the unfolded domain.
    This value-independent surrogate reconstructs the dense observation map so
    its exact PyTorch VJP can produce the reduced-domain Yee current profile.

    Args:
        cotangents: Dense complex cotangents shaped
            ``(frequency, 6, Nx, Ny, Nz)``.
        monitor: Placed sparse forward detector carrying frequency metadata.
        objects: Forward FDTD-X object container used for boundary padding.
        config: Reduced-domain FDTD-X simulation configuration.

    Returns:
        Reduced-domain Yee-grid cotangents with six components.
    """
    reduced_shape = tuple(int(size) for size in objects.volume.grid_shape)
    with torch.enable_grad():
        yee_fields = torch.zeros(
            (cotangents.shape[0], len(_COMPONENTS), *reduced_shape),
            dtype=cotangents.dtype,
            device=cotangents.device,
            requires_grad=True,
        )
        electric = yee_to_colocate_interpolate(
            yee_fields[:, :3], objects, config, is_E=True
        )
        magnetic = yee_to_colocate_interpolate(
            yee_fields[:, 3:], objects, config, is_E=False
        )
        frequencies = torch.as_tensor(
            [wave.get_frequency() for wave in monitor.wave_characters],
            dtype=magnetic.real.dtype,
            device=magnetic.device,
        )
        theta = 2 * torch.pi * frequencies * config.time_step_duration / 2
        magnetic = magnetic * (torch.exp(1j * theta) * torch.cos(theta)).reshape(
            -1, 1, 1, 1, 1
        )
        reference_outputs = unfold_phasor(
            torch.cat((electric, magnetic), dim=1),
            monitor=monitor,
            components=_COMPONENTS,
            config=config,
        )
        if reference_outputs.shape != cotangents.shape:
            raise ValueError(
                "Dense-reference observation shape does not match sparse "
                f"cotangents: reference={tuple(reference_outputs.shape)}, "
                f"cotangents={tuple(cotangents.shape)}."
            )
        return torch.autograd.grad(
            outputs=reference_outputs,
            inputs=yee_fields,
            grad_outputs=cotangents,
            retain_graph=False,
            create_graph=False,
        )[0]


def _pullback_dense_cotangents_like_reference(
    cotangents: Tensor,
    *,
    monitor: Any,
    objects: ObjectContainer,
    config: SimulationConfig,
    wave_characters: Sequence[Any] | None = None,
) -> Tensor:
    """Apply the analytical transpose of the dense observation map.

    The forward observation map is symmetry unfolding after electric/magnetic
    Yee colocation and magnetic half-step phase correction. This function
    applies those operators' conjugate transposes in reverse order without
    constructing dummy fields or an autograd graph.

    Args:
        cotangents: Dense complex cotangents shaped
            ``(frequency, 6, Nx, Ny, Nz)``.
        monitor: Placed detector carrying symmetry and frequency metadata.
        objects: FDTD-X object container defining the reduced grid and padding.
        config: Reduced-domain FDTD-X simulation configuration.
        wave_characters: Optional frequency subset corresponding to the first
            axis of ``cotangents``. By default, use all monitor frequencies.

    Returns:
        Reduced-domain Yee-grid cotangents with six components.
    """
    if cotangents.ndim != 5 or cotangents.shape[1] != len(_COMPONENTS):
        raise ValueError(
            "Expected cotangents shaped (frequency, 6, Nx, Ny, Nz), got "
            f"{tuple(cotangents.shape)}."
        )
    selected_waves = tuple(
        monitor.wave_characters if wave_characters is None else wave_characters
    )
    if cotangents.shape[0] != len(selected_waves):
        raise ValueError(
            "Cotangent frequency axis does not match monitor waves: "
            f"cotangents={cotangents.shape[0]}, "
            f"waves={len(selected_waves)}."
        )

    folded = _fold_unfolded_phasor_cotangents(
        cotangents, monitor=monitor, config=config
    )
    reduced_shape = tuple(int(size) for size in objects.volume.grid_shape)
    if tuple(folded.shape[-3:]) != reduced_shape:
        raise ValueError(
            "Folded cotangent shape does not match the reduced Yee grid: "
            f"cotangents={tuple(folded.shape[-3:])}, grid={reduced_shape}."
        )

    output = torch.empty_like(folded)
    electric = _colocation_interpolation_transpose(
        folded[:, :3], objects=objects, config=config, is_electric=True
    )
    output[:, :3].copy_(electric)
    del electric

    frequencies = torch.as_tensor(
        [wave.get_frequency() for wave in selected_waves],
        dtype=folded.real.dtype,
        device=folded.device,
    )
    theta = torch.pi * frequencies * config.time_step_duration
    phase = (torch.exp(1j * theta) * torch.cos(theta)).conj().reshape(-1, 1, 1, 1, 1)
    magnetic = _colocation_interpolation_transpose(
        folded[:, 3:] * phase,
        objects=objects,
        config=config,
        is_electric=False,
    )
    output[:, 3:].copy_(magnetic)
    return output


def _cotangents_to_current_profiles_torch(
    cotangents: Tensor,
    components: Sequence[str],
) -> tuple[Tensor | None, Tensor | None]:
    profile_shape = (cotangents.shape[0], 3, *cotangents.shape[2:])
    electric = torch.zeros(
        profile_shape, dtype=torch.complex64, device=cotangents.device
    )
    magnetic = torch.zeros(
        profile_shape, dtype=torch.complex64, device=cotangents.device
    )
    has_e = False
    has_h = False
    for idx, component in enumerate(components):
        current = -1j * torch.conj(cotangents[:, idx].to(torch.complex64))
        if component in _E_COMPONENTS:
            electric[:, _E_COMPONENTS[component]] = current
            has_e = True
        elif component in _H_COMPONENTS:
            magnetic[:, _H_COMPONENTS[component]] = -current
            has_h = True
        else:
            raise ValueError(f"Unsupported phasor component {component!r}")
    return (electric if has_e else None), (magnetic if has_h else None)


def _fit_separable_current_profile_torch(
    cotangents: Tensor,
    components: Sequence[str],
) -> tuple[Tensor | None, Tensor | None, Tensor]:
    electric_profiles, magnetic_profiles = _cotangents_to_current_profiles_torch(
        cotangents, components
    )
    flat_parts = []
    if electric_profiles is not None:
        flat_parts.append(electric_profiles.reshape(electric_profiles.shape[0], -1))
    if magnetic_profiles is not None:
        flat_parts.append(magnetic_profiles.reshape(magnetic_profiles.shape[0], -1))
    if len(flat_parts) == 0:
        raise ValueError("No electric or magnetic cotangent components found.")

    combined = torch.cat(flat_parts, dim=1)
    energy = torch.linalg.norm(combined, dim=1)
    ref_idx = int(torch.argmax(energy).item())
    ref_profile = combined[ref_idx]
    denom = torch.sum(torch.conj(ref_profile) * ref_profile)
    if torch.abs(denom) == 0:
        alpha = torch.zeros(
            (combined.shape[0],), dtype=torch.complex64, device=cotangents.device
        )
    else:
        alpha = torch.sum(torch.conj(ref_profile)[None, :] * combined, dim=1) / denom

    template_e = None
    template_m = None
    if electric_profiles is not None:
        template_e = electric_profiles[ref_idx]
    if magnetic_profiles is not None:
        template_m = magnetic_profiles[ref_idx]
    return template_e, template_m, alpha.to(torch.complex64), ref_idx


def _factor_frequency_dependent_current_profiles_torch(
    cotangents: Tensor,
    components: Sequence[str],
    *,
    frequencies: Sequence[float],
    center_frequency: float,
    order: int,
) -> tuple[Tensor | None, Tensor | None, Tensor, dict[str, Any]]:
    """Convert Yee cotangents and factor their smooth spatial current family."""
    from thirdparty.fdtdx.objects.sources.current_profiles import (
        factor_current_profiles_torch,
    )

    electric, magnetic = _cotangents_to_current_profiles_torch(cotangents, components)
    return factor_current_profiles_torch(
        electric,
        magnetic,
        frequencies=frequencies,
        center_frequency=center_frequency,
        order=order,
    )


def _build_distributed_adjoint_source(
    *,
    adj_monitor: Any,
    source_template: Any,
    cotangents: Any,
    center_on_monitor: bool = False,
    adjoint_group_info: Sequence[dict[str, Any]] | None = None,
    global_freq_idx: int | None = None,
    source_partial_grid_shape: Sequence[int] | None = None,
) -> DistributedCurrentSource:
    # Cotangents must be Yee-grid dL/dE and dL/dH. Use exact_interpolation=False
    # in PhasorDetector to disable interpolation.
    components = tuple(getattr(adj_monitor, "components", _COMPONENTS))
    blocks = None
    electric_profiles = None
    magnetic_profiles = None
    if (
        _MAPS_SPARSE_ADJOINT_SOURCE
        and CurrentSourceBlock is not None
        and adjoint_group_info
        and global_freq_idx is not None
    ):
        blocks = _build_sparse_current_blocks(
            cotangents=cotangents,
            components=components,
            adjoint_group_info=adjoint_group_info,
            global_freq_idx=global_freq_idx,
        )
    if not blocks:
        electric_profiles, magnetic_profiles = _cotangents_to_current_profiles(
            cotangents, components
        )
    temporal_profile = getattr(
        source_template, "temporal_profile", SingleFrequencyProfile()
    )
    if (
        center_on_monitor
        and isinstance(temporal_profile, GaussianPulseProfile)
        and len(adj_monitor.wave_characters) == 1
    ):
        temporal_profile = temporal_profile.aset(
            "center_wave", adj_monitor.wave_characters[0]
        )
    source = DistributedCurrentSource(
        name=f"{adj_monitor.name}_adjoint_source",
        partial_grid_shape=(
            tuple(int(size) for size in source_partial_grid_shape)
            if source_partial_grid_shape is not None
            else (
                tuple(int(size) for size in cotangents.shape[-3:])
                if blocks
                else adj_monitor.partial_grid_shape
            )
        ),
        wave_characters=adj_monitor.wave_characters,
        temporal_profile=temporal_profile,
        electric_profiles=electric_profiles,
        magnetic_profiles=magnetic_profiles,
        blocks=blocks,
        static_amplitude_factor=SCALE,
    )

    return source


def _build_separable_broadband_adjoint_source(
    *,
    adj_monitor: Any,
    source_template: Any,
    cotangents: Tensor,
    symmetry: Sequence[int] = (0, 0, 0),
    source_partial_grid_shape: Sequence[int] | None = None,
) -> tuple[DistributedCurrentSource, Tensor]:
    components = tuple(getattr(adj_monitor, "components", _COMPONENTS))
    template_e, template_m, alpha, template_idx = _fit_separable_current_profile_torch(
        cotangents, components
    )
    nfreq = 1  # cotangents.shape[0]
    electric_profiles = None
    magnetic_profiles = None
    if template_e is not None:
        electric_profiles = _torch_to_jax(
            template_e.unsqueeze(0).expand(nfreq, -1, -1, -1, -1).contiguous()
        )
    if template_m is not None:
        magnetic_profiles = _torch_to_jax(
            template_m.unsqueeze(0).expand(nfreq, -1, -1, -1, -1).contiguous()
        )
    source_shape = tuple(
        int(size)
        for size in (
            adj_monitor.partial_grid_shape
            if source_partial_grid_shape is None
            else source_partial_grid_shape
        )
    )
    expected_profile_shape = tuple(
        target // 2 if symmetry[axis] else target
        for axis, target in enumerate(source_shape)
    )
    for axis, target in enumerate(expected_profile_shape):
        spatial_axis = axis + 2
        profile_size = (
            electric_profiles.shape[spatial_axis]
            if electric_profiles is not None
            else magnetic_profiles.shape[spatial_axis]
        )
        if profile_size != target:
            raise ValueError(
                "Adjoint source profile shape does not match monitor placement: "
                f"axis={axis}, profile={profile_size}, target={target}"
            )
    temporal_profile = getattr(
        source_template, "temporal_profile", SingleFrequencyProfile()
    )
    center_wave = (
        temporal_profile.center_wave
        if hasattr(temporal_profile, "center_wave")
        else None
    )
    source = DistributedCurrentSource(
        name=f"{adj_monitor.name}_adjoint_source",
        partial_grid_shape=source_shape,
        wave_characters=(center_wave,),  # should use the same as forward simulation
        temporal_profile=temporal_profile,
        electric_profiles=electric_profiles,
        magnetic_profiles=magnetic_profiles,
        static_amplitude_factor=SCALE,
    )
    return source, alpha


def _build_factored_chebyshev_adjoint_source(
    *,
    adj_monitor: Any,
    source_template: Any,
    cotangents: Tensor,
    order: int,
    symmetry: Sequence[int] = (0, 0, 0),
    source_partial_grid_shape: Sequence[int] | None = None,
) -> tuple[DistributedCurrentSource, Tensor, dict[str, Any]]:
    """Build a broadband source with separate smooth profiles and VJP weights.

    Args:
        adj_monitor: Detector defining target frequencies and field components.
        source_template: Forward source providing the Gaussian temporal profile.
        cotangents: Yee-grid objective cotangents shaped ``(F, C, Nx, Ny, Nz)``.
        order: Number of Chebyshev spatial and temporal terms.
        symmetry: Active simulation mirror parities.
        source_partial_grid_shape: Optional full-domain placement shape for
            cotangents reconstructed from sparse monitors.

    Returns:
        Unplaced source, complex post-weights, and spatial-fit diagnostics.

    Raises:
        ValueError: If the temporal profile, fitting order, or placement shape
            is incompatible with the supplied current samples.
    """
    if FactoredChebyshevCurrentSource is None:
        raise RuntimeError("Factored Chebyshev current-source support is unavailable.")
    temporal_profile = getattr(
        source_template, "temporal_profile", SingleFrequencyProfile()
    )
    if not isinstance(temporal_profile, GaussianPulseProfile):
        raise ValueError("Factored broadband adjoints require a Gaussian pulse.")
    target_waves = tuple(adj_monitor.wave_characters)
    frequencies = tuple(float(wave.get_frequency()) for wave in target_waves)
    center_frequency = float(temporal_profile.center_wave.get_frequency())
    electric, magnetic, weights, diagnostics = (
        _factor_frequency_dependent_current_profiles_torch(
            cotangents,
            tuple(getattr(adj_monitor, "components", _COMPONENTS)),
            frequencies=frequencies,
            center_frequency=center_frequency,
            order=order,
        )
    )
    # Keep the injected current scale and global phase comparable to the
    # strongest original sample. Unit-profile injection alone can change the
    # decay detector's float32 dynamic range. This is an exact common rescaling
    # of the whole smooth family, never a projection onto the anchor profile.
    reference_scale = weights[torch.argmax(weights.abs())]
    if electric is not None:
        electric = electric * reference_scale
    if magnetic is not None:
        magnetic = magnetic * reference_scale
    weights = weights / reference_scale
    source_shape = tuple(
        int(size)
        for size in (
            adj_monitor.partial_grid_shape
            if source_partial_grid_shape is None
            else source_partial_grid_shape
        )
    )
    expected_profile_shape = tuple(
        target // 2 if symmetry[axis] else target
        for axis, target in enumerate(source_shape)
    )
    profile = electric if electric is not None else magnetic
    for axis, target in enumerate(expected_profile_shape):
        if profile.shape[axis + 2] != target:
            raise ValueError(
                "Adjoint source profile shape does not match monitor placement: "
                f"axis={axis}, profile={profile.shape[axis + 2]}, target={target}"
            )
    polynomial_half_width = max(
        abs(frequency - center_frequency) for frequency in frequencies
    )
    # Only the number of entries matters for the temporal Chebyshev basis; the
    # physical sample frequencies are stored separately.
    basis_waves = target_waves[:order]
    source = FactoredChebyshevCurrentSource(
        name=f"{adj_monitor.name}_factored_chebyshev_adjoint_source",
        partial_grid_shape=source_shape,
        wave_characters=basis_waves,
        sample_wave_characters=target_waves,
        polynomial_half_width=polynomial_half_width,
        temporal_profile=temporal_profile,
        electric_profiles=(None if electric is None else _torch_to_jax(electric)),
        magnetic_profiles=(None if magnetic is None else _torch_to_jax(magnetic)),
        static_amplitude_factor=SCALE,
    )
    diagnostics["order"] = float(order)
    return source, weights, diagnostics


def _plan_broadband_adjoint_groups(
    *,
    cotangents: Tensor,
    groups: dict[str, dict[str, Tensor]],
    monitor: Any,
    source_template: Any,
    config: Any,
) -> dict[str, int] | None:
    """Choose bounded smooth profile fits, or request per-frequency fallback.

    The plan never increases simulation duration. It is used only when the
    number of spatial groups is less than the number of independent frequency
    runs. A failed group sends the whole batch to the reference path so
    overlapping group contributions cannot be counted twice.

    Args:
        cotangents: Pulled-back Yee-grid objective cotangents.
        groups: Frequency indices and spatial indices for each source group.
        monitor: Detector providing target waves and component ordering.
        source_template: Forward source providing the Gaussian pulse.
        config: Existing simulation duration and time-step configuration.

    Returns:
        Accepted spatial order for every group, or None to use independent
        per-frequency adjoints for the batch.
    """
    from thirdparty.fdtdx.objects.sources.chebyshev import (
        chebyshev_temporal_diagnostics,
    )

    if _BROADBAND_ADJOINT_METHOD == "legacy":
        return {name: 1 for name in groups}
    if _BROADBAND_ADJOINT_METHOD not in ("auto", "chebyshev"):
        raise ValueError(
            "MAPS_BROADBAND_ADJOINT_METHOD must be auto, chebyshev, or legacy."
        )
    profile = getattr(source_template, "temporal_profile", None)
    if not isinstance(profile, GaussianPulseProfile):
        logging.warning(
            "Broadband adjoint fallback: the source is not a Gaussian pulse."
        )
        return None
    coverage = torch.zeros(
        (cotangents.shape[0], *cotangents.shape[-3:]),
        device=cotangents.device,
        dtype=torch.int16,
    )
    for group in groups.values():
        xs, ys, zs = group["spatial_indices"].to(cotangents.device).unbind(dim=1)
        for index in group["freq_indices"].tolist():
            coverage[index, xs, ys, zs] += 1
    active = cotangents.abs().amax(dim=1) > 0
    if torch.any(coverage[active] != 1):
        logging.warning(
            "Broadband adjoint fallback: groups omit or duplicate active currents."
        )
        return None
    del coverage, active
    center = profile.center_wave.get_frequency()
    plan = {}
    for name, group in groups.items():
        indices = [int(index) for index in group["freq_indices"].tolist()]
        frequencies = [
            monitor.wave_characters[index].get_frequency() for index in indices
        ]
        current = extract_group_grad_compact_torch(cotangents, group)
        orders = (
            (min(_BROADBAND_ADJOINT_PROFILE_ORDER, len(indices)),)
            if _BROADBAND_ADJOINT_PROFILE_ORDER > 0
            else tuple(range(1, min(5, len(indices)) + 1))
        )
        accepted = False
        reason = "no acceptable profile fit"
        for order in orders:
            try:
                _, _, _, diagnostics = (
                    _factor_frequency_dependent_current_profiles_torch(
                        current,
                        tuple(monitor.components),
                        frequencies=frequencies,
                        center_frequency=center,
                        order=order,
                    )
                )
                diagnostics.update(
                    chebyshev_temporal_diagnostics(
                        temporal_profile=profile,
                        frequencies=np.asarray(frequencies),
                        coefficient_norms=np.asarray(
                            diagnostics["coefficient_norm_by_term"]
                        ),
                        polynomial_half_width=diagnostics["polynomial_half_width_hz"],
                        time_step_duration=config.time_step_duration,
                        num_time_steps=config.time_steps_total,
                    )
                )
                peak_bound = diagnostics["waveform_peak_norm_bound"]
                response_error = diagnostics["temporal_basis_error_max"]
                if diagnostics["fit_relative_l2_max"] > _BROADBAND_ADJOINT_PROFILE_RTOL:
                    reason = (
                        f"profile residual {diagnostics['fit_relative_l2_max']:.3g}"
                    )
                    continue
                if diagnostics["fit_condition"] > 1e5:
                    reason = f"spatial fit condition {diagnostics['fit_condition']:.3g}"
                    continue
                if diagnostics["adjacent_overlap_min"] < _BROADBAND_ADJOINT_MIN_OVERLAP:
                    reason = f"adjacent profile overlap {diagnostics['adjacent_overlap_min']:.3g}"
                    continue
                if (
                    not np.isfinite(peak_bound)
                    or peak_bound > _BROADBAND_ADJOINT_MAX_COEFFICIENT_NORM
                ):
                    reason = f"waveform amplification bound {peak_bound:.3g}"
                    continue
                if response_error > 1e-3:
                    reason = f"discrete temporal-basis error {response_error:.3g}"
                    continue
                plan[name] = order
                print(
                    f"[fdtdx-adjoint] group={name}, factored Chebyshev {diagnostics}",
                    flush=True,
                )
                accepted = True
                break
            except ValueError as error:
                reason = str(error)
        del current
        if not accepted:
            logging.warning("Broadband adjoint fallback for %s: %s", name, reason)
            return None
    return plan


def clone_phasor_detector(
    monitor: Any,
    *,
    freq_idx: int | Tuple[int, ...] | None = None,
    components: Sequence[str] | None = None,
    name: str | None = None,
    partial_grid_shape: tuple[int | None, int | None, int | None] | None = None,
    exact_interpolation: bool = False,
) -> Any:
    """Clone a phasor detector with selected metadata.

    Args:
        monitor: Detector supplying geometry and default metadata.
        freq_idx: Optional frequency index or ordered index tuple to retain.
        components: Optional field-component sequence for the clone.
        name: Optional replacement detector name.
        partial_grid_shape: Optional replacement detector grid shape.
        exact_interpolation: Whether FDTD-X should colocate recorded fields.

    Returns:
        A new FDTD-X phasor detector.
    """
    if isinstance(freq_idx, int):
        freq_idx = (freq_idx,)
    monitor = fdtdx.PhasorDetector(
        name=monitor.name if name is None else name,
        partial_grid_shape=(
            monitor.partial_grid_shape
            if partial_grid_shape is None
            else partial_grid_shape
        ),
        partial_real_shape=getattr(monitor, "partial_real_shape", None),
        partial_real_position=getattr(monitor, "partial_real_position", None),
        wave_characters=(
            tuple(monitor.wave_characters[i] for i in freq_idx)
            if freq_idx is not None
            else monitor.wave_characters
        ),
        components=tuple(monitor.components if components is None else components),
        scaling_mode=monitor.scaling_mode,
        plot=False,
        switch=monitor.switch,  # we downsample by 8x, note that the field amplitude also reduce proprtionally
        # Adjoint detectors remain on the Yee grid; debug/reference detectors
        # may opt into fdtdx's native exact interpolation.
        exact_interpolation=exact_interpolation,
    )

    return monitor


def clone_phasor_detector_for_waves(
    monitor: Any,
    *,
    wave_characters: Sequence[Any],
    components: Sequence[str] | None = None,
    exact_interpolation: bool = False,
) -> Any:
    """Clone a detector for an ordered subset of its configured frequencies.

    Args:
        monitor: Source detector whose frequency metadata is cloned.
        wave_characters: Waves to retain, in the output detector's frequency order.
        components: Optional field components for the cloned detector.
        exact_interpolation: Whether FDTD-X should colocate the recorded fields.

    Returns:
        A detector with the same name and geometry as ``monitor``.

    Raises:
        ValueError: If a requested wave is absent or maps ambiguously.
    """
    available_waves = tuple(monitor.wave_characters)
    frequency_indices = []
    for requested_wave in wave_characters:
        requested_frequency = float(requested_wave.get_frequency())
        matches = [
            index
            for index, available_wave in enumerate(available_waves)
            if abs(float(available_wave.get_frequency()) - requested_frequency)
            <= 1e-9 * max(abs(requested_frequency), 1.0)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Detector {monitor.name!r} has {len(matches)} matches for "
                f"frequency {requested_frequency:.12g}; expected exactly one."
            )
        frequency_indices.append(matches[0])
    return clone_phasor_detector(
        monitor,
        freq_idx=tuple(frequency_indices),
        components=components,
        exact_interpolation=exact_interpolation,
    )


def _clone_phasor_detector_for_frequency(
    monitor: Any,
    *,
    freq_idx: int | Tuple[int, ...] | None = None,
    components: Sequence[str] | None = None,
) -> Any:
    return clone_phasor_detector(
        monitor,
        freq_idx=freq_idx,
        components=components,
        name="adjoint_field_detector",
        exact_interpolation=True,
    )


def _derivative_stretch(
    omega: Tensor,
    alpha: Tensor,
    kappa: Tensor,
    sigma: Tensor,
) -> Tensor:
    denom = kappa.to(torch.complex64) + sigma.to(torch.complex64) / (
        alpha.to(torch.complex64) + 1j * omega[..., None, None, None] * constants.eps0
    )
    return 1.0 / denom


def _apply_inverse_permeability(inv_mu: Any, curl: Tensor) -> Tensor:
    if isinstance(inv_mu, Tensor):
        inv_mu_t = inv_mu.to(device=curl.device)
    elif hasattr(inv_mu, "__dlpack__") and not isinstance(inv_mu, np.ndarray):
        inv_mu_t = _jax_to_torch(inv_mu).to(device=curl.device)
    else:
        inv_mu_t = torch.as_tensor(inv_mu, device=curl.device)
    if inv_mu_t.ndim == 0:
        return curl * inv_mu_t.to(curl.dtype)
    if inv_mu_t.ndim == 4 and inv_mu_t.shape[0] == 3:
        return curl * inv_mu_t.to(curl.dtype).unsqueeze(0)
    if inv_mu_t.ndim == 4 and inv_mu_t.shape[0] == 9:
        inv_mu_t = inv_mu_t.reshape(3, 3, *inv_mu_t.shape[1:]).to(curl.dtype)
        return torch.einsum("abxyz,fbxyz->faxyz", inv_mu_t, curl)
    raise ValueError(f"Unsupported inv_permeabilities shape: {tuple(inv_mu_t.shape)}")


def _axis_widths_for_yee(
    config,
    axis: int,
    n_cells: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Return normalized axis spacings for Yee-grid derivatives.

    On a uniform grid, this returns an all-ones vector.
    On a nonuniform grid, this returns cell widths divided by the
    grid's reference spacing (min_spacing), matching FDTDX's
    normalized FDTD convention.
    """
    grid = getattr(config, "resolved_grid", None)

    if grid is not None:
        widths = _jax_to_torch(grid.cell_widths(axis))
        ref_spacing = float(grid.min_spacing)
    else:
        if hasattr(config, "uniform_spacing"):
            ref_spacing = float(config.uniform_spacing())
            widths = torch.full((n_cells,), ref_spacing)
        else:
            ref_spacing = float(config.resolution)
            widths = torch.full((n_cells,), ref_spacing)

    widths = torch.as_tensor(widths, device=device, dtype=dtype)
    if widths.ndim != 1 or widths.numel() != n_cells:
        raise ValueError(
            f"Axis {axis} spacing has shape {tuple(widths.shape)}, expected ({n_cells},)."
        )

    # Normalize to the solver's internal length scale.
    return widths / ref_spacing


def _derivative_stretch_runtime(
    omega: Tensor,
    pml_a: Tensor,
    pml_b: Tensor,
    pml_inv_kappa: Tensor,
    dt: float,
) -> Tensor:
    """
    Frequency-domain equivalent of the CPML recurrence used in latest FDTDX.

    omega: [nfreq]
    pml_a/pml_b/pml_inv_kappa: broadcastable to the derivative tensor
    """
    z = torch.exp(1j * omega[..., None, None, None] * dt).to(torch.complex64)

    return pml_inv_kappa.to(torch.complex64) + pml_a.to(torch.complex64) / (
        z - pml_b.to(torch.complex64)
    )


def _derive_magnetic_phasor_from_electric(
    electric_phasor: Tensor,
    *,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    wave_characters: Sequence[Any],
) -> Tensor:
    """
    Derive H phasor from E phasor using discrete Yee curl.

    Supports both uniform and nonuniform rectilinear grids.
    The nonuniform metric uses spacing normalized by min_spacing.
    """
    if electric_phasor.ndim != 5 or electric_phasor.shape[1] != 3:
        raise ValueError(
            "Expected electric_phasor shape [nfreq, 3, Nx, Ny, Nz], "
            f"got {tuple(electric_phasor.shape)}"
        )

    device = electric_phasor.device
    cdtype = electric_phasor.dtype
    rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64

    e_pad = pad_fields_for_boundaries(electric_phasor, objects, config)

    ex = e_pad[:, 0]
    ey = e_pad[:, 1]
    ez = e_pad[:, 2]

    xs = slice(1, -1)
    ys = slice(1, -1)
    zs = slice(1, -1)

    # Raw differences on the Yee stencil.
    dy_ez = ez[:, xs, 2:, zs] - ez[:, xs, ys, zs]
    dz_ey = ey[:, xs, ys, 2:] - ey[:, xs, ys, zs]

    dz_ex = ex[:, xs, ys, 2:] - ex[:, xs, ys, zs]
    dx_ez = ez[:, 2:, ys, zs] - ez[:, xs, ys, zs]

    dx_ey = ey[:, 2:, ys, zs] - ey[:, xs, ys, zs]
    dy_ex = ex[:, xs, 2:, zs] - ex[:, xs, ys, zs]

    nx = electric_phasor.shape[2]
    ny = electric_phasor.shape[3]
    nz = electric_phasor.shape[4]

    # Normalized spacings: widths / min_spacing.
    dx = _axis_widths_for_yee(config, 0, nx, device=device, dtype=rdtype)
    dy = _axis_widths_for_yee(config, 1, ny, device=device, dtype=rdtype)
    dz = _axis_widths_for_yee(config, 2, nz, device=device, dtype=rdtype)

    dx_b = dx.view(1, -1, 1, 1).to(cdtype)
    dy_b = dy.view(1, 1, -1, 1).to(cdtype)
    dz_b = dz.view(1, 1, 1, -1).to(cdtype)

    omega = (
        2
        * torch.pi
        * torch.as_tensor(
            [wave.get_frequency() for wave in wave_characters],
            dtype=rdtype,
            device=device,
        )
    )

    # sigma = arrays.sigma
    # kappa = arrays.kappa
    # alpha = arrays.alpha

    alpha = jnp.zeros((6, nx, ny, nz), dtype=jnp.float32)
    kappa = jnp.ones((6, nx, ny, nz), dtype=jnp.float32)
    sigma = jnp.zeros((6, nx, ny, nz), dtype=jnp.float32)

    electric_conductivity = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    magnetic_conductivity = jnp.zeros((nx, ny, nz), dtype=jnp.float32)

    for boundary in getattr(objects, "boundary_objects", []):
        if hasattr(boundary, "modify_arrays"):
            out = boundary.modify_arrays(
                alpha=alpha,
                kappa=kappa,
                sigma=sigma,
                electric_conductivity=electric_conductivity,
                magnetic_conductivity=magnetic_conductivity,
            )
            alpha = out["alpha"]
            kappa = out["kappa"]
            sigma = out["sigma"]
            electric_conductivity = out["electric_conductivity"]
            magnetic_conductivity = out["magnetic_conductivity"]

    alpha = _jax_to_torch(alpha).to(device=device, dtype=rdtype)
    kappa = _jax_to_torch(kappa).to(device=device, dtype=rdtype)
    sigma = _jax_to_torch(sigma).to(device=device, dtype=rdtype)

    alpha3 = _jax_to_torch(alpha[3]).to(device=device, dtype=rdtype)
    alpha4 = _jax_to_torch(alpha[4]).to(device=device, dtype=rdtype)
    alpha5 = _jax_to_torch(alpha[5]).to(device=device, dtype=rdtype)

    kappa3 = _jax_to_torch(kappa[3]).to(device=device, dtype=rdtype)
    kappa4 = _jax_to_torch(kappa[4]).to(device=device, dtype=rdtype)
    kappa5 = _jax_to_torch(kappa[5]).to(device=device, dtype=rdtype)

    sigma3 = _jax_to_torch(sigma[3]).to(device=device, dtype=rdtype)
    sigma4 = _jax_to_torch(sigma[4]).to(device=device, dtype=rdtype)
    sigma5 = _jax_to_torch(sigma[5]).to(device=device, dtype=rdtype)

    h_stretch_x = _derivative_stretch(omega, alpha3, kappa3, sigma3).to(cdtype)
    h_stretch_y = _derivative_stretch(omega, alpha4, kappa4, sigma4).to(cdtype)
    h_stretch_z = _derivative_stretch(omega, alpha5, kappa5, sigma5).to(cdtype)

    curl = torch.empty(
        (
            electric_phasor.shape[0],
            3,
            electric_phasor.shape[2],
            electric_phasor.shape[3],
            electric_phasor.shape[4],
        ),
        dtype=cdtype,
        device=device,
    )

    # Metric-aware, but normalized to min_spacing.
    curl[:, 0] = h_stretch_y * (dy_ez.to(cdtype) / dy_b) - h_stretch_z * (
        dz_ey.to(cdtype) / dz_b
    )
    curl[:, 1] = h_stretch_z * (dz_ex.to(cdtype) / dz_b) - h_stretch_x * (
        dx_ez.to(cdtype) / dx_b
    )
    curl[:, 2] = h_stretch_x * (dx_ey.to(cdtype) / dx_b) - h_stretch_y * (
        dy_ex.to(cdtype) / dy_b
    )

    inv_mu = arrays.inv_permeabilities
    h_curl = _apply_inverse_permeability(inv_mu, curl)

    discrete_denominator = (
        torch.exp(1j * omega[:, None, None, None, None] * config.time_step_duration)
        - 1.0
    ).to(cdtype)

    h_curl.mul_((config.courant_number / discrete_denominator).to(h_curl.dtype))
    return h_curl.to(cdtype)


def _derive_magnetic_phasor_from_electric_new(
    electric_phasor: Tensor,
    *,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    wave_characters: Sequence[Any],
) -> Tensor:
    if electric_phasor.ndim != 5 or electric_phasor.shape[1] != 3:
        raise ValueError(
            "Expected electric_phasor shape [nfreq, 3, Nx, Ny, Nz], "
            f"got {tuple(electric_phasor.shape)}"
        )

    import jax.numpy as jnp
    from fdtdx.constants import c as c0
    from fdtdx.constants import eps0

    device = electric_phasor.device
    cdtype = electric_phasor.dtype
    rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64

    def _metric_scale_torch(axis: int, n_cells: int, stencil: str) -> Tensor:
        if not config.has_nonuniform_grid:
            shape = [1, 1, 1, 1]
            shape[axis + 1] = n_cells
            return torch.ones(tuple(shape), device=device, dtype=rdtype)

        grid = config.resolved_grid
        assert grid is not None
        widths = _jax_to_torch(grid.cell_widths(axis)).to(device=device, dtype=rdtype)
        if widths.ndim != 1 or widths.numel() != n_cells:
            raise ValueError(
                f"Axis {axis} widths have shape {tuple(widths.shape)}, expected ({n_cells},)."
            )

        if stencil == "backward":
            prev_widths = torch.cat([widths[:1], widths[:-1]])
            widths = 0.5 * (widths + prev_widths)
        elif stencil != "forward":
            raise ValueError(f"Unknown derivative stencil: {stencil}")

        reference_spacing = c0 * config.time_step_duration / config.courant_number
        scale = reference_spacing / widths

        shape = [1, 1, 1, 1]
        shape[axis + 1] = n_cells
        return scale.view(tuple(shape))

    def _build_legacy_pml_profiles(
        nx: int,
        ny: int,
        nz: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        jdtype = jnp.float32 if rdtype == torch.float32 else jnp.float64

        alpha = jnp.zeros((6, nx, ny, nz), dtype=jdtype)
        kappa = jnp.ones((6, nx, ny, nz), dtype=jdtype)
        sigma = jnp.zeros((6, nx, ny, nz), dtype=jdtype)
        electric_conductivity = jnp.zeros((nx, ny, nz), dtype=jdtype)
        magnetic_conductivity = jnp.zeros((nx, ny, nz), dtype=jdtype)

        pml_objects = getattr(objects, "pml_objects", None)
        if pml_objects is None:
            pml_objects = [
                b
                for b in getattr(objects, "boundary_objects", [])
                if hasattr(b, "modify_arrays")
            ]

        for pml in pml_objects:
            out = pml.modify_arrays(
                alpha=alpha,
                kappa=kappa,
                sigma=sigma,
                electric_conductivity=electric_conductivity,
                magnetic_conductivity=magnetic_conductivity,
            )
            alpha = out["alpha"]
            kappa = out["kappa"]
            sigma = out["sigma"]
            electric_conductivity = out["electric_conductivity"]
            magnetic_conductivity = out["magnetic_conductivity"]

        return (
            _jax_to_torch(alpha).to(device=device, dtype=rdtype),
            _jax_to_torch(kappa).to(device=device, dtype=rdtype),
            _jax_to_torch(sigma).to(device=device, dtype=rdtype),
        )

    e_pad = pad_fields_for_boundaries(electric_phasor, objects, config)

    ex = e_pad[:, 0]
    ey = e_pad[:, 1]
    ez = e_pad[:, 2]

    xs = slice(1, -1)
    ys = slice(1, -1)
    zs = slice(1, -1)

    dy_ez = ez[:, xs, 2:, zs] - ez[:, xs, ys, zs]
    dz_ey = ey[:, xs, ys, 2:] - ey[:, xs, ys, zs]

    dz_ex = ex[:, xs, ys, 2:] - ex[:, xs, ys, zs]
    dx_ez = ez[:, 2:, ys, zs] - ez[:, xs, ys, zs]

    dx_ey = ey[:, 2:, ys, zs] - ey[:, xs, ys, zs]
    dy_ex = ex[:, xs, 2:, zs] - ex[:, xs, ys, zs]

    nx, ny, nz = electric_phasor.shape[2:]
    dx_scale = _metric_scale_torch(0, nx, "forward")
    dy_scale = _metric_scale_torch(1, ny, "forward")
    dz_scale = _metric_scale_torch(2, nz, "forward")

    omega = (
        2
        * torch.pi
        * torch.as_tensor(
            [wave.get_frequency() for wave in wave_characters],
            dtype=rdtype,
            device=device,
        )
    )

    alpha, kappa, sigma = _build_legacy_pml_profiles(nx, ny, nz)

    h_stretch_x = _derivative_stretch(omega, alpha[3], kappa[3], sigma[3]).to(cdtype)
    h_stretch_y = _derivative_stretch(omega, alpha[4], kappa[4], sigma[4]).to(cdtype)
    h_stretch_z = _derivative_stretch(omega, alpha[5], kappa[5], sigma[5]).to(cdtype)

    dx_scale = dx_scale.to(cdtype)
    dy_scale = dy_scale.to(cdtype)
    dz_scale = dz_scale.to(cdtype)

    curl = torch.empty(
        (electric_phasor.shape[0], 3, nx, ny, nz), dtype=cdtype, device=device
    )

    curl[:, 0] = h_stretch_y * (dy_ez.to(cdtype) * dy_scale) - h_stretch_z * (
        dz_ey.to(cdtype) * dz_scale
    )
    curl[:, 1] = h_stretch_z * (dz_ex.to(cdtype) * dz_scale) - h_stretch_x * (
        dx_ez.to(cdtype) * dx_scale
    )
    curl[:, 2] = h_stretch_x * (dx_ey.to(cdtype) * dx_scale) - h_stretch_y * (
        dy_ex.to(cdtype) * dy_scale
    )

    h_curl = _apply_inverse_permeability(arrays.inv_permeabilities, curl)

    discrete_denominator = (
        torch.exp(1j * omega[:, None, None, None, None] * config.time_step_duration)
        - 1.0
    ).to(cdtype)

    h_curl.mul_((config.courant_number / discrete_denominator).to(h_curl.dtype))
    return h_curl.to(cdtype)


def _permittivity_gradient_from_phasors(
    *,
    forward_phasor: Any,
    adjoint_phasor: Any,
    wave_characters: Sequence[Any],
    need_interpolated_phasors: bool = False,
    forward_phasor_is_colocated: bool = False,
    adjoint_phasor_is_colocated: bool = False,
    complex_permittivity: bool = False,
    objects=None,
    config=None,
):
    """Compute the permittivity sensitivity from forward and adjoint phasors.

    Args:
        forward_phasor: Forward fields with shape ``(nfreq, component, Nx, Ny,
            Nz)``.
        adjoint_phasor: Adjoint fields with the same layout.
        wave_characters: Frequencies represented by the first axis.
        need_interpolated_phasors: Whether Yee-sampled fields need colocation.
        forward_phasor_is_colocated: Whether ``forward_phasor`` is already
            colocated.
        adjoint_phasor_is_colocated: Whether ``adjoint_phasor`` is already
            colocated.
        complex_permittivity: Return the standard complex Torch gradient for
            independent real and imaginary permittivity channels. For real
            permittivity, preserve the historical real ``dL/d epsilon``.
        objects: FDTDX object container used for Yee interpolation.
        config: FDTDX simulation configuration.

    Returns:
        A real gradient for real permittivity, or a complex gradient ``g``
        satisfying ``dL = Re(conj(g) * d_epsilon)`` for complex permittivity.
    """
    # forward_phasor (nfreq, component, Nx, Ny, Nz), adjoint_phasor (nfreq, component, Nx, Ny, Nz)
    forward_e = forward_phasor[:, :3]  # (nfreq, 3, Nx, Ny, Nz)
    adjoint_e = adjoint_phasor[:, :3]  # (nfreq, 3, Nx, Ny, Nz)

    if need_interpolated_phasors:
        interpolation_config = config
        resolved_grid = getattr(config, "resolved_grid", None)
        if resolved_grid is not None and tuple(resolved_grid.shape) != tuple(
            forward_phasor.shape[-3:]
        ):
            interpolation_config = config.aset(
                "grid", unfold_grid(resolved_grid, config.symmetry)
            ).aset("symmetry", (0, 0, 0))
        if not forward_phasor_is_colocated:
            forward_e = yee_to_colocate_interpolate(
                forward_e, objects, interpolation_config, is_E=True
            )
        if not adjoint_phasor_is_colocated:
            adjoint_e = yee_to_colocate_interpolate(
                adjoint_e, objects, interpolation_config, is_E=True
            )
    # adjoint_e = -jnp.asarray(adjoint_phasor)[:, :3]
    # grad = jnp.zeros(forward_e.shape[2:], dtype=jnp.float32)
    omegas = (
        2
        * torch.pi
        * torch.tensor(
            [wave.get_frequency() for wave in wave_characters],
            dtype=torch.complex64,
            device=forward_phasor.device,
        )
    )  # [nfreq]
    """
    https://www.flexcompute.com/assets/pdf/learning-center/inverse-design/Tutorial_02_Adjoint-Method.pdf
    page 7.
    """
    dA_deps = -(omegas**2) * constants.mu0 * constants.eps0  # [nfreq]
    overlap = (forward_e * adjoint_e).sum(1)  # [nfreq, Nx, Ny, Nz]
    sensitivity = -2 * (dA_deps[..., None, None, None] * overlap).sum(
        0
    )  # accumulate over frequencies, [Nx, Ny, Nz]
    if complex_permittivity:
        # The real-valued legacy result is Re(S). For a complex material map,
        # PyTorch expects g such that dL = Re(conj(g) * d_epsilon), hence
        # g = conj(S). This exposes independent real and imaginary channels.
        return sensitivity.conj()
    grad = sensitivity.real

    # for freq_idx, wave in enumerate(wave_characters):
    #     omega = 2 * jnp.pi * wave.get_frequency()
    #     dA_deps = -(omega**2) * constants.mu0 * constants.eps0
    #     overlap = jnp.sum(forward_e[freq_idx] * adjoint_e[freq_idx], axis=0)
    #     grad = grad + (-2.0 * jnp.real(dA_deps * overlap)).astype(jnp.float32)
    return grad


class FDTDXSolveTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        eps_r: Tensor,
        arrays: ArrayContainer,
        objects: ObjectContainer,
        config: SimulationConfig,
        key: jax.Array,
        monitor: Any,
        native_h_monitor: Any = None,
        native_h_slice: tuple[slice, slice, slice] | None = None,
        field_grid_metadata: dict[str, Any] | None = None,
        prepare_adjoint_simulation: Any = None,
        adjoint_group_info: Any = None,
    ):
        _, arrays = fdtdx.run_fdtd(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            stopping_condition=fdtdx.DetectorDecayAfterSourceCondition(
                detector_name="decay",
                decay_by=_DECAY_BY,
                source_decay_by=_DECAY_BY,
                detector_interval=1000,
            ),
            show_progress=True,
        )
        detector_phasors = extract_phasor_detector_states(arrays)
        ctx.full_spatial_shape = tuple(int(size) for size in eps_r.shape)

        if _MAPS_SPARSE_PHASOR:
            sources = objects.sources
            native_outputs = {}
            for detector_name, phasor in detector_phasors.items():
                detector = objects[detector_name]
                if _MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM and len(sources) > 0:
                    native_outputs[detector_name], _ = (
                        _normalize_phasor_by_source_spectrum(
                            phasor,
                            source_template=sources[0],
                            monitor=detector,
                            config=config,
                        )
                    )
                else:
                    native_outputs[detector_name] = phasor

            ctx.eps_device = eps_r.device
            ctx.eps_dtype = eps_r.dtype
            ctx.complex_permittivity = torch.is_complex(eps_r)
            # ctx.arrays = arrays
            ctx.objects = objects
            ctx.config = config
            ctx.key = key
            ctx.prepare_adjoint_simulation = prepare_adjoint_simulation
            ctx.adjoint_group_info = adjoint_group_info
            ctx.phasor_spectrum = None
            unfolded_outputs = {
                detector_name: unfold_phasor(
                    detector_fields,
                    monitor=objects[detector_name],
                    components=tuple(objects[detector_name].components),
                    config=config,
                )
                for detector_name, detector_fields in native_outputs.items()
            }
            dense_forward_phasor = _sparse_phasors_to_dense(
                unfolded_outputs,
                objects=objects,
                monitor=monitor,
                spatial_shape=eps_r.shape,
                config=config,
                device=eps_r.device,
                dtype=eps_r.dtype,
            )
            monitor_template = (
                monitor[0] if isinstance(monitor, (tuple, list)) else monitor
            )
            ctx.fwd_monitor = objects[monitor_template.name]
            if dense_forward_phasor.ndim != 5:
                raise RuntimeError(
                    "Sparse FDTD dense forward phasor must have shape "
                    f"(frequency, component, x, y, z), got {tuple(dense_forward_phasor.shape)}"
                )
            if dense_forward_phasor.shape[1] != len(_COMPONENTS):
                raise RuntimeError(
                    "Sparse FDTD dense forward phasor must contain all six field "
                    f"components, got {dense_forward_phasor.shape[1]}"
                )
            if tuple(dense_forward_phasor.shape[-3:]) != tuple(eps_r.shape):
                raise RuntimeError(
                    "Sparse FDTD dense forward phasor must cover the full eps grid, "
                    f"got {tuple(dense_forward_phasor.shape[-3:])}, expected {tuple(eps_r.shape)}"
                )
            ctx.forward_phasor = dense_forward_phasor
            if _MAPS_COMPARE_SPARSE_DENSE:
                if dense_forward_phasor.shape[0] != ctx.forward_phasor.shape[0]:
                    print(
                        "[fdtdx sparse-dense] shape mismatch: "
                        f"full={tuple(dense_forward_phasor.shape)} "
                        f"sparse={tuple(ctx.forward_phasor.shape)}",
                        flush=True,
                    )
                else:
                    errors = []
                    reference_values = []
                    for detector_name, detector_fields in unfolded_outputs.items():
                        if detector_fields.ndim != 5:
                            continue
                        detector = objects[detector_name]
                        grid_slice = getattr(detector, "grid_slice", None)
                        if grid_slice is None:
                            continue
                        spatial_slices = _unfolded_detector_spatial_slices(
                            detector,
                            detector_fields,
                            dense_forward_phasor.shape[2:],
                            config,
                        )
                        mapped_shape = tuple(
                            spatial_slice.stop - spatial_slice.start
                            for spatial_slice in spatial_slices
                        )
                        print(
                            "[fdtdx sparse-dense] detector="
                            f"{detector_name!r}, components={tuple(detector.components)!r}, "
                            f"native_grid_slice={grid_slice!r}, "
                            f"unfolded_shape={tuple(detector_fields.shape[2:])}, "
                            f"reference_slice={tuple(spatial_slices)!r}, "
                            f"reference_shape={mapped_shape!r}",
                            flush=True,
                        )
                        for component_index, component in enumerate(
                            detector.components
                        ):
                            if component not in _COMPONENTS:
                                continue
                            dense_value = dense_forward_phasor[
                                : detector_fields.shape[0],
                                _COMPONENTS.index(component),
                                *spatial_slices,
                            ]
                            detector_value = detector_fields[:, component_index]
                            if dense_value.shape != detector_value.shape:
                                print(
                                    "[fdtdx sparse-dense] component shape mismatch: "
                                    f"detector={detector_name!r}, component={component!r}, "
                                    f"dense={tuple(dense_value.shape)}, "
                                    f"sparse={tuple(detector_value.shape)}",
                                    flush=True,
                                )
                                continue
                            difference = dense_value - detector_value
                            reference_values.append(dense_value)
                            errors.append(difference)
                            ref_norm = torch.linalg.vector_norm(
                                dense_value.reshape(dense_value.shape[0], -1),
                                dim=1,
                            )
                            error_norm = torch.linalg.vector_norm(
                                difference.reshape(difference.shape[0], -1),
                                dim=1,
                            )
                            print(
                                "[fdtdx sparse-dense] component="
                                f"{component}: max_abs={float(difference.abs().max().detach().cpu()):.6e}, "
                                f"rms={float(torch.sqrt(torch.mean(difference.abs() ** 2)).detach().cpu()):.6e}, "
                                f"relative_l2={float((error_norm / ref_norm.clamp_min(1e-30)).max().detach().cpu()):.6e}",
                                flush=True,
                            )
                            if _MAPS_WRITE_SPARSE_DENSE_PLOTS:
                                plot_sparse_dense_comparison(
                                    filepath=(
                                        Path("figs/fdtdx_sparse_dense_compare")
                                        / f"{detector_name}_{component}.png"
                                    ),
                                    reference=dense_value,
                                    sparse=detector_value,
                                    detector_name=detector_name,
                                    component=component,
                                )
                    if not errors:
                        print(
                            "[fdtdx sparse-dense] no spatial detector samples",
                            flush=True,
                        )
                    else:
                        difference = torch.cat([error.reshape(-1) for error in errors])
                        reference_sample = torch.cat(
                            [value.reshape(-1) for value in reference_values]
                        )
                        max_error = float(difference.abs().max().detach().cpu())
                        reference_norm = float(
                            reference_sample.abs().max().detach().cpu()
                        )
                        relative_error = max_error / max(reference_norm, 1e-30)
                        print(
                            "[fdtdx sparse-dense] "
                            f"max_abs={max_error:.6e}, "
                            f"relative={relative_error:.6e}",
                            flush=True,
                        )
            return ctx.forward_phasor, arrays

        if monitor.name not in detector_phasors:
            raise KeyError(f"Monitor {monitor.name!r} has no recorded phasor state.")
        primary_phasor = detector_phasors[monitor.name]

        for name, state in arrays.detector_states.items():
            if name != monitor.name and "mode" in name:
                det = objects[name]
                overlap = det.compute_overlap(state)
                print(f"[DEBUG] Detector {name} overlap: {overlap}", flush=True)
        # if config.has_symmetry:
        #     E_fields = fdtdx.unfold_fields(field=E_fields, symmetry=config.symmetry, field_type="E")
        #     arrays = fdtdx.unfold_detector_states(arrays, objects, config)

        raw_outputs = primary_phasor
        sources = objects.sources
        if _MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM and len(sources) > 0:
            native_outputs, phasor_spectrum = _normalize_phasor_by_source_spectrum(
                raw_outputs,
                source_template=sources[0],
                monitor=monitor,
                config=config,
            )
        else:
            native_outputs = raw_outputs
            phasor_spectrum = torch.ones(
                raw_outputs.shape[0],
                dtype=raw_outputs.dtype,
                device=raw_outputs.device,
            )

        ctx.eps_device = eps_r.device
        ctx.eps_dtype = eps_r.dtype
        ctx.complex_permittivity = torch.is_complex(eps_r)
        # ctx.arrays = arrays
        ctx.objects = objects
        ctx.config = config
        ctx.key = key
        ctx.fwd_monitor = monitor
        ctx.prepare_adjoint_simulation = prepare_adjoint_simulation
        ctx.adjoint_group_info = adjoint_group_info
        ctx.phasor_spectrum = phasor_spectrum
        ctx.forward_phasor = native_outputs
        return native_outputs, arrays

    @staticmethod
    def backward(ctx, *grad_outputs):
        if grad_outputs[0] is None:
            # if len(grad_outputs) != 1 or grad_outputs[0] is None:
            return None, None, None, None, None, None, None, None, None, None, None

        # phasor_spectrum = ctx.phasor_spectrum.to(grad_outputs[0].device)
        # denom = phasor_spectrum.conj().reshape(
        #     (phasor_spectrum.shape[0],) + (1,) * (grad_outputs[0].ndim - 1)
        # )
        # if _MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM:
        #     raw_cotangents = grad_outputs[0] / denom
        # else:
        #     raw_cotangents = grad_outputs[0]
        raw_cotangents = grad_outputs[0]
        if _MAPS_SPARSE_PHASOR:
            adjoint_cotangents = _pullback_dense_cotangents_like_reference(
                raw_cotangents,
                monitor=ctx.fwd_monitor,
                objects=ctx.objects,
                config=ctx.config,
            )
        else:
            adjoint_cotangents = raw_cotangents
        if not _MAPS_SPARSE_PHASOR and getattr(
            ctx.fwd_monitor, "exact_interpolation", False
        ):
            adjoint_cotangents = _pullback_colocated_cotangents_to_yee(
                adjoint_cotangents,
                tuple(ctx.fwd_monitor.components),
                objects=ctx.objects,
                config=ctx.config,
            )
        ### first we need to determine how to run adjoint simulation
        # per single frequency for all ports
        #   or per spatial group (same port) for corresonding frequencies
        # whichever is smaller number of runs.
        ### [Warning] RULE OF THUMB: Cannot run broadband simulation if the spatial locations of the gradients for different frequencies do not overlap.
        # Otherwise, the unwanted freq components from the broadband sources will all be recorded and mixed up by the phasor detector
        groups = build_adjoint_groups_from_monitor_info(
            adjoint_cotangents,
            ctx.adjoint_group_info,
            threshold=0.0,
        )
        group_source = "monitor metadata"
        if len(groups) == 0:
            groups = group_adjoint_spatial_support(adjoint_cotangents, threshold=0.0)
            group_source = "cotangent support fallback"
        n_freq = adjoint_cotangents.shape[0]
        n_groups = len(groups)
        fwd_wls_um = [
            float(wave.get_wavelength() / 1e-6)
            for wave in ctx.fwd_monitor.wave_characters
        ]
        group_summary = {
            name: [int(i) for i in group["freq_indices"].detach().cpu().tolist()]
            for name, group in groups.items()
        }
        if _WRITE_ADJOINT_DEBUG_PLOTS:
            print(
                "[fdtdx-adjoint] "
                f"n_freq={n_freq}, fwd_wls_um={fwd_wls_um}, "
                f"group_source={group_source}, groups={group_summary}",
                flush=True,
            )

        broadband_plan = None
        if (
            _MAPS_ENABLE_BROADBAND_ADJOINT
            and n_groups > 0
            and n_freq > n_groups
            and _MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM
        ):
            broadband_plan = _plan_broadband_adjoint_groups(
                cotangents=adjoint_cotangents,
                groups=groups,
                monitor=ctx.fwd_monitor,
                source_template=ctx.objects.sources[0],
                config=ctx.config,
            )

        if _MAPS_SPARSE_PHASOR:
            # The full pullback is required only for the existing grouping
            # decision. Per-run pullbacks below stay alive through FDTD-X so
            # DLPack-backed source profiles cannot outlive their Torch owner.
            del adjoint_cotangents
            cotangents = None
            if raw_cotangents.is_cuda:
                torch.cuda.empty_cache()
        else:
            cotangents = _torch_to_jax(adjoint_cotangents)

        adjoint_reduced_shape: tuple[int, int, int] | None = None
        adjoint_symmetry: tuple[int, int, int] | None = None
        if broadband_plan is None:
            # one adjoint run per frequency, with sources at all relevant locations for that frequency.
            print(f"Running independent adjoints: n_freq={n_freq}, n_groups={n_groups}")
            # Run adjoint simulation per frequency
            grad_eps = None
            adjoint_sources_by_freq: list[Tensor] = []
            adjoint_phasors_by_freq: list[Tensor] = []
            grad_eps_by_freq: list[Tensor] = []
            adjoint_components = ("Ex", "Ey", "Ez")
            for freq_idx in range(n_freq):
                adj_monitor = _clone_phasor_detector_for_frequency(
                    ctx.fwd_monitor,
                    freq_idx=freq_idx,
                    components=adjoint_components,
                )
                if _MAPS_SPARSE_PHASOR:
                    selected_waves = (ctx.fwd_monitor.wave_characters[freq_idx],)
                    source_cotangents = _pullback_dense_cotangents_like_reference(
                        raw_cotangents[freq_idx : freq_idx + 1],
                        monitor=ctx.fwd_monitor,
                        objects=ctx.objects,
                        config=ctx.config,
                        wave_characters=selected_waves,
                    )
                    adj_monitor = clone_phasor_detector(
                        adj_monitor,
                        name=adj_monitor.name,
                        components=adjoint_components,
                        exact_interpolation=True,
                    )
                else:
                    source_cotangents = cotangents[freq_idx : freq_idx + 1]
                adj_source = _build_distributed_adjoint_source(
                    adj_monitor=adj_monitor,
                    source_template=ctx.objects.sources[0],
                    cotangents=source_cotangents,
                    center_on_monitor=True,
                    adjoint_group_info=ctx.adjoint_group_info,
                    global_freq_idx=freq_idx,
                    source_partial_grid_shape=(
                        ctx.full_spatial_shape if _MAPS_SPARSE_PHASOR else None
                    ),
                )
                adj_arrays, adj_objects, adj_config, adj_key, adj_monitor = (
                    ctx.prepare_adjoint_simulation(
                        source=adj_source,
                        source_grid_margins=(
                            tuple((0, int(size)) for size in ctx.full_spatial_shape)
                            if _MAPS_SPARSE_PHASOR
                            else _object_grid_slice_bounds(ctx.fwd_monitor)
                        ),
                        monitor=adj_monitor,
                    )
                )
                adjoint_reduced_shape = tuple(
                    int(size) for size in adj_arrays.inv_permittivities.shape[-3:]
                )
                adjoint_symmetry = tuple(int(value) for value in adj_config.symmetry)

                _, adj_arrays = fdtdx.run_fdtd(
                    arrays=adj_arrays,
                    objects=adj_objects,
                    config=adj_config,
                    key=adj_key,
                    stopping_condition=fdtdx.DetectorDecayAfterSourceCondition(
                        detector_name="decay",
                        decay_by=_DECAY_BY,
                        source_decay_by=_DECAY_BY,
                        detector_interval=1000,
                    ),
                    show_progress=True,
                )
                if _MAPS_SPARSE_PHASOR:
                    adjoint_phasor = _adjoint_phasors_to_dense(
                        adj_arrays,
                        objects=adj_objects,
                        monitor=adj_monitor,
                        source_template=adj_source,
                        spatial_shape=ctx.full_spatial_shape,
                        config=adj_config,
                    )
                else:
                    adjoint_phasor = _jax_to_torch(
                        adj_arrays.detector_states[adj_monitor.name]["phasor"]
                    )[0]
                    if _MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM:
                        adjoint_phasor, _ = (
                            _post_normalize_adjoint_phasor_by_source_spectrum(
                                adjoint_phasor,
                                source_template=adj_source,
                                monitor=adj_monitor,
                                config=adj_config,
                            )
                        )
                if _WRITE_ADJOINT_DEBUG_PLOTS:
                    adjoint_source_profile = _source_profile_for_debug(
                        adj_source,
                        grid_shape=adjoint_phasor.shape[-3:],
                    ).to(device=adjoint_phasor.device, dtype=adjoint_phasor.dtype)

                single_grad = _permittivity_gradient_from_phasors(
                    forward_phasor=ctx.forward_phasor[freq_idx : freq_idx + 1],
                    adjoint_phasor=adjoint_phasor,
                    wave_characters=(ctx.fwd_monitor.wave_characters[freq_idx],),
                    objects=ctx.objects,
                    config=ctx.config,
                    need_interpolated_phasors=True,
                    forward_phasor_is_colocated=_MAPS_SPARSE_PHASOR
                    or getattr(ctx.fwd_monitor, "exact_interpolation", False),
                    adjoint_phasor_is_colocated=True,
                    complex_permittivity=ctx.complex_permittivity,
                )
                if _WRITE_ADJOINT_DEBUG_PLOTS:
                    adjoint_sources_by_freq.append(adjoint_source_profile)
                    adjoint_phasors_by_freq.append(adjoint_phasor)
                    grad_eps_by_freq.append(single_grad)
                grad_eps = single_grad if grad_eps is None else grad_eps + single_grad
                # Do not let this full simulation state overlap preparation of
                # the next frequency. Debug lists above intentionally retain
                # only the tensors required for the requested diagnostics.
                del adj_arrays, adj_objects, adj_config, adj_key
                del adj_source, adj_monitor, adjoint_phasor
                del single_grad
                del source_cotangents
                if _WRITE_ADJOINT_DEBUG_PLOTS:
                    del adjoint_source_profile
            if _WRITE_ADJOINT_DEBUG_PLOTS:
                _write_adjoint_debug_plot(
                    monitor=ctx.fwd_monitor,
                    phasor_gradients=raw_cotangents,
                    forward_phasors=ctx.forward_phasor,
                    adjoint_sources=torch.cat(adjoint_sources_by_freq, dim=0),
                    adjoint_phasors=torch.cat(adjoint_phasors_by_freq, dim=0),
                    grad_eps_by_freq=torch.stack(grad_eps_by_freq, dim=0),
                )
        else:
            # Each accepted spatial group injects its smooth profile expansion
            # in one run. Arbitrary complex VJP weights are restored after DTFT
            # normalization and before the existing material-gradient kernel.
            print(
                f"[Experimental] Running adjoint simulation per spatial group: n_freq={n_freq} > n_groups={n_groups}"
            )
            grad_eps = None
            broadband_debug_sources: list[Tensor] = []
            broadband_debug_forward_phasors: list[Tensor] = []
            broadband_debug_phasors: list[Tensor] = []
            broadband_debug_gradients: list[Tensor] = []
            broadband_debug_waves: list[Any] = []
            broadband_debug_phasor_gradients: list[Tensor] = []
            adjoint_components = ("Ex", "Ey", "Ez")
            ## we run adjoint simulation per spatial group, each group has multiple frequencies but same spatial support. this can save some runs when there are many frequencies but only few spatial supports.
            for group_name, group in groups.items():
                # freq_indices = group["freq_indices"]
                # spatial_indices = group["spatial_indices"]
                if _MAPS_SPARSE_PHASOR:
                    group_freq_indices = group["freq_indices"].to(raw_cotangents.device)
                    selected_waves = tuple(
                        ctx.fwd_monitor.wave_characters[int(index)]
                        for index in group_freq_indices.tolist()
                    )
                    selected_cotangents = _pullback_dense_cotangents_like_reference(
                        raw_cotangents.index_select(0, group_freq_indices),
                        monitor=ctx.fwd_monitor,
                        objects=ctx.objects,
                        config=ctx.config,
                        wave_characters=selected_waves,
                    )
                    local_group = {
                        "freq_indices": torch.arange(
                            group_freq_indices.numel(),
                            dtype=torch.long,
                            device=raw_cotangents.device,
                        ),
                        "spatial_indices": group["spatial_indices"],
                    }
                    cotangents_group = extract_group_grad_compact_torch(
                        selected_cotangents, group=local_group
                    )
                    del selected_cotangents, local_group
                else:
                    cotangents_group = extract_group_grad_compact_torch(
                        adjoint_cotangents, group=group
                    )
                source_cotangents_group = cotangents_group
                freq_indices = tuple(int(i) for i in group["freq_indices"].tolist())
                adj_monitor = _clone_phasor_detector_for_frequency(
                    ctx.fwd_monitor,
                    freq_idx=freq_indices,
                    components=adjoint_components,
                )
                if _MAPS_SPARSE_PHASOR:
                    adj_monitor = clone_phasor_detector(
                        adj_monitor,
                        name=adj_monitor.name,
                        components=adjoint_components,
                        exact_interpolation=True,
                    )
                source_kwargs = dict(
                    adj_monitor=adj_monitor,
                    source_template=ctx.objects.sources[0],
                    cotangents=source_cotangents_group,
                    symmetry=ctx.config.symmetry,
                    source_partial_grid_shape=(
                        ctx.full_spatial_shape if _MAPS_SPARSE_PHASOR else None
                    ),
                )
                if _BROADBAND_ADJOINT_METHOD == "legacy":
                    adj_source, adjoint_vjp_scale = (
                        _build_separable_broadband_adjoint_source(**source_kwargs)
                    )
                else:
                    adj_source, adjoint_vjp_scale, _ = (
                        _build_factored_chebyshev_adjoint_source(
                            **source_kwargs, order=broadband_plan[group_name]
                        )
                    )
                adj_arrays, adj_objects, adj_config, adj_key, adj_monitor = (
                    ctx.prepare_adjoint_simulation(
                        source=adj_source,
                        source_grid_margins=(
                            tuple((0, int(size)) for size in ctx.full_spatial_shape)
                            if _MAPS_SPARSE_PHASOR
                            else _object_grid_slice_bounds(ctx.fwd_monitor)
                        ),
                        monitor=adj_monitor,
                    )
                )
                adjoint_reduced_shape = tuple(
                    int(size) for size in adj_arrays.inv_permittivities.shape[-3:]
                )
                adjoint_symmetry = tuple(int(value) for value in adj_config.symmetry)

                _, adj_arrays = fdtdx.run_fdtd(
                    arrays=adj_arrays,
                    objects=adj_objects,
                    config=adj_config,
                    key=adj_key,
                    stopping_condition=fdtdx.DetectorDecayAfterSourceCondition(
                        detector_name="decay",
                        decay_by=_DECAY_BY,
                        source_decay_by=_DECAY_BY,
                        detector_interval=1000,
                    ),
                    show_progress=True,
                )
                if _MAPS_SPARSE_PHASOR:
                    adjoint_phasor = _adjoint_phasors_to_dense(
                        adj_arrays,
                        objects=adj_objects,
                        monitor=adj_monitor,
                        source_template=adj_source,
                        spatial_shape=ctx.full_spatial_shape,
                        config=adj_config,
                    )
                else:
                    adjoint_phasor = _jax_to_torch(
                        adj_arrays.detector_states[adj_monitor.name]["phasor"]
                    )[0]
                    if _MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM:
                        adjoint_phasor, _ = (
                            _post_normalize_adjoint_phasor_by_source_spectrum(
                                adjoint_phasor,
                                source_template=adj_source,
                                monitor=adj_monitor,
                                config=adj_config,
                            )
                        )
                adjoint_phasor = adjoint_phasor * adjoint_vjp_scale.reshape(
                    (adjoint_vjp_scale.shape[0],) + (1,) * (adjoint_phasor.ndim - 1)
                )

                if _WRITE_ADJOINT_DEBUG_PLOTS:
                    adjoint_source_profile = _source_profile_for_debug(
                        adj_source,
                        grid_shape=adjoint_phasor.shape[-3:],
                    ).to(device=adjoint_phasor.device, dtype=adjoint_phasor.dtype)
                cur_grad_eps = _permittivity_gradient_from_phasors(
                    forward_phasor=ctx.forward_phasor.index_select(
                        0, group["freq_indices"]
                    ),
                    adjoint_phasor=adjoint_phasor,
                    wave_characters=[
                        ctx.fwd_monitor.wave_characters[i.item()]
                        for i in group["freq_indices"]
                    ],
                    objects=ctx.objects,
                    config=ctx.config,
                    need_interpolated_phasors=True,
                    forward_phasor_is_colocated=_MAPS_SPARSE_PHASOR
                    or getattr(ctx.fwd_monitor, "exact_interpolation", False),
                    adjoint_phasor_is_colocated=True,
                    complex_permittivity=ctx.complex_permittivity,
                )
                grad_eps = cur_grad_eps if grad_eps is None else grad_eps + cur_grad_eps

                if _WRITE_ADJOINT_DEBUG_PLOTS:
                    group_freq_indices = tuple(
                        int(i) for i in group["freq_indices"].tolist()
                    )
                    center_idx = len(group_freq_indices) // 2
                    center_freq_idx = group_freq_indices[center_idx]
                    center_scale = adjoint_vjp_scale[center_idx]
                    profile_index = (
                        center_idx if adjoint_source_profile.shape[0] > 1 else 0
                    )
                    broadband_debug_sources.append(
                        (
                            adjoint_source_profile[profile_index : profile_index + 1]
                            * center_scale
                        ).detach()
                    )
                    broadband_debug_forward_phasors.append(
                        ctx.forward_phasor[
                            center_freq_idx : center_freq_idx + 1
                        ].detach()
                    )
                    broadband_debug_phasors.append(
                        adjoint_phasor[center_idx : center_idx + 1].detach()
                    )
                    broadband_debug_gradients.append(
                        _permittivity_gradient_from_phasors(
                            forward_phasor=ctx.forward_phasor[
                                center_freq_idx : center_freq_idx + 1
                            ],
                            adjoint_phasor=adjoint_phasor[center_idx : center_idx + 1],
                            wave_characters=(
                                ctx.fwd_monitor.wave_characters[center_freq_idx],
                            ),
                            objects=ctx.objects,
                            config=ctx.config,
                            need_interpolated_phasors=True,
                            forward_phasor_is_colocated=_MAPS_SPARSE_PHASOR
                            or getattr(ctx.fwd_monitor, "exact_interpolation", False),
                            adjoint_phasor_is_colocated=True,
                            complex_permittivity=ctx.complex_permittivity,
                        ).unsqueeze(0)
                    )
                    broadband_debug_waves.append(
                        ctx.fwd_monitor.wave_characters[center_freq_idx]
                    )
                    broadband_debug_phasor_gradients.append(
                        raw_cotangents[center_freq_idx : center_freq_idx + 1].detach()
                    )

                # A later spatial group creates another complete FDTD-X state;
                # release this group's state before constructing that state.
                del adj_arrays, adj_objects, adj_config, adj_key
                del adj_source, adj_monitor, adjoint_phasor, adjoint_vjp_scale
                del cur_grad_eps
                del cotangents_group, source_cotangents_group
                if _WRITE_ADJOINT_DEBUG_PLOTS:
                    del adjoint_source_profile

            if _WRITE_ADJOINT_DEBUG_PLOTS and broadband_debug_phasors:
                _write_adjoint_debug_plot(
                    monitor=ctx.fwd_monitor,
                    phasor_gradients=torch.cat(broadband_debug_phasor_gradients, dim=0),
                    forward_phasors=torch.cat(broadband_debug_forward_phasors, dim=0),
                    adjoint_sources=torch.cat(broadband_debug_sources, dim=0),
                    adjoint_phasors=torch.cat(broadband_debug_phasors, dim=0),
                    grad_eps_by_freq=torch.cat(broadband_debug_gradients, dim=0),
                    wave_characters=broadband_debug_waves,
                )

        if adjoint_reduced_shape is None or adjoint_symmetry is None:
            raise RuntimeError("Adjoint simulation produced no gradient contribution.")
        if not _MAPS_SPARSE_PHASOR:
            grad_eps = unfold_array(
                arr=grad_eps, symmetry=adjoint_symmetry, spatial_axes=(0, 1, 2)
            )
        else:
            grad_shape = tuple(int(size) for size in grad_eps.shape)
            if grad_shape == adjoint_reduced_shape:
                grad_eps = unfold_array(
                    arr=grad_eps, symmetry=adjoint_symmetry, spatial_axes=(0, 1, 2)
                )
            elif grad_shape != ctx.full_spatial_shape:
                raise ValueError(
                    "Adjoint gradient has unexpected spatial shape: "
                    f"got={grad_shape}, reduced={adjoint_reduced_shape}, "
                    f"full={ctx.full_spatial_shape}"
                )
        # print(grad_eps.shape, grad_eps.dtype)
        # grad_eps = _to_torch(grad_eps, device=ctx.eps_device).to(ctx.eps_dtype)
        return grad_eps, None, None, None, None, None, None, None, None, None, None


class FDTDXSolveTorch(nn.Module):
    """Small module wrapper around the fdtdx custom autograd function."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        eps_r: Tensor,
        arrays: ArrayContainer,
        objects: ObjectContainer,
        config: SimulationConfig,
        key: jax.Array,
        monitor: Any,
        native_h_monitor: Any = None,
        native_h_slice: tuple[slice, slice, slice] | None = None,
        field_grid_metadata: dict[str, Any] | None = None,
        prepare_adjoint_simulation: Any = None,
        adjoint_group_info: Any = None,
    ) -> tuple[Tensor, ArrayContainer]:
        return FDTDXSolveTorchFunction.apply(
            eps_r,
            arrays,
            objects,
            config,
            key,
            monitor,
            native_h_monitor,
            native_h_slice,
            field_grid_metadata,
            prepare_adjoint_simulation,
            adjoint_group_info,
        )
