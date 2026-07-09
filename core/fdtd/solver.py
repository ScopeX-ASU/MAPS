"""
Torch autodiff bridge for fdtdx-backed FDTD solves.

The MAPS side keeps the same high-level contract as the FDFD sparse solver:
forward returns field tensors and backward returns a gradient with respect to
the external permittivity grid.  The concrete fdtdx runtime is intentionally
injected, because scene construction, monitor placement, and adjoint-source
construction are device/application specific.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from itertools import count
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
from fdtdx.objects.sources.distributed import DistributedCurrentSource
from fdtdx.objects.sources.profile import GaussianPulseProfile, SingleFrequencyProfile
from torch import Tensor, nn

from core.utils import (
    _jax_to_torch,
    _torch_to_jax,
    pad_fields_for_boundaries,
    yee_to_colocate_interpolate,
)

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
SCALE = 3e-12
_MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM = (
    os.environ.get("MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM", "1") == "1"
)
_WRITE_ADJOINT_DEBUG_PLOTS = os.environ.get("MAPS_FDTDX_ADJOINT_DEBUG", "0") == "1"
_MAPS_ENABLE_BROADBAND_ADJOINT = (
    os.environ.get("MAPS_ENABLE_BROADBAND_ADJOINT", "0") == "1"
)
_MAPS_ENABLE_PHASOR_H = os.environ.get("MAPS_ENABLE_PHASOR_H", "0") == "1"
_MAPS_PHASOR_H_NATIVE_SLAB_CELLS = int(
    os.environ.get("MAPS_PHASOR_H_NATIVE_SLAB_CELLS", "3")
)
if not _MAPS_ENABLE_PHASOR_H:
    logging.warning(
        "MAPS_ENABLE_PHASOR_H is disabled. H phasors are analytically derived from the E phasors."
        + " H fields on the source plane are still recorded by an extra PhasorDetector."
        + " This can potentially speedup FDTD and save memory."
    )
if _MAPS_ENABLE_BROADBAND_ADJOINT:
    logging.warning(
        "_MAPS_ENABLE_BROADBAND_ADJOINT is enabled. This is an experimental feature that may produce incorrect gradients."
        + " Make sure you understand that each spatial group need to have the same spatial profile from the same port and from a broadband objective."
        + " Please compare with the single-freq adjoint gradients to verify."
    )
_ADJOINT_DEBUG_DIR = os.environ.get(
    "MAPS_FDTDX_ADJOINT_DEBUG_DIR", "figs/fdtdx_adjoint_debug"
)
_ADJOINT_DEBUG_DIR = Path(_ADJOINT_DEBUG_DIR)
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


def phasor_detector_components() -> tuple[str, ...]:
    return _COMPONENTS if _MAPS_ENABLE_PHASOR_H else ("Ex", "Ey", "Ez")


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

        spatial_indices = _slice_to_spatial_indices(monitor_slice).to(grad.device)
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
    """Analytic Gaussian-pulse coefficient seen by an fdtdx phasor detector.

    The detector demodulates real fields with exp(+i omega t).  For an fdtdx
    Gaussian pulse g(t) cos(omega0 t + phi0), this returns the diagonal
    coefficient at each requested omega.  It compensates pulse spectrum only;
    it does not de-mix finite-window cross-frequency leakage.
    """
    profile = getattr(source_or_profile, "temporal_profile", source_or_profile)
    if not isinstance(profile, GaussianPulseProfile):
        return torch.ones(
            len(wave_characters),
            dtype=dtype,
            device=device,
        )

    device = torch.device("cpu") if device is None else torch.device(device)
    real_dtype = torch.float64
    dt = torch.tensor(float(config.time_step_duration), dtype=real_dtype, device=device)
    omega = torch.tensor(
        [2 * np.pi * wave.get_frequency() for wave in wave_characters],
        dtype=real_dtype,
        device=device,
    )
    omega0 = torch.tensor(
        2 * np.pi * profile.center_wave.get_frequency(),
        dtype=real_dtype,
        device=device,
    )
    spectral_width_hz = torch.tensor(
        profile.spectral_width.get_frequency(),
        dtype=real_dtype,
        device=device,
    )
    sigma_t = 1.0 / (2 * torch.pi * spectral_width_hz)
    t0 = 6.0 * sigma_t
    phi0 = torch.tensor(
        float(getattr(profile.center_wave, "phase_shift", 0.0)),
        dtype=real_dtype,
        device=device,
    )

    prefactor = 0.5 * (2 * torch.pi) ** 0.5 * sigma_t
    pos = torch.exp(-0.5 * (sigma_t * (omega - omega0)) ** 2) * torch.exp(
        1j * ((omega - omega0) * t0 - phi0)
    )
    neg = torch.exp(-0.5 * (sigma_t * (omega + omega0)) ** 2) * torch.exp(
        1j * ((omega + omega0) * t0 + phi0)
    )
    spectrum = prefactor * (pos + neg)

    scaling_mode = getattr(detector, "scaling_mode", "continuous")
    if scaling_mode == "pulse":
        # PhasorDetector pulse mode samples every switch.interval steps and
        # multiplies by switch.interval / 2.  The sampled sum is approximated
        # by integral / (switch.interval * dt), leaving 1 / (2 * dt).
        detector_scale = 1.0 / (2.0 * dt)
    else:
        detector_scale = 1.0
    spectrum = spectrum * detector_scale
    if normalize_to_max:
        spectrum = spectrum / spectrum.abs().max()

    if include_static_amplitude:
        spectrum = spectrum * float(
            getattr(source_or_profile, "static_amplitude_factor", 1.0)
        )
        spectrum = spectrum * float(getattr(source_or_profile, "amplitude", 1.0))

    return spectrum.to(dtype=dtype)


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

    nx, ny, nz = eps_grad_np.shape[1:]
    z_idx = nz // 2 if plane_indices[0] is None else int(plane_indices[0])
    y_idx = ny // 2 if plane_indices[1] is None else int(plane_indices[1])
    x_idx = nx // 2 if plane_indices[2] is None else int(plane_indices[2])
    plane_titles = (
        f"x-y, z={z_idx}",
        f"x-z, y={y_idx}",
        f"y-z, x={x_idx}",
    )

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
        source_e_norm = np.linalg.norm(src_np[:, :3], axis=1, keepdims=True)
        quantity_specs.append(
            ("adjoint source", src_np, field_comp_idx, np.real, "RdBu_r")
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
            views = _plane_views(vol, z_idx=z_idx, y_idx=y_idx, x_idx=x_idx)
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


def _write_adjoint_debug_plot(
    *,
    monitor: Any,
    phasor_gradients: Tensor,
    forward_phasors: Tensor | None = None,
    adjoint_sources: Tensor | None = None,
    adjoint_phasors: Tensor,
    grad_eps_by_freq: Tensor,
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
        wave_characters=monitor.wave_characters,
        gradient_component=component,
        field_component=component,
        max_freqs=min(4, len(monitor.wave_characters)),
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


def _build_distributed_adjoint_source(
    *,
    adj_monitor: Any,
    source_template: Any,
    cotangents: Any,
    center_on_monitor: bool = False,
) -> DistributedCurrentSource:
    # Cotangents must be Yee-grid dL/dE and dL/dH. Use exact_interpolation=False
    # in PhasorDetector to disable interpolation.
    components = tuple(getattr(adj_monitor, "components", _COMPONENTS))
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
        partial_grid_shape=adj_monitor.partial_grid_shape,
        partial_real_position=(0, 0, 0),
        wave_characters=adj_monitor.wave_characters,
        temporal_profile=temporal_profile,
        electric_profiles=electric_profiles,
        magnetic_profiles=magnetic_profiles,
        static_amplitude_factor=SCALE,
    )

    return source


def _build_separable_broadband_adjoint_source(
    *,
    adj_monitor: Any,
    source_template: Any,
    cotangents: Tensor,
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
        partial_grid_shape=adj_monitor.partial_grid_shape,
        partial_real_position=(0, 0, 0),
        wave_characters=(center_wave,),  # should use the same as forward simulation
        temporal_profile=temporal_profile,
        electric_profiles=electric_profiles,
        magnetic_profiles=magnetic_profiles,
        static_amplitude_factor=SCALE,
    )
    return source, alpha


def clone_phasor_detector(
    monitor: Any,
    *,
    freq_idx: int | Tuple[int, ...] | None = None,
    components: Sequence[str] | None = None,
    name: str | None = None,
    partial_grid_shape: tuple[int | None, int | None, int | None] | None = None,
) -> Any:
    if isinstance(freq_idx, int):
        freq_idx = (freq_idx,)
    """Return a detector clone with optional component and frequency selection."""
    monitor = fdtdx.PhasorDetector(
        name=monitor.name if name is None else name,
        partial_grid_shape=(
            monitor.partial_grid_shape
            if partial_grid_shape is None
            else partial_grid_shape
        ),
        wave_characters=(
            tuple(monitor.wave_characters[i] for i in freq_idx)
            if freq_idx is not None
            else monitor.wave_characters
        ),
        components=tuple(monitor.components if components is None else components),
        scaling_mode=monitor.scaling_mode,
        plot=False,
        switch=monitor.switch,  # we downsample by 8x, note that the field amplitude also reduce proprtionally
        ## we need the Yee's grid's gradients to create correct adjoint source.
        ## for objective computation (flux/eigenmode/overlap), we will do interpolation there
        exact_interpolation=False,
    )

    return monitor


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
    objects=None,
    config=None,
):
    """Compute scalar dL/d eps_r from forward and adjoint E-field phasors."""
    # forward_phasor (nfreq, component, Nx, Ny, Nz), adjoint_phasor (nfreq, component, Nx, Ny, Nz)
    forward_e = forward_phasor[:, :3]  # (nfreq, 3, Nx, Ny, Nz)
    adjoint_e = adjoint_phasor[:, :3]  # (nfreq, 3, Nx, Ny, Nz)

    if need_interpolated_phasors:
        forward_e = yee_to_colocate_interpolate(forward_e, objects, config, is_E=True)
        adjoint_e = yee_to_colocate_interpolate(adjoint_e, objects, config, is_E=True)
        # after interpolation, forward_e and adjoint_e are (nfreq,
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
    grad = -2 * (dA_deps[..., None, None, None] * overlap).real.sum(
        0
    )  # accumulate over frequencies, [Nx, Ny, Nz]

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
        ## if symmetry, then this is reduced Yee grid E fields.
        E_fields = arrays.detector_states[monitor.name]["phasor"][0]
        # if config.has_symmetry:
        #     E_fields = fdtdx.unfold_fields(field=E_fields, symmetry=config.symmetry, field_type="E")
        #     arrays = fdtdx.unfold_detector_states(arrays, objects, config)

        # (latent=1, freq, component, Nx, Ny, Nz) on yee's grid. not co-located.
        # raw_outputs = _jax_to_torch(arrays.detector_states[monitor.name]["phasor"])[0]
        raw_outputs = _jax_to_torch(E_fields)
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
        ctx.arrays = arrays
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
        cotangents = _torch_to_jax(raw_cotangents)

        ### first we need to determine how to run adjoint simulation
        # per single frequency for all ports
        #   or per spatial group (same port) for corresonding frequencies
        # whichever is smaller number of runs.
        ### [Warning] RULE OF THUMB: Cannot run broadband simulation if the spatial locations of the gradients for different frequencies do not overlap.
        # Otherwise, the unwanted freq components from the broadband sources will all be recorded and mixed up by the phasor detector
        groups = build_adjoint_groups_from_monitor_info(
            raw_cotangents,
            ctx.adjoint_group_info,
            threshold=1e-12,
        )
        group_source = "monitor metadata"
        if len(groups) == 0:
            groups = group_adjoint_spatial_support(raw_cotangents, threshold=1e-12)
            group_source = "cotangent support fallback"
        n_freq = cotangents.shape[0]
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

        if n_groups == 0 or n_freq <= n_groups or (not _MAPS_ENABLE_BROADBAND_ADJOINT):
            # one adjoint run per frequency, with sources at all relevant locations for that frequency.
            print(
                f"Running adjoint simulation per frequency: n_freq={n_freq} <= n_groups={n_groups}"
            )
            # Run adjoint simulation per frequency
            grad_eps = None
            adjoint_sources_by_freq: list[Tensor] = []
            adjoint_phasors_by_freq: list[Tensor] = []
            grad_eps_by_freq: list[Tensor] = []
            for freq_idx in range(cotangents.shape[0]):
                adj_monitor = _clone_phasor_detector_for_frequency(
                    ctx.fwd_monitor,
                    freq_idx=freq_idx,
                    components=("Ex", "Ey", "Ez"),
                )
                adj_source = _build_distributed_adjoint_source(
                    adj_monitor=adj_monitor,
                    source_template=ctx.objects.sources[0],
                    cotangents=cotangents[freq_idx : freq_idx + 1],
                    center_on_monitor=True,
                )

                adj_arrays, adj_objects, adj_config, adj_key, adj_monitor = (
                    ctx.prepare_adjoint_simulation(
                        source=adj_source,
                        source_grid_margins=_object_grid_slice_bounds(ctx.fwd_monitor),
                        monitor=adj_monitor,
                    )
                )

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

                if adj_source.electric_profiles is not None:
                    adjoint_source_profile = _jax_to_torch(adj_source.electric_profiles)
                else:
                    adjoint_source_profile = torch.zeros(
                        (1, 3, *adj_monitor.grid_shape),
                        dtype=adjoint_phasor.dtype,
                        device=adjoint_phasor.device,
                    )

                single_grad = _permittivity_gradient_from_phasors(
                    forward_phasor=ctx.forward_phasor[freq_idx : freq_idx + 1],
                    adjoint_phasor=adjoint_phasor,
                    wave_characters=(ctx.fwd_monitor.wave_characters[freq_idx],),
                    objects=ctx.objects,
                    config=ctx.config,
                    need_interpolated_phasors=True,
                )
                adjoint_sources_by_freq.append(adjoint_source_profile)
                adjoint_phasors_by_freq.append(adjoint_phasor)
                grad_eps_by_freq.append(single_grad)
                grad_eps = single_grad if grad_eps is None else grad_eps + single_grad
            _write_adjoint_debug_plot(
                monitor=ctx.fwd_monitor,
                phasor_gradients=raw_cotangents,
                forward_phasors=ctx.forward_phasor,
                adjoint_sources=torch.cat(adjoint_sources_by_freq, dim=0),
                adjoint_phasors=torch.cat(adjoint_phasors_by_freq, dim=0),
                grad_eps_by_freq=torch.stack(grad_eps_by_freq, dim=0),
            )
        else:
            # Option B: one multi-frequency adjoint run only for sources that share the same spatial/source-location group.
            """https://github.com/flexcompute/tidy3d/pull/1830
            Here's how it handles multi-frequency adjoint problems:
            if the number of frequencies in the objective function is 1, it just does what we did before (per-freq)
            if the number of "ports" in the objective function (to put another way, if we can formulate a broadband adjoint source with one source). We do broadband adjoint with a single source and then post-normalize the adjoint fields using the vjp values.
            """
            print(
                f"[Experimental] Running adjoint simulation per spatial group: n_freq={n_freq} > n_groups={n_groups}"
            )
            grad_eps = None
            ## we run adjoint simulation per spatial group, each group has multiple frequencies but same spatial support. this can save some runs when there are many frequencies but only few spatial supports.
            for group in groups.values():
                # freq_indices = group["freq_indices"]
                # spatial_indices = group["spatial_indices"]
                cotangents_group = extract_group_grad_compact_torch(
                    raw_cotangents, group=group
                )
                freq_indices = tuple(int(i) for i in group["freq_indices"].tolist())
                adj_monitor = _clone_phasor_detector_for_frequency(
                    ctx.fwd_monitor,
                    freq_idx=freq_indices,
                    components=("Ex", "Ey", "Ez"),
                )
                adj_source, adjoint_vjp_scale = (
                    _build_separable_broadband_adjoint_source(
                        adj_monitor=adj_monitor,
                        source_template=ctx.objects.sources[0],
                        cotangents=cotangents_group,
                    )
                )

                adj_arrays, adj_objects, adj_config, adj_key, adj_monitor = (
                    ctx.prepare_adjoint_simulation(
                        source=adj_source,
                        source_grid_margins=_object_grid_slice_bounds(ctx.fwd_monitor),
                        monitor=adj_monitor,
                    )
                )

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

                if adj_source.electric_profiles is not None:
                    adjoint_source_profile = _jax_to_torch(adj_source.electric_profiles)
                else:
                    adjoint_source_profile = torch.zeros(
                        (cotangents.shape[0], 3, *adj_monitor.grid_shape),
                        dtype=adjoint_phasor.dtype,
                        device=adjoint_phasor.device,
                    )
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
                )
                grad_eps = cur_grad_eps if grad_eps is None else grad_eps + cur_grad_eps

        grad_eps = unfold_array(
            arr=grad_eps, symmetry=adj_config.symmetry, spatial_axes=(0, 1, 2)
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
    ) -> tuple[Tensor, ...]:
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
