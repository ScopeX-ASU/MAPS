"""Plotting and measured profiling for the near-field example notebooks.

Conversions to NumPy are confined to reporting. Projection and loss evaluation
remain in PyTorch. Intensity plots show squared field magnitude, not power flux.
"""

from __future__ import annotations

from html import escape
from statistics import median
from textwrap import fill
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display

from core.nearfield import Surface


def _array(value):
    """Return a detached CPU array for reporting only."""
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _grid(field, x, y):
    """Validate one frequency and reshape samples in y-major order."""
    values = _array(field)
    if values.ndim == 3 and values.shape[0] == 1 and values.shape[1] == len(x) * len(y):
        values = values[0]
    if values.shape == (len(x) * len(y), 3):
        values = values.reshape(len(y), len(x), 3)
    if values.shape != (len(y), len(x), 3):
        raise ValueError("Select one frequency with field shape (Ny*Nx,3) or (Ny,Nx,3)")
    return values


def field_components(
    field,
    x_um,
    y_um,
    *,
    title,
    quantity="E",
    units="V/m",
    view="amplitude_phase",
    coordinate_labels=("x (µm)", "y (µm)"),
):
    """Display all Cartesian components with common amplitude scales.

    Args:
        field: Single-frequency complex field, shape (Ny*Nx,3) or (Ny,Nx,3).
        x_um: Horizontal sample coordinates in micrometers.
        y_um: Vertical sample coordinates in micrometers.
        title: Figure title.
        quantity: Field symbol used in component labels.
        units: Physical units of field amplitude.
        view: 'amplitude_phase' or 'real_imag'.
        coordinate_labels: Horizontal and vertical coordinate labels.

    Returns:
        Displayed Matplotlib figure. Phase is masked below 1% of peak amplitude.
    """
    values = _grid(field, x_um, y_um)
    if view not in ("amplitude_phase", "real_imag"):
        raise ValueError("view must be amplitude_phase or real_imag")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    peak = max(float(np.abs(values).max()), np.finfo(float).tiny)
    for component, axis_name in enumerate("xyz"):
        value = values[..., component]
        if view == "amplitude_phase":
            panels = [
                (np.abs(value), 0, peak, "viridis", units),
                (
                    np.ma.masked_where(np.abs(value) < 0.01 * peak, np.angle(value)),
                    -np.pi,
                    np.pi,
                    "twilight",
                    "rad",
                ),
            ]
        else:
            panels = [
                (value.real, -peak, peak, "RdBu_r", units),
                (value.imag, -peak, peak, "RdBu_r", units),
            ]
        for row, (data, lower, upper, cmap, label) in enumerate(panels):
            ax = axes[row, component]
            mesh = ax.pcolormesh(
                _array(x_um),
                _array(y_um),
                data,
                shading="auto",
                cmap=cmap,
                vmin=lower,
                vmax=upper,
            )
            mode = (
                ("amplitude", "phase")[row]
                if view == "amplitude_phase"
                else ("real", "imag")
            )
            ax.set(
                title=f"{quantity}{axis_name}: {mode}",
                xlabel=coordinate_labels[0],
                ylabel=coordinate_labels[1],
                aspect="equal",
            )
            fig.colorbar(mesh, ax=ax, label=label)
    fig.suptitle(title)
    plt.show()
    return fig


def field_comparison(
    reference,
    candidate,
    x_um,
    y_um,
    *,
    names=("reference", "method"),
    quantity="E",
    units="V/m",
    component=0,
):
    """Display shared-scale fields and errors without renormalizing methods.

    Args:
        reference: Complex reference vector field at one frequency.
        candidate: Complex candidate field on identical coordinates.
        x_um: Horizontal coordinates in micrometers.
        y_um: Vertical coordinates in micrometers.
        names: Reference and candidate labels.
        quantity: Field symbol.
        units: Field amplitude units.
        component: Cartesian component index used for phase comparison.

    Returns:
        Displayed figure. Complex error uses global peak reference vector magnitude;
        phase error excludes samples below 1% of the reference component peak.
    """
    ref, val = _grid(reference, x_um, y_um), _grid(candidate, x_um, y_um)
    ref_norm, val_norm = np.linalg.norm(ref, axis=-1), np.linalg.norm(val, axis=-1)
    peak = max(float(ref_norm.max()), np.finfo(float).tiny)
    intensity_max = max(
        float((ref_norm**2).max()), float((val_norm**2).max()), np.finfo(float).tiny
    )
    phase_peak = max(float(np.abs(ref[..., component]).max()), np.finfo(float).tiny)
    mask = (np.abs(ref[..., component]) < 0.01 * phase_peak) | (
        np.abs(val[..., component]) < 0.01 * phase_peak
    )
    phase_error = np.ma.masked_where(
        mask, np.angle(val[..., component] * ref[..., component].conj())
    )
    difference = (val_norm**2 - ref_norm**2) / peak**2
    diff_max = max(float(np.abs(difference).max()), 1e-15)
    panels = [
        (
            ref_norm**2,
            f"{names[0]}: |{quantity}|²",
            "viridis",
            0,
            intensity_max,
            f"({units})²",
        ),
        (
            val_norm**2,
            f"{names[1]}: |{quantity}|²",
            "viridis",
            0,
            intensity_max,
            f"({units})²",
        ),
        (
            difference,
            "Intensity difference / reference peak",
            "RdBu_r",
            -diff_max,
            diff_max,
            "dimensionless",
        ),
        (
            np.linalg.norm(val - ref, axis=-1) / peak,
            "Complex vector error / reference peak",
            "magma",
            0,
            None,
            "dimensionless",
        ),
        (
            np.abs(ref[..., component]),
            f"Reference |{quantity}{'xyz'[component]}|",
            "viridis",
            0,
            phase_peak,
            units,
        ),
        (
            phase_error,
            f"{quantity}{'xyz'[component]} phase error (weak fields masked)",
            "twilight",
            -np.pi,
            np.pi,
            "rad",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (data, title, cmap, lower, upper, label) in zip(axes.flat, panels):
        mesh = ax.pcolormesh(
            _array(x_um),
            _array(y_um),
            data,
            shading="auto",
            cmap=cmap,
            vmin=lower,
            vmax=upper,
        )
        ax.set(title=title, xlabel="x (µm)", ylabel="y (µm)", aspect="equal")
        fig.colorbar(mesh, ax=ax, label=label)
    fig.suptitle(
        f"{names[1]} versus {names[0]} — identical coordinates and absolute scales"
    )
    plt.show()
    return fig


def error_summary(reference, candidate, *, label="comparison", strong_fraction=1e-2):
    """Summarize complex errors, distinguishing strong and weak field samples.

    Args:
        reference: Complex reference vector field with final dimension three.
        candidate: Candidate field with identical shape.
        label: Row label.
        strong_fraction: Strong-field threshold relative to peak vector amplitude.

    Returns:
        Dictionary of dimensionless L2 errors, phase RMS in radians, and counts.
        Weak-field RMS error uses global reference peak, avoiding division by nulls.
    """
    ref, val = _array(reference).reshape(-1, 3), _array(candidate).reshape(-1, 3)
    if ref.shape != val.shape or not 0 < strong_fraction < 1:
        raise ValueError("Shapes must match and strong_fraction must lie in (0,1)")
    norm = np.linalg.norm(ref, axis=-1)
    peak = max(float(norm.max()), np.finfo(float).tiny)
    strong = norm >= strong_fraction * peak
    phase_valid = (np.abs(ref) >= strong_fraction * peak) & (
        np.abs(val) >= strong_fraction * peak
    )
    denominator = max(float(np.linalg.norm(ref)), np.finfo(float).tiny)
    phase = np.angle(val * ref.conj())
    result = dict(
        label=label,
        relative_complex_l2=float(np.linalg.norm(val - ref) / denominator),
        relative_amplitude_l2=float(
            np.linalg.norm(np.abs(val) - np.abs(ref)) / denominator
        ),
        phase_rms_rad=(
            float(np.sqrt(np.mean(phase[phase_valid] ** 2)))
            if phase_valid.any()
            else None
        ),
        strong_relative_l2=(
            float(
                np.linalg.norm((val - ref)[strong])
                / max(float(np.linalg.norm(ref[strong])), np.finfo(float).tiny)
            )
            if strong.any()
            else None
        ),
        weak_absolute_over_peak=(
            float(
                np.sqrt(np.mean(np.linalg.norm((val - ref)[~strong], axis=-1) ** 2))
                / peak
            )
            if (~strong).any()
            else None
        ),
        strong_count=int(strong.sum()),
        weak_count=int((~strong).sum()),
    )
    return result


def _table(records):
    """Render report dictionaries without requiring pandas."""
    keys = list(dict.fromkeys(key for record in records for key in record))

    def cell(value):
        if value is None:
            return "N/A"
        return escape(f"{value:.5g}" if isinstance(value, float) else str(value))

    display(
        HTML(
            "<table><thead><tr>"
            + "".join(f"<th>{escape(key)}</th>" for key in keys)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>"
                + "".join(f"<td>{cell(record.get(key))}</td>" for key in keys)
                + "</tr>"
                for record in records
            )
            + "</tbody></table>"
        )
    )


def show_errors(records):
    """Display error dictionaries as a table; missing metrics appear as N/A."""
    _table(records)


def surface_copy(surface, *, device=None, dtype=None, requires_grad=False):
    """Create an independent fixed-geometry profiling fixture.

    Args:
        surface: Source Surface.
        device: Target device, defaulting to source device.
        dtype: complex64 or complex128, defaulting to source dtype.
        requires_grad: Whether cloned E/H samples are independent gradient leaves.

    Returns:
        Surface with detached geometry and independently cloned E/H fields.
    """
    dtype = dtype or surface.electric_fields.dtype
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("dtype must be complex64 or complex128")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    device = device or surface.positions_m.device
    geometry = [
        value.detach().to(device=device, dtype=real_dtype).clone()
        for value in (surface.positions_m, surface.normals, surface.weights_m2)
    ]
    fields = [
        value.detach()
        .to(device=device, dtype=dtype)
        .clone()
        .requires_grad_(requires_grad)
        for value in (surface.electric_fields, surface.magnetic_fields)
    ]
    return Surface(
        *geometry,
        *fields,
        surface.frequencies_hz.detach().to(device=device, dtype=real_dtype).clone(),
    )


def benchmark_methods(methods, *, device, backward_inputs=(), warmups=1, repeats=3):
    """Measure synchronized forward/backward latency and allocated CUDA memory.

    Args:
        methods: Mapping from labels to zero-argument field-producing callables.
        device: Device on which the callables execute.
        backward_inputs: Gradient leaves used by each callable; omit for forward only.
        warmups: Number of untimed warm executions.
        repeats: Number of timed repetitions summarized by median.

    Returns:
        Report dictionaries. First-observed times include any lazy setup but do not
        isolate compilation. CPU memory is unmeasured. CUDA allocation is reported
        both absolutely and above the live allocation before the method. Backward
        uses sum of squared output magnitudes and never accumulates leaf .grad.
    """
    device = torch.device(device)
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def outputs(value):
        return (value,) if torch.is_tensor(value) else tuple(value)

    def call_forward(fn):
        sync()
        start = perf_counter()
        with torch.no_grad():
            result = outputs(fn())
        sync()
        elapsed = (perf_counter() - start) * 1000
        metadata = (str(result[0].dtype), tuple(result[0].shape))
        del result
        return elapsed, metadata

    def call_gradient(fn):
        sync()
        start = perf_counter()
        result = outputs(fn())
        loss = sum(value.abs().square().sum() for value in result)
        sync()
        forward_end = perf_counter()
        gradients = torch.autograd.grad(loss, backward_inputs)
        sync()
        end = perf_counter()
        del gradients, loss, result
        return (forward_end - start) * 1000, (end - forward_end) * 1000

    records = []
    for name, fn in methods.items():
        baseline = (
            torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        first, (dtype, shape) = call_forward(fn)
        first_gradient = call_gradient(fn) if backward_inputs else (None, None)
        for _ in range(warmups):
            call_forward(fn)
            if backward_inputs:
                call_gradient(fn)
        forward_times, backward_times, total_times = [], [], []
        for _ in range(repeats):
            elapsed, _ = call_forward(fn)
            forward_times.append(elapsed)
            if backward_inputs:
                fw, bw = call_gradient(fn)
                backward_times.append(bw)
                total_times.append(fw + bw)
        peak = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        records.append(
            dict(
                method=name,
                device=str(device),
                dtype=dtype,
                output_shape=str(shape),
                first_forward_ms=first,
                warm_forward_ms=median(forward_times),
                first_gradient_forward_ms=first_gradient[0],
                first_backward_ms=first_gradient[1],
                warm_backward_ms=median(backward_times) if backward_times else None,
                warm_forward_backward_ms=median(total_times) if total_times else None,
                peak_allocated_mib=peak / 2**20 if peak is not None else None,
                peak_increment_mib=(
                    (peak - baseline) / 2**20 if peak is not None else None
                ),
                warmups=warmups,
                repeats=repeats,
            )
        )
    return records


def show_benchmarks(records):
    """Display measured records and latency/allocation plots; CPU memory is N/A."""
    devices = {record["device"] for record in records}
    hardware = [
        (
            torch.cuda.get_device_name(torch.device(device))
            if torch.device(device).type == "cuda"
            else "CPU"
        )
        for device in sorted(devices)
    ]
    display(
        HTML(
            f"<p>Hardware: {escape(', '.join(hardware))}; PyTorch {escape(torch.__version__)}.</p>"
        )
    )
    _table(records)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    labels = [fill(record["method"], width=23) for record in records]
    x = np.arange(len(records))
    for ax, keys, ylabel in zip(
        axes,
        [
            ("first_forward_ms", "warm_forward_ms"),
            ("warm_backward_ms", "warm_forward_backward_ms"),
            ("peak_allocated_mib", "peak_increment_mib"),
        ],
        ["Forward latency (ms)", "Gradient latency (ms)", "CUDA allocation (MiB)"],
    ):
        readable = {
            "first_forward_ms": "First observed",
            "warm_forward_ms": "Warm forward",
            "warm_backward_ms": "Backward",
            "warm_forward_backward_ms": "Forward + backward",
            "peak_allocated_mib": "Peak allocated",
            "peak_increment_mib": "Peak increase",
        }
        available = any(
            record.get(key) is not None for record in records for key in keys
        )
        for offset, key in enumerate(keys):
            values = [
                np.nan if record.get(key) is None else record[key] for record in records
            ]
            if available:
                ax.bar(
                    x + (offset - 0.5) * 0.36, values, width=0.36, label=readable[key]
                )
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylabel(ylabel)
        if available:
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "N/A: no measurement",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_yticks([])
    fig.suptitle(
        "Measured calls; first observed includes lazy setup; CPU memory unmeasured"
    )
    plt.show()
