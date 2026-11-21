"""Cost-based dispatch for exact near-field projection.

The automatic selector only uses exact-shape benchmark records supplied by
the caller. It performs no profiling, calibration, or device work beyond
reading the current CUDA device identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Iterable, Mapping

import torch

from .projector import Surface, project_nearfield

_BENCHMARK = "core.nearfield.project_nearfield"
_MIN_REPEATS = 3
_MAX_REFERENCE_ERROR = 0.01
_TRITON_SOURCE_BLOCK_SIZE = 16
_TRITON_OBSERVATION_BLOCK_SIZE = 32
_TRITON_BACKWARD_SOURCE_CHUNK_SIZE = 256
_TRITON_BACKWARD_OBSERVATION_CHUNK_SIZE = 128


@dataclass(frozen=True)
class BackendDecision:
    """A backend choice and the evidence used to make it.

    Attributes:
        backend: Selected public backend name.
        reason: Human-readable explanation suitable for logging by the caller.
        estimated_seconds: Estimated total time for ``expected_calls`` or
            ``None`` when no matching measurement supports the decision.
    """

    backend: str
    reason: str
    estimated_seconds: float | None


def _requires_backward(
    surface: Surface,
    observation_points_m: torch.Tensor,
    eps_r: float | torch.Tensor,
    mu_r: float | torch.Tensor,
) -> bool:
    """Detect differentiable field, geometry, frequency, or material inputs.

    Args:
        surface: Sampled fields and source geometry.
        observation_points_m: Target positions in meters.
        eps_r: Relative permittivity, possibly a differentiable scalar tensor.
        mu_r: Relative permeability, possibly a differentiable scalar tensor.

    Returns:
        Whether any projection input tensor requires gradients.
    """
    values = (
        surface.positions_m,
        surface.normals,
        surface.weights_m2,
        surface.electric_fields,
        surface.magnetic_fields,
        surface.frequencies_hz,
        observation_points_m,
        eps_r,
        mu_r,
    )
    return any(
        isinstance(value, torch.Tensor) and value.requires_grad for value in values
    )


def _finite_positive(value: Any) -> bool:
    """Check that a measured duration is a finite positive non-Boolean number."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _record_cost(
    record: Mapping[str, Any],
    *,
    surface: Surface,
    observation_points_m: torch.Tensor,
    source_chunk_size: int,
    observation_chunk_size: int,
    triton_observation_block_size: int,
    triton_source_block_size: int,
    expected_calls: int,
    requires_backward: bool,
    hardware: Mapping[str, Any],
) -> tuple[str | None, float | None]:
    """Validate one record and return its backend and projected cost."""
    try:
        if record.get("benchmark") != _BENCHMARK:
            return "wrong benchmark", None
        backend = record.get("backend")
        if backend not in ("torch", "triton"):
            return "unsupported benchmark backend", None
        rec_hardware = record["hardware"]
        if any(rec_hardware.get(key) != value for key, value in hardware.items()):
            return "hardware or software version mismatch", None
        measurements = record["measurements"]
        config = record["configuration"]
        if (
            measurements.get("surface_points") != surface.positions_m.shape[0]
            or measurements.get("observation_points") != observation_points_m.shape[0]
            or measurements.get("frequencies") != surface.frequencies_hz.shape[0]
            or measurements.get("field_dtype") != str(surface.electric_fields.dtype)
        ):
            return "shape or dtype mismatch", None
        if config.get("real_dtype") != str(surface.positions_m.dtype).removeprefix(
            "torch."
        ):
            return "real dtype mismatch", None
        repeats = config.get("measured_warm_repeats")
        if (
            not isinstance(repeats, int)
            or isinstance(repeats, bool)
            or repeats < _MIN_REPEATS
        ):
            return "fewer than three measured warm repeats", None
        reference = record.get("precision_reference")
        if not isinstance(reference, Mapping):
            return "missing precision reference", None
        errors = (
            reference.get("relative_l2_E_vs_complex128"),
            reference.get("relative_l2_H_vs_complex128"),
        )
        if any(
            not isinstance(error, (int, float))
            or not math.isfinite(error)
            or error < 0
            or error > _MAX_REFERENCE_ERROR
            for error in errors
        ):
            return "precision reference exceeds one percent or is invalid", None
        if backend == "torch":
            if (
                config.get("source_chunk_applies") is not True
                or config.get("checkpoint_chunks_applies") is not True
                or config.get("checkpoint_chunks") is not True
                or config.get("source_chunk_size") != source_chunk_size
                or config.get("observation_chunk_size") != observation_chunk_size
            ):
                return "Torch chunk settings mismatch", None
            warm_key = (
                "warm_forward_backward_median_seconds"
                if requires_backward
                else "warm_forward_median_seconds"
            )
            first_key = (
                "first_forward_backward_seconds"
                if requires_backward
                else "first_forward_seconds"
            )
        else:
            # The benchmark currently uses the Triton backend's fixed source
            # block of 16; source_chunk_size is not used by that path.
            triton_tensors = (
                surface.positions_m,
                surface.normals,
                surface.weights_m2,
                surface.electric_fields,
                surface.magnetic_fields,
                surface.frequencies_hz,
                observation_points_m,
            )
            if (
                surface.positions_m.device.type != "cuda"
                or any(
                    tensor.dtype != torch.float32
                    for tensor in (
                        surface.positions_m,
                        surface.normals,
                        surface.weights_m2,
                        surface.frequencies_hz,
                        observation_points_m,
                    )
                )
                or surface.electric_fields.dtype != torch.complex64
                or surface.magnetic_fields.dtype != torch.complex64
                or any(
                    tensor.device != surface.positions_m.device
                    for tensor in triton_tensors
                )
            ):
                return (
                    "Triton requires float32 geometry and complex64 fields on CUDA",
                    None,
                )
            if (
                config.get("triton_observation_block_size")
                != triton_observation_block_size
                or config.get("source_chunk_applies") is not False
                or config.get("checkpoint_chunks_applies") is not False
                or config.get("triton_source_block_size") != triton_source_block_size
            ):
                return "Triton block settings mismatch", None
            if requires_backward and (
                config.get("triton_backward_source_chunk_size")
                != _TRITON_BACKWARD_SOURCE_CHUNK_SIZE
                or config.get("triton_backward_observation_chunk_size")
                != _TRITON_BACKWARD_OBSERVATION_CHUNK_SIZE
            ):
                return "Triton backward chunk settings mismatch", None
            warm_key = (
                "warm_forward_backward_median_seconds"
                if requires_backward
                else "warm_forward_median_seconds"
            )
            first_key = (
                "first_forward_backward_seconds"
                if requires_backward
                else "first_forward_seconds"
            )
        first = measurements.get(first_key)
        warm = measurements.get(warm_key)
        if not _finite_positive(first) or not _finite_positive(warm):
            return "missing or invalid measured timing", None
        if requires_backward:
            gradient_forward = measurements.get("first_gradient_forward_seconds")
            first_backward = measurements.get("first_backward_seconds")
            first_forward = measurements.get("first_forward_seconds")
            warm_forward = measurements.get("warm_forward_median_seconds")
            if (
                not _finite_positive(gradient_forward)
                or not _finite_positive(first_backward)
                or not _finite_positive(first_forward)
                or not _finite_positive(warm_forward)
            ):
                return "missing or invalid first-forward timing", None
            # The benchmark compiles Triton in its initial no-grad forward
            # before measuring the first gradient call. Add that separately
            # measured cold-call premium back to a one-shot gradient estimate.
            first += max(0.0, first_forward - warm_forward)
        # First-call cost is paid once, then the measured warm median for each
        # subsequent call. No size or shape extrapolation is performed.
        return backend, first + (expected_calls - 1) * warm
    except (KeyError, TypeError, AttributeError):
        return "malformed benchmark record", None


def select_backend(
    surface: Surface,
    observation_points_m: torch.Tensor,
    *,
    calibration_records: Iterable[Mapping[str, Any]] = (),
    expected_calls: int = 1,
    requires_backward: bool | None = None,
    source_chunk_size: int = 256,
    observation_chunk_size: int = 128,
    triton_observation_block_size: int | None = None,
    triton_source_block_size: int = _TRITON_SOURCE_BLOCK_SIZE,
    eps_r: float | torch.Tensor = 1.0,
    mu_r: float | torch.Tensor = 1.0,
) -> BackendDecision:
    """Choose a direct near-field backend from validated exact-shape records.

    Args:
        surface: Sampled source surface.
        observation_points_m: Observation locations, shape ``(M, 3)``.
        calibration_records: JSON-like benchmark records produced by
            ``benchmarks/nearfield/projector.py``.
        expected_calls: Number of projections expected for this input shape.
        requires_backward: Whether any projected input will require a
            backward pass. If omitted, inferred from input ``requires_grad``.
        source_chunk_size: Torch source chunk. Triton backward uses source
            chunks of 256.
        observation_chunk_size: Torch observation chunk. Triton backward uses
            observation chunks of 128.
        triton_observation_block_size: Triton forward observation block. If
            omitted, uses 32.
        triton_source_block_size: Triton forward source tile size.
        eps_r: Relative permittivity, passed through to the Torch projector.
        mu_r: Relative permeability, passed through to the Torch projector.

    Returns:
        A decision containing the selected backend, reason, and estimated cost.

    Notes:
        Candidates must match exact P/M/F, dtypes, current GPU name, PyTorch
        and CUDA versions, chunk configuration, at least three measured warm
        repeats, and a complex128 reference error no greater than one percent.
        Stale or malformed records are ignored. There is no extrapolation.
    """
    if (
        not isinstance(expected_calls, int)
        or isinstance(expected_calls, bool)
        or expected_calls < 1
    ):
        raise ValueError("expected_calls must be a positive integer")
    if source_chunk_size <= 0 or observation_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if triton_observation_block_size is None:
        triton_observation_block_size = _TRITON_OBSERVATION_BLOCK_SIZE
    if triton_observation_block_size <= 0 or triton_source_block_size <= 0:
        raise ValueError("Triton block sizes must be positive")
    if observation_points_m.ndim != 2 or observation_points_m.shape[-1] != 3:
        raise ValueError("observation_points_m must have shape (M, 3)")
    backward = (
        _requires_backward(surface, observation_points_m, eps_r, mu_r)
        if requires_backward is None
        else requires_backward
    )
    if not isinstance(backward, bool):
        raise TypeError("requires_backward must be bool or None")
    if surface.positions_m.device.type != "cuda":
        return BackendDecision(
            "direct_torch", "AUTO uses Torch on non-CUDA devices", None
        )
    device = surface.positions_m.device
    hardware = {
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    candidates: list[tuple[str, float]] = []
    rejected: list[str] = []
    try:
        records = tuple(calibration_records)
    except TypeError:
        records = ()
        rejected.append("calibration records are not iterable")
    for record in records:
        if not isinstance(record, Mapping):
            rejected.append("record is not a JSON object")
            continue
        backend, cost = _record_cost(
            record,
            surface=surface,
            observation_points_m=observation_points_m,
            source_chunk_size=source_chunk_size,
            observation_chunk_size=observation_chunk_size,
            triton_observation_block_size=triton_observation_block_size,
            triton_source_block_size=triton_source_block_size,
            expected_calls=expected_calls,
            requires_backward=backward,
            hardware=hardware,
        )
        if backend is None or cost is None:
            rejected.append(backend or "invalid record")
        elif backend == "triton" and find_spec("triton") is None:
            rejected.append("Triton package is unavailable")
        else:
            candidates.append((backend, cost))
    if not candidates:
        detail = (
            "; ".join(dict.fromkeys(rejected)) if rejected else "no records supplied"
        )
        return BackendDecision(
            "direct_torch",
            f"no applicable validated calibration record ({detail})",
            None,
        )
    chosen, cost = min(candidates, key=lambda item: item[1])
    reason = (
        f"lowest measured first-plus-warm cost for {expected_calls} call(s)"
        f" using {'forward and backward' if backward else 'forward'} timings"
    )
    if rejected:
        reason += f"; ignored {len(rejected)} record(s): " + "; ".join(
            dict.fromkeys(rejected)
        )
    return BackendDecision(f"direct_{chosen}", reason, cost)


def project(
    surface: Surface,
    observation_points_m: torch.Tensor,
    backend: str = "direct_torch",
    *,
    calibration_records: Iterable[Mapping[str, Any]] = (),
    expected_calls: int = 1,
    requires_backward: bool | None = None,
    source_chunk_size: int = 256,
    observation_chunk_size: int = 128,
    triton_observation_block_size: int | None = None,
    triton_source_block_size: int = _TRITON_SOURCE_BLOCK_SIZE,
    eps_r: float | torch.Tensor = 1.0,
    mu_r: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project equivalent currents using Torch, Triton, or calibrated AUTO.

    Args:
        surface: Sampled source surface.
        observation_points_m: Observation locations, shape ``(M, 3)``.
        backend: ``"direct_torch"``, ``"direct_triton"``, or ``"auto"``.
        calibration_records: Caller-supplied records used only by ``"auto"``.
        expected_calls: Expected reuse count supplied to AUTO's cost estimate.
        requires_backward: Backward expectation; inferred from inputs when None.
        source_chunk_size: Torch source chunk size. Triton backward uses
            chunks of 256.
        observation_chunk_size: Torch observation chunk size. Triton backward
            uses chunks of 128.
        triton_observation_block_size: Triton forward observation block. If
            omitted, uses 32.
        triton_source_block_size: Triton forward source tile size.
        eps_r: Positive homogeneous-medium relative permittivity.
        mu_r: Positive homogeneous-medium relative permeability.

    Returns:
        Complex electric and magnetic fields shaped ``(F, M, 3)``.

    Raises:
        ValueError: If ``backend`` is unknown or backend inputs are invalid.
        RuntimeError: If explicitly requested Triton is unavailable or
            unsupported for the input device and dtype.

    Notes:
        AUTO uses exact-shape validated measurements only. It does not profile,
        compile candidate kernels, or estimate costs for unseen shapes.
        Planar far-field and FFT propagation are separate mathematical APIs.
    """
    if backend not in ("direct_torch", "direct_triton", "auto"):
        raise ValueError("backend must be 'direct_torch', 'direct_triton', or 'auto'")
    if backend == "auto":
        decision = select_backend(
            surface,
            observation_points_m,
            calibration_records=calibration_records,
            expected_calls=expected_calls,
            requires_backward=requires_backward,
            source_chunk_size=source_chunk_size,
            observation_chunk_size=observation_chunk_size,
            triton_observation_block_size=triton_observation_block_size,
            triton_source_block_size=triton_source_block_size,
            eps_r=eps_r,
            mu_r=mu_r,
        )
        backend = decision.backend
    if backend == "direct_torch":
        return project_nearfield(
            surface,
            observation_points_m,
            eps_r=eps_r,
            mu_r=mu_r,
            source_chunk_size=source_chunk_size,
            observation_chunk_size=observation_chunk_size,
        )
    # Keep Triton optional for callers that use only the direct Torch path.
    from .triton_backend import project_nearfield_triton

    return project_nearfield_triton(
        surface,
        observation_points_m,
        eps_r=eps_r,
        mu_r=mu_r,
        observation_block_size=(
            observation_chunk_size
            if triton_observation_block_size is None
            else triton_observation_block_size
        ),
        source_block_size=triton_source_block_size,
    )
