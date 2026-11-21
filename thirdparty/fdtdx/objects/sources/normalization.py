"""Complex source-spectrum normalization under FDTD-X detector conventions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np


def discrete_source_spectrum(
    *,
    source: Any,
    wave_characters: Sequence[Any],
    config: Any,
    detector: Any | None = None,
    include_static_amplitude: bool = False,
) -> np.ndarray:
    """Evaluate the complex source DTFT exactly as a phasor detector samples it.

    Args:
        source: Source exposing ``temporal_profile`` and ``wave_character``.
        wave_characters: Frequencies at which to evaluate the normalization.
        config: Placed FDTD-X simulation configuration.
        detector: Optional placed phasor detector. Its switch, DFT stride, and
            scaling mode determine the sampled time indices and scale.
        include_static_amplitude: Whether to include source-level amplitude
            multipliers. MAPS normally leaves these unnormalized so deliberate
            user scaling remains visible in monitor outputs.

    Returns:
        Complex normalization coefficients shaped ``(num_frequencies,)``.

    Notes:
        The transform uses the detector convention ``exp(+i omega t)`` and the
        actual truncated, discretely sampled real source waveform. For a
        Chebyshev mode source, the Yee-delay phase is already incorporated into
        its reconstructed spatial coefficients, so its scalar normalization is
        the parent Gaussian-pulse spectrum returned here.
    """
    count = int(config.time_steps_total)
    dt = float(config.time_step_duration)
    if detector is None:
        indices = np.arange(count, dtype=np.int64)
        detector_scale = 2.0 / count
    else:
        on = np.asarray(
            detector.switch.calculate_on_list(
                num_total_time_steps=count,
                time_step_duration=dt,
            ),
            dtype=bool,
        )
        indices = np.flatnonzero(on)
        stride = max(1, int(getattr(detector, "_dft_stride", 1)))
        if stride > 1:
            indices = indices[::stride]
        scaling_mode = getattr(detector, "scaling_mode", "continuous")
        if scaling_mode == "continuous":
            detector_scale = 2.0 / len(indices)
        elif scaling_mode == "pulse":
            detector_scale = float(stride)
        else:
            raise ValueError(f"Unsupported phasor scaling mode {scaling_mode!r}.")
    if indices.size == 0:
        return np.zeros(len(wave_characters), dtype=np.complex128)

    times = indices.astype(np.float64) * dt
    profile = getattr(source, "temporal_profile", source)
    carrier_wave = getattr(source, "wave_character", None)
    if carrier_wave is None or not hasattr(carrier_wave, "get_period"):
        carrier_wave = profile.center_wave
    signal = np.asarray(
        profile.get_amplitude(
            time=jnp.asarray(times),
            period=carrier_wave.get_period(),
            phase_shift=carrier_wave.phase_shift,
        ),
        dtype=np.float64,
    )
    frequencies = np.asarray(
        [wave.get_frequency() for wave in wave_characters], dtype=np.float64
    )
    spectrum = detector_scale * (
        np.exp(2j * np.pi * frequencies[:, None] * times[None, :]) @ signal
    )
    if include_static_amplitude:
        spectrum *= float(getattr(source, "static_amplitude_factor", 1.0))
        spectrum *= float(getattr(source, "amplitude", 1.0))
    return spectrum
