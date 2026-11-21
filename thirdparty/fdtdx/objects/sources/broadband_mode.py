"""Frequency-dependent broadband eigenmode source for forward FDTD runs."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.mode import ModePlaneSource
from fdtdx.objects.sources.profile import CustomTimeSignalProfile


def synthesize_isolated_mode_waveforms(
    *,
    wave_characters: tuple[WaveCharacter, ...],
    temporal_profile,
    carrier_wave: WaveCharacter,
    time_step_duration: float,
    num_time_steps: int,
    temporal_oversampling: int = 8,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Synthesize real waveforms that isolate mode profiles at target frequencies.

    Each output row is a real time signal. Under fdtdx's phasor convention its
    spectrum equals the original source spectrum at the corresponding target
    frequency and is zero at every other target. A zero-DC constraint prevents
    the source from depositing a static field after shutoff.

    Args:
        wave_characters: Distinct target frequencies, one per mode profile.
        temporal_profile: Original broadband temporal profile whose target
            spectrum is preserved.
        carrier_wave: Carrier passed to ``temporal_profile``.
        time_step_duration: FDTD time step in seconds.
        num_time_steps: Total configured simulation steps.
        temporal_oversampling: Sampling factor for the stored custom waveform.
            Values above one reduce interpolation error at fractional Yee time
            offsets without changing the FDTD time step.

    Returns:
        A tuple containing waveforms shaped
        ``(nfreq, (num_time_steps - 1) * temporal_oversampling + 1)``, their
        achieved complex target spectra shaped ``(nfreq, nfreq)``, the real
        synthesis-system condition number, and the ratio between the largest
        synthesized sample and the largest original-profile sample.

    Raises:
        ValueError: If frequencies are duplicated, fewer than two time samples
            are available, or the real synthesis system is rank deficient.

    Notes:
        The synthesis window is deliberately longer than the ordinary MAPS
        Gaussian pulse when necessary to resolve nearby target frequencies.
        This trades time stepping for a well-conditioned one-run source.
    """
    if num_time_steps < 2:
        raise ValueError("Broadband mode synthesis requires at least two time steps.")
    if temporal_oversampling < 1:
        raise ValueError("temporal_oversampling must be at least one.")
    frequencies = np.asarray(
        [wave.get_frequency() for wave in wave_characters], dtype=np.float64
    )
    if frequencies.size < 2:
        raise ValueError("Broadband mode synthesis requires at least two frequencies.")
    if np.unique(frequencies).size != frequencies.size:
        raise ValueError("Broadband mode synthesis frequencies must be distinct.")

    dt = float(time_step_duration)
    time = np.arange(num_time_steps, dtype=np.float64) * dt
    original = np.asarray(
        temporal_profile.get_amplitude(
            time=jnp.asarray(time),
            period=carrier_wave.get_period(),
            phase_shift=carrier_wave.phase_shift,
        ),
        dtype=np.float64,
    )
    demodulation = np.exp(1j * 2.0 * np.pi * frequencies[:, None] * time[None])
    detector_scale = 2.0 / num_time_steps
    target_spectrum = detector_scale * (demodulation @ original)

    sorted_frequencies = np.sort(frequencies)
    min_spacing = float(np.min(np.diff(sorted_frequencies)))
    # sigma ~= 1 / (2 pi df) makes adjacent Gaussian-windowed tones
    # distinguishable. The first half of the configured duration bounds a
    # 12-sigma source window; the second half remains available for propagation
    # and ring-down before the phasor accumulation ends.
    requested_sigma = 1.0 / (2.0 * np.pi * min_spacing)
    max_sigma = (num_time_steps - 1) * dt / 24.0
    sigma_t = min(requested_sigma, max_sigma)
    center_time = 6.0 * sigma_t
    envelope = np.exp(-0.5 * ((time - center_time) / sigma_t) ** 2)

    candidates = []
    for frequency in frequencies:
        phase = 2.0 * np.pi * frequency * time
        candidates.extend((envelope * np.cos(phase), envelope * np.sin(phase)))
    candidates.append(envelope)
    candidates = np.stack(candidates, axis=1)

    candidate_spectrum = detector_scale * (demodulation @ candidates)
    dc_response = np.mean(candidates, axis=0, keepdims=True)
    system = np.concatenate(
        (candidate_spectrum.real, candidate_spectrum.imag, dc_response), axis=0
    )
    condition_number = float(np.linalg.cond(system))
    if (
        not np.isfinite(condition_number)
        or np.linalg.matrix_rank(system) < system.shape[0]
    ):
        raise ValueError(
            "Broadband mode temporal synthesis is rank deficient; increase the "
            "simulation duration or separate the target frequencies."
        )

    dense_time = (
        np.arange((num_time_steps - 1) * temporal_oversampling + 1, dtype=np.float64)
        * dt
        / temporal_oversampling
    )
    dense_envelope = np.exp(-0.5 * ((dense_time - center_time) / sigma_t) ** 2)
    dense_candidates = []
    for frequency in frequencies:
        dense_phase = 2.0 * np.pi * frequency * dense_time
        dense_candidates.extend(
            (
                dense_envelope * np.cos(dense_phase),
                dense_envelope * np.sin(dense_phase),
            )
        )
    dense_candidates.append(dense_envelope)
    dense_candidates = np.stack(dense_candidates, axis=1)

    waveforms = []
    achieved_columns = []
    for target_index in range(frequencies.size):
        desired = np.zeros(frequencies.size, dtype=np.complex128)
        desired[target_index] = target_spectrum[target_index]
        rhs = np.concatenate((desired.real, desired.imag, np.zeros(1)))
        coefficients = np.linalg.solve(system, rhs)
        waveforms.append(dense_candidates @ coefficients)
        achieved_columns.append(candidate_spectrum @ coefficients)
    waveforms = np.stack(waveforms, axis=0)
    achieved = np.stack(achieved_columns, axis=1)
    original_peak = max(float(np.max(np.abs(original))), np.finfo(float).tiny)
    peak_amplification = float(np.max(np.abs(waveforms)) / original_peak)
    return waveforms, achieved, condition_number, peak_amplification


@autoinit
class BroadbandModePlaneSource(ModePlaneSource):
    """Experimental exact-isolation mode source for diagnostic comparisons.

    The source solves the requested mode independently for each target wave and
    injects all solved profiles during one FDTD run. Finite-duration temporal
    waveforms are calibrated using the same discrete phasor convention as the
    detector, so the profile at each target frequency is not contaminated by
    the other sampled profiles. Closely spaced frequencies can make this
    construction ill-conditioned; production broadband profile interpolation
    should use a low-order Chebyshev representation instead.
    """

    wave_characters: tuple[WaveCharacter, ...] = frozen_field()
    max_synthesis_condition: float = frozen_field(default=1e6)
    max_spectrum_relative_error: float = frozen_field(default=1e-5)
    max_waveform_peak_amplification: float = frozen_field(default=10.0)
    min_frequency_spacing_cycles: float = frozen_field(default=1.0)
    temporal_oversampling: int = frozen_field(default=8)

    _mode_components: tuple[ModePlaneSource, ...] = private_field()
    _synthesis_condition: float = private_field()
    _spectrum_relative_error: float = private_field()
    _waveform_peak_amplification: float = private_field()
    _frequency_spacing_cycles: float = private_field()

    def __post_init__(self):
        if len(self.wave_characters) < 2:
            raise ValueError(
                "BroadbandModePlaneSource requires at least two wave_characters."
            )
        if self.temporal_oversampling < 1:
            raise ValueError("temporal_oversampling must be at least one.")
        if self.max_synthesis_condition <= 0:
            raise ValueError("max_synthesis_condition must be positive.")
        if self.max_spectrum_relative_error <= 0:
            raise ValueError("max_spectrum_relative_error must be positive.")
        if self.max_waveform_peak_amplification <= 0:
            raise ValueError("max_waveform_peak_amplification must be positive.")
        if self.min_frequency_spacing_cycles <= 0:
            raise ValueError("min_frequency_spacing_cycles must be positive.")

    def apply(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
        electric_conductivity: jax.Array | None = None,
        dispersive_c4: jax.Array | None = None,
    ):
        if electric_conductivity is not None:
            raise NotImplementedError(
                "BroadbandModePlaneSource does not yet support electric "
                "conductivity because conductive eigenmodes are complex and "
                "CustomTimeSignalProfile has no phase-shifted quadrature channel."
            )
        waveforms, achieved, condition_number, peak_amplification = (
            synthesize_isolated_mode_waveforms(
                wave_characters=self.wave_characters,
                temporal_profile=self.temporal_profile,
                carrier_wave=self.wave_character,
                time_step_duration=self._config.time_step_duration,
                num_time_steps=self._config.time_steps_total,
                temporal_oversampling=self.temporal_oversampling,
            )
        )
        desired = np.zeros_like(achieved)
        for index in range(len(self.wave_characters)):
            desired[index, index] = achieved[index, index]
        stored_waveforms = np.asarray(jnp.asarray(waveforms, dtype=self._config.dtype))
        coarse_waveforms = stored_waveforms[:, :: self.temporal_oversampling]
        sample_time = (
            np.arange(self._config.time_steps_total, dtype=np.float64)
            * self._config.time_step_duration
        )
        frequencies = np.asarray(
            [wave.get_frequency() for wave in self.wave_characters],
            dtype=np.float64,
        )
        demodulation = np.exp(
            1j * 2.0 * np.pi * frequencies[:, None] * sample_time[None]
        )
        achieved_stored = (
            2.0 / self._config.time_steps_total * (demodulation @ coarse_waveforms.T)
        )
        spectrum_scale = max(float(np.max(np.abs(desired))), np.finfo(float).tiny)
        spectrum_error = float(
            np.max(np.abs(achieved_stored - desired)) / spectrum_scale
        )
        frequencies = np.sort(frequencies)
        min_spacing = float(np.min(np.diff(frequencies)))
        duration = self._config.time_steps_total * self._config.time_step_duration
        spacing_cycles = min_spacing * duration
        spacing_duration = self.min_frequency_spacing_cycles / min_spacing
        if condition_number > self.max_synthesis_condition:
            condition_duration = (
                duration * (condition_number / self.max_synthesis_condition) ** 0.25
            )
            peak_duration = duration * (
                peak_amplification / self.max_waveform_peak_amplification
            ) ** (1.0 / 3.0)
            recommended_duration = max(
                condition_duration, peak_duration, spacing_duration
            )
            raise ValueError(
                "Broadband mode temporal synthesis is too poorly conditioned "
                "for a reliable stored-dtype FDTD run: "
                f"condition={condition_number:.3e}, "
                f"limit={self.max_synthesis_condition:.3e}. Increase the "
                f"simulation duration from {duration:.3e} s to at least "
                f"approximately {recommended_duration:.3e} s, reduce the "
                "frequency count, or use the center-profile ModePlaneSource."
            )
        if (
            peak_amplification > self.max_waveform_peak_amplification
            or spacing_cycles < self.min_frequency_spacing_cycles
        ):
            peak_duration = duration * (
                peak_amplification / self.max_waveform_peak_amplification
            ) ** (1.0 / 3.0)
            recommended_duration = max(peak_duration, spacing_duration)
            warnings.warn(
                "Broadband mode frequencies are only weakly resolved by the "
                "configured duration: "
                f"peak amplification={peak_amplification:.3e}, "
                f"limit={self.max_waveform_peak_amplification:.3e}, "
                f"minimum-spacing cycles={spacing_cycles:.3e}, "
                f"target={self.min_frequency_spacing_cycles:.3e}. "
                f"Increase the simulation duration from {duration:.3e} s to "
                f"approximately {recommended_duration:.3e} s or reduce the "
                "frequency count. Absolute source-amplitude error can remain "
                "even when the synthesis residual is small.",
                RuntimeWarning,
                stacklevel=2,
            )
        if spectrum_error > self.max_spectrum_relative_error:
            raise ValueError(
                "Broadband mode temporal synthesis missed its spectrum target: "
                f"error={spectrum_error:.3e}, limit={self.max_spectrum_relative_error:.3e}."
            )

        components = []
        for index, wave in enumerate(self.wave_characters):
            temporal_profile = CustomTimeSignalProfile(
                signal=jnp.asarray(stored_waveforms[index]),
                time_step_duration=(
                    self._config.time_step_duration / self.temporal_oversampling
                ),
            )
            component = self.aset("wave_character", wave).aset(
                "temporal_profile", temporal_profile
            )
            component = ModePlaneSource.apply(
                component,
                key=key,
                inv_permittivities=inv_permittivities,
                inv_permeabilities=inv_permeabilities,
                dispersive_c1=dispersive_c1,
                dispersive_c2=dispersive_c2,
                dispersive_c3=dispersive_c3,
                electric_conductivity=electric_conductivity,
                dispersive_c4=dispersive_c4,
            )
            components.append(component)

        center_index = min(
            range(len(self.wave_characters)),
            key=lambda index: abs(
                self.wave_characters[index].get_frequency()
                - self.wave_character.get_frequency()
            ),
        )
        center_component = components[center_index]
        self = self.aset("_mode_components", tuple(components), create_new_ok=True)
        self = self.aset("_synthesis_condition", condition_number, create_new_ok=True)
        self = self.aset("_spectrum_relative_error", spectrum_error, create_new_ok=True)
        self = self.aset(
            "_waveform_peak_amplification", peak_amplification, create_new_ok=True
        )
        self = self.aset(
            "_frequency_spacing_cycles", spacing_cycles, create_new_ok=True
        )
        self = self.aset("_E", center_component._E, create_new_ok=True)
        self = self.aset("_H", center_component._H, create_new_ok=True)
        self = self.aset("_neff", center_component._neff, create_new_ok=True)
        self = self.aset(
            "_inv_permittivity", center_component._inv_permittivity, create_new_ok=True
        )
        self = self.aset(
            "_inv_permeability", center_component._inv_permeability, create_new_ok=True
        )
        self = self.aset(
            "_time_offset_E", center_component._time_offset_E, create_new_ok=True
        )
        self = self.aset(
            "_time_offset_H", center_component._time_offset_H, create_new_ok=True
        )
        return self

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        for component in self._mode_components:
            E = ModePlaneSource.update_E(
                component,
                E=E,
                inv_permittivities=inv_permittivities,
                inv_permeabilities=inv_permeabilities,
                time_step=time_step,
                inverse=inverse,
            )
        return E

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        for component in self._mode_components:
            H = ModePlaneSource.update_H(
                component,
                H=H,
                inv_permittivities=inv_permittivities,
                inv_permeabilities=inv_permeabilities,
                time_step=time_step,
                inverse=inverse,
            )
        return H
