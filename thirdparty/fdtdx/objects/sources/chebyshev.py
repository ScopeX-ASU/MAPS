"""Stable Chebyshev broadband-profile utilities and distributed source."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.profile import GaussianPulseProfile

from .distributed import DistributedCurrentSource


def chebyshev_sample_frequencies(
    *,
    center_frequency: float,
    spectral_width: float,
    num_freqs: int,
    band_sigma: float = 1.5,
) -> np.ndarray:
    """Return ascending first-kind Chebyshev nodes over a Gaussian source band.

    Args:
        center_frequency: Pulse center frequency in Hz.
        spectral_width: Gaussian spectral standard deviation in Hz.
        num_freqs: Number of profile samples. One returns only the center.
        band_sigma: Half-width of the interpolation band in spectral standard
            deviations.

    Returns:
        Sampling frequencies with shape ``(num_freqs,)``.

    Raises:
        ValueError: If an argument is outside its physical range or the lower
            interpolation edge is not positive.
    """
    if num_freqs < 1 or num_freqs > 20:
        raise ValueError("num_freqs must be between 1 and 20.")
    if center_frequency <= 0 or spectral_width <= 0 or band_sigma <= 0:
        raise ValueError("Frequencies and band_sigma must be positive.")
    if num_freqs == 1:
        return np.asarray([center_frequency], dtype=np.float64)
    half_width = band_sigma * spectral_width
    if center_frequency <= half_width:
        raise ValueError(
            "The Chebyshev source band must contain only positive frequencies."
        )
    angles = np.pi * (2 * np.arange(num_freqs) + 1) / (2 * num_freqs)
    nodes = np.cos(angles)[::-1]
    return center_frequency + half_width * nodes


def align_sampled_profiles(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Phase-align sampled complex profiles to the sample nearest band center.

    Args:
        samples: Complex array shaped ``(num_freqs, ...)``.

    Returns:
        Aligned samples and their unit-magnitude phase factors.

    Notes:
        This tracks a smooth branch only when the requested mode index does not
        cross another mode. Small adjacent overlaps trigger a warning.
    """
    values = np.asarray(samples)
    if values.ndim < 2:
        raise ValueError("samples must have a frequency and spatial axis.")
    reference_index = values.shape[0] // 2
    reference = values[reference_index].reshape(-1)
    aligned = values.astype(np.complex128, copy=True)
    phases = np.ones(values.shape[0], dtype=np.complex128)
    for indices in (
        range(reference_index - 1, -1, -1),
        range(reference_index + 1, values.shape[0]),
    ):
        previous = reference
        for index in indices:
            current = values[index].reshape(-1)
            overlap = np.vdot(previous, current)
            coherence = abs(overlap) / (
                np.linalg.norm(previous) * np.linalg.norm(current)
                + np.finfo(float).tiny
            )
            if coherence < 0.5:
                warnings.warn(
                    f"Possible mode-branch crossing at Chebyshev sample {index}: "
                    f"adjacent overlap={coherence:.3f}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            phase = np.exp(-1j * np.angle(overlap)) if overlap != 0 else 1.0
            phases[index] = phase
            aligned[index] *= phase
            previous = aligned[index].reshape(-1)
    return aligned, phases


def chebyshev_coefficients(samples: np.ndarray) -> np.ndarray:
    """Compute stable first-kind Chebyshev coefficients from node samples."""
    values = np.asarray(samples)
    count = values.shape[0]
    if count == 1:
        return values.copy()
    angles = np.pi * (2 * np.arange(count) + 1) / (2 * count)
    # Samples are stored at ascending nodes, opposite the conventional angle order.
    descending = values[::-1]
    transform = np.cos(np.arange(count)[:, None] * angles[None, :])
    coefficients = (2.0 / count) * np.tensordot(transform, descending, axes=(1, 0))
    coefficients[0] *= 0.5
    return coefficients


def chebyshev_temporal_basis(
    *,
    temporal_profile: GaussianPulseProfile,
    time_step_duration: float,
    num_time_steps: int,
    num_terms: int,
    band_sigma: float = 1.5,
    polynomial_half_width: float | None = None,
) -> np.ndarray:
    """Build compact analytic waveforms for Chebyshev frequency basis functions.

    A polynomial in frequency detuning is a finite combination of derivatives
    of the Gaussian envelope. Evaluating those derivatives analytically avoids
    FFT truncation, spectral masking, and long ringing tails.

    Args:
        temporal_profile: Gaussian carrier and spectral width.
        time_step_duration: Source sample interval in seconds.
        num_time_steps: Number of samples to construct.
        num_terms: Number of Chebyshev temporal basis terms.
        band_sigma: Default polynomial half-width in Gaussian spectral widths.
        polynomial_half_width: Optional explicit polynomial half-width in Hz,
            overriding ``band_sigma`` for fits at objective frequencies.

    Returns:
        Complex analytic waveforms shaped ``(num_terms, num_time_steps)``.
    """
    time = np.arange(num_time_steps, dtype=np.float64) * time_step_duration
    center_wave = temporal_profile.center_wave
    center = center_wave.get_frequency()
    half_width = (
        band_sigma * temporal_profile.spectral_width.get_frequency()
        if polynomial_half_width is None
        else float(polynomial_half_width)
    )
    if half_width <= 0:
        raise ValueError("polynomial_half_width must be positive.")
    sigma_t = 1.0 / (2.0 * np.pi * temporal_profile.spectral_width.get_frequency())
    offset = time - 6.0 * sigma_t
    normalized_time = offset / sigma_t
    envelope = np.exp(-0.5 * normalized_time**2)
    carrier = np.exp(-1j * (2.0 * np.pi * center * time + center_wave.phase_shift))
    basis = []
    for order in range(num_terms):
        cheb = np.polynomial.Chebyshev.basis(order)
        power_coefficients = cheb.convert(kind=np.polynomial.Polynomial).coef
        waveform = np.zeros(num_time_steps, dtype=np.complex128)
        for derivative_order, coefficient in enumerate(power_coefficients):
            hermite_coefficients = np.zeros(derivative_order + 1)
            hermite_coefficients[-1] = 1.0
            hermite = np.polynomial.hermite_e.hermeval(
                normalized_time, hermite_coefficients
            )
            envelope_derivative = (
                (-1.0) ** derivative_order
                * hermite
                * envelope
                / sigma_t**derivative_order
            )
            waveform += (
                coefficient
                * (1j / (2.0 * np.pi * half_width)) ** derivative_order
                * envelope_derivative
                * carrier
            )
        basis.append(waveform)
    return np.asarray(basis)


def chebyshev_temporal_diagnostics(
    *,
    temporal_profile: GaussianPulseProfile,
    frequencies: np.ndarray,
    coefficient_norms: np.ndarray,
    polynomial_half_width: float,
    time_step_duration: float,
    num_time_steps: int,
) -> dict[str, float]:
    """Bound sampled current-spectrum error and temporal amplification.

    Args:
        temporal_profile: Gaussian carrier and envelope used for injection.
        frequencies: Target frequencies in Hz.
        coefficient_norms: Spatial coefficient norms, one per basis term.
        polynomial_half_width: Chebyshev coordinate scale in Hz.
        time_step_duration: Actual FDTD time step in seconds.
        num_time_steps: Number of source samples in the configured simulation.

    Returns:
        A norm-weighted bound on discrete basis error, a bound on peak source
        norm, and the minimum relative normalization spectrum.

    Raises:
        ValueError: If the normalization spectrum is nearly zero.
    """
    basis = (
        chebyshev_temporal_basis(
            temporal_profile=temporal_profile,
            time_step_duration=time_step_duration,
            num_time_steps=num_time_steps,
            num_terms=len(coefficient_norms),
            polynomial_half_width=polynomial_half_width,
        )
        .astype(np.complex64)
        .astype(np.complex128)
    )
    peak_bound = float(np.sum(coefficient_norms * np.max(np.abs(basis), axis=1)))
    times = np.arange(num_time_steps) * time_step_duration
    dft = np.exp(2j * np.pi * np.asarray(frequencies)[:, None] * times)
    real_response = dft @ basis.real.T
    imag_response = dft @ basis.imag.T
    base = real_response[:, 0]
    largest = float(np.max(np.abs(base)))
    relative_min = float(np.min(np.abs(base)) / largest) if largest > 0 else 0.0
    if relative_min < 1e-8:
        raise ValueError("Source spectrum is too small at an objective frequency.")
    center = temporal_profile.center_wave.get_frequency()
    coordinate = (np.asarray(frequencies) - center) / polynomial_half_width
    expected = np.polynomial.chebyshev.chebvander(
        coordinate, len(coefficient_norms) - 1
    )
    response_error = max(
        np.max(np.abs(real_response / base[:, None] - expected) @ coefficient_norms),
        np.max(
            np.abs(imag_response / base[:, None] + 1j * expected) @ coefficient_norms
        ),
    )
    return {
        "temporal_basis_error_max": float(response_error),
        "waveform_peak_norm_bound": peak_bound,
        "source_spectrum_min_relative": relative_min,
    }


@autoinit
class ChebyshevDistributedCurrentSource(DistributedCurrentSource):
    """Distributed source whose profiles vary smoothly over a Gaussian band.

    ``electric_profiles`` and ``magnetic_profiles`` contain profile samples at
    ``sample_wave_characters``. During ``apply`` they are phase-aligned,
    transformed to Chebyshev coefficients, and paired with compact temporal
    basis signals. Objective/monitor frequencies do not enter this transform.
    """

    sample_wave_characters: tuple[WaveCharacter, ...] = frozen_field()
    band_sigma: float = frozen_field(default=1.5)
    _basis_waveforms: jax.Array = private_field()

    def __post_init__(self):
        if len(self.sample_wave_characters) != len(self.wave_characters):
            raise ValueError("sample_wave_characters and profile counts must match.")
        if len(self.sample_wave_characters) > 20:
            raise ValueError("Chebyshev broadband sources support at most 20 samples.")
        if not isinstance(self.temporal_profile, GaussianPulseProfile):
            raise TypeError("Chebyshev broadband sources require GaussianPulseProfile.")
        expected = chebyshev_sample_frequencies(
            center_frequency=self.temporal_profile.center_wave.get_frequency(),
            spectral_width=self.temporal_profile.spectral_width.get_frequency(),
            num_freqs=len(self.sample_wave_characters),
            band_sigma=self.band_sigma,
        )
        actual = np.asarray(
            [wave.get_frequency() for wave in self.sample_wave_characters]
        )
        if not np.allclose(actual, expected, rtol=1e-10, atol=0.0):
            raise ValueError(
                "sample_wave_characters must be the Chebyshev nodes returned "
                "by chebyshev_sample_frequencies for this temporal profile."
            )

    def apply(self, *args, **kwargs):
        electric = None
        magnetic = None
        phases = None
        if self.electric_profiles is not None and self.magnetic_profiles is not None:
            combined = np.concatenate(
                (
                    np.asarray(self.electric_profiles),
                    np.asarray(self.magnetic_profiles),
                ),
                axis=1,
            )
            aligned, phases = align_sampled_profiles(combined)
            electric = chebyshev_coefficients(aligned[:, :3])
            magnetic = chebyshev_coefficients(aligned[:, 3:])
        elif self.electric_profiles is not None:
            aligned, phases = align_sampled_profiles(np.asarray(self.electric_profiles))
            electric = chebyshev_coefficients(aligned)
        elif self.magnetic_profiles is not None:
            aligned, phases = align_sampled_profiles(np.asarray(self.magnetic_profiles))
            magnetic = chebyshev_coefficients(aligned)
        del phases
        basis = chebyshev_temporal_basis(
            temporal_profile=self.temporal_profile,
            time_step_duration=self._config.time_step_duration,
            num_time_steps=self._config.time_steps_total,
            num_terms=len(self.sample_wave_characters),
            band_sigma=self.band_sigma,
        )
        self = self.aset(
            "electric_profiles", None if electric is None else jnp.asarray(electric)
        )
        self = self.aset(
            "magnetic_profiles", None if magnetic is None else jnp.asarray(magnetic)
        )
        self = self.aset("_basis_waveforms", jnp.asarray(basis), create_new_ok=True)
        # The parent validates the coefficient leading axis against wave_characters.
        return super().apply(*args, **kwargs)

    def _complex_amplitudes(self, time_step: jax.Array) -> jax.Array:
        index = jnp.asarray(time_step, dtype=jnp.int32)
        values = self._basis_waveforms[
            :, jnp.clip(index, 0, self._basis_waveforms.shape[1] - 1)
        ]
        valid = (index >= 0) & (index < self._basis_waveforms.shape[1])
        return jnp.where(valid, values, 0.0).astype(jnp.complex64)


@autoinit
class FactoredChebyshevCurrentSource(DistributedCurrentSource):
    """Inject smooth spatial profiles fitted at arbitrary target frequencies.

    The input profiles are samples of a phase-tracked, unit-scale spatial
    family. Arbitrary objective amplitudes are intentionally excluded and are
    applied to the recorded adjoint fields by the caller. During placement the
    samples are least-squares fitted in a Chebyshev basis centered on the
    Gaussian carrier, and each spatial coefficient is paired with its compact
    analytic temporal basis waveform.

    Attributes:
        sample_wave_characters: Frequencies corresponding to input profiles.
        polynomial_half_width: Frequency scale used to map samples into the
            Chebyshev coordinate. All samples must map into ``[-1, 1]``.
    """

    sample_wave_characters: tuple[WaveCharacter, ...] = frozen_field()
    polynomial_half_width: float = frozen_field()
    _basis_waveforms: jax.Array = private_field()

    def __post_init__(self):
        if self.electric_profiles is None and self.magnetic_profiles is None:
            raise ValueError("At least one electric or magnetic profile is required.")
        if self.synthesis != "direct" or self.blocks:
            raise ValueError(
                "Factored Chebyshev sources require direct dense profiles."
            )
        if not isinstance(self.temporal_profile, GaussianPulseProfile):
            raise TypeError("Factored Chebyshev sources require GaussianPulseProfile.")
        if len(self.sample_wave_characters) == 0:
            raise ValueError("At least one profile sample is required.")
        if len(self.wave_characters) == 0 or len(self.wave_characters) > 20:
            raise ValueError("The Chebyshev basis must contain between 1 and 20 terms.")
        if len(self.wave_characters) > len(self.sample_wave_characters):
            raise ValueError("The number of basis terms cannot exceed profile samples.")
        if self.polynomial_half_width <= 0:
            raise ValueError("polynomial_half_width must be positive.")

    def _validate_profiles(self) -> None:
        """Accept sample profiles at placement and coefficient profiles at apply."""
        if self.blocks:
            raise ValueError(
                "Factored Chebyshev sources currently require dense profiles."
            )
        for profile in (self.electric_profiles, self.magnetic_profiles):
            if profile is None:
                continue
            if tuple(profile.shape[1:]) != (3, *self.grid_shape):
                raise ValueError(
                    "Current profile spatial shape differs from source placement."
                )
            if profile.shape[0] not in (
                len(self.sample_wave_characters),
                len(self.wave_characters),
            ):
                raise ValueError(
                    "Current profiles have an invalid leading frequency axis."
                )

    def _fit_profiles(self, profiles: jax.Array | None) -> jax.Array | None:
        """Fit source samples using only the small frequency matrix inverse.

        Args:
            profiles: Optional complex current samples with a leading sample
                axis and remaining shape ``(3, *grid_shape)``.

        Returns:
            Chebyshev spatial coefficients, or None for an absent current type.
        """
        if profiles is None:
            return None
        values = np.asarray(profiles)
        expected = (len(self.sample_wave_characters), 3, *self.grid_shape)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Profile samples must have shape {expected}, got {tuple(values.shape)}"
            )
        center = self.temporal_profile.center_wave.get_frequency()
        frequencies = np.asarray(
            [wave.get_frequency() for wave in self.sample_wave_characters],
            dtype=np.float64,
        )
        coordinate = (frequencies - center) / self.polynomial_half_width
        if np.max(np.abs(coordinate)) > 1.0 + 1e-10:
            raise ValueError("Profile frequencies fall outside the Chebyshev interval.")
        flattened = values.reshape(values.shape[0], -1)
        vandermonde = np.polynomial.chebyshev.chebvander(
            coordinate, len(self.wave_characters) - 1
        )
        coefficients = np.linalg.pinv(vandermonde) @ flattened
        return jnp.asarray(
            coefficients.reshape((len(self.wave_characters), *values.shape[1:]))
        )

    def apply(self, *args, **kwargs):
        """Fit profiles and prepare the compact time-domain injection basis.

        Args:
            *args: Positional arguments of ``DistributedCurrentSource.apply``.
            **kwargs: Material arrays and key accepted by the parent method.

        Returns:
            Applied source with coefficient profiles and sampled waveforms.
        """
        electric = self._fit_profiles(self.electric_profiles)
        magnetic = self._fit_profiles(self.magnetic_profiles)
        basis = chebyshev_temporal_basis(
            temporal_profile=self.temporal_profile,
            time_step_duration=self._config.time_step_duration,
            num_time_steps=self._config.time_steps_total,
            num_terms=len(self.wave_characters),
            polynomial_half_width=self.polynomial_half_width,
        )
        self = self.aset("electric_profiles", electric)
        self = self.aset("magnetic_profiles", magnetic)
        self = self.aset("_basis_waveforms", jnp.asarray(basis), create_new_ok=True)
        return DistributedCurrentSource.apply(self, *args, **kwargs)

    def _complex_amplitudes(self, time_step: jax.Array) -> jax.Array:
        index = jnp.asarray(time_step, dtype=jnp.int32)
        values = self._basis_waveforms[
            :, jnp.clip(index, 0, self._basis_waveforms.shape[1] - 1)
        ]
        valid = (index >= 0) & (index < self._basis_waveforms.shape[1])
        return jnp.where(valid, values, 0.0).astype(jnp.complex64)

    def active_until_step(self, source_decay_by: float) -> int:
        """Return a conservative pulse-off step for all injected basis terms.

        Args:
            source_decay_by: Relative source-envelope cutoff in ``(0, 1)``.

        Returns:
            First absolute time step after the coefficient-weighted envelope
            bound falls below the relative cutoff for the rest of the run.
        """
        coefficient_norms = np.zeros(len(self.wave_characters), dtype=np.float64)
        for profiles in (self.electric_profiles, self.magnetic_profiles):
            if profiles is not None:
                flattened = np.asarray(profiles).reshape(len(self.wave_characters), -1)
                coefficient_norms += np.linalg.norm(flattened, axis=1) ** 2
        coefficient_norms = np.sqrt(coefficient_norms)
        bound = coefficient_norms @ np.abs(np.asarray(self._basis_waveforms))
        peak = float(np.max(bound))
        active = np.flatnonzero(bound > peak * source_decay_by)
        return int(active[-1]) + 1 if active.size else 0
