"""Tidy3D-style low-order Chebyshev broadband mode source."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.mode import ModePlaneSource
from fdtdx.objects.sources.profile import CustomTimeSignalProfile, GaussianPulseProfile

from .chebyshev import (
    align_sampled_profiles,
    chebyshev_coefficients,
    chebyshev_sample_frequencies,
    chebyshev_temporal_basis,
)


@autoinit
class ChebyshevModePlaneSource(ModePlaneSource):
    """Mode source using a low-order smooth broadband profile expansion.

    The mode is solved at first-kind Chebyshev nodes over the useful Gaussian
    pulse band. Modal fields are combined with their component-wise Yee delay
    phases, phase-tracked using the raw eigenmodes, and transformed to Chebyshev
    coefficients. Compact analytic Gaussian-derivative waveforms inject those
    coefficients without a second travel-time shift. Objective and detector
    frequencies do not affect the source construction.
    """

    num_freqs: int = frozen_field(default=3)
    band_sigma: float = frozen_field(default=1.5)
    mode_overlap_warning: float = frozen_field(default=0.5)
    temporal_oversampling: int = frozen_field(default=8)
    _mode_components: tuple[ModePlaneSource, ...] = private_field()
    _sample_wave_characters: tuple[WaveCharacter, ...] = private_field()
    _sample_neff: jax.Array = private_field()

    def __post_init__(self):
        if self.num_freqs < 2 or self.num_freqs > 20:
            raise ValueError("ChebyshevModePlaneSource num_freqs must be in [2, 20].")
        if self.band_sigma <= 0:
            raise ValueError("band_sigma must be positive.")
        if self.temporal_oversampling < 1:
            raise ValueError("temporal_oversampling must be at least one.")
        if not isinstance(self.temporal_profile, GaussianPulseProfile):
            raise TypeError("ChebyshevModePlaneSource requires GaussianPulseProfile.")

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
        if dispersive_c1 is not None or electric_conductivity is not None:
            raise NotImplementedError(
                "ChebyshevModePlaneSource currently requires nondispersive, "
                "nonconductive media because each temporal basis needs its own "
                "dispersive H-side filter or conductive quadrature treatment."
            )
        center_frequency = self.temporal_profile.center_wave.get_frequency()
        sample_frequencies = chebyshev_sample_frequencies(
            center_frequency=center_frequency,
            spectral_width=self.temporal_profile.spectral_width.get_frequency(),
            num_freqs=self.num_freqs,
            band_sigma=self.band_sigma,
        )
        sample_waves = tuple(
            WaveCharacter(frequency=float(frequency))
            for frequency in sample_frequencies
        )
        solved = []
        for wave in sample_waves:
            component = self.aset("wave_character", wave)
            component = ModePlaneSource.apply(
                component,
                key=key,
                inv_permittivities=inv_permittivities,
                inv_permeabilities=inv_permeabilities,
            )
            solved.append(component)

        center_index = next(
            (
                index
                for index, wave in enumerate(sample_waves)
                if np.isclose(
                    wave.get_frequency(), center_frequency, rtol=1e-12, atol=0.0
                )
            ),
            None,
        )
        if center_index is None:
            center_component = ModePlaneSource.apply(
                self,
                key=key,
                inv_permittivities=inv_permittivities,
                inv_permeabilities=inv_permeabilities,
            )
        else:
            center_component = solved[center_index]

        raw_sampled_fields = []
        effective_sampled_fields = []
        for component in solved:
            angular_frequency = 2.0 * np.pi * component.wave_character.get_frequency()
            electric_delay = np.exp(
                -1j
                * angular_frequency
                * np.asarray(component._time_offset_E)
                * self._config.time_step_duration
            )
            magnetic_delay = np.exp(
                -1j
                * angular_frequency
                * np.asarray(component._time_offset_H)
                * self._config.time_step_duration
            )
            raw_fields = np.concatenate(
                (np.asarray(component._E), np.asarray(component._H)), axis=0
            )
            effective_fields = np.concatenate(
                (
                    np.asarray(component._E) * electric_delay,
                    np.asarray(component._H) * magnetic_delay,
                ),
                axis=0,
            )
            raw_sampled_fields.append(raw_fields)
            effective_sampled_fields.append(effective_fields)
        raw_sampled_fields = np.stack(raw_sampled_fields)
        effective_sampled_fields = np.stack(effective_sampled_fields)
        raw_aligned, alignment_phases = align_sampled_profiles(raw_sampled_fields)
        phase_shape = (alignment_phases.shape[0],) + (1,) * (
            effective_sampled_fields.ndim - 1
        )
        aligned = effective_sampled_fields * alignment_phases.reshape(phase_shape)
        center_raw_fields = np.concatenate(
            (np.asarray(center_component._E), np.asarray(center_component._H)), axis=0
        )
        raw_center_polynomial = np.polynomial.chebyshev.chebval(
            0.0, chebyshev_coefficients(raw_aligned)
        )
        center_overlap = np.vdot(
            center_raw_fields.reshape(-1), raw_center_polynomial.reshape(-1)
        )
        if center_overlap != 0:
            aligned *= np.exp(-1j * np.angle(center_overlap))
        coefficients = chebyshev_coefficients(aligned)
        stored_num_steps = (
            self._config.time_steps_total - 1
        ) * self.temporal_oversampling + 1
        basis = chebyshev_temporal_basis(
            temporal_profile=self.temporal_profile,
            time_step_duration=(
                self._config.time_step_duration / self.temporal_oversampling
            ),
            num_time_steps=stored_num_steps,
            num_terms=self.num_freqs,
            band_sigma=self.band_sigma,
        )
        components = []
        dtype = self._config.dtype
        for order in range(self.num_freqs):
            for spatial, signal in (
                (np.real(coefficients[order]), np.real(basis[order])),
                (-np.imag(coefficients[order]), np.imag(basis[order])),
            ):
                if np.max(np.abs(spatial)) < 1e-12 or np.max(np.abs(signal)) < 1e-12:
                    continue
                profile = CustomTimeSignalProfile(
                    signal=jnp.asarray(signal, dtype=dtype),
                    time_step_duration=(
                        self._config.time_step_duration / self.temporal_oversampling
                    ),
                )
                component = center_component.aset("temporal_profile", profile)
                component = component.aset(
                    "_E", jnp.asarray(spatial[:3], dtype=dtype), create_new_ok=True
                )
                component = component.aset(
                    "_H", jnp.asarray(spatial[3:], dtype=dtype), create_new_ok=True
                )
                component = component.aset(
                    "_temporal_H_filter", None, create_new_ok=True
                )
                component = component.aset(
                    "_time_offset_E",
                    jnp.zeros_like(center_component._time_offset_E),
                    create_new_ok=True,
                )
                component = component.aset(
                    "_time_offset_H",
                    jnp.zeros_like(center_component._time_offset_H),
                    create_new_ok=True,
                )
                components.append(component)

        self = self.aset("_mode_components", tuple(components), create_new_ok=True)
        self = self.aset("_sample_wave_characters", sample_waves, create_new_ok=True)
        self = self.aset(
            "_sample_neff",
            jnp.asarray([component._neff for component in solved]),
            create_new_ok=True,
        )
        for name in (
            "_E",
            "_H",
            "_neff",
            "_inv_permittivity",
            "_inv_permeability",
            "_time_offset_E",
            "_time_offset_H",
        ):
            self = self.aset(name, getattr(center_component, name), create_new_ok=True)
        return self

    def update_E(self, E, inv_permittivities, inv_permeabilities, time_step, inverse):
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

    def update_H(self, H, inv_permittivities, inv_permeabilities, time_step, inverse):
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
