from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.profile import GaussianPulseProfile
from fdtdx.objects.sources.source import Source


class CurrentSourceBlock(NamedTuple):
    """Localized current profiles and their global Yee-grid slice."""

    grid_slice: tuple[slice, slice, slice]
    electric_profiles: jax.Array | None
    magnetic_profiles: jax.Array | None


@autoinit
class DistributedCurrentSource(Source):
    """Soft distributed electric and magnetic current source.

    This source injects complex frequency-domain current profiles over its grid
    slice by direct phasor synthesis. It is intended for adjoint sources built
    from DFT-field cotangents, where each frequency sample has its own complex
    spatial profile.
    """

    #: Frequency samples corresponding to the leading axis of the profiles.
    wave_characters: tuple[WaveCharacter, ...] = frozen_field()

    #: Complex electric current profiles, shape ``(nfreq, 3, *grid_shape)``.
    electric_profiles: jax.Array | None = frozen_field(default=None)

    #: Complex magnetic current profiles, shape ``(nfreq, 3, *grid_shape)``.
    magnetic_profiles: jax.Array | None = frozen_field(default=None)

    #: Optional localized dense current blocks in global grid coordinates.
    blocks: tuple[CurrentSourceBlock, ...] | None = frozen_field(default=None)

    #: Synthesis mode. ``direct`` sums the requested phasors at every time step.
    synthesis: str = frozen_field(default="direct")

    #: Additional source amplitude.
    amplitude: float = frozen_field(default=1.0)

    def __post_init__(self):
        if self.synthesis != "direct":
            raise ValueError(
                f"Only direct synthesis is currently supported, got {self.synthesis!r}"
            )
        if len(self.wave_characters) == 0:
            raise ValueError(
                "DistributedCurrentSource requires at least one wave_character"
            )
        if (
            self.electric_profiles is None
            and self.magnetic_profiles is None
            and not self.blocks
        ):
            raise ValueError("At least one current profile or block must be provided")
        if self.blocks:
            for block in self.blocks:
                if block.electric_profiles is None and block.magnetic_profiles is None:
                    raise ValueError(
                        "Each current source block must contain E or H profiles"
                    )

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
        self = super().apply(
            key=key,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            dispersive_c1=dispersive_c1,
            dispersive_c2=dispersive_c2,
            dispersive_c3=dispersive_c3,
            electric_conductivity=electric_conductivity,
            dispersive_c4=dispersive_c4,
        )
        self._validate_profiles()
        return self

    def place_on_grid(
        self,
        grid_slice_tuple,
        config,
        key: jax.Array,
    ):
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        self._validate_profiles()
        return self

    def _validate_profiles(self) -> None:
        if self.blocks:
            for block in self.blocks:
                expected = (
                    len(self.wave_characters),
                    3,
                    *tuple(axis.stop - axis.start for axis in block.grid_slice),
                )
                for name, profile in (
                    ("electric_profiles", block.electric_profiles),
                    ("magnetic_profiles", block.magnetic_profiles),
                ):
                    if profile is not None and tuple(profile.shape) != expected:
                        raise ValueError(
                            f"block {name} must have shape {expected}, got {tuple(profile.shape)}"
                        )
            return
        expected = (len(self.wave_characters), 3, *self.grid_shape)
        for name, profile in (
            ("electric_profiles", self.electric_profiles),
            ("magnetic_profiles", self.magnetic_profiles),
        ):
            if profile is None:
                continue
            if tuple(profile.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(profile.shape)}"
                )

    def _complex_amplitudes(self, time_step: jax.Array) -> jax.Array:
        dt = self._config.time_step_duration
        time = time_step * dt
        if isinstance(self.temporal_profile, GaussianPulseProfile):
            spectral_width_hz = self.temporal_profile.spectral_width.get_frequency()
            sigma_t = 1.0 / (2 * jnp.pi * spectral_width_hz)
            t0 = 6 * sigma_t
            envelope = jnp.exp(-((time - t0) ** 2) / (2 * sigma_t**2))
            amps = [
                envelope
                * jnp.exp(
                    -1j * (2 * jnp.pi * wave.get_frequency() * time + wave.phase_shift)
                )
                for wave in self.wave_characters
            ]
            return jnp.asarray(amps, dtype=jnp.complex64)

        amps = [
            self.temporal_profile.get_complex_amplitude(
                time=time,
                period=wave.get_period(),
                phase_shift=wave.phase_shift,
            )
            for wave in self.wave_characters
        ]
        return jnp.asarray(amps, dtype=jnp.complex64)

    def _synthesize(self, profiles: jax.Array, time_step: jax.Array) -> jax.Array:
        amplitudes = self._complex_amplitudes(time_step)
        scale = self.amplitude * self.static_amplitude_factor
        field = jnp.sum(profiles * amplitudes[:, None, None, None, None], axis=0)
        return (scale * jnp.real(field)).astype(self._config.dtype)

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permeabilities
        if self.blocks:
            c = self._config.courant_number
            sign = -1.0 if not inverse else 1.0
            for block in self.blocks:
                if block.electric_profiles is None:
                    continue
                current = self._synthesize(block.electric_profiles, time_step)
                sl = block.grid_slice
                injection = c * inv_permittivities[:, *sl] * current
                E = E.at[:, *sl].add(sign * injection.astype(E.dtype))
            return E
        if self.electric_profiles is None:
            return E

        c = self._config.courant_number
        sign = -1.0 if not inverse else 1.0
        current = self._synthesize(self.electric_profiles, time_step)
        injection = c * inv_permittivities[:, *self.grid_slice] * current
        return E.at[:, *self.grid_slice].add(sign * injection.astype(E.dtype))

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities
        if self.blocks:
            c = self._config.courant_number
            sign = -1.0 if not inverse else 1.0
            for block in self.blocks:
                if block.magnetic_profiles is None:
                    continue
                current = self._synthesize(block.magnetic_profiles, time_step)
                sl = block.grid_slice
                if (
                    isinstance(inv_permeabilities, jax.Array)
                    and inv_permeabilities.ndim > 0
                ):
                    inv_mu = inv_permeabilities[:, *sl]
                else:
                    inv_mu = inv_permeabilities
                injection = c * inv_mu * current
                H = H.at[:, *sl].add(sign * injection.astype(H.dtype))
            return H
        if self.magnetic_profiles is None:
            return H

        c = self._config.courant_number
        sign = -1.0 if not inverse else 1.0
        current = self._synthesize(self.magnetic_profiles, time_step)
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_mu = inv_permeabilities[:, *self.grid_slice]
        else:
            inv_mu = inv_permeabilities
        injection = c * inv_mu * current
        return H.at[:, *self.grid_slice].add(sign * injection.astype(H.dtype))


def make_gaussian_distributed_current_source(
    *,
    center_wave: WaveCharacter,
    spectral_width: WaveCharacter,
    **kwargs,
) -> DistributedCurrentSource:
    """Construct a distributed current source with a Gaussian pulse profile."""
    return DistributedCurrentSource(
        temporal_profile=GaussianPulseProfile(
            center_wave=center_wave,
            spectral_width=spectral_width,
        ),
        **kwargs,
    )
