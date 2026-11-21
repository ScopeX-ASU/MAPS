"""Temporal synthesis contracts for the forward broadband mode source."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")
pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from thirdparty.fdtdx.fdtd.stop_condition import _gaussian_pulse_off_step  # noqa: E402
from thirdparty.fdtdx.objects.sources.broadband_mode import (  # noqa: E402
    BroadbandModePlaneSource,
    synthesize_isolated_mode_waveforms,
)

pytestmark = [pytest.mark.unit, pytest.mark.fdtdx]


def test_isolated_mode_waveforms_match_discrete_complex_targets_and_zero_dc():
    """Every synthesized waveform selects one requested frequency exactly."""
    waves = tuple(
        fdtdx.WaveCharacter(wavelength=wavelength)
        for wavelength in (1.295e-6, 1.310e-6, 1.325e-6)
    )
    carrier = waves[1]
    profile = fdtdx.GaussianPulseProfile(
        center_wave=carrier,
        spectral_width=fdtdx.WaveCharacter(wavelength=1.310e-6 * 20),
    )
    dt = 1.0e-16
    num_steps = 8000
    waveforms, achieved, condition, peak_amplification = (
        synthesize_isolated_mode_waveforms(
            wave_characters=waves,
            temporal_profile=profile,
            carrier_wave=carrier,
            time_step_duration=dt,
            num_time_steps=num_steps,
        )
    )

    time = np.arange(num_steps) * dt
    original = np.asarray(
        profile.get_amplitude(
            time=jnp.asarray(time),
            period=carrier.get_period(),
            phase_shift=carrier.phase_shift,
        )
    )
    demodulation = np.exp(
        1j
        * 2
        * np.pi
        * np.asarray([wave.get_frequency() for wave in waves])[:, None]
        * time[None]
    )
    target = (2.0 / num_steps) * (demodulation @ original)
    expected = np.diag(target)

    assert waveforms.shape == (len(waves), (num_steps - 1) * 8 + 1)
    assert np.isfinite(condition)
    assert np.isfinite(peak_amplification)
    assert np.allclose(achieved, expected, rtol=2e-6, atol=2e-7)
    assert np.allclose(np.mean(waveforms[:, ::8], axis=1), 0.0, atol=1e-9)


def test_isolated_mode_waveforms_reject_duplicate_frequencies():
    """Duplicate target frequencies make profile selection undefined."""
    wave = fdtdx.WaveCharacter(wavelength=1.31e-6)
    profile = fdtdx.GaussianPulseProfile(
        center_wave=wave,
        spectral_width=fdtdx.WaveCharacter(wavelength=1.31e-6 * 20),
    )
    with pytest.raises(ValueError, match="distinct"):
        synthesize_isolated_mode_waveforms(
            wave_characters=(wave, wave),
            temporal_profile=profile,
            carrier_wave=wave,
            time_step_duration=1e-16,
            num_time_steps=8000,
        )


def test_longer_duration_improves_close_frequency_conditioning():
    """Close wavelength samples need a longer temporal observation window."""
    waves = tuple(
        fdtdx.WaveCharacter(wavelength=wavelength)
        for wavelength in (1.3075e-6, 1.3100e-6, 1.3125e-6)
    )
    carrier = waves[1]
    profile = fdtdx.GaussianPulseProfile(
        center_wave=carrier,
        spectral_width=fdtdx.WaveCharacter(wavelength=1.310e-6 * 20),
    )

    _, _, short_condition, short_peak = synthesize_isolated_mode_waveforms(
        wave_characters=waves,
        temporal_profile=profile,
        carrier_wave=carrier,
        time_step_duration=1e-16,
        num_time_steps=8000,
    )
    _, _, long_condition, long_peak = synthesize_isolated_mode_waveforms(
        wave_characters=waves,
        temporal_profile=profile,
        carrier_wave=carrier,
        time_step_duration=1e-16,
        num_time_steps=32000,
    )

    assert short_peak > 10.0
    assert long_peak < 1.0
    assert long_condition < short_condition / 100


def _broadband_source():
    """Build an unplaced broadband source for apply-path contract tests."""
    waves = tuple(
        fdtdx.WaveCharacter(wavelength=wavelength)
        for wavelength in (1.295e-6, 1.310e-6, 1.325e-6)
    )
    profile = fdtdx.GaussianPulseProfile(
        center_wave=waves[1],
        spectral_width=fdtdx.WaveCharacter(wavelength=1.310e-6 * 20),
    )
    source = BroadbandModePlaneSource(
        name="broadband",
        partial_grid_shape=(1, None, None),
        wave_character=waves[1],
        wave_characters=waves,
        direction="+",
        mode_index=0,
        temporal_profile=profile,
    )
    return source.aset(
        "_config",
        SimpleNamespace(
            time_step_duration=1e-16,
            time_steps_total=8000,
            dtype=jnp.float32,
        ),
    )


def test_broadband_source_forwards_dispersion_to_each_mode(monkeypatch):
    """Every wavelength solve must see the simulation's dispersive material."""
    calls = []

    def fake_apply(component, **kwargs):
        calls.append(kwargs)
        for name, value in (
            ("_E", jnp.zeros((3, 1, 2, 2))),
            ("_H", jnp.zeros((3, 1, 2, 2))),
            ("_neff", jnp.asarray(2.0)),
            ("_inv_permittivity", jnp.ones((3, 1, 2, 2))),
            ("_inv_permeability", jnp.asarray(1.0)),
            ("_time_offset_E", jnp.zeros((3, 1, 2, 2))),
            ("_time_offset_H", jnp.zeros((3, 1, 2, 2))),
        ):
            component = component.aset(name, value, create_new_ok=True)
        return component

    monkeypatch.setattr(
        "thirdparty.fdtdx.objects.sources.broadband_mode.ModePlaneSource.apply",
        fake_apply,
    )
    c1 = jnp.ones((1, 1, 1, 2, 2))
    c2 = 2 * c1
    c3 = 3 * c1
    c4 = 4 * c1
    close_waves = tuple(
        fdtdx.WaveCharacter(wavelength=wavelength)
        for wavelength in (1.3075e-6, 1.3100e-6, 1.3125e-6)
    )
    source = _broadband_source().aset("wave_characters", close_waves)
    with pytest.warns(RuntimeWarning, match="weakly resolved"):
        source.apply(
            key=jnp.asarray([0, 0], dtype=jnp.uint32),
            inv_permittivities=jnp.ones((3, 1, 2, 2)),
            inv_permeabilities=1.0,
            dispersive_c1=c1,
            dispersive_c2=c2,
            dispersive_c3=c3,
            dispersive_c4=c4,
        )

    assert len(calls) == 3
    assert all(call["dispersive_c1"] is c1 for call in calls)
    assert all(call["dispersive_c2"] is c2 for call in calls)
    assert all(call["dispersive_c3"] is c3 for call in calls)
    assert all(call["dispersive_c4"] is c4 for call in calls)


def test_broadband_source_rejects_conductivity_quadrature():
    """Sampled signals cannot yet inject the complex conductive mode phase."""
    with pytest.raises(NotImplementedError, match="quadrature"):
        _broadband_source().apply(
            key=jnp.asarray([0, 0], dtype=jnp.uint32),
            inv_permittivities=jnp.ones((3, 1, 2, 2)),
            inv_permeabilities=1.0,
            electric_conductivity=jnp.ones((1, 1, 2, 2)),
        )


def test_broadband_source_rejects_unreliable_close_spacing():
    """Unsafe temporal conditioning must stop before launching FDTD."""
    close_waves = tuple(
        fdtdx.WaveCharacter(wavelength=wavelength)
        for wavelength in (1.309e-6, 1.310e-6, 1.311e-6)
    )
    source = _broadband_source().aset("wave_characters", close_waves)

    with pytest.raises(ValueError, match="poorly conditioned"):
        source.apply(
            key=jnp.asarray([0, 0], dtype=jnp.uint32),
            inv_permittivities=jnp.ones((3, 1, 2, 2)),
            inv_permeabilities=1.0,
        )


def test_broadband_source_off_step_uses_synthesized_component_duration():
    """Early stopping must not use the shorter parent Gaussian duration."""
    signal = np.zeros(801, dtype=np.float32)
    signal[120:641] = 1.0
    component = SimpleNamespace(
        temporal_profile=SimpleNamespace(signal=signal, time_step_duration=0.125e-15)
    )
    source = SimpleNamespace(_mode_components=(component,), temporal_profile=object())
    config = SimpleNamespace(time_step_duration=1e-15, time_steps_total=200)

    off_step = _gaussian_pulse_off_step(
        source, config, first_on_step=3, source_decay_by=1e-3
    )

    assert off_step == 84
