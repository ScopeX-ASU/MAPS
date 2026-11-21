"""Contracts for stable Chebyshev broadband profile sources."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")
pytest.importorskip("jax")

from thirdparty.fdtdx.objects.sources.chebyshev import (  # noqa: E402
    align_sampled_profiles,
    chebyshev_coefficients,
    chebyshev_sample_frequencies,
    chebyshev_temporal_basis,
)
from thirdparty.fdtdx.objects.sources.normalization import (
    discrete_source_spectrum,
)  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.fdtdx]


@pytest.mark.parametrize("num_freqs", [1, 2, 3, 5])
def test_chebyshev_nodes_do_not_depend_on_objective_spacing(num_freqs):
    """Sampling is determined solely by the pulse band and interpolation order."""
    nodes = chebyshev_sample_frequencies(
        center_frequency=200e12,
        spectral_width=10e12,
        num_freqs=num_freqs,
    )
    assert nodes.shape == (num_freqs,)
    assert np.all(np.diff(nodes) > 0) if num_freqs > 1 else nodes[0] == 200e12
    assert np.all(nodes >= 185e12)
    assert np.all(nodes <= 215e12)


@pytest.mark.parametrize("num_freqs", [1, 2, 3, 5])
def test_chebyshev_transform_reconstructs_smooth_profile_at_nodes(num_freqs):
    """The DCT transform reproduces every sampled spatial profile."""
    if num_freqs == 1:
        nodes = np.asarray([0.0])
    else:
        angles = np.pi * (2 * np.arange(num_freqs) + 1) / (2 * num_freqs)
        nodes = np.cos(angles)[::-1]
    samples = np.stack((1 + 0.2 * nodes + 0.05 * nodes**2, -0.3 + 0.1j * nodes), axis=1)
    coefficients = chebyshev_coefficients(samples)
    reconstructed = np.stack(
        [np.polynomial.chebyshev.chebval(node, coefficients) for node in nodes]
    )
    np.testing.assert_allclose(reconstructed, samples, rtol=1e-12, atol=1e-12)


def test_profile_phase_alignment_removes_arbitrary_solver_phase():
    """Mode-solver phase choices must not become artificial frequency variation."""
    base = np.asarray([1.0, 2.0j, -0.5])
    phases = np.exp(1j * np.asarray([-1.2, 0.0, 0.7]))
    samples = phases[:, None] * base[None]
    aligned, _ = align_sampled_profiles(samples)
    np.testing.assert_allclose(
        aligned, np.broadcast_to(base, aligned.shape), atol=1e-12
    )


def test_temporal_basis_multiplies_base_spectrum_without_long_isolation_solve():
    """Temporal terms implement Chebyshev multipliers on the pulse spectrum."""
    center = fdtdx.WaveCharacter(frequency=200e12)
    profile = fdtdx.GaussianPulseProfile(
        center_wave=center,
        spectral_width=fdtdx.WaveCharacter(frequency=10e12),
    )
    dt = 1e-16
    count = 8192
    basis = chebyshev_temporal_basis(
        temporal_profile=profile,
        time_step_duration=dt,
        num_time_steps=count,
        num_terms=3,
    )
    assert basis.shape == (3, count)
    assert np.max(np.abs(basis[0])) <= 1.01
    # All terms retain the compact Gaussian pulse support instead of extending
    # to the inverse spacing of arbitrary objective frequencies.
    tail = np.max(np.abs(basis[:, count // 2 :]))
    assert tail < 1e-5

    time = np.arange(count) * dt
    for normalized_frequency in (-0.9, -0.3, 0.2, 0.8):
        frequency = 200e12 + normalized_frequency * 15e12
        response = np.exp(1j * 2 * np.pi * frequency * time) @ basis.T
        relative_response = response / response[0]
        expected = np.asarray(
            [
                np.polynomial.chebyshev.chebval(
                    normalized_frequency, [0.0] * order + [1.0]
                )
                for order in range(3)
            ]
        )
        np.testing.assert_allclose(relative_response, expected, atol=2e-5)


@pytest.mark.parametrize("scaling_mode, interval", [("continuous", 1), ("pulse", 8)])
def test_discrete_source_spectrum_matches_detector_sampling(scaling_mode, interval):
    """Normalization uses the detector's exact time indices and complex DFT."""
    center = fdtdx.WaveCharacter(frequency=200e12)
    profile = fdtdx.GaussianPulseProfile(
        center_wave=center,
        spectral_width=fdtdx.WaveCharacter(frequency=10e12),
    )
    source = SimpleNamespace(
        temporal_profile=profile,
        wave_character=center,
        static_amplitude_factor=1.0,
        amplitude=1.0,
    )
    count = 4096
    dt = 1e-16
    config = SimpleNamespace(time_steps_total=count, time_step_duration=dt)
    detector = SimpleNamespace(
        switch=fdtdx.OnOffSwitch(interval=interval),
        scaling_mode=scaling_mode,
        _dft_stride=1,
    )
    waves = (
        fdtdx.WaveCharacter(frequency=195e12),
        fdtdx.WaveCharacter(frequency=205e12),
    )
    actual = discrete_source_spectrum(
        source=source,
        wave_characters=waves,
        config=config,
        detector=detector,
    )

    indices = np.arange(0, count, interval)
    times = indices * dt
    signal = np.asarray(
        profile.get_amplitude(
            time=jnp.asarray(times),
            period=center.get_period(),
            phase_shift=center.phase_shift,
        )
    )
    scale = 2.0 / len(indices) if scaling_mode == "continuous" else 1.0
    expected = scale * np.asarray(
        [
            np.sum(signal * np.exp(2j * np.pi * wave.get_frequency() * times))
            for wave in waves
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_discrete_source_spectrum_uses_profile_carrier_for_distributed_source():
    """Adjoint sources without a concrete wave_character use center_wave."""
    center = fdtdx.WaveCharacter(frequency=200e12)
    profile = fdtdx.GaussianPulseProfile(
        center_wave=center,
        spectral_width=fdtdx.WaveCharacter(frequency=10e12),
    )
    source = SimpleNamespace(temporal_profile=profile, wave_character=object())
    config = SimpleNamespace(time_steps_total=256, time_step_duration=1e-16)
    spectrum = discrete_source_spectrum(
        source=source,
        wave_characters=(center,),
        config=config,
    )
    assert spectrum.shape == (1,)
    assert np.isfinite(spectrum).all()
