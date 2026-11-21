"""Frequency-dependent adjoint current and arbitrary VJP weight contracts."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from core.fdtd import solver
from thirdparty.fdtdx import FactoredChebyshevCurrentSource, fdtdx

pytestmark = [pytest.mark.unit, pytest.mark.fdtdx, pytest.mark.gradient]


def _profile_fixture(span=10e12):
    frequencies = np.linspace(200e12 - span, 200e12 + span, 5)
    waves = tuple(fdtdx.WaveCharacter(frequency=float(f)) for f in frequencies)
    center = fdtdx.WaveCharacter(frequency=200e12)
    profile = fdtdx.GaussianPulseProfile(
        center_wave=center, spectral_width=fdtdx.WaveCharacter(frequency=10e12)
    )
    x = np.linspace(-1.0, 1.0, 5)
    # Unit-norm smooth physical family with measurable shape variation.
    family = np.zeros((5, 6, 2, 1, 1), dtype=np.complex64)
    family[:, 1, 0, 0, 0] = np.cos(0.1 * x)
    family[:, 5, 1, 0, 0] = 1j * np.sin(0.1 * x)
    weights = np.asarray([1, -1, 1j, -2j, 0.3 + 0.8j], dtype=np.complex64)
    currents = torch.from_numpy(weights[:, None, None, None, None] * family)
    cotangents = (1j * currents).conj()
    cotangents[:, 3:] *= -1
    return frequencies, waves, profile, currents, cotangents


@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_factorization_preserves_arbitrary_objective_phase_and_sign(order):
    frequencies, _, _, currents, cotangents = _profile_fixture()
    electric, magnetic, weights, diagnostics = (
        solver._factor_frequency_dependent_current_profiles_torch(
            cotangents,
            solver._COMPONENTS,
            frequencies=frequencies,
            center_frequency=200e12,
            order=order,
        )
    )
    reconstructed = (
        torch.cat((electric, magnetic), dim=1) * weights[:, None, None, None, None]
    )
    torch.testing.assert_close(reconstructed, currents, rtol=3e-6, atol=1e-7)
    assert diagnostics["adjacent_overlap_min"] > 0.99
    if order == 1:
        assert diagnostics["fit_relative_l2_max"] > 0.05
    if order >= 3:
        assert diagnostics["fit_relative_l2_max"] < 1e-4


@pytest.mark.parametrize("order", [3, 5])
def test_sampled_source_dtft_reproduces_frequency_dependent_e_and_h_currents(order):
    frequencies, waves, profile, currents, cotangents = _profile_fixture()
    electric, magnetic, weights, _ = (
        solver._factor_frequency_dependent_current_profiles_torch(
            cotangents,
            solver._COMPONENTS,
            frequencies=frequencies,
            center_frequency=200e12,
            order=order,
        )
    )
    config = fdtdx.SimulationConfig(
        time=0.4e-12,
        grid=fdtdx.UniformGrid(spacing=60e-9),
        dtype=jnp.float32,
        backend="cpu",
    )
    source = FactoredChebyshevCurrentSource(
        name="source",
        partial_grid_shape=(2, 1, 1),
        wave_characters=waves[:order],
        sample_wave_characters=waves,
        polynomial_half_width=10e12,
        temporal_profile=profile,
        electric_profiles=jnp.asarray(electric.numpy()),
        magnetic_profiles=jnp.asarray(magnetic.numpy()),
    )
    source = source.place_on_grid(
        grid_slice_tuple=((0, 2), (0, 1), (0, 1)),
        config=config,
        key=jax.random.PRNGKey(0),
    )
    source = source.apply(
        key=jax.random.PRNGKey(0),
        inv_permittivities=jnp.ones((3, 2, 1, 1)),
        inv_permeabilities=1.0,
    )
    off_step = source.active_until_step(1e-3)
    assert 0 < off_step < config.time_steps_total
    time = np.arange(config.time_steps_total) * config.time_step_duration
    dft = np.exp(2j * np.pi * frequencies[:, None] * time)
    spectrum = fdtdx.discrete_source_spectrum(
        source=source,
        wave_characters=waves,
        config=config,
    )
    values = []
    for spatial in (source.electric_profiles, source.magnetic_profiles):
        signal = np.asarray(
            jax.vmap(lambda step: source._synthesize(spatial, step))(
                jnp.arange(config.time_steps_total)
            )
        ).reshape(config.time_steps_total, -1)
        phasors = (2.0 / config.time_steps_total) * (dft @ signal) / spectrum[:, None]
        values.append(phasors.reshape(5, 3, 2, 1, 1))
    reconstructed = (
        np.concatenate(values, axis=1) * weights.numpy()[:, None, None, None, None]
    )
    # Three terms approximate the sinusoidal family rather than interpolate
    # all five samples. Measure the full current norm so small transverse
    # components do not impose an unrelated relative-error requirement.
    relative_error = np.linalg.norm(reconstructed - currents.numpy()) / np.linalg.norm(
        currents.numpy()
    )
    assert relative_error < 1e-4
    if order == 5:
        np.testing.assert_allclose(
            reconstructed, currents.numpy(), rtol=3e-4, atol=3e-5
        )


def test_planner_uses_order_one_for_invariant_profiles_and_rejects_orthogonal_profiles(
    monkeypatch,
):
    frequencies, waves, profile, _, cotangents = _profile_fixture()
    monitor = SimpleNamespace(wave_characters=waves, components=solver._COMPONENTS)
    config = SimpleNamespace(time_steps_total=4096, time_step_duration=1e-16)
    groups = {
        "port": {
            "freq_indices": torch.arange(5),
            "spatial_indices": torch.tensor([[0, 0, 0], [1, 0, 0]]),
        }
    }
    monkeypatch.setattr(solver, "_BROADBAND_ADJOINT_METHOD", "auto")
    monkeypatch.setattr(solver, "_BROADBAND_ADJOINT_PROFILE_ORDER", 0)
    invariant = cotangents[:, :].clone()
    invariant[:, 5] = 0
    invariant[:, 1, 0, 0, 0] = torch.tensor([1, -1, 1j, 2j, 0.5])
    plan = solver._plan_broadband_adjoint_groups(
        cotangents=invariant,
        groups=groups,
        monitor=monitor,
        source_template=SimpleNamespace(temporal_profile=profile),
        config=config,
    )
    assert plan == {"port": 1}
    orthogonal = torch.zeros_like(cotangents)
    for index in range(5):
        orthogonal[index, index, index % 2, 0, 0] = 1
    assert (
        solver._plan_broadband_adjoint_groups(
            cotangents=orthogonal,
            groups=groups,
            monitor=monitor,
            source_template=SimpleNamespace(temporal_profile=profile),
            config=config,
        )
        is None
    )


def test_grouping_keeps_small_nonzero_currents_and_interpolation_halo():
    """Grouping must not truncate a physical current using an absolute scale."""
    cotangent = torch.zeros((3, 6, 5, 3, 3), dtype=torch.complex64)
    cotangent[:, 1, 1:4, 1, 1] = 1e-15
    groups = solver.build_adjoint_groups_from_monitor_info(
        cotangent,
        [
            {
                "group_id": "port",
                "monitor_slice": (slice(2, 3), slice(1, 2), slice(1, 2)),
                "freq_indices": (0, 1, 2),
            }
        ],
    )
    extracted = solver.extract_group_grad_compact_torch(cotangent, groups["port"])
    torch.testing.assert_close(extracted, cotangent, rtol=0, atol=0)


def test_sparse_current_blocks_accept_strided_conjugated_torch_cotangents():
    """A cropped Torch view must preserve all complex values across DLPack."""
    cotangent = torch.arange(750).reshape(1, 6, 5, 5, 5).to(torch.complex64)
    cotangent = (cotangent * (1 + 2j)).conj()
    entries = [
        {"monitor_slice": (slice(2, 3), slice(2, 3), slice(2, 3)), "freq_indices": (0,)}
    ]
    (block,) = solver._build_sparse_current_blocks(
        cotangents=cotangent,
        components=solver._COMPONENTS,
        adjoint_group_info=entries,
        global_freq_idx=0,
    )
    expected_e, expected_m = solver._cotangents_to_current_profiles(
        cotangent[:, :, 1:4, 1:4, 1:4].resolve_conj().numpy(),
        solver._COMPONENTS,
    )
    np.testing.assert_array_equal(
        np.asarray(block.electric_profiles), np.asarray(expected_e)
    )
    np.testing.assert_array_equal(
        np.asarray(block.magnetic_profiles), np.asarray(expected_m)
    )


def test_source_builder_preserves_current_scale_and_all_complex_vjp_weights():
    """A common source gauge avoids changing decay-detector dynamic range."""
    _, waves, profile, currents, cotangents = _profile_fixture()
    source, weights, _ = solver._build_factored_chebyshev_adjoint_source(
        adj_monitor=SimpleNamespace(
            wave_characters=waves,
            components=solver._COMPONENTS,
            partial_grid_shape=(2, 1, 1),
            name="monitor",
        ),
        source_template=SimpleNamespace(temporal_profile=profile),
        cotangents=cotangents,
        order=3,
    )
    samples = np.concatenate(
        (np.asarray(source.electric_profiles), np.asarray(source.magnetic_profiles)),
        axis=1,
    )
    np.testing.assert_allclose(
        samples * weights.numpy()[:, None, None, None, None],
        currents.numpy(),
        rtol=3e-6,
        atol=1e-7,
    )
    anchor = int(currents.reshape(5, -1).norm(dim=1).argmax())
    np.testing.assert_allclose(
        samples[anchor], currents.numpy()[anchor], rtol=3e-6, atol=1e-7
    )
    np.testing.assert_allclose(weights[anchor].numpy(), 1, atol=1e-7)


def test_temporal_diagnostics_detect_unsafe_amplification():
    """Closely spaced samples must not hide large temporal derivative amplitudes."""
    from thirdparty.fdtdx.objects.sources.chebyshev import (
        chebyshev_temporal_diagnostics,
    )

    _, _, profile, _, _ = _profile_fixture()
    diagnostics = chebyshev_temporal_diagnostics(
        temporal_profile=profile,
        frequencies=np.asarray([200e12 - 1e10, 200e12 + 1e10]),
        coefficient_norms=np.ones(3),
        polynomial_half_width=1e10,
        time_step_duration=1e-16,
        num_time_steps=4096,
    )
    assert diagnostics["waveform_peak_norm_bound"] > 100


def test_stopping_condition_uses_the_composite_source_envelope():
    """A Gaussian carrier's cutoff cannot truncate derivative-basis tails."""
    from thirdparty.fdtdx.fdtd.stop_condition import _gaussian_pulse_off_step

    source = SimpleNamespace(active_until_step=lambda relative_cutoff: 900)
    config = SimpleNamespace(time_steps_total=1000)
    assert _gaussian_pulse_off_step(source, config, 0, 1e-3) == 900


def test_adjoint_field_metrics_are_invariant_to_small_static_current_scaling():
    """Complex gain diagnostics must remain meaningful for tiny adjoint fields."""
    def _metrics(value: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        value = np.asarray(value).reshape(-1).astype(np.complex128)
        reference = np.asarray(reference).reshape(-1).astype(np.complex128)
        common_scale = np.max(np.abs(reference))
        if common_scale > 0:
            value = value / common_scale
            reference = reference / common_scale
        ref_norm = np.linalg.norm(reference)
        value_norm = np.linalg.norm(value)
        difference = value - reference
        coherence = np.abs(np.vdot(reference, value)) / (ref_norm * value_norm + 1e-30)
        scale = np.vdot(value, reference) / (np.vdot(value, value) + 1e-30)
        complex_gain = np.vdot(reference, value) / (
            np.vdot(reference, reference) + 1e-30
        )
        aligned = scale * value
        return {
            "relative_l2": float(np.linalg.norm(difference) / (ref_norm + 1e-30)),
            "complex_coherence": float(coherence),
            "aligned_relative_l2": float(
                np.linalg.norm(aligned - reference) / (ref_norm + 1e-30)
            ),
            "max_normalized": float(
                np.max(np.abs(difference)) / (np.max(np.abs(reference)) + 1e-30)
            ),
            "norm_ratio": float(value_norm / (ref_norm + 1e-30)),
            "complex_gain_real": float(np.real(complex_gain)),
            "complex_gain_imag": float(np.imag(complex_gain)),
            "complex_gain_magnitude_error": float(np.abs(complex_gain) - 1.0),
            "complex_gain_phase_rad": float(np.angle(complex_gain)),
        }

    reference = np.asarray([1, 2j, -3], dtype=complex)
    gain = 1.02 * np.exp(-0.03j)
    normal = _metrics(gain * reference, reference)
    tiny = _metrics(1e-20 * gain * reference, 1e-20 * reference)
    np.testing.assert_allclose(tiny["complex_gain_magnitude_error"], 0.02, atol=1e-12)
    np.testing.assert_allclose(tiny["complex_gain_phase_rad"], -0.03, atol=1e-12)
    np.testing.assert_allclose(tiny["aligned_relative_l2"], 0, atol=1e-12)
    for key in normal:
        np.testing.assert_allclose(tiny[key], normal[key], atol=1e-12)


@pytest.mark.parametrize("coverage", ["missing", "duplicate"])
def test_planner_falls_back_when_metadata_does_not_partition_current_support(
    monkeypatch, coverage
):
    """No active current may be omitted or counted twice by port grouping."""
    _, waves, profile, _, cotangents = _profile_fixture()
    monitor = SimpleNamespace(wave_characters=waves, components=solver._COMPONENTS)
    group = {
        "freq_indices": torch.arange(5),
        "spatial_indices": torch.tensor([[0, 0, 0], [1, 0, 0]]),
    }
    groups = {"first": group}
    if coverage == "duplicate":
        groups["second"] = group
    else:
        group["spatial_indices"] = torch.tensor([[0, 0, 0]])
    monkeypatch.setattr(solver, "_BROADBAND_ADJOINT_METHOD", "auto")
    assert (
        solver._plan_broadband_adjoint_groups(
            cotangents=cotangents,
            groups=groups,
            monitor=monitor,
            source_template=SimpleNamespace(temporal_profile=profile),
            config=SimpleNamespace(time_steps_total=4096, time_step_duration=1e-16),
        )
        is None
    )
