"""Unit tests for the complex-permittivity to FDTDX conductivity bridge."""

import numpy as np
import pytest
import torch

fdtdx = pytest.importorskip("fdtdx")
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from core.fdtd.fdtd import _split_complex_permittivity_for_fdtdx
from core.fdtd.solver import _permittivity_gradient_from_phasors


class _Wave:
    """Minimal wave-character test double."""

    def get_frequency(self):
        return 2.0e14


def test_complex_permittivity_matches_fdtdx_material_constructor():
    """The spatial conversion agrees with FDTDX's scalar material conversion."""
    wavelength_m = 1.55e-6
    dt = 2.0e-16
    courant = 0.99
    epsilon = 5.6**2 - 0.75**2 + 1j * (2.0 * 5.6 * 0.75)
    epsilon_map = torch.full((2, 3, 1), epsilon, dtype=torch.complex64)

    real_map, sigma_map = _split_complex_permittivity_for_fdtdx(
        epsilon_map,
        wavelength_m=wavelength_m,
        time_step_duration=dt,
        courant_number=courant,
    )

    material = fdtdx.Material.from_complex_permittivity(
        epsilon,
        wavelength=wavelength_m,
    )
    expected_sigma = material.electric_conductivity[0] * (
        fdtdx.constants.c * dt / courant
    )

    assert real_map.dtype == torch.float32
    torch.testing.assert_close(real_map, torch.full_like(real_map, epsilon.real))
    torch.testing.assert_close(
        sigma_map,
        torch.full_like(sigma_map, expected_sigma),
        rtol=1e-6,
        atol=1e-12,
    )


def test_explicit_conductivity_is_combined_before_fdtdx_scaling():
    """Existing physical conductivity is retained alongside PCM loss."""
    wavelength_m = 1.55e-6
    dt = 2.0e-16
    courant = 0.99
    epsilon = torch.full((2, 2, 1), 2.0 + 0.1j, dtype=torch.complex64)
    explicit_sigma = torch.full((2, 2, 1), 7.0, dtype=torch.float32)

    _, sigma_map = _split_complex_permittivity_for_fdtdx(
        epsilon,
        wavelength_m=wavelength_m,
        time_step_duration=dt,
        courant_number=courant,
        electrical_conductivity_map=explicit_sigma,
    )

    expected_physical_sigma = (
        2.0 * np.pi * fdtdx.constants.c / wavelength_m
    ) * fdtdx.constants.eps0 * 0.1 + 7.0e6
    expected = expected_physical_sigma * fdtdx.constants.c * dt / courant
    torch.testing.assert_close(
        sigma_map,
        torch.full_like(sigma_map, expected),
        rtol=1e-6,
        atol=1e-12,
    )


def test_complex_phasor_sensitivity_exposes_both_real_channels():
    """Complex FDTDX sensitivity follows the standard Torch convention."""
    forward = torch.tensor(
        [[[[[1.0 + 2.0j]], [[2.0 - 1.0j]], [[0.5 + 0.25j]]]]],
        dtype=torch.complex64,
    )
    adjoint = torch.tensor(
        [[[[[0.25 - 0.5j]], [[1.0 + 0.25j]], [[-0.5 + 0.75j]]]]],
        dtype=torch.complex64,
    )
    gradient = _permittivity_gradient_from_phasors(
        forward_phasor=forward,
        adjoint_phasor=adjoint,
        wave_characters=(_Wave(),),
        complex_permittivity=True,
    )

    omega = 2.0 * np.pi * 2.0e14
    raw_sensitivity = (
        -2.0
        * (-(omega**2) * fdtdx.constants.mu0 * fdtdx.constants.eps0)
        * (forward * adjoint).sum(1)[0]
    )
    torch.testing.assert_close(gradient, raw_sensitivity.conj())


def test_maps_place_objects_accepts_conductivity_without_conductive_material():
    """The MAPS placement extension creates the conductivity array on demand."""
    config = fdtdx.SimulationConfig(
        time=1e-15,
        grid=fdtdx.UniformGrid(spacing=200e-9),
        dtype=jnp.float32,
        courant_factor=0.8,
        backend="cpu",
    )
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(2.4e-6, 2.4e-6, 2.4e-6),
        material=fdtdx.Material(permittivity=2.0),
    )
    _, arrays, _, _, _ = fdtdx.place_objects(
        object_list=[volume],
        config=config,
        constraints=[],
        key=jax.random.PRNGKey(1),
        override_electric_conductivity=jnp.full((1, 12, 12, 12), 0.25),
    )

    assert arrays.electric_conductivity is not None
    assert arrays.electric_conductivity.shape == (1, 12, 12, 12)
    assert float(arrays.electric_conductivity[0, 0, 0, 0]) == pytest.approx(0.25)
