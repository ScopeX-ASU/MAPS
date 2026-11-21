"""Regression tests for complex-permittivity Ceviche-Torch gradients."""

import numpy as np
import torch

from core.fdfd.fdfd import fdfd_ez
from core.invdes.models.base_optimization import (
    _blend_design_region,
    _optimization_epsilon_tensor,
)
from core.utils import energy_constraint_penalty
from thirdparty.ceviche.constants import C_0


def _loss_for_single_pixel(epsilon_value: complex) -> float:
    """Solve a small Ez problem after perturbing one permittivity pixel."""
    shape = (5, 5)
    pixel = (3, 3)
    omega = 2 * np.pi * C_0 / 1.55e-6
    source = torch.zeros(shape, dtype=torch.complex128)
    source[2, 1] = 1.0 + 0.3j

    epsilon = torch.full(shape, 2.25 + 0.08j, dtype=torch.complex128)
    epsilon[pixel] = epsilon_value
    simulation = fdfd_ez(
        omega,
        0.2e-6,
        epsilon,
        [1, 1],
        sym_precond=False,
    )
    simulation.set_cache_mode(True)
    _, _, ez = simulation.solve(source, slice_name="test", mode=1, temp=0)
    return float(ez[pixel].abs().square())


def test_complex_epsilon_gradient_matches_real_and_imaginary_finite_differences():
    """The complex autograd gradient has two physically correct real channels."""
    shape = (5, 5)
    pixel = (3, 3)
    omega = 2 * np.pi * C_0 / 1.55e-6
    source = torch.zeros(shape, dtype=torch.complex128)
    source[2, 1] = 1.0 + 0.3j

    epsilon = torch.full(
        shape,
        2.25 + 0.08j,
        dtype=torch.complex128,
        requires_grad=True,
    )
    simulation = fdfd_ez(
        omega,
        0.2e-6,
        epsilon,
        [1, 1],
        sym_precond=False,
    )
    simulation.set_cache_mode(True)
    _, _, ez = simulation.solve(source, slice_name="test", mode=1, temp=0)
    ez[pixel].abs().square().backward()

    base = 2.25 + 0.08j
    step = 1e-6
    finite_difference_real = (
        _loss_for_single_pixel(base + step) - _loss_for_single_pixel(base - step)
    ) / (2 * step)
    finite_difference_imag = (
        _loss_for_single_pixel(base + 1j * step)
        - _loss_for_single_pixel(base - 1j * step)
    ) / (2 * step)

    assert torch.is_complex(epsilon.grad)
    assert np.isclose(
        epsilon.grad[pixel].real.item(),
        finite_difference_real,
        rtol=2e-5,
        atol=1e-16,
    )
    assert np.isclose(
        epsilon.grad[pixel].imag.item(),
        finite_difference_imag,
        rtol=2e-5,
        atol=1e-16,
    )


def test_complex_material_fraction_and_energy_penalty_are_differentiable():
    """A complex endpoint mix remains differentiable through the energy penalty."""
    rho = torch.tensor(
        [[0.0, 0.25], [0.5, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    eps_clad = 2.1 + 0.01j
    eps_core = 3.2 + 0.18j
    epsilon = eps_clad + rho * (eps_core - eps_clad)
    field = torch.ones((2, 2), dtype=torch.complex128)

    penalty = energy_constraint_penalty(
        field,
        epsilon,
        eps_core=eps_core,
        eps_clad=eps_clad,
    )
    penalty.backward()

    assert torch.isfinite(penalty)
    assert torch.isfinite(rho.grad).all()


def test_complex_design_region_is_not_cast_to_real_during_blending():
    """A real background map is promoted before inserting complex PCM values."""
    device = type("Device", (), {})()
    device.epsilon_map = np.ones((2, 2), dtype=np.float32)
    device._epsilon_inputs_are_complex = lambda: True

    epsilon_map = _optimization_epsilon_tensor(device, torch.device("cpu"))
    _blend_design_region(
        epsilon_map,
        (slice(0, 2), slice(0, 2)),
        torch.full((2, 2), 2.0 + 0.15j, dtype=torch.complex128),
    )

    assert epsilon_map.dtype == torch.complex64
    assert torch.allclose(
        epsilon_map.imag,
        torch.full((2, 2), 0.15, dtype=torch.float32),
    )
