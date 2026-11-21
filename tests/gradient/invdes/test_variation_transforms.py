"""Analytical geometry references and independent finite-difference gradients."""

import numpy as np
import pytest
import torch

from core.invdes.variation import (
    aggregate_objectives,
    interval_coverage,
    sequential_objective,
    translate,
)

pytestmark = [pytest.mark.gradient, pytest.mark.numerical]


def test_zero_integer_fractional_offsets_and_domain_padding():
    values = torch.zeros((5, 4), dtype=torch.float64, requires_grad=True)
    impulse = values + torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    coords = (np.arange(5) * 0.1, np.arange(4) * 0.2)
    assert translate(impulse, coords, (0, 0)) is impulse
    integer = translate(impulse, coords, (0.1, 0.2))
    expected = torch.zeros_like(integer)
    expected[2, 2] = 1
    torch.testing.assert_close(integer, expected, rtol=0, atol=1e-15)
    fractional = translate(impulse, coords, (0.05, 0))
    expected.zero_()
    expected[1:3, 1] = 0.5
    torch.testing.assert_close(fractional, expected, rtol=0, atol=1e-15)
    assert not translate(impulse, coords, (10, 0)).any()


def test_nonuniform_coordinates_and_fractional_vertical_coverage():
    values = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    shifted = translate(values, ([0, 0.1, 0.4], [0, 0.2]), (0.05, 0))
    torch.testing.assert_close(
        shifted[:, 0], torch.tensor([0.0, 0.5, 1 / 6], dtype=torch.float64)
    )
    coverage = interval_coverage([0, 0.1, 0.3, 0.6], 0.05, 0.4, values)
    torch.testing.assert_close(
        coverage, torch.tensor([0.5, 1.0, 1 / 3], dtype=torch.float64)
    )


@pytest.mark.parametrize("mode", ["mean", "mean_std", "minimum", "smooth_min"])
def test_sequential_robust_gradient_matches_finite_difference(mode):
    coords = (np.arange(4) * 0.1, np.arange(3) * 0.2)
    design = (
        torch.linspace(0.1, 0.9, 12, dtype=torch.float64).reshape(4, 3).requires_grad_()
    )
    offsets = [(0.0, 0.0), (0.05, -0.1)]

    def objective(theta):
        materials = (translate(theta, coords, offset) for offset in offsets)
        return sequential_objective(
            materials,
            lambda eps: -(eps * torch.arange(1, 13, dtype=eps.dtype).reshape(4, 3))
            .square()
            .sum(),
            weights=[1, 3],
            aggregation=dict(mode=mode, penalty=0.2, temperature=0.3),
        )[0]

    result = objective(design)
    (grad,) = torch.autograd.grad(result, design)
    h = 1e-6
    for index in [(0, 0), (2, 1), (3, 2)]:
        plus, minus = design.detach().clone(), design.detach().clone()
        plus[index] += h
        minus[index] -= h
        fd = (objective(plus) - objective(minus)) / (2 * h)
        torch.testing.assert_close(grad[index], fd, rtol=2e-7, atol=2e-7)


def test_weighting_zero_variance_zero_weight_and_custom_aggregation():
    values = torch.tensor([2.0, 2.0], dtype=torch.float64, requires_grad=True)
    mean_std = aggregate_objectives(values, [1, 3], mode="mean_std")
    (grad,) = torch.autograd.grad(mean_std, values)
    torch.testing.assert_close(grad, torch.tensor([0.25, 0.75], dtype=torch.float64))
    assert aggregate_objectives(torch.tensor([-9.0, 2.0]), [0, 1], mode="minimum") == 2
    assert aggregate_objectives(values, mode=lambda v, w: (v * w).sum()) == 2


def test_mutable_solver_state_backward_completes_before_next_scenario():
    state = {"scale": 0}

    class Stateful(torch.autograd.Function):
        @staticmethod
        def forward(ctx, eps):
            ctx.save_for_backward(eps)
            return (eps * state["scale"]).sum()

        @staticmethod
        def backward(ctx, cotangent):
            (eps,) = ctx.saved_tensors
            return torch.ones_like(eps) * state["scale"] * cotangent

    design = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

    def evaluate(eps):
        state["scale"] += 1
        return Stateful.apply(eps)

    result, _ = sequential_objective([design, design], evaluate)
    result.backward()
    torch.testing.assert_close(design.grad, torch.tensor([1.5], dtype=torch.float64))
