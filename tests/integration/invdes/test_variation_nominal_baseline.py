"""Pre-change baseline for BaseOptimization's nominal public forward contract."""

import pytest
import torch
from torch import nn

from core.invdes.models.base_optimization import BaseOptimization

pytestmark = pytest.mark.integration


class SmallOptimization(BaseOptimization):
    """Exercise the real forward lifecycle with an analytical objective."""

    def __init__(self):
        nn.Module.__init__(self)
        self.design = nn.Parameter(torch.tensor([0.2, 0.7], dtype=torch.float64))
        self.objective = type(
            "Objective", (), {"breakdown": {}, "runtime_thermal_states": {}}
        )()
        self.build_count = 0

    def build_device(self, sharpness=1, ls_knots=None):
        self.build_count += 1
        eps = self.design * sharpness
        return eps, {"region": eps}, eps, {"region": eps}, {}

    def _build_runtime_thermal_material_maps_from_regions(self, regions):
        return None

    def _snapshot_runtime_thermal_states(self):
        pass

    def objective_layer(self, maps, custom_source=None):
        value = maps[0].square().sum()
        self.objective.breakdown = {"power": {"weight": 1, "value": value}}
        return value


def test_nominal_forward_value_gradient_and_results():
    model = SmallOptimization()
    result = model(sharpness=2)
    assert set(result) == {"obj", "breakdown", "region", "eps_map"}
    torch.testing.assert_close(
        result["eps_map"], torch.tensor([0.4, 1.4], dtype=torch.float64), rtol=0, atol=0
    )
    torch.testing.assert_close(result["obj"], torch.tensor(2.12, dtype=torch.float64))
    (-result["obj"]).backward()
    torch.testing.assert_close(
        model.design.grad, torch.tensor([-1.6, -5.6], dtype=torch.float64)
    )
    assert model.build_count == 1


def test_disabled_forward_and_checkpoint_schema_are_unchanged():
    model = SmallOptimization()
    before = model.state_dict()
    model.configure_variation({"enabled": False})
    after = model.state_dict()
    assert set(before) == set(after)
    nominal = model(sharpness=2)
    (-nominal["obj"]).backward()
    assert set(nominal) == {"obj", "breakdown", "region", "eps_map"}
    assert model.build_count == 1
    torch.testing.assert_close(
        model.design.grad, torch.tensor([-1.6, -5.6], dtype=torch.float64)
    )


def test_robust_forward_shared_design_and_checkpoint_resume():
    from types import SimpleNamespace

    settings = dict(
        enabled=True,
        seed=7,
        num_samples=2,
        distributions=[dict(params=["offset"], type="uniform", loc=-0.1, scale=0.2)],
    )

    def prepare():
        model = SmallOptimization()
        model.device = SimpleNamespace(active_region_cfgs={})
        model.configure_variation(
            settings,
            materializer=lambda nominal, scenario: nominal[0] + scenario["offset"],
        )
        return model

    model = prepare()
    model.variation.step = 4
    results = model(sharpness=2)
    assert model.build_count == 1
    expected = torch.stack(
        [
            (model.design * 2 + s["offset"]).square().sum()
            for s in results["scenarios"]["scenarios"]
        ]
    ).mean()
    (expected_grad,) = torch.autograd.grad(-expected, model.design)
    (-results["obj"]).backward()
    torch.testing.assert_close(results["obj"], expected)
    torch.testing.assert_close(model.design.grad, expected_grad)
    checkpoint = model.state_dict()
    restored = prepare()
    restored.load_state_dict(checkpoint)
    assert restored.variation.step == 4
    model.variation.step = restored.variation.step = 5
    assert restored.variation.sampler.get(5) == model.variation.sampler.get(5)


class PlotOptimization(SmallOptimization):
    """Analytical 3-D objective using the real runtime material plotting lifecycle."""

    def __init__(self, clone_runtime=True):
        super().__init__()
        from types import SimpleNamespace

        self.design = nn.Parameter(
            torch.linspace(0.2, 0.8, 24, dtype=torch.float64).reshape(2, 3, 4)
        )
        self.device = SimpleNamespace(active_region_cfgs={})
        self.clone_runtime = clone_runtime
        self._thermal_states = {}
        self._thermal_state = None

    def objective_layer(self, maps, custom_source=None):
        eps = maps[0].clone() if self.clone_runtime else maps[0]
        self.objective.runtime_thermal_states = {
            300: {"mode": "legacy_temp", "eps": eps}
        }
        value = eps.square().sum()
        self.objective.breakdown = {"power": {"weight": 1, "value": value}}
        return value

    def _snapshot_runtime_thermal_states(self):
        BaseOptimization._snapshot_runtime_thermal_states(self)


@pytest.mark.parametrize("clone_runtime", [False, True])
def test_zero_variation_eps_gradient_matches_nominal_three_view_planes(clone_runtime):
    nominal = PlotOptimization(clone_runtime)
    result = nominal(sharpness=2)
    (-result["obj"]).backward()
    expected = nominal._resolve_plot_eps_map(("in", 1.31, "Hz2", 300)).grad
    assert expected is not None

    varied = PlotOptimization(clone_runtime)
    varied.configure_variation(
        dict(enabled=True, scenarios=[{}]), materializer=lambda maps, scenario: maps[0]
    )
    result = varied(sharpness=2)
    runtime_eps = varied._resolve_plot_eps_map(("in", 1.31, "Hz2", 300))
    assert (
        runtime_eps.grad is None
    )  # Diagnostic gradients belong to outer loss backward.
    (-result["obj"]).backward()
    assert runtime_eps.grad is not None
    for plane in [
        (slice(None), slice(None), 1),
        (slice(None), 1, slice(None)),
        (1, slice(None), slice(None)),
    ]:
        torch.testing.assert_close(
            runtime_eps.grad[plane], expected[plane], rtol=0, atol=0
        )
    torch.testing.assert_close(varied.design.grad, nominal.design.grad, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["mean", "mean_std", "minimum", "smooth_min"])
def test_last_scenario_eps_gradient_has_outer_loss_sign_and_aggregation_weight(mode):
    from core.invdes.variation import aggregate_objectives

    model = PlotOptimization()
    model.configure_variation(
        dict(
            enabled=True,
            scenarios=[{"offset": -0.1}, {"offset": 0.1}],
            weights=[1, 3],
            aggregation=dict(mode=mode, penalty=0.2, temperature=0.3),
        ),
        materializer=lambda nominal, scenario: nominal[0] + scenario["offset"],
    )
    result = model(sharpness=2)
    runtime_eps = model._resolve_plot_eps_map(("in", 1.31, "Hz2", 300))
    values = result["scenario_objectives"].clone().requires_grad_()
    objective = aggregate_objectives(
        values, [1, 3], mode=mode, penalty=0.2, temperature=0.3
    )
    (coefficients,) = torch.autograd.grad(-objective, values)
    (-result["obj"]).backward()
    torch.testing.assert_close(
        runtime_eps.grad,
        coefficients[-1] * 2 * runtime_eps.detach(),
        rtol=1e-12,
        atol=1e-12,
    )
