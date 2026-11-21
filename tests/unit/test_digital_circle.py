from types import SimpleNamespace

import pytest
import torch

from core.invdes.models.layers.parametrization.circle import (
    CircleParameterization,
    Circles,
    _circle_coverage,
)


def _circle(max_radius_um=None, small_circle_policy=None):
    region = {
        "center": (0.0, 0.0),
        "size": (1.0, 1.0),
        "eps": 12.0,
        "eps_bg": 1.0,
    }
    geometry_cfgs = {
        "batch_dims": (1, 1),
        "material_optimization": True,
        "border": 0.0,
        "coverage_samples": 16,
        "gradient_step_um": 0.025,
    }
    if max_radius_um is not None:
        geometry_cfgs["max_radius_um"] = max_radius_um
    if small_circle_policy is not None:
        geometry_cfgs["small_circle_policy"] = small_circle_policy
    circles = Circles(
        sim_cfg={"resolution": 40},
        geometry_cfgs=geometry_cfgs,
        design_region_cfg=region,
        operation_device=torch.device("cpu"),
    )
    with torch.no_grad():
        circles.centers.copy_(torch.tensor([[[0.07, -0.08]]]))
        circles.sizes[..., 0].fill_(0.16)
        circles.material_rho.fill_(0.7)
    circles.double()
    return circles


def test_circle_coverage_uses_cartesian_y_up_rows():
    coverage = _circle_coverage(
        torch.tensor([0.0, 0.25]),
        torch.tensor(0.1),
        region_center=(0.0, 0.0),
        region_size=(1.0, 1.0),
        grid_step=0.1,
        shape=(10, 10),
        samples_per_pixel=8,
    )
    # The circle is above the region center. In y_up image layout it must
    # occupy rows near the top (small row indices).
    row_mass = coverage.sum(dim=0)
    assert row_mass[:4].sum() > row_mass[6:].sum()


def test_circle_material_gradient_matches_finite_difference():
    circles = _circle()
    circles.set_optimization_stage("material")
    weights = torch.arange(1600, dtype=torch.float32).reshape(40, 40) / 1600.0
    loss = (circles() * weights).sum()
    loss.backward()
    analytic = circles.material_rho.grad[0, 0].detach()

    delta = 1e-3
    with torch.no_grad():
        circles.material_rho[0, 0] += delta
        plus = (circles() * weights).sum().detach()
        circles.material_rho[0, 0] -= 2.0 * delta
        minus = (circles() * weights).sum().detach()
        circles.material_rho[0, 0] += delta
    finite_difference = (plus - minus) / (2.0 * delta)
    assert analytic == pytest.approx(finite_difference.item(), rel=1e-6, abs=1e-3)


@pytest.mark.parametrize("parameter", ["radius", "center_x", "center_y"])
def test_circle_shape_gradients_match_finite_difference(parameter):
    circles = _circle()
    circles.set_optimization_stage("joint", optimize_centers=True)
    weights = torch.arange(1600, dtype=torch.float32).reshape(40, 40) / 1600.0
    loss = (circles() * weights).sum()
    loss.backward()
    if parameter == "radius":
        analytic = circles.sizes.grad[0, 0, 0].detach()
        parameter_tensor = circles.sizes[..., 0]
    elif parameter == "center_x":
        analytic = circles.centers.grad[0, 0, 0].detach()
        parameter_tensor = circles.centers[..., 0]
    else:
        analytic = circles.centers.grad[0, 0, 1].detach()
        parameter_tensor = circles.centers[..., 1]

    delta = circles.gradient_step_um
    with torch.no_grad():
        parameter_tensor[0, 0] += delta
        plus = (circles() * weights).sum().detach()
        parameter_tensor[0, 0] -= 2.0 * delta
        minus = (circles() * weights).sum().detach()
        parameter_tensor[0, 0] += delta
    finite_difference = (plus - minus) / (2.0 * delta)
    assert analytic == pytest.approx(finite_difference.item(), rel=1e-6, abs=1e-3)


def test_circle_shape_gradients_follow_material_polarity():
    high_index = _circle()
    low_index = _circle()
    high_index.geometry_cfgs["material_background_value"] = 0.0
    low_index.geometry_cfgs["material_background_value"] = 1.0
    with torch.no_grad():
        high_index.material_state.fill_(1.0)
        low_index.material_state.fill_(0.0)
    high_index.set_optimization_stage("shape")
    low_index.set_optimization_stage("shape")

    weights = torch.arange(1600, dtype=torch.float32).reshape(40, 40) / 1600.0
    high_center, high_radius, _ = high_index.compute_derivatives(weights, None)
    low_center, low_radius, _ = low_index.compute_derivatives(weights, None)

    assert torch.allclose(low_center, -high_center, atol=1e-6)
    assert torch.allclose(low_radius, -high_radius, atol=1e-6)


def test_circle_material_to_shape_uses_area_fraction():
    circles = _circle()
    circles.material_rho.data.fill_(0.25)
    fraction = circles.transfer_material_to_shape(
        eps_low=1.0,
        eps_high=12.0,
        eps_host=1.0,
        eps_inclusion=12.0,
    )
    assert fraction.item() == pytest.approx(0.25)
    assert circles.sizes[0, 0, 0].item() == pytest.approx(0.16 * 0.5)


def test_circle_etch_polarity_maps_low_material_to_larger_radius():
    circles = _circle()
    circles.material_rho.data.fill_(0.25)
    fraction = circles.transfer_material_to_shape(
        eps_low=2.0,
        eps_high=12.0,
        eps_host=12.0,
        eps_inclusion=2.0,
    )
    assert fraction.item() == pytest.approx(0.75)
    assert circles.sizes[0, 0, 0].item() == pytest.approx(0.16 * 0.75**0.5)
    assert circles.material_state.item() == pytest.approx(0.0)


def test_small_circle_policy_removes_or_clamps_tiny_radii():
    circles = _circle()
    with torch.no_grad():
        circles.radius.fill_(0.5e-4)
    assert circles._effective_radii().item() == pytest.approx(0.0)

    circles.small_circle_policy = "clamp"
    assert circles._effective_radii().item() == pytest.approx(1.0e-4)


def test_small_circle_policy_snap_by_radius_uses_half_minimum_threshold():
    circles = _circle()
    with torch.no_grad():
        circles.radius.fill_(0.49e-4)
    assert circles._effective_radii().item() == pytest.approx(0.0)

    with torch.no_grad():
        circles.radius.fill_(0.5e-4)
    circles.small_circle_policy = "snap_by_radius"
    assert circles._effective_radii().item() == pytest.approx(1.0e-4)

    with torch.no_grad():
        circles.radius.fill_(0.75e-4)
    assert circles._effective_radii().item() == pytest.approx(1.0e-4)

    with torch.no_grad():
        circles.radius.fill_(0.2)
    assert circles._effective_radii().item() == pytest.approx(0.2)


def test_small_circle_policy_snap_by_area_uses_half_area_threshold():
    circles = _circle(small_circle_policy="snap_by_area")
    with torch.no_grad():
        circles.radius.fill_(0.7e-4)
    assert circles._effective_radii().item() == pytest.approx(0.0)

    with torch.no_grad():
        circles.radius.fill_(0.71e-4)
    assert circles._effective_radii().item() == pytest.approx(1.0e-4)

    with torch.no_grad():
        circles.radius.fill_(0.75e-4)
    assert circles._effective_radii().item() == pytest.approx(1.0e-4)

    with torch.no_grad():
        circles.radius.fill_(0.2)
    assert circles._effective_radii().item() == pytest.approx(0.2)


def test_max_radius_config_clamps_to_user_and_lattice_bounds():
    circles = _circle(max_radius_um=0.1)
    with torch.no_grad():
        circles.radius.fill_(0.16)
    assert circles.max_radius_um == pytest.approx(0.1)
    assert circles._max_radius() == pytest.approx(0.1)
    assert circles._effective_radii().item() == pytest.approx(0.1)

    lattice_limited = _circle(max_radius_um=1.0)
    assert lattice_limited._max_radius() == pytest.approx(0.5)
    with torch.no_grad():
        lattice_limited.radius.fill_(0.6)
    assert lattice_limited._effective_radii().item() == pytest.approx(0.5)


@pytest.mark.parametrize("policy", ["remove", "snap_by_radius", "snap_by_area"])
def test_removed_zero_radius_gets_center_cell_recovery_gradient(policy):
    circles = _circle()
    circles.set_optimization_stage("shape")
    circles.small_circle_policy = policy
    with torch.no_grad():
        circles.radius.zero_()

    # The center (0.07, -0.08) falls in cell (22, 23) for this y-up grid.
    weights = torch.zeros((40, 40), dtype=torch.float64)
    weights[22, 23] = 1.0
    (circles() * weights).sum().backward()

    assert circles.sizes.grad[0, 0, 0].item() == pytest.approx(
        circles.zero_radius_gradient_scale
    )


def test_circle_and_cylinder_parameterization_registry_shapes():
    region = {
        "center": (0.0, 0.0),
        "size": (1.0, 1.0),
        "eps": 12.0,
        "eps_bg": 1.0,
    }
    device = SimpleNamespace(
        sim_cfg={"resolution": 20},
        design_region_masks={"design_region_1": (slice(0, 20), slice(0, 20))},
        design_region_cfgs={"design_region_1": region},
    )
    for method, key in (("circle", "circles"), ("cylinder", "cylinders")):
        parameterization = CircleParameterization(
            device=device,
            hr_device=device,
            sim_cfg={"resolution": 20},
            region_name="design_region_1",
            cfgs={
                "method": method,
                "geometry_cfgs": {
                    "batch_dims": (2, 2),
                    "material_optimization": True,
                },
                "init_method": "2d_array_full",
            },
            operation_device=torch.device("cpu"),
        )
        assert key in parameterization.weights
        assert parameterization.weights[key].radius.shape == (2, 2)
