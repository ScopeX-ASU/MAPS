from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.invdes.invdesign import clamp_material_parameters
from core.invdes.models.base_optimization import BaseOptimization
from core.invdes.models.layers.parametrization.box import (
    Boxes,
    BoxParameterization,
    _gpu_raster_to_torch_xy,
)
from core.invdes.models.layers.parametrization.material_projection import (
    EffectiveMediumProjector,
)


def _region_devices():
    region = {
        "center": (0.0, 0.0),
        "size": (2.0, 2.0),
        "eps": 12.0,
        "eps_bg": 1.0,
    }
    masks = {"design_region_1": (slice(0, 20), slice(0, 20))}
    return (
        SimpleNamespace(
            design_region_masks=masks,
            design_region_cfgs={"design_region_1": region},
        ),
        SimpleNamespace(
            sim_cfg={"resolution": 10},
            design_region_masks=masks,
            design_region_cfgs={"design_region_1": region},
        ),
        region,
    )


def _boxes(**geometry_updates):
    _, hr_device, region = _region_devices()
    geometry_cfgs = {
        "batch_dims": (2, 1),
        "size_dim": 2,
        "material_optimization": True,
    }
    geometry_cfgs.update(geometry_updates)
    boxes = Boxes(
        sim_cfg=hr_device.sim_cfg,
        geometry_cfgs=geometry_cfgs,
        design_region_mask=hr_device.design_region_masks["design_region_1"],
        design_region_cfg=region,
        operation_device=torch.device("cpu"),
    )
    with torch.no_grad():
        boxes.centers.copy_(torch.tensor([[[-0.5, 0.0]], [[0.5, 0.0]]]))
        boxes.sizes.fill_(0.4)
        boxes.material_rho.copy_(torch.tensor([[0.25], [0.75]]))
    return boxes


def test_gpu_raster_is_reoriented_to_torch_cartesian_y():
    raster = np.zeros((2, 3, 1), dtype=np.float32)
    raster[:, :, 0] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    oriented = _gpu_raster_to_torch_xy(raster)

    np.testing.assert_array_equal(
        oriented[:, :, 0],
        np.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]], dtype=np.float32),
    )
    assert oriented.flags.c_contiguous


def test_projector_linear_eps_and_box_scale():
    projector = EffectiveMediumProjector()
    rho = torch.tensor([0.25, 0.75])
    fill = projector.rho_to_fill_fraction(
        rho,
        eps_low=1.0,
        eps_high=12.0,
        eps_host=1.0,
        eps_inclusion=12.0,
    )
    assert torch.allclose(fill, rho)
    assert torch.allclose(
        projector.fill_fraction_to_scale(fill),
        torch.tensor([[0.5, 0.5], [3**0.5 / 2, 3**0.5 / 2]]),
    )
    # Reversing host/inclusion polarity makes the same effective material a
    # low-index hole fraction rather than silently assuming a high-index post.
    hole_fill = projector.rho_to_fill_fraction(
        rho,
        eps_low=1.0,
        eps_high=12.0,
        eps_host=12.0,
        eps_inclusion=1.0,
    )
    assert torch.allclose(hole_fill, 1.0 - rho)


def test_box_material_values_and_gradient():
    boxes = _boxes()
    boxes.set_optimization_stage("material")
    density = boxes()
    assert density.shape == (20, 20)
    assert torch.any(torch.isclose(density, torch.tensor(0.25)))
    assert torch.any(torch.isclose(density, torch.tensor(0.75)))

    loss = density.sum()
    loss.backward()
    assert boxes.material_rho.grad is not None

    # The density is linear in each material value while geometry is fixed.
    with torch.no_grad():
        baseline = loss.detach()
        delta = 1e-3
        boxes.material_rho[0, 0] += delta
        plus = boxes().sum().detach()
        boxes.material_rho[0, 0] -= 2 * delta
        minus = boxes().sum().detach()
        boxes.material_rho[0, 0] += delta
    finite_difference = (plus - minus) / (2 * delta)
    assert torch.allclose(boxes.material_rho.grad[0, 0], finite_difference, atol=1e-3)
    assert baseline.item() == pytest.approx(loss.item())


def test_box_material_gradient_covers_all_lattice_entries():
    _, hr_device, region = _region_devices()
    boxes = Boxes(
        sim_cfg=hr_device.sim_cfg,
        geometry_cfgs={
            "batch_dims": (4, 4),
            "size_dim": 2,
            "material_optimization": True,
            "phi_resolution": (20, 20),
            "border": 0.0,
            "material_background_value": 1.0,
        },
        design_region_mask=hr_device.design_region_masks["design_region_1"],
        design_region_cfg=region,
        operation_device=torch.device("cpu"),
    )
    with torch.no_grad():
        centers = torch.linspace(-0.75, 0.75, 4)
        xx, yy = torch.meshgrid(centers, centers, indexing="ij")
        boxes.centers[..., 0].copy_(xx)
        boxes.centers[..., 1].copy_(yy)
        boxes.sizes.fill_(0.5)
    boxes.set_optimization_stage("material")

    boxes().sum().backward()

    gradient = boxes.material_rho.grad
    assert gradient.shape == (4, 4)
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient != 0)


@pytest.mark.parametrize("background, value, sign", [(0.0, 1.0, 1.0), (1.0, 0.0, -1.0)])
def test_box_size_gradient_uses_edge_difference_and_material_polarity(
    background, value, sign
):
    boxes = _boxes(material_background_value=background)
    with torch.no_grad():
        boxes.material_state.fill_(value)
        boxes.sizes[0, 0].copy_(torch.tensor([0.63, 0.47]))
    boxes.set_optimization_stage("shape")

    _, grad_sizes, _ = boxes.compute_derivatives(torch.ones((20, 20)), None)
    expected = sign * torch.tensor([0.47, 0.63]) / (0.1**2)
    assert torch.allclose(grad_sizes[0, 0], expected, atol=1e-5)


def test_clamp_material_parameters_keeps_only_material_in_legal_range():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.material_rho = torch.nn.Parameter(
                torch.tensor([[-0.2, 0.4], [1.2, 2.0]])
            )
            self.other_parameter = torch.nn.Parameter(torch.tensor([-2.0, 2.0]))

    model = Model()
    invdes = SimpleNamespace(devOptimization=model)

    clamp_material_parameters(invdes)

    assert torch.all((model.material_rho >= 0.0) & (model.material_rho <= 1.0))
    assert torch.equal(model.other_parameter, torch.tensor([-2.0, 2.0]))


def test_box_stage_transfer_and_stage_flags():
    boxes = _boxes()
    boxes.set_optimization_stage("material")
    boxes.transfer_material_to_shape()

    assert torch.equal(boxes.material_state, torch.tensor([[0.0], [1.0]]))
    max_size = torch.tensor(0.8)
    assert boxes.sizes[0, 0, 0] == pytest.approx(max_size.item() * 0.5)
    assert boxes.sizes[1, 0, 0] == pytest.approx(max_size.item() * (0.75**0.5))

    boxes.set_optimization_stage("shape")
    assert not boxes.centers.requires_grad
    assert boxes.sizes.requires_grad
    assert not boxes.material_rho.requires_grad
    assert torch.allclose(boxes().max(), torch.tensor(1.0))

    boxes.set_optimization_stage("joint", optimize_centers=True)
    assert boxes.centers.requires_grad
    assert boxes.sizes.requires_grad
    assert boxes.material_rho.requires_grad


def test_box_size_bounds_accept_scalar_or_per_axis_values():
    boxes = _boxes(min_size_um=(0.1, 0.2), max_size_um=(0.5, 0.6))
    with torch.no_grad():
        boxes.sizes[0, 0].copy_(torch.tensor([0.4, 0.7]))

    assert boxes.min_size_um == (0.1, 0.2)
    assert boxes.max_size_um == (0.5, 0.6)
    assert torch.allclose(boxes._effective_box_sizes()[0, 0], torch.tensor([0.4, 0.6]))

    boxes = _boxes(min_size_um=0.1, max_size_um=0.5)
    with torch.no_grad():
        boxes.sizes[0, 0].copy_(torch.tensor([0.7, 0.7]))
    assert torch.allclose(boxes._effective_box_sizes()[0, 0], torch.tensor([0.5, 0.5]))


@pytest.mark.parametrize("policy", ["remove", "clamp"])
def test_small_box_remove_and_clamp_policies(policy):
    boxes = _boxes(min_size_um=0.2, small_box_policy=policy)
    with torch.no_grad():
        boxes.sizes[0, 0].copy_(torch.tensor([0.1, 0.4]))

    expected = (
        torch.tensor([0.0, 0.0]) if policy == "remove" else torch.tensor([0.2, 0.4])
    )
    assert torch.allclose(boxes._effective_box_sizes()[0, 0], expected)


def test_small_box_snap_by_area_policy():
    boxes = _boxes(min_size_um=0.2, small_box_policy="snap_by_area")
    with torch.no_grad():
        boxes.sizes[0, 0].copy_(torch.tensor([0.1, 0.1]))
    assert torch.allclose(boxes._effective_box_sizes()[0, 0], torch.tensor([0.0, 0.0]))

    with torch.no_grad():
        boxes.sizes[0, 0].copy_(torch.tensor([0.15, 0.15]))
    assert torch.allclose(boxes._effective_box_sizes()[0, 0], torch.tensor([0.2, 0.2]))


def test_small_box_snap_by_min_side_policy():
    boxes = _boxes(min_size_um=0.2, small_box_policy="snap_by_min_side")
    with torch.no_grad():
        boxes.sizes[0, 0].copy_(torch.tensor([0.09, 0.4]))
    assert torch.allclose(boxes._effective_box_sizes()[0, 0], torch.tensor([0.0, 0.0]))

    with torch.no_grad():
        boxes.sizes[0, 0].copy_(torch.tensor([0.11, 0.4]))
    assert torch.allclose(boxes._effective_box_sizes()[0, 0], torch.tensor([0.2, 0.4]))


def test_box_shared_size_uses_square_geometry_and_aggregates_gradients():
    boxes = _boxes(share_size=True)
    with torch.no_grad():
        boxes.sizes[0, 0].copy_(torch.tensor([0.3, 0.5]))
    assert torch.allclose(boxes._effective_box_sizes()[0, 0], torch.tensor([0.3, 0.3]))

    boxes.set_optimization_stage("shape")
    upstream = torch.arange(400, dtype=torch.float32).reshape(20, 20)
    _, shared_grad, _ = boxes.compute_derivatives(upstream, None)

    independent = _boxes()
    with torch.no_grad():
        independent.sizes[0, 0].fill_(0.3)
    independent.set_optimization_stage("shape")
    _, independent_grad, _ = independent.compute_derivatives(upstream, None)

    assert torch.allclose(shared_grad[0, 0, 0], independent_grad[0, 0].sum())
    assert shared_grad[0, 0, 1] == 0


def test_param_gradient_plot_uses_active_box_sizes_not_stale_material_gradient():
    region = torch.nn.Module()
    region.cfgs = {"method": "box", "dims": (0, 1)}
    region.design_region_cfg = {"size": (2.0, 2.0)}
    region.sizes = torch.nn.Parameter(torch.zeros(2, 2, 2))
    region.material_rho = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    region.sizes.grad = torch.zeros_like(region.sizes)
    region.sizes.grad[..., 0] = 3.0
    region.sizes.grad[..., 1] = 4.0
    region.material_rho.grad = torch.ones_like(region.material_rho)

    optimization = SimpleNamespace(design_region_param_dict={"design_region_1": region})
    plot_data = BaseOptimization._get_region_param_grad_plot_data(
        optimization, "design_region_1"
    )

    assert plot_data["name"] == "sizes_x"
    assert torch.allclose(plot_data["grad"], torch.full((2, 2), 3.0))


def test_full_array_initialization():
    device, hr_device, _ = _region_devices()
    parameterization = BoxParameterization(
        device=device,
        hr_device=hr_device,
        sim_cfg=hr_device.sim_cfg,
        region_name="design_region_1",
        cfgs={
            "method": "box",
            "geometry_cfgs": {
                "batch_dims": (2, 2),
                "size_dim": 2,
                "material_optimization": True,
            },
            "init_method": "2d_array_full",
            "material_init": 0.4,
        },
        operation_device=torch.device("cpu"),
    )
    boxes = parameterization.weights["boxes"]
    assert torch.allclose(boxes.sizes[..., 0], torch.full((2, 2), 0.8))
    assert torch.allclose(boxes.sizes[..., 1], torch.full((2, 2), 0.8))
    assert torch.allclose(boxes.material_rho, torch.full((2, 2), 0.4))


def test_box_callable_initialization():
    device, hr_device, region = _region_devices()

    def init_fn(weight_dict, param_cfg, region_cfg):
        del param_cfg, region_cfg
        with torch.no_grad():
            weight_dict["boxes"].sizes.fill_(0.125)

    parameterization = BoxParameterization(
        device=device,
        hr_device=hr_device,
        sim_cfg=hr_device.sim_cfg,
        region_name="design_region_1",
        cfgs={
            "method": "box",
            "geometry_cfgs": {"batch_dims": (2, 2), "size_dim": 2},
            "init_method": init_fn,
        },
        operation_device=torch.device("cpu"),
    )

    assert torch.allclose(
        parameterization.weights["boxes"].sizes,
        torch.full((2, 2, 2), 0.125),
    )


def test_box_phi_grid_is_uniform_and_resamples_by_coordinates():
    region = {
        "center": (0.0, 0.0),
        "size": (2.0, 2.0),
        "eps": 12.0,
        "eps_bg": 1.0,
    }
    coords = (
        np.array([-1.4, -0.8, -0.2, 0.5, 1.3]),
        np.array([-1.3, -0.7, -0.1, 0.6, 1.2]),
    )
    masks = {"design_region_1": (slice(1, 4), slice(1, 4))}
    grid_info = {"epsilon_map": {"coords": coords}}
    device = SimpleNamespace(
        design_region_masks=masks,
        design_region_cfgs={"design_region_1": region},
        grid_info_dict=grid_info,
    )
    hr_device = SimpleNamespace(
        sim_cfg={"resolution": 100},
        design_region_masks=masks,
        design_region_cfgs={"design_region_1": region},
        grid_info_dict=grid_info,
    )
    parameterization = BoxParameterization(
        device=device,
        hr_device=hr_device,
        sim_cfg=hr_device.sim_cfg,
        region_name="design_region_1",
        cfgs={
            "method": "box",
            "geometry_cfgs": {
                "batch_dims": (2, 2),
                "size_dim": 2,
                "material_optimization": True,
            },
            "phi_resolution": 4,
            "init_method": "2d_array_full",
        },
        operation_device=torch.device("cpu"),
    )
    boxes = parameterization.weights["boxes"]
    assert boxes.phi_shape == (8, 8)
    assert boxes.grid_step == pytest.approx(0.25)

    source_x = boxes.uniform_parameter_grid_coords()[0]
    source_density = source_x[:, None].expand(8, 8)
    sampled = parameterization._resample_uniform_parameter_density(
        source_density,
        device,
        masks["design_region_1"],
    )
    expected_x = torch.as_tensor(coords[0][1:4], dtype=sampled.dtype)
    expected_x = expected_x.clamp(source_x[0], source_x[-1])
    assert sampled.shape == (3, 3)
    assert torch.allclose(sampled[:, 1], expected_x, atol=1e-6)


def test_legacy_box_path_remains_binary_and_differentiable():
    _, hr_device, region = _region_devices()
    boxes = Boxes(
        sim_cfg=hr_device.sim_cfg,
        geometry_cfgs={"batch_dims": (1, 1), "size_dim": 2},
        design_region_mask=hr_device.design_region_masks["design_region_1"],
        design_region_cfg=region,
        operation_device=torch.device("cpu"),
    )
    with torch.no_grad():
        boxes.centers.copy_(torch.tensor([[[0.0, 0.0]]]))
        boxes.sizes.fill_(0.4)
    density = boxes()
    assert torch.all((density == 0) | (density == 1))
    density.sum().backward()
    assert boxes.sizes.grad is not None
    assert boxes.material_rho is None


def test_box_builds_xy_density_before_3d_extrusion():
    _, hr_device, region = _region_devices()
    region = dict(region, size=(2.0, 2.0, 0.22))
    boxes = Boxes(
        sim_cfg=hr_device.sim_cfg,
        geometry_cfgs={
            "batch_dims": (1, 1),
            "size_dim": 2,
            "material_optimization": True,
        },
        design_region_mask=(slice(0, 20), slice(0, 20), slice(0, 2)),
        design_region_cfg=region,
        operation_device=torch.device("cpu"),
    )
    assert boxes().shape == (20, 20)
