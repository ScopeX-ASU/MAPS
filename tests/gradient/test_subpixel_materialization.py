"""Geometric coverage is arithmetic in epsilon, not optimization density."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from core.invdes.models.base_optimization import BaseOptimization
from core.invdes.models.layers.parametrization.base_parametrization import (
    BaseParametrization,
)
from core.invdes.models.materialization.coverage import average_region_cells

pytestmark = [pytest.mark.unit, pytest.mark.gradient]


def _parameter(mode="inverse_eps"):
    p = BaseParametrization.__new__(BaseParametrization)
    nn.Module.__init__(p)
    p.cfgs = dict(dims=(0, 1), denorm_mode=mode, extrude_angle=0.0, transform=[])
    p.design_region_cfg = dict(
        eps=12.0, eps_bg=2.0, size=(2.0, 2.0, 0.22), center=(0, 0, 0)
    )
    p.design_region_mask = (slice(0, 1), slice(0, 1), slice(0, 3))
    p.hr_design_region_mask = (slice(0, 2), slice(0, 2), slice(0, 3))
    p.device = SimpleNamespace()
    p.hr_device = SimpleNamespace()
    return p


@pytest.mark.parametrize("mode", ["linear_eps", "inverse_eps", "linear_index"])
def test_full_silicon_extrusion_matches_fixed_port_and_single_coverage(mode):
    p = _parameter(mode)
    weights = (np.array([0.5]), np.ones(1), np.array([0.25, 1.0, 0.75]))
    base = torch.full((1, 1, 3), 2.0, dtype=torch.float64)
    rho, mask, extra = p._extrude_density_tensor(
        torch.ones((1, 1), dtype=torch.float64),
        entire_eps=base,
        region_mask=p.design_region_mask,
        axis_weights=weights,
        grid_step=0.04,
    )
    eps = p.denormalize_permittivity(rho)
    opt = BaseOptimization.__new__(BaseOptimization)
    nn.Module.__init__(opt)
    opt.device = SimpleNamespace(
        design_region_masks={"r": mask},
        design_region_axis_weights={"r": weights},
        design_region_mask_weights={},
    )
    region = nn.Module()
    region.cfgs = p.cfgs
    opt.design_region_param_dict = nn.ModuleDict({"r": region})
    opt._apply_region_property_map(base, "r", eps, region_weight_override=extra)
    expected = 2 + 10 * torch.tensor([0.125, 0.5, 0.375], dtype=torch.float64)
    torch.testing.assert_close(base[0, 0], expected, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("mode", ["inverse_eps", "linear_index"])
def test_lateral_subpixel_averages_physical_permittivity(mode):
    p = _parameter(mode)
    density = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    _, low = p.density_transform(density, p.cfgs, 20, None, None)
    actual = p.denormalize_permittivity(low)
    torch.testing.assert_close(
        actual, torch.tensor([[9.5]], dtype=torch.float64), rtol=1e-7, atol=1e-7
    )


def test_existing_partial_port_is_not_blended_through_z_coverage_twice():
    p = _parameter()
    z = np.array([0.25, 1.0, 0.75])
    baseline = (2 + 10 * torch.tensor(z, dtype=torch.float64)).reshape(1, 1, 3)
    rho, _, _ = p._extrude_density_tensor(
        torch.ones((1, 1), dtype=torch.float64),
        entire_eps=baseline,
        region_mask=p.design_region_mask,
        axis_weights=(np.ones(1), np.ones(1), z),
        grid_step=0.04,
    )
    torch.testing.assert_close(p.denormalize_permittivity(rho), baseline)


def test_nonlinear_lateral_materialization_gradient_matches_finite_difference():
    p = _parameter()
    density = torch.tensor(
        [[0.1, 0.4], [0.7, 0.9]], dtype=torch.float64, requires_grad=True
    )

    def evaluate(x):
        _, low = p.density_transform(x, p.cfgs, 20, None, None)
        return p.denormalize_permittivity(low)

    assert torch.autograd.gradcheck(evaluate, (density,))


def test_fractional_extrusion_gradient_matches_finite_difference():
    p = _parameter()
    weights = (np.ones(1), np.ones(1), np.array([0.25, 1.0, 0.75]))
    base = torch.full((1, 1, 3), 2.0, dtype=torch.float64)

    def evaluate(density):
        rho, _, _ = p._extrude_density_tensor(
            density,
            entire_eps=base,
            region_mask=p.design_region_mask,
            axis_weights=weights,
            grid_step=0.04,
        )
        return p.denormalize_permittivity(rho).sum()

    x = torch.tensor([[0.63]], dtype=torch.float64, requires_grad=True)
    (derivative,) = torch.autograd.grad(evaluate(x), x)
    h = 1e-5
    finite_difference = (evaluate(x.detach() + h) - evaluate(x.detach() - h)) / (2 * h)
    # Extrusion intentionally averages its reverse-mode sensitivity over z.
    torch.testing.assert_close(
        derivative.squeeze() * sum(weights[2]), finite_difference, rtol=1e-6, atol=1e-7
    )


def test_nonuniform_shifted_cells_average_by_overlap_not_center_sampling():
    values = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    # Region [2,4] x [-1,1], with source cell edges [2,3,4] and [-1,0,1].
    actual = average_region_cells(
        values,
        ([2.5, 3.5], [-0.5, 0.5]),
        ([1.5, 2.25, 3.5, 4.5], [-1.5, 0.5, 1.5]),
        ((2.0, 4.0), (-1.0, 1.0)),
    )
    wx = torch.tensor([[1.0, 0.0], [0.6, 0.4], [0.0, 1.0]], dtype=torch.float64)
    wy = torch.tensor([[2 / 3, 1 / 3], [0.0, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(actual, wx @ values @ wy.T)
    assert torch.autograd.gradcheck(
        lambda x: average_region_cells(
            x,
            ([2.5, 3.5], [-0.5, 0.5]),
            ([1.5, 2.25, 3.5, 4.5], [-1.5, 0.5, 1.5]),
            ((2.0, 4.0), (-1.0, 1.0)),
        ),
        (values,),
    )


def test_noninteger_single_axis_downsampling_preserves_physical_percentages():
    from core.invdes.models.materialization.coverage import average_uniform_cells

    fraction = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float64, requires_grad=True
    )
    actual = average_uniform_cells(fraction, target_size=(2, 2))
    expected = torch.tensor([[1 / 3, 1 / 3], [1.0, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(
        lambda x: average_uniform_cells(x, target_size=(2, 2)),
        (fraction,),
    )


def test_native_levelset_hook_uses_physical_cell_averages():
    from core.invdes.models.layers.parametrization.levelset import (
        LevelSetParameterization,
    )

    p = _parameter()
    p.device.optical_grid_metadata = {
        "coords": (np.array([0.0]), np.array([0.0]), np.array([0.0])),
        "boundaries": (
            np.array([-1.0, 1.0]),
            np.array([-1.0, 1.0]),
            np.array([-0.11, 0.11]),
        ),
    }
    p.params = {"phi": (torch.tensor([-1.0, 1.0]), torch.tensor([-1.0, 1.0]))}
    p.operation_device = torch.device("cpu")
    # Bind actual subclass methods without starting a simulation or mode solve.
    from types import MethodType

    p._region_coordinates = MethodType(LevelSetParameterization._region_coordinates, p)
    p._resample_density_to_native = MethodType(
        LevelSetParameterization._resample_density_to_native, p
    )
    _, low = p.density_transform(
        torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float64),
        p.cfgs,
        20,
        None,
        None,
    )
    torch.testing.assert_close(
        p.denormalize_permittivity(low), torch.tensor([[9.5]], dtype=torch.float64)
    )


@pytest.mark.gpu
def test_actual_wdm_all_solid_design_joins_fixed_port(require_cuda):
    """Exercise the real raster grid and overlap cells without an FDTD solve."""
    from core.invdes.models.base_optimization import DefaultSimulationConfig
    from core.invdes.models.layers import WDM

    config = DefaultSimulationConfig()
    config.update(
        dict(
            solver="fdtdx",
            resolution=25,
            wl_cen=1.56,
            wl_width=0.04,
            n_wl=2,
            border_width=[0, 0, 1.5, 1.5, 0.7, 0.7],
            PML=[0.2, 0.2, 0.2],
        )
    )
    device = WDM(
        material_r1="Si",
        sim_cfg=config,
        box_size=(6, 6, 0.22),
        port_len=(1.5, 1.5),
        port_width=(0.5, 0.5),
        port_height=(0.22, 0.22),
        is_3d=True,
        mode="Hz1",
        num_outports=2,
        port_box_margin=4.5,
        device=torch.device("cuda:0"),
    )
    p = _parameter()
    name = "design_region_1"
    p.design_region_cfg = device.design_region_cfgs[name]
    mask = device.design_region_masks[name]
    base = torch.as_tensor(np.real(device.epsilon_map), device="cuda:0").clone()
    density = torch.ones(tuple(s.stop - s.start for s in mask[:2]), device=base.device)
    rho, support, coverage = p._extrude_density_tensor(
        density,
        entire_eps=base,
        region_mask=mask,
        axis_weights=device.design_region_axis_weights[name],
        grid_step=device.grid_step,
    )
    opt = BaseOptimization.__new__(BaseOptimization)
    nn.Module.__init__(opt)
    opt.device = device
    opt.design_region_param_dict = nn.ModuleDict({name: p})
    opt._apply_region_property_map(
        base,
        name,
        p.denormalize_permittivity(rho),
        region_mask_override=support,
        region_weight_override=coverage,
    )
    coords = device.grid_info_dict["epsilon_map"]["coords"]
    y = int(np.argmin(abs(coords[1])))
    port_x = int(np.argmin(abs(coords[0] + 3.5)))
    for x in (mask[0].start, mask[0].start + 1, (mask[0].start + mask[0].stop) // 2):
        torch.testing.assert_close(base[x, y], base[port_x, y], rtol=2e-6, atol=2e-6)
