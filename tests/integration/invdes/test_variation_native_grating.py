"""Native-grid comparison against an independently reconstructed grating device."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import GratingCoupler
from core.invdes.models.layers.parametrization.extrude import extrude
from core.invdes.variation import MapsGeometryTransform

pytestmark = [pytest.mark.integration, pytest.mark.numerical, pytest.mark.gpu]


@pytest.mark.parametrize("border_size", [(0.0, 0.0), (0.1, 0.1)])
def test_native_etch_matches_reconstructed_grating(require_cuda, border_size):
    sim = DefaultSimulationConfig()
    sim.update(
        dict(
            solver="fdtdx",
            resolution=20,
            symmetry=(0, 0, 0),
            cell_size=(2.8, 1.8, 3.2),
            PML=[0.3, 0.3, 0.3],
            border_width=[0.8, 0, 0.3, 0.3, 0, 0],
            wl_cen=1.31,
            wl_width=0.0,
            n_wl=1,
        )
    )
    device = GratingCoupler(
        material_r=12.25,
        material_layers=[12.25, 4.0],
        thickness_layers=[0.22, 0.4],
        etch_depth_layers=[0.16, 0.3],
        gap_layers=[0.0, 0.1],
        material_buried=2.0736,
        material_cladding=2.0736,
        sim_cfg=sim,
        design_region_size=(1.2, 1.2),
        design_region_border_size=border_size,
        port_len=0.8,
        port_width=0.4,
        port_height=0.22,
        top_port_len=1.5,
        substrate_thickness=0.3,
        substrate_gap=0.6,
        is_3d=True,
        device=torch.device("cuda:0"),
        verbose=False,
    )
    rebuilt = device.copy(etch_depth_layers=[0.17, 0.3])
    info = device._get_raster_grid_info("epsilon_map")
    for a, b in zip(
        info["boundaries"], rebuilt._get_raster_grid_info("epsilon_map")["boundaries"]
    ):
        np.testing.assert_array_equal(a, b)
    base = torch.as_tensor(device.epsilon_map.real, device="cuda:0")
    scalar = torch.tensor(0.3, device="cuda:0", requires_grad=True)
    regions = {}
    for name, cfg in device.design_region_cfgs.items():
        mask = device.design_region_masks[name]
        plane = scalar * torch.ones(base[mask].shape[:2], device="cuda:0")
        density = extrude(
            plane,
            2,
            base[mask].shape[2],
            extrude_angle=0,
            extrude_weights=device.design_region_axis_weights[name][2],
            base_permittivity=(base[mask] - cfg["eps_bg"])
            / (cfg["eps"] - cfg["eps_bg"]),
        )
        regions[name] = {
            "permittivity": cfg["eps_bg"] + density * (cfg["eps"] - cfg["eps_bg"])
        }
    nominal = (base, {}, base, {}, regions)
    model = SimpleNamespace(device=device, epsilon_map=base)
    varied = MapsGeometryTransform(model)(nominal, {"grating_layer_1.etch_depth": 0.01})
    expected = torch.as_tensor(rebuilt.epsilon_map.real.copy(), device="cuda:0")
    for name, cfg in rebuilt.design_region_cfgs.items():
        mask = rebuilt.design_region_masks[name]
        plane = scalar * torch.ones(expected[mask].shape[:2], device="cuda:0")
        density = extrude(
            plane,
            2,
            expected[mask].shape[2],
            extrude_angle=0,
            extrude_weights=rebuilt.design_region_axis_weights[name][2],
            base_permittivity=(expected[mask].detach() - cfg["eps_bg"])
            / (cfg["eps"] - cfg["eps_bg"]),
        )
        region = cfg["eps_bg"] + density * (cfg["eps"] - cfg["eps_bg"])
        axes = rebuilt.design_region_axis_weights[name]
        weight = torch.as_tensor(
            axes[0][:, None, None] * axes[1][None, :, None], device="cuda:0"
        )
        expected[mask] = expected[mask] * (1 - weight) + region * weight
    # Float32 native CuPy rasterization and endpoint inversion can accumulate
    # several ulps. This bound is relative to material permittivity (~12).
    torch.testing.assert_close(varied, expected, rtol=2e-6, atol=2e-6)
    varied.sum().backward()
    assert scalar.grad is not None and torch.isfinite(scalar.grad) and scalar.grad > 0
