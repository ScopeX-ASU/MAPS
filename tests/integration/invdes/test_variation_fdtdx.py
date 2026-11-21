"""Small GPU FDTDx optimization contract with shared bilayer parameters."""

import pytest
import torch
from benchmark_grating_coupler_variation import benchmark, build_model, variation_config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.fdtdx,
    pytest.mark.gpu,
    pytest.mark.slow,
]


def test_nonuniform_gaussian_source_and_recorder_cover_physical_aperture(
    require_cuda, monkeypatch
):
    """Gaussian launch and sparse source-plane recording share physical bounds."""
    import importlib

    import numpy as np

    for module_name in (
        "core.invdes.models.layers.device_base",
        "core.fdtd.fdtd",
        "core.fdtd.solver",
    ):
        monkeypatch.setattr(
            importlib.import_module(module_name), "_MAPS_SPARSE_PHASOR", True
        )

    torch.manual_seed(541)
    model = build_model(
        torch.device("cuda:0"),
        resolution=20,
        max_time=3e-14,
        optical_grid=dict(
            mode="fdtdx_rectilinear_nonuniform",
            wavelength=1.31,
            **{
                f"grid_{axis}": dict(type="auto", min_steps_per_wvl=15)
                for axis in "xyz"
            },
        ),
    )
    prototype = next(iter(model.objective.port_profiles["in_slice_1"].values()))[0]
    result = model(sharpness=2.0)
    simulation = next(iter(model.objective.sims.values()))
    source = simulation.objects.sources[0]
    edges = simulation._native_boundaries()
    info = model.device.port_monitor_slices_info["in_slice_1"]
    for axis in (0, 1):
        lower, upper = source.grid_slice_tuple[axis]
        requested_lower = (info["center"][axis] - info["size"][axis] / 2) * 1e-6
        requested_upper = (info["center"][axis] + info["size"][axis] / 2) * 1e-6
        assert edges[axis][lower] <= requested_lower + 1e-12
        assert edges[axis][upper] >= requested_upper - 1e-12
        assert requested_lower - edges[axis][lower] <= np.diff(edges[axis]).max()
        assert edges[axis][upper] - requested_upper <= np.diff(edges[axis]).max()
    recorder = simulation.objects["field_detector_port_in_slice_1_Hz2"]
    assert recorder.grid_slice_tuple == source.grid_slice_tuple
    local_edges = [np.asarray(axis) for axis in source._local_edge_coordinates()]
    transverse = [
        0.5 * (axis[:-1] + axis[1:]) - axis[-1] / 2 for axis in local_edges[:2]
    ]
    expected = np.exp(
        -(transverse[0][:, None] ** 2 + transverse[1][None, :] ** 2)
        / source.waist_radius**2
    )
    injected = np.abs(np.asarray(source._E)[1, ..., 0])
    np.testing.assert_allclose(
        injected / injected.max(), expected / expected.max(), rtol=2e-6, atol=1e-7
    )
    assert (
        next(iter(model.objective.port_profiles["in_slice_1"].values()))[0] is prototype
    )
    (-result["obj"]).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_fdtdx_robust_step_has_finite_shared_gradients_and_restores_sources(
    require_cuda,
):
    torch.manual_seed(541)
    model = build_model(torch.device("cuda:0"), resolution=20, max_time=3e-14)
    profiles = model.objective.port_profiles["in_slice_1"]
    source_before = {key: value[0] for key, value in profiles.items()}
    parameters_before = [p.detach().clone() for p in model.parameters()]
    records = benchmark(model, steps=1, variation=variation_config(42, 2))
    record = records[0]
    assert record["objective"] > 0 and record["gradient_norm"] > 0
    assert len(record["scenario_objectives"]) == 2
    assert record["objective"] == pytest.approx(
        sum(record["scenario_objectives"]) / 2, rel=2e-6
    )
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert any(
        not torch.equal(old, new)
        for old, new in zip(parameters_before, model.parameters())
    )
    plotted_eps = model._resolve_plot_eps_map(("in_slice_1", 1.31, "Hz2", 300))
    assert plotted_eps.grad is not None and torch.isfinite(plotted_eps.grad).all()
    assert plotted_eps.grad.abs().max() > 0
    assert all(
        model.objective.port_profiles["in_slice_1"][key][0] is source
        for key, source in source_before.items()
    )


def test_zero_variation_material_gradient_matches_nominal_plot_planes(require_cuda):
    """A single nominal scenario preserves real FDTD gradients in all three views."""
    import numpy as np

    torch.manual_seed(541)
    model = build_model(torch.device("cuda:0"), resolution=20, max_time=3e-14)
    field_key = ("in_slice_1", 1.31, "Hz2", 300)
    nominal = model(sharpness=2.0)
    (-nominal["obj"]).backward()
    nominal_eps = model._resolve_plot_eps_map(field_key)
    assert nominal_eps.grad is not None
    expected = nominal_eps.grad.detach().clone()
    expected_params = [p.grad.detach().clone() for p in model.parameters()]
    assert expected.abs().max() > 0
    model.zero_grad(set_to_none=True)
    model.configure_variation(dict(enabled=True, scenarios=[{}]))
    varied = model(sharpness=2.0)
    plotted_eps = model._resolve_plot_eps_map(field_key)
    assert plotted_eps.grad is None
    (-varied["obj"]).backward()
    assert plotted_eps.grad is not None
    torch.testing.assert_close(
        varied["obj"], nominal["obj"].detach().reshape(()), rtol=1e-4, atol=1e-12
    )
    torch.testing.assert_close(plotted_eps.grad, expected, rtol=1e-4, atol=1e-12)
    coords = model.device._get_raster_grid_info("epsilon_map")["coords"]
    for cfg in model.device.design_region_cfgs.values():
        centers = [
            int(np.argmin(np.abs(np.asarray(axis) - center)))
            for axis, center in zip(coords, cfg["center"])
        ]
        for fixed_axis in (2, 1, 0):
            plane = [slice(None)] * 3
            plane[fixed_axis] = centers[fixed_axis]
            torch.testing.assert_close(
                plotted_eps.grad[tuple(plane)],
                expected[tuple(plane)],
                rtol=1e-4,
                atol=1e-12,
            )
    for p, expected_param in zip(model.parameters(), expected_params):
        torch.testing.assert_close(p.grad, expected_param, rtol=1e-4, atol=1e-12)
