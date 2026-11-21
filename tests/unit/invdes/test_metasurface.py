"""Contracts for the image-forming diffractive metasurface example."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.invdes.examples.metasurface_3d import make_maps_target
from core.invdes.models.base_optimization import BaseOptimization
from core.invdes.models.layers import DiffractiveMetasurface
from core.invdes.models.layers.device_base import N_Ports
from core.invdes.models.layers.objective import (
    TargetIntensityAngularSpectrumObjective,
    TargetIntensityObjective,
)
from core.invdes.models.layers.utils import plot_monitor_fields
from core.utils import Slice3D


def _objective(target):
    return TargetIntensityObjective(
        sims={},
        port_slices={"image_plane": (slice(None), slice(None), 1)},
        in_slice_name="in_slice_1",
        out_slice_name="image_plane",
        in_mode="Hz1",
        target_wls=(1.0,),
        target_temps=(300,),
        target_intensity=target,
    )


def test_target_intensity_score_matches_reference_normalization():
    """A field with the target intensity has unit score at any amplitude."""
    target = torch.tensor([[0.0, 0.25], [0.5, 1.0]])
    amplitude = (7.0 * target).sqrt().to(torch.complex64).requires_grad_()
    field = torch.zeros((2, 2, 3), dtype=torch.complex64)
    field[:, :, 1] = amplitude
    objective = _objective(target)

    score = objective({("in_slice_1", 1.0, "Hz1", 300): {"Ex": field}})

    assert score.item() == pytest.approx(1.0, abs=1e-6)
    score.backward()
    assert amplitude.grad is not None
    assert torch.isfinite(amplitude.grad).all()
    torch.testing.assert_close(objective.last_target, target)
    torch.testing.assert_close(objective.last_intensity, target)


def test_target_intensity_resamples_and_uses_vector_electric_field():
    """The target follows monitor resolution and all E components contribute."""
    target = torch.eye(3)
    ex = torch.ones((4, 5, 2), dtype=torch.complex64, requires_grad=True)
    ey = 2j * torch.ones_like(ex)
    ez = 0.5 * torch.ones_like(ex)
    objective = _objective(target)

    score = objective(
        {
            ("in_slice_1", 1.0, "Hz1", 300): {
                "Ex": ex,
                "Ey": ey,
                "Ez": ez,
            }
        }
    )

    assert score.ndim == 0
    assert objective.last_target.shape == (4, 5)
    assert objective.last_intensity.shape == (4, 5)
    score.backward()
    assert ex.grad is not None


def test_periodic_angular_spectrum_objective_projects_and_backpropagates():
    """The metasurface objective propagates the near plane through Torch FFT."""
    generator = torch.Generator().manual_seed(4)
    ex = torch.complex(
        torch.rand((4, 5, 3), generator=generator),
        torch.rand((4, 5, 3), generator=generator),
    ).requires_grad_()
    ey = (0.4j * ex).requires_grad_()
    objective = TargetIntensityAngularSpectrumObjective(
        sims={},
        port_slices={"nearfield_projection": (slice(None), slice(None), 1)},
        in_slice_name="in_slice_1",
        out_slice_name="nearfield_projection",
        in_mode="Hz1",
        target_wls=(1.0,),
        target_temps=(300,),
        target_intensity=torch.eye(4, 5),
        projection_distance_um=5.8,
        grid_step_um=0.1,
    )

    score = objective({("in_slice_1", 1.0, "Hz1", 300): {"Ex": ex, "Ey": ey}})

    assert score.ndim == 0
    assert objective.last_nearfield_intensity.shape == (4, 5)
    expected_intensity = sum(
        objective.last_projected_fields[component].abs().square()
        for component in ("Ex", "Ey", "Ez")
    )
    torch.testing.assert_close(objective.last_projected_intensity, expected_intensity)
    assert objective.last_intensity.shape == (4, 5)
    assert set(objective.last_nearfield_fields) == {
        "Ex",
        "Ey",
        "spacing_um",
        "plane_axes",
    }
    assert set(objective.last_projected_fields) == {
        "Ex",
        "Ey",
        "Ez",
        "Hx",
        "Hy",
        "Hz",
        "spacing_um",
        "plane_axes",
    }
    assert objective.last_nearfield_fields["plane_axes"] == ("x", "y")
    assert objective.last_projected_fields["plane_axes"] == ("x", "y")
    score.backward()
    assert ex.grad is not None
    assert torch.isfinite(ex.grad).all()


def test_angular_spectrum_objective_selects_projected_components():
    """Configured components select the propagated intensity, not its inputs."""
    ex = torch.ones((3, 4, 2), dtype=torch.complex64, requires_grad=True)
    ey = (0.3j * torch.ones_like(ex)).requires_grad_()
    objective = TargetIntensityAngularSpectrumObjective(
        sims={},
        port_slices={"nearfield_projection": (slice(None), slice(None), 1)},
        in_slice_name="in_slice_1",
        out_slice_name="nearfield_projection",
        in_mode="Hz1",
        target_wls=(1.0,),
        target_temps=(300,),
        target_intensity=torch.ones((3, 4)),
        field_components=("Ex",),
        projection_distance_um=2.0,
        grid_step_um=0.1,
    )

    objective({("in_slice_1", 1.0, "Hz1", 300): {"Ex": ex, "Ey": ey}})

    torch.testing.assert_close(
        objective.last_projected_intensity,
        objective.last_projected_fields["Ex"].abs().square(),
    )


def test_base_optimization_plots_retained_monitor_fields(tmp_path):
    """The standard plot configuration can render near and far diagnostics."""
    values = torch.ones((4, 5), dtype=torch.complex64)
    objective_fn = SimpleNamespace(
        last_nearfield_fields={
            "Ex": values,
            "Ey": 0.5j * values,
            "spacing_um": (0.1, 0.2),
        },
        last_projected_fields={
            "Ex": 2 * values,
            "Ey": values,
            "Ez": 0.25j * values,
            "spacing_um": (0.1, 0.2),
        },
    )
    optimization = object.__new__(BaseOptimization)
    optimization.objective = SimpleNamespace(Js={"image_match": {"fn": objective_fn}})
    optimization.sim_cfg = {"plot_root": str(tmp_path)}

    outputs = optimization.plot_monitor_diagnostics(
        objective_name="image_match",
        monitor_plots=[
            {
                "objective": "image_match",
                "fields": ("nearfield", "farfield"),
                "components": "Ey",
                "quantities": "phase",
            }
        ],
        plot_filename="step_0001_image_match.png",
    )

    assert {os.path.basename(path) for path in outputs} == {
        "step_0001_image_match_nearfield.png",
        "step_0001_image_match_farfield.png",
    }
    assert all((tmp_path / path.split("/")[-1]).is_file() for path in outputs)


def test_monitor_plot_uses_component_rows_and_quantity_columns(tmp_path, monkeypatch):
    """Vector and scalar selectors produce the requested row-major panels."""
    from matplotlib.axes import Axes

    values = torch.ones((3, 4), dtype=torch.complex64)
    output = tmp_path / "total_e.png"
    panel_titles = []
    original_set_title = Axes.set_title

    def capture_title(axis, label, *args, **kwargs):
        panel_titles.append(label)
        return original_set_title(axis, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", capture_title)

    plot_monitor_fields(
        {
            "Ex": values,
            "Ey": 2j * values,
            "Ez": 0.5 * values,
            "spacing_um": (0.1, 0.2),
            "plane_axes": ("x", "y"),
        },
        filepath=str(output),
        title="Total E",
        components=("E", "Ey", "Ez"),
        quantities=("intensity", "phase"),
        phase_reference="Ey",
    )

    assert output.is_file()
    assert panel_titles == [
        "|E|² (raw)",
        "phase(E) [Ey ref]",
        "|Ey|² (raw)",
        "phase(Ey)",
        "|Ez|² (raw)",
        "phase(Ez)",
    ]


def test_maps_target_is_nonnegative_and_nontrivial():
    """The self-contained example target needs no external image asset."""
    target = make_maps_target(96)

    assert target.shape == (96, 96)
    assert target.dtype == np.float32
    assert 0.0 <= float(target.min()) < float(target.max()) <= 1.0


def test_device_delegates_asymmetric_z_bounds_to_base_recenter(monkeypatch):
    """The substrate covers the design interface by one nominal grid cell."""
    captured = {}

    def capture_base_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(N_Ports, "__init__", capture_base_init)
    device = DiffractiveMetasurface(
        sim_cfg={"wl_cen": 1.0, "cell_size": None},
        aperture=4.0,
        thickness=1.5,
        substrate_thickness=3.0,
        source_distance=1.5,
        image_distance=6.1,
        device="cpu",
    )

    assert captured["sim_cfg"]["cell_size"] is None
    input_port = captured["port_cfgs"]["in_port_1"]
    output_port = captured["port_cfgs"]["out_port_1"]
    substrate_bounds = (
        input_port["center"][2] - input_port["size"][2] / 2,
        input_port["center"][2] + input_port["size"][2] / 2,
    )
    design_region = captured["design_region_cfgs"]["design_region"]
    design_bottom = design_region["center"][2] - design_region["size"][2] / 2
    nearfield_z = output_port["center"][2] + output_port["size"][2] / 2

    assert substrate_bounds == pytest.approx((-3.75, -0.70))
    assert substrate_bounds[1] - design_bottom == pytest.approx(1 / 20)
    assert output_port["size"][2] == pytest.approx(0.3)
    assert nearfield_z == pytest.approx(1.05)
    assert device.substrate_thickness == pytest.approx(3.0)


def test_fdtdx_plane_wave_builder_uses_native_uniform_tfsf_source():
    """The 3-D plane-wave path builds FDTD-X's concrete TFSF source."""
    fdtdx = pytest.importorskip("fdtdx")
    device = object.__new__(N_Ports)
    device.port_monitor_slices_info = {
        "source": {
            "center": (0.0, 0.0, -1.5),
            "size": (4.0, 4.0, 0.0),
            "direction": "z+",
        }
    }

    profiles = device.insert_plane_wave_fdtd3d(
        source_slice_name="source",
        eps=None,
        slice=None,
        wl_cen=1.0,
        source_modes=("Hz1", "Hz2"),
        direction="z+",
    )

    source_x = profiles[((1.0, 0, 1), "Hz1")][0]
    source_y = profiles[((1.0, 0, 1), "Hz2")][0]
    assert isinstance(source_x, fdtdx.UniformPlaneSource)
    assert source_x.partial_grid_shape == (None, None, 1)
    assert source_x.partial_real_shape[:2] == pytest.approx((4e-6, 4e-6))
    assert source_x.partial_real_shape[2] is None
    assert source_x.fixed_E_polarization_vector == (1.0, 0.0, 0.0)
    assert source_y.fixed_E_polarization_vector == (0.0, 1.0, 0.0)
    assert source_x.direction == "+"


def test_near2far_homogeneity_check_uses_native_grid_slice():
    """Export-grid indices must never index the native epsilon tensor."""
    device = object.__new__(N_Ports)
    device.dim = 3
    device.axes = ("x", "y", "z")
    device.epsilon_map = np.ones((4, 5, 95), dtype=np.float32)
    device.port_monitor_slices_native = {}
    export_slice = Slice3D(x=slice(0, 4), y=slice(0, 5), z=103)
    native_slice = (slice(0, 4), slice(0, 5), 94)

    def register_slice(*_args, **_kwargs):
        device.port_monitor_slices_native["nearfield"] = native_slice
        return export_slice

    device.add_monitor_slice = register_slice
    returned = device.build_near2far_slice(
        slice_name="nearfield",
        center=(0.0, 0.0, 1.0),
        size=(4.0, 4.0, 0.0),
        direction="z+",
    )

    assert returned is export_slice


@pytest.mark.parametrize("target", [torch.zeros(4, 4), torch.zeros(4)])
def test_target_intensity_rejects_invalid_target(target):
    """Empty and non-image targets fail before an optimization starts."""
    with pytest.raises(ValueError):
        _objective(target)
