"""Reproduce the Tidy3D thermally tuned ring resonator tutorial with MAPS.

This example mirrors the main geometry and material settings from:
https://www.flexcompute.com/tidy3d/examples/notebooks/ThermallyTunedRingResonator/

Two important notes about the current MAPS heat stack:
1. The heat solver currently supports Dirichlet and Neumann faces, but not a
   Robin / convection boundary. This example approximates the tutorial's top
   convection boundary with a fixed-temperature top face.
2. The optical and thermal domains are different in the tutorial. To preserve
   that here without changing solver internals, this example uses a custom
   device class with a smaller optical box and a larger thermal-only box.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
from core.invdes.models.base_optimization import (
    BaseOptimization,
    DefaultSimulationConfig,
)
from core.invdes.models.layers import ThermoOpticMRR
from core.invdes.models.layers.thermal_control import canonicalize_currents
from core.utils import set_torch_deterministic

sys.path.pop(0)


def _zero_fusion(breakdown):
    for item in breakdown.values():
        value = item["value"]
        if isinstance(value, torch.Tensor):
            return value.new_zeros(())
    return 0.0


def _build_sim_cfg(*, n_wl: int, resolution: int, plot_root: str):
    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="fdtdx",
            border_width=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            PML=[0.5, 0.5, 0.5],
            max_time=2e-13 * 6 * 4,  # 2e-13 propagates ~1/6 loop, 4 is 4 loops
            resolution=resolution,
            optical_grid={
                "mode": "fdtdx_rectilinear_nonuniform",
                "wavelength": 1.55,
                "grid_x": {"type": "auto", "min_steps_per_wvl": 15},
                "grid_y": {"type": "auto", "min_steps_per_wvl": 15},
                "grid_z": {"type": "auto", "min_steps_per_wvl": 15},
            },
            plot_root=plot_root,
            wl_cen=1.55,
            wl_width=0,
            n_wl=n_wl,
            numerical_solver="solve_direct",
            use_autodiff=False,
            heat={
                "backend": "jax",
                "mesh_type": "fixed_rectilinear_nonuniform",
                "use_tidy3d": True,
                "grid_spec": {
                    "wavelength": 1.55,
                    "grid_x": {"type": "auto", "min_steps_per_wvl": 12},
                    "grid_y": {"type": "auto", "min_steps_per_wvl": 12},
                    "grid_z": {"type": "auto", "min_steps_per_wvl": 25},
                },
                "padding": {
                    "distance": [3.9, 3.9, 3.0, 3.0, 0.0, 0.0],
                    "mode": "replicate",
                    "source_mode": "constant",
                    "scheme": "graded",
                    "growth_rate": 1.07,
                    "max_scale": 3.0,
                },
                "use_tidy3d": True,
                "build_source_maps": True,
                "combine_sources": True,
                "requires_temp_grad": False,
                "solver_options": {
                    "dirichlet_mode": "reduced_free_dof",
                    "jax_solver": {
                        "method": "cg",
                        "precond": True,
                        "solve_dtype": "float64",
                        "check_residual": True,
                        "residual_rtol": 1e-6,
                        "residual_atol": 1e-8,
                    },
                },
                "adjoint_solver_options": {
                    "dirichlet_mode": "reduced_free_dof",
                    "jax_solver": {
                        "method": "cg",
                        "precond": True,
                        "solve_dtype": "float64",
                        "check_residual": True,
                        "residual_rtol": 1e-6,
                        "residual_atol": 1e-8,
                    },
                },
            },
        )
    )
    return sim_cfg


def _build_objective_cfgs(wls, heater_current, mode="Hz1"):
    off_currents = [{"heater_1": 0.0}]
    on_currents = [{"heater_1": float(heater_current)}]
    targets = [float(wl) for wl in wls]
    return dict(
        through_off=dict(
            weight=0.0,
            in_slice_name="in_slice_1",
            out_slice_name="through_slice",
            wl=targets,
            currents=off_currents,
            in_mode=mode,
            out_modes=(mode,),
            type="eigenmode",
            direction="x+",
            requires_grad=False,
        ),
        drop_off=dict(
            weight=0.0,
            in_slice_name="in_slice_1",
            out_slice_name="drop_slice",
            wl=targets,
            currents=off_currents,
            in_mode=mode,
            out_modes=(mode,),
            type="eigenmode",
            direction="x-",
            requires_grad=False,
        ),
        through_on=dict(
            weight=0.0,
            in_slice_name="in_slice_1",
            out_slice_name="through_slice",
            wl=targets,
            currents=on_currents,
            in_mode=mode,
            out_modes=(mode,),
            type="eigenmode",
            direction="x+",
            requires_grad=False,
        ),
        drop_on=dict(
            weight=0.0,
            in_slice_name="in_slice_1",
            out_slice_name="drop_slice",
            wl=targets,
            currents=on_currents,
            in_mode=mode,
            out_modes=(mode,),
            type="eigenmode",
            direction="x-",
            requires_grad=False,
        ),
        _fusion_func=_zero_fusion,
    )


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _ring_mask_from_coords(coords, *, ring_radius, w_wg, z_wg, h_wg):
    xs, ys, zs = np.meshgrid(*coords, indexing="ij")
    radial = np.sqrt(xs**2 + ys**2)
    return (
        (radial >= ring_radius - w_wg / 2.0)
        & (radial <= ring_radius + w_wg / 2.0)
        & (np.abs(zs - z_wg) <= max(h_wg / 2.0, 1e-9))
    )


def _summarize_runtime_state(device, state):
    optical_temperature = _to_numpy(state["optical_temperature"])
    mask = _ring_mask_from_coords(
        device.coords,
        ring_radius=device.ring_radius,
        w_wg=device.w_wg,
        z_wg=device.z_wg,
        h_wg=device.h_wg,
    )
    ring_temperature = optical_temperature[mask]
    return {
        "temperature_min_K": float(optical_temperature.min()),
        "temperature_max_K": float(optical_temperature.max()),
        "ring_mean_temperature_K": float(ring_temperature.mean()),
        "ring_max_temperature_K": float(ring_temperature.max()),
    }


def _plot_spectra(output_dir: Path, wavelengths, spectra):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(wavelengths, spectra["through_off"], label="0 mA")
    ax[0].plot(wavelengths, spectra["through_on"], label="15 mA")
    ax[0].set_title("Through port")
    ax[0].set_xlabel("Wavelength (um)")
    ax[0].set_ylabel("Transmission")
    ax[0].set_ylim([0, 1.05])
    ax[0].grid(True, alpha=0.25)
    ax[0].legend()

    ax[1].plot(wavelengths, spectra["drop_off"], label="0 mA")
    ax[1].plot(wavelengths, spectra["drop_on"], label="15 mA")
    ax[1].set_title("Drop port")
    ax[1].set_xlabel("Wavelength (um)")
    ax[1].set_ylabel("Transmission")
    ax[1].set_ylim([0, 1.05])
    ax[1].grid(True, alpha=0.25)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "spectra.png", dpi=180)
    plt.close(fig)


def plot_device_maps(
    device,
    plot_root,
    map_names=("epsilon", "conductivity", "thermo_optic_coeff"),
    filename="device_maps.jpg",
    cmap="Greys",
    value_mode="real",
    dpi=300,
    overlay_monitors=True,
    overlay_pml=True,
    overlay_heat_sources=True,
):
    """
    Plot selected device maps in rows.

    Rows:
        each map in `map_names`

    Columns:
        0: z = 0
        1: y = 0
        2: x = 0
        3: z = device.z_heater

    Parameters
    ----------
    device:
        Device object with plot_property(...) method.
    plot_root:
        Directory to save the figure.
    map_names:
        List/tuple of property names to plot, e.g.
        ["epsilon_map", "conductivity_map", "thermo_optic_coeff_map"].
    filename:
        Output image filename.
    """

    os.makedirs(plot_root, exist_ok=True)

    map_names = list(map_names)
    n_rows = len(map_names)
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 6 * n_rows),
        squeeze=False,
    )

    slices = [
        dict(z=0, title_suffix="z=0"),
        dict(y=0, title_suffix="y=0"),
        dict(x=0, title_suffix="x=0"),
        dict(z=device.z_heater, title_suffix=f"heater z={device.z_heater:.3f}"),
    ]

    for row, map_name in enumerate(map_names):
        for col, slice_info in enumerate(slices):
            ax = axes[row, col]

            title_suffix = slice_info.pop("title_suffix")
            slice_kwargs = dict(slice_info)

            device.plot_property(
                map_name,
                cmap=cmap,
                value_mode=value_mode,
                ax=ax,
                overlay_monitors=overlay_monitors,
                overlay_pml=overlay_pml,
                overlay_heat_sources=overlay_heat_sources,
                title=f"{map_name} slice at {title_suffix}",
                **slice_kwargs,
            )

            # Restore title_suffix because pop mutates the dict.
            slice_info["title_suffix"] = title_suffix

    plt.tight_layout()

    save_path = os.path.join(plot_root, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return save_path


def run_example(gpu_id: int = 0):
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device(f"cuda:{gpu_id}")
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(41)

    wl_cen = 1.55
    resolution = 0
    n_wl = 1
    resolution = 40
    heater_current = 15e-3
    mode = "Hz1"
    exp_name = "mrr_3d_nonuniform"

    sim_cfg = _build_sim_cfg(
        n_wl=n_wl,
        resolution=resolution,
        plot_root=f"./figs/{exp_name}",
    )
    wavelengths = np.linspace(
        sim_cfg["wl_cen"] - sim_cfg["wl_width"] / 2.0,
        sim_cfg["wl_cen"] + sim_cfg["wl_width"] / 2.0,
        sim_cfg["n_wl"],
    )

    device = ThermoOpticMRR(
        material_r1="Si",
        material_r2="Si",
        material_bg="SiO2",
        sim_cfg=sim_cfg,
        ring_radius=5.0,
        ring_width=0.5,
        ring_height=0.22,
        couple_gap=0.1,
        h_clad=2.8,
        h_box=2.0,
        h_wafer=0.5,
        port_len=(12.2, 12.2),
        port_width=(0.5, 0.5),
        port_height=(0.22, 0.22),
        heater_material="TiN",
        heater_gap=2.0,
        heater_height=0.14,
        heater_width=2.0,
        mode=mode,
        is_3d=True,
        heater_on_current=heater_current,
        device=operation_device,
    )
    hr_device = device.copy(resolution=resolution)

    opt = BaseOptimization(
        device=device,
        hr_device=hr_device,
        design_region_param_cfgs={},
        sim_cfg=sim_cfg,
        obj_cfgs=_build_objective_cfgs(wavelengths, heater_current, mode),
        operation_device=operation_device,
        verbose=True,
    ).to(operation_device)

    plot_device_maps(
        device,
        sim_cfg["plot_root"],
        map_names=[
            "epsilon",
            "conductivity",
            "thermo_optic_coeff",
        ],
    )

    # fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    # device.plot_eps(
    #     z=0,
    #     cmap="Greys",
    #     value_mode="real",
    #     ax=axes[0],
    #     overlay_monitors=False,
    #     overlay_pml=True,
    #     overlay_heat_sources=True,
    #     title="epsilon slice at z=0",
    # )
    # device.plot_eps(
    #     z=device.z_heater,
    #     cmap="Greys",
    #     ax=axes[1],
    #     overlay_monitors=False,
    #     overlay_pml=True,
    #     overlay_heat_sources=True,
    #     title=f"epsilon slice at heater z={device.z_heater:.3f}",
    # )
    # plt.tight_layout()
    # import os
    # os.makedirs(sim_cfg["plot_root"], exist_ok=True)
    # plt.savefig(os.path.join(sim_cfg["plot_root"], "device.jpg"), dpi=300)
    # exit(0)

    results = opt.forward(sharpness=1)

    pol = "Ey"
    heat_source_names = ("heater_1",)
    off_control_key = canonicalize_currents(
        {"heater_1": 0.0}, heat_source_names=heat_source_names
    )
    on_control_key = canonicalize_currents(
        {"heater_1": float(round(heater_current, 15))},
        heat_source_names=heat_source_names,
    )

    plot_kwargs = dict(
        obj=results["breakdown"]["through_off"]["value"],
        plot_filename=f"test_field_{pol}_off.jpg",
        field_key=(("in_slice_1", wl_cen, mode, off_control_key)),
        field_component="Ey",
        in_slice_name="in_slice_1",
        exclude_slice_names=[],
        thermal_map_name="optical_temperature",
        show_delta_eps=True,
        eps_grad=False,
        param_grad=False,
    )
    opt.plot(**plot_kwargs)

    plot_kwargs = dict(
        obj=results["breakdown"]["through_on"]["value"],
        plot_filename=f"test_field_{pol}_on.jpg",
        field_key=(("in_slice_1", wl_cen, mode, on_control_key)),
        field_component="Ey",
        in_slice_name="in_slice_1",
        exclude_slice_names=[],
        thermal_map_name="optical_temperature",
        show_delta_eps=True,
        eps_grad=False,
        param_grad=False,
    )
    opt.plot(**plot_kwargs)

    # del results

    # breakdown = opt.objective.breakdown
    # spectra = {
    #     "through_off": _to_numpy(breakdown["through_off"]["value"]).astype(float),
    #     "drop_off": _to_numpy(breakdown["drop_off"]["value"]).astype(float),
    #     "through_on": _to_numpy(breakdown["through_on"]["value"]).astype(float),
    #     "drop_on": _to_numpy(breakdown["drop_on"]["value"]).astype(float),
    # }

    # through_min_off_idx = int(np.argmin(spectra["through_off"]))
    # through_min_on_idx = int(np.argmin(spectra["through_on"]))
    # drop_max_off_idx = int(np.argmax(spectra["drop_off"]))
    # drop_max_on_idx = int(np.argmax(spectra["drop_on"]))

    # thermal_states = opt.objective.runtime_thermal_states
    # off_key = (("heater_1", 0.0),)
    # on_key = (("heater_1", float(round(heater_current, 15))),)
    # thermal_summary = {
    #     "off": _summarize_runtime_state(device, thermal_states[off_key]),
    #     "on": _summarize_runtime_state(device, thermal_states[on_key]),
    # }


if __name__ == "__main__":
    run_example(gpu_id=0)
