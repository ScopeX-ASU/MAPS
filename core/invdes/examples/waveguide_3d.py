"""Run the source normalization for a straight 3D waveguide.

This example intentionally contains no design region or optimization. It is
useful for checking the eigenmode source coefficient used to normalize source
power in a device run.
"""

import os
import sys
from functools import partial

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

import torch
from pyutils.config import Config
from pyutils.general import logger

from core.invdes.invdesign import InvDesign
from core.invdes.models.base_optimization import BaseOptimization
from core.invdes.models.layers.device_base import N_Ports
from core.invdes.run_management import OptimizationRunManager
from core.utils import material_fn_dict

sys.path.pop(0)


class StraightWaveguide3D(N_Ports):
    """A straight two-port waveguide used only for normalization."""

    def __init__(
        self,
        sim_cfg,
        waveguide_length=1.0,
        port_length=1.5,
        waveguide_width=0.5,
        thickness=0.22,
        material="Si",
        background_material="SiO2",
        mode="Hz1",
        sidewall_angle: float = 0.0,
        device=torch.device("cuda:0"),
    ):
        self.mode = mode
        grid_step = 1 / sim_cfg["resolution"]

        if isinstance(material, str):
            eps_waveguide_fn = material_fn_dict[material]
            if "_eff" in material:
                eps_waveguide_fn = partial(eps_waveguide_fn, thickness=thickness)
        else:
            eps_waveguide_fn = lambda wl: material
        eps_waveguide = eps_waveguide_fn(sim_cfg["wl_cen"])

        if isinstance(background_material, str):
            eps_background_fn = material_fn_dict[background_material]
            if "_eff" in background_material:
                eps_background_fn = partial(eps_background_fn, thickness=thickness)
        else:
            eps_background_fn = lambda wl: background_material
        eps_background = eps_background_fn(sim_cfg["wl_cen"])

        half_length = waveguide_length / 2

        port_cfgs = dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[0, 0, 0],
                size=[waveguide_length + port_length * 2, waveguide_width, thickness],
                eps=eps_waveguide,
                sidewall_angle=sidewall_angle,
            ),
            out_port_1=dict(
                type="box",
                direction="x",
                center=[0, 0, 0],
                size=[waveguide_length + port_length * 2, waveguide_width, thickness],
                eps=eps_waveguide,
                sidewall_angle=sidewall_angle,
            ),
        )

        super().__init__(
            eps_bg=eps_background,
            sim_cfg=sim_cfg,
            port_cfgs=port_cfgs,
            geometry_cfgs=dict(),
            design_region_cfgs=dict(),
            device=device,
        )

    def init_monitors(self, verbose=True):
        """Create source, source-power, and output monitor slices."""
        print("Initializing source and output monitor slices...", flush=True)
        port_length = self.port_cfgs["in_port_1"]["size"][0]
        offset = 0.5 + self.sim_cfg["PML"][0]
        rel_loc = offset / port_length
        border_width_y = self.sim_cfg["border_width"][2]
        border_width_z = self.sim_cfg["border_width"][4]

        source_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_slice_1",
            rel_loc=rel_loc,
            rel_width=(
                self.port_cfgs["in_port_1"]["size"][1]
                + 2 * border_width_y
                - self.sim_cfg["PML"][1] * 2
            )
            / self.port_cfgs["in_port_1"]["size"][1],
            rel_height=(
                self.port_cfgs["in_port_1"]["size"][2]
                + 2 * border_width_z
                - self.sim_cfg["PML"][2] * 2
            )
            / self.port_cfgs["in_port_1"]["size"][2],
        )
        output_slice = self.build_port_monitor_slice(
            port_name="out_port_1",
            slice_name="out_slice_1",
            rel_loc=1 - rel_loc,
            rel_width=(
                self.port_cfgs["out_port_1"]["size"][1]
                + 2 * border_width_y
                - self.sim_cfg["PML"][1] * 2
            )
            / self.port_cfgs["out_port_1"]["size"][1],
            rel_height=(
                self.port_cfgs["out_port_1"]["size"][2]
                + 2 * border_width_z
                - self.sim_cfg["PML"][2] * 2
            )
            / self.port_cfgs["out_port_1"]["size"][2],
        )
        self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="source_power_slice",
            rel_loc=(offset + 0.3) / port_length,
            rel_width=(
                self.port_cfgs["in_port_1"]["size"][1]
                + 2 * border_width_y
                - self.sim_cfg["PML"][1] * 2
            )
            / self.port_cfgs["in_port_1"]["size"][1],
            rel_height=(
                self.port_cfgs["in_port_1"]["size"][2]
                + 2 * border_width_z
                - self.sim_cfg["PML"][2] * 2
            )
            / self.port_cfgs["in_port_1"]["size"][2],
        )
        return source_slice, output_slice

    def norm_run(self, verbose=True):
        print("Running normalization run...", flush=True)
        """Register the eigenmode source and the output mode monitor."""
        source_profiles = self.build_norm_sources(
            source_modes=(self.mode,),
            input_port_name="in_port_1",
            input_slice_name="in_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=True,
        )
        output_profiles = self.build_norm_sources(
            source_modes=(self.mode,),
            input_port_name="out_port_1",
            input_slice_name="out_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )
        source_power_profiles = self.build_norm_sources(
            source_modes=(self.mode,),
            input_port_name="in_port_1",
            input_slice_name="source_power_slice",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )
        print("Done normalization run...")
        return source_profiles, source_power_profiles, output_profiles


class StraightWaveguideOptimization(BaseOptimization):
    """Optimization wrapper that only performs a forward simulation."""

    def __init__(self, device, sim_cfg, obj_cfgs, operation_device):
        super().__init__(
            device=device,
            hr_device=device,
            design_region_param_cfgs={},
            sim_cfg=sim_cfg,
            obj_cfgs=obj_cfgs,
            operation_device=operation_device,
        )
        # InvDesign needs an optimizer parameter even though this run has no update.
        self.forward_only_parameter = torch.nn.Parameter(
            torch.zeros(1, device=operation_device)
        )


def main():
    wl_cen = 1.310
    sidewall_angle = 0
    exp_name = "waveguide_3d_normalization"
    work_root = "./runs"
    run_manager = OptimizationRunManager(
        root=work_root,
        experiment="waveguide_3d",
        tag=exp_name,
    )
    run_manager.attach_logger(logger)
    sim_cfg = dict(
        cell_size="None",
        # symmetry=(0, -1, 0),
        solver="fdtdx",
        resolution=50,
        subpixel=False,
        wl_cen=wl_cen,
        wl_width=0.0,
        n_wl=1,
        max_time=3e-13,
        border_width=[0, 0, 1.5, 1.5, 1.3, 1.3],
        PML=[wl_cen / 2 + 0.2] * 3,
        plot_root=f"./figs/{exp_name}/",
        optical_grid={
            "mode": "fdtdx_rectilinear_nonuniform",
            "wavelength": wl_cen,
            "grid_x": {"type": "auto", "min_steps_per_wvl": 25},
            "grid_y": {"type": "auto", "min_steps_per_wvl": 25},
            "grid_z": {"type": "auto", "min_steps_per_wvl": 25},
        },
    )
    run_manager.bind_simulation_config(sim_cfg)
    device = StraightWaveguide3D(
        # material=1.9809**2,
        material="Si",
        # background_material=1.4519**2,
        background_material="SiO2",
        # thickness=0.3,
        thickness=0.22,
        # waveguide_width=0.8,
        waveguide_width=0.5,
        waveguide_length=1.0,
        sim_cfg=sim_cfg,
        sidewall_angle=sidewall_angle,
        mode="Hz1",
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    mode = "Hz1"
    obj_cfgs = dict(
        source_power=dict(
            weight=1,
            in_slice_name="in_slice_1",
            out_slice_name="source_power_slice",
            in_mode=mode,
            wl=[wl_cen],
            temp=[300],
            out_modes=(mode,),
            type="eigenmode",
            direction="x+",
            requires_grad=False,
            requires_adjoint=False,
        ),
        output_power=dict(
            weight=0,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            in_mode=mode,
            wl=[wl_cen],
            temp=[300],
            out_modes=(mode,),
            type="eigenmode",
            direction="x+",
            requires_grad=False,
            requires_adjoint=False,
        ),
        output_flux=dict(
            weight=0,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            in_mode=mode,
            wl=[wl_cen],
            temp=[300],
            out_modes=(mode,),
            type="flux",
            direction="x",
            requires_grad=False,
            requires_adjoint=False,
        ),
        _fusion_func=lambda breakdown: (
            breakdown["source_power"]["value"],
            {},
        ),
        override=True,
    )

    operation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimization = StraightWaveguideOptimization(
        device=device,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(
        devOptimization=optimization,
        run_manager=run_manager,
        optimizer=Config(name="Adam", lr=1e-2),
        run=Config(start_epoch=0, n_epochs=1, backward=False),
        plot_cfgs=Config(
            plot=True,
            interval=1,
            plot_name=f"{exp_name}",
            objs=["output_power"],
            field_keys=[("in_slice_1", wl_cen, mode, 300)],
            in_slice_names=["in_slice_1"],
            exclude_slice_names=[],
            field_component="Ey",
            eps_grad=True,
            param_grad=True,
        ),
        checkpoint_cfgs=Config(save_model=False, dump_gds=False),
        debug_cfgs=Config(clear_step_results=False),
    )
    invdesign.optimize()
    breakdown = invdesign.results["breakdown"]
    source_power = breakdown["source_power"]["value"]
    output_power = breakdown["output_power"]["value"]
    print(f"source normalized eigenmode power: {source_power.item():.12g}")
    print(f"output normalized eigenmode power: {output_power.item():.12g}")
    print(f"source eigenmode amplitude coefficient: {source_power.sqrt().item():.12g}")
    print(f"output eigenmode amplitude coefficient: {output_power.sqrt().item():.12g}")
    print("Source field monitor: in_slice_1_field_monitor")
    print("Source coefficient template: source_power_slice")
    print("Output mode monitor: out_slice_1")

    source_power_slice = device.port_monitor_slices_info["source_power_slice"]
    out_slice_1 = device.port_monitor_slices_info["out_slice_1"]
    distance = out_slice_1["center"][0] - source_power_slice["center"][0]
    ## dB/cm conversion
    insertion_loss = -10 * torch.log10(output_power / source_power)
    insertion_loss_per_cm = insertion_loss / (distance * 1e-6) * 1e-2
    print(
        f"Insertion loss: {insertion_loss.item():.4f} dB, Insertion loss per cm: {insertion_loss_per_cm.item():.4f} dB/cm"
    )


if __name__ == "__main__":
    main()
