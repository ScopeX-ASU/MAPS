"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
import numpy as np
import torch
from pyutils.config import Config
from pyutils.general import logger

from core.invdes.invdesign import InvDesign
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.crossing import CrossingOptimization
from core.invdes.models.four_port_coupler import FourPortOptimization
from core.invdes.models.layers.crossing import Crossing
from core.invdes.models.layers.four_port_device import FourPortCoupler
from core.invdes.run_management import OptimizationRunManager
from core.utils import set_torch_deterministic

sys.path.pop(0)


def generate_coupler4_3d(gpu_id, mfs):
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    splitter1_3_region_size = (5.0, 5.0, 0.22)
    port_len = 1.5
    pol = "Hz"
    wl_cen = 1.550
    n_wl = 1
    wl_width = 0.0
    # n_wl = 3
    # wl_width = 0.02
    wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)
    resolution = 100

    input_port_width = 0.5
    output_port_width = 0.5

    sidewall_angle = 0
    # Both port and extrusion angles use the same signed degree convention.

    exp_name = f"coupler4_1550_3d_opt_mfs{mfs * 1000:.0f}_{splitter1_3_region_size[0]:.1f}x{splitter1_3_region_size[1]:.1f}_nosym"
    work_root = "./runs"
    resume_checkpoint = os.environ.get("MAPS_RESUME_CHECKPOINT")

    run_manager = OptimizationRunManager(
        root=work_root,
        experiment="coupler4_3d_1550",
        tag=exp_name,
        resume=resume_checkpoint is not None,
        resume_from=resume_checkpoint,
    )
    run_manager.attach_logger(logger)

    sim_cfg.update(
        dict(
            solver="fdtdx",
            border_width=[
                0,
                0,
                0,
                0,
                1.3,
                1.3,
            ],  # left, right, lower, upper, containing PML
            # symmetry=(0, -1, 0),  # PEC on y direction.
            max_time=6e-13,
            subpixel=False,
            resolution=resolution,
            optical_grid={
                "mode": "fdtdx_rectilinear_nonuniform",
                "wavelength": wl_cen,
                "grid_x": {"type": "auto", "min_steps_per_wvl": 20},
                "grid_y": {"type": "auto", "min_steps_per_wvl": 20},
                "grid_z": {"type": "auto", "min_steps_per_wvl": 15},
            },
            PML=[wl_cen / 2, wl_cen / 2, wl_cen / 2 - 0.2],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
        )
    )
    run_manager.bind_simulation_config(sim_cfg)

    def fom_func(breakdown):
        input_power = breakdown["input_trans"]["value"].detach().abs().square()
        breakdown["input_trans"]["value"] = input_power.item()
        weights = torch.tensor(
            [0.4 / (3e-2), 0.1 / (0.15), 0.1 / (7e-3), 0.0], device=input_power.device
        )[..., None]
        weights = weights / weights.sum()
        for key, obj in breakdown.items():
            if key == "smatrix":
                s_matrix = obj["value"].reshape(3, n_wl)  # shape (num_outports, n_wl)
                # print(s_matrix.shape, s_matrix)
                s_matrix = torch.concat(
                    [s_matrix, s_matrix[2:3, ...].data], dim=0
                )  # [4, n_wl]
                breakdown["smatrix"]["value"] = (
                    (s_matrix / input_power**0.5).detach().abs().square().view(-1)
                )
                power = (torch.abs(s_matrix) ** 2) / input_power
                # maximize mean power and penalize std to enforce equal split
                mean_p = torch.mean(power, 0)
                std_p = torch.std(power, 0)
                fom = (mean_p - 0.5 * std_p).mean()
                # fom = -(power - 0.25).pow(2).mean()
                # fom = -(power - 0.25).pow(2).mul(weights).sum()
                # grad: out_slice_1: 3e-2
                # grad: out_slice_2: 7e-3
                # grad: refl_slice_1: 1.5e-1
                # fom = -(power - 0.245).mul(weights).pow(2).sum()
                # fom = power.mul(weights).sum()
                # fom = power[0] * weights[0] + power[2] * weights[2]
                # fom = power[0] * weights[0] + power[2] * weights[2] + power[1] * weights[1]

            if "energy_constraint" in key:
                fom += obj["weight"] * obj["value"]

        return fom, {
            "sum_T": {"weight": 1.0, "value": mean_p.data.mean().data},
            "std_T": {"weight": -0.5, "value": -std_p.data.mean().data},
        }

    obj_cfgs = dict(_fusion_func=fom_func)

    device = Crossing(
        material_r1=3.4811**2,
        material_r2=1.4491**2,
        material_bg=1.4491**2,
        sim_cfg=sim_cfg,
        box_size=splitter1_3_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        port_height=(splitter1_3_region_size[2], splitter1_3_region_size[2]),
        device=operation_device,
        port_sidewall_angle=sidewall_angle,
        mode=f"{pol}1",
        is_3d=True,
    )
    hr_device = device.copy(resolution=resolution, optical_grid=None)
    print(device)

    design_region_param_cfgs = dict()
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[50, 50],
            sigma=2 / 50,
            transform=[
                dict(type="mirror_symmetry", dims=[1]),
                # dict(type="transpose_symmetry", rot_k=0),
                # dict(type="mirror_symmetry", dims=[0, 1]),
                dict(
                    type="blur",
                    mfs=mfs,
                    resolutions=[hr_device.resolution, hr_device.resolution],
                    dim="xy",
                ),
                dict(type="binarize"),
            ],
            init_method="constant_0",
            # init_method="diamond_0.3",
            denorm_mode="linear_eps",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
            dims=(0, 1),
            reference_plane="middle",
            extrude_angle=sidewall_angle,
        )

    obj_cfgs = dict(
        smatrix=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name=["in_slice_1"],
            out_slice_name=["out_slice_1", "refl_slice_1", "top_slice"],
            # out_slice_name=["out_slice_1", "out_slice_2", "out_slice_3"],
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,  #
            temp=[300],
            out_modes=(
                f"{pol}1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="smatrix",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction=["x+", "x-", "y+"],
            # direction=["x+", "y+", "y-"],
        ),
        _fusion_func=fom_func,
        override=True,
    )

    obj_cfgs[f"input_trans"] = dict(
        weight=1,
        #### objective is evaluated at this port
        in_slice_name="in_slice_1",
        out_slice_name=f"refl_slice_1",
        #### objective is evaluated at all points by sweeping the wavelength and modes
        in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
        wl=wls,  #
        temp=[300],
        out_modes=(
            f"{pol}1",
        ),  # can evaluate on multiple output modes and get average transmission
        type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
        energy=False,
        direction="x+",
        requires_grad=False,
        requires_adjoint=False,
    )
    obj_cfgs[f"energy_constraint"] = dict(
        # weight=-400,
        weight=-100,
        # weight=0,
        #### objective is evaluated at this port
        in_slice_name=f"in_slice_1",
        out_slice_name="out_slice_1",
        #### objective is evaluated at all points by sweeping the wavelength and modes
        wl=[wl_cen],
        temp=[300],
        in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
        out_modes=(
            f"{pol}1",
        ),  # can evaluate on multiple output modes and get average transmission
        type="energy_constraint",
        requires_grad=True,
        requires_adjoint=False,
    )

    opt = CrossingOptimization(
        device=device,
        hr_device=hr_device,
        design_region_param_cfgs=design_region_param_cfgs,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)

    invdesign = InvDesign(
        devOptimization=opt,
        run_manager=run_manager,
        optimizer=Config(
            name="Adam",
            # lr=0.3,
            lr=0.1,
            init_v=1e-5,
        ),
        lr_scheduler=Config(
            name="cosine",
            # lr_min=0.15,
            lr_min=0.05,
        ),
        sharp_scheduler=Config(
            mode="cosine",
            name="sharpness",
            init_sharp=32,
            final_sharp=32,
        ),
        run=Config(
            start_epoch=0,
            n_epochs=50,
            norm_grad=None,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=2,
            plot_name=f"{exp_name}",
            objs=["sum_T"],
            field_keys=[
                ("in_slice_1", wl_cen, f"{pol}1", 300),
            ],
            in_slice_names=["in_slice_1"],
            filename_suffixes=["s1"],
            field_component="Hz",
            exclude_port_names=["bot_port"],
            eps_grad=True,
            param_grad=True,
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
            upsample_eps_to_1nm=True,
            load_ckpt=resume_checkpoint,
            resume=resume_checkpoint is not None,
        ),
    )
    invdesign.optimize()


if __name__ == "__main__":
    gpu_id = 0
    # for mfs in [0.07, 0.09, 0.11, 0.13, 0.15]:
    for mfs in [0.1]:
        generate_coupler4_3d(gpu_id, mfs)
