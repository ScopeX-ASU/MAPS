"""
Date: 2025-01-04 20:49:15
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-16 14:31:07
FilePath: /MAPS/core/invdes/examples/bending.py
"""

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
import torch
from pyutils.config import Config
from pyutils.torch_train import load_model

from core.invdes.invdesign import InvDesign
from core.invdes.models import BendingOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import Bending
from core.utils import interface_field_penalty, set_torch_deterministic

sys.path.pop(0)

if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    thickness = 0.22
    bending_region_size = (3, 3, thickness)
    port_len = 1.5

    input_port_width = 0.5
    output_port_width = 0.5

    exp_name = "bending_3d_nonuniform"
    mode = "Hz1"
    resolution = 50
    wl_cen = 1.55
    wl_width = 0.00
    n_wl = 1

    # wl_width = 0.00
    # n_wl = 1

    wls = torch.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl).tolist()

    sim_cfg.update(
        dict(
            solver="fdtdx",
            border_width=[
                0,
                port_len / 1.5,
                port_len / 1.5,
                0,
                1.4,
                1.4,
            ],
            max_time=2e-13,
            subpixel=False,
            resolution=resolution,
            optical_grid={
                "mode": "fdtdx_rectilinear_nonuniform",
                "wavelength": wl_cen,
                "grid_x": {"type": "auto", "min_steps_per_wvl": 20},
                "grid_y": {"type": "auto", "min_steps_per_wvl": 20},
                "grid_z": {"type": "auto", "min_steps_per_wvl": 20},
            },
            plot_root=f"./figs/{exp_name}",
            PML=[wl_cen / 2, wl_cen / 2, wl_cen / 2],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
        )
    )

    device = Bending(
        material_r1="Si",
        sim_cfg=sim_cfg,
        bending_region_size=bending_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        port_height=(thickness, thickness),
        device=operation_device,
        is_3d=True,
        mode=mode,
    )

    # hr_device = device.copy(resolution=310)
    hr_device = device.copy(resolution=50)
    print(device)

    design_region_param_cfgs = dict()
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[50, 50],
            sigma=2 / 50,  # 2 * levelset knot grid step
            transform=[
                dict(type="transpose_symmetry", rot_k=3),
                # dict(
                #     type="blur",
                #     mfs=0.05,
                #     resolutions=[hr_device.resolution, hr_device.resolution],
                #     dim="xy",
                # ),
                dict(type="binarize"),
            ],
            # init_method="bending_0.5",
            init_method="constant_0",
            # init_method="constant_0.5",
            # denorm_mode="linear_eps",
            denorm_mode="inverse_eps",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
            dims=(0, 1),
            extrude_direction="-",
            extrude_angle=80.0,
        )

    obj_cfgs = dict(
        fwd_trans=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,  # broadband monitor
            temp=[300],
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            # type="flux",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction="y+",
        ),
        ## input power monitor is needed to normalize the transmission, as fdtd didn't run normalization run.
        input_trans=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="refl_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,
            temp=[300],
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            # type="flux",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction="x+",
            requires_grad=False,  # only used for normalization, no adjoint run is needed.
        ),
        energy_penalty=dict(
            weight=-100,  # in 3D, grad is only through epsilon, so weights need to be large.
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,
            temp=[300],
            type="energy_constraint",
            requires_grad=True,  # grad is needed, to avoid it from being detached.
            requires_adjoint=False,  # grad is not related to fields, thus no adjoint run is needed.
        ),
        override=True,
    )

    def fom_func(breakdown):
        trans = (
            breakdown["fwd_trans"]["value"] / breakdown["input_trans"]["value"].detach()
        )
        spectrum = {}
        fom = 0
        for wl, s in zip(wls, trans):
            spectrum[f"|s21|^2 at {wl:.2f}"] = {"weight": 1, "value": s.item()}
            fom = fom + s
        penalty = breakdown["energy_penalty"]["value"]
        fom = fom + penalty * breakdown["energy_penalty"]["weight"]
        spectrum["avg|s21|^2"] = {"weight": 1, "value": trans.mean().item()}
        return fom, spectrum | {
            "energy_penalty": {
                "weight": breakdown["energy_penalty"]["weight"],
                "value": penalty.item(),
            }
        }

    obj_cfgs["_fusion_func"] = fom_func

    opt = BendingOptimization(
        device=device,
        hr_device=hr_device,
        design_region_param_cfgs=design_region_param_cfgs,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
        obj_cfgs=obj_cfgs,
    ).to(operation_device)

    def call_back(invdesign):
        for region_name in device.design_region_cfgs.keys():
            rho = invdesign.devOptimization.design_region_param_dict[region_name]
            with torch.no_grad():
                grad = rho.weights["ls_knots"].grad.abs()
                threshold = 0.000 * grad.max()
                rho.weights["ls_knots"].data = torch.where(
                    grad < threshold,
                    torch.clamp(rho.weights["ls_knots"].data - 0.2, -0.5, 0.5),
                    rho.weights["ls_knots"].data,
                )

    invdesign = InvDesign(
        devOptimization=opt,
        optimizer=dict(
            name="Adam",
            lr=0.2,
            weight_decay=0,
        ),
        run=Config(
            start_epoch=0,
            n_epochs=20,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=0.2,
        ),
        sharp_scheduler=Config(
            mode="linear",
            name="sharpness",
            init_sharp=4,
            final_sharp=20,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=1,
            plot_name=f"{exp_name}",
            objs=["avg|s21|^2"],
            field_keys=[("in_slice_1", wl_cen, mode, 300)],
            in_slice_names=["in_slice_1"],
            exclude_slice_names=[],
            field_component="Ey",
            eps_grad=True,
            param_grad=True,
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
            upsample_eps_to_1nm=True,
        ),
        # after_step_callbacks=[call_back],
    )
    invdesign.optimize()
