"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu (jiaqigu@asu.edu)
LastEditTime: 2026-02-09 12:18:23
FilePath: /MAPS_ilt/core/invdes/examples/mdm_SiN.py
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

from core.invdes import builder
from core.invdes.invdesign import InvDesign
from core.invdes.models import MDMOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MDM
from core.utils import set_torch_deterministic

sys.path.pop(0)
if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    # mdm_region_size = (20, 9)
    wl_cen = 1.55
    wl_width = 0.0
    n_wl = 1
    thickness = 0.4
    mdm_region_size = (20, 9, thickness)
    # mdm_region_size = (16, 7, thickness)
    # mdm_region_size = (10, 4, thickness)
    port_len = 1.5
    pol = "Hz"

    input_port_width = 4
    # input_port_width = 2
    output_port_width = 0.85
    num_outports = 5
    # num_outports = 2
    # exp_name = f"mdm_3d_opt-port-{num_outports}_SiN_{mdm_region_size[0]}x{mdm_region_size[1]}x{mdm_region_size[2]}_adam"
    # exp_name = f"mdm_3d_opt-port-{num_outports}_SiN_{mdm_region_size[0]}x{mdm_region_size[1]}x{mdm_region_size[2]}_nesterov_bb_static_2d"
    exp_name = f"mdm_3d_opt-port-{num_outports}_SiN_{mdm_region_size[0]}x{mdm_region_size[1]}x{mdm_region_size[2]}_cg_prp_2d"
    # exp_name = f"mdm_3d_opt-port-{num_outports}_SiN_{mdm_region_size[0]}x{mdm_region_size[1]}x{mdm_region_size[2]}_lbfgs"

    sim_cfg.update(
        dict(
            solver="fdtdx",
            # border_width=[port_len, port_len, 2, 2],
            border_width=[0, 0, 1.1, 1.1, 0.9, 0.9],
            resolution=20,
            plot_root=f"./figs/{exp_name}",
            PML=[0.2, 0.2, 0.2],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
        )
    )

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        product = 0
        for i in range(1, num_outports + 1):
            trans = (
                breakdown[f"mode{i}_trans"]["value"]
                / (breakdown[f"mode{i}_input_trans"]["value"].detach())
            ).mean()
            for j in range(1, num_outports + 1):
                if j != i:
                    xtalk = (
                        breakdown[f"mode{i}_xtalk_p{j}"]["value"]
                        / (breakdown[f"mode{i}_input_trans"]["value"].detach())
                    ).mean()
                    fom = fom + breakdown[f"mode{i}_xtalk_p{j}"]["weight"] * xtalk
                    breakdown[f"mode{i}_xtalk_p{j}"]["value"] = xtalk.item()
            fom = fom - breakdown[f"mode{i}_trans"]["weight"] * (trans - 1) ** 4

            product = product + torch.log(trans + 1e-3)
            breakdown[f"mode{i}_trans"]["value"] = trans.item()

        ## if only sum all transmission, lbfgs cannot balance them well.

        ## do not use direct product! ill-conditioned!
        # for i in range(1, n_wl + 1):
        #     product = product * breakdown[f"wl{i}_trans"]["value"]

        ## this sum-of-log formulation is more numerically stable and also encourages all transmissions to be high (since log is dominated by the smallest value)
        # for i in range(1, num_outports + 1):
        #     product = product + torch.log(breakdown[f"mode{i}_trans"]["value"] + 1e-3)
        fom = fom  # + product  # * 10
        return fom, {"trans_product": {"weight": 1, "value": product}}

    device = MDM(
        material_r1="Si3N4",
        sim_cfg=sim_cfg,
        box_size=mdm_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        port_height=(thickness, thickness),
        num_outports=num_outports,
        port_box_margin=0.675,
        is_3d=True,
        pol=pol,
        device=operation_device,
    )

    hr_device = device.copy(resolution=20)
    print(device)

    obj_cfgs = {}
    wls = torch.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl).tolist()
    for i in range(1, num_outports + 1):
        obj_cfgs[f"mode{i}_trans"] = dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name=f"out_slice_{i}",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=wls,
            temp=[300],
            in_mode=f"{pol}{i}",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                f"{pol}1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",
            direction="x+",
        )
        for j in range(1, num_outports + 1):
            if j != i:
                obj_cfgs[f"mode{i}_xtalk_p{j}"] = dict(
                    weight=-0,
                    #### objective is evaluated at this port
                    in_slice_name="in_slice_1",
                    out_slice_name=f"out_slice_{j}",
                    #### objective is evaluated at all points by sweeping the wavelength and modes
                    wl=wls,
                    temp=[300],
                    in_mode=f"{pol}{i}",  # only one source mode is supported, cannot input multiple modes at the same time
                    out_modes=(
                        f"{pol}1",
                    ),  # can evaluate on multiple output modes and get average transmission
                    type="eigenmode",
                    direction="x+",
                )
        obj_cfgs[f"mode{i}_input_trans"] = dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name=f"refl_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=wls,
            temp=[300],
            in_mode=f"{pol}{i}",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                f"{pol}{i}",
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",
            direction="x+",
        )

    obj_cfgs["_fusion_func"] = fom_func
    obj_cfgs["override"] = True

    design_region_param_cfgs = dict()
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[20, 20],
            sigma=1 / 20,
            transform=[
                dict(
                    type="blur",
                    mfs=0.1,
                    resolutions=[hr_device.resolution, hr_device.resolution],
                    dim="xy",
                ),
                dict(type="binarize"),
            ],
            # init_method="constant_0.2",
            init_method="constant_0.499",
            # denorm_mode="inverse_eps",
            denorm_mode="linear_eps",
            # interpolation="bilinear",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
            dims=(0, 1),
            extrude_direction="-",
            extrude_angle=90.0,
        )

    opt = MDMOptimization(
        device=device,
        hr_device=hr_device,
        design_region_param_cfgs=design_region_param_cfgs,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(
        devOptimization=opt,
        run=Config(
            n_epochs=50,
            # norm_grad="abs_mean",
        ),
        optimizer=Config(
            # name="Adam",
            # lr=0.1,
            # init_v=1e-10,
            # name="lbfgs",
            # line_search_fn="strong_wolfe",
            # lr=0.1,
            # tolerance_change=1e-3,
            # name="nesterov",
            # lr=1,
            # alg="bb_static",
            # constraint_fn=lambda x: x.clamp_(-0.5, 0.5),
            # max_iter=1,
            # max_eval=1,
            # weight_decay=0,
            name="cg_prp",
            lr=10,
        ),
        lr_scheduler=Config(
            name="cosine",
            # lr_min=0.1,
            lr_min=0.05,
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
            objs=[f"mode{i}_trans" for i in range(1, num_outports + 1)],
            field_keys=[
                (f"in_slice_1", wl_cen, f"{pol}{i}", 300)
                for i in range(1, num_outports + 1)
            ],
            in_slice_names=["in_slice_1"] * num_outports,
            # exclude_slice_names=[["out_slice_2"], ["out_slice_1"]],
            exclude_slice_titles=[
                [f"out_slice_{i}"] for i in range(1, num_outports + 1)
            ],
            field_component="Ey",
            eps_grad=True,
            param_grad=True,
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
            resume=True,
            load_ckpt="/data/jiaqigu/projects/MAPS_fdtdx/figs/mdm_opt-port-5_SiN_neff1.7_20x9_lbfgs/mdm_opt-port-5_SiN_neff1.7_20x9_lbfgs_epoch-49_err-4.8937_epoch-49.pt",
        ),
    )
    invdesign.optimize()

    # invdesign = InvDesign(
    #     devOptimization=opt,
    #     run=Config(
    #         n_epochs=10,
    #         start_epoch=40,
    #         # norm_grad="abs_mean",
    #     ),
    #     optimizer=Config(
    #         # name="Adam",
    #         # lr=0.02,
    #         # init_v=1e-11,
    #         # name="lbfgs",
    #         # line_search_fn="strong_wolfe",
    #         # lr=2,
    #         # tolerance_change=1e-2,
    #         name="nesterov",
    #         lr=1,
    #         alg="bb_static",
    #         # max_iter=1,
    #         # max_eval=1,
    #         # weight_decay=0,
    #     ),
    #     lr_scheduler=Config(
    #         name="cosine",
    #         lr_min=0.1,
    #     ),
    #     sharp_scheduler=Config(
    #         mode="linear",
    #         name="sharpness",
    #         init_sharp=20,
    #         final_sharp=20,
    #     ),
    #     plot_cfgs=Config(
    #         plot=True,
    #         interval=1,
    #         plot_name=f"{exp_name}",
    #         objs=[f"mode{i}_trans" for i in range(1, num_outports + 1)],
    #         field_keys=[
    #             (f"in_slice_1", wl_cen, f"{pol}{i}", 300)
    #             for i in range(1, num_outports + 1)
    #         ],
    #         in_slice_names=["in_slice_1"] * num_outports,
    #         # exclude_slice_names=[["out_slice_2"], ["out_slice_1"]],
    #         exclude_slice_titles=[
    #             [f"out_slice_{i}"] for i in range(1, num_outports + 1)
    #         ],
    #         field_component="Ey",
    #         eps_grad=True,
    #         param_grad=True,
    #     ),
    #     checkpoint_cfgs=Config(
    #         save_model=False,
    #         ckpt_name=f"{exp_name}",
    #         dump_gds=True,
    #         gds_name=f"{exp_name}",
    #     ),
    # )
    # invdesign.optimize()
