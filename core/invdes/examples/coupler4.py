"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
"""

import os
import sys
from copy import deepcopy

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
import csv
import math

import numpy as np
import torch
from pyutils.config import Config
from pyutils.general import logger

from core.invdes.invdesign import InvDesign
from core.invdes.models import CrossingOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import Crossing
from core.invdes.run_management import OptimizationRunManager
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
    test_sim_cfg = DefaultSimulationConfig()

    crossing_region_size = (5, 5)
    port_len = 1.5

    input_port_width = 0.5
    output_port_width = 0.5

    wl_cen = 1.55
    wl_width = 0.01
    n_wl = 3
    n_wl_test = 11
    wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)
    test_wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl_test)
    exp_comment = "eff_eps_tgt_24_MFS_100_nosym"
    # exp_comment = "test"
    exp_name = f"coupler4_port4_s{crossing_region_size[0]}x{crossing_region_size[1]}_c-{exp_comment}"
    work_root = "./runs"
    resume_checkpoint = os.environ.get("MAPS_RESUME_CHECKPOINT")
    mode = "Hz1"

    run_manager = OptimizationRunManager(
        root=work_root,
        experiment="coupler4",
        tag=exp_name,
        resume=resume_checkpoint is not None,
        resume_from=resume_checkpoint,
    )
    run_manager.attach_logger(logger)

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 0, 0],
            resolution=50,
            subpixel=False,
            symmetry=(0, -1, 0),
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
        )
    )
    run_manager.bind_simulation_config(sim_cfg)
    test_sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 0, 0],
            resolution=50,
            subpixel=False,
            symmetry=(0, -1, 0),
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl_test,
        )
    )
    run_manager.bind_simulation_config(test_sim_cfg)

    device = Crossing(
        material_r1=2.85334**2,
        material_r2=1.4491**2,
        material_bg=1.4491**2,
        sim_cfg=sim_cfg,
        box_size=crossing_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
        mode=mode,
    )

    hr_device = device.copy(resolution=100)

    test_device = Crossing(
        material_r1=2.85334**2,
        material_r2=1.4491**2,
        material_bg=1.4491**2,
        sim_cfg=test_sim_cfg,
        box_size=crossing_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
        mode=mode,
    )
    test_hr_device = test_device.copy(resolution=100)

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        # for _, obj in breakdown.items():
        #     fom = fom + obj["weight"] * obj["value"]
        # for name, obj in breakdown.items():
        # if "rad" in name:
        #     fom = fom + obj["weight"] * obj["value"]
        ## add extra contrast ratio
        target = 0.25**0.5
        balance = 0
        for wl in wls:
            balance += (
                (breakdown[f"fwd_trans_{wl}"]["value"] - target).abs() ** 2
                + (breakdown[f"refl_trans_{wl}"]["value"] - target).abs() ** 2
                + (breakdown[f"top_cross_talk_{wl}"]["value"] - target).abs() ** 2
                + (breakdown[f"bot_cross_talk_{wl}"]["value"] - target).abs() ** 2
            )
        balance = balance / len(wls)
        fom = fom - 30 * balance
        # balance = (breakdown["refl_trans"]["value"]/breakdown["fwd_trans"]["value"].data - 1)**2 + (breakdown["top_cross_talk"]["value"]/breakdown["fwd_trans"]["value"].data - 1)**2 + (breakdown["bot_cross_talk"]["value"]/breakdown["fwd_trans"]["value"].data - 1)**2
        # balance = torch.stack(
        #     [
        #         breakdown["fwd_trans"]["value"],
        #         breakdown["refl_trans"]["value"],
        #         breakdown["top_cross_talk"]["value"],
        #         breakdown["bot_cross_talk"]["value"],
        #     ]
        # )
        # balance = torch.nn.functional.kl_div(
        #     torch.tensor([0.25, 0.25, 0.25, 0.25], device=balance.device),
        #     torch.softmax(balance, dim=-1),
        #     reduction="sum",
        # )
        # balance = (breakdown["fwd_trans"]["value"] - 0.33)**2 + (breakdown["top_cross_talk"]["value"] - 0.33)**2 + (breakdown["bot_cross_talk"]["value"] - 0.33)**2 + (breakdown["refl_trans"]["value"])**2
        # fom = fom - 1 * balance

        for key, obj in breakdown.items():
            if "energy_penalty" in key:
                fom = fom + obj["weight"] * obj["value"]
        return fom, {"balance": {"weight": -10, "value": balance}}

    # wls = [1.55]
    obj_cfgs_base = dict(
        fwd_trans=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,
            temp=[300],
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction="x+",
            energy=False,
        ),
        refl_trans=dict(
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
            type="eigenmode",
            direction="x-",
            energy=False,
        ),
        top_cross_talk=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="top_slice",
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,
            temp=[300],
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",
            direction="y+",
            energy=False,
        ),
        bot_cross_talk=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="bot_slice",
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,
            temp=[300],
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",
            direction="y-",
            energy=False,
        ),
        rad_trans_xp=dict(
            weight=-0.1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="rad_slice_xp",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=wls,
            temp=[300],
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="flux",
            direction="x",
        ),
        rad_trans_xm=dict(
            weight=-0.1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="rad_slice_xm",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=wls,
            temp=[300],
            in_mode=mode,  # only one source mode is supsliceed, cannot input multiple modes at the same time
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="flux",
            direction="x",
        ),
        rad_trans_yp=dict(
            weight=-0.1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="rad_slice_yp",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=wls,
            temp=[300],
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="flux",
            direction="y",
        ),
        rad_trans_ym=dict(
            weight=-0.1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="rad_slice_ym",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=wls,
            temp=[300],
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="flux",
            direction="y",
        ),
        energy_penalty=dict(
            weight=-0.5,
            # weight=-5000,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=[wl_cen],
            temp=[300],
            in_mode=f"{mode}",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                f"{mode}",
            ),  # can evaluate on multiple output modes and get average transmission
            type="energy_constraint",
            requires_grad=True,
        ),
        _fusion_func=fom_func,
    )

    obj_cfgs = {"_fusion_func": obj_cfgs_base.pop("_fusion_func")}
    for wl in wls:
        for name, obj in obj_cfgs_base.items():
            obj_cfgs[f"{name}_{wl}"] = deepcopy(obj)
            obj_cfgs[f"{name}_{wl}"]["wl"] = [wl]
    # print(obj_cfgs)
    obj_cfgs["override"] = True

    test_obj_cfgs = dict(
        fwd_trans=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            wl=test_wls,
            temp=[300],
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction="x+",
        ),
        refl_trans=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="refl_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            wl=test_wls,
            temp=[300],
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",
            direction="x-",
        ),
        top_cross_talk=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="top_slice",
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            wl=test_wls,
            temp=[300],
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",
            direction="y+",
        ),
        bot_cross_talk=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="bot_slice",
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            wl=test_wls,
            temp=[300],
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",
            direction="y-",
        ),
        rad_trans_xp=dict(
            weight=-0.1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="rad_slice_xp",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=test_wls,
            temp=[300],
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="flux",
            direction="x",
        ),
        rad_trans_xm=dict(
            weight=-0.1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="rad_slice_xm",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=test_wls,
            temp=[300],
            in_mode=mode,  # only one source mode is supsliceed, cannot input multiple modes at the same time
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="flux",
            direction="x",
        ),
        rad_trans_yp=dict(
            weight=-0.1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="rad_slice_yp",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=test_wls,
            temp=[300],
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="flux",
            direction="y",
        ),
        rad_trans_ym=dict(
            weight=-0.1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="rad_slice_ym",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=test_wls,
            temp=[300],
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="flux",
            direction="y",
        ),
    )

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
                    # mfs=0.05,
                    mfs=0.1,
                    resolutions=[hr_device.resolution, hr_device.resolution],
                    dim="xy",
                ),
                # dict(type="fft", mfs=0.1, resolutions=[310, 310], dim="xy"),
                dict(type="binarize"),
                # dict(type="transpose_symmetry", flag=True),
            ],
            # init_method="ones",
            # init_method="checkerboard",
            # init_method="ball",
            init_method="diamond_0.3",
            # init_method="diamond_0.4",
            # init_method="constant_0.35",
            # init_method="crossing",
            denorm_mode="linear_eps",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
        )

    opt = CrossingOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        design_region_param_cfgs=design_region_param_cfgs,
        operation_device=operation_device,
    ).to(operation_device)

    invdesign = InvDesign(
        devOptimization=opt,
        run_manager=run_manager,
        run=Config(
            n_epochs=150,
        ),
        optimizer=Config(
            # name="lbfgs",
            # line_search_fn="strong_wolfe",
            name="Adam",
            # lr=1e-2,
            lr=0.02,
            weight_decay=0,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=0.01,
        ),
        sharp_scheduler=Config(
            mode="cosine",
            name="sharpness",
            init_sharp=1,
            final_sharp=32,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=5,
            plot_name=f"coupler4_{'port4'}_s{crossing_region_size[0]}x{crossing_region_size[1]}",
            objs=[f"fwd_trans_{wl_cen}"],
            field_keys=[
                ("in_slice_1", wl_cen, f"{mode}", 300),
            ],
            in_slice_names=["in_slice_1"],
            filename_suffixes=["s1"],
            exclude_port_names=[],
            eps_grad=True,
            param_grad=True,
        ),
        checkpoint_cfgs=Config(
            save_model=True,
            ckpt_name=f"coupler4_{'port4'}_s{crossing_region_size[0]}x{crossing_region_size[1]}_c-{exp_comment}",
            dump_gds=True,
            gds_name=f"coupler4_{'port4'}_s{crossing_region_size[0]}x{crossing_region_size[1]}",
            upsample_eps_to_1nm=True,
            load_ckpt=resume_checkpoint,
            resume=resume_checkpoint is not None,
        ),
    )
    invdesign.optimize()

    # invdesign = InvDesign(
    #     devOptimization=opt,
    #     optimizer=Config(
    #         # name="lbfgs",
    #         # line_search_fn="strong_wolfe",
    #         name="Adam",
    #         lr=1e-2,
    #         weight_decay=0,
    #     ),
    #     run=Config(
    #         n_epochs=100,
    #     ),
    # )
    # invdesign.optimize(
    #     plot=True,
    #     # plot_filename=f"coupler4_{'port3'}",
    #     plot_filename=f"coupler4_{'port4'}_s{crossing_region_size[0]}x{crossing_region_size[1]}",
    #     objs=[f"fwd_trans_{wl_cen}"],
    #     field_keys=[("in_slice_1", wl_cen, mode, 300)],
    #     in_slice_names=["in_slice_1"],
    #     exclude_slice_names=[],
    #     dump_gds=True,
    #     save_model=True,
    #     ckpt_name=f"coupler4_{'port4'}_s{crossing_region_size[0]}x{crossing_region_size[1]}_c-{exp_comment}",
    # )
    test_opt = CrossingOptimization(
        device=test_device,
        hr_device=test_hr_device,
        sim_cfg=test_sim_cfg,
        obj_cfgs=test_obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)
    test_ls_knots = {}
    for _, param in opt.named_parameters():
        test_ls_knots["crossing_region"] = param.data
    with torch.no_grad():
        _ = test_opt(
            sharpness=256,
            ls_knots=test_ls_knots,
        )
    fwd_trans_list = []
    refl_trans_list = []
    top_cross_talk_list = []
    bot_cross_talk_list = []
    s_params = test_opt.objective.s_params
    wl_list = []
    for (wl, pol), _ in test_opt.objective.sims.items():
        wl_list.append(wl)
    for wl in wl_list:
        fwd_trans_list.append(
            10
            * math.log10(
                s_params[("in_slice_1", "out_slice_1", mode, wl, mode, 300)][
                    "s_p"
                ].item()
            )
        )
        refl_trans_list.append(
            10
            * math.log10(
                s_params[("in_slice_1", "refl_slice_1", mode, wl, mode, 300)][
                    "s_m"
                ].item()
            )
        )
        top_cross_talk_list.append(
            10
            * math.log10(
                s_params[("in_slice_1", "top_slice", mode, wl, mode, 300)]["s_p"].item()
            )
        )
        bot_cross_talk_list.append(
            10
            * math.log10(
                s_params[("in_slice_1", "bot_slice", mode, wl, mode, 300)]["s_m"].item()
            )
        )
    # Store these lists to CSV file
    csv_filename = run_manager.path("metrics", "split_ratio.csv")
    with open(csv_filename, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(["Wavelength", "S31", "S11", "S21", "S41"])
        # Write the data row by row
        for i, wl in enumerate(wl_list):
            csv_writer.writerow(
                [
                    wl,
                    fwd_trans_list[i],
                    refl_trans_list[i],
                    top_cross_talk_list[i],
                    bot_cross_talk_list[i],
                ]
            )

    run_manager.register_artifact("split_ratio_csv", csv_filename, checksum=True)
    print(f"Data successfully written to {csv_filename}")
