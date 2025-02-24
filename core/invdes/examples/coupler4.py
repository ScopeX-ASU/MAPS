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
import numpy as np
import torch

from core.invdes.invdesign import InvDesign
from core.invdes.models import (
    CrossingOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import Crossing
from core.utils import set_torch_deterministic
from pyutils.config import Config
import csv
import math
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

    crossing_region_size = (8, 8)
    port_len = 2

    input_port_width = 0.48
    output_port_width = 0.48

    wl_cen = 1.55
    wl_width = 0.01
    n_wl = 3
    n_wl_test = 11
    wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)
    test_wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl_test)
    exp_comment = "eff_eps_tgt_24_MFS_100_adam"
    # exp_comment = "test"
    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 0, 0],
            resolution=50,
            # plot_root=f"./figs/coupler4_{'port3'}",
            plot_root=f"./figs/coupler4_{'port4'}_s{crossing_region_size[0]}x{crossing_region_size[1]}_c-{exp_comment}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
        )
    )
    test_sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 0, 0],
            resolution=50,
            plot_root=f"./figs/coupler4_{'port4'}_s{crossing_region_size[0]}x{crossing_region_size[1]}_c-{exp_comment}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl_test,
        )
    )

    device = Crossing(
        sim_cfg=sim_cfg,
        box_size=crossing_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )

    hr_device = device.copy(resolution=310)

    test_device = Crossing(
        sim_cfg=test_sim_cfg,
        box_size=crossing_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )
    test_hr_device = test_device.copy(resolution=310)

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        # for _, obj in breakdown.items():
        #     fom = fom + obj["weight"] * obj["value"]
        for name, obj in breakdown.items():
            if "rad" in name:
                fom = fom + obj["weight"] * obj["value"]
        ## add extra contrast ratio
        target = 0.24
        balance = 0
        for wl in wls:
            balance += (
                (breakdown[f"fwd_trans_{wl}"]["value"] - target) ** 2
                + (breakdown[f"refl_trans_{wl}"]["value"] - target) ** 2
                + (breakdown[f"top_cross_talk_{wl}"]["value"] - target) ** 2
                + (breakdown[f"bot_cross_talk_{wl}"]["value"] - target) ** 2
            )
        fom = fom - 10 * balance
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
        return fom, {"balance": {"weight": -10, "value": balance}}

    # wls = [1.55]
    obj_cfgs_base = dict(
        fwd_trans=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,
            temp=[300],
            out_modes=(
                "Ez1",
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
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,
            temp=[300],
            out_modes=(
                "Ez1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",
            direction="x-",
        ),
        top_cross_talk=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="top_slice",
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,
            temp=[300],
            out_modes=(
                "Ez1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",
            direction="y+",
        ),
        bot_cross_talk=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="bot_slice",
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,
            temp=[300],
            out_modes=(
                "Ez1",
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
            wl=wls,
            temp=[300],
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                "Ez1",
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
            in_mode="Ez1",  # only one source mode is supsliceed, cannot input multiple modes at the same time
            out_modes=(
                "Ez1",
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
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                "Ez1",
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
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                "Ez1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="flux",
            direction="y",
        ),
        _fusion_func=fom_func,
    )

    obj_cfgs = {"_fusion_func": obj_cfgs_base.pop("_fusion_func")}
    for wl in wls:
        for name, obj in obj_cfgs_base.items():
            obj_cfgs[f"{name}_{wl}"] = deepcopy(obj)
            obj_cfgs[f"{name}_{wl}"]["wl"] = [wl]
    print(obj_cfgs)

    test_obj_cfgs = dict(
        fwd_trans=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=test_wls,
            temp=[300],
            out_modes=(
                "Ez1",
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
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=test_wls,
            temp=[300],
            out_modes=(
                "Ez1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",
            direction="x-",
        ),
        top_cross_talk=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="top_slice",
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=test_wls,
            temp=[300],
            out_modes=(
                "Ez1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",
            direction="y+",
        ),
        bot_cross_talk=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="bot_slice",
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=test_wls,
            temp=[300],
            out_modes=(
                "Ez1",
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
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                "Ez1",
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
            in_mode="Ez1",  # only one source mode is supsliceed, cannot input multiple modes at the same time
            out_modes=(
                "Ez1",
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
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                "Ez1",
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
            in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                "Ez1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="flux",
            direction="y",
        ),
    )

    opt = CrossingOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(
        devOptimization=opt,
        optimizer=Config(
            # name="lbfgs",
            # line_search_fn="strong_wolfe",
            name="Adam",
            lr=1e-2,
            weight_decay=0,
        ),
        run=Config(
            n_epochs=100,
        ),
    )
    invdesign.optimize(
        plot=True,
        # plot_filename=f"coupler4_{'port3'}",
        plot_filename=f"coupler4_{'port4'}_s{crossing_region_size[0]}x{crossing_region_size[1]}",
        objs=[f"fwd_trans_{wl_cen}"],
        field_keys=[("in_slice_1", wl_cen, "Ez1", 300)],
        in_slice_names=["in_slice_1"],
        exclude_slice_names=[],
        dump_gds=True,
        save_model=True,
        ckpt_name=f"coupler4_{'port4'}_s{crossing_region_size[0]}x{crossing_region_size[1]}_c-{exp_comment}",
    )
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
        fwd_trans_list.append(10*math.log10(s_params[('in_slice_1', 'out_slice_1', 'Ez1', wl, 'Ez1', 300)]["s_p"].item()))
        refl_trans_list.append(10*math.log10(s_params[('in_slice_1', 'refl_slice_1', 'Ez1', wl, 'Ez1', 300)]["s_m"].item()))
        top_cross_talk_list.append(10*math.log10(s_params[('in_slice_1', 'top_slice', 'Ez1', wl, 'Ez1', 300)]["s_p"].item()))
        bot_cross_talk_list.append(10*math.log10(s_params[('in_slice_1', 'bot_slice', 'Ez1', wl, 'Ez1', 300)]["s_m"].item()))
    # Store these lists to CSV file
    csv_filename = f"./figs/coupler4_{'port4'}_s{crossing_region_size[0]}x{crossing_region_size[1]}_c-{exp_comment}" + '/split_ratio.csv'
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(['Wavelength', 'S31', 'S11', 'S21', 'S41'])
        # Write the data row by row
        for i, wl in enumerate(wl_list):
            csv_writer.writerow([wl, fwd_trans_list[i], refl_trans_list[i], top_cross_talk_list[i], bot_cross_talk_list[i]])

    print(f"Data successfully written to {csv_filename}")
