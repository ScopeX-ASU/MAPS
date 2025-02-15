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

sys.path.pop(0)
if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    crossing_region_size = (5, 5)
    port_len = 2

    input_port_width = 0.48
    output_port_width = 0.48

    wl_cen = 1.55
    wl_width = 0.01
    n_wl = 3
    wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 0, 0],
            resolution=50,
            # plot_root=f"./figs/coupler4_{'port3'}",
            plot_root=f"./figs/coupler4_{'port4'}_s{crossing_region_size[0]}x{crossing_region_size[1]}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
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
    # print(device)

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        # for _, obj in breakdown.items():
        #     fom = fom + obj["weight"] * obj["value"]
        for name, obj in breakdown.items():
            if "rad" in name:
                fom = fom + obj["weight"] * obj["value"]
        ## add extra contrast ratio
        target = 0.25
        balance = 0
        for wl in wls:
            balance += (
                (breakdown[f"fwd_trans_{wl}"]["value"] - target) ** 2
                + (breakdown[f"refl_trans_{wl}"]["value"] - target) ** 2
                + (breakdown[f"top_cross_talk_{wl}"]["value"] - target) ** 2
                + (breakdown[f"bot_cross_talk_{wl}"]["value"] - target) ** 2
            )
        fom = fom - 1 * balance
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
        return fom, {"balance": {"weight": -1, "value": balance}}

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

    opt = CrossingOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(devOptimization=opt)
    invdesign.optimize(
        plot=True,
        # plot_filename=f"coupler4_{'port3'}",
        plot_filename=f"coupler4_{'port4'}_s{crossing_region_size[0]}x{crossing_region_size[1]}",
        objs=[f"fwd_trans_{wl_cen}"],
        field_keys=[("in_slice_1", wl_cen, "Ez1", 300)],
        in_slice_names=["in_slice_1"],
        exclude_slice_names=[],
        dump_gds=True,
    )
