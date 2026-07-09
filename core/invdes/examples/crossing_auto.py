"""
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-25 23:21:36
FilePath: /MAPS/core/invdes/examples/crossing_auto.py
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

from core.invdes.autotune import AutoTune
from core.invdes.invdesign import InvDesign
from core.invdes.models import CrossingOptimization
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

    def eval_obj_fn(iter: int, params):
        ## this is important to make it deterministic
        set_torch_deterministic(int(41 + 500))
        # crossing_region_size = (1.6, 1.6) # this is now the search variable
        crossing_region_size = (
            params["design_region_size"],
            params["design_region_size"],
        )
        port_len = 1.8

        input_port_width = 0.48
        output_port_width = 0.48

        sim_cfg.update(
            dict(
                solver="ceviche_torch",
                border_width=[0, 0, 0, 0],
                resolution=50,
                plot_root=f"./figs/crossing_{'autotune'}",
                PML=[0.5, 0.5],
                neural_solver=None,
                numerical_solver="solve_direct",
                use_autodiff=False,
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
        print(device)

        obj_cfgs = dict(
            fwd_trans=dict(
                weight=1,
                #### objective is evaluated at this port
                in_slice_name="in_slice_1",
                out_slice_name="out_slice_1",
                #### objective is evaluated at all points by sweeping the wavelength and modes
                in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                wl=[1.55],
                temp=[300],
                out_modes=(
                    "Ez1",
                ),  # can evaluate on multiple output modes and get average transmission
                type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
                direction="x+",
            ),
            refl_trans=dict(
                weight=-0.1,
                #### objective is evaluated at this port
                in_slice_name="in_slice_1",
                out_slice_name="refl_slice_1",
                #### objective is evaluated at all points by sweeping the wavelength and modes
                in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                wl=[1.55],
                temp=[300],
                out_modes=(
                    "Ez1",
                ),  # can evaluate on multiple output modes and get average transmission
                type="flux_minus_src",
                direction="x",
            ),
            top_cross_talk=dict(
                weight=-0.1,
                #### objective is evaluated at this port
                in_slice_name="in_slice_1",
                out_slice_name="top_slice",
                in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                wl=[1.55],
                temp=[300],
                out_modes=(
                    "Ez1",
                ),  # can evaluate on multiple output modes and get average transmission
                type="flux",
                direction="y+",
            ),
            bot_cross_talk=dict(
                weight=-0.1,
                #### objective is evaluated at this port
                in_slice_name="in_slice_1",
                out_slice_name="bot_slice",
                in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                wl=[1.55],
                temp=[300],
                out_modes=(
                    "Ez1",
                ),  # can evaluate on multiple output modes and get average transmission
                type="flux",
                direction="y-",
            ),
            rad_trans_xp=dict(
                weight=0,
                #### objective is evaluated at this port
                in_slice_name="in_slice_1",
                out_slice_name="rad_slice_xp",
                #### objective is evaluated at all points by sweeping the wavelength and modes
                wl=[1.55],
                temp=[300],
                in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                out_modes=(
                    "Ez1",
                ),  # can evaluate on multiple output modes and get average transmission
                type="flux",
                direction="x",
            ),
            rad_trans_xm=dict(
                weight=0,
                #### objective is evaluated at this port
                in_slice_name="in_slice_1",
                out_slice_name="rad_slice_xm",
                #### objective is evaluated at all points by sweeping the wavelength and modes
                wl=[1.55],
                temp=[300],
                in_mode="Ez1",  # only one source mode is supsliceed, cannot input multiple modes at the same time
                out_modes=(
                    "Ez1",
                ),  # can evaluate on multiple output modes and get average transmission
                type="flux",
                direction="x",
            ),
            rad_trans_yp=dict(
                weight=0,
                #### objective is evaluated at this port
                in_slice_name="in_slice_1",
                out_slice_name="rad_slice_yp",
                #### objective is evaluated at all points by sweeping the wavelength and modes
                wl=[1.55],
                temp=[300],
                in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                out_modes=(
                    "Ez1",
                ),  # can evaluate on multiple output modes and get average transmission
                type="flux",
                direction="y",
            ),
            rad_trans_ym=dict(
                weight=0,
                #### objective is evaluated at this port
                in_slice_name="in_slice_1",
                out_slice_name="rad_slice_ym",
                #### objective is evaluated at all points by sweeping the wavelength and modes
                wl=[1.55],
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
            verbose=False,
        ).to(operation_device)
        invdesign = InvDesign(
            devOptimization=opt,
            run=Config(
                n_epochs=10,
            ),
            optimizer=Config(
                # name="adam",
                name="nesterov",
                lr=1e-2,
                use_bb=False,
            ),
        )
        invdesign.optimize(
            plot=True,
            plot_filename=f"crossing_{'autotune'}_{iter}",
            objs=["fwd_trans"],
            field_keys=[("in_slice_1", 1.55, "Ez1", 300)],
            in_slice_names=["in_slice_1"],
            exclude_slice_names=[],
            dump_gds=True,
            verbose=False,
        )
        obj = invdesign.results["obj"]
        return obj, invdesign

    autotuner = AutoTune(
        eval_obj_fn=eval_obj_fn,
        sampler="BoTorchSampler",
        params_cfgs=Config(
            design_region_size=dict(
                type="float",
                low=3,
                high=5,
                step=0.1,
                log=False,
            )
        ),
    )
    autotuner.search()
