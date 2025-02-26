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
from core.invdes.models import (
    ModeMuxOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import ModeMux
from core.utils import set_torch_deterministic

sys.path.pop(0)
if __name__ == "__main__":
    gpu_id = 3
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()
    root = "mode_mux_autotune"

    def eval_obj_fn(iter: int, params):
        mode_mux_region_size = (
            params["design_region_size_x"],
            params["design_region_size_y"],
        )
        port_len = 1.8

        input_port_width = 0.48
        output_port_width = 0.8
        etch_thickness = params["etch_thickness"]

        exp_comment = f"{etch_thickness}"

        sim_cfg.update(
            dict(
                solver="ceviche_torch",
                # border_width=[port_len, port_len, 2, 2],
                border_width=[0, 0, 2, 2],
                resolution=50,
                plot_root=f"./figs/{root}",
                PML=[0.5, 0.5],
                neural_solver=None,
                numerical_solver="solve_direct",
                use_autodiff=False,
            )
        )

        def fom_func(breakdown):
            ## maximization fom
            fom = 0

            ## add extra temp mul
            product = (
                breakdown["mode1_trans"]["value"] * breakdown["mode2_trans"]["value"]
            )
            fom = fom + product * 5
            return fom, {"trans_product": {"weight": 1, "value": product}}

        obj_cfgs = dict(_fusion_func=fom_func)
        device = ModeMux(
            etch_thickness=etch_thickness,
            sim_cfg=sim_cfg,
            box_size=mode_mux_region_size,
            port_len=(port_len, port_len),
            port_width=(input_port_width, output_port_width),
            device=operation_device,
        )

        hr_device = device.copy(resolution=310)
        print(device)
        opt = ModeMuxOptimization(
            device=device,
            hr_device=hr_device,
            sim_cfg=sim_cfg,
            obj_cfgs=obj_cfgs,
            operation_device=operation_device,
        ).to(operation_device)
        invdesign = InvDesign(
            devOptimization=opt,
            optimizer=Config(
                # name="adam",
                name="nesterov",
                lr=1e-2,
                use_bb=False,
            ),
            run=Config(
                n_epochs=15,
            ),
            lr_scheduler=Config(
                name="cosine",
                lr_min=8e-3,
            ),
            sharp_scheduler=Config(
                mode="cosine",
                name="sharpness",
                init_sharp=0.1,
                final_sharp=20,
            ),
        )
        invdesign.optimize(
            plot=True,
            plot_filename=f"mode_mux_{iter}_{mode_mux_region_size[0]}x{mode_mux_region_size[1]}_{exp_comment}",
            objs=["mode1_trans", "mode2_trans"],
            field_keys=[
                ("in_slice_1", 1.55, "Ez1", 300),
                ("in_slice_2", 1.55, "Ez1", 300),
            ],
            in_slice_names=["in_slice_1", "in_slice_2"],
            exclude_slice_names=[],
        )

        obj = invdesign.results["obj"].item()
        return obj, invdesign

    autotuner = AutoTune(
        log_path=f"./log/{root}/{root}.log",
        eval_obj_fn=eval_obj_fn,
        sampler="BoTorchSampler",
        params_cfgs=Config(
            design_region_size_x=dict(
                type="float",
                low=5,
                high=7,
                step=0.1,
                log=False,
            ),
            design_region_size_y=dict(
                type="float",
                low=4,
                high=6,
                step=0.1,
                log=False,
            ),
            etch_thickness=dict(
                type="float",
                low=0.15,
                high=0.18,
                step=0.005,
                log=False,
            ),
        ),
        run=Config(
            n_epochs=100,
        ),
    )
    autotuner.add_init_guesses(
        # [dict(design_region_size_x=6.4, design_region_size_y=4.0, etch_thickness=0.15)
        [
            {
                "design_region_size_x": 6.4,
                "design_region_size_y": 4.0,
                "etch_thickness": 0.15,
            },
            {
                "design_region_size_x": 6.6,
                "design_region_size_y": 5.3,
                "etch_thickness": 0.17,
            },
            {
                "design_region_size_x": 5.1,
                "design_region_size_y": 5.4,
                "etch_thickness": 0.155,
            },
        ]
    )
    autotuner.search()
