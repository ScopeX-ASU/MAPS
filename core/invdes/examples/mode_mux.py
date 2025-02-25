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
from core.invdes.models import (
    ModeMuxOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import ModeMux
from core.utils import set_torch_deterministic
from core.invdes.invdesign import InvDesign

sys.path.pop(0)
if __name__ == "__main__":
    gpu_id = 3
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    mode_mux_region_size = (6, 5)
    port_len = 1.8

    input_port_width = 0.8
    output_port_width = 0.8

    exp_comment = "180_220"

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            # border_width=[port_len, port_len, 2, 2],
            border_width=[0, 0, 2, 2],
            resolution=50,
            plot_root=f"./figs/mode_mux_{mode_mux_region_size[0]}x{mode_mux_region_size[1]}_{exp_comment}",
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
        product = breakdown["mode1_trans"]["value"] * breakdown["mode2_trans"]["value"]
        fom = fom + product * 5
        return fom, {"trans_product": {"weight": 1, "value": product}}

    obj_cfgs = dict(_fusion_func=fom_func)
    device = ModeMux(
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
    )
    invdesign.optimize(
        plot=True,
        plot_filename=f"mode_mux_{'init_try_test'}",
        objs=["mode1_trans", "mode2_trans"],
        field_keys=[("in_slice_1", 1.55, "Ez1", 300), ("in_slice_2", 1.55, "Ez2", 300)],
        in_slice_names=["in_slice_1", "in_slice_2"],
        exclude_slice_names=[],
    )
