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
    TDMOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import TDM
from core.utils import set_torch_deterministic
from core.invdes.invdesign import InvDesign

sys.path.pop(0)
if __name__ == "__main__":
    gpu_id = 1
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    mdm_region_size = (6, 6)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.48

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            # border_width=[port_len, port_len, 2, 2],
            border_width=[0, 0, 2, 2],
            resolution=50,
            plot_root=f"./figs/tdm_{'init_try'}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        for key, obj in breakdown.items():
            if key in {"temp1_trans", "temp2_trans"}:
                continue
            fom = fom + obj["weight"] * obj["value"]

        ## add extra temp mul
        product = breakdown["temp1_trans"]["value"] * breakdown["temp2_trans"]["value"] * 5
        fom = fom + product
        return fom, {"trans_product": {"weight": 5, "value": product}}

    obj_cfgs = dict(_fusion_func=fom_func)

    device = TDM(
        sim_cfg=sim_cfg,
        box_size=mdm_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )

    hr_device = device.copy(resolution=50)
    print(device)
    opt = TDMOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(devOptimization=opt)
    invdesign.optimize(
        plot=True,
        plot_filename=f"tdm_{'init_try'}",
        objs=["temp1_trans", "temp2_trans"],
        field_keys=[("in_port_1", 1.55, 1, 300), ("in_port_1", 1.55, 1, 360)],
        in_port_names=["in_port_1", "in_port_1"],
        exclude_port_names=[],
    )
