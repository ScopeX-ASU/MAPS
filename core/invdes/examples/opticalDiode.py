"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
"""

import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "/home/pingchua/projects/MAPS")
)
sys.path.insert(0, project_root)
import torch
from pyutils.config import Config

from core.invdes import builder
from core.invdes.models import (
    opticalDiodeOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import opticalDiode
from core.utils import set_torch_deterministic
from core.invdes.invdesign import InvDesign

if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    opticaldiode_region_size = (3, 3)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.8

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            # border_width=[port_len, port_len, 2, 2],
            border_width=[0, 0, 2, 2],
            resolution=50,
            plot_root=f"./figs/opticalDiode_{'init_try'}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        for _, obj in breakdown.items():
            fom = fom + obj["weight"] * obj["value"]

        ## add extra contrast ratio
        contrast = breakdown["bwd_trans"]["value"] / breakdown["fwd_trans"]["value"]
        fom = fom - contrast
        return fom, {"contrast": {"weight": -1, "value": contrast}}

    obj_cfgs = dict(_fusion_func=fom_func)

    device = opticalDiode(
        sim_cfg=sim_cfg,
        box_size=opticaldiode_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )

    hr_device = device.copy(resolution=310)
    print(device)
    opt = opticalDiodeOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
        obj_cfgs=obj_cfgs,
    ).to(operation_device)
    invdesign = InvDesign(devOptimization=opt)
    invdesign.optimize(
        plot=True,
        plot_filename=f"opticalDiode_{'init_try'}",
        objs=["fwd_trans", "bwd_trans"],
        field_keys=[("in_port_1", 1.55, 1, 300), ("out_port_1", 1.55, 1, 300)],
        in_port_names=["in_port_1", "out_port_1"],
        exclude_port_names=[],
    )
