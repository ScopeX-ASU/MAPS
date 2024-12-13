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

from core.invdes.invdesign import InvDesign
from core.invdes.models import (
    edgeCouplerOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import edgeCoupler
from core.utils import set_torch_deterministic

if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    edgeCoupler_region_size = (1.6, 1.6)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 1.6

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 2, 2],
            resolution=50,
            plot_root=f"./figs/edgeCoupler_{'init_try'}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )

    device = edgeCoupler(
        sim_cfg=sim_cfg,
        box_size=edgeCoupler_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )

    hr_device = device.copy(resolution=310)
    print(device)
    opt = edgeCouplerOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(devOptimization=opt)
    invdesign.optimize(
        plot=True,
        plot_filename=f"edgeCoupler_{'init_try'}",
        objs=["fwd_trans"],
        field_keys=[("in_port_1", 1.55, 1, 300)],
        in_port_names=["in_port_1"],
        exclude_port_names=[],
        dump_gds=True,
    )
