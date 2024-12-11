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
import numpy as np

from core.invdes import builder
from core.invdes.models import (
    WDMOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import WDM
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

    mdm_region_size = (3, 3)
    port_len = 1.8

    input_port_width = 0.8
    output_port_width = 0.8

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            # border_width=[port_len, port_len, 2, 2],
            border_width=[0, 0, 2, 2],
            resolution=50,
            plot_root=f"./figs/wdm_{'init_try'}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=1.55,
            wl_width=0.6,
            n_wl=2,
        )
    )

    device = WDM(
        sim_cfg=sim_cfg,
        box_size=mdm_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )

    hr_device = device.copy(resolution=50)
    print(device)
    opt = WDMOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(devOptimization=opt)
    invdesign.optimize(
        plot=True,
        plot_filename=f"wdm_{'init_try'}",
        objs=["wl1_trans", "wl2_trans"],
        field_keys=[("in_port_1", wl, 1, 300) for wl in np.linspace(sim_cfg["wl_cen"] - sim_cfg["wl_width"]/2, sim_cfg["wl_cen"] + sim_cfg["wl_width"]/2, sim_cfg["n_wl"])],
        in_port_names=["in_port_1", "in_port_1"],
        exclude_port_names=[],
    )
