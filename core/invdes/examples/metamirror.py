"""
Date: 2025-01-04 20:49:15
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-16 16:43:18
FilePath: /MAPS/core/invdes/examples/metamirror.py
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
from pyutils.torch_train import load_model

from core.invdes.invdesign import InvDesign
from core.invdes.models import MetaMirrorOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MetaMirror
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

    # exp_name = "Ez1_adam"
    exp_name = "Ez1_nesterov"

    input_port_width = 0.22
    output_port_width = 0.22

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 1.5, 0, 1.5],
            resolution=100,
            plot_root=f"./figs/metamirror_{exp_name}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )

    device = MetaMirror(
        sim_cfg=sim_cfg,
        aperture=1.5,
        ridge_height_max=0.06,
        port_len=(5, 4),
        port_width=(input_port_width, output_port_width),
        mirror_size=(0.316, 0.316),
        device=operation_device,
    )

    hr_device = device.copy(resolution=310)
    print(device)
    opt = MetaMirrorOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    checkpoint = None
    if checkpoint is not None:
        load_model(
            opt,
            checkpoint,
            ignore_size_mismatch=0,
        )
    invdesign = InvDesign(
        devOptimization=opt,
        # optimizer=dict(
        #     name="lbfgs",
        #     lr=1e-2,
        #     line_search_fn="strong_wolfe",
        #     weight_decay=0,
        # ),
        optimizer=Config(
            name="nesterov",
            lr=1e-2,
            use_bb=False,
        ),
    )
    invdesign.optimize(
        plot=True,
        plot_filename=f"metamirror_{exp_name}",
        objs=["fwd_trans"],
        field_keys=[("in_slice_1", 1.55, "Ez1", 300)],
        in_slice_names=["in_slice_1"],
        exclude_slice_names=[],
        dump_gds=False,
        save_model=False,
    )
