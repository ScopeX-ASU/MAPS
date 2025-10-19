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

from core.invdes.invdesign import InvDesign
from core.invdes.models import EdgeCouplerOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import EdgeCoupler
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

    # edge_coupler_region_size = (1.6, 1.6)
    edge_coupler_region_size = (2.5, 2.5)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 1.6
    exp_name = "edge_coupler_opt"

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 2, 2],
            resolution=50,
            plot_root=f"./figs/{exp_name}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )

    device = EdgeCoupler(
        sim_cfg=sim_cfg,
        box_size=edge_coupler_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        out_slice_size=2.5,
        out_slice_dx=0.1,
        device=operation_device,
    )

    hr_device = device.copy(resolution=310)
    print(device)
    opt = EdgeCouplerOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(
        devOptimization=opt,
        run=Config(
            n_epochs=100,
        ),
        sharp_scheduler=Config(
            mode="cosine",
            name="sharpness",
            init_sharp=1,
            final_sharp=256,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=5,
            plot_name=f"{exp_name}",
            objs=["fwd_trans"],
            field_keys=[("in_slice_1", 1.55, "Ez1", 300)],
            in_slice_names=["in_slice_1"],
            exclude_port_names=[],
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
        ),
    )
    invdesign.optimize()
