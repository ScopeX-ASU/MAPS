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

from core.invdes.invdesign import InvDesign
from core.invdes.models import (
    MetaLensOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MetaLens
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

    # mdm_region_size = (6, 6)
    # port_len = 1.8

    # input_port_width = 0.8
    # output_port_width = 0.8

    wl = 0.832

    sim_cfg.update(
        dict(
            # solver="ceviche",
            solver="ceviche_torch",
            numerical_solver="solve_direct",
            use_autodiff=False,
            neural_solver=None,
            border_width=[0, 0, 1.5, 1.5],
            PML=[0.5, 0.5],
            resolution=100,
            wl_cen=wl,
            plot_root="./figs/metalens_near2far",
        )
    )

    device = MetaLens(
        material_bg="Air",
        sim_cfg=sim_cfg,
        aperture=3,
        port_len=(1.5, 5),
        substrate_depth=0.75,
        ridge_height_max=0.75,
        nearfield_dx=0.3,
        farfield_dxs=(4,),
        farfield_sizes=(1,),
        device=operation_device,
    )

    hr_device = device.copy(resolution=100)
    print(device)
    opt = MetaLensOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(
        devOptimization=opt,
    )
    invdesign.optimize(
        plot=True,
        plot_filename=f"metalens_{'init_try'}",
        objs=["fwd_trans"],
        field_keys=[("in_port_1", wl, 1, 300)],
        in_port_names=["in_port_1"],
        exclude_port_names=["farfield_region"],
    )
