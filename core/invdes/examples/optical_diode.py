"""
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-16 16:51:36
FilePath: /MAPS/core/invdes/examples/optical_diode.py
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

from core.invdes.invdesign import InvDesign
from core.invdes.models import (
    OpticalDiodeOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import OpticalDiode
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

    opticaldiode_region_size = (7, 6)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.8
    exp_name = "180_220"
    # exp_name = "adam"

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            # border_width=[port_len, port_len, 2, 2],
            border_width=[0, 0, 2, 2],
            resolution=50,
            plot_root=f"./figs/optical_diode_{opticaldiode_region_size[0]}_{opticaldiode_region_size[1]}_{exp_name}",
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

    device = OpticalDiode(
        sim_cfg=sim_cfg,
        box_size=opticaldiode_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )

    hr_device = device.copy(resolution=310)
    print(device)
    opt = OpticalDiodeOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
        obj_cfgs=obj_cfgs,
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
        plot_filename=f"optical_diode_{exp_name}",
        objs=["fwd_trans", "bwd_trans"],
        field_keys=[
            ("in_slice_1", 1.55, "Ez1", 300),
            ("out_slice_1", 1.55, "Ez1", 300),
        ],
        in_slice_names=["in_slice_1", "out_slice_1"],
        exclude_slice_names=[],
        dump_gds=True,
    )
