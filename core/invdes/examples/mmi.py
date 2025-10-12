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
from autograd.numpy import array as npa

from core.invdes.invdesign import InvDesign
from core.invdes.models import MMIOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MMI
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

    mmi_region_size = (3, 3)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.48

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 1, 1],  # left, right, lower, upper, containing PML
            resolution=50,
            plot_root=f"./figs/mmi_{'init_try'}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )
    num_outports = 3

    def fom_func(breakdown):
        ## maximization fom
        fom = 1
        s_param_list = []
        for key, obj in breakdown.items():
            if "fwd_trans" in key:
                fom = fom * obj["value"]
            elif "phase" in key:
                s_param_list.append(obj["value"])
        if len(s_param_list) == 0:
            return fom
        if isinstance(s_param_list[0], torch.Tensor):
            s_params = torch.tensor(s_param_list)
            s_params_std = torch.std(s_params)
            fom = fom - s_params_std
        else:
            s_params = npa.array(s_param_list)
            s_params_std = npa.std(s_params)
            fom = fom - s_params_std
        # maximize the forward transmission and minimize the standard deviation of the s-parameters
        return fom, {"phase std": {"weight": -1, "value": s_params_std}}

    obj_cfgs = dict(_fusion_func=fom_func)

    device = MMI(
        sim_cfg=sim_cfg,
        box_size=mmi_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        num_outports=num_outports,
        device=operation_device,
    )

    hr_device = device.copy(resolution=100)
    print(device)
    opt = MMIOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
        obj_cfgs=obj_cfgs,
    ).to(operation_device)
    invdesign = InvDesign(devOptimization=opt)
    invdesign.optimize(
        plot=True,
        plot_filename=f"mmi_{'init_try'}",
        objs=["fwd_trans_1"],
        field_keys=[("in_port_1", 1.55, 1, 300)],
        in_port_names=["in_port_1"],
        exclude_port_names=[],
        dump_gds=True,
    )
