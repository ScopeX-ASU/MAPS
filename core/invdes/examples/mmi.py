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
from pyutils.config import Config

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

    mmi_region_size = (4, 4)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.48
    exp_name = "mmi_opt"

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 1, 1],  # left, right, lower, upper, containing PML
            resolution=100,
            plot_root=f"./figs/{exp_name}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )
    num_inports = 3
    num_outports = 3

    target_unitary_smatrix = torch.randn(
        (num_inports, num_outports), dtype=torch.complex64, device=operation_device
    )
    u, s, v = torch.linalg.svd(target_unitary_smatrix)
    target_unitary_smatrix = torch.matmul(u, v) * 0.95
    target_norm = torch.norm(target_unitary_smatrix)
    print(f"target unitary smatrix:\n{target_unitary_smatrix}")
    """ e.g., target unitary smatrix:
    [[-0.1627-0.6450j,  0.6381-0.2135j, -0.0702-0.0481j],
     [-0.5426-0.0271j, -0.0188+0.5920j, -0.5033+0.0567j],
     [-0.2311+0.3339j,  0.2916+0.1178j,  0.2743-0.7506j]]

    trained matrix, 0.581 relative L2 norm error
    [[-0.1511-0.6146j,  0.6196-0.2099j, -0.0684-0.0495j,]
     [-0.4896-0.0319j, -0.0191+0.5528j, -0.4856+0.0568j,]
     [-0.2111+0.2967j,  0.2755+0.1021j,  0.2669-0.7242j]
    """

    def fom_func(breakdown):
        ## maximization fom
        for key, obj in breakdown.items():
            if key == "smatrix":
                s_matrix = obj["value"]  # shape (num_outports, num_inports)
                fom = (
                    -torch.norm(s_matrix - target_unitary_smatrix.flatten())
                    / target_norm
                )
        # for key, obj in breakdown.items():
        #     if "fwd_trans" in key:
        #         fom = fom * obj["value"]
        #     elif "phase" in key:
        #         s_param_list.append(obj["value"])
        # if len(s_param_list) == 0:
        #     return fom
        # if isinstance(s_param_list[0], torch.Tensor):
        #     s_params = torch.tensor(s_param_list)
        #     s_params_std = torch.std(s_params)
        #     fom = fom - s_params_std
        # else:
        #     s_params = npa.array(s_param_list)
        #     s_params_std = npa.std(s_params)
        #     fom = fom - s_params_std
        # maximize the forward transmission and minimize the standard deviation of the s-parameters
        return fom, {"smatrix_err": {"weight": -1, "value": fom}}

    obj_cfgs = dict(_fusion_func=fom_func)

    device = MMI(
        sim_cfg=sim_cfg,
        box_size=mmi_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        num_inports=num_inports,
        num_outports=num_outports,
        device=operation_device,
        port_box_margin=0.5,
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
    invdesign = InvDesign(
        devOptimization=opt,
        run=Config(
            n_epochs=100,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=5,
            plot_name=f"{exp_name}",
            objs=["smatrix_err", "smatrix_err", "smatrix_err"],
            field_keys=[
                ("in_slice_1", 1.55, "Ez1", 300),
                ("in_slice_2", 1.55, "Ez1", 300),
                ("in_slice_3", 1.55, "Ez1", 300),
            ],
            in_slice_names=["in_slice_1", "in_slice_2", "in_slice_3"],
            filename_suffixes=["s1", "s2", "s3"],
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
