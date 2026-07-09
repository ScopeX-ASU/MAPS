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
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.four_port_coupler import FourPortOptimization
from core.invdes.models.layers.four_port_device import FourPortCoupler
from core.utils import set_torch_deterministic

sys.path.pop(0)


def generate_splitter1_3(gpu_id, mfs):
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    splitter1_3_region_size = (4.0, 4.0)
    port_len = 2

    input_port_width = 0.5
    output_port_width = 0.5
    exp_name = f"splitter1_3_opt_mfs{mfs * 1000:.0f}_500_100ls"

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 0, 0],  # left, right, lower, upper, containing PML
            resolution=100,
            plot_root=f"./figs/{exp_name}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=1.55,
            wl_width=0.0,
            n_wl=1,
        )
    )

    def fom_func(breakdown):
        for key, obj in breakdown.items():
            if key == "smatrix":
                s_matrix = obj["value"]  # shape (num_inports, num_outports)
                power = torch.abs(s_matrix) ** 2
                # maximize mean power and penalize std to enforce equal split
                mean_p = torch.mean(power)
                std_p = torch.std(power)
                fom = mean_p - 0.5 * std_p
        return fom, {
            "sum_T": {"weight": 1.0, "value": mean_p},
            "std_T": {"weight": -0.5, "value": -std_p},
        }

    obj_cfgs = dict(_fusion_func=fom_func)

    device = FourPortCoupler(
        sim_cfg=sim_cfg,
        box_size=(4.0, 4.0),
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
        port_box_margin=0.5,
    )
    hr_device = device.copy(resolution=500)
    print(device)

    design_region_param_cfgs = dict()
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[100, 100],
            transform=[
                dict(type="mirror_symmetry", dims=[1]),
                dict(
                    type="blur",
                    mfs=mfs,
                    resolutions=[hr_device.resolution, hr_device.resolution],
                    dim="xy",
                ),
                dict(type="binarize"),
            ],
            init_method="random",
            denorm_mode="linear_eps",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
        )

    opt = FourPortOptimization(
        device=device,
        hr_device=hr_device,
        design_region_param_cfgs=design_region_param_cfgs,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)

    invdesign = InvDesign(
        devOptimization=opt,
        run=Config(
            n_epochs=100,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=20,
            plot_name=f"{exp_name}",
            objs=["sum_T"],
            field_keys=[
                ("in_slice_1", 1.55, "Ez1", 300),
            ],
            in_slice_names=["in_slice_1"],
            filename_suffixes=["s1"],
            exclude_port_names=[],
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
            upsample_eps_to_1nm=True,
        ),
    )
    invdesign.optimize()


if __name__ == "__main__":
    gpu_id = 1
    # for mfs in [0.07, 0.09, 0.11, 0.13, 0.15]:
    for mfs in [0.11]:
        generate_splitter1_3(gpu_id, mfs)
