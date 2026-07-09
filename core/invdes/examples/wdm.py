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
import numpy as np
import torch
from pyutils.config import Config

from core.invdes.invdesign import InvDesign
from core.invdes.models import WDMOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import WDM
from core.utils import set_torch_deterministic

sys.path.pop(0)


def generate_wdm(gpu_id, mfs):
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    wdm_region_size = [6, 6]
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.48
    mode = "Ez1"
    exp_name = f"wdm_opt_mfs{mfs * 1000:.0f}_500_100ls"

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            # border_width=[port_len, port_len, 2, 2],
            border_width=[0, 0, 2, 2],
            resolution=50,
            plot_root=f"./figs/{exp_name}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=1.55,
            wl_width=0.02,
            n_wl=2,
        )
    )

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        for key, obj in breakdown.items():
            if "smatrix" in key or "rad_trans" in key or "refl_trans" in key:
                continue
            if obj["weight"] < 0:
                continue
            fom = fom + obj["weight"] * obj["value"]

        ## add extra temp mul
        product = breakdown["wl1_trans"]["value"] * breakdown["wl2_trans"]["value"]
        fom = fom  # + product * 10
        return fom, {"trans_product": {"weight": 1, "value": product}}

    obj_cfgs = dict(_fusion_func=fom_func)

    device = WDM(
        sim_cfg=sim_cfg,
        box_size=wdm_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
        mode=mode,
    )

    hr_device = device.copy(resolution=50)
    print(device)

    design_region_param_cfgs = dict()
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[50, 50],
            transform=[
                dict(
                    type="blur",
                    mfs=mfs,
                    resolutions=[hr_device.resolution, hr_device.resolution],
                    dim="xy",
                ),
                dict(type="binarize"),
            ],
            init_method="ones",
            denorm_mode="linear_eps",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
        )

    opt = WDMOptimization(
        device=device,
        hr_device=hr_device,
        design_region_param_cfgs=design_region_param_cfgs,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(
        devOptimization=opt,
        optimizer=dict(
            name="Adam",
            lr=2e-2,
            weight_decay=0,
        ),
        run=Config(
            n_epochs=100,
            # n_epochs=20,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=1,
            plot_name=f"{exp_name}",
            objs=["wl1_trans", "wl2_trans"],
            field_keys=[
                ("in_slice_1", wl, mode, 300)
                for wl in np.linspace(
                    sim_cfg["wl_cen"] - sim_cfg["wl_width"] / 2,
                    sim_cfg["wl_cen"] + sim_cfg["wl_width"] / 2,
                    sim_cfg["n_wl"],
                )
            ],
            in_slice_names=["in_slice_1", "in_slice_1"],
            eps_grad=True,
            field_component="Ez",
            exclude_slice_names=[],
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
    gpu_id = 3
    # for mfs in [0.07, 0.09, 0.11, 0.13, 0.15]:
    # for mfs in [0.11]:
    for mfs in [0.016]:
        generate_wdm(gpu_id, mfs)
