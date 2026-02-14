"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu (jiaqigu@asu.edu)
LastEditTime: 2026-02-09 12:18:23
FilePath: /MAPS_ilt/core/invdes/examples/mdm_SiN.py
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

from core.invdes import builder
from core.invdes.invdesign import InvDesign
from core.invdes.models import MDMOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MDM
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

    # mdm_region_size = (20, 9)
    mdm_region_size = (17, 9)
    port_len = 1.8

    input_port_width = 5
    output_port_width = 0.85
    num_outports = 5
    exp_name = f"mdm_opt-port-{num_outports}_SiN_neff1.7_{mdm_region_size[0]}x{mdm_region_size[1]}"

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            # border_width=[port_len, port_len, 2, 2],
            border_width=[0, 0, 2, 2],
            resolution=25,
            plot_root=f"./figs/{exp_name}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        for key, obj in breakdown.items():
            # if key in {f"wl{i}_trans" for i in range(1, n_wl + 1)}:
            #     continue
            fom = fom + obj["weight"] * obj["value"]

        product = 0

        ## if only sum all transmission, lbfgs cannot balance them well.

        ## do not use direct product! ill-conditioned!
        # for i in range(1, n_wl + 1):
        #     product = product * breakdown[f"wl{i}_trans"]["value"]

        ## this sum-of-log formulation is more numerically stable and also encourages all transmissions to be high (since log is dominated by the smallest value)
        for i in range(1, num_outports + 1):
            product = product + torch.log(breakdown[f"mode{i}_trans"]["value"] + 1e-3)
        fom = fom + product * 10
        return fom, {"trans_product": {"weight": 1, "value": product}}

    device = MDM(
        material_r1="SiN_eff",
        sim_cfg=sim_cfg,
        box_size=mdm_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        num_outports=num_outports,
        port_box_margin=0.5,
        device=operation_device,
    )

    hr_device = device.copy(resolution=100)
    print(device)

    obj_cfgs = {}
    for i in range(1, num_outports + 1):
        obj_cfgs[f"mode{i}_trans"] = dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name=f"out_slice_{i}",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=[1.55],
            temp=[300],
            in_mode=f"Ez{i}",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                "Ez1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",
            direction="x+",
        )

    obj_cfgs["_fusion_func"] = fom_func
    obj_cfgs["override"] = True

    design_region_param_cfgs = dict()
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[100, 100],
            transform=[
                dict(
                    type="blur",
                    mfs=0.1,
                    resolutions=[hr_device.resolution, hr_device.resolution],
                    dim="xy",
                ),
                dict(type="binarize"),
            ],
            init_method="ones",
            denorm_mode="linear_eps",
            # interpolation="bilinear",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
        )

    opt = MDMOptimization(
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
        optimizer=Config(
            # name="Adam",
            lr=0.1,
            name="lbfgs",
            line_search_fn="strong_wolfe",
            # lr=1e-2,
            # weight_decay=0,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=5,
            plot_name=f"{exp_name}",
            objs=[f"mode{i}_trans" for i in range(1, num_outports + 1)],
            field_keys=[
                (f"in_slice_1", 1.55, f"Ez{i}", 300) for i in range(1, num_outports + 1)
            ],
            in_slice_names=["in_slice_1"] * num_outports,
            # exclude_slice_names=[["out_slice_2"], ["out_slice_1"]],
            exclude_slice_titles=[
                [f"out_slice_{i}"] for i in range(1, num_outports + 1)
            ],
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
        ),
    )
    invdesign.optimize()
