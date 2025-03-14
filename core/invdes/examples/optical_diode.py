"""
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-03-13 21:42:03
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
from core.utils import SharpnessScheduler, set_torch_deterministic

sys.path.pop(0)


class CustomInvDesign(InvDesign):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mfs_scheduler = SharpnessScheduler(
            initial_sharp=self._cfg.mfs_scheduler.init_sharp,
            final_sharp=self._cfg.mfs_scheduler.final_sharp,
            total_steps=self._cfg.run.n_epochs,
            mode="cosine",
        )

    def _before_step_callbacks(self, feed_dict):
        mfs = self.mfs_scheduler.get_sharpness()
        self.devOptimization.design_region_param_dict["design_region_1"].cfgs[
            "transform"
        ][1]["mfs"] = mfs
        self._log = f", mfs={mfs:.3f} um"
        self.mfs_scheduler.step()
        return feed_dict


if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    opticaldiode_region_size = (6, 4)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.8
    thickness_r1 = 0.22
    thickness_r2 = 0.15
    init_mfs = 0.35
    final_mfs = 0.2

    exp_name = f"optical_diode_{thickness_r1}_{thickness_r2}_mfs={init_mfs}-{final_mfs}"


    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            # border_width=[port_len, port_len, 2, 2],
            border_width=[0, 0, 2, 2],
            resolution=50,
            plot_root=f"./figs/{exp_name}_{opticaldiode_region_size[0]}_{opticaldiode_region_size[1]}",
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
        material_r1="Si_eff",
        material_r2="Si_eff",
        thickness_r1=thickness_r1,
        thickness_r2=thickness_r2,
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
    invdesign = CustomInvDesign(
        devOptimization=opt,
        optimizer=Config(
            # name="adam",
            name="nesterov",
            lr=1e-2,
            use_bb=False,
        ),
        mfs_scheduler=Config(
            mode="cosine",
            name="sharpness",
            init_sharp=init_mfs,
            final_sharp=final_mfs,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=5,
            plot_name=f"{exp_name}",
            objs=["fwd_trans", "bwd_trans"],
            field_keys=[
                ("in_slice_1", 1.55, "Ez1", 300),
                ("out_slice_1", 1.55, "Ez1", 300),
            ],
            in_slice_names=["in_slice_1", "out_slice_1"],
            exclude_slice_names=[],
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
        ),
    )
    invdesign.optimize()
