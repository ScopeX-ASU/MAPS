"""
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-12-15 03:40:49
FilePath: /MAPS/core/invdes/examples/metalens.py
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

from core.invdes.invdesign import InvDesign
from core.invdes.models import (
    MetaLensOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MetaLens
from core.utils import set_torch_deterministic
from pyutils.config import Config

sys.path.pop(0)
if __name__ == "__main__":
    gpu_id = 0
    # gpu_id = 1
    # gpu_id = 2
    # gpu_id = 3
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

    wl = 0.850
    initialization_file = "core/invdes/initialization/Si_metalens1D_for_850nm_FL30um.mat"
    # initialization_file = "core/invdes/initialization/Si_metalens1D_for_850nm_FL50um.mat"
    # initialization_file = "core/invdes/initialization/Si_metalens1D_for_850nm_FL60um.mat"
    # initialization_file = "core/invdes/initialization/Si_metalens1D_for_850nm_FL80um.mat"
    # initialization_file = "core/invdes/initialization/Si_metalens1D_for_850nm_FL30um_EDOF4.mat"
    # initialization_file = "core/invdes/initialization/Si_metalens1D_for_850nm_FL30um_EDOF5.mat"
    # initialization_file = "core/invdes/initialization/Si_metalens1D_for_850nm_FL100um.mat"
    sim_cfg.update(
        dict(
            # solver="ceviche",
            solver="ceviche_torch",
            numerical_solver="solve_direct",
            use_autodiff=False,
            neural_solver=None,
            border_width=[0, 0, 0, 0],
            PML=[0.5, 0.5],
            resolution=100,
            wl_cen=wl,
            plot_root="./figs/metalens_near2far_FL30_init",
            # plot_root="./figs/metalens_near2far_FL50",
            # plot_root="./figs/metalens_near2far_FL60",
            # plot_root="./figs/metalens_near2far_FL80",
            # plot_root="./figs/metalens_near2far_FL30_EDOF4",
            # plot_root="./figs/metalens_near2far_FL30_EDOF5",
            # plot_root="./figs/metalens_near2far_FL100",
        )
    )

    device = MetaLens(
        material_bg="Air",
        sim_cfg=sim_cfg,
        # aperture=20,
        aperture=20,
        port_len=(1.5, 2),
        port_width=(22.2, 2),
        substrate_depth=0,
        ridge_height_max=0.75,
        nearfield_dx=0.3,
        nearfield_size=21,
        # farfield_dxs=((20, 50),),
        farfield_dxs=((30, 37.2),),
        # farfield_dxs=((50, 70),),
        # farfield_dxs=((60, 89),),
        # farfield_dxs=((80, 131),),
        # farfield_dxs=((30, 100),),
        # farfield_dxs=((30, 140),),
        # farfield_dxs=((100, 180),),
        farfield_sizes=(2,),
        device=operation_device,
    )

    hr_device = device.copy(resolution=100)

    # def fom_func(breakdown):
    #     ## maximization fom
    #     fom = 0
    #     local_fom = 10
    #     for name, obj in breakdown.items():
    #         if name.startswith("fwd_trans"):
    #             local_fom = local_fom * obj["value"]
    #             local_weight = obj["weight"]
    #         else:
    #             fom = fom + obj["weight"] * obj["value"]
    #     fom = fom + local_fom * local_weight

    #     return fom, {"trans_prod": {"weight": local_weight, "value": local_fom}}

    # obj_cfgs = dict(_fusion_func=fom_func)

    print(device)
    opt = MetaLensOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
        initialization_file=initialization_file,
    ).to(operation_device)
    invdesign = InvDesign(
        devOptimization=opt,
        sharp_scheduler=Config(
            name="sharpness",
            init_sharp=256,
            final_sharp=256,
        ),
        run=Config(
            n_epochs=1,
        ),
    )
    invdesign.optimize(
        plot=True,
        plot_filename=f"metalens_{'init_try'}",
        objs=["fwd_trans"],
        field_keys=[("in_port_1", wl, 1, 300)],
        in_port_names=["in_port_1"],
        exclude_port_names=["farfield_region"],
    )
