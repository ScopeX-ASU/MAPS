"""
Date: 2024-12-13 02:46:01
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-01 23:31:24
FilePath: /MAPS/unitest/test_mesh_interp.py
"""

"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from core.invdes.invdesign import InvDesign
from core.invdes.models import (
    BendingOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import Bending
from core.utils import set_torch_deterministic

# import cProfile
# import pstats
sys.path.pop(0)


if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    bending_region_size = (1.6, 1.6)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.48

    Ezs = []
    res_list = [25, 50, 100, 200, 400]
    for res in res_list:
        set_torch_deterministic(int(41 + 500))
        sim_cfg.update(
            dict(
                solver="ceviche_torch",
                border_width=[0, port_len, port_len, 0],
                resolution=res,
                plot_root=f"./figs/bending_test_mesh",
                PML=[0.5, 0.5],
                neural_solver=None,
                numerical_solver="solve_direct",
                # numerical_solver="solve_iterative",
                use_autodiff=False,
            )
        )

        device = Bending(
            sim_cfg=sim_cfg,
            bending_region_size=bending_region_size,
            port_len=(port_len, port_len),
            port_width=(input_port_width, output_port_width),
            device=operation_device,
        )

        hr_device = device.copy(resolution=200)
        print(device)
        opt = BendingOptimization(
            device=device,
            hr_device=hr_device,
            sim_cfg=sim_cfg,
            operation_device=operation_device,
        ).to(operation_device)
        invdesign = InvDesign(devOptimization=opt)
        # with cProfile.Profile() as profile:
        results = invdesign.devOptimization.forward(sharpness=256)
        field_keys = ("in_port_1", 1.55, 1, 300)
        Ez = opt.objective.solutions[field_keys]["Ez"]

        plot_filename = f"bending_test_mesh"
        invdesign.devOptimization.plot(
            eps_map=invdesign.devOptimization._eps_map,
            obj=results["breakdown"]["fwd_trans"]["value"],
            plot_filename=plot_filename + f"_{res}.jpg",
            # field_key=("in_port_1", 1.55, 1),
            field_key=field_keys,
            field_component="Ez",
            in_port_name="in_port_1",
            # exclude_port_names=exclude_port_names[j],
        )

        print(f"resolution: {res}")
        print(Ez.shape)
        print(
            Ez[
                Ez.shape[0] // 2 : Ez.shape[0] // 2 + 1,
                Ez.shape[1] // 2 : Ez.shape[1] // 2 + 1,
            ]
        )
        Ezs.append(Ez)
    
    ## downsample to the same lowest resolution
    for i in range(1, len(Ezs)):
        n = Ezs[i].shape[0] // Ezs[0].shape[0]
        
        # Ezs[i] = Ezs[i][n//2-1::n,n//2-1::n]
        Ezs[i] = Ezs[i][0::n,0::n]

    ## we want to compare the Ez with the highesst resolution one with MSE error
    for i in range(0, len(Ezs)-1):
        # n = Ezs[-1].shape[0] // Ezs[i].shape[0]
        Ez = torch.view_as_real(Ezs[i])
        Ez_t = torch.view_as_real(Ezs[-1])
        print(f"MSE between resolution {res_list[-1]} and resolution {res_list[0] * 2 ** i}: {torch.nn.functional.mse_loss(Ez, Ez_t)}")
    
    ## we want to compare the Ez with the highesst resolution one with MSE error
    for i in range(0, len(Ezs)-1):
       
        Ez1 = torch.view_as_real(Ezs[i])
        Ez2 = torch.view_as_real(Ezs[i+1])
        Ez_t = torch.view_as_real(Ezs[-1])
        ### Richardson Extrapolation
        Ez_interp = -1/3 * Ez1 + 4/3 * Ez2
        print(f"MSE between resolution {res_list[-1]} and extrapolated resolution ({res_list[0] * 2 ** i},{25 * 2 ** (i+1)}): {torch.nn.functional.mse_loss(Ez_interp, Ez_t)}")

    # invdesign.optimize(
    #     plot=True,
    #     plot_filename=f"bending_{'init_try'}",
    #     objs=["fwd_trans"],
    #     field_keys=[("in_port_1", 1.55, 1, 300)],
    #     in_port_names=["in_port_1"],
    #     exclude_port_names=[],
    #     dump_gds=True,
    # )
    # profile_result = pstats.Stats(profile)
    # # profile_result.sort_stats(pstats.SortKey.TIME)
    # profile_result.sort_stats(pstats.SortKey.CUMULATIVE)
    # profile_result.print_stats(30)
