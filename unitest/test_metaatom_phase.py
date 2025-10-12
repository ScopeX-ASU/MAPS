import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import csv
import math

import h5py
import numpy as np
import torch
from pyutils.config import Config

from core.invdes import builder
from core.invdes.invdesign import InvDesign
from core.invdes.models import MetaLensOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MetaLens
from core.utils import print_stat, set_torch_deterministic

sys.path.pop(0)


def get_mid_weight(l, w, period=0.3):
    return (w * l) / (period - w)


if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    sim_cfg = DefaultSimulationConfig()

    wl = 0.850
    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            numerical_solver="solve_direct",
            use_autodiff=False,
            neural_solver=None,
            border_width=[0, 0, 0, 0],
            PML=[0.5, 0],
            resolution=50,
            wl_cen=wl,
            plot_root="./figs/metaatom_init_try",
        )
    )

    device = MetaLens(
        material_bg="Air",
        material_sub="SiO2",
        sim_cfg=sim_cfg,
        aperture=0.3,
        port_len=(1, 1),
        port_width=(0.3, 0.3),
        substrate_depth=0,
        ridge_height_max=0.75,
        nearfield_dx=0.3,
        nearfield_size=0.3,
        farfield_dxs=((30, 37.2),),
        farfield_sizes=(0.3,),
        device=operation_device,
    )

    hr_device = device.copy(resolution=200)
    # phase_shift_recoder_cfg = dict(
    #                 near_field_phase_record=dict(
    #                     weight=0,
    #                     #### objective is evaluated at this port
    #                     in_slice_name="in_slice_1",
    #                     out_slice_name="nearfield_1",
    #                     #### objective is evaluated at all points by sweeping the wavelength and modes
    #                     in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
    #                     wl=[1.55],
    #                     temp=[300],
    #                     out_modes=(
    #                         1,
    #                     ),  # can evaluate on multiple output modes and get average transmission
    #                     type="phase_recoder",
    #                     direction="x+",
    #                 )
    #             ),
    print(device)
    opt = MetaLensOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        # obj_cfgs=phase_shift_recoder_cfg,
        operation_device=operation_device,
    ).to(operation_device)

    assert len(list(opt.parameters())) == 1, "there should be only one parameter"
    width = torch.linspace(0.01, 0.14, 140)
    w = []
    phase_std = []
    phase_mean = []
    for i in width:
        for p in opt.parameters():
            init_weight = [-0.05, get_mid_weight(0.05, round(i.item(), 3)), -0.05]
            p.data = torch.tensor(init_weight, device=operation_device).unsqueeze(0)

        results = opt.forward(sharpness=256)
        w.append(round(i.item(), 3))
        phase_std.append(
            opt.objective.phase_shift[("in_slice_1", "nearfield_1", 0.85, "Hz1", 300)][
                "phase_std"
            ].item()
        )
        phase_mean.append(
            opt.objective.phase_shift[("in_slice_1", "nearfield_1", 0.85, "Hz1", 300)][
                "phase_mean"
            ].item()
        )
        opt.plot(
            eps_map=opt._eps_map,
            obj=None,
            plot_filename=f"metaatom_init_try_{round(i.item(), 3)}.png",
            field_key=("in_slice_1", 0.85, "Hz1", 300),
            field_component="Hz",
            in_slice_name="in_slice_1",
            exclude_slice_names=[],
        )
    w = np.array(w)
    phase_mean = np.array(phase_mean)
    phase_mean = (phase_mean[0] - phase_mean) % (2 * np.pi)
    phase_std = np.array(phase_std)
    # save them to csv
    with open("./unitest/metaatom_phase.csv", mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["width", "phase_mean", "phase_std"])
        for i in range(len(w)):
            writer.writerow([w[i], phase_mean[i], phase_std[i]])
    # invdesign = InvDesign(
    #     devOptimization=opt,
    #     sharp_scheduler=Config(
    #         name="sharpness",
    #         init_sharp=256,
    #         final_sharp=256,
    #     ),
    #     run=Config(
    #         n_epochs=1,
    #     ),
    # )
    # invdesign.optimize(
    #     plot=True,
    #     plot_filename=f"metalens_{'init_try'}",
    #     objs=["fwd_trans"],
    #     field_keys=[("in_port_1", wl, 1, 300)],
    #     in_port_names=["in_port_1"],
    #     exclude_port_names=["farfield_region"],
    # )
