import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
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
import h5py
from core.invdes import builder
from core.utils import print_stat
import numpy as np
import csv
import math
import random
sys.path.pop(0)

def get_mid_weight(l, w):
    return (w*l)/(0.3-w)

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
            resolution=200,
            wl_cen=wl,
            plot_root="./figs/metaatom_trustworthy",
        )
    )
    max_num_pillar_single_side = 10 
    num_perturb_piller_single_side = 5
    num_perturb = 5
    pillar_widths = []
    perturb_widths_dict = {}
    for i in range(0, num_perturb):
        perturb_widths_dict[i] = [round(random.uniform(0, 0.295), 2) for _ in range(2 * num_perturb_piller_single_side)]
    center_atom_phase_dict = {}
    center_atom_phase_mean_dict = {}
    for num_pillar_single_side in range(0, max_num_pillar_single_side+1):
        if num_pillar_single_side == 0:
            pillar_widths.append(round(random.uniform(0, 0.295), 2))
        else:
            pillar_widths = [round(random.uniform(0, 0.295), 2)] + pillar_widths + [round(random.uniform(0, 0.295), 2)]
        device = MetaLens(
            material_bg="Air",
            material_sub="SiO2",
            sim_cfg=sim_cfg,
            aperture=0.3 * ((num_pillar_single_side+num_perturb_piller_single_side) * 2 + 1),
            port_len=(1, 1),
            port_width=(0.3 * ((num_pillar_single_side+num_perturb_piller_single_side) * 2 + 1), 0.3),
            substrate_depth=0,
            ridge_height_max=0.75,
            nearfield_dx=0.3,
            nearfield_size=0.3,
            farfield_dxs=((30, 37.2),),
            farfield_sizes=(0.3,),
            device=operation_device,
        )

        hr_device = device.copy(resolution=200)
        print(device)
        opt = MetaLensOptimization(
            device=device,
            hr_device=hr_device,
            sim_cfg=sim_cfg,
            # obj_cfgs=phase_shift_recoder_cfg,
            operation_device=operation_device,
        ).to(operation_device)

        assert len(list(opt.parameters())) == 1, "there should be only one parameter"
        center_atom_phase_dict[num_pillar_single_side] = []
        center_atom_phase_mean_dict[num_pillar_single_side] = []
        for perturb_time in range(num_perturb):
            for p in opt.parameters():
                # init_weight = [-0.05, get_mid_weight(0.05, round(i.item(), 2)), -0.05]
                extend_pillar_widths = perturb_widths_dict[perturb_time][:num_perturb_piller_single_side] + pillar_widths + perturb_widths_dict[perturb_time][-num_perturb_piller_single_side:]
                print("this is the pillar_widths: ", extend_pillar_widths, flush=True)
                ls_knots = [get_mid_weight(0.05, round(extend_pillar_widths[j], 2)) for j in range(len(extend_pillar_widths))]
                ls_knots = torch.tensor(ls_knots, device=operation_device)
                weight = torch.zeros((1 + 2 * len(extend_pillar_widths)), device=operation_device)
                weight[::2] = -0.05
                weight[1::2] = ls_knots
                weight = weight.unsqueeze(0)
                p.data = weight
            results = opt.forward(sharpness=256)
            # w.append(round(i.item(), 2))
            center_atom_phase = opt.objective.phase_shift[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["phase"]
            center_atom_phase_mean = opt.objective.phase_shift[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["phase_mean"]
            center_atom_phase_dict[num_pillar_single_side].append(center_atom_phase)
            center_atom_phase_mean_dict[num_pillar_single_side].append(center_atom_phase_mean.item())

            opt.plot(
                eps_map=opt._eps_map,
                obj=None,
                plot_filename=f"metaatom_nPillarSingleSide-{num_pillar_single_side}_PerturbID_{perturb_time}.png",
                field_key=("in_slice_1", 0.85, "Hz1", 300),
                field_component="Hz",
                in_slice_name="in_slice_1",
                exclude_slice_names=[],
            )
    num_pillars_list = []
    mad_list = []
    for key, value in center_atom_phase_dict.items():
        num_pillars_list.append(key)
        value = torch.stack(value).squeeze().detach() # 5 x 60
        # plot the 5 curves whose length is 60 overlap in one figure
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(value.shape[0]):
            plt.plot(value[i].cpu().numpy(), label=f"perturb_{i}")
        # plt.ylim(0, math.pi * 4)
        plt.legend()
        plt.savefig(f"./unitest/metaatom_nPillarSingleSide-{key}_phase.png")
        plt.close()

        mu = value.mean(dim=0)
        mad = value.sub(mu).abs().mean().item()
        mad_list.append(mad)
    num_pillars_list = np.array(num_pillars_list)
    # save them to csv
    with open('./unitest/metaatom_phase.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["num_pillars", "mad"])
        for i in range(len(num_pillars_list)):
            writer.writerow([num_pillars_list[i], mad_list[i]])
