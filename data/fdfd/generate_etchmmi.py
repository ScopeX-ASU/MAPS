import os
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from ceviche import fdfd_ez as ceviche_fdfd_ez
from ceviche.constants import *
from pyutils.general import print_stat

from core.models import (
    IsolatorOptimization,
    MetaCouplerOptimization,
    MetaMirrorOptimization,
    BendingOptimization,
    EtchMMIOptimization,
)
from core.models.base_optimization import BaseOptimization, DefaultSimulationConfig
from core.models.fdfd.fdfd import fdfd_ez
from core.models.fdfd.utils import torch_sparse_to_scipy_sparse
from core.models.layers import Isolator, MetaCoupler, MetaMirror, Bending, EtchMMI
from core.models.layers.device_base import N_Ports, Si_eps
from core.models.layers.utils import plot_eps_field
from core.utils import set_torch_deterministic
from torch_sparse import spspmm
import argparse
import random

def compare_designs(design_regions_1, design_regions_2):
    similarity = []
    for k, v in design_regions_1.items():
        v1 = v
        v2 = design_regions_2[k]
        similarity.append(F.cosine_similarity(v1.flatten(), v2.flatten(), dim=0))
    return torch.mean(torch.stack(similarity)).item()

def etchmmi_opt(device_id, operation_device):
    sim_cfg = DefaultSimulationConfig()

    bending_region_size = (1.6, 1.6)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.48

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, port_len, port_len, 0],
            resolution=50,
            plot_root=f"./figs/larger_mfs_bending_{device_id}",
            PML=[0.5, 0.5],
        )
    )

    device = EtchMMI(
        sim_cfg=sim_cfg, 
        bending_region_size=bending_region_size,
        port_len=(port_len, port_len),
        port_width=(
            input_port_width,
            output_port_width
        ), 
        device=operation_device
    )
    hr_device = device.copy(resolution=310)
    print(device)
    opt = EtchMMIOptimization(device=device, hr_device=hr_device, sim_cfg=sim_cfg, operation_device=operation_device).to(operation_device)
    print(opt)

    optimizer = torch.optim.Adam(opt.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=70, eta_min=0.0002
    )
    last_design_region_dict = None
    # for step in range(10):
    for step in range(10):
        optimizer.zero_grad()
        results = opt.forward(sharpness=1 + 2 * step)
        print(f"Step {step}:", end=" ")
        for k, obj in results["breakdown"].items():
            print(f"{k}: {obj['value']:.3f}", end=", ")
        print()

        (-results["obj"]).backward()
        current_design_region_dict = opt.get_design_region_eps_dict()
        filename_h5 = f"./data/fdfd/bending/mfs_raw_larger/bending_id-{device_id}_opt_step_{step}.h5"
        filename_yml = f"./data/fdfd/bending/mfs_raw_larger/bending_id-{device_id}.yml"
        if last_design_region_dict is None:
            opt.dump_data(filename_h5=filename_h5, filename_yml=filename_yml, step=step)
            last_design_region_dict = current_design_region_dict
            opt.plot(
                eps_map=opt._eps_map,
                obj=results["breakdown"]["fwd_trans"]["value"],
                plot_filename="bending_opt_step_{}_fwd.png".format(step),
                field_key=("in_port_1", 1.55, 1),
                field_component="Ez",
                in_port_name="in_port_1",
                exclude_port_names=["refl_port_2"],
            )
        else:
            cosine_similarity = compare_designs(last_design_region_dict, current_design_region_dict)
            if cosine_similarity < 0.996 or step == 9:
                opt.dump_data(filename_h5=filename_h5, filename_yml=filename_yml, step=step)
                last_design_region_dict = current_design_region_dict
                opt.plot(
                    eps_map=opt._eps_map,
                    obj=results["breakdown"]["fwd_trans"]["value"],
                    plot_filename="bending_opt_step_{}_fwd.png".format(step),
                    field_key=("in_port_1", 1.55, 1),
                    field_component="Ez",
                    in_port_name="in_port_1",
                    exclude_port_names=["refl_port_2"],
                )
        # for p in opt.parameters():
        #     print(p.grad)
        # print_stat(list(opt.parameters())[0], f"step {step}: grad: ")
        optimizer.step()
        scheduler.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    random_seed = parser.parse_args().random_seed
    gpu_id = parser.parse_args().gpu_id
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41+random_seed))
    etchmmi_opt(random_seed, device)

if __name__ == "__main__":
    main()
