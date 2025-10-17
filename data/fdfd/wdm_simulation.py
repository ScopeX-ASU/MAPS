import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse
import random

import torch
import torch.nn.functional as F

from core.invdes.models import WDMOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import WDM
from core.utils import (
    DeterministicCtx,
    SharpnessScheduler,
    print_stat,
    set_torch_deterministic,
)
from thirdparty.ceviche.constants import *


def compare_designs(design_regions_1, design_regions_2):
    similarity = []
    for k, v in design_regions_1.items():
        v1 = v
        v2 = design_regions_2[k]
        similarity.append(F.cosine_similarity(v1.flatten(), v2.flatten(), dim=0))
    return torch.mean(torch.stack(similarity)).item()


def wdm_simulation(
    device_id,
    operation_device,
    each_step=False,
    include_perturb=False,
    perturb_probs=[0.05, 0.1, 0.15],
    image_path="/home/hzhou144/projects/MAPS_local/data/fdfd/wdm/raw_opt_traj_ptb/wdm_id-0_opt_step_85-in_slice_1-1.56-Ez1-300.png",
):
    set_torch_deterministic(int(device_id))
    dump_data_path = f"./data/fdfd/wdm/raw_opt_traj_ptb"
    sim_cfg = DefaultSimulationConfig()
    target_img_size = 512
    resolution = 50
    target_cell_size = target_img_size / resolution  # 10.24
    port_len = round(random.uniform(2, 2.4) * resolution) / resolution

    wdm_region_size = [
        round((target_cell_size - 2 * port_len) * resolution) / resolution,
        round((target_cell_size - 2 * port_len) * resolution) / resolution,
    ]
    assert (
        round(wdm_region_size[0] + 2 * port_len, 2) == target_cell_size
    ), f"right hand side: {wdm_region_size[0] + 2 * port_len}, target_cell_size: {target_cell_size}"

    input_port_width = 0.48
    output_port_width = 0.48

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, port_len, port_len],
            resolution=resolution,
            plot_root=f"./data/fdfd/wdm/plot_opt_traj_ptb/wdm_{device_id}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=1.55,
            wl_width=0.02,
            n_wl=2,
        )
    )

    device = WDM(
        sim_cfg=sim_cfg,
        box_size=wdm_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )
    # hr_device = device.copy(resolution=310)
    hr_device = device.copy(resolution=1000)
    print(device)
    opt = WDMOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    print(opt)
    
    results = opt.evaluation(image_path)
    print(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--image_path", type=str, default="/home/hzhou144/projects/MAPS_local/data/fdfd/wdm/prefab/corrected_design_prediction.png")
    random_seed = parser.parse_args().random_seed
    gpu_id = parser.parse_args().gpu_id
    image_path = parser.parse_args().image_path
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + random_seed))
    wdm_simulation(random_seed, device, image_path=image_path)


if __name__ == "__main__":
    main()
