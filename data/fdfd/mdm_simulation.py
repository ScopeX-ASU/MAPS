import os
import re
import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.invdes.models import MDMOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MDM
from core.utils import (
    DeterministicCtx,
    SharpnessScheduler,
    material_fn_dict,
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


def load_prediction_image(opt, image_dir="./data/fdfd/mdm/prefab_images"):
    prefab_dir = Path(image_dir)
    if not prefab_dir.exists():
        raise FileNotFoundError(f"Prefab image directory not found: {prefab_dir}")

    image_paths = sorted(prefab_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No prefab images found in {prefab_dir}")

    ez1_images = [path for path in image_paths if "Ez1" in path.name]
    candidates = ez1_images if ez1_images else image_paths

    def _extract_step(path):
        match = re.search(r"_opt_step_(\d+)", path.name)
        return int(match.group(1)) if match else -1

    image_path = max(candidates, key=_extract_step)

    with Image.open(image_path) as img:
        bitmap = np.array(img.convert("L"), dtype=np.uint8)

    region_mask = next(iter(opt.device.design_region_masks.values()))
    target_height = region_mask.x.stop - region_mask.x.start
    target_width = region_mask.y.stop - region_mask.y.start

    resized = Image.fromarray(bitmap).resize(
        (target_width, target_height), resample=Image.NEAREST
    )
    normalized = np.asarray(resized, dtype=np.float32) / 255.0
    binary_mask = normalized >= 0.5

    wavelength = opt.sim_cfg.get("wl_cen", 1.55)
    eps_si = float(material_fn_dict["Si_eff"](wavelength))
    eps_sio2 = float(material_fn_dict["SiO2"](wavelength))

    eps_map = torch.tensor(
        opt.device.epsilon_map, dtype=torch.float32, device=opt.operation_device
    )
    binary_mask_tensor = torch.from_numpy(binary_mask.astype(np.bool_)).to(
        opt.operation_device
    )
    region_eps = torch.full(
        (target_height, target_width),
        eps_sio2,
        dtype=torch.float32,
        device=opt.operation_device,
    )
    region_eps.masked_fill_(binary_mask_tensor, eps_si)

    eps_map[region_mask.x, region_mask.y] = region_eps
    return eps_map


def mdm_simulation(
    device_id,
    operation_device,
    each_step=False,
    include_perturb=False,
    perturb_probs=[0.05, 0.1, 0.15],
    image_path="/home/hzhou144/projects/MAPS_local/data/fdfd/mdm/raw_opt_traj_ptb/mdm_id-0_opt_step_63-in_slice_1-1.55-Ez2-300.png",
):
    set_torch_deterministic(int(device_id))
    # dump_data_path = f"./data/fdfd/mdm/raw_opt_traj_ptb"
    sim_cfg = DefaultSimulationConfig()
    target_img_size = 256
    resolution = 50
    target_cell_size = target_img_size / resolution  # 5.12
    port_len = round(random.uniform(1, 1.2) * resolution) / resolution

    mdm_region_size = [
        round((target_cell_size - 2 * port_len) * resolution) / resolution,
        round((target_cell_size - 2 * port_len) * resolution) / resolution,
    ]
    assert (
        round(mdm_region_size[0] + 2 * port_len, 2) == target_cell_size
    ), f"right hand side: {mdm_region_size[0] + 2 * port_len}, target_cell_size: {target_cell_size}"

    input_port_width = 0.8
    output_port_width = 0.8

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, port_len, port_len],
            resolution=resolution,
            plot_root=f"./data/fdfd/mdm/plot_opt_traj_ptb/mdm_{device_id}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )

    device = MDM(
        sim_cfg=sim_cfg,
        box_size=mdm_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )
    # hr_device = device.copy(resolution=310)
    hr_device = device.copy(resolution=1000)
    print(device)
    opt = MDMOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    print(opt)

    ## load predicted image
    # eps = load_prediction_image(opt)
    results = opt.evaluation(image_path)
    print(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--image_path",
        type=str,
        default="/home/hzhou144/projects/MAPS_local/data/fdfd/mdm/prefab/corrected_design_prediction.png",
    )
    # parser.add_argument("--each_step", type=bool, default=False)
    # parser.add_argument("--include_perturb", type=int, default=0)
    random_seed = parser.parse_args().random_seed
    gpu_id = parser.parse_args().gpu_id
    image_path = parser.parse_args().image_path
    # each_step = parser.parse_args().each_step
    # include_perturb = parser.parse_args().include_perturb
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + random_seed))
    mdm_simulation(random_seed, device, image_path=image_path)


if __name__ == "__main__":
    main()
