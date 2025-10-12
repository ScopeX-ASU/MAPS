"""
Date: 2024-10-03 02:27:36
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-06 18:59:45
FilePath: /MAPS/data/fdfd/generate_metacoupler.py
"""

import argparse

import torch
import torch.nn.functional as F

from core.invdes.models import MetaCouplerOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MetaCoupler
from core.utils import set_torch_deterministic
from thirdparty.ceviche.constants import *


def compare_designs(design_regions_1, design_regions_2):
    similarity = []
    for k, v in design_regions_1.items():
        v1 = v
        v2 = design_regions_2[k]
        similarity.append(F.cosine_similarity(v1.flatten(), v2.flatten(), dim=0))
    return torch.mean(torch.stack(similarity)).item()


def metacoupler_opt(device_id, operation_device):
    sim_cfg = DefaultSimulationConfig()
    # ------------------- Parameters for the metacoupler -------------------
    # total_height = 6
    # total_width = 6
    # aperture = random.uniform(5, 7)
    # aperture = int(aperture * 50) / 50
    # border_height = total_height - aperture / 2
    # ridge_height_max = random.uniform(0.8, 1.2)
    # ridge_height_max = int(ridge_height_max * 50) / 50
    # port_len = total_width - 3 * ridge_height_max
    # # input_port_width = random.uniform(5, aperture)
    # # input_port_width = int(input_port_width * 50) / 50
    # input_port_width = aperture
    # # output_port_width = random.uniform(2.8, 3.2)
    # # output_port_width = int(output_port_width * 50) / 50
    # output_port_width = 3

    # sim_cfg.update(
    #     dict(
    #         solver="ceviche_torch",
    #         border_width=[0, 0, border_height, border_height],
    #         resolution=50,
    #         plot_root=f"./figs/mfs_metacoupler_{device_id}",
    #         # plot_root="./figs/metacoupler_subpixel",
    #         # plot_root="./figs/metacoupler_periodic",
    #     )
    # )
    # ------------------- Parameters for the metacoupler -------------------
    total_height = 2.8
    total_width = 2.8
    aperture = 4
    border_height = total_height - aperture / 2
    ridge_height_max = 1
    port_len = total_width - 0.5 * ridge_height_max
    input_port_width = 3
    output_port_width = 3

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, border_height, border_height],
            resolution=50,
            plot_root=f"./figs/mfs_metacoupler_1_layer_{device_id}",
            PML=[0.5, 0.5],
        )
    )

    device = MetaCoupler(
        sim_cfg=sim_cfg,
        aperture=aperture,
        n_layers=1,  # here to simplify the problem, we only consider 1 layer
        ridge_height_max=ridge_height_max,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )
    hr_device = device.copy(resolution=310)
    print(device)
    opt = MetaCouplerOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    print(opt)

    n_epoch = 100
    optimizer = torch.optim.Adam(opt.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=70, eta_min=0.0002
    )
    last_design_region_dict = None
    for step in range(n_epoch):
        # for step in range(1):
        optimizer.zero_grad()
        # results = opt.forward(sharpness=1 + 2 * step)
        results = opt.forward(sharpness=256)
        print(f"Step {step}:", end=" ")
        for k, obj in results["breakdown"].items():
            print(f"{k}: {obj['value']:.3f}", end=", ")
        print()

        (-results["obj"]).backward()
        current_design_region_dict = opt.get_design_region_eps_dict()
        filename_h5 = f"./data/fdfd/metacoupler/mfs_raw_1_layer/metacoupler_id-{device_id}_opt_step_{step}.h5"
        filename_yml = (
            f"./data/fdfd/metacoupler/mfs_raw_1_layer/metacoupler_id-{device_id}.yml"
        )
        if last_design_region_dict is None:
            opt.dump_data(filename_h5=filename_h5, filename_yml=filename_yml, step=step)
            last_design_region_dict = current_design_region_dict
            opt.plot(
                eps_map=opt._eps_map,
                obj=results["breakdown"]["fwd_trans"]["value"],
                plot_filename="metacoupler_opt_step_{}_fwd.png".format(step),
                field_key=("in_slice_1", 1.55, "Ez1", 300),
                field_component="Ez",
                in_slice_name="in_slice_1",
                exclude_slice_names=["refl_slice_2"],
            )
            opt.plot(
                eps_map=opt._eps_map,
                obj=results["breakdown"]["bwd_trans"]["value"],
                plot_filename="metacoupler_opt_step_{}_bwd.png".format(step),
                field_key=("out_slice_1", 1.55, "Ez1", 300),
                field_component="Ez",
                in_slice_name="out_slice_1",
                exclude_slice_names=["refl_slice_1"],
            )
        else:
            cosine_similarity = compare_designs(
                last_design_region_dict, current_design_region_dict
            )
            if cosine_similarity < 0.998 or step == 9:
                opt.dump_data(
                    filename_h5=filename_h5, filename_yml=filename_yml, step=step
                )
                last_design_region_dict = current_design_region_dict
                opt.plot(
                    eps_map=opt._eps_map,
                    obj=results["breakdown"]["fwd_trans"]["value"],
                    plot_filename="metacoupler_opt_step_{}_fwd.png".format(step),
                    field_key=("in_slice_1", 1.55, "Ez1", 300),
                    field_component="Ez",
                    in_slice_name="in_slice_1",
                    exclude_slice_names=["refl_slice_2"],
                )
                opt.plot(
                    eps_map=opt._eps_map,
                    obj=results["breakdown"]["bwd_trans"]["value"],
                    plot_filename="metacoupler_opt_step_{}_bwd.png".format(step),
                    field_key=("out_slice_1", 1.55, "Ez1", 300),
                    field_component="Ez",
                    in_slice_name="out_slice_1",
                    exclude_slice_names=["refl_slice_1"],
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
    set_torch_deterministic(int(41 + random_seed))
    metacoupler_opt(random_seed, device)


if __name__ == "__main__":
    main()

    # metacoupler_opt(0, operation_device="cuda:0")
