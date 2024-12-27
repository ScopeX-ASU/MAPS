import os
import sys

# Add the project root to sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)
import argparse
import random

import torch
import torch.nn.functional as F

from core.invdes.models import (
    MDMOptimization,
)
from core.invdes.models.base_optimization import (
    DefaultSimulationConfig,
)
from core.invdes.models.layers import MDM
from core.utils import set_torch_deterministic
from thirdparty.ceviche.ceviche.constants import *

from core.utils import DeterministicCtx, print_stat

def compare_designs(design_regions_1, design_regions_2):
    similarity = []
    for k, v in design_regions_1.items():
        v1 = v
        v2 = design_regions_2[k]
        similarity.append(F.cosine_similarity(v1.flatten(), v2.flatten(), dim=0))
    return torch.mean(torch.stack(similarity)).item()


def mdm_opt(device_id, operation_device, perturb_probs=[0.1, 0.3, 0.5]):
    set_torch_deterministic(int(device_id))
    dump_data_path = f"./data/fdfd/mdm/raw_small"
    sim_cfg = DefaultSimulationConfig()
    target_img_size = 256
    resolution = 50
    target_cell_size = target_img_size / resolution # 10.24
    port_len = round(random.uniform(1, 1.2) * resolution) / resolution

    mdm_region_size = [
        round((target_cell_size - 2 * port_len) * resolution) / resolution,
        round((target_cell_size - 2 * port_len) * resolution) / resolution,
    ]
    assert round(mdm_region_size[0] + 2 * port_len, 2) == target_cell_size, f"right hand side: {mdm_region_size[0] + 2 * port_len}, target_cell_size: {target_cell_size}"

    input_port_width = 0.8
    output_port_width = 0.8

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, port_len, port_len],
            resolution=resolution,
            plot_root=f"./data/fdfd/mdm/plot_small/mdm_{device_id}",
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
    hr_device = device.copy(resolution=50)
    print(device)
    opt = MDMOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    print(opt)

    optimizer = torch.optim.Adam(opt.parameters(), lr=0.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=0.0004
    )
    last_design_region_dict = None

    def perturb_and_dump(step, flip_prob=0.1, i=None):
        """
        Perturb parameters, perform forward and backward passes, and dump data.
        """
        assert i is not None, "The perturb_and_dump function requires an index i"
        with DeterministicCtx(seed=42 + step + i):
            # Save the original parameters and optimizer state
            original_params = [p.clone().detach() for p in opt.parameters()]
            optimizer_state = optimizer.state_dict()

            try:
                # Perturb parameters with noise
                with torch.no_grad():
                    for p in opt.parameters():
                        mask = torch.rand_like(p) < flip_prob
                        p.data[mask] =  -1 * p.data[mask]
                        # p.data.add_(torch.randn_like(p) * perturb_scale)

                # Forward and backward pass (isolate computation graph)
                optimizer.zero_grad(set_to_none=True)
                results_perturbed = opt.forward(sharpness=1 + 2 * step)

                print(f"Pert {step}:", end=" ")
                for k, obj in results_perturbed["breakdown"].items():
                    print(f"{k}: {obj['value']:.3f}", end=", ")
                print()

                (-results_perturbed["obj"]).backward()

                # Dump data for the perturbed model
                filename_h5 = dump_data_path + f"/mdm_id-{device_id}_opt_step_{step}_perturbed_{i}.h5"
                filename_yml = dump_data_path + f"/mdm_id-{device_id}_perturbed_{i}.yml"
                opt.dump_data(filename_h5=filename_h5, filename_yml=filename_yml, step=step)

                opt.plot(
                    eps_map=opt._eps_map,
                    obj=results["breakdown"]["mode1_trans"]["value"],
                    plot_filename=f"mdm_opt_step_{step}_mode1_fwd_perturbed_{i}.png",
                    field_key=("in_port_1", 1.55, 1, 300),
                    field_component="Ez",
                    in_port_name="in_port_1",
                    exclude_port_names=[],
                )
                opt.plot(
                    eps_map=opt._eps_map,
                    obj=results["breakdown"]["mode2_trans"]["value"],
                    plot_filename=f"mdm_opt_step_{step}_mode2_fwd_perturbed_{i}.png",
                    field_key=("in_port_1", 1.55, 2, 300),
                    field_component="Ez",
                    in_port_name="in_port_1",
                    exclude_port_names=[],
                )

            finally:
                # Restore the original parameters and optimizer state
                with torch.no_grad():
                    for p, original_p in zip(opt.parameters(), original_params):
                        p.copy_(original_p)
                optimizer.load_state_dict(optimizer_state)
                optimizer.zero_grad(set_to_none=True)  # Clear gradients completely


    for step in range(50):
        # for step in range(1):
        optimizer.zero_grad()
        results = opt.forward(sharpness=1 + 4 * step)
        # results = opt.forward(sharpness=256)
        print(f"Step {step}:", end=" ")
        for k, obj in results["breakdown"].items():
            print(f"{k}: {obj['value']:.3f}", end=", ")
        print()

        (-results["obj"]).backward()
        current_design_region_dict = opt.get_design_region_eps_dict()
        filename_h5 = dump_data_path + f"/mdm_id-{device_id}_opt_step_{step}.h5"
        filename_yml = dump_data_path + f"/mdm_id-{device_id}.yml"
        if last_design_region_dict is None:
            opt.dump_data(filename_h5=filename_h5, filename_yml=filename_yml, step=step)
            last_design_region_dict = current_design_region_dict
            dumped_data = True
            opt.plot(
                eps_map=opt._eps_map,
                obj=results["breakdown"]["mode1_trans"]["value"],
                plot_filename="mdm_opt_step_{}_mode1_fwd.png".format(step),
                field_key=("in_port_1", 1.55, 1, 300),
                field_component="Ez",
                in_port_name="in_port_1",
                exclude_port_names=[],
            )
            opt.plot(
                eps_map=opt._eps_map,
                obj=results["breakdown"]["mode2_trans"]["value"],
                plot_filename="mdm_opt_step_{}_mode2_fwd.png".format(step),
                field_key=("in_port_1", 1.55, 2, 300),
                field_component="Ez",
                in_port_name="in_port_1",
                exclude_port_names=[],
            )
        else:
            cosine_similarity = compare_designs(
                last_design_region_dict, current_design_region_dict
            )
            if cosine_similarity < 0.996 or step == 49:
                opt.dump_data(
                    filename_h5=filename_h5, filename_yml=filename_yml, step=step
                )
                last_design_region_dict = current_design_region_dict
                dumped_data = True
                opt.plot(
                    eps_map=opt._eps_map,
                    obj=results["breakdown"]["mode1_trans"]["value"],
                    plot_filename="mdm_opt_step_{}_mode1_fwd.png".format(step),
                    field_key=("in_port_1", 1.55, 1, 300),
                    field_component="Ez",
                    in_port_name="in_port_1",
                    exclude_port_names=[],
                )
                opt.plot(
                    eps_map=opt._eps_map,
                    obj=results["breakdown"]["mode2_trans"]["value"],
                    plot_filename="mdm_opt_step_{}_mode2_fwd.png".format(step),
                    field_key=("in_port_1", 1.55, 2, 300),
                    field_component="Ez",
                    in_port_name="in_port_1",
                    exclude_port_names=[],
                )
        # for p in opt.parameters():
        #     print(p.grad)
        # print_stat(list(opt.parameters())[0], f"step {step}: grad: ")
        optimizer.step()
        scheduler.step()
        if dumped_data:
            for i, prob in enumerate(perturb_probs):
                perturb_and_dump(step, flip_prob=prob, i=i)
            dumped_data = False
            # quit()


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
    mdm_opt(random_seed, device)


if __name__ == "__main__":
    main()
