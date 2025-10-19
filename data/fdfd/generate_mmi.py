import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse
import random

import torch
import torch.nn.functional as F

from core.invdes.models import MMIOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MMI
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


def mmi_opt(
    device_id,
    operation_device,
    each_step=False,
    include_perturb=False,
    perturb_probs=[0.05, 0.1, 0.15],
    mfs=0.05,
):
    set_torch_deterministic(int(device_id))
    device_name = "mmi"
    dump_data_path = f"./data/fdfd/{device_name}/raw_opt_traj_ptb"
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
    num_inports = 3
    num_outports = 3

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, port_len, port_len],
            resolution=resolution,
            plot_root=f"./data/fdfd/{device_name}/plot_opt_traj_ptb/{device_name}_{device_id}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=1.55,
            wl_width=0,
            n_wl=1,
        )
    )

    device = MMI(
        sim_cfg=sim_cfg,
        box_size=wdm_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        num_inports=num_inports,
        num_outports=num_outports,
        port_box_margin=1,
        device=operation_device,
    )
    hr_device = device.copy(resolution=310)
    print(device)

    target_unitary_smatrix = torch.randn(
        (num_inports, num_outports), dtype=torch.complex64, device=operation_device
    )
    u, s, v = torch.linalg.svd(target_unitary_smatrix)
    target_unitary_smatrix = torch.matmul(u, v) * 0.95
    target_norm = torch.norm(target_unitary_smatrix)
    print(f"target unitary smatrix:\n{target_unitary_smatrix}")
    """ e.g., target unitary smatrix:
    [[-0.1627-0.6450j,  0.6381-0.2135j, -0.0702-0.0481j],
     [-0.5426-0.0271j, -0.0188+0.5920j, -0.5033+0.0567j],
     [-0.2311+0.3339j,  0.2916+0.1178j,  0.2743-0.7506j]]

    trained matrix, 0.0581 relative L2 norm error
    [[-0.1511-0.6146j,  0.6196-0.2099j, -0.0684-0.0495j,]
     [-0.4896-0.0319j, -0.0191+0.5528j, -0.4856+0.0568j,]
     [-0.2111+0.2967j,  0.2755+0.1021j,  0.2669-0.7242j]
    """

    def fom_func(breakdown):
        ## maximization fom
        for key, obj in breakdown.items():
            if key == "smatrix":
                s_matrix = obj["value"]  # shape (num_outports, num_inports)
                fom = (
                    -torch.norm(s_matrix - target_unitary_smatrix.flatten())
                    / target_norm
                )
        return fom, {"smatrix_err": {"weight": -1, "value": fom}}

    obj_cfgs = dict(_fusion_func=fom_func)

    opt = MMIOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)
    for region_name in device.design_region_cfgs:
        opt.design_region_param_cfgs[region_name]["transform"] = [
                dict(
                type="fft",
                mfs=mfs,
                resolutions=[hr_device.resolution, hr_device.resolution],
                dim="xy",
            ),
            dict(type="binarize"),
        ]
        
    print(opt)
    n_epoch = 1
    optimizer = torch.optim.Adam(opt.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epoch, eta_min=0.0002
    )
    sharp_scheduler = SharpnessScheduler(
        initial_sharp=256, final_sharp=256, total_steps=n_epoch
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
                        p.data[mask] = -1 * p.data[mask]
                        # p.data.add_(torch.randn_like(p) * perturb_scale)

                # Forward and backward pass (isolate computation graph)
                optimizer.zero_grad(set_to_none=True)
                results_perturbed = opt.forward(sharpness=1 + 2 * step)

                print(f"Pert {step}:", end=" ")
                for k, obj in results_perturbed["breakdown"].items():
                    print(f"{k}: {obj['value']}", end=", ")
                print()

                (-results_perturbed["obj"]).backward()

                # Dump data for the perturbed model
                filename_h5 = (
                    dump_data_path
                    + f"/{device_name}_id-{device_id}_opt_step_{step}_perturbed_{i}.h5"
                )
                filename_yml = dump_data_path + f"/{device_name}_id-{device_id}_perturbed_{i}.yml"
                opt.dump_data(
                    filename_h5=filename_h5, filename_yml=filename_yml, step=step
                )

                for in_port_idx in range(1, num_inports + 1):
                    opt.plot(
                        eps_map=opt._eps_map,
                        obj=results["breakdown"]["smatrix_err"]["value"],
                        plot_filename=f"{device_name}_opt_step_{step}_s{in_port_idx}_perturbed_{i}.png",
                        field_key=(f"in_slice_{in_port_idx}", 1.55, "Ez1", 300),
                        field_component="Ez",
                        in_slice_name=f"in_slice_{in_port_idx}",
                        exclude_slice_names=[],
                    )

            finally:
                # Restore the original parameters and optimizer state
                with torch.no_grad():
                    for p, original_p in zip(opt.parameters(), original_params):
                        p.copy_(original_p)
                optimizer.load_state_dict(optimizer_state)
                optimizer.zero_grad(set_to_none=True)  # Clear gradients completely

    early_stop_threshold = 1e-3  # Define a threshold for detecting convergence
    patience = 3  # Number of epochs to wait for changes before stopping
    breakdown_history = []  # To store the breakdown history

    for step in range(n_epoch):
        # for step in range(1):
        optimizer.zero_grad()
        sharpness = sharp_scheduler.get_sharpness()
        results = opt.forward(sharpness=sharpness)
        # results = opt.forward(sharpness=256)
        print(f"Step {step}:", end=" ")
        for k, obj in results["breakdown"].items():
            print(f"{k}: {obj['value']}", end=", ")
        print()

        (-results["obj"]).backward()
        current_design_region_dict = opt.get_design_region_eps_dict()
        filename_h5 = dump_data_path + f"/{device_name}_id-{device_id}_opt_step_{step}.h5"
        filename_yml = dump_data_path + f"/{device_name}_id-{device_id}.yml"

        # Store the current breakdown for early stopping
        current_breakdown = {k: obj["value"] for k, obj in results["breakdown"].items() if not (isinstance(obj["value"], torch.Tensor) and obj["value"].numel() > 1)}
        breakdown_history.append(current_breakdown)

        # Keep only the last `patience` results in the history
        if len(breakdown_history) > patience:
            breakdown_history.pop(0)

        # Check for convergence
        if len(breakdown_history) == patience:
            changes = [
                max(
                    abs(current_breakdown[k] - previous_breakdown[k])
                    for k in current_breakdown.keys()
                )
                for previous_breakdown in breakdown_history[:-1]
            ]
            if all(change < early_stop_threshold for change in changes):
                print(
                    f"Early stopping at step {step}: No significant changes in {patience} epochs."
                )
                break

        if last_design_region_dict is None:
            opt.dump_data(filename_h5=filename_h5, filename_yml=filename_yml, step=step)
            last_design_region_dict = current_design_region_dict
            dumped_data = True
            for in_port_idx in range(1, num_inports + 1):
                opt.plot(
                    eps_map=opt._eps_map,
                    obj=results["breakdown"]["smatrix_err"]["value"],
                    plot_filename=f"{device_name}_opt_step_{step}_s{in_port_idx}.png",
                    field_key=(f"in_slice_{in_port_idx}", 1.55, "Ez1", 300),
                    field_component="Ez",
                    in_slice_name=f"in_slice_{in_port_idx}",
                    exclude_slice_names=[],
                )
        else:
            cosine_similarity = compare_designs(
                last_design_region_dict, current_design_region_dict
            )
            if cosine_similarity < 0.996 or step == n_epoch - 1 or each_step:
                opt.dump_data(
                    filename_h5=filename_h5, filename_yml=filename_yml, step=step
                )
                last_design_region_dict = current_design_region_dict
                dumped_data = True
                for in_port_idx in range(1, num_inports + 1):
                    opt.plot(
                        eps_map=opt._eps_map,
                        obj=results["breakdown"]["smatrix_err"]["value"],
                        plot_filename=f"{device_name}_opt_step_{step}_s{in_port_idx}.png",
                        field_key=(f"in_slice_{in_port_idx}", 1.55, "Ez1", 300),
                        field_component="Ez",
                        in_slice_name=f"in_slice_{in_port_idx}",
                        exclude_slice_names=[],
                    )
        # for p in opt.parameters():
        #     print(p.grad)
        # print_stat(list(opt.parameters())[0], f"step {step}: grad: ")
        optimizer.step()
        scheduler.step()
        sharp_scheduler.step()
        if dumped_data and include_perturb:
            for i, prob in enumerate(perturb_probs):
                perturb_and_dump(step, flip_prob=prob, i=i)
            dumped_data = False
        #     # quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--each_step", type=int, default=0)
    parser.add_argument("--include_perturb", type=int, default=0)
    parser.add_argument("--mfs", type=float, default=0.05)
    args = parser.parse_args()
    random_seed = args.random_seed
    gpu_id = args.gpu_id
    each_step = args.each_step
    include_perturb = args.include_perturb
    mfs = args.mfs
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + random_seed))
    mmi_opt(random_seed, device, each_step, include_perturb, mfs=mfs)


if __name__ == "__main__":
    main()
