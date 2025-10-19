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
import argparse
import random

import torch
import torch.nn.functional as F

from autograd.numpy import array as npa
from pyutils.config import Config

from core.invdes.invdesign import InvDesign
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
    perturb_probs=[0.1, 0.3, 0.5],
):
    # sim_cfg = DefaultSimulationConfig()

    # mmi_region_size = (4, 4)
    # target_img_size = 256
    # target_mmi_size = (4, 4)
    # resolution = 50
    # # target_cell_size = target_img_size / resolution
    # target_region_size = target_mmi_size * resolution
    # ## may calculated by the mmi size
    # port_len = round(random.uniform(1.6, 1.8) * resolution) / resolution
    # target_cell_size = target_mmi_size[0] + port_len


    # input_port_width = 0.48
    # output_port_width = 0.48
    # exp_name = "mmi_opt"

    # sim_cfg.update(
    #     dict(
    #         solver="ceviche_torch",
    #         border_width=[0, 0, 1, 1],  # left, right, lower, upper, containing PML
    #         resolution=100,
    #         plot_root=f"./figs/{exp_name}",
    #         PML=[0.5, 0.5],
    #         neural_solver=None,
    #         numerical_solver="solve_direct",
    #         use_autodiff=False,
    #     )
    # )
    # num_inports = 4
    # num_outports = 4
    set_torch_deterministic(int(device_id))
    dump_data_path = f"./data/fdfd/mmi/raw_opt_traj_ptb"
    sim_cfg = DefaultSimulationConfig()

    mmi_region_size = (4, 4)
    # mmi_region_size = (6, 6
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.48
    exp_name = "mmi_opt"

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 1, 1],  # left, right, lower, upper, containing PML
            resolution=100,
            plot_root=f"./figs/{exp_name}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )
    num_inports = 3
    num_outports = 3

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
        # for key, obj in breakdown.items():
        #     if "fwd_trans" in key:
        #         fom = fom * obj["value"]
        #     elif "phase" in key:
        #         s_param_list.append(obj["value"])
        # if len(s_param_list) == 0:
        #     return fom
        # if isinstance(s_param_list[0], torch.Tensor):
        #     s_params = torch.tensor(s_param_list)
        #     s_params_std = torch.std(s_params)
        #     fom = fom - s_params_std
        # else:
        #     s_params = npa.array(s_param_list)
        #     s_params_std = npa.std(s_params)
        #     fom = fom - s_params_std
        # maximize the forward transmission and minimize the standard deviation of the s-parameters
        return fom, {"smatrix_err": {"weight": -1, "value": fom}}

    obj_cfgs = dict(_fusion_func=fom_func)

    device = MMI(
        sim_cfg=sim_cfg,
        box_size=mmi_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        num_inports=num_inports,
        num_outports=num_outports,
        device=operation_device,
        port_box_margin=0.5,
    )

    hr_device = device.copy(resolution=1000)
    print(device)
    opt = MMIOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
        obj_cfgs=obj_cfgs,
    ).to(operation_device)

    print(opt)
    n_epoch = 2
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
                    print(f"{k}: {obj['value']:.3f}", end=", ")
                print()

                (-results_perturbed["obj"]).backward()

                # Dump data for the perturbed model
                filename_h5 = (
                    dump_data_path
                    + f"/bending_id-{device_id}_opt_step_{step}_perturbed_{i}.h5"
                )
                filename_yml = (
                    dump_data_path + f"/bending_id-{device_id}_perturbed_{i}.yml"
                )
                opt.dump_data(
                    filename_h5=filename_h5, filename_yml=filename_yml, step=step
                )

                opt.plot(
                    eps_map=opt._eps_map,
                    obj=results["breakdown"]["fwd_trans"]["value"],
                    plot_filename=f"bending_opt_step_{step}_fwd_perturbed_{i}.png",
                    field_key=("in_slice_1", 1.55, "Ez1", 300),
                    field_component="Ez",
                    in_slice_name="in_slice_1",
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
        # print(f"Step {step}:", end=" ")
        # for k, obj in results["breakdown"].items():
        #     print(f"{k}: {obj['value']:.3f}", end=", ")
        # print()

        (-results["obj"]).backward()
        current_design_region_dict = opt.get_design_region_eps_dict()
        filename_h5 = dump_data_path + f"/mmi_id-{device_id}_opt_step_{step}.h5"
        filename_yml = dump_data_path + f"/mmi_id-{device_id}.yml"

        # Store the current breakdown for early stopping
        current_breakdown = {k: obj["value"] for k, obj in results["breakdown"].items()}
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
            # opt.plot(
            #     eps_map=opt._eps_map,
            #     obj=results["breakdown"]["fwd_trans"]["value"],
            #     plot_filename="bending_opt_step_{}_fwd.png".format(step),
            #     field_key=("in_slice_1", 1.55, "Ez1", 300),
            #     field_component="Ez",
            #     in_slice_name="in_slice_1",
            #     exclude_slice_names=[],
            # )
        else:
            cosine_similarity = compare_designs(
                last_design_region_dict, current_design_region_dict
            )
            if cosine_similarity < 0.9 or step == n_epoch - 1 or each_step or step % 10 == 0:
                opt.dump_data(
                    filename_h5=filename_h5, filename_yml=filename_yml, step=step
                )
                last_design_region_dict = current_design_region_dict
                dumped_data = True
                # opt.plot(
                #     eps_map=opt._eps_map,
                #     obj=results["breakdown"]["fwd_trans"]["value"],
                #     plot_filename="bending_opt_step_{}_fwd.png".format(step),
                #     field_key=("in_slice_1", 1.55, "Ez1", 300),
                #     field_component="Ez",
                #     in_slice_name="in_slice_1",
                #     exclude_slice_names=[],
                # )
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
    random_seed = parser.parse_args().random_seed
    gpu_id = parser.parse_args().gpu_id
    each_step = parser.parse_args().each_step
    include_perturb = parser.parse_args().include_perturb
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + random_seed))
    mmi_opt(random_seed, device, each_step, include_perturb)

if __name__ == "__main__":
    main()