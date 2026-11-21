import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Make sure project root is importable (same as user's wrapper does)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from pyutils.config import Config
from pyutils.general import logger as lg

from core.invdes.models import MMIOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.four_port_coupler import FourPortOptimization

# Import our device
from core.invdes.models.layers.four_port_device import FourPortCoupler
from core.utils import DeterministicCtx, SharpnessScheduler, set_torch_deterministic
from thirdparty.ceviche.constants import *


def compare_designs(design_regions_1, design_regions_2):
    sims = []
    for k in design_regions_1.keys():
        v1 = design_regions_1[k].flatten()
        v2 = design_regions_2[k].flatten()
        sims.append(F.cosine_similarity(v1, v2, dim=0))
    return torch.mean(torch.stack(sims)).item()


def fourport_opt(
    device_id,
    operation_device,
    each_step=False,
    include_perturb=False,
    perturb_probs=[0.05, 0.1, 0.15],
):
    set_torch_deterministic(int(device_id))
    dump_data_path = f"./data/fdfd/four_port/raw_opt_traj_ptb/mfs_50"

    port_len = 1.8
    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 0, 0],
            resolution=100,
            plot_root=f"./data/fdfd/four_port/plot_opt_traj_ptb/mfs_50/four_port_{device_id}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=1.55,
            wl_width=0.0,
            n_wl=1,
        )
    )

    # Build device (box and ports similar to MMI)
    device = FourPortCoupler(
        sim_cfg=sim_cfg,
        box_size=(4.0, 4.0),
        port_len=(port_len, port_len),
        port_width=(0.48, 0.48),
        device=operation_device,
        port_box_margin=0.5,
    )
    hr_device = device.copy(resolution=1000)

    # ----- Objective: equal power split to three outputs -----
    def fom_func(breakdown):
        for key, obj in breakdown.items():
            if key == "smatrix":
                s_matrix = obj["value"]  # shape (num_inports, num_outports)
                power = torch.abs(s_matrix) ** 2
                # maximize mean power and penalize std to enforce equal split
                mean_p = torch.mean(power)
                std_p = torch.std(power)
                fom = mean_p - 0.5 * std_p
        return fom, {
            "sum_T": {"weight": 1.0, "value": mean_p},
            "std_T": {"weight": -0.5, "value": -std_p},
        }

    obj_cfgs = dict(_fusion_func=fom_func)

    opt = FourPortOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)

    n_epoch = 100
    optimizer = torch.optim.Adam(opt.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epoch, eta_min=0.0002
    )
    sharp_scheduler = SharpnessScheduler(
        initial_sharp=1, final_sharp=256, total_steps=n_epoch
    )

    last_design_region_dict = None
    dumped_data = False

    def _format_breakdown(breakdown):
        parts = []
        scalar_values = {}
        for k, obj in breakdown.items():
            value = obj["value"]
            if torch.is_tensor(value):
                if value.numel() == 1:
                    scalar = value.detach().item()
                    parts.append(f"{k}: {scalar:.3f}")
                    scalar_values[k] = scalar
                else:
                    mean_val = value.detach().mean().item()
                    parts.append(f"{k}: {mean_val:.3f} (mean)")
            else:
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    parts.append(f"{k}: <non-scalar>")
                else:
                    parts.append(f"{k}: {scalar:.3f}")
                    scalar_values[k] = scalar
        return parts, scalar_values

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

                parts, _ = _format_breakdown(results_perturbed["breakdown"])
                print(f"Pert {step}: " + ", ".join(parts))

                (-results_perturbed["obj"]).backward()

                # Dump data for the perturbed model
                filename_h5 = (
                    dump_data_path
                    + f"/four_port_id-{device_id}_opt_step_{step}_perturbed_{i}.h5"
                )
                filename_yml = (
                    dump_data_path + f"/four_port_id-{device_id}_perturbed_{i}.yml"
                )
                opt.dump_data(
                    filename_h5=filename_h5, filename_yml=filename_yml, step=step
                )

                opt.plot(
                    eps_map=opt._eps_map,
                    obj=results_perturbed["obj"],
                    plot_filename=(f"four_port_opt_step_{step}_fwd_perturbed_{i}.png"),
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
        parts, scalar_breakdown = _format_breakdown(results["breakdown"])
        print(f"Step {step}: " + ", ".join(parts))

        (-results["obj"]).backward()
        current_design_region_dict = opt.get_design_region_eps_dict()
        filename_h5 = dump_data_path + f"/four_port_id-{device_id}_opt_step_{step}.h5"
        filename_yml = dump_data_path + f"/four_port_id-{device_id}.yml"

        # Store the current breakdown for early stopping
        if scalar_breakdown:
            breakdown_history.append(scalar_breakdown)

        # Keep only the last `patience` results in the history
        if len(breakdown_history) > patience:
            breakdown_history.pop(0)

        # Check for convergence
        if len(breakdown_history) == patience:
            changes = []
            for previous_breakdown in breakdown_history[:-1]:
                shared_keys = set(previous_breakdown.keys()) & set(
                    breakdown_history[-1].keys()
                )
                if not shared_keys:
                    continue
                change = max(
                    abs(breakdown_history[-1][k] - previous_breakdown[k])
                    for k in shared_keys
                )
                changes.append(change)
            if changes and all(change < early_stop_threshold for change in changes):
                print(
                    f"Early stopping at step {step}: No significant changes in {patience} epochs."
                )
                break

        if last_design_region_dict is None:
            opt.dump_data(filename_h5=filename_h5, filename_yml=filename_yml, step=step)
            last_design_region_dict = current_design_region_dict
            dumped_data = True
            opt.plot(
                eps_map=opt._eps_map,
                obj=results["obj"],
                plot_filename=f"four_port_opt_step_{step}_fwd.png",
                field_key=("in_slice_1", 1.55, "Ez1", 300),
                field_component="Ez",
                in_slice_name="in_slice_1",
                exclude_slice_names=[],
            )
        else:
            cosine_similarity = compare_designs(
                last_design_region_dict, current_design_region_dict
            )
            if (
                cosine_similarity < 0.996
                or step == n_epoch - 1
                or each_step
                or step % 5 == 0
            ):
                opt.dump_data(
                    filename_h5=filename_h5,
                    filename_yml=filename_yml,
                    step=step,
                    use_high_res_eps=True,
                    binarize_eps=True,
                )
                last_design_region_dict = current_design_region_dict
                dumped_data = True
                opt.plot(
                    eps_map=opt._eps_map,
                    obj=results["obj"],
                    plot_filename=f"four_port_opt_step_{step}_fwd.png",
                    field_key=("in_slice_1", 1.55, "Ez1", 300),
                    field_component="Ez",
                    in_slice_name="in_slice_1",
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
    random_seed = parser.parse_args().random_seed
    gpu_id = parser.parse_args().gpu_id
    each_step = parser.parse_args().each_step
    include_perturb = parser.parse_args().include_perturb
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + random_seed))
    fourport_opt(random_seed, device, each_step, include_perturb)


if __name__ == "__main__":
    main()
