"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
"""

import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../MAPS"))
sys.path.insert(0, project_root)
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
from pyutils.config import Config
from pyutils.general import logger as lg
from pyutils.torch_train import load_model, set_torch_deterministic

from core.invdes.invdesign import InvDesign
from core.invdes.models import BendingOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import Bending
from core.train import builder
from core.utils import SharpnessScheduler, cal_total_field_adj_src_from_fwd_field
from core.utils import train_configs as configs


class dual_predictor(nn.Module):
    def __init__(self, model_fwd, model_adj, switch_epoch):
        super(dual_predictor, self).__init__()
        self.model_fwd = model_fwd
        self.model_adj = model_adj
        self.switch_epoch = switch_epoch
        print(
            "will swith to predicted field from epoch: ", self.switch_epoch, flush=True
        )

    def forward(
        self,
        data,
        epoch=1,
    ):
        eps = data["eps_map"]
        src = {}
        wl = data["wavelength"]
        mode = data["mode"]
        temp = data["temp"]
        in_slice_name = data["input_slice"]
        src = data["src_profile"]
        fwd_Ez_field = self.model_fwd(
            eps,
            src,
            monitor_slices=data["monitor_slices"],
            monitor_slice_list=None,
            in_slice_name=in_slice_name,
            wl=wl,
            temp=temp,
        )
        with torch.enable_grad():
            fwd_field, adj_source, monitor_slice_list = (
                cal_total_field_adj_src_from_fwd_field(
                    Ez4adj=(
                        fwd_Ez_field
                        if epoch >= self.switch_epoch
                        else data["fwd_field"][:, -2:, ...]
                    ),
                    Ez4fullfield=fwd_Ez_field,
                    # Ez=data["fwd_field"][:, -2:, ...],
                    eps=eps,
                    ht_ms=data[
                        "ht_m"
                    ],  # this two only used for adjoint field calculation, we don't need it here in forward pass
                    et_ms=data["et_m"],
                    monitors=data["monitor_slices"],
                    pml_mask=self.model_fwd.pml_mask,
                    return_adj_src=True,
                    sim=self.model_fwd.sim,
                    opt_cfg_file_path=data["opt_cfg_file_path"],
                    wl=wl,
                    mode=mode,
                    temp=temp,
                    src_in_slice_name=in_slice_name,
                )
            )
        adj_source = adj_source.detach()
        adj_Ez_field = self.model_adj(
            eps,
            adj_source,
            monitor_slices=data["monitor_slices"],
            monitor_slice_list=monitor_slice_list,
            in_slice_name=in_slice_name,
            wl=wl,
            temp=temp,
        )
        adj_field, _, _ = cal_total_field_adj_src_from_fwd_field(
            Ez4adj=adj_Ez_field,
            Ez4fullfield=adj_Ez_field,
            eps=eps,
            ht_ms=data["ht_m"],
            et_ms=data["et_m"],
            monitors=data["monitor_slices"],
            pml_mask=self.model_adj.pml_mask,
            return_adj_src=False,
            sim=self.model_adj.sim,
            opt_cfg_file_path=data["opt_cfg_file_path"],
            wl=wl,
            mode=mode,
            temp=temp,
            src_in_slice_name=in_slice_name,
        )
        return {
            "forward_field": fwd_field,
            "adjoint_field": adj_field,
            "adjoint_source": adj_source,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive=True)
    configs.update(opts)
    # Convert tuple strings to actual tuples
    if hasattr(configs.model_fwd, "mode_list"):
        if configs.model_fwd.type != "FNO2d":
            assert hasattr(
                configs.model_fwd, "kernel_list"
            ), "kernel_list should be defined if mode_list is defined"
            configs["model_fwd"]["mode_list"] = [(50, 50)] * len(
                configs["model_fwd"]["kernel_list"]
            )
        else:
            configs["model_fwd"]["mode_list"] = [(50, 50)] * 4
    if hasattr(configs.model_adj, "mode_list"):
        if configs.model_adj.type != "FNO2d":
            assert hasattr(
                configs.model_adj, "kernel_list"
            ), "kernel_list should be defined if mode_list is defined"
            configs["model_adj"]["mode_list"] = [(50, 50)] * len(
                configs["model_adj"]["kernel_list"]
            )
        else:
            configs["model_adj"]["mode_list"] = [(50, 50)] * 4
    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
        print("cuda is available and set to device: ", device, flush=True)
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False
    print("this is the config: \n", configs, flush=True)
    if int(configs.run.deterministic) == True:
        set_torch_deterministic(int(configs.run.random_state))

    configs.model_fwd.device = device
    model_fwd = builder.make_model(**configs.model_fwd)
    print("this is the model: \n", model_fwd, flush=True)
    configs.model_adj.device = device
    model_adj = builder.make_model(**configs.model_adj)
    print("this is the model: \n", model_adj, flush=True)
    switch_epoch = int(getattr(configs.run, "switch_epoch", 1))
    model = dual_predictor(model_fwd, model_adj, switch_epoch)

    # load model:
    if (
        int(configs.checkpoint.resume)
        and len(configs.checkpoint.restore_checkpoint) > 0
    ):
        load_model(
            model,
            configs.checkpoint.restore_checkpoint,
            ignore_size_mismatch=int(configs.checkpoint.no_linear),
        )

    neural_solver = {
        "fwd_solver": model.model_fwd,
        "adj_solver": model.model_adj,
    }

    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()
    sim_cfg_gt = DefaultSimulationConfig()

    bending_region_size = (1.6, 1.6)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.48

    # sim_cfg.update(
    #     dict(
    #         solver="ceviche_torch",
    #         border_width=[0, port_len, port_len, 0],
    #         resolution=50,
    #         plot_root=f"./figs/bending_{'nn_solver'}",
    #         PML=[0.5, 0.5],
    #         neural_solver=None,
    #         numerical_solver="solve_direct",
    #         use_autodiff=False,
    #     )
    # )
    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, port_len, port_len, 0],
            resolution=50,
            plot_root=f"./figs/bending_{'nn_solver'}",
            PML=[0.5, 0.5],
            neural_solver=neural_solver,
            numerical_solver="none",
            use_autodiff=False,
        )
    )
    sim_cfg_gt.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, port_len, port_len, 0],
            resolution=50,
            plot_root=f"./figs/bending_{'nn_solver'}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )

    device = Bending(
        sim_cfg=sim_cfg,
        bending_region_size=bending_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )
    device_gt = Bending(
        sim_cfg=sim_cfg_gt,
        bending_region_size=bending_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )

    hr_device = device.copy(resolution=310)
    hr_device_gt = device_gt.copy(resolution=310)
    print(device)

    opt = BendingOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    opt_gt = BendingOptimization(
        device=device_gt,
        hr_device=hr_device_gt,
        sim_cfg=sim_cfg_gt,
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
    fwd_trans_NN = []
    fwd_trans_GT = []
    for step in range(n_epoch):
        # for step in range(1):
        optimizer.zero_grad()
        sharpness = sharp_scheduler.get_sharpness()
        results = opt.forward(sharpness=sharpness)
        # results = opt.forward(sharpness=256)
        print(f"Step {step}:", end=" ")
        for k, obj in results["breakdown"].items():
            print(f"{k}: {obj['value']:.3f}", end=", ")
        print()
        fwd_trans_NN.append(results["breakdown"]["fwd_trans"]["value"].item())

        (-results["obj"]).backward()

        opt.plot(
            eps_map=opt._eps_map,
            obj=results["breakdown"]["fwd_trans"]["value"],
            plot_filename="bending_opt_step_{}_fwd_NN.png".format(step),
            field_key=("in_slice_1", 1.55, 1, 300),
            field_component="Ez",
            in_slice_name="in_slice_1",
            exclude_slice_names=[],
        )
        # copy the parameters to the gt model
        for p, p_gt in zip(opt.parameters(), opt_gt.parameters()):
            p_gt.data = p.data.clone().detach()

        results_gt = opt_gt.forward(sharpness=sharpness)
        print(f"GT Step {step}:", end=" ")
        for k, obj in results_gt["breakdown"].items():
            print(f"{k}: {obj['value']:.3f}", end=", ")
        print()
        fwd_trans_GT.append(results_gt["breakdown"]["fwd_trans"]["value"].item())
        opt_gt.plot(
            eps_map=opt_gt._eps_map,
            obj=results_gt["breakdown"]["fwd_trans"]["value"],
            plot_filename="bending_opt_step_{}_fwd_GT.png".format(step),
            field_key=("in_slice_1", 1.55, 1, 300),
            field_component="Ez",
            in_slice_name="in_slice_1",
            exclude_slice_names=[],
        )

        optimizer.step()
        scheduler.step()
        sharp_scheduler.step()

    # save the results to csv file:

    fwd_trans_GT = np.array(fwd_trans_GT)
    fwd_trans_NN = np.array(fwd_trans_NN)
    file_path = "./unitest/nn_as_solver.csv"
    # Write to CSV using csv module
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["fwd_trans_GT", "fwd_trans_NN"])
        # Write rows
        for gt, nn in zip(fwd_trans_GT, fwd_trans_NN):
            writer.writerow([gt, nn])
