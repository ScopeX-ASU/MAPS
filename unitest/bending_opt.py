import os
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from ceviche import fdfd_ez as ceviche_fdfd_ez
from ceviche.constants import *
from pyutils.general import print_stat
from pyutils.config import configs
from core.models import (
    IsolatorOptimization,
    MetaCouplerOptimization,
    MetaMirrorOptimization,
    BendingOptimization,
)
from core.models.base_optimization import BaseOptimization, DefaultSimulationConfig
from core.models.fdfd.fdfd import fdfd_ez
from core.models.fdfd.utils import torch_sparse_to_scipy_sparse
from core.models.layers import Isolator, MetaCoupler, MetaMirror, Bending
from core.models.layers.device_base import N_Ports, Si_eps
from core.models.layers.utils import plot_eps_field
from core.utils import set_torch_deterministic
from torch_sparse import spspmm
from core import builder
import argparse
import random
from pyutils.general import AverageMeter, logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)

def compare_designs(design_regions_1, design_regions_2):
    similarity = []
    for k, v in design_regions_1.items():
        v1 = v
        v2 = design_regions_2[k]
        similarity.append(F.cosine_similarity(v1.flatten(), v2.flatten(), dim=0))
    return torch.mean(torch.stack(similarity)).item()

def bending_opt(
        device_id, 
        operation_device,
        neural_solver=None,
        numerical_solver="solve_direct",
    ):
    sim_cfg = DefaultSimulationConfig()

    bending_region_size = (1.6, 1.6)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.48

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            neural_solver=neural_solver,
            numerical_solver=numerical_solver,
            border_width=[0, port_len, port_len, 0],
            resolution=50,
            plot_root=f"./figs/test_mfs_bending_{device_id}",
            PML=[0.5, 0.5],
        )
    )

    device = Bending(
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
    opt = BendingOptimization(device=device, hr_device=hr_device, sim_cfg=sim_cfg, operation_device=operation_device).to(operation_device)
    print(opt)
    init_lr = 1e4
    optimizer = torch.optim.Adam(opt.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=70, eta_min=init_lr*0.01
    )
    last_design_region_dict = None
    for step in range(10):
    # for step in range(1):
        optimizer.zero_grad()
        results = opt.forward(sharpness=1 + 2 * step)
        # results = opt.forward(sharpness=256)
        print(f"Step {step}:", end=" ")
        for k, obj in results["breakdown"].items():
            print(f"{k}: {obj['value']:.3f}", end=", ")
        print()

        (-results["obj"]).backward()
        if neural_solver is not None:
            for p in opt.parameters():
                if p.grad is not None:
                    max_grad = p.grad.data.abs().max()  # Get the maximum absolute gradient value
                    if max_grad > 1e3:  # Only scale if the maximum exceeds the threshold
                        scale_factor = 1e3 / max_grad  # Compute the scale factor
                        p.grad.data.mul_(scale_factor)  # Scale the gradient

        current_design_region_dict = opt.get_design_region_eps_dict()
        filename_h5 = f"./data/fdfd/bending/mfs_raw_test/bending_id-{device_id}_opt_step_{step}.h5"
        filename_yml = f"./data/fdfd/bending/mfs_raw_test/bending_id-{device_id}.yml"
        if last_design_region_dict is None:
            # opt.dump_data(filename_h5=filename_h5, filename_yml=filename_yml, step=step)
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
                # opt.dump_data(filename_h5=filename_h5, filename_yml=filename_yml, step=step)
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
        optimizer.step()
        scheduler.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
        print("cuda is available and set to device: ", device, flush=True)
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic(int(configs.run.random_state))

    model_fwd = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
    )
    lg.info(model_fwd)
    if model_fwd.train_field == "adj":
        assert not configs.run.include_adjoint_NN, "when only adj field is trained, we should not include another adjoint NN"

    if configs.run.include_adjoint_NN:
        model_adj = builder.make_model(
            device,
            int(configs.run.random_state) if int(configs.run.deterministic) else None,
        )
        model_adj.train_field = "adj"
        lg.info(model_adj)
    else:
        model_adj = None

    if (
        int(configs.checkpoint.resume)
        and len(configs.checkpoint.restore_checkpoint_fwd) > 0
        and len(configs.checkpoint.restore_checkpoint_adj) > 0
    ):
        load_model(
            model_fwd,
            configs.checkpoint.restore_checkpoint_fwd,
            ignore_size_mismatch=int(configs.checkpoint.no_linear),
        )
        if model_adj is not None:
            load_model(
                model_adj,
                configs.checkpoint.restore_checkpoint_adj,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )

    bending_opt(
        int(configs.run.random_state), 
        device,
        neural_solver={"fwd_solver": model_fwd, "adj_solver": model_adj},
        numerical_solver="none",
    )

if __name__ == "__main__":
    main()
