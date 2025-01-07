import argparse
import copy
import os
import random
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter, print_stat
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from torch_sparse import spspmm

from core import builder
from core.models import (
    BendingOptimization,
    IsolatorOptimization,
    MetaCouplerOptimization,
    MetaMirrorOptimization,
)
from core.models.base_optimization import BaseOptimization, DefaultSimulationConfig
from core.models.fdfd.fdfd import fdfd_ez
from core.models.fdfd.utils import torch_sparse_to_scipy_sparse
from core.models.layers import Bending, Isolator, MetaCoupler, MetaMirror
from core.models.layers.device_base import N_Ports, Si_eps
from core.models.layers.utils import plot_eps_field
from core.utils import set_torch_deterministic
from thirdparty.ceviche import fdfd_ez as ceviche_fdfd_ez
from thirdparty.ceviche.constants import *
from thirdparty.ceviche.utils import make_sparse

DEFAULT_ITERATIVE_METHOD = "lgmres"

# dict of iterative methods supported (name: function)
ITERATIVE_METHODS = {
    "bicg": spl.bicg,
    "bicgstab": spl.bicgstab,
    "cg": spl.cg,
    "cgs": spl.cgs,
    "gmres": spl.gmres,
    "lgmres": spl.lgmres,
    "qmr": spl.qmr,
    "gcrotmk": spl.gcrotmk,
}

# convergence tolerance for iterative solvers.
ATOL = 1e-8

""" ========================== SOLVER FUNCTIONS ========================== """


def solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD, **kwargs):
    """Iterative solver"""

    # error checking on the method name (https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
    try:
        solver_fn = ITERATIVE_METHODS[iterative_method]
    except:
        raise ValueError(
            "iterative method {} not found.\n supported methods are:\n {}".format(
                iterative_method, ITERATIVE_METHODS
            )
        )

    # call the solver using scipy's API
    x, info = solver_fn(A, b, atol=ATOL, **kwargs)
    return x


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
    use_autodiff=False,
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
            use_autodiff=use_autodiff,
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
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )
    hr_device = device.copy(resolution=310)
    print(device)
    opt = BendingOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    print(opt)
    # init_lr = 1e4
    init_lr = 2e-2
    optimizer = torch.optim.Adam(opt.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=70, eta_min=init_lr * 0.01
    )
    fwd_trans = []
    fwd_trans_gt = []
    time_list = []
    for step in range(10):
        # for step in range(1):
        optimizer.zero_grad()
        with torch.no_grad():
            opt.switch_solver(None, "solve_direct", False)
            results_gt = opt.forward(sharpness=1 + 2 * step)
            print(f"***Step {step}:", end=" ")
            for k, obj in results_gt["breakdown"].items():
                print(f"{k}: {obj['value']:.3f}", end=", ")
            print()
            fwd_trans_gt.append(
                results_gt["breakdown"]["fwd_trans"]["value"].detach().cpu().numpy()
            )
            opt.switch_solver(neural_solver, numerical_solver, use_autodiff)
            opt.plot(
                eps_map=opt._eps_map,
                obj=results_gt["breakdown"]["fwd_trans"]["value"],
                plot_filename="bending_opt_step_{}_fwd_GT.png".format(step),
                field_key=("in_port_1", 1.55, 1),
                field_component="Ez",
                in_port_name="in_port_1",
                exclude_port_names=["refl_port_2"],
            )
        start_time = time.time()
        results = opt.forward(sharpness=1 + 2 * step)
        # results = opt.forward(sharpness=256)
        print(f"Step {step}:", end=" ")
        for k, obj in results["breakdown"].items():
            print(f"{k}: {obj['value']:.3f}", end=", ")
        print()
        fwd_trans.append(
            results["breakdown"]["fwd_trans"]["value"].detach().cpu().numpy()
        )
        (-results["obj"]).backward()
        end_time = time.time()
        time_list.append(end_time - start_time)
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
        #     print("this is the grad", p.grad, flush=True)
        if neural_solver is not None and numerical_solver == "none":
            for p in opt.parameters():
                if p.grad is not None:
                    max_grad = (
                        p.grad.data.abs().max()
                    )  # Get the maximum absolute gradient value
                    if (
                        max_grad > 1e3
                    ):  # Only scale if the maximum exceeds the threshold
                        scale_factor = 1e3 / max_grad  # Compute the scale factor
                        p.grad.data.mul_(scale_factor)  # Scale the gradient

        optimizer.step()
        scheduler.step()
    fwd_trans = np.array(fwd_trans)
    fwd_trans_gt = np.array(fwd_trans_gt)
    time_list = np.array(time_list)
    fwd_trans_tot = np.stack([fwd_trans, fwd_trans_gt], axis=1)
    # save it to a csv file
    np.savetxt(
        f"./unitest/fwd_trans.csv",
        fwd_trans_tot,
        delimiter=",",
        header="Pred_fwd_trans,GT_fwd_trans",
        comments="",
    )
    np.savetxt(
        f"./unitest/time.csv", time_list, delimiter=",", header="time", comments=""
    )


def test_speed(
    device_id,
    operation_device,
    neural_solver=None,
    numerical_solver="solve_direct",
    use_autodiff=False,
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
            use_autodiff=use_autodiff,
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
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )
    hr_device = device.copy(resolution=310)
    print(device)
    opt = BendingOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    with torch.no_grad():
        # only to build all the required thing needed calculating
        print("begin to init forward", flush=True)
        results_init = opt.forward(sharpness=1 + 2 * 0)
        print("finish init forward", flush=True)

    eps = opt._eps_map
    eps_vec = eps.flatten()
    # eps = eps.to(operation_device)
    keys = list(opt.objective.sims.keys())
    assert len(keys) == 1
    entries_a, indices_a = opt.objective.sims[keys[0]].A
    src_keys = list(opt.norm_run_profiles.keys())
    Jz = opt.norm_run_profiles["in_port_1"][(1.55, 1)][0]
    Jz_torch = torch.from_numpy(Jz).to(operation_device).to(torch.complex64)
    b_vec = Jz.flatten() * 1j * (2 * np.pi * C_0 / (1.55 * MICRON_UNIT))
    A = make_sparse(entries_a, indices_a, (eps_vec.shape[0], eps_vec.shape[0]))
    print("begin to solve the linear system", flush=True)
    start_time = time.time()
    Ez_vec = solve_iterative(A, b_vec, rtol=1e-2)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds", flush=True)
    x = neural_solver(
        eps.unsqueeze(0),
        Jz_torch.unsqueeze(0),
    )
    x = x.detach().cpu().numpy()
    start_time = time.time()
    Ez_vec = solve_iterative(A, b_vec, rtol=1e-2, x0=x)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds", flush=True)
    quit()


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
        assert (
            not configs.run.include_adjoint_NN
        ), "when only adj field is trained, we should not include another adjoint NN"

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

    test_speed(
        int(configs.run.random_state),
        device,
        neural_solver=model_fwd,
        # neural_solver=None,
        numerical_solver="solve_direct",
        use_autodiff=False,
    )


if __name__ == "__main__":
    main()
