"""
Date: 2024-10-03 02:27:36
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-06 01:41:28
FilePath: /Metasurface-Opt/unitest/test_device_base.py
"""

"""
Date: 2024-10-03 02:27:36
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-04 00:50:55
FilePath: /Metasurface-Opt/unitest/test_device_base.py
"""
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch
from ceviche import fdfd_ez as ceviche_fdfd_ez
from ceviche.constants import *
from pyutils.general import print_stat
from pyutils.general import TimerCtx
from core.invdes.models import (
    IsolatorOptimization,
    MetaCouplerOptimization,
    MetaMirrorOptimization,
)
from core.invdes.models.base_optimization import (
    BaseOptimization,
    DefaultSimulationConfig,
)
from core.fdfd.fdfd import fdfd_ez
from core.fdfd.utils import torch_sparse_to_scipy_sparse
from core.invdes.models.layers import Isolator, MetaCoupler, MetaMirror
from core.invdes.models.layers.device_base import N_Ports, Si_eps
from core.invdes.models.layers.utils import plot_eps_field
from core.utils import set_torch_deterministic
from torch_sparse import spspmm


def test_device_base():
    device = N_Ports(
        port_cfgs=dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-1.5, 0],
                size=[3, 0.48],
                eps=Si_eps(1.55),
            ),
            out_port_1=dict(
                type="box",
                direction="x",
                center=[1.5, 0],
                size=[3, 0.6],
                eps=Si_eps(1.55),
            ),
        ),
    )
    print(device)
    src_slice = device.build_port_monitor_slice(
        port_name="in_port_1", rel_loc=0.4, rel_width=2
    )
    out_slice = device.build_port_monitor_slice(
        port_name="out_port_1", rel_loc=0.6, rel_width=2
    )
    radiation_monitor = device.build_radiation_monitor()
    plot_eps_field(
        radiation_monitor,
        device.epsilon_map,
        monitors=[(src_slice, "r"), (out_slice, "b"), (radiation_monitor, "g")],
        filepath="./figs/device_base_eps_field.png",
        x_width=device.cell_size[0],
        y_height=device.cell_size[1],
        NPML=device.NPML,
    )
    input_SCALE, norm_sources = device.build_norm_sources(
        source_modes=(1,),
        input_port_name="in_port_1",
        wl_cen=1.55,
        wl_width=0,
        n_wl=1,
        plot=True,
    )


def test_metamirror():
    device = MetaMirror(device="cuda:0")
    print(device)
    # src_slice = device.build_port_monitor_slice(
    #     port_name="in_port_1", rel_loc=0.4, rel_width=2
    # )
    # out_slice = device.build_port_monitor_slice(
    #     port_name="out_port_1", rel_loc=0.6, rel_width=2
    # )
    # radiation_monitor = device.build_radiation_monitor()
    # plot_eps_field(
    #     radiation_monitor,
    #     device.epsilon_map,
    #     monitors=[(src_slice, "r"), (out_slice, "b"), (radiation_monitor, "g")],
    #     filepath="./figs/metamirror_eps_field.png",
    #     x_width=device.cell_size[0],
    #     y_height=device.cell_size[1],
    #     NPML=device.NPML,
    # )
    device.init_monitors()
    device.norm_run()
    fields = device.solve(
        device.epsilon_map,
        source_profiles=device.port_sources_dict["in_port_1"],
        grid_step=device.grid_step,
    )
    plot_eps_field(
        list(fields.values())[0]["Ez"],
        device.epsilon_map,
        filepath="./figs/metamirror_solve_field.png",
        x_width=device.cell_size[0],
        y_height=device.cell_size[1],
        NPML=device.NPML,
    )


def test_metamirror_opt():
    set_torch_deterministic(seed=59)
    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            border_width=[0, 1.5, 0, 1.5],
            resolution=100,
            wl_cen=1.55,
            wl_width=0,
            n_wl=1,
            # plot_root="./figs/metamirror",
            plot_root="./figs/metamirror_tanh_subpixel",
        )
    )

    device = MetaMirror(sim_cfg=sim_cfg, device="cuda:0")
    print(device)
    hr_device = device.copy(resolution=310)
    opt = MetaMirrorOptimization(device=device, hr_device=hr_device, sim_cfg=sim_cfg)

    optimizer = torch.optim.Adam(opt.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=70, eta_min=0.0002
    )

    for step in range(70):
        optimizer.zero_grad()
        results = opt.forward(sharpness=1 + 2 * step)
        opt.plot(
            eps_map=opt._eps_map,
            obj=opt._obj,
            plot_filename="metamirrir_opt_step_{}.png".format(step),
            field_key=("in_port_1", 1.55, 1),
            field_component="Ez",
            in_port_name="in_port_1",
        )

        print(f"Step {step}:", end=" ")
        for k, obj in results["breakdown"].items():
            print(f"{k}: {obj['value']:.3f}", end=", ")
        print()
        (-results["obj"]).backward()
        # print_stat(list(opt.parameters())[0], f"step {step}: grad: ")
        optimizer.step()
        scheduler.step()


def test_metacoupler_opt():
    set_torch_deterministic(seed=59)
    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            # solver="ceviche",
            solver="ceviche_torch",
            numerical_solver="solve_direct",
            use_autodiff=False,
            neural_solver=None,
            border_width=[0, 0, 6, 6],
            resolution=20,
            plot_root="./figs/metacoupler_subpixel",
            # plot_root="./figs/metacoupler_periodic",
        )
    )

    device = MetaCoupler(sim_cfg=sim_cfg, device="cuda:0")
    hr_device = device.copy(resolution=100)
    print(device)
    opt = MetaCouplerOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
    )
    print(opt)

    optimizer = torch.optim.Adam(opt.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=70, eta_min=0.0002
    )

    for step in range(100):
        with TimerCtx() as t:
            optimizer.zero_grad()
            results = opt.forward(sharpness=1 + 2 * step)
            opt.plot(
                eps_map=opt._eps_map,
                obj=results["breakdown"]["fwd_trans"]["value"],
                plot_filename="metacoupler_opt_step_{}_fwd.png".format(step),
                field_key=("in_port_1", 1.55, 1),
                field_component="Ez",
                in_port_name="in_port_1",
                exclude_port_names=["refl_port_2"],
            )
            opt.plot(
                eps_map=opt._eps_map,
                obj=results["breakdown"]["bwd_trans"]["value"],
                plot_filename="metacoupler_opt_step_{}_bwd.png".format(step),
                field_key=("out_port_1", 1.55, 1),
                field_component="Ez",
                in_port_name="out_port_1",
                exclude_port_names=["refl_port_1"],
            )
            print(f"Step {step}:", end=" ")
            for k, obj in results["breakdown"].items():
                print(f"{k}: {obj['value']:.3f}", end=", ")
            print()
            (-results["obj"]).backward()
            # for p in opt.parameters():
            #     print(p.grad)
            # if step % 5 == 0:
            #     opt.dump_data(f"./data/fdfd/metacoupler/test2_metacoupler_opt_step_{step}.h5")
            # print_stat(list(opt.parameters())[0], f"step {step}: grad: ")
            optimizer.step()
            scheduler.step()
        print(f"Step {step} took {t.interval:.3f} s")


def test_isolator_opt():
    name = "isolator"
    set_torch_deterministic(seed=59)
    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            # solver="ceviche",
            solver="ceviche_torch",
            border_width=[0, 0, 2, 2],
            resolution=50,
            plot_root=f"./figs/{name}",
        )
    )

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        for _, obj in breakdown.items():
            fom = fom + obj["weight"] * obj["value"]

        ## add extra contrast ratio
        contrast = breakdown["bwd_trans"]["value"] / breakdown["fwd_trans"]["value"]
        fom = fom - contrast
        return fom, {"contrast": {"weight": -1, "value": contrast}}

    obj_cfgs = dict(_fusion_func=fom_func)

    device = Isolator(sim_cfg=sim_cfg, device="cuda:0")
    hr_device = device.copy(resolution=310)
    print(device, flush=True)
    print(hr_device, flush=True)
    opt = IsolatorOptimization(
        device=device, hr_device=hr_device, sim_cfg=sim_cfg, obj_cfgs=obj_cfgs
    )
    print(opt)

    optimizer = torch.optim.Adam(opt.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=70, eta_min=0.0002
    )

    for step in range(70):
        optimizer.zero_grad()
        results = opt.forward(sharpness=1 + 2 * step)
        opt.plot(
            eps_map=opt._eps_map,
            obj=results["breakdown"]["fwd_trans"]["value"],
            plot_filename=f"{name}_opt_step_{step}_fwd.png",
            field_key=("in_port_1", 1.55, 1),
            field_component="Ez",
            in_port_name="in_port_1",
            exclude_port_names=["refl_port_2"],
        )
        opt.plot(
            eps_map=opt._eps_map,
            obj=results["breakdown"]["bwd_trans"]["value"],
            plot_filename=f"{name}_opt_step_{step}_bwd.png",
            field_key=("out_port_1", 1.55, 1),
            field_component="Ez",
            in_port_name="out_port_1",
            exclude_port_names=["refl_port_1"],
        )
        print(f"Step {step}:", end=" ")
        for k, obj in results["breakdown"].items():
            print(f"{k}: {obj['value']:.3f}", end=", ")
        print()
        (-results["obj"]).backward()

        # print_stat(list(opt.parameters())[0].grad, f"solver={sim_cfg['solver']}, step {step}: grad: ")
        # print(list(opt.parameters())[0].grad)
        optimizer.step()
        scheduler.step()


def test_fdtd_ez_torch():
    dev = "cpu"
    name = "isolator"
    set_torch_deterministic(seed=59)
    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="ceviche",
            border_width=[0, 0, 2, 2],
            resolution=20,
            plot_root=f"./figs/{name}",
        )
    )

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        for _, obj in breakdown.items():
            fom = fom + obj["weight"] * obj["value"]

        ## add extra contrast ratio
        contrast = breakdown["bwd_trans"]["value"] / breakdown["fwd_trans"]["value"]
        fom = fom - contrast
        return fom, {"contrast": {"weight": -1, "value": contrast}}

    obj_cfgs = dict(_fusion_func=fom_func)

    device = Isolator(sim_cfg=sim_cfg, device=dev)
    hr_device = device.copy(resolution=20)
    print(device, flush=True)
    print(hr_device, flush=True)
    opt = IsolatorOptimization(
        device=device, hr_device=hr_device, sim_cfg=sim_cfg, obj_cfgs=obj_cfgs
    )
    print(opt)

    eps_r = torch.from_numpy(device.epsilon_map).to(dev)
    source = torch.from_numpy(device.port_sources_dict["in_port_1"][(1.55, 1)][0]).to(
        dev
    )

    omega = 2 * np.pi / (1.55e-6)
    dl = device.grid_step * 1e-6
    # sim = fdfd_ez(omega, dl, eps_r[:2,:2], device=dev, npml=[1,1])
    # sim = fdfd_ez(omega, dl, eps_r[:2,:2], npml=[1,1])
    # hx, hy, ez = sim.solve(source[:2,:2])

    # c_sim = ceviche_fdfd_ez(omega, dl, device.epsilon_map[:2,:2], npml=[1,1])
    # c_hx, c_hy, c_ez = c_sim.solve(source.cpu().numpy()[:2,:2])

    sim = fdfd_ez(omega, dl, eps_r, npml=[1, 1])
    hx, hy, ez = sim.solve(source)

    c_sim = ceviche_fdfd_ez(omega, dl, device.epsilon_map, npml=[1, 1])
    c_hx, c_hy, c_ez = c_sim.solve(source.cpu().numpy())
    print(ez)
    print(c_ez)
    quit()
    # print((torch_sparse_to_scipy_sparse(sim.Dxf) - c_sim.Dxf.tocoo()))
    # print((torch_sparse_to_scipy_sparse(sim.Dxb) - c_sim.Dxb.tocoo()))
    # print((torch_sparse_to_scipy_sparse(sim.Dyf) - c_sim.Dyf.tocoo()))
    # print((torch_sparse_to_scipy_sparse(sim.Dyb) - c_sim.Dyb.tocoo()))

    # sparse_tensor_A = torch.sparse_coo_tensor(
    #     indices=torch.tensor(sim._make_A(sim.eps_r.flatten())[1]).to(dev),
    #     values=torch.tensor(sim._make_A(sim.eps_r.flatten())[0]).to(dev),
    #     size=(4,),
    #     device=dev,
    # )
    # A = torch_sparse_to_scipy_sparse(sparse_tensor_A) # this sim._make_A returns a tuple (entries_a, indices_a)
    # c_A_v, c_A_i = c_sim._make_A(c_sim.eps_r.flatten())

    print(torch_sparse_to_scipy_sparse(sim.Dxf))
    print("----------------")
    print(c_sim.Dxf.tocoo())
    print()
    print((torch_sparse_to_scipy_sparse(sim.Dxb) - c_sim.Dxb.tocoo()))
    print()
    print((torch_sparse_to_scipy_sparse(sim.Dyf) - c_sim.Dyf.tocoo()))
    print()
    print((torch_sparse_to_scipy_sparse(sim.Dyb) - c_sim.Dyb.tocoo()))
    print()

    # c_A = sp.coo_matrix((c_A_v, c_A_i), shape=A.shape)
    # print("A error:", A - c_A)

    C = (
        -1
        / MU_0
        * (
            sim.Dxf.coalesce() @ sim.Dxb.coalesce()
            # torch_sparse_to_scipy_sparse(sim.Dxf).dot(torch_sparse_to_scipy_sparse(sim.Dxb))
            #    + torch.sparse.mm(sim.Dyf, sim.Dyb)
        )
    )

    c_C = (
        -1
        / MU_0
        * (
            c_sim.Dxf @ c_sim.Dxb
            # + c_sim.Dyf.dot(c_sim.Dyb)
        )
    ).tocoo()
    print("C:", C)
    print("c_C:", c_C)
    print("C error", C - c_C)


if __name__ == "__main__":
    # test_device_base()
    # test_metamirror()
    # test_metamirror_opt()
    test_metacoupler_opt()
    # test_isolator_opt()
    # test_fdtd_ez_torch()
