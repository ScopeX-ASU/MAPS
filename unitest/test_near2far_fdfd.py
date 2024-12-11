"""
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-12-08 16:31:47
FilePath: /MAPS/unitest/test_near2far_fdfd.py
"""
import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "/home/pingchua/projects/MAPS")
)
sys.path.insert(0, project_root)
import torch
from ceviche.constants import *
from matplotlib import pyplot as plt

from core.fdfd.near2far import get_farfields_Rayleigh_Sommerfeld
from core.invdes.models import (
    MetaLensOptimization,
)
from core.invdes.models.base_optimization import (
    DefaultSimulationConfig,
)
from core.invdes.models.layers import MetaLens
from core.utils import set_torch_deterministic
import numpy as np
import copy


def test_near2far():
    set_torch_deterministic(seed=59)
    sim_cfg = DefaultSimulationConfig()
    wl = 0.5
    sim_cfg.update(
        dict(
            # solver="ceviche",
            solver="ceviche_torch",
            numerical_solver="solve_direct",
            use_autodiff=False,
            neural_solver=None,
            border_width=[0, 0, 1.5, 1.5],
            PML=[0.8, 0.8],
            resolution=50,
            wl_cen=wl,
            plot_root="./figs/metalens_near2far",
        )
    )

    device = MetaLens(
        sim_cfg=sim_cfg,
        device="cuda:0",
        port_len=(1.5, 10),
        substrate_depth=0.75,
        ridge_height_max=0.75,
        nearfield_dx=0.2,
        farfield_dxs=(8,),
        farfield_sizes=(4,),
    )
    hr_device = device.copy(resolution=50)
    print(device)
    opt = MetaLensOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
    )
    print(opt)

    results = opt.forward(sharpness=80)
    opt.plot(
        eps_map=opt._eps_map,
        obj=results["breakdown"]["fwd_trans"]["value"],
        plot_filename="metalens_opt_step_{}_fwd.png".format(1),
        field_key=("in_port_1", wl, 1, 300),
        field_component="Ez",
        in_port_name="in_port_1",
    )
    near_field_points = device.port_monitor_slices_info["nearfield"]
    far_field_points = device.port_monitor_slices_info["farfield_1"]

    far_field_points_p1 = copy.deepcopy(far_field_points)
    far_field_points_p1["xs"] = far_field_points_p1["xs"] + device.grid_step
    print(near_field_points)
    print(far_field_points)

    Ez = opt.objective.solutions[("in_port_1", wl, 1, 300)]["Ez"]
    Hx = opt.objective.solutions[("in_port_1", wl, 1, 300)]["Hx"]
    Hy = opt.objective.solutions[("in_port_1", wl, 1, 300)]["Hy"]

    Ez_farfield = Ez[device.port_monitor_slices["farfield_1"]]
    Hx_farfield = Hx[device.port_monitor_slices["farfield_1"]][0:-1]
    Hy_farfield = Hy[device.port_monitor_slices["farfield_1"]]
    # print(Ez_farfield.abs())

    Ez_farfield_2 = get_farfields_Rayleigh_Sommerfeld(
        nearfield_slice=device.port_monitor_slices["nearfield"],
        nearfield_slice_info=device.port_monitor_slices_info["nearfield"],
        fields=Ez[None, ..., None],
        farfield_x=None,
        farfield_slice_info=far_field_points,
        freqs=torch.tensor([1 / wl], device=Ez.device),
        eps=1,
        mu=MU_0,
        dL=device.grid_step,
        component="Ez",
    )["Ez"][0, :, 0]
    Ez_farfield_2_p1 = get_farfields_Rayleigh_Sommerfeld(
        nearfield_slice=device.port_monitor_slices["nearfield"],
        nearfield_slice_info=device.port_monitor_slices_info["nearfield"],
        fields=Ez[None, ..., None],
        farfield_x=None,
        farfield_slice_info=far_field_points_p1,
        freqs=torch.tensor([1 / wl], device=Ez.device),
        eps=1,
        mu=MU_0,
        dL=device.grid_step,
        component="Ez",
    )["Ez"][0, :, 0]
    # print(Ez_farfield_2.abs())
    omega = 2 * np.pi * C_0 / (wl * 1e-6)
    Hx_farfield_2 = (Ez_farfield_2[0:-1] - Ez_farfield_2[1:]) / (device.grid_step * 1e-6) * (-1 / 1j / omega / MU_0)
    Hy_farfield_2 = (Ez_farfield_2_p1 - Ez_farfield_2) / (device.grid_step * 1e-6) * (-1 / 1j / omega / MU_0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(
        far_field_points["ys"],
        Ez_farfield.abs().detach().cpu().numpy(),
        label="fdfd",
        color="r",
    )
    ax.plot(
        far_field_points["ys"],
        Ez_farfield_2.abs().detach().cpu().numpy(),
        label="near2far",
        color="b",
    )
    ax.legend()
    plt.savefig("./figs/metalens_near2far/farfield_E.png")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    print("this is farfield_points", type(far_field_points["ys"]))
    ax.plot(
        far_field_points["ys"][0:-1],
        Hx_farfield.abs().detach().cpu().numpy(),
        label="fdfd",
        color="r",
    )
    ax.plot(
        far_field_points["ys"][0:-1],
        Hx_farfield_2.abs().detach().cpu().numpy(),
        label="near2far",
        color="b",
    )
    ax.legend()
    plt.savefig("./figs/metalens_near2far/farfield_Hx.png")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(
        far_field_points["ys"],
        Hy_farfield.abs().detach().cpu().numpy(),
        label="fdfd",
        color="r",
    )
    ax.plot(
        far_field_points["ys"],
        Hy_farfield_2.abs().detach().cpu().numpy(),
        label="near2far",
        color="b",
    )
    ax.legend()
    plt.savefig("./figs/metalens_near2far/farfield_Hy.png")
    plt.close()

    # Compute derivative matrices


if __name__ == "__main__":
    test_near2far()
