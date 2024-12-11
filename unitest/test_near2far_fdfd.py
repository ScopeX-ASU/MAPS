"""
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-12-08 16:31:47
FilePath: /MAPS/unitest/test_near2far_fdfd.py
"""

import torch
from ceviche.constants import *
from matplotlib import pyplot as plt

from core.fdfd.near2far import (
    get_farfields_GreenFunction,
    get_farfields_Rayleigh_Sommerfeld,
)
from core.invdes.models import (
    MetaLensOptimization,
)
from core.invdes.models.base_optimization import (
    DefaultSimulationConfig,
)
from core.invdes.models.layers import MetaLens
from core.utils import set_torch_deterministic


def test_near2far():
    set_torch_deterministic(seed=56)
    sim_cfg = DefaultSimulationConfig()
    wl = 0.832
    sim_cfg.update(
        dict(
            # solver="ceviche",
            solver="ceviche_torch",
            numerical_solver="solve_direct",
            use_autodiff=False,
            neural_solver=None,
            border_width=[0, 0, 3, 3],
            PML=[0.8, 0.8],
            resolution=50,
            wl_cen=wl,
            plot_root="./figs/metalens_near2far",
        )
    )

    device = MetaLens(
        material_bg="SiO2",
        sim_cfg=sim_cfg,
        device="cuda:0",
        aperture=3,
        port_len=(1.5, 11),
        substrate_depth=0.75,
        ridge_height_max=0.75,
        nearfield_dx=0.9,
        farfield_dxs=(10,),
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
        field_key=("in_port_1", wl, 1),
        field_component="Ez",
        in_port_name="in_port_1",
        exclude_port_names=["farfield_region"],
    )
    # near_field_points = device.port_monitor_slices_info["nearfield"]
    far_field_points = device.port_monitor_slices_info["farfield_1"]
    # print(near_field_points)
    print(far_field_points)

    Ez = opt.objective.solutions[("in_port_1", wl, 1)]["Ez"]
    Hx = opt.objective.solutions[("in_port_1", wl, 1)]["Hx"]
    Hy = opt.objective.solutions[("in_port_1", wl, 1)]["Hy"]

    Ez_farfield = Ez[device.port_monitor_slices["farfield_1"]]
    print(Ez_farfield.abs())

    # Ez_farfield_2 = get_farfields_Rayleigh_Sommerfeld(
    #     nearfield_slice=device.port_monitor_slices["nearfield"],
    #     nearfield_slice_info=device.port_monitor_slices_info["nearfield"],
    #     fields=Ez[None, ..., None],
    #     farfield_x=None,
    #     farfield_slice_info=far_field_points,
    #     freqs=torch.tensor([1 / wl], device=Ez.device),
    #     eps=1,
    #     mu=MU_0,
    #     dL=device.grid_step,
    #     component="Ez",
    # )["Ez"][0, :, 0]

    with torch.inference_mode():
        Ez_farfield_2 = get_farfields_GreenFunction(
            nearfield_slices=[
                device.port_monitor_slices[f"nearfield_{i}"] for i in range(1, 4)
            ],
            nearfield_slices_info=[
                device.port_monitor_slices_info[f"nearfield_{i}"] for i in range(1, 4)
            ],
            Ez=Ez[None, ..., None],
            Hx=Hx[None, ..., None],
            Hy=Hy[None, ..., None],
            farfield_x=None,
            farfield_slice_info=far_field_points,
            freqs=torch.tensor([1 / wl], device=Ez.device),
            eps=device.eps_bg,
            mu=MU_0,
            dL=device.grid_step,
            component="Ez",
            sign="+",
        )["Ez"][0, ..., 0]

    print(Ez_farfield_2.abs())

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
    plt.savefig("./figs/metalens_near2far/farfield.png")
    plt.close()
    # exit(0)

    # Compute derivative matrices
    # Ez_farfield_region = get_farfields_Rayleigh_Sommerfeld(
    #     nearfield_slice=device.port_monitor_slices["nearfield"],
    #     nearfield_slice_info=device.port_monitor_slices_info["nearfield"],
    #     fields=Ez[None, ..., None],
    #     farfield_x=None,
    #     farfield_slice_info=device.port_monitor_slices_info["farfield_region"],
    #     freqs=torch.tensor([1 / wl], device=Ez.device),
    #     eps=1.44**2,
    #     mu=MU_0,
    #     dL=device.grid_step,
    #     component="Ez",
    # )["Ez"][0, ..., 0]

    with torch.inference_mode():
        Ez_farfield_region = get_farfields_GreenFunction(
            nearfield_slices=[
                device.port_monitor_slices[f"nearfield_{i}"] for i in range(1, 4)
            ],
            nearfield_slices_info=[
                device.port_monitor_slices_info[f"nearfield_{i}"] for i in range(1, 4)
            ],
            Ez=Ez[None, ..., None],
            Hx=Hx[None, ..., None],
            Hy=Hy[None, ..., None],
            farfield_x=None,
            farfield_slice_info=device.port_monitor_slices_info["farfield_region"],
            freqs=torch.tensor([1 / wl], device=Ez.device),
            eps=device.eps_bg,
            mu=MU_0,
            dL=device.grid_step,
            component="Ez",
            sign="+",
        )["Ez"][0, ..., 0]

    farfield_region = device.port_monitor_slices["farfield_region"]
    print(farfield_region)
    print(device.port_monitor_slices_info["farfield_region"])
    fig, ax = plt.subplots(2, 1, figsize=(5, 5))
    print(Ez.shape, Ez_farfield_region.shape)
    Ez_farfield_region = torch.cat(
        [Ez[: farfield_region.x[0, 0]], Ez_farfield_region], dim=0
    )
    min_val = torch.min(Ez.real).item()
    max_val = torch.max(Ez.real).item()
    ax[0].imshow(
        Ez_farfield_region.abs().detach().cpu().numpy().T,
        # Ez_farfield_region.real.detach().cpu().numpy().T,
        cmap="magma",
        # cmap="RdBu",
        vmin=min_val,
        vmax=max_val,
    )
    ax[1].imshow(
        Ez.abs().detach().cpu().numpy().T,
        # Ez.real.detach().cpu().numpy().T,
        cmap="magma",
        # cmap="RdBu",
        vmin=min_val,
        vmax=max_val,
    )
    plt.tight_layout()
    plt.savefig("./figs/metalens_near2far/farfield_ext.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    test_near2far()
