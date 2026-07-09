"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2026-05-08 20:18:18
LastEditors: Jiaqi Gu (jiaqigu@asu.edu)
LastEditTime: 2026-05-08 21:50:18
FilePath: /MAPS_fdtdx/core/invdes/examples/grating_coupler_3d.py
"""

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
import torch
from pyutils.config import Config

from core.invdes.invdesign import InvDesign
from core.invdes.models import GratingCouplerOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import GratingCoupler
from core.utils import set_torch_deterministic

sys.path.pop(0)
if __name__ == "__main__":
    """Reference: https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd16BilayerCoupler"""
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    # edge_coupler_region_size = (1.6, 1.6)
    edge_coupler_region_size = (8, 8)
    # edge_coupler_region_size = (4, 4)
    # port_len = 1.5
    port_len = 1.9
    top_port_len = 2.5
    port_height = 0.22

    input_port_width = 0.5
    # input_port_width = 1.0
    wl_cen = 1.310
    # exp_name = f"grating_coupler_{wl_cen}_{edge_coupler_region_size[0]}x{edge_coupler_region_size[1]}_uniform25_sym"
    # exp_name = f"grating_coupler_{wl_cen}_{edge_coupler_region_size[0]}x{edge_coupler_region_size[1]}_autogrid10_sym"
    exp_name = f"grating_coupler_{wl_cen}_{edge_coupler_region_size[0]}x{edge_coupler_region_size[1]}_autogrid10_sym_new"
    wl_width = 0.0
    n_wl = 1
    in_mode = "Hz2"  # z-direction Y-polarized gaussian beam
    out_mode = "Hz1"  # TE00 mode in the waveguide
    resolution = 50

    sim_cfg.update(
        dict(
            solver="fdtdx",
            subpixel=True,
            # border_width=[0, port_len, 0.6, 0.6, 0.7, 0.7],
            # border_width=[0, port_len, 0.6, 0.6, 1.3, 1.3],
            # border_width=[port_len, 0, 1.0, 1.0, 0.9, 0.0],
            border_width=[port_len, 0, 1.0, 1.0, 0.0, 0.0],
            resolution=resolution,
            max_time=4e-13,
            symmetry=(0, -1, 0),  # PEC on y direction.
            optical_grid={
                "mode": "fdtdx_rectilinear_nonuniform",
                "wavelength": wl_cen,
                "grid_x": {"type": "auto", "min_steps_per_wvl": 10},
                "grid_y": {"type": "auto", "min_steps_per_wvl": 10},
                "grid_z": {"type": "auto", "min_steps_per_wvl": 15},
            },
            plot_root=f"./figs/{exp_name}",
            # PML=[0.4, 0.4, 0.4],
            PML=[0.8, 0.8, 0.8],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
        )
    )

    device = GratingCoupler(
        material_r=3.5**2,
        # material_r=1.444**2,
        material_layers=[3.5**2, 2**2],
        # material_layers=[1.444**2, 1.444**2],
        # material_bg="SiO2",
        material_bg=1.44**2,
        thickness_layers=[0.22, 0.6],
        etch_depth_layers=[0.16, 0.6],
        gap_layers=(0.0, 0.1),
        sim_cfg=sim_cfg,
        design_region_size=edge_coupler_region_size,
        port_len=port_len,
        port_width=input_port_width,
        port_height=port_height,
        top_port_len=top_port_len,
        substrate_thickness=1.0,
        substrate_gap=2.0,
        # gaussian_waist=5.6568,
        waist_radius=1,  # SMF-28 1310nm
        waist_distance=-5.0,  # SMF-28 1310nm
        is_3d=True,
        in_mode=in_mode,
        out_mode=out_mode,
        device=operation_device,
    )

    # import matplotlib.pyplot as plt
    # print(device.epsilon_map.shape)
    # fig, ax = plt.subplots(1, 5, figsize=(20, 3))
    # for i in range(5):
    #     ax[i].imshow(device.epsilon_map.real[..., device.epsilon_map.shape[-1]//2-i].T, origin="lower", cmap="Greys")
    # plt.savefig(f"{exp_name}_init.png", dpi=300)
    # exit(0)

    hr_device = device.copy(resolution=resolution)
    print(device)

    wls = torch.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl).tolist()
    obj_cfgs = dict(override=True)

    obj_cfgs[f"fwd_trans"] = dict(
        weight=0.1,
        #### objective is evaluated at this port
        in_slice_name="in_slice_1",
        out_slice_name="out_slice_1",
        #### objective is evaluated at all points by sweeping the wavelength and modes
        in_mode=in_mode,  # only one source mode is supported, cannot input multiple modes at the same time
        wl=wls,
        temp=[300],
        out_modes=(
            out_mode,
        ),  # can evaluate on multiple output modes and get average transmission
        type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
        direction="x+",
    )
    obj_cfgs[f"output_flux"] = dict(
        weight=1,
        #### objective is evaluated at this port
        in_slice_name="in_slice_1",
        out_slice_name="out_slice_1",
        #### objective is evaluated at all points by sweeping the wavelength and modes
        in_mode=in_mode,  # only one source mode is supported, cannot input multiple modes at the same time
        wl=wls,
        temp=[300],
        out_modes=(
            out_mode,
        ),  # can evaluate on multiple output modes and get average transmission
        type="flux",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
        direction="x",
        requires_grad=False,
    )
    obj_cfgs[f"input_flux"] = dict(
        weight=1,
        #### objective is evaluated at this port
        in_slice_name="in_slice_1",
        out_slice_name="in_monitor_slice_1",
        #### objective is evaluated at all points by sweeping the wavelength and modes
        in_mode=in_mode,  # only one source mode is supported, cannot input multiple modes at the same time
        wl=wls,
        temp=[300],
        out_modes=(
            out_mode,
        ),  # can evaluate on multiple output modes and get average transmission
        type="flux",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
        direction="z",
        requires_grad=False,
    )
    obj_cfgs[f"refl_flux"] = dict(
        weight=1,
        #### objective is evaluated at this port
        in_slice_name="in_slice_1",
        out_slice_name="refl_slice_1",
        #### objective is evaluated at all points by sweeping the wavelength and modes
        in_mode=in_mode,  # only one source mode is supported, cannot input multiple modes at the same time
        wl=wls,
        temp=[300],
        out_modes=(
            out_mode,
        ),  # can evaluate on multiple output modes and get average transmission
        type="flux",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
        direction="z",
        requires_grad=False,
    )

    def fom_func(breakdown):
        ## input_flux = |Pin - Pref|; refl_flux = |Pref|; Pref > 0; |Pin| = input_flux + refl_flux
        trans = (breakdown["fwd_trans"]["value"]) / (
            breakdown["refl_flux"]["value"].detach()
            + breakdown["input_flux"]["value"].detach()
        )

        trans_flux = (breakdown["output_flux"]["value"]).detach() / (
            breakdown["refl_flux"]["value"].detach()
            + breakdown["input_flux"]["value"].detach()
        )
        spectrum = {}
        fom = 0
        for wl, s, s_flux in zip(wls, trans, trans_flux):
            spectrum[f"|s21|^2 at {wl:.2f}"] = {"weight": 1, "value": s_flux.item()}
            fom = fom - breakdown["fwd_trans"]["weight"] * (s - 1) ** 2
        spectrum["avg|s21|^2"] = {"weight": 1, "value": trans_flux.mean().item()}
        return fom, spectrum

    obj_cfgs["_fusion_func"] = fom_func

    design_region_param_cfgs = dict()
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[50, 50],
            sigma=2 / 50,
            transform=[
                dict(type="mirror_symmetry", dims=[1]),
                # dict(
                #     type="blur",
                #     mfs=0.05,
                #     resolutions=[hr_device.resolution, hr_device.resolution],
                #     dim="xy",
                # ),
                dict(type="binarize"),
            ],
            init_method="random_0.5",
            # init_method="constant_0.3",
            # init_method="constant_0.0",
            # init_method="random",
            denorm_mode="linear_eps",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
            dims=(0, 1),
            extrude_direction="-",
            extrude_angle=90.0,
        )

    opt = GratingCouplerOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        design_region_param_cfgs=design_region_param_cfgs,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(
        devOptimization=opt,
        run=Config(
            n_epochs=100,
        ),
        optimizer=Config(
            name="Adam",
            # lr=0.25,
            lr=0.1,
            # init_v=1e-7,
            weight_decay=0,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=0.05,
        ),
        sharp_scheduler=Config(
            mode="cosine",
            name="sharpness",
            init_sharp=5,
            final_sharp=5,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=1,
            plot_name=f"{exp_name}",
            objs=["avg|s21|^2"],
            field_keys=[("in_slice_1", wl_cen, in_mode, 300)],
            in_slice_names=["in_slice_1"],
            exclude_port_names=[],
            field_component="Ey",
            eps_grad=True,
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
        ),
    )
    invdesign.optimize()
