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
from autograd.numpy import array as npa
from pyutils.config import Config

from core.invdes.invdesign import InvDesign
from core.invdes.models import MMIOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MMI
from core.utils import set_torch_deterministic

sys.path.pop(0)
if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    mmi_region_size = (2, 2)
    port_len = 1.5
    wl_cen = 1.55
    branch_sep = 0.2

    input_port_width = 0.5
    output_port_width = 0.5
    num_inports = 1
    num_outports = 2
    resolution = 100
    pol = "Hz"

    material_r1 = "Si_eff"
    exp_name = (
        f"y_branch_{material_r1}_{mmi_region_size[0]}x{mmi_region_size[1]}"
    )

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            symmetry=(0, -1, 0),  # PEC on y direction.
            optical_grid={
                "mode": "fdtdx_rectilinear_nonuniform",
                "wavelength": wl_cen,
                "grid_x": {"type": "auto", "min_steps_per_wvl": 25},
                "grid_y": {"type": "auto", "min_steps_per_wvl": 25},
            },
            border_width=[0, 0, 1, 1],  # left, right, lower, upper, containing PML
            resolution=resolution,
            plot_root=f"./figs/{exp_name}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        for key, obj in breakdown.items():
            if "fwd" in key:
                fom -= obj["weight"] * (obj["value"] - 1) ** 2
            elif "refl" in key:
                fom += obj["weight"] * obj["value"]
            elif "energy_constraint" in key:
                fom += obj["weight"] * obj["value"]

        return fom, {}

    obj_cfgs = dict(
        fwd_trans_p1=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=[wl_cen],  #
            temp=[300],
            out_modes=(
                f"{pol}1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction="x+",
        ),
        fwd_trans_p2=dict(
            weight=1,
            in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_2",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=[wl_cen],  #
            temp=[300],
            out_modes=(
                f"{pol}1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction="x+",
        ),
        refl_trans=dict(
            weight=-1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="refl_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=[wl_cen],  #
            temp=[300],
            out_modes=(
                f"{pol}1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="flux_minus_src",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction="x",
        ),
        energy_constraint=dict(
            weight=-0.1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=[wl_cen],
            temp=[300],
            in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                f"{pol}1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="energy_constraint",
        ),
        _fusion_func=fom_func,
        override=True,
    )

    device = MMI(
        material_r1=material_r1,
        sim_cfg=sim_cfg,
        box_size=mmi_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        num_inports=num_inports,
        num_outports=num_outports,
        device=operation_device,
        port_box_margin=(
            mmi_region_size[1]
            - output_port_width * num_outports
            - branch_sep * (num_outports - 1)
        )
        / 2,
        rel_port_width=2.5,
        output_sine_bend_length=3,
        output_sine_bend_expand=0.25,
        pol=pol,
    )

    hr_device = device.copy(resolution=resolution)
    print(device)

    design_region_param_cfgs = {}
    for region_name in device.design_region_cfgs.keys():
        # design_region_param_cfgs[region_name] = dict(
        #     method="levelset",
        #     rho_resolution=[25, 25],
        #     sigma=2 / resolution,
        #     transform=[
        #         dict(type="mirror_symmetry", dims=[1]),
        #         # dict(
        #         #     type="blur",
        #         #     mfs=0.1,
        #         #     resolutions=[hr_device.resolution, hr_device.resolution],
        #         #     dim="xy",
        #         # ),
        #         dict(type="binarize"),
        #     ],
        #     # init_method="random",
        #     init_method="constant_0.1",
        #     denorm_mode="linear_eps",
        #     interpolation="gaussian_linear",
        #     binary_projection=dict(
        #         fw_threshold=100,
        #         bw_threshold=100,
        #         mode="regular",
        #     ),
        # )

        design_region_param_cfgs[region_name] = dict(
            method="edgebox",
            geometry_cfgs=dict(
                enabled_edges=("y-", "y+"),
                pair_symmetry="mirror",
                box_center=(0.0, 0.0),
                box_center_unit="relative",
                box_size=(1.0, 0.5),
                box_size_unit="relative",  # relative or um
                rho_resolution={"y-": 5, "y+": 5},
                default_rho_resolution=5,
                degree=3,
                border_band_um=0.02,
                constraints={},
                # constraints={"y+": (0, 1.0)},
            ),
            transform=[
                dict(type="mirror_symmetry", dims=[1]),
                dict(
                    type="blur",
                    mfs=0.1,
                    resolutions=[hr_device.resolution, hr_device.resolution],
                    dim="xy",
                ),
                dict(type="binarize"),
            ],
            # init_method="random",
            init_method="zeros",
            denorm_mode="linear_eps",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
        )

    opt = MMIOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
        design_region_param_cfgs=design_region_param_cfgs,
        obj_cfgs=obj_cfgs,
    ).to(operation_device)
    invdesign = InvDesign(
        devOptimization=opt,
        optimizer=Config(
            name="Adam",
            # lr=1e-2,
            lr=0.1,
            # init_v=1e-7,
            # name="lbfgs",
            # line_search_fn="strong_wolfe",
            # lr=1e4,
            # weight_decay=0,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=0.1,
        ),
        sharp_scheduler=Config(
            mode="cosine",
            name="sharpness",
            init_sharp=1,
            final_sharp=32,
        ),
        run=Config(
            n_epochs=100,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=5,
            objs=["fwd_trans_p1"],
            plot_name=f"{exp_name}",
            field_keys=[
                (f"in_slice_{i + 1}", wl_cen, f"{pol}1", 300)
                for i in range(num_inports)
            ],
            in_slice_names=[f"in_slice_{i + 1}" for i in range(num_inports)],
            filename_suffixes=[f"s{i + 1}" for i in range(num_inports)],
            exclude_port_names=[],
            eps_grad=True,
            param_grad=True,
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
            upsample_eps_to_1nm=True,
        ),
        after_step_callbacks=[],  # no callbacks for edgebox, need to use clamp for levelset
    )
    invdesign.optimize()
