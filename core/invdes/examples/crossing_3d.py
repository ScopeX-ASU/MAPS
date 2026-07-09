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
from core.invdes.models import CrossingOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import Crossing
from core.utils import set_torch_deterministic

sys.path.pop(0)

if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))

    sim_cfg = DefaultSimulationConfig()

    thickness = 0.22
    crossing_region_size = (4, 4, thickness)
    port_len = 1.5
    port_width = 0.5
    mode = "Hz1"
    wl_cen = 1.55
    wl_width = 0.06
    n_wl = 7
    resolution = 25
    exp_name = (
        f"crossing_3d_Si_{crossing_region_size[0]}x"
        # f"{crossing_region_size[1]}x{crossing_region_size[2]}_cg_hs-dy"
        # f"{crossing_region_size[1]}x{crossing_region_size[2]}_cg_hz"
        # f"{crossing_region_size[1]}x{crossing_region_size[2]}_cg_ls"
        f"{crossing_region_size[1]}x{crossing_region_size[2]}_adam"
    )

    sim_cfg.update(
        dict(
            solver="fdtdx",
            border_width=[0, 0, 0, 0, 0.7, 0.7],
            resolution=resolution,
            plot_root=f"./figs/{exp_name}",
            PML=[0.2, 0.2, 0.2],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
        )
    )

    device = Crossing(
        material_r1="Si",
        sim_cfg=sim_cfg,
        box_size=crossing_region_size,
        port_len=(port_len, port_len),
        port_width=(port_width, port_width),
        port_height=(thickness, thickness),
        is_3d=True,
        mode=mode,
        device=operation_device,
    )

    hr_device = device.copy(resolution=resolution)
    print(device)

    wls = torch.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl).tolist()

    obj_cfgs = dict(
        fwd_trans=dict(
            weight=1,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            in_mode=mode,
            wl=wls,
            temp=[300],
            out_modes=(mode,),
            type="eigenmode",
            direction="x+",
        ),
        input_trans=dict(
            weight=1,
            in_slice_name="in_slice_1",
            out_slice_name="refl_slice_1",
            in_mode=mode,
            wl=wls,
            temp=[300],
            out_modes=(mode,),
            type="eigenmode",
            direction="x+",
            requires_grad=False,  # only used for normalization, no adjoint run is needed.
        ),
        top_cross_talk=dict(
            weight=-0.5,
            in_slice_name="in_slice_1",
            out_slice_name="top_slice",
            in_mode=mode,
            wl=wls,
            temp=[300],
            out_modes=(mode,),
            type="eigenmode",
            direction="y+",
        ),
        bot_cross_talk=dict(
            weight=-0.5,
            in_slice_name="in_slice_1",
            out_slice_name="bot_slice",
            in_mode=mode,
            wl=wls,
            temp=[300],
            out_modes=(mode,),
            type="eigenmode",
            direction="y-",
        ),
        override=True,
    )

    def fom_func(breakdown):
        input_trans = breakdown["input_trans"]["value"].detach()
        through = breakdown["fwd_trans"]["value"] / input_trans
        top_xt = breakdown["top_cross_talk"]["value"] / input_trans
        bot_xt = breakdown["bot_cross_talk"]["value"] / input_trans

        spectrum = {}
        fom = 0
        for wl, s21, s31, s41 in zip(wls, through, top_xt, bot_xt):
            spectrum[f"|s21|^2 at {wl:.2f}"] = {"weight": 1, "value": s21.item()}
            spectrum[f"|s31|^2 at {wl:.2f}"] = {"weight": 1, "value": s31.item()}
            spectrum[f"|s41|^2 at {wl:.2f}"] = {"weight": 1, "value": s41.item()}
            fom = (
                fom
                + breakdown["fwd_trans"]["weight"] * s21
                + breakdown["top_cross_talk"]["weight"] * s31
                + breakdown["bot_cross_talk"]["weight"] * s41
            )

        spectrum["avg|s21|^2"] = {"weight": 1, "value": through.mean().item()}
        spectrum["avg|s31|^2"] = {"weight": 1, "value": top_xt.mean().item()}
        spectrum["avg|s41|^2"] = {"weight": 1, "value": bot_xt.mean().item()}
        return fom, spectrum

    obj_cfgs["_fusion_func"] = fom_func

    design_region_param_cfgs = dict()
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[resolution, resolution],
            sigma=1 / resolution,
            transform=[
                dict(type="transpose_symmetry", rot_k=0),
                dict(type="mirror_symmetry", dims=[0, 1]),
                dict(
                    type="blur",
                    mfs=0.08,
                    resolutions=[hr_device.resolution, hr_device.resolution],
                    dim="xy",
                ),
                dict(type="binarize"),
            ],
            # init_method="diamond_0.3",
            init_method="constant_0.3",
            denorm_mode="inverse_eps",
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

    opt = CrossingOptimization(
        device=device,
        hr_device=hr_device,
        design_region_param_cfgs=design_region_param_cfgs,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)

    invdesign = InvDesign(
        devOptimization=opt,
        optimizer=Config(
            name="Adam",
            lr=0.2,
            # init_v=1e-10,
            # weight_decay=0,
            # name="cg_prp",
            # name="cg_hs-dy",
            # name="cg_hz",
            # name="cg_ls",
            # lr=10,
            # name="nesterov",
            # lr=1,
            # alg="bb_static",
            # constraint_fn=lambda x: x.clamp_(-0.5, 0.5),
        ),
        run=Config(
            start_epoch=0,
            n_epochs=30,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=0.02,
        ),
        sharp_scheduler=Config(
            mode="linear",
            name="sharpness",
            init_sharp=4,
            final_sharp=20,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=1,
            plot_name=f"{exp_name}",
            objs=["avg|s21|^2"],
            field_keys=[("in_slice_1", wl_cen, mode, 300)],
            in_slice_names=["in_slice_1"],
            exclude_slice_names=[],
            field_component="Ey",
            eps_grad=True,
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
            upsample_eps_to_1nm=True,
        ),
    )

    invdesign.optimize()
