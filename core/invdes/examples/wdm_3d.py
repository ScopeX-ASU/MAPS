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
import numpy as np
import torch
from pyutils.config import Config

from core.invdes.invdesign import InvDesign
from core.invdes.models import WDMOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import WDM
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

    thickness = 0.22
    mdm_region_size = (6, 6, thickness)
    port_len = 1.5

    input_port_width = 0.5
    output_port_width = 0.5
    num_outports = 2
    wl_cen = 1.56
    wl_width = 0.04
    n_wl = 2
    mode = "Hz1"
    exp_name = f"wdm_opt-port-{num_outports}_Si_3d_{mdm_region_size[0]}x{mdm_region_size[1]}x{mdm_region_size[2]}"

    sim_cfg.update(
        dict(
            solver="fdtdx",
            # border_width=[port_len, port_len, 2, 2],
            border_width=[0, 0, 1.5, 1.5, 0.7, 0.7],
            resolution=25,
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

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        trans_list = []
        ref_power = breakdown["wl1_input_trans"]["value"].detach()
        for wl_id in range(1, n_wl + 1):
            out_key = f"wl{wl_id}_trans"
            in_key = f"wl{wl_id}_input_trans"
            in_power = breakdown[in_key]["value"].detach()
            trans = breakdown[out_key]["value"] / in_power
            # trans = trans * (ref_power / in_power).detach()
            trans_list.append(trans)
            breakdown[out_key]["value"] = trans
            fom = fom + trans * breakdown[out_key]["weight"]
            # for j in range(1, num_outports + 1):
            #     other_key = f"wl{wl_id}_trans_p{j}"
            #     if other_key in breakdown:
            #         crosstalk = breakdown[other_key]["value"] / breakdown[in_key]["value"].detach()
            #         breakdown[other_key]["value"] = crosstalk
            #         fom = fom + crosstalk * breakdown[other_key]["weight"]
            # del breakdown[in_key]  # we don't need this in the backward pass, and it can be large since it's the input flux

        product = 0

        ## if only sum all transmission, lbfgs cannot balance them well.

        ## do not use direct product! ill-conditioned!
        # for i in range(1, n_wl + 1):
        #     product = product * breakdown[f"wl{i}_trans"]["value"]

        ## this sum-of-log formulation is more numerically stable and also encourages all transmissions to be high (since log is dominated by the smallest value)
        # for i in range(1, n_wl + 1):
        #     product = product + torch.log(trans_list[i - 1] + 1e-3)
        fom = fom  # + product * 10
        return fom, {"trans_product": {"weight": 1, "value": product}}

    def build_wdm_obj_cfgs(
        wls,  # e.g. [1.54, 1.56, 1.58, ...]
        desired_out_slices,  # e.g. ["out_slice_1", "out_slice_2", ...], same length as wls
        *,
        in_slice="in_slice_1",
        all_out_slices=("out_slice_1", "out_slice_2"),
        refl_slice="refl_slice_1",
        weights=dict(trans=1.0, xtalk=-2.0),
        temp=300,
        in_mode=mode,
        out_modes=(mode,),
        prop_dir="x+",
    ):
        """
        Returns a dict of objectives keyed like:
        wl{idx}_trans, wl{idx}_trans_p{j}, wl{idx}_refl_trans, wl{idx}_rad_trans_{dir}
        where idx starts at 1 in order of `wls`.
        """
        assert len(wls) == len(
            desired_out_slices
        ), "wls and desired_out_slices must have same length"

        cfg = {}
        for i, (wl, desired_out) in enumerate(zip(wls, desired_out_slices), start=1):
            wl_tag = f"wl{i}"

            # 1) Desired transmission (positive weight)
            cfg[f"{wl_tag}_trans"] = dict(
                weight=weights["trans"],
                in_slice_name=in_slice,
                out_slice_name=desired_out,
                wl=[wl],
                temp=[temp],
                in_mode=in_mode,
                out_modes=out_modes,
                type="eigenmode",
                direction=prop_dir,
            )

            # 2) Crosstalk penalties to other outputs (negative weight)
            for j, out_s in enumerate(all_out_slices, start=1):
                if out_s == desired_out:
                    continue
                cfg[f"{wl_tag}_trans_p{j}"] = dict(
                    weight=weights["xtalk"],
                    in_slice_name=in_slice,
                    out_slice_name=out_s,
                    wl=[wl],
                    temp=[temp],
                    in_mode=in_mode,
                    out_modes=out_modes,
                    type="eigenmode",
                    direction=prop_dir,
                )

            # 3) Reflection penalty at input (negative weight)
            cfg[f"{wl_tag}_input_trans"] = dict(
                weight=1,
                in_slice_name=in_slice,
                out_slice_name=refl_slice,
                wl=[wl],
                temp=[temp],
                in_mode=in_mode,
                out_modes=out_modes,
                type="eigenmode",
                direction=prop_dir,  # input source monitor
                requires_grad=False,  # only for normalization, not for adjoint simuation
            )

        return cfg

    # ---- Example: your 2-wavelength case ----
    wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl).tolist()
    # wls = [round(wl, 2) for wl in wls]
    desired_out_slices = [
        f"out_slice_{i}" for i in range(1, n_wl + 1)
    ]  # 1.54 -> port 1, 1.56 -> port 2
    obj_cfgs = build_wdm_obj_cfgs(
        wls,
        desired_out_slices,
        in_slice="in_slice_1",
        all_out_slices=tuple(desired_out_slices),
        refl_slice="refl_slice_1",
        weights=dict(trans=1, xtalk=-0.2),
        temp=300,
        in_mode=mode,
        out_modes=(mode,),
        prop_dir="x+",
    )

    obj_cfgs["_fusion_func"] = fom_func
    obj_cfgs["override"] = (
        True  # this will override the default obj_cfgs in WDMOptimization
    )

    device = WDM(
        material_r1="Si",
        sim_cfg=sim_cfg,
        box_size=mdm_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        port_height=(thickness, thickness),
        is_3d=True,
        mode=mode,
        num_outports=num_outports,
        port_box_margin=4.5,
        device=operation_device,
    )

    # hr_device = device.copy(resolution=1000)
    hr_device = device.copy(resolution=50)

    design_region_param_cfgs = dict()
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[50, 50],
            sigma=1 / 50,  # 2 * levelset knot grid step
            transform=[
                # dict(
                #     type="blur",
                #     mfs=0.100,
                #     resolutions=[hr_device.resolution, hr_device.resolution],
                #     dim="xy",
                # ),
                dict(type="binarize"),
            ],  # there is no symmetry in this design region
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

    print(device)
    opt = WDMOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        design_region_param_cfgs=design_region_param_cfgs,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)

    # print(opt.hr_eps_map.shape)

    ## first use lbfgs with fixed sharpness to get a good initial design.
    # invdesign = InvDesign(
    #     devOptimization=opt,
    #     optimizer=dict(
    #         name="lbfgs",
    #         lr=1,
    #         line_search_fn="strong_wolfe",
    #     ),
    #     run=Config(
    #         n_epochs=2,
    #     ),
    #     lr_scheduler=Config(
    #         name="cosine",
    #         lr_min=1,
    #     ),
    #     sharp_scheduler=Config(
    #         mode="cosine",
    #         name="sharpness",
    #         init_sharp=4,
    #         final_sharp=4,
    #     ),
    #     plot_cfgs=Config(
    #         plot=True,
    #         interval=5,
    #         plot_name=f"{exp_name}",
    #         objs=[f"wl{i}_trans" for i in range(1, n_wl + 1)],
    #         field_keys=[
    #             ("in_slice_1", wl, "Ez1", 300)
    #             for wl in np.linspace(
    #                 sim_cfg["wl_cen"] - sim_cfg["wl_width"] / 2,
    #                 sim_cfg["wl_cen"] + sim_cfg["wl_width"] / 2,
    #                 sim_cfg["n_wl"],
    #             )
    #         ],
    #         in_slice_names=["in_slice_1" for _ in range(sim_cfg["n_wl"])],
    #         exclude_slice_names=[],
    #     ),
    #     checkpoint_cfgs=Config(
    #         save_model=False,
    #         ckpt_name=f"{exp_name}",
    #         dump_gds=True,
    #         gds_name=f"{exp_name}",
    #     ),
    # )

    # invdesign.optimize()

    ## then switch to Adam for further optimization and better convergence
    invdesign = InvDesign(
        devOptimization=opt,
        optimizer=dict(
            # name="lbfgs",
            # lr=0.5,
            # line_search_fn="strong_wolfe",
            name="Adam",
            lr=0.2,
        ),
        run=Config(
            n_epochs=50,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=0.002,
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
            objs=[f"wl{i}_trans" for i in range(1, n_wl + 1)],
            field_keys=[
                ("in_slice_1", wl, mode, 300)
                for wl in np.linspace(
                    sim_cfg["wl_cen"] - sim_cfg["wl_width"] / 2,
                    sim_cfg["wl_cen"] + sim_cfg["wl_width"] / 2,
                    sim_cfg["n_wl"],
                )
            ],
            in_slice_names=["in_slice_1" for _ in range(sim_cfg["n_wl"])],
            exclude_slice_names=[],
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
