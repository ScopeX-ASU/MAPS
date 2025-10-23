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

    mdm_region_size = (24, 24)
    port_len = 1.8

    input_port_width = 0.85
    output_port_width = 0.85
    num_outports = 4
    wl_cen = 1.56
    wl_width = 0.06
    n_wl = 4
    exp_name = f"wdm_opt-port-{num_outports}_SiN_neff1.7_{mdm_region_size[0]}x{mdm_region_size[1]}"

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            # border_width=[port_len, port_len, 2, 2],
            border_width=[0, 0, 2, 2],
            resolution=25,
            plot_root=f"./figs/{exp_name}",
            PML=[0.5, 0.5],
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
        for key, obj in breakdown.items():
            # if key in {f"wl{i}_trans" for i in range(1, n_wl + 1)}:
            #     continue
            fom = fom + obj["weight"] * obj["value"]

        ## add extra temp mul
        product = 1
        # for i in range(1, n_wl + 1):
        #     product = product * breakdown[f"wl{i}_trans"]["value"]
        # fom = fom + product * 10
        return fom, {"trans_product": {"weight": 1, "value": product}}

    def build_wdm_obj_cfgs(
        wls,                              # e.g. [1.54, 1.56, 1.58, ...]
        desired_out_slices,               # e.g. ["out_slice_1", "out_slice_2", ...], same length as wls
        *,
        in_slice="in_slice_1",
        all_out_slices=("out_slice_1", "out_slice_2"),
        refl_slice="refl_slice_1",
        rad_slices=dict(xp="rad_slice_xp", xm="rad_slice_xm", yp="rad_slice_yp", ym="rad_slice_ym"),
        weights=dict(trans=1.0, xtalk=-2.0, refl=-1.0, rad=-2.0),
        temp=300,
        in_mode="Ez1",
        out_modes=("Ez1",),
        prop_dir="x+"
    ):
        """
        Returns a dict of objectives keyed like:
        wl{idx}_trans, wl{idx}_trans_p{j}, wl{idx}_refl_trans, wl{idx}_rad_trans_{dir}
        where idx starts at 1 in order of `wls`.
        """
        assert len(wls) == len(desired_out_slices), "wls and desired_out_slices must have same length"

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
            cfg[f"{wl_tag}_refl_trans"] = dict(
                weight=weights["refl"],
                in_slice_name=in_slice,
                out_slice_name=refl_slice,
                wl=[wl],
                temp=[temp],
                in_mode=in_mode,
                out_modes=out_modes,
                type="flux_minus_src",
                direction="x",  # reflection along x
            )

            # 4) Radiation penalties (negative weight) in ±x, ±y
            for dir_key, rad_slice in rad_slices.items():
                cfg[f"{wl_tag}_rad_trans_{dir_key}"] = dict(
                    weight=weights["rad"],
                    in_slice_name=in_slice,
                    out_slice_name=rad_slice,
                    wl=[wl],
                    temp=[temp],
                    in_mode=in_mode,
                    out_modes=out_modes,
                    type="flux",
                    direction="x" if dir_key in ("xp", "xm") else "y",
                )

        return cfg

    # ---- Example: your 2-wavelength case ----
    wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl).tolist()
    wls = [round(wl, 2) for wl in wls]
    desired_out_slices = [f"out_slice_{i}" for i in range(1, n_wl + 1)]  # 1.54 -> port 1, 1.56 -> port 2
    obj_cfgs = build_wdm_obj_cfgs(
        wls,
        desired_out_slices,
        in_slice="in_slice_1",
        all_out_slices=tuple(desired_out_slices),
        refl_slice="refl_slice_1",
        rad_slices=dict(xp="rad_slice_xp", xm="rad_slice_xm", yp="rad_slice_yp", ym="rad_slice_ym"),
        weights=dict(trans=1, xtalk=-0.2, refl=-0.1, rad=-0.2),
        temp=300,
        in_mode="Ez1",
        out_modes=("Ez1",),
        prop_dir="x+",
    )

    obj_cfgs["_fusion_func"] = fom_func

    device = WDM(
        material_r1="SiN_eff",
        sim_cfg=sim_cfg,
        box_size=mdm_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        num_outports=num_outports,
        port_box_margin=4.5,
        device=operation_device,
    )

    hr_device = device.copy(resolution=200)
    print(device)
    opt = WDMOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(
        devOptimization=opt,
        optimizer=dict(
            # name="Adam",
            name="lbfgs",
            lr=0.1,
            # weight_decay=0.005,
            # use_bb=False,
            line_search_fn="strong_wolfe",
        ),
        run=Config(
            n_epochs=100,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=5,
            plot_name=f"{exp_name}",
            objs=[f"wl{i}_trans" for i in range(1, n_wl + 1)],
            field_keys=[
                ("in_slice_1", wl, "Ez1", 300)
                for wl in np.linspace(
                    sim_cfg["wl_cen"] - sim_cfg["wl_width"] / 2,
                    sim_cfg["wl_cen"] + sim_cfg["wl_width"] / 2,
                    sim_cfg["n_wl"],
                )
            ],
            in_slice_names=["in_slice_1" for _ in range(sim_cfg["n_wl"])],
            exclude_slice_names=[],
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
        ),
    )
    invdesign.optimize()
