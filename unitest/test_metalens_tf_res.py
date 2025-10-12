"""
in this file, we will test the transfer function of metalens by the following methods:

we first random generate input wavefront [32], and then simulate the output wavefront [32] by FDFD.

and then use minimum square method to fit the transfer matrix [32 * 32] of metalens.

see how the transfer matrix is close to a diagonal one.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pyutils.general import ensure_dir
from test_metaatom_phase import get_mid_weight

from core.invdes.models import MetaLensOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MetaLens
from core.utils import DeterministicCtx, set_torch_deterministic

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_torch_deterministic(41)
    num_meta_atom = 32
    res = 50
    wl = 0.85
    atom_period = 0.3
    pillar_material = "Si"
    near_field_dx = 0.3
    with DeterministicCtx(seed=41):

        ls_knot = torch.randn(2 * num_meta_atom + 1, device=device)
        ls_knot[::2] = -0.05 * torch.ones(num_meta_atom + 1, device=device)
        width = torch.rand(num_meta_atom, device=device) * (0.14 - 0.03) + 0.03
        width = [get_mid_weight(0.05, round(w.item(), 2), atom_period) for w in width]
        width = torch.tensor(width, device=device)
        # width = torch.ones(num_meta_atom, device=device) * 0.05
        ls_knot[1::2] = width
        knots_value = {"design_region_0": ls_knot.unsqueeze(0)}

    sim_cfg = DefaultSimulationConfig()
    exp_name = f"test_sample_period"
    # exp_name = f"metalens_TF_wl-{wl}_p-{atom_period}_mat-{pillar_material}"
    plot_root = f"./figs/{exp_name}"
    ensure_dir(plot_root)

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            numerical_solver="solve_direct",
            use_autodiff=False,
            neural_solver=None,
            border_width=[0, 0, 0.5, 0.5],
            PML=[0.5, 0.5],
            resolution=res,
            wl_cen=wl,
            plot_root=plot_root,
        )
    )

    metalens = MetaLens(
        material_bg="Air",
        material_r=pillar_material,
        material_sub="SiO2",
        sim_cfg=sim_cfg,
        aperture=atom_period * num_meta_atom,
        port_len=(1, 2),
        port_width=(atom_period * num_meta_atom, atom_period),
        substrate_depth=0,
        ridge_height_max=0.75,
        nearfield_dx=near_field_dx,
        nearfield_size=atom_period * num_meta_atom,
        farfield_dxs=((30, 37.2),),
        farfield_sizes=(atom_period,),
        device=device,
    )
    hr_metalens = metalens.copy(resolution=200)
    design_region_param_cfgs = dict(
        rho_resolution=[0, 2 / atom_period],
    )
    obj_cfgs = dict(
        near_field_response_record=dict(
            wl=[wl],
        ),
    )
    opt = MetaLensOptimization(
        device=metalens,
        design_region_param_cfgs=design_region_param_cfgs,
        hr_device=hr_metalens,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=device,
    ).to(device)

    norm_l2_list = []
    input_sample_period_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    A_ref = None
    for input_sample_period in input_sample_period_list:
        # for input_sample_period in [15, 5, 1]:
        print(f"input_sample_period={input_sample_period}", flush=True)
        prob_sources = torch.eye(
            round(num_meta_atom * atom_period * res // input_sample_period),
            device=device,
        )
        response_list = []
        for i in range(num_meta_atom):
            # source_i = interpolate_1d(sources[i], torch.linspace(0, num_meta_atom*0.3, num_meta_atom), torch.linspace(0, num_meta_atom*0.3, int(num_meta_atom*0.3*res)), method="gaussian")
            source_i = prob_sources[i].repeat_interleave(int(input_sample_period))
            source_zero_padding = torch.zeros(int(0.5 * res), device=device)
            source_i = torch.cat([source_zero_padding, source_i, source_zero_padding])
            custom_source = dict(
                source=source_i,
                slice_name="in_slice_1",
                mode="Hz1",
                wl=wl,
                direction="x+",
            )
            with torch.no_grad():
                _ = opt(
                    sharpness=256, ls_knots=knots_value, custom_source=custom_source
                )
                # opt.plot(
                #     plot_filename=f"nearfield_source_{i}.png",
                #     eps_map=opt._eps_map,
                #     obj=None,
                #     field_key=("in_slice_1", wl, "Hz1", 300),
                #     field_component="Hz",
                #     in_slice_name="in_slice_1",
                #     exclude_slice_names=["in_slice_1", "refl_slice_1", "nearfield_1"],
                # )
                # read the output wavefront
                response = opt.objective.response[
                    ("in_slice_1", "nearfield_1", wl, "Hz1", 300)
                ]["fz"]
                response = response.squeeze()
                out_response = response[
                    int(input_sample_period) // 2 :: int(input_sample_period)
                ]
                assert len(out_response) == round(
                    num_meta_atom * atom_period * res // input_sample_period
                ), f"len(out_phase)={len(out_response)}"
                response_list.append(out_response)
        out_response = torch.stack(response_list)
        # fit the transfer matrix
        A = out_response.transpose(0, 1)

        # interpolate A to the same size as GT_response using nearest interpolation
        # Separate real and imaginary parts
        A_real = A.real
        A_imag = A.imag

        # Interpolate real and imaginary parts separately
        A_real_resized = F.interpolate(
            A_real.unsqueeze(0).unsqueeze(0), size=(480, 480), mode="nearest"
        ).squeeze()
        A_imag_resized = F.interpolate(
            A_imag.unsqueeze(0).unsqueeze(0), size=(480, 480), mode="nearest"
        ).squeeze()

        # Recombine into a complex tensor
        A = A_real_resized + 1j * A_imag_resized

        if input_sample_period == 1:
            A_ref = A

        # calculate the normalized L2 distance
        norm_l2 = torch.norm(A - A_ref) / (torch.norm(A_ref) + 1e-12)
        norm_l2_list.append(norm_l2.item())
        print(
            f"norm_l2={norm_l2} @ input_sample_period={input_sample_period}", flush=True
        )

    input_sample_period = np.array(input_sample_period_list)
    norm_l2 = np.array(norm_l2_list)

    # write it to csv file
    with open(f"{plot_root}/norm_l2.csv", "w") as f:
        f.write("input_sample_period,norm_l2\n")
        for i in range(len(input_sample_period)):
            f.write(f"{input_sample_period[i]},{norm_l2[i]}\n")
