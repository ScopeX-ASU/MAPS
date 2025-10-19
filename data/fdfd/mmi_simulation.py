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
import argparse
import random
import numpy as np

import torch
import torch.nn.functional as F

from autograd.numpy import array as npa
from pyutils.config import Config

from core.invdes.invdesign import InvDesign
from core.invdes.models import MMIOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MMI
from core.utils import (
    DeterministicCtx,
    SharpnessScheduler,
    print_stat,
    set_torch_deterministic,
)
from thirdparty.ceviche.constants import *


def compare_designs(design_regions_1, design_regions_2):
    similarity = []
    for k, v in design_regions_1.items():
        v1 = v
        v2 = design_regions_2[k]
        similarity.append(F.cosine_similarity(v1.flatten(), v2.flatten(), dim=0))
    return torch.mean(torch.stack(similarity)).item()


def _align_smatrix_phases(current: np.ndarray, reference: np.ndarray, atol: float = 1e-12):
    """Align per-port phases of `current` to match `reference` up to numerical tolerance."""
    aligned = current.copy()
    num_inputs, num_outputs = reference.shape

    col_factors = np.ones(num_outputs, dtype=np.complex128)
    for col in range(num_outputs):
        row = int(np.argmax(np.abs(reference[:, col])))
        ref_entry = reference[row, col]
        cur_entry = aligned[row, col]
        if abs(ref_entry) <= atol or abs(cur_entry) <= atol:
            continue
        phase = np.angle(cur_entry) - np.angle(ref_entry)
        factor = np.exp(-1j * phase)
        col_factors[col] = factor
        aligned[:, col] *= factor

    row_factors = np.ones(num_inputs, dtype=np.complex128)
    for row in range(num_inputs):
        col = int(np.argmax(np.abs(reference[row, :])))
        ref_entry = reference[row, col]
        cur_entry = aligned[row, col]
        if abs(ref_entry) <= atol or abs(cur_entry) <= atol:
            continue
        phase = np.angle(cur_entry) - np.angle(ref_entry)
        factor = np.exp(-1j * phase)
        row_factors[row] = factor
        aligned[row, :] *= factor

    return aligned, row_factors, col_factors


def mmi_simulation(
    device_id,
    operation_device,
    each_step=False,
    include_perturb=False,
    perturb_probs=[0.05, 0.1, 0.15],
    image_path="/home/hzhou144/projects/MAPS_local/data/fdfd/wdm/raw_opt_traj_ptb/wdm_id-0_opt_step_85-in_slice_1-1.56-Ez1-300.png",
    raw_data="/home/hzhou144/projects/MAPS_local/data/fdfd/mmi/raw_opt_traj_ptb/mmi_id-0_opt_step_0-in_slice_3-1.55-Ez1-300.h5",
):
    
    set_torch_deterministic(int(device_id))
    dump_data_path = f"./data/fdfd/mmi/raw_opt_traj_ptb"
    sim_cfg = DefaultSimulationConfig()

    mmi_region_size = (4, 4)
    # mmi_region_size = (6, 6)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.48
    exp_name = "mmi_opt"

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 1, 1],  # left, right, lower, upper, containing PML
            resolution=100,
            plot_root=f"./figs/{exp_name}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )
    num_inports = 3
    num_outports = 3

    target_unitary_smatrix = torch.randn(
        (num_inports, num_outports), dtype=torch.complex64, device=operation_device
    )
    u, s, v = torch.linalg.svd(target_unitary_smatrix)
    target_unitary_smatrix = torch.matmul(u, v) * 0.95
    target_norm = torch.norm(target_unitary_smatrix)
    print(f"target unitary smatrix:\n{target_unitary_smatrix}")
    """ e.g., target unitary smatrix:
    [[-0.1627-0.6450j,  0.6381-0.2135j, -0.0702-0.0481j],
     [-0.5426-0.0271j, -0.0188+0.5920j, -0.5033+0.0567j],
     [-0.2311+0.3339j,  0.2916+0.1178j,  0.2743-0.7506j]]

    trained matrix, 0.0581 relative L2 norm error
    [[-0.1511-0.6146j,  0.6196-0.2099j, -0.0684-0.0495j,]
     [-0.4896-0.0319j, -0.0191+0.5528j, -0.4856+0.0568j,]
     [-0.2111+0.2967j,  0.2755+0.1021j,  0.2669-0.7242j]
    """

    def fom_func(breakdown):
        ## maximization fom
        for key, obj in breakdown.items():
            if key == "smatrix":
                s_matrix = obj["value"]  # shape (num_outports, num_inports)
                fom = (
                    -torch.norm(s_matrix - target_unitary_smatrix.flatten())
                    / target_norm
                )
        # for key, obj in breakdown.items():
        #     if "fwd_trans" in key:
        #         fom = fom * obj["value"]
        #     elif "phase" in key:
        #         s_param_list.append(obj["value"])
        # if len(s_param_list) == 0:
        #     return fom
        # if isinstance(s_param_list[0], torch.Tensor):
        #     s_params = torch.tensor(s_param_list)
        #     s_params_std = torch.std(s_params)
        #     fom = fom - s_params_std
        # else:
        #     s_params = npa.array(s_param_list)
        #     s_params_std = npa.std(s_params)
        #     fom = fom - s_params_std
        # maximize the forward transmission and minimize the standard deviation of the s-parameters
        return fom, {"smatrix_err": {"weight": -1, "value": fom}}

    obj_cfgs = dict(_fusion_func=fom_func)

    device = MMI(
        sim_cfg=sim_cfg,
        box_size=mmi_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        num_inports=num_inports,
        num_outports=num_outports,
        device=operation_device,
        port_box_margin=0.5,
    )

    hr_device = device.copy(resolution=1000)
    print(device)
    opt = MMIOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
        obj_cfgs=obj_cfgs,
    ).to(operation_device)

    print(opt)

    results = opt.evaluation(image_path, raw_data, read_S_parameter=True)
    current_s_matrix = results["breakdown"]["smatrix"]["value"].cpu().numpy()
    num_inports = opt.device.num_inports
    num_outports = opt.device.num_outports
    current_matrix = current_s_matrix.reshape(num_inports, num_outports)

    true_s_matrix = results.get("true_s_matrix")
    if true_s_matrix is not None:
        reference_matrix = true_s_matrix.reshape(num_outports, num_inports)
        # raw_error = np.linalg.norm(current_matrix - reference_matrix, ord=2)
        aligned_matrix, row_phase, col_phase = _align_smatrix_phases(
            current_matrix, reference_matrix
        )
        diff = aligned_matrix - reference_matrix
        aligned_error = np.linalg.norm(diff, ord='fro') / np.linalg.norm(reference_matrix, ord='fro')

        # print("current_s_matrix:", current_matrix.reshape(-1))
        print("true_s_matrix:", reference_matrix.reshape(-1))
        # print("s_matrix_error:", raw_error)
        print("aligned_s_matrix:", aligned_matrix.reshape(-1))
        print("aligned_s_matrix_error:", aligned_error)
        # print(
        #     "input_phase_offsets_deg:",
        #     np.degrees(np.angle(row_phase)),
        # )
        # print(
        #     "output_phase_offsets_deg:",
        #     np.degrees(np.angle(col_phase)),
        # )
    else:
        print("current_s_matrix:", current_matrix.reshape(-1))
        print("No ground-truth S matrix stored in the HDF5 file.")
    ## need to read in the true S_parameter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    # parser.add_argument("--image_path", type=str, default="/home/hzhou144/projects/MAPS_local/data/fdfd/mmi/raw_opt_traj_ptb/mmi_id-0_opt_step_0-in_slice_3-1.55-Ez1-300.png")
    parser.add_argument("--image_path", type=str, default="/home/hzhou144/projects/MAPS_local/data/fdfd/data_fig/fdfd/mmi/prefab_prediction_wo_correction.png")
    # parser.add_argument("--image_path", type=str, default="/home/hzhou144/projects/MAPS_local/data/fdfd/data_fig/fdfd/mmi/corrected_design_prediction.png")
    parser.add_argument("--raw_data", type=str, default="/home/hzhou144/projects/MAPS_local/data/fdfd/mmi/raw_opt_traj_ptb/mmi_id-0_opt_step_0-in_slice_3-1.55-Ez1-300.h5")
    random_seed = parser.parse_args().random_seed
    gpu_id = parser.parse_args().gpu_id
    image_path = parser.parse_args().image_path
    raw_data = parser.parse_args().raw_data
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + random_seed))
    mmi_simulation(random_seed, device, image_path=image_path, raw_data=raw_data)

if __name__ == "__main__":
    main()
