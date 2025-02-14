import torch
from torch import nn
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)
import torch

from core.invdes.invdesign import InvDesign
from core.invdes.models import (
    MetaLensOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MetaLens
from core.utils import set_torch_deterministic
from pyutils.config import Config
import h5py
from core.invdes import builder
from core.utils import print_stat
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random
sys.path.pop(0)
from test_metaatom_phase import get_mid_weight
from pyutils.general import ensure_dir

import torch
import torch.nn.functional as F
import h5py

def interpolate_1d(input_tensor, x0, x1, method="linear"):
    """
    Perform 1D interpolation on a tensor.
    
    Args:
        input_tensor (torch.Tensor): 1D tensor of shape (N,)
        x0 (torch.Tensor): Original positions of shape (N,)
        x1 (torch.Tensor): Target positions of shape (M,)
        method (str): Interpolation method ("linear" or "gaussian").
    
    Returns:
        torch.Tensor: Interpolated tensor of shape (M,)
    """
    if method == "linear":
        # linear interpolation
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        return F.interpolate(input_tensor, size=x1.shape[0], mode="linear", align_corners=False).squeeze()
    elif method == "gaussian":
        sigma = 0.1
        dist_sq = (x1.reshape(-1, 1) - x0.reshape(1, -1)).square().to(input_tensor.device)
        weights = (-dist_sq / (2 * sigma ** 2)).exp()
        weights = weights / weights.sum(dim=1, keepdim=True)
        return weights @ input_tensor
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")
    
def response_matching_loss(total_response, target_response, target_phase_variants):
    """
    Computes the MSE loss between total_phase and the closest value
    in the three versions of target_phase_shift: original, +2π, and -2π.
    
    Args:
        total_phase (torch.Tensor): Tensor of shape (N,) representing the computed phase.
        target_phase_shift (torch.Tensor): Tensor of shape (N,) representing the target phase shift.
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Compute absolute differences (3, N)
    target_phase = torch.angle(target_response)
    target_mag = torch.abs(target_response)

    total_phase = torch.angle(total_response)
    total_mag = torch.abs(total_response)
    # begin calculate the phase loss
    abs_diffs = torch.abs(target_phase_variants - total_phase.unsqueeze(0))  # Broadcasting

    # Find the index of the closest match at each point
    closest_indices = torch.argmin(abs_diffs, dim=0)  # Shape (N,)

    # Gather the closest matching values
    closest_values = target_phase_variants[closest_indices, torch.arange(total_phase.shape[0])]

    phase_error = closest_values - total_phase

    weight_map = (target_mag - torch.min(target_mag)) / (torch.max(target_mag) - torch.min(target_mag))

    phase_normalized_L2 = torch.norm(phase_error * weight_map) / (torch.norm(closest_values) + 1e-12)

    # begin calculate the magnitude loss

    mag_error = target_mag - total_mag
    mag_normalized_L2 = torch.norm(mag_error) / (torch.norm(target_mag) + 1e-12)

    print("this is the mag NL2norm: ", mag_normalized_L2.item(), "this is the phase NL2norm: ", phase_normalized_L2.item(), flush=True)
    return mag_normalized_L2 + phase_normalized_L2

def find_closest_width(lut, target_phase):
    """
    Given a LUT (dictionary) where keys are widths and values are phase shifts,
    find the width whose phase shift is closest to the target phase.

    Parameters:
    lut (dict): Dictionary with widths as keys and phase shifts as values.
    target_phase (float): The desired phase shift.

    Returns:
    float: The width corresponding to the closest phase shift.
    """
    closest_width = min(lut, key=lambda w: abs(lut[w] - target_phase))
    return closest_width

class PatchMetalens(nn.Module):
    def __init__(
        self,
        atom_period: float,
        patch_size: int,
        num_atom: int,
        probing_region_size: int,
        target_phase_response: torch.Tensor,
        LUT: dict = None,
        device: torch.device = torch.device("cuda:0"),
    ):
        super(PatchMetalens, self).__init__()
        self.atom_period = atom_period
        self.patch_size = patch_size
        self.num_atom = num_atom
        self.probing_region_size = probing_region_size
        self.num_dummy_atom = patch_size // 2 * 2
        self.target_phase_response = target_phase_response # this is used to initialize the metalens
        self.LUT = LUT
        self.device = device
        self.build_param()
        self.build_patch()
    
    def build_param(self):
        self.pillar_ls_knots = nn.Parameter(
            -0.05 * torch.ones((self.num_atom + self.num_dummy_atom), device=self.device)
        )
        if self.LUT is None:
            self.pillar_ls_knots.data = self.pillar_ls_knots.data + 0.01 * torch.randn_like(self.pillar_ls_knots.data)
        else:
            for i in range(self.num_atom):
                print(f"this is the width for idx {i} for the phase shift {self.target_phase_response[i, i].item()}", find_closest_width(self.LUT, self.target_phase_response[i, i].item()), flush=True)
                self.pillar_ls_knots.data[i + self.num_dummy_atom // 2] = get_mid_weight(0.05, find_closest_width(self.LUT, self.target_phase_response[i, i].item()))

    def build_patch(self):
        sim_cfg = DefaultSimulationConfig()
        total_sim_cfg = DefaultSimulationConfig()

        wl = 0.850
        sim_cfg.update(
            dict(
                solver="ceviche_torch",
                numerical_solver="solve_direct",
                use_autodiff=False,
                neural_solver=None,
                border_width=[0, 0, 0, 0],
                PML=[0.5, 0.5],
                resolution=50,
                wl_cen=wl,
                plot_root="./figs/patched_metalens",
            )
        )
        total_sim_cfg.update(
            dict(
                solver="ceviche_torch",
                numerical_solver="solve_direct",
                use_autodiff=False,
                neural_solver=None,
                border_width=[0, 0, 0.5, 0.5],
                PML=[0.5, 0.5],
                resolution=50,
                wl_cen=wl,
                plot_root="./figs/patched_metalens",
            )
        )
        patch_metalens = MetaLens(
            material_bg="Air",
            material_r = "Si",
            material_sub="SiO2",
            sim_cfg=sim_cfg,
            aperture=0.3 * self.patch_size,
            port_len=(1, 1),
            port_width=(0.3 * self.patch_size, 0.3),
            substrate_depth=0,
            ridge_height_max=0.75,
            nearfield_dx=0.3,
            nearfield_size=0.3 * self.probing_region_size,
            farfield_dxs=((30, 37.2),),
            farfield_sizes=(0.3,),
            device=self.device,
        )
        total_metalens = MetaLens(
            material_bg="Air",
            material_r = "Si",
            material_sub="SiO2",
            sim_cfg=total_sim_cfg,
            aperture=0.3 * self.num_atom,
            port_len=(1, 1),
            port_width=(0.3 * self.num_atom, 0.3),
            substrate_depth=0,
            ridge_height_max=0.75,
            nearfield_dx=0.3,
            nearfield_size=0.3 * self.num_atom,
            farfield_dxs=((30, 37.2),),
            farfield_sizes=(0.3,),
            device=self.device,
        )
        hr_patch_metalens = patch_metalens.copy(resolution=200)
        hr_total_metalens = total_metalens.copy(resolution=200)
        self.opt = MetaLensOptimization(
            device=patch_metalens,
            hr_device=hr_patch_metalens,
            sim_cfg=sim_cfg,
            operation_device=self.device,
        ).to(self.device)

        self.total_opt = MetaLensOptimization(
            device=total_metalens,
            hr_device=hr_total_metalens,
            sim_cfg=total_sim_cfg,
            operation_device=self.device,
        )

    def set_ref_response(
            self, 
            ref_mag: torch.Tensor, 
            ref_phase: torch.Tensor,
        ):
        self.ref_phase = ref_phase
        self.ref_mag = ref_mag

    def forward(self, sharpness):
        # in each time of forward, we need simulate the transfer matrix using the stiched patch metaatom
        # in each of the for loop, we need to run 13 times of simulation for differnet input port
        total_response = torch.zeros(
            (
                self.num_atom + self.num_dummy_atom, 
                self.num_atom + self.num_dummy_atom,
            ), 
            dtype=torch.complex128,
            device=self.device,
        )
        total_ls_knot = -0.05 * torch.ones(2 * (self.num_atom + self.num_dummy_atom) + 1, device=self.device)
        for i in range(self.num_atom):
            center_knot_idx = 2 * i + 1 + self.num_dummy_atom
            # total_ls_knot[1::2] = self.pillar_ls_knots
            self.level_set_knots = total_ls_knot.clone()
            self.level_set_knots[1::2] = self.pillar_ls_knots
            knots_value = {"design_region_0": self.level_set_knots[
                center_knot_idx - 2 * (self.patch_size // 2) - 1 : center_knot_idx + 2 * (self.patch_size // 2 + 1)
            ].unsqueeze(0)}
            source = torch.zeros(self.patch_size, device=self.device)
            source[self.patch_size // 2] = 1
            source = source.repeat_interleave(int(self.atom_period * 50))
            custom_source = dict(
                source=source,
                slice_name="in_slice_1",
                mode="Hz1",
                wl=0.85,
                direction="x+",
            )
            _ = self.opt(sharpness=sharpness, ls_knots=knots_value, custom_source=custom_source)
            response = self.opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["fz"].squeeze()
            response = response[
                int(self.atom_period * 50) // 2 :: int(self.atom_period * 50)
            ]
            assert len(response) == self.probing_region_size, f"{self.probing_region_size}!={len(response)}"
            row_idx = i + self.num_dummy_atom // 2
            col_start = row_idx - self.probing_region_size // 2
            col_end   = col_start + self.probing_region_size

            # Construct a partial matrix that depends on `response`
            partial_matrix = torch.zeros_like(total_response)
            partial_matrix[row_idx, col_start:col_end] = response
            
            # Now accumulate
            total_response = total_response + partial_matrix
            # self.opt.plot(
            #     plot_filename=f"patched_metalens_sharp-{sharpness}_{i}.png",
            #     eps_map=None,
            #     obj=None,
            #     field_key=("in_slice_1", 0.85, "Hz1", 300),
            #     field_component="Hz",
            #     in_slice_name="in_slice_1",
            #     exclude_slice_names=[],
            # )
            # print_stat(trust_worthy_phase)
            # trust_worthy_phase_list.append(trust_worthy_phase)
            # print("this is the shape of the trust_worthy_phase", trust_worthy_phase.shape, flush=True)
        total_response = total_response[
            self.num_dummy_atom // 2 : -self.num_dummy_atom // 2,
            self.num_dummy_atom // 2 : -self.num_dummy_atom // 2,
        ].transpose(0, 1)
        # plot the total_phase (1 D tensor)
        # plt.figure()
        # plt.plot(total_phase.detach().cpu().numpy())
        # plt.savefig("./figs/stiched_phase.png")
        # plt.close()
        return total_response


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_torch_deterministic()
    # read LUT from csv file
    csv_file = "./unitest/metaatom_phase_response.csv"
    LUT = {}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if float(row[0]) > 0.14:
                break
            LUT[float(row[0])] = float(row[1])
    # print(LUT)
    # quit()
    # read ONN weights from pt file
    # -----------------------------------------
    # checkpoint_file_name = "/home/pingchua/projects/MetaONN/checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0010_wb-16_ib-16_rotm-fixed_c-two_atom_wise_no_smooth_cweight_acc-92.38_epoch-55.pt"
    # checkpoint = torch.load(checkpoint_file_name)
    # model_comment = checkpoint_file_name.split("_c-")[-1].split("_acc-")[0]
    # state_dict = checkpoint["state_dict"]

    # target_response = state_dict["features.conv1.conv._conv_pos.weight"].squeeze()[0].to(device)
    # -----------------------------------------
    # read transfer matrix from h5 file
    transfer_matrix_file = "figs/metalens_TF_wl-0.85_p-0.3_mat-Si/transfer_matrix.h5"
    model_comment = transfer_matrix_file.split("_TF_")[-1].split("/")[0]
    with h5py.File(transfer_matrix_file, "r") as f:
        A = torch.tensor(f["transfer_matrix"], device=device)
    target_response = A
    target_phase_response = torch.angle(target_response)

    atom_period = 0.3
    patch_size = 17
    num_atom = 32

    # Create 3 variants of the target phase shift
    target_phase_variants = torch.stack([
        target_phase_response,         # Original
        target_phase_response + 2 * torch.pi,  # Shifted +2π
        target_phase_response - 2 * torch.pi   # Shifted -2π
    ], dim=0)  # Shape (3, N)

    patch_metalens = PatchMetalens(
        atom_period=atom_period,
        patch_size=patch_size,
        num_atom=num_atom,
        probing_region_size=13,
        target_phase_response=target_phase_response,
        LUT=LUT,
        device=device,
    )
    # Define the optimizer
    num_epoch = 100
    lr = 2e-3
    # lr = 5e-2
    optimizer = torch.optim.Adam(patch_metalens.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=lr * 1e-2
    )

    # before we optimize the metalens, we need to simulate the ref phase:
    ref_level_set_knots = -0.05 * torch.ones(2 * (patch_metalens.num_atom + patch_metalens.num_dummy_atom) + 1, device=device)
    ref_level_set_knots = ref_level_set_knots + 0.001 * torch.randn_like(ref_level_set_knots)
    _ = patch_metalens.total_opt(sharpness=256, ls_knots={"design_region_0": ref_level_set_knots[16:-16].unsqueeze(0)})
    ref_phase = patch_metalens.total_opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["phase"].squeeze().mean().to(torch.float32)
    ref_mag = patch_metalens.total_opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["mag"].squeeze().mean().to(torch.float32)
    patch_metalens.set_ref_response(ref_mag, ref_phase)
    plot_root = f"./figs/ONN_{model_comment}/"
    ensure_dir(plot_root)
    sources = torch.eye(num_atom, device=device)
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        stiched_response = patch_metalens(sharpness=epoch + 120)
        loss = response_matching_loss(stiched_response, target_response, target_phase_variants)
        loss.backward()
        # for name, param in patch_metalens.named_parameters():
        #     if param.grad is not None:
        #         print(name, torch.norm(param.grad))
        # quit()
        optimizer.step()
        scheduler.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")
        if epoch > num_epoch - 110:
            with torch.no_grad():
                full_wave_response = torch.zeros((num_atom, num_atom), device=device, dtype=torch.complex128)
                for idx in range(num_atom):
                    source_i = sources[idx].repeat_interleave(int(atom_period * 50))
                    if idx == 0:
                        plt.figure()
                        plt.plot(source_i.cpu().numpy())
                        plt.savefig(plot_root + f"source_{epoch}.png")
                    custom_source = dict(
                        source=source_i,
                        slice_name="in_slice_1",
                        mode="Hz1",
                        wl=0.85,
                        direction="x+",
                    )
                    _ = patch_metalens.total_opt(
                        sharpness=256, 
                        ls_knots={"design_region_0": patch_metalens.level_set_knots[16:-16].unsqueeze(0)},
                        custom_source=custom_source
                    )
                    if idx == 0:
                        patch_metalens.total_opt.plot(
                            plot_filename=f"total_metalens_epoch_{epoch}.png",
                            eps_map=patch_metalens.total_opt._eps_map,
                            obj=None,
                            field_key=("in_slice_1", 0.85, "Hz1", 300),
                            field_component="Hz",
                            in_slice_name="in_slice_1",
                            exclude_slice_names=[],
                        )
                    response = patch_metalens.total_opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["fz"].squeeze()
                    response = response[int(atom_period * 50) // 2 :: int(atom_period * 50)]
                    assert len(response) == num_atom, f"{num_atom}!={len(response)}"
                    full_wave_response[idx] = response
                full_wave_response = full_wave_response.transpose(0, 1)
        else:
            full_phase = None
            full_mag = None
        
        figure, ax = plt.subplots(1, 4, figsize=(20, 5))
        im0 = ax[0].imshow(target_response.abs().cpu().numpy())
        ax[0].set_title("Target Magnitude")
        figure.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(stiched_response.detach().abs().cpu().numpy())
        ax[1].set_title("Stitched Magnitude")
        figure.colorbar(im1, ax=ax[1])
        im2 = ax[2].imshow(full_wave_response.abs().cpu().numpy())
        ax[2].set_title("Full Magnitude")
        figure.colorbar(im2, ax=ax[2])
        im3 = ax[3].imshow((full_wave_response - target_response).abs().cpu().numpy())
        ax[3].set_title("Difference Magnitude")
        figure.colorbar(im3, ax=ax[3])
        plt.savefig(plot_root + f"epoch_{epoch}.png")
