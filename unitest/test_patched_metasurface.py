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
    
def phase_matching_loss(total_phase, target_variants):
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
    abs_diffs = torch.abs(target_variants - total_phase.unsqueeze(0))  # Broadcasting

    # Find the index of the closest match at each point
    closest_indices = torch.argmin(abs_diffs, dim=0)  # Shape (N,)

    # Gather the closest matching values
    closest_values = target_variants[closest_indices, torch.arange(target_phase_shift.shape[0])]

    # Compute mean squared error loss
    loss = torch.nn.functional.mse_loss(closest_values, total_phase)

    return loss

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
        target_phase_shift: torch.Tensor,
        LUT: dict = None,
        device: torch.device = torch.device("cuda:0"),
    ):
        super(PatchMetalens, self).__init__()
        self.atom_period = atom_period
        self.patch_size = patch_size
        self.num_atom = num_atom
        self.num_dummy_atom = patch_size // 2 * 2
        self.target_phase_shift = target_phase_shift # this is used to initialize the metalens
        self.target_phase_shift = self.target_phase_shift.chunk(self.num_atom)
        self.target_phase_shift = [phase_shift.mean() for phase_shift in self.target_phase_shift]
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
                print(f"this is the width for idx {i} for the phase shift {self.target_phase_shift[i].item()}", find_closest_width(self.LUT, self.target_phase_shift[i].item()), flush=True)
                self.pillar_ls_knots.data[i + self.num_dummy_atom // 2] = get_mid_weight(0.05, find_closest_width(self.LUT, self.target_phase_shift[i].item()))

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
                PML=[0.5, 0],
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
            nearfield_size=0.3,
            farfield_dxs=((30, 37.2),),
            farfield_sizes=(0.3,),
            device=self.device,
        )
        total_metalens = MetaLens(
            material_bg="Air",
            material_r = "Si",
            material_sub="SiO2",
            sim_cfg=sim_cfg,
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
            # obj_cfgs=phase_shift_recoder_cfg,
            operation_device=self.device,
        ).to(self.device)

        self.total_opt = MetaLensOptimization(
            device=total_metalens,
            hr_device=hr_total_metalens,
            sim_cfg=total_sim_cfg,
            # obj_cfgs=phase_shift_recoder_cfg,
            operation_device=self.device,
        )

    def set_ref_phase(self, ref_phase):
        self.ref_phase = ref_phase

    def forward(self, sharpness):
        # set the level set knots to the self.opt parameters and 
        # then use the forward of the self.opt to get the phase shift mask
        # then stich the trust worthy region to the phase shift mask
        # design_region_param_dict.design_region_0.weights.ls_knots
        trust_worthy_phase_list = []
        total_ls_knot = -0.05 * torch.ones(2 * (self.num_atom + self.num_dummy_atom) + 1, device=self.device)
        for i in range(self.num_atom):
            center_knot_idx = 2 * i + 1 + self.num_dummy_atom
            # total_ls_knot[1::2] = self.pillar_ls_knots
            self.level_set_knots = total_ls_knot.clone()
            self.level_set_knots[1::2] = self.pillar_ls_knots
            knots_value = {"design_region_0": self.level_set_knots[
                center_knot_idx - 2 * (self.patch_size // 2) - 1 : center_knot_idx + 2 * (self.patch_size // 2 + 1)
            ].unsqueeze(0)}
            _ = self.opt(sharpness=sharpness, ls_knots=knots_value)
            trust_worthy_phase = (self.ref_phase - self.opt.objective.phase_shift[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["phase"]) % (2 * torch.pi)
            self.opt.plot(
                plot_filename=f"patched_metalens_sharp-{sharpness}_{i}.png",
                eps_map=None,
                obj=None,
                field_key=("in_slice_1", 0.85, "Hz1", 300),
                field_component="Hz",
                in_slice_name="in_slice_1",
                exclude_slice_names=[],
            )
            # print_stat(trust_worthy_phase)
            # trust_worthy_phase_list.append(trust_worthy_phase)
            # print("this is the shape of the trust_worthy_phase", trust_worthy_phase.shape, flush=True)
            trust_worthy_phase_list.append(trust_worthy_phase[:, :-1])
        total_phase = torch.cat(trust_worthy_phase_list, dim=1).squeeze().to(torch.float32)
        # plot the total_phase (1 D tensor)
        # plt.figure()
        # plt.plot(total_phase.detach().cpu().numpy())
        # plt.savefig("./figs/stiched_phase.png")
        # plt.close()
        return total_phase


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_torch_deterministic()
    # read LUT from csv file
    csv_file = "./unitest/metaatom_phase.csv"
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
    checkpoint_file_name = "/home/pingchua/projects/MetaONN/checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0010_wb-16_ib-16_rotm-fixed_c-two_atom_wise_no_smooth_acc-92.07_epoch-80.pt"
    checkpoint = torch.load(checkpoint_file_name)
    model_comment = checkpoint_file_name.split("_c-")[-1].split("_acc-")[0]
    state_dict = checkpoint["state_dict"]

    target_phase_shift = state_dict["features.conv1.conv._conv_pos.weight"].squeeze()[0].to(device)
    target_phase_shift = target_phase_shift % (2 * torch.pi)
    target_phase_shift = interpolate_1d(target_phase_shift, torch.linspace(0, 32*0.3, target_phase_shift.shape[0]), torch.linspace(0, 32*0.3, 480), method="gaussian")
    # plt.figure()
    # plt.plot(target_phase_shift.detach().cpu().numpy())
    # plt.savefig("./figs/target_phase_shift.png")
    # plt.close()
    # quit()

    # # Generate a 1D tensor with 480 points within the range (-3.14, 3.14)
    # num_points = 480
    # x_tensor = torch.linspace(-3.14, 3.14, num_points).to(device)

    # # Define a more complex function for the phase shift mask
    # target_phase_shift = (
    #     1.2 * torch.sin(1.5 * x_tensor) * torch.cos(2.5 * x_tensor)
    #     + 0.8 * torch.sin(3.2 * x_tensor + 0.5)
    #     + 0.5 * torch.cos(4.8 * x_tensor - 1)
    #     + 0.3 * torch.sin(6.5 * x_tensor)
    # )

    # # Normalize to range [0, 2pi]
    # target_phase_shift = (target_phase_shift - target_phase_shift.min()) / (
    #     target_phase_shift.max() - target_phase_shift.min()
    # ) * (2 * torch.pi)

    # target_phase_shift = target_phase_shift.to(device)

    atom_period = 0.3
    patch_size = 17
    num_atom = 32

    # Create 3 variants of the target phase shift
    target_variants = torch.stack([
        target_phase_shift,         # Original
        target_phase_shift + 2 * torch.pi,  # Shifted +2π
        target_phase_shift - 2 * torch.pi   # Shifted -2π
    ], dim=0)  # Shape (3, N)

    patch_metalens = PatchMetalens(
        atom_period=atom_period,
        patch_size=patch_size,
        num_atom=num_atom,
        target_phase_shift=target_phase_shift,
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
    ref_phase = patch_metalens.total_opt.objective.phase_shift[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["phase"].squeeze().mean().to(torch.float32)
    patch_metalens.set_ref_phase(ref_phase)
    plot_root = f"./figs/ONN_{model_comment}/"
    ensure_dir(plot_root)
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        total_phase = patch_metalens(sharpness=epoch + 120)
        # total_phase = patch_metalens(sharpness=256)
        # loss = torch.nn.functional.mse_loss(total_phase, target_phase_shift)
        loss = phase_matching_loss(total_phase, target_variants)
        loss.backward()
        # for name, param in patch_metalens.named_parameters():
        #     if param.grad is not None:
        #         print(name, torch.norm(param.grad))
        optimizer.step()
        scheduler.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")
        if epoch > num_epoch - 110:
            with torch.no_grad():
                _ = patch_metalens.total_opt(sharpness=256, ls_knots={"design_region_0": patch_metalens.level_set_knots[16:-16].unsqueeze(0)})
                full_phase = (ref_phase - patch_metalens.total_opt.objective.phase_shift[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["phase"].squeeze().to(torch.float32)) % (2 * torch.pi)
        else:
            full_phase = None
        plt.figure()
        plt.plot(target_phase_shift.detach().cpu().numpy(), label="target_phase")
        plt.plot(total_phase.detach().cpu().numpy(), label="stiched_phase")
        if full_phase is not None:
            plt.plot(full_phase.detach().cpu().numpy(), label="full_phase")
        plt.legend()
        plt.savefig(plot_root + f"stich_metasurfact_epoch_{epoch}.png")
        plt.close()
