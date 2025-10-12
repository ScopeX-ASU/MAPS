import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import csv
import math
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyutils.config import Config

from core.invdes import builder
from core.invdes.invdesign import InvDesign
from core.invdes.models import MetaLensOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MetaLens
from core.utils import SharpnessScheduler, print_stat, set_torch_deterministic

sys.path.pop(0)
import h5py
import torch
import torch.nn.functional as F
from pyutils.general import ensure_dir

from core.invdes.optimizer.nesterov import NesterovAcceleratedGradientOptimizer


def get_mid_weight(l, w, period=0.3):
    return (w * l) / (period - w)


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
        return F.interpolate(
            input_tensor, size=x1.shape[0], mode="linear", align_corners=False
        ).squeeze()
    elif method == "gaussian":
        sigma = 0.1
        dist_sq = (
            (x1.reshape(-1, 1) - x0.reshape(1, -1)).square().to(input_tensor.device)
        )
        weights = (-dist_sq / (2 * sigma**2)).exp()
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
    abs_diffs = torch.abs(
        target_phase_variants - total_phase.unsqueeze(0)
    )  # Broadcasting

    # Find the index of the closest match at each point
    closest_indices = torch.argmin(abs_diffs, dim=0)  # Shape (N,)

    # Gather the closest matching values
    closest_values = target_phase_variants[
        closest_indices, torch.arange(total_phase.shape[0])
    ]

    phase_error = closest_values - total_phase

    weight_map = (target_mag - torch.min(target_mag)) / (
        torch.max(target_mag) - torch.min(target_mag)
    )

    phase_normalized_L2 = torch.norm(phase_error * weight_map) / (
        torch.norm(closest_values) + 1e-12
    )

    # begin calculate the magnitude loss

    mag_error = target_mag - total_mag
    mag_normalized_L2 = torch.norm(mag_error) / (torch.norm(target_mag) + 1e-12)

    print(
        "this is the mag NL2norm: ",
        mag_normalized_L2.item(),
        "this is the phase NL2norm: ",
        phase_normalized_L2.item(),
        flush=True,
    )
    return (
        mag_normalized_L2 + phase_normalized_L2,
        round(mag_normalized_L2.item(), 3),
        round(phase_normalized_L2.item(), 3),
    )


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
        target_dx: float = 0.3,
        plot_root: str = "./figs/patched_metalens",
    ):
        super(PatchMetalens, self).__init__()
        self.atom_period = atom_period
        self.patch_size = patch_size
        self.num_atom = num_atom
        self.probing_region_size = probing_region_size
        self.target_phase_response = (
            target_phase_response  # this is used to initialize the metalens
        )
        self.LUT = LUT
        self.device = device
        self.target_dx = target_dx
        self.plot_root = plot_root
        self.build_param()
        self.build_patch()

    def build_param(self):
        self.pillar_ls_knots = nn.Parameter(
            -0.05 * torch.ones((self.num_atom), device=self.device)
        )
        if self.LUT is None:
            self.pillar_ls_knots.data = (
                self.pillar_ls_knots.data
                + 0.01 * torch.randn_like(self.pillar_ls_knots.data)
            )
        else:
            for i in range(self.num_atom):
                print(
                    f"this is the width for idx {i} for the phase shift {self.target_phase_response[i, i].item()}",
                    find_closest_width(
                        self.LUT, self.target_phase_response[i, i].item()
                    ),
                    flush=True,
                )
                self.pillar_ls_knots.data[i] = get_mid_weight(
                    0.05,
                    find_closest_width(
                        self.LUT, self.target_phase_response[i, i].item()
                    ),
                )

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
                border_width=[0, 0, 0.5, 0.5],
                PML=[0.5, 0.5],
                resolution=50,
                wl_cen=wl,
                plot_root=self.plot_root,
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
                plot_root=self.plot_root,
            )
        )
        self.opt = nn.ModuleDict()
        for patch_size in range(self.patch_size // 2 + 1, self.patch_size + 1):
            if patch_size > self.num_atom:
                break
            patch_metalens = MetaLens(
                material_bg="Air",
                material_r="Si",
                material_sub="SiO2",
                sim_cfg=sim_cfg,
                aperture=0.3 * patch_size,
                port_len=(1, 1),
                port_width=(0.3 * patch_size, 0.3),
                substrate_depth=0,
                ridge_height_max=0.75,
                nearfield_dx=0.3,
                nearfield_offset=0,  # only support y direction offset for now, so this is only a scalar, not a tuple
                nearfield_size=0.3 * min(self.probing_region_size, patch_size),
                farfield_dxs=((self.target_dx, self.target_dx + 1 / 50),),
                farfield_sizes=(self.atom_period * self.num_atom,),
                farfield_offset=0,
                device=self.device,
            )
            hr_patch_metalens = patch_metalens.copy(resolution=200)
            self.opt[str(patch_size)] = MetaLensOptimization(
                device=patch_metalens,
                hr_device=hr_patch_metalens,
                sim_cfg=sim_cfg,
                operation_device=self.device,
            ).to(self.device)
            # the key is from 9 (0+1+8) to 17 (8+1+8)
        total_metalens = MetaLens(
            material_bg="Air",
            material_r="Si",
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
        hr_total_metalens = total_metalens.copy(resolution=200)
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
                self.num_atom,
                self.num_atom,
            ),
            dtype=torch.complex128,
            device=self.device,
        )
        total_ls_knot = -0.05 * torch.ones(2 * (self.num_atom) + 1, device=self.device)
        for i in range(self.num_atom):
            center_knot_idx = 2 * i + 1
            # total_ls_knot[1::2] = self.pillar_ls_knots
            self.level_set_knots = total_ls_knot.clone()
            self.level_set_knots[1::2] = self.pillar_ls_knots
            if i >= self.patch_size // 2 and i < self.num_atom - self.patch_size // 2:
                knots_value = {
                    "design_region_0": self.level_set_knots[
                        center_knot_idx
                        - 2 * (self.patch_size // 2)
                        - 1 : center_knot_idx
                        + 2 * (self.patch_size // 2 + 1)
                    ].unsqueeze(0)
                }
                required_patch = self.patch_size
            elif i < self.patch_size // 2:
                knots_value = {
                    "design_region_0": self.level_set_knots[
                        : center_knot_idx + 2 * (self.patch_size // 2 + 1)
                    ].unsqueeze(0)
                }
                required_patch = min(i + self.patch_size // 2 + 1, self.num_atom)
            else:
                knots_value = {
                    "design_region_0": self.level_set_knots[
                        center_knot_idx - 2 * (self.patch_size // 2) - 1 :
                    ].unsqueeze(0)
                }
                required_patch = self.num_atom - i + self.patch_size // 2
            opt = self.opt[str(required_patch)]
            source = torch.zeros(required_patch, device=self.device)
            if i >= self.patch_size // 2 and i < self.num_atom - self.patch_size // 2:
                source[self.patch_size // 2] = 1
            elif i < self.patch_size // 2:
                source[i] = 1
            else:
                source[-(self.num_atom - i)] = 1
            source = source.repeat_interleave(int(self.atom_period * 50))
            source_zero_padding = torch.zeros(int(0.5 * 50), device=self.device)
            source = torch.cat([source_zero_padding, source, source_zero_padding])
            custom_source = dict(
                source=source,
                slice_name="in_slice_1",
                mode="Hz1",
                wl=0.85,
                direction="x+",
            )
            _ = opt(
                sharpness=sharpness, ls_knots=knots_value, custom_source=custom_source
            )
            near_field_response_fz = opt.objective.response[
                ("in_slice_1", "nearfield_1", 0.85, "Hz1", 300)
            ]["fz"].squeeze()
            near_field_response_fx = opt.objective.response[
                ("in_slice_1", "nearfield_1", 0.85, "Hz1", 300)
            ]["fx"].squeeze()
            near_field_response_fy = opt.objective.response[
                ("in_slice_1", "nearfield_1", 0.85, "Hz1", 300)
            ]["fy"].squeeze()

            # near 2 far projection

            response = response[
                int(self.atom_period * 50) // 2 :: int(self.atom_period * 50)
            ]
            assert len(response) == required_patch, f"{required_patch}!={len(response)}"
            row_idx = i
            if i >= self.patch_size // 2 and i < self.num_atom - self.patch_size // 2:
                col_start = row_idx - self.probing_region_size // 2
                col_end = col_start + self.probing_region_size
            elif i < self.patch_size // 2:
                col_start = 0
                col_end = required_patch
            else:
                col_end = self.num_atom
                col_start = col_end - required_patch

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
        total_response = total_response.transpose(0, 1)
        # plot the total_phase (1 D tensor)
        # plt.figure()
        # plt.plot(total_phase.detach().cpu().numpy())
        # plt.savefig("./figs/stiched_phase.png")
        # plt.close()
        return total_response / torch.max(total_response.abs())


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    set_torch_deterministic()
    # read LUT from csv file
    near_field_dx = 0.3
    atom_period = 0.3
    patch_size = 17
    num_atom = 32
    csv_file = f"./unitest/metaatom_phase_response_fsdx-{near_field_dx}.csv"
    LUT = {}
    with open(csv_file, mode="r") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if float(row[0]) > 0.14:
                break
            LUT[float(row[0])] = float(row[1])
    near2far_matrix = None
    # print(LUT)
    # quit()
    # read ONN weights from pt file
    # -----------------------------------------
    # checkpoint_file_name = "/home/pingchua/projects/MetaONN/checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_wb-16_ib-16_rotm-fixed_c-pac_test_3_acc-87.72_epoch-1.pt"
    # checkpoint = torch.load(checkpoint_file_name)
    # model_comment = checkpoint_file_name.split("_c-")[-1].split("_acc-")[0]
    # plot_root = f"./figs/ONN_{model_comment}_farfield/"
    # ensure_dir(plot_root)
    # state_dict = checkpoint["state_dict"]
    # # print("this is the state_dict keys: ", list(state_dict.keys()))
    # # quit()
    # W_1 = state_dict["features.conv1.conv._conv_pos.metalens.0_1.W_buffer"].squeeze().to(device)
    # W_3 = state_dict["features.conv1.conv._conv_pos.metalens.0_3.W_buffer"].squeeze().to(device)
    # W_5 = state_dict["features.conv1.conv._conv_pos.metalens.0_5.W_buffer"].squeeze().to(device)
    # W_7 = state_dict["features.conv1.conv._conv_pos.metalens.0_7.W_buffer"].squeeze().to(device)
    # W_13 = state_dict["features.conv1.conv._conv_pos.metalens.0_13.W_buffer"].squeeze().to(device)
    # target_response = state_dict["features.conv1.conv._conv_pos.W_buffer"].squeeze()[0].to(device).to(torch.complex128)
    # near2far_matrix = state_dict["features.conv1.conv._conv_pos.near2far_buffer"].squeeze()[0].to(device).to(torch.complex128)
    # target_response = target_response / torch.max(target_response.abs())
    # target_response_H = target_response.T.conj()
    # response_matmul = torch.matmul(target_response, target_response_H)

    # plt.figure()
    # plt.imshow(W_1.abs().cpu().numpy())
    # plt.savefig(f"./figs/ONN_{model_comment}/target_response_1.png")
    # plt.close()
    # plt.figure()
    # plt.imshow(W_3.abs().cpu().numpy())
    # plt.savefig(f"./figs/ONN_{model_comment}/target_response_3.png")
    # plt.close()
    # plt.figure()
    # plt.imshow(W_5.abs().cpu().numpy())
    # plt.savefig(f"./figs/ONN_{model_comment}/target_response_5.png")
    # plt.close()
    # plt.imshow(W_7.abs().cpu().numpy())
    # plt.savefig(f"./figs/ONN_{model_comment}/target_response_7.png")
    # plt.close()
    # plt.imshow(W_13.abs().cpu().numpy())
    # plt.savefig(f"./figs/ONN_{model_comment}/target_response_13.png")
    # plt.close()
    # plt.imshow(target_response.abs().cpu().numpy())
    # plt.savefig(f"./figs/ONN_{model_comment}/target_response.png")
    # plt.close()
    # plt.imshow(response_matmul.abs().cpu().numpy())
    # plt.savefig(f"./figs/ONN_{model_comment}/response_matmul.png")
    # plt.close()
    # quit()
    # -----------------------------------------
    # read transfer matrix from h5 file
    transfer_matrix_file = (
        f"figs/metalens_TF_fsdx-{near_field_dx}_wl-0.85_p-0.3_mat-Si/transfer_matrix.h5"
    )
    model_comment = transfer_matrix_file.split("_TF_")[-1].split("/")[0]
    plot_root = f"./figs/TF_fsdx-{near_field_dx}_ps-{patch_size}_{model_comment}/"
    ensure_dir(plot_root)
    with h5py.File(transfer_matrix_file, "r") as f:
        A = torch.tensor(f["transfer_matrix"], device=device)
    target_response = A
    target_response = target_response / torch.max(target_response.abs())
    target_response_H = A.T.conj()
    response_matmul = torch.matmul(target_response, target_response_H)
    # plt.figure()
    # plt.imshow(response_matmul.abs().cpu().numpy())
    # plt.colorbar()
    # plt.savefig(f"./figs/TF_{model_comment}/target_response.png")
    # plt.close()
    # quit()
    # -----------------------------------------
    target_phase_response_near = torch.angle(target_response)
    if near2far_matrix is not None:
        target_phase_response_far = torch.angle(
            torch.matmul(near2far_matrix, target_response)
        )
    else:
        target_phase_response_far = None

    # Create 3 variants of the target phase shift
    target_phase_variants = torch.stack(
        [
            (
                target_phase_response_near
                if target_phase_response_far is None
                else target_phase_response_far
            ),  # Original
            (
                (target_phase_response_near + 2 * torch.pi)
                if target_phase_response_far is None
                else (target_phase_response_far + 2 * torch.pi)
            ),
            (
                (target_phase_response_near - 2 * torch.pi)
                if target_phase_response_far is None
                else (target_phase_response_far - 2 * torch.pi)
            ),
        ],
        dim=0,
    )  # Shape (3, N)

    patch_metalens = PatchMetalens(
        atom_period=atom_period,
        patch_size=patch_size,
        num_atom=num_atom,
        probing_region_size=patch_size,
        target_phase_response=target_phase_response_near,
        LUT=LUT,
        device=device,
        target_dx=near_field_dx,
        plot_root=plot_root,
    )
    # Define the optimizer
    num_epoch = 100
    lr = 1e-3
    # lr = 5e-2
    optimizer = torch.optim.Adam(patch_metalens.parameters(), lr=lr)
    # optimizer = NesterovAcceleratedGradientOptimizer(
    #     [patch_metalens.pillar_ls_knots], lr=lr,
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=lr * 1e-2
    )

    sharp_scheduler = SharpnessScheduler(
        initial_sharp=10,
        final_sharp=256,
        total_steps=num_epoch,
    )

    # before we optimize the metalens, we need to simulate the ref phase:
    # ref_level_set_knots = -0.05 * torch.ones(2 * (patch_metalens.num_atom) + 1, device=device)
    # ref_level_set_knots = ref_level_set_knots + 0.001 * torch.randn_like(ref_level_set_knots)
    # _ = patch_metalens.total_opt(sharpness=256, ls_knots={"design_region_0": ref_level_set_knots.unsqueeze(0)})
    # ref_phase = patch_metalens.total_opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["phase"].squeeze().mean().to(torch.float32)
    # ref_mag = patch_metalens.total_opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["mag"].squeeze().mean().to(torch.float32)
    # patch_metalens.set_ref_response(ref_mag, ref_phase)
    class Closure(object):
        def __init__(
            self,
            optimizer,  # optimizer
            devOptimization,  # device optimization model,
        ):
            self.results = None
            self.stiched_response = None
            self.phase_loss = None
            self.mag_loss = None
            self.optimizer = optimizer
            self.devOptimization = devOptimization
            self.sharpness = 1

        def __call__(self):
            # clear grad here
            self.optimizer.zero_grad()

            stiched_response = self.devOptimization.forward(sharpness=self.sharpness)
            loss, mag_loss, phase_loss = response_matching_loss(
                torch.matmul(
                    (
                        near2far_matrix
                        if near2far_matrix is not None
                        else torch.eye(num_atom, device=device, dtype=torch.complex128)
                    ),
                    stiched_response,
                ),
                torch.matmul(
                    (
                        near2far_matrix
                        if near2far_matrix is not None
                        else torch.eye(num_atom, device=device, dtype=torch.complex128)
                    ),
                    target_response,
                ),
                target_phase_variants,
            )
            loss.backward()

            self.stiched_response = stiched_response
            self.phase_loss = phase_loss
            self.mag_loss = mag_loss
            self.results = loss

            return loss

    closure = Closure(
        optimizer=optimizer,
        devOptimization=patch_metalens,
    )

    sources = torch.eye(num_atom, device=device)
    for epoch in range(num_epoch):
        sharpness = sharp_scheduler.get_sharpness()
        closure.sharpness = sharpness
        optimizer.step(closure)
        stiched_response = closure.stiched_response
        phase_loss = closure.phase_loss
        mag_loss = closure.mag_loss
        loss = closure.results

        # stiched_response = patch_metalens(sharpness=sharpness)
        # loss, mag_loss, phase_loss = response_matching_loss(
        #     torch.matmul(
        #         near2far_matrix if near2far_matrix is not None else torch.eye(num_atom, device=device, dtype=torch.complex128),
        #         stiched_response,
        #     ),
        #     torch.matmul(
        #         near2far_matrix if near2far_matrix is not None else torch.eye(num_atom, device=device, dtype=torch.complex128),
        #         target_response,
        #     ),
        #     target_phase_variants,
        # )
        # loss.backward()
        # for name, param in patch_metalens.named_parameters():
        #     if param.grad is not None:
        #         print(name, torch.norm(param.grad))
        # quit()
        scheduler.step()
        sharp_scheduler.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")
        if epoch > num_epoch - 110:
            with torch.no_grad():
                full_wave_response = torch.zeros(
                    (num_atom, num_atom), device=device, dtype=torch.complex128
                )
                for idx in range(num_atom):
                    source_i = sources[idx].repeat_interleave(int(atom_period * 50))
                    source_zero_padding = torch.zeros(int(0.5 * 50), device=device)
                    source_i = torch.cat(
                        [source_zero_padding, source_i, source_zero_padding]
                    )
                    custom_source = dict(
                        source=source_i,
                        slice_name="in_slice_1",
                        mode="Hz1",
                        wl=0.85,
                        direction="x+",
                    )
                    _ = patch_metalens.total_opt(
                        sharpness=256,
                        ls_knots={
                            "design_region_0": patch_metalens.level_set_knots.unsqueeze(
                                0
                            )
                        },
                        custom_source=custom_source,
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
                    response = patch_metalens.total_opt.objective.response[
                        ("in_slice_1", "nearfield_1", 0.85, "Hz1", 300)
                    ]["fz"].squeeze()
                    response = response[
                        int(atom_period * 50) // 2 :: int(atom_period * 50)
                    ]
                    assert len(response) == num_atom, f"{num_atom}!={len(response)}"
                    full_wave_response[idx] = response
                full_wave_response = full_wave_response.transpose(0, 1)
                full_wave_response = full_wave_response / torch.max(
                    full_wave_response.abs()
                )
        else:
            full_phase = None
            full_mag = None

        figure, ax = plt.subplots(1, 4, figsize=(20, 5))
        im0 = ax[0].imshow(
            torch.matmul(
                (
                    near2far_matrix
                    if near2far_matrix is not None
                    else torch.eye(num_atom, device=device, dtype=torch.complex128)
                ),
                target_response,
            )
            .abs()
            .cpu()
            .numpy()
        )
        ax[0].set_title("Target Magnitude")
        figure.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(
            torch.matmul(
                (
                    near2far_matrix
                    if near2far_matrix is not None
                    else torch.eye(num_atom, device=device, dtype=torch.complex128)
                ),
                stiched_response,
            )
            .detach()
            .abs()
            .cpu()
            .numpy()
        )
        ax[1].set_title("Stitched Magnitude")
        figure.colorbar(im1, ax=ax[1])
        im2 = ax[2].imshow(
            torch.matmul(
                (
                    near2far_matrix
                    if near2far_matrix is not None
                    else torch.eye(num_atom, device=device, dtype=torch.complex128)
                ),
                full_wave_response,
            )
            .abs()
            .cpu()
            .numpy()
        )
        ax[2].set_title("Full Magnitude")
        figure.colorbar(im2, ax=ax[2])
        im3 = ax[3].imshow(
            torch.matmul(
                (
                    near2far_matrix
                    if near2far_matrix is not None
                    else torch.eye(num_atom, device=device, dtype=torch.complex128)
                ),
                full_wave_response - target_response,
            )
            .abs()
            .cpu()
            .numpy()
        )
        ax[3].set_title("Difference Magnitude")
        figure.colorbar(im3, ax=ax[3])
        plt.savefig(
            plot_root + f"epoch_{epoch}_magNL2-{mag_loss}_phaseNL2-{phase_loss}.png"
        )
        plt.close()
