"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-05-10 20:34:02
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-12-26 00:11:01
"""
#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Callable, Dict, Iterable
import torch.amp as amp
# import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter, logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler
import torch.fft
from core import builder
from core.utils import DeterministicCtx
import wandb
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
from core.utils import print_stat, plot_fields

def single_batch_check(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Criterion,
    aux_criterions: Dict,
    epoch: int = 0,
    mixup_fn: Callable = None,
    device: torch.device = torch.device("cuda:0"),
    grad_scaler=None,
) -> None:
    model.train()
    step = epoch * len(train_loader)

    mse_meter = AverageMeter("mse")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
    aux_output_weight = getattr(configs.criterion, "aux_output_weight", 0)

    # poynting_loss = PoyntingLoss(configs.model.grid_step, wavelength=1.55)
    data_counter = 0
    total_data = len(train_loader.dataset)
    rand_idx = len(train_loader.dataset) // train_loader.batch_size - 1
    rand_idx = random.randint(0, rand_idx)
    for batch_idx, (eps_map, adj_srcs, gradient, field_solutions, s_params, src_profiles, fields_adj, field_normalizer, design_region_mask, incident_field, ht_m, et_m, monitor_slices, As) in enumerate(train_loader):
        eps_map = eps_map.to(device, non_blocking=True)
        gradient = gradient.to(device, non_blocking=True)
        for key, field in field_solutions.items():
            field = torch.view_as_real(field).permute(0, 1, 4, 2, 3)
            field = field.flatten(1, 2)
            field_solutions[key] = field.to(device, non_blocking=True)
        for key, s_param in s_params.items():
            s_params[key] = s_param.to(device, non_blocking=True)
        for key, adj_src in adj_srcs.items():
            adj_srcs[key] = adj_src.to(device, non_blocking=True)
        for key, src_profile in src_profiles.items():
            src_profiles[key] = src_profile.to(device, non_blocking=True)
        for key, field_adj in fields_adj.items():
            field_adj = torch.view_as_real(field_adj).permute(0, 1, 4, 2, 3)
            field_adj = field_adj.flatten(1, 2)
            fields_adj[key] = field_adj.to(device, non_blocking=True)
        for key, field_norm in field_normalizer.items():
            field_normalizer[key] = field_norm.to(device, non_blocking=True)
        for key, field in incident_field.items():
            incident_field[key] = field.to(device, non_blocking=True)
        for key, monitor_slice in monitor_slices.items():
            monitor_slices[key] = monitor_slice.to(device, non_blocking=True)
        # for key, design_region in design_region_mask.items():
        #     design_region_mask[key] = design_region.to(device, non_blocking=True)
        for key, ht in ht_m.items():
            if key.endswith("-origin_size"):
                continue
            else:
                size = ht_m[key + "-origin_size"]
                ht_list = []
                for i in range(size.shape[0]):
                    item_to_add = torch.view_as_real(ht[i]).permute(1, 0).unsqueeze(0)
                    item_to_add = F.interpolate(item_to_add, size=size[i].item(), mode='linear', align_corners=True)
                    item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                    ht_list.append(torch.view_as_complex(item_to_add).to(device, non_blocking=True))
                ht_m[key] = ht_list
        for key, et in et_m.items():
            if key.endswith("-origin_size"):
                continue
            else:
                size = et_m[key + "-origin_size"]
                et_list = []
                for i in range(size.shape[0]):
                    item_to_add = torch.view_as_real(et[i]).permute(1, 0).unsqueeze(0)
                    item_to_add = F.interpolate(item_to_add, size=size[i].item(), mode='linear', align_corners=True)
                    item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                    et_list.append(torch.view_as_complex(item_to_add).to(device, non_blocking=True))
                et_m[key] = et_list
        for key, A in As.items():
            As[key] = A.to(device, non_blocking=True)

        data_counter += eps_map.shape[0]

        if mixup_fn is not None:
            eps_map, adj_src, gradient, field_solutions, s_params = mixup_fn(eps_map, adj_src, gradient, field_solutions, s_params)
        if batch_idx == rand_idx:
            break

    for iter in range(10000):
        with amp.autocast('cuda', enabled=False):
            output = model( # now only suppose that the output is the gradient of the field
                eps_map, 
                src_profiles,
                adj_srcs,
                incident_field, 
            )
            if type(output) == tuple:
                output, aux_output = output
            else:
                aux_output = None
            regression_loss = criterion(output[:, -2:, ...], field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...], torch.ones_like(output[:, -2:, ...]).to(device))
            # regression_loss = regression_loss + criterion(aux_output, fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"], torch.ones_like(aux_output).to(device))
            mse_meter.update(regression_loss.item())
            loss = regression_loss
            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                if name == "maxwell_residual":
                    # TODO, this is incorrect now
                    aux_loss = weight * aux_criterion(output, gradient)
                loss = loss + aux_loss
                aux_meters[name].update(aux_loss.item())
                
            if aux_output is not None and aux_output_weight > 0:
                # TODO, this is incorrect now
                aux_output_loss = aux_output_weight * F.mse_loss(
                    aux_output, field_solutions
                )  # field magnitude learning
                loss = loss + aux_output_loss
            else:
                aux_output_loss = None

        grad_scaler.scale(loss).backward()

        grad_scaler.unscale_(optimizer)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()

        step += 1

        if iter % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} Regression Loss: {:.4e}".format(
                epoch,
                data_counter,
                total_data,
                100.0 * data_counter / total_data,
                loss.data.item(),
                regression_loss.data.item(),
            )
            for name, aux_meter in aux_meters.items():
                log += f" {name}: {aux_meter.val:.4e}"
            if aux_output_loss is not None:
                log += f" aux_output_loss: {aux_output_loss.item()}"
            lg.info(log)

            # mlflow.log_metrics({"train_loss": loss.item()}, step=step)
            wandb.log(
                {
                    "train_running_loss": loss.item(),
                    "global_step": step,
                },
            )
        if iter % 20 == 0:
            dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
            os.makedirs(dir_path, exist_ok=True)
            filepath = os.path.join(dir_path, f"epoch_{epoch}_sbc.png")
            # plot_fields(
            #     fields=output.clone().detach(),
            #     ground_truth=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
            #     filepath=filepath,
            # )
            # plot_fouier_transform(
            #     field=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
            #     filepath=filepath.replace(".png", "_fft.png"),
            # )
            # quit()
    return None

def plot_fourier_eps(
    eps_map: torch.Tensor,
    filepath: str,
) -> None:
    eps_map = 1 / eps_map[0]
    eps_map0 = eps_map.cpu().numpy()
    plt.imshow(eps_map0)
    plt.colorbar()
    plt.savefig(filepath.replace(".png", "_org_eps.png"))
    eps_map = torch.fft.fft2(eps_map)           # 2D FFT
    eps_map = torch.fft.fftshift(eps_map)       # Shift zero frequency to center
    eps_map = torch.abs(eps_map) 
    print_stat(eps_map)
    eps_map = eps_map.cpu().numpy()
    plt.imshow(eps_map)
    plt.colorbar()
    plt.savefig(filepath)
    plt.close()

def plot_fouier_transform(
    field: torch.Tensor,
    filepath: str,
) -> None:
    field = field.reshape(field.shape[0], -1, 2, field.shape[-2], field.shape[-1]).permute(0, 1, 3, 4, 2).contiguous()
    field = torch.abs(torch.view_as_complex(field)).squeeze()
    field = field[0]
    field0 = field.cpu().numpy()
    plt.imshow(field0)
    plt.savefig(filepath.replace(".png", "_org_field.png"))
    field = torch.fft.fft2(field)           # 2D FFT
    field = torch.fft.fftshift(field)       # Shift zero frequency to center
    field = torch.abs(field) 
    field = field.cpu().numpy()
    plt.imshow(field)
    plt.savefig(filepath)
    plt.close()

def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Dict,
    mixup_fn: Callable = None,
    device: torch.device = torch.device("cuda:0"),
    plot: bool = False,
    grad_scaler=None,
) -> None:
    torch.autograd.set_detect_anomaly(True)
    model.train()
    step = epoch * len(train_loader)

    mse_meter = AverageMeter("mse")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}

    data_counter = 0
    total_data = len(train_loader.dataset)
    for batch_idx, (eps_map, adj_srcs, gradient, field_solutions, s_params, src_profiles, fields_adj, field_normalizer, design_region_mask, incident_field, ht_m, et_m, monitor_slices, As) in enumerate(train_loader):
        eps_map = eps_map.to(device, non_blocking=True)
        gradient = gradient.to(device, non_blocking=True)
        for key, field in field_solutions.items():
            field = torch.view_as_real(field).permute(0, 1, 4, 2, 3)
            field = field.flatten(1, 2)
            field_solutions[key] = field.to(device, non_blocking=True)
        for key, s_param in s_params.items():
            s_params[key] = s_param.to(device, non_blocking=True)
        for key, adj_src in adj_srcs.items():
            adj_srcs[key] = adj_src.to(device, non_blocking=True)
        for key, src_profile in src_profiles.items():
            src_profiles[key] = src_profile.to(device, non_blocking=True)
        for key, field_adj in fields_adj.items():
            field_adj = torch.view_as_real(field_adj).permute(0, 1, 4, 2, 3)
            field_adj = field_adj.flatten(1, 2)
            fields_adj[key] = field_adj.to(device, non_blocking=True)
        for key, field_norm in field_normalizer.items():
            field_normalizer[key] = field_norm.to(device, non_blocking=True)
        for key, field in incident_field.items():
            incident_field[key] = field.to(device, non_blocking=True)
        for key, monitor_slice in monitor_slices.items():
            monitor_slices[key] = monitor_slice.to(device, non_blocking=True)
        # for key, design_region in design_region_mask.items():
        #     design_region_mask[key] = design_region.to(device, non_blocking=True)
        for key, ht in ht_m.items():
            if key.endswith("-origin_size"):
                continue
            else:
                size = ht_m[key + "-origin_size"]
                ht_list = []
                for i in range(size.shape[0]):
                    item_to_add = torch.view_as_real(ht[i]).permute(1, 0).unsqueeze(0)
                    item_to_add = F.interpolate(item_to_add, size=size[i].item(), mode='linear', align_corners=True)
                    item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                    ht_list.append(torch.view_as_complex(item_to_add).to(device, non_blocking=True))
                ht_m[key] = ht_list
        for key, et in et_m.items():
            if key.endswith("-origin_size"):
                continue
            else:
                size = et_m[key + "-origin_size"]
                et_list = []
                for i in range(size.shape[0]):
                    item_to_add = torch.view_as_real(et[i]).permute(1, 0).unsqueeze(0)
                    item_to_add = F.interpolate(item_to_add, size=size[i].item(), mode='linear', align_corners=True)
                    item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                    et_list.append(torch.view_as_complex(item_to_add).to(device, non_blocking=True))
                et_m[key] = et_list
        for key, A in As.items():
            As[key] = A.to(device, non_blocking=True)

        data_counter += eps_map.shape[0]
        if mixup_fn is not None:
            eps_map, adj_src, gradient, field_solutions, s_params = mixup_fn(eps_map, adj_src, gradient, field_solutions, s_params)

        with amp.autocast('cuda', enabled=grad_scaler._enabled):
            output = model( # now only suppose that the output is the gradient of the field
                eps_map, 
                src_profiles,
                adj_srcs,
                incident_field, 
            )
            if type(output) == tuple and len(output) == 2:
                output, aux_output = output
                output_correction = None
            elif type(output) == tuple and len(output) == 3:
                output, aux_output, output_correction = output
            else:
                aux_output = None
                output_correction = None
            regression_loss = criterion(
                output[:, -2:, ...], 
                field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...], 
                torch.ones_like(output[:, -2:, ...]).to(device)
            )
            mse_meter.update(regression_loss.item())
            loss = regression_loss
            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                if name == "maxwell_residual_loss":
                    aux_loss = weight * aux_criterion(
                        Ez=output, 
                        # Ez=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                        eps_r=eps_map, 
                        source=src_profiles["source_profile-wl-1.55-port-in_port_1-mode-1"], 
                        target_size=eps_map.shape[-2:],
                        As=As,
                    )
                elif name == "grad_loss":
                    aux_loss = weight * aux_criterion(
                        forward_fields=output,
                        # forward_fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                        backward_fields=field_solutions["field_solutions-wl-1.55-port-out_port_1-mode-1"][:, -2:, ...],
                        adjoint_fields=aux_output,  
                        # adjoint_fields=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                        backward_adjoint_fields = fields_adj['fields_adj-wl-1.55-port-in_port_1-mode-1'][:, -2:, ...],
                        target_gradient=gradient,
                        gradient_multiplier=field_normalizer,
                        # dr_mask=None,
                        dr_mask=design_region_mask,
                    )
                elif name == "s_param_loss":
                    aux_loss = weight * aux_criterion(
                        fields=output, 
                        # fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                        ht_m=ht_m['ht_m-wl-1.55-port-out_port_1-mode-1'],
                        et_m=et_m['et_m-wl-1.55-port-out_port_1-mode-1'],
                        monitor_slices=monitor_slices, # 'port_slice-out_port_1_x', 'port_slice-out_port_1_y'
                        target_SParam=s_params['s_params-fwd_trans-1.55-1'],
                    )
                elif name == "err_corr_Ez":
                    assert model.err_correction
                    aux_loss = weight * aux_criterion(
                        output_correction[:, -2:, ...], 
                        field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...], 
                        torch.ones_like(output[:, -2:, ...]).to(device)
                    )
                elif name == "err_corr_Hx":
                    assert model.err_correction
                    aux_loss = weight * aux_criterion(
                        output_correction[:, :2, ...],
                        field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, :2, ...],
                        torch.ones_like(output[:, :2, ...]).to(device)
                    )
                elif name == "err_corr_Hy":
                    assert model.err_correction
                    aux_loss = weight * aux_criterion(
                        output_correction[:, 2:4, ...],
                        field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, 2:4, ...],
                        torch.ones_like(output[:, 2:4, ...]).to(device)
                    )
                elif name == "Hx_loss":
                    aux_loss = weight * aux_criterion(
                        output[:, :2, ...], 
                        field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, :2, ...], 
                        torch.ones_like(output[:, :2, ...]).to(device)
                    )
                elif name == "Hy_loss":
                    aux_loss = weight * aux_criterion(
                        output[:, 2:4, ...], 
                        field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, 2:4, ...], 
                        torch.ones_like(output[:, 2:4, ...]).to(device)
                    )
                loss = loss + aux_loss
                aux_meters[name].update(aux_loss.item())
                
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()

        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} Regression Loss: {:.4e}".format(
                epoch,
                data_counter,
                total_data,
                100.0 * data_counter / total_data,
                loss.data.item(),
                regression_loss.data.item(),
            )
            for name, aux_meter in aux_meters.items():
                log += f" {name}: {aux_meter.val:.4e}"
            lg.info(log)

            wandb.log(
                {
                    "train_running_loss": loss.item(),
                    "global_step": step,
                },
            )

    scheduler.step()
    avg_regression_loss = mse_meter.avg
    lg.info(f"Train Regression Loss: {avg_regression_loss:.4e}")

    wandb.log(
        {
            "train_loss": avg_regression_loss,
            "epoch": epoch,
            "lr": get_learning_rate(optimizer),
        },
    )

    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_train.png")
        plot_fields(
            fields=output.clone().detach(),
            ground_truth=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
            filepath=filepath,
        )
        if output_correction is not None:
            filepath = os.path.join(dir_path, f"epoch_{epoch}_train_corr.png")
            plot_fields(
                fields=output_correction.clone().detach(),
                ground_truth=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath,
            )

def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    log_criterions: Dict,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = True,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    log_meters = {name: AverageMeter(name) for name in log_criterions}
    with torch.no_grad(), DeterministicCtx(42):
        for batch_idx, (eps_map, adj_srcs, gradient, field_solutions, s_params, src_profiles, fields_adj, field_normalizer, design_region_mask, incident_field, ht_m, et_m, monitor_slices, As) in enumerate(validation_loader):
            eps_map = eps_map.to(device, non_blocking=True)
            gradient = gradient.to(device, non_blocking=True)
            for key, field in field_solutions.items():
                field = torch.view_as_real(field).permute(0, 1, 4, 2, 3)
                field = field.flatten(1, 2)
                field_solutions[key] = field.to(device, non_blocking=True)
            for key, s_param in s_params.items():
                s_params[key] = s_param.to(device, non_blocking=True)
            for key, adj_src in adj_srcs.items():
                adj_srcs[key] = adj_src.to(device, non_blocking=True)
            for key, src_profile in src_profiles.items():
                src_profiles[key] = src_profile.to(device, non_blocking=True)
            for key, field_adj in fields_adj.items():
                field_adj = torch.view_as_real(field_adj).permute(0, 1, 4, 2, 3)
                field_adj = field_adj.flatten(1, 2)
                fields_adj[key] = field_adj.to(device, non_blocking=True)
            for key, field_norm in field_normalizer.items():
                field_normalizer[key] = field_norm.to(device, non_blocking=True)
            for key, field in incident_field.items():
                incident_field[key] = field.to(device, non_blocking=True)
            for key, monitor_slice in monitor_slices.items():
                monitor_slices[key] = monitor_slice.to(device, non_blocking=True)
            # for key, design_region in design_region_mask.items():
            #     design_region_mask[key] = design_region.to(device, non_blocking=True)
            for key, ht in ht_m.items():
                if key.endswith("-origin_size"):
                    continue
                else:
                    size = ht_m[key + "-origin_size"]
                    ht_list = []
                    for i in range(size.shape[0]):
                        item_to_add = torch.view_as_real(ht[i]).permute(1, 0).unsqueeze(0)
                        item_to_add = F.interpolate(item_to_add, size=size[i].item(), mode='linear', align_corners=True)
                        item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                        ht_list.append(torch.view_as_complex(item_to_add).to(device, non_blocking=True))
                    ht_m[key] = ht_list
            for key, et in et_m.items():
                if key.endswith("-origin_size"):
                    continue
                else:
                    size = et_m[key + "-origin_size"]
                    et_list = []
                    for i in range(size.shape[0]):
                        item_to_add = torch.view_as_real(et[i]).permute(1, 0).unsqueeze(0)
                        item_to_add = F.interpolate(item_to_add, size=size[i].item(), mode='linear', align_corners=True)
                        item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                        et_list.append(torch.view_as_complex(item_to_add).to(device, non_blocking=True))
                    et_m[key] = et_list
            for key, A in As.items():
                As[key] = A.to(device, non_blocking=True)

            if mixup_fn is not None:
                eps_map, adj_src, gradient, field_solutions, s_params = mixup_fn(eps_map, adj_src, gradient, field_solutions, s_params)

            with amp.autocast('cuda', enabled=False):
                output = model( # now only suppose that the output is the gradient of the field
                    eps_map, 
                    src_profiles,
                    adj_srcs, 
                    incident_field, 
                )
                if type(output) == tuple and len(output) == 2:
                    output, aux_output = output
                    output_correction = None
                elif type(output) == tuple and len(output) == 3:
                    output, aux_output, output_correction = output
                else:
                    aux_output = None
                    output_correction = None
                # TODO not sure how to distinguish the test criterion if test is all field while train is the Ez
                val_loss = criterion(
                    output[:, -2:, ...] if output_correction is None else output_correction[:, -2:, ...],
                    field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...], 
                    torch.ones_like(output[:, -2:, ...]).to(device)
                )
                mse_meter.update(val_loss.item())
                loss = val_loss

                for name, config in log_criterions.items():
                    log_criterion, _ = config
                    if name == "maxwell_residual_loss":
                        log_loss = log_criterion(
                            Ez=output, 
                            # Ez=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                            eps_r=eps_map, 
                            source=src_profiles["source_profile-wl-1.55-port-in_port_1-mode-1"], 
                            target_size=eps_map.shape[-2:],
                            As=As,
                        )
                    elif name == "grad_loss":
                        log_loss = log_criterion(
                            forward_fields=output,
                            # forward_fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                            backward_fields=field_solutions["field_solutions-wl-1.55-port-out_port_1-mode-1"][:, -2:, ...],
                            adjoint_fields=aux_output,  
                            # adjoint_fields=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                            backward_adjoint_fields = fields_adj['fields_adj-wl-1.55-port-in_port_1-mode-1'][:, -2:, ...],
                            target_gradient=gradient,
                            gradient_multiplier=field_normalizer,
                            # dr_mask=None,
                            dr_mask=design_region_mask,
                        )
                    elif name == "s_param_loss":
                        log_loss = log_criterion(
                            fields=output, 
                            # fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                            ht_m=ht_m['ht_m-wl-1.55-port-out_port_1-mode-1'],
                            et_m=et_m['et_m-wl-1.55-port-out_port_1-mode-1'],
                            monitor_slices=monitor_slices, # 'port_slice-out_port_1_x', 'port_slice-out_port_1_y'
                            target_SParam=s_params['s_params-fwd_trans-1.55-1'],
                        )
                    elif name == "err_corr_Ez":
                        assert model.err_correction
                        log_loss = log_criterion(
                            output_correction[:, -2:, ...], 
                            field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...], 
                            torch.ones_like(output[:, -2:, ...]).to(device)
                        )
                    elif name == "err_corr_Hx":
                        assert model.err_correction
                        log_loss = log_criterion(
                            output_correction[:, :2, ...],
                            field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, :2, ...],
                            torch.ones_like(output[:, :2, ...]).to(device)
                        )
                    elif name == "err_corr_Hy":
                        assert model.err_correction
                        log_loss = log_criterion(
                            output_correction[:, 2:4, ...],
                            field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, 2:4, ...],
                            torch.ones_like(output[:, 2:4, ...]).to(device)
                        )
                    elif name == "Hx_loss":
                        log_loss = log_criterion(
                            output[:, :2, ...], 
                            field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, :2, ...], 
                            torch.ones_like(output[:, :2, ...]).to(device)
                        )
                    elif name == "Hy_loss":
                        log_loss = log_criterion(
                            output[:, 2:4, ...], 
                            field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, 2:4, ...], 
                            torch.ones_like(output[:, 2:4, ...]).to(device)
                        )
                    loss = loss + log_loss
                    log_meters[name].update(log_loss.item())
    if "err_corr_Ez" in log_criterions.keys():
        loss_to_append = log_meters["err_corr_Ez"].avg
        if "err_corr_Hx" in log_criterions.keys():
            assert "err_corr_Hy" in log_criterions.keys(), "H field loss must appear together"
            loss_to_append += log_meters["err_corr_Hx"].avg + log_meters["err_corr_Hy"].avg
    elif "Hy_loss" in log_criterions.keys():
        assert "Hx_loss" in log_criterions.keys(), "H field loss must appear together"
        loss_to_append = log_meters["Hx_loss"].avg + log_meters["Hy_loss"].avg + mse_meter.avg
    else:
        loss_to_append = mse_meter.avg
    loss_vector.append(loss_to_append)

    log_info = "\nValidation set: Average loss: {:.4e}".format(mse_meter.avg)
    for name, log_meter in log_meters.items():
        log_info += f" {name}: {log_meter.val:.4e}"

    lg.info(log_info)
    wandb.log(
        {
            "val_loss": loss_to_append,
            "epoch": epoch,
        },
    )

    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_valid.png")
        plot_fields(
            fields=output.clone().detach(),
            ground_truth=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
            filepath=filepath,
        )
        if output_correction is not None:
            filepath = os.path.join(dir_path, f"epoch_{epoch}_valid_corr.png")
            plot_fields(
                fields=output_correction.clone().detach(),
                ground_truth=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath,
            )


def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    log_criterions: Dict,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    log_meters = {name: AverageMeter(name) for name in log_criterions}
    with torch.no_grad(), DeterministicCtx(42):
        for batch_idx, (eps_map, adj_srcs, gradient, field_solutions, s_params, src_profiles, fields_adj, field_normalizer, design_region_mask, incident_field, ht_m, et_m, monitor_slices, As) in enumerate(test_loader):
            eps_map = eps_map.to(device, non_blocking=True)
            gradient = gradient.to(device, non_blocking=True)
            for key, field in field_solutions.items():
                field = torch.view_as_real(field).permute(0, 1, 4, 2, 3)
                field = field.flatten(1, 2)
                field_solutions[key] = field.to(device, non_blocking=True)
            for key, s_param in s_params.items():
                s_params[key] = s_param.to(device, non_blocking=True)
            for key, adj_src in adj_srcs.items():
                adj_srcs[key] = adj_src.to(device, non_blocking=True)
            for key, src_profile in src_profiles.items():
                src_profiles[key] = src_profile.to(device, non_blocking=True)
            for key, field_adj in fields_adj.items():
                field_adj = torch.view_as_real(field_adj).permute(0, 1, 4, 2, 3)
                field_adj = field_adj.flatten(1, 2)
                fields_adj[key] = field_adj.to(device, non_blocking=True)
            for key, field_norm in field_normalizer.items():
                field_normalizer[key] = field_norm.to(device, non_blocking=True)
            for key, field in incident_field.items():
                incident_field[key] = field.to(device, non_blocking=True)
            for key, monitor_slice in monitor_slices.items():
                monitor_slices[key] = monitor_slice.to(device, non_blocking=True)
            # for key, design_region in design_region_mask.items():
            #     design_region_mask[key] = design_region.to(device, non_blocking=True)
            for key, ht in ht_m.items():
                if key.endswith("-origin_size"):
                    continue
                else:
                    size = ht_m[key + "-origin_size"]
                    ht_list = []
                    for i in range(size.shape[0]):
                        item_to_add = torch.view_as_real(ht[i]).permute(1, 0).unsqueeze(0)
                        item_to_add = F.interpolate(item_to_add, size=size[i].item(), mode='linear', align_corners=True)
                        item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                        ht_list.append(torch.view_as_complex(item_to_add).to(device, non_blocking=True))
                    ht_m[key] = ht_list
            for key, et in et_m.items():
                if key.endswith("-origin_size"):
                    continue
                else:
                    size = et_m[key + "-origin_size"]
                    et_list = []
                    for i in range(size.shape[0]):
                        item_to_add = torch.view_as_real(et[i]).permute(1, 0).unsqueeze(0)
                        item_to_add = F.interpolate(item_to_add, size=size[i].item(), mode='linear', align_corners=True)
                        item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                        et_list.append(torch.view_as_complex(item_to_add).to(device, non_blocking=True))
                    et_m[key] = et_list
            for key, A in As.items():
                As[key] = A.to(device, non_blocking=True)

            if mixup_fn is not None:
                eps_map, adj_src, gradient, field_solutions, s_params = mixup_fn(eps_map, adj_src, gradient, field_solutions, s_params)

            with amp.autocast('cuda', enabled=False):
                output = model( # now only suppose that the output is the gradient of the field
                    eps_map, 
                    src_profiles,
                    adj_srcs, 
                    incident_field, 
                )
                if type(output) == tuple and len(output) == 2:
                    output, aux_output = output
                    output_correction = None
                elif type(output) == tuple and len(output) == 3:
                    output, aux_output, output_correction = output
                else:
                    aux_output = None
                    output_correction = None
                # TODO not sure how to distinguish the test criterion if test is all field while train is the Ez
                val_loss = criterion(
                    output[:, -2:, ...] if output_correction is None else output_correction[:, -2:, ...],
                    field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...], 
                    torch.ones_like(output[:, -2:, ...]).to(device)
                )
                mse_meter.update(val_loss.item())
                loss = val_loss
                for name, config in log_criterions.items():
                    log_criterion, _ = config
                    if name == "maxwell_residual_loss":
                        log_loss = log_criterion(
                            Ez=output, 
                            # Ez=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                            eps_r=eps_map, 
                            source=src_profiles["source_profile-wl-1.55-port-in_port_1-mode-1"], 
                            target_size=eps_map.shape[-2:],
                            As=As,
                        )
                    elif name == "grad_loss":
                        log_loss = log_criterion(
                            forward_fields=output,
                            # forward_fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                            backward_fields=field_solutions["field_solutions-wl-1.55-port-out_port_1-mode-1"][:, -2:, ...],
                            adjoint_fields=aux_output,  
                            # adjoint_fields=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                            backward_adjoint_fields = fields_adj['fields_adj-wl-1.55-port-in_port_1-mode-1'][:, -2:, ...],
                            target_gradient=gradient,
                            gradient_multiplier=field_normalizer,
                            # dr_mask=None,
                            dr_mask=design_region_mask,
                        )
                    elif name == "s_param_loss":
                        log_loss = log_criterion(
                            fields=output, 
                            # fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                            ht_m=ht_m['ht_m-wl-1.55-port-out_port_1-mode-1'],
                            et_m=et_m['et_m-wl-1.55-port-out_port_1-mode-1'],
                            monitor_slices=monitor_slices, # 'port_slice-out_port_1_x', 'port_slice-out_port_1_y'
                            target_SParam=s_params['s_params-fwd_trans-1.55-1'],
                        )
                    elif name == "err_corr_Ez":
                        assert model.err_correction
                        log_loss = log_criterion(
                            output_correction[:, -2:, ...], 
                            field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...], 
                            torch.ones_like(output[:, -2:, ...]).to(device)
                        )
                    elif name == "err_corr_Hx":
                        assert model.err_correction
                        log_loss = log_criterion(
                            output_correction[:, :2, ...],
                            field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, :2, ...],
                            torch.ones_like(output[:, :2, ...]).to(device)
                        )
                    elif name == "err_corr_Hy":
                        assert model.err_correction
                        log_loss = log_criterion(
                            output_correction[:, 2:4, ...],
                            field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, 2:4, ...],
                            torch.ones_like(output[:, 2:4, ...]).to(device)
                        )
                    elif name == "Hx_loss":
                        log_loss = log_criterion(
                            output[:, :2, ...], 
                            field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, :2, ...], 
                            torch.ones_like(output[:, :2, ...]).to(device)
                        )
                    elif name == "Hy_loss":
                        log_loss = log_criterion(
                            output[:, 2:4, ...], 
                            field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, 2:4, ...], 
                            torch.ones_like(output[:, 2:4, ...]).to(device)
                        )
                    loss = loss + log_loss
                    log_meters[name].update(log_loss.item())
    if "err_corr_Ez" in log_criterions.keys():
        loss_to_append = log_meters["err_corr_Ez"].avg
        if "err_corr_Hx" in log_criterions.keys():
            assert "err_corr_Hy" in log_criterions.keys(), "H field loss must appear together"
            loss_to_append += log_meters["err_corr_Hx"].avg + log_meters["err_corr_Hy"].avg
    elif "Hy_loss" in log_criterions.keys():
        assert "Hx_loss" in log_criterions.keys(), "H field loss must appear together"
        loss_to_append = log_meters["Hx_loss"].avg + log_meters["Hy_loss"].avg + mse_meter.avg
    else:
        loss_to_append = mse_meter.avg
    loss_vector.append(loss_to_append)

    log_info = "\nTest set: Average loss: {:.4e}".format(mse_meter.avg)
    for name, log_meter in log_meters.items():
        log_info += f" {name}: {log_meter.val:.4e}"

    lg.info(log_info)
    wandb.log(
        {
            "test_loss": mse_meter.avg,
            "epoch": epoch,
        },
    )

    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_test.png")
        plot_fields(
            fields=output.clone().detach(),
            ground_truth=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
            filepath=filepath,
        )
        if output_correction is not None:
            filepath = os.path.join(dir_path, f"epoch_{epoch}_test_corr.png")
            plot_fields(
                fields=output_correction.clone().detach(),
                ground_truth=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath,
            )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
        print("cuda is available and set to device: ", device, flush=True)
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic(int(configs.run.random_state))

    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
    )
    lg.info(model)


    train_loader, validation_loader, test_loader = builder.make_dataloader()

    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )
    test_criterion = builder.make_criterion(
        configs.test_criterion.name, configs.test_criterion
    ).to(device)

    optimizer = builder.make_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    scheduler = builder.make_scheduler(optimizer, config_file=configs.lr_scheduler)
    aux_criterions = {
        name: [builder.make_criterion(name, cfg=config), float(config.weight)]
        for name, config in configs.aux_criterion.items()
        if float(config.weight) > 0
    }
    print("aux criterions used in training: ", aux_criterions, flush=True)

    log_criterions = {
        name: [builder.make_criterion(name, cfg=config), float(config.weight)]
        for name, config in configs.log_criterion.items()
        if float(config.weight) > 0
    }
    print("criterions to be printed: ", log_criterions, flush=True)

    mixup_config = configs.dataset.augment
    # mixup_fn = MixupAll(**mixup_config)
    # test_mixup_fn = MixupAll(**configs.dataset.test_augment)
    mixup_fn = None
    test_mixup_fn = None
    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=10,
        metric_name="err",
        format="{:.4f}",
    )

    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    wandb.login()
    tag = wandb.util.generate_id()
    group = f"{datetime.date.today()}"
    name = f"{configs.run.wandb.name}-{datetime.datetime.now().hour:02d}{datetime.datetime.now().minute:02d}{datetime.datetime.now().second:02d}-{tag}"
    configs.run.pid = os.getpid()
    run = wandb.init(
        project=configs.run.wandb.project,
        # entity=configs.run.wandb.entity,
        group=group,
        name=name,
        id=tag,
        # Track hyperparameters and run metadata
        config=configs,
    )

    lossv, accv = [0], [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {name} starts. Group: {group}, Run ID: ({run.id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        lg.info(configs)
        if (
            int(configs.checkpoint.resume)
            and len(configs.checkpoint.restore_checkpoint) > 0
        ):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )
            lg.info("Validate resumed model...")
            test(
                model,
                test_loader,
                epoch,
                test_criterion,
                aux_criterions,
                [],
                [],
                device,
                mixup_fn=test_mixup_fn,
                plot=configs.plot.test,
            )
            quit()
        for epoch in range(1, int(configs.run.n_epochs) + 1):
            # single_batch_check(
            #     model,
            #     train_loader,
            #     optimizer,
            #     criterion,
            #     aux_criterions,
            #     epoch,
            #     mixup_fn,
            #     device,
            #     grad_scaler=grad_scaler,
            # )
            # quit()
            train(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                aux_criterions,
                mixup_fn,
                device,
                plot=configs.plot.train,
                grad_scaler=grad_scaler,
            )

            if validation_loader is not None:
                validate(
                    model,
                    validation_loader,
                    epoch,
                    test_criterion,
                    log_criterions,
                    lossv,
                    accv,
                    device,
                    mixup_fn=test_mixup_fn,
                    plot=configs.plot.valid,
                )
            if epoch > int(configs.run.n_epochs) - 21:
                test(
                    model,
                    test_loader,
                    epoch,
                    test_criterion,
                    log_criterions,
                    [],
                    [],
                    device,
                    mixup_fn=test_mixup_fn,
                    plot=configs.plot.test,
                )
                saver.save_model(
                    model,
                    lossv[-1],
                    epoch=epoch,
                    path=checkpoint,
                    save_model=False,
                    print_msg=True,
                )
        wandb.finish()
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
