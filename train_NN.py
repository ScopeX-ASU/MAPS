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
import mlflow
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
    for batch_idx, (eps_map, adj_srcs, gradient, field_solutions, s_params, src_profiles, fields_adj, field_normalizer, design_region_mask, orgion_size, incident_field) in enumerate(train_loader):
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
        # for key, design_region in design_region_mask.items():
        #     design_region_mask[key] = design_region.to(device, non_blocking=True)

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
            plot_fourier_eps(eps_map, "./figs/eps_map.png")
            if type(output) == tuple:
                output, aux_output = output
            else:
                aux_output = None
            # print("this is the key of the field solutions: ", field_solutions.keys())
            # print("this is the key of the fields adj: ", fields_adj.keys())
            # print("this is the key of the field normalizer: ", field_normalizer.keys()) "field_adj_normalizer-wl-1.55-port-in_port_1-mode-1"
            # quit()
            regression_loss = criterion(output, field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...], torch.ones_like(output).to(device))
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
            plot_fields(
                fields=output.clone().detach(),
                ground_truth=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                filepath=filepath,
            )
            plot_fouier_transform(
                field=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                filepath=filepath.replace(".png", "_fft.png"),
            )
            quit()
    return None

def plot_fourier_eps(
    eps_map: torch.Tensor,
    filepath: str,
) -> None:
    eps_map = eps_map[0]
    eps_map0 = eps_map.cpu().numpy()
    plt.imshow(eps_map0)
    plt.savefig(filepath.replace(".png", "_org_eps.png"))
    eps_map = torch.fft.fft2(eps_map)           # 2D FFT
    eps_map = torch.fft.fftshift(eps_map)       # Shift zero frequency to center
    eps_map = torch.abs(eps_map) 
    eps_map = eps_map.cpu().numpy()
    plt.imshow(eps_map)
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
    aux_output_weight = getattr(configs.criterion, "aux_output_weight", 0)

    data_counter = 0
    total_data = len(train_loader.dataset)
    for batch_idx, (eps_map, adj_srcs, gradient, field_solutions, s_params, src_profiles, fields_adj, field_normalizer, design_region_mask, orgion_size, incident_field) in enumerate(train_loader):
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
        # for key, design_region in design_region_mask.items():
        #     design_region_mask[key] = design_region.to(device, non_blocking=True)

        data_counter += eps_map.shape[0]

        if mixup_fn is not None:
            eps_map, adj_src, gradient, field_solutions, s_params = mixup_fn(eps_map, adj_src, gradient, field_solutions, s_params)

        # # visualize the input data
        # plt.imshow(eps_map[0].detach().cpu().numpy())
        # plt.savefig(f"./figs/eps_map_{epoch}.png")
        # plt.close()
        # plt.imshow(gradient[0].detach().cpu().numpy())
        # plt.savefig(f"./figs/gradient_{epoch}.png")
        # plt.close()

        # dummy_key = next(iter(src_profiles.keys()))
        # dummy_src_profile = src_profiles[dummy_key]
        # print("this is the shape of the src profile: ", dummy_src_profile.shape)
        # src_profile = torch.abs(dummy_src_profile[0])
        # plt.imshow(src_profile.abs().detach().cpu().numpy())
        # plt.savefig(f"./figs/src_profile_{epoch}.png")
        # plt.close()

        # dummy_key = next(iter(adj_srcs.keys()))
        # dummy_adjsrc_profile = adj_srcs[dummy_key]
        # adjsrc_profile = torch.abs(dummy_adjsrc_profile[0])
        # plt.imshow(adjsrc_profile.abs().detach().cpu().numpy())
        # plt.savefig(f"./figs/adjsrc_profile_{epoch}.png")
        # plt.close()

        # dummy_key = next(iter(field_solutions.keys()))
        # dummy_field_solution = field_solutions[dummy_key]
        # field_solution = dummy_field_solution[0][0]
        # plt.imshow(torch.abs(field_solution).detach().cpu().numpy())
        # plt.savefig(f"./figs/field_solution_{epoch}.png")
        # plt.close()

        # dummy_key = next(iter(fields_adj.keys()))
        # dummy_fields_adj = fields_adj[dummy_key]
        # fields_adj = dummy_fields_adj[0][0]
        # plt.imshow(torch.abs(fields_adj).detach().cpu().numpy())
        # plt.savefig(f"./figs/fields_adj_{epoch}.png")
        # plt.close()
        # quit()

        with amp.autocast('cuda', enabled=grad_scaler._enabled):
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
            # print("this is the key of the field solutions: ", field_solutions.keys())
            # print("this is the key of the fields adj: ", fields_adj.keys())
            # print("this is the key of the field normalizer: ", field_normalizer.keys()) "field_adj_normalizer-wl-1.55-port-in_port_1-mode-1"
            # quit()
            regression_loss = criterion(output, field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...], torch.ones_like(output).to(device))
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

    scheduler.step()
    avg_regression_loss = mse_meter.avg
    lg.info(f"Train Regression Loss: {avg_regression_loss:.4e}")
    # mlflow.log_metrics(
    #     {"train_regression": avg_regression_loss, "lr": get_learning_rate(optimizer)},
    #     step=epoch,
    # )
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
            ground_truth=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
            filepath=filepath,
        )

def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = True,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    with torch.no_grad(), DeterministicCtx(42):
        for batch_idx, (eps_map, adj_srcs, gradient, field_solutions, s_params, src_profiles, fields_adj, field_normalizer, design_region_mask, orgion_size, incident_field) in enumerate(validation_loader):
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
            # for key, design_region in design_region_mask.items():
            #     design_region_mask[key] = design_region.to(device, non_blocking=True)

            if mixup_fn is not None:
                eps_map, adj_src, gradient, field_solutions, s_params = mixup_fn(eps_map, adj_src, gradient, field_solutions, s_params)

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
                val_loss = criterion(output, field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...], torch.ones_like(output).to(device))
                # val_loss = val_loss + criterion(aux_output, fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"], torch.ones_like(aux_output).to(device))
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nValidation set: Average loss: {:.4e}\n".format(mse_meter.avg))
    # mlflow.log_metrics({"val_loss": mse_meter.avg}, step=epoch)
    wandb.log(
        {
            "val_loss": mse_meter.avg,
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
            ground_truth=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
            filepath=filepath,
        )


def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    with torch.no_grad(), DeterministicCtx(42):
        for batch_idx, (eps_map, adj_srcs, gradient, field_solutions, s_params, src_profiles, fields_adj, field_normalizer, design_region_mask, orgion_size, incident_field) in enumerate(test_loader):
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
            # for key, design_region in design_region_mask.items():
            #     design_region_mask[key] = design_region.to(device, non_blocking=True)

            if mixup_fn is not None:
                eps_map, adj_src, gradient, field_solutions, s_params = mixup_fn(eps_map, adj_src, gradient, field_solutions, s_params)

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
                val_loss = criterion(output, field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...], torch.ones_like(output).to(device))
                # val_loss = val_loss + criterion(aux_output, fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"], torch.ones_like(aux_output).to(device))
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nTest set: Average loss: {:.4e}\n".format(mse_meter.avg))
    # mlflow.log_metrics({"test_loss": mse_meter.avg}, step=epoch)
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
            ground_truth=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
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
    print(aux_criterions)
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
                0,
                test_criterion,
                [],
                [],
                device,
                mixup_fn=test_mixup_fn,
                plot=configs.plot.test,
            )
        for epoch in range(1, int(configs.run.n_epochs) + 1):
            single_batch_check(
                model,
                train_loader,
                optimizer,
                criterion,
                aux_criterions,
                epoch,
                mixup_fn,
                device,
                grad_scaler=grad_scaler,
            )
            quit()
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
