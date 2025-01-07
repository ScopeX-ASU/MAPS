"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-05-10 20:34:02
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-06 19:00:15
"""

#!/usr/bin/env python
# coding=UTF-8
import argparse
import datetime
import os
import random
from typing import Callable, Dict, Iterable

# import mlflow
import torch
import torch.amp as amp
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler

import wandb
from core.train import builder
from core.train.models.utils import from_Ez_to_Hx_Hy
from core.utils import (
    DeterministicCtx,
    cal_total_field_adj_src_from_fwd_field,
    plot_fields,
)
from thirdparty.ceviche.constants import *


def single_batch_check(
    model_fwd: nn.Module,
    model_adj: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Criterion,
    aux_criterions: Dict,
    epoch: int = 0,
    mixup_fn: Callable = None,
    device: torch.device = torch.device("cuda:0"),
    grad_scaler=None,
) -> None:
    model_fwd.train()
    if model_adj is not None:
        model_adj.train()
    step = epoch * len(train_loader)

    mse_meter = AverageMeter("mse")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}

    data_counter = 0
    total_data = len(train_loader.dataset)
    rand_idx = len(train_loader.dataset) // train_loader.batch_size - 1
    rand_idx = random.randint(0, rand_idx)
    for batch_idx, (
        eps_map,
        adj_srcs,
        gradient,
        field_solutions,
        s_params,
        src_profiles,
        fields_adj,
        field_normalizer,
        design_region_mask,
        ht_m,
        et_m,
        monitor_slices,
        As,
    ) in enumerate(train_loader):
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
        # for key, field in incident_field.items():
        #     incident_field[key] = field.to(device, non_blocking=True)
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
                    item_to_add = F.interpolate(
                        item_to_add,
                        size=size[i].item(),
                        mode="linear",
                        align_corners=True,
                    )
                    item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                    ht_list.append(
                        torch.view_as_complex(item_to_add).to(device, non_blocking=True)
                    )
                ht_m[key] = ht_list
        for key, et in et_m.items():
            if key.endswith("-origin_size"):
                continue
            else:
                size = et_m[key + "-origin_size"]
                et_list = []
                for i in range(size.shape[0]):
                    item_to_add = torch.view_as_real(et[i]).permute(1, 0).unsqueeze(0)
                    item_to_add = F.interpolate(
                        item_to_add,
                        size=size[i].item(),
                        mode="linear",
                        align_corners=True,
                    )
                    item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                    et_list.append(
                        torch.view_as_complex(item_to_add).to(device, non_blocking=True)
                    )
                et_m[key] = et_list
        for key, A in As.items():
            As[key] = A.to(device, non_blocking=True)

        data_counter += eps_map.shape[0]
        if mixup_fn is not None:
            eps_map, adj_src, gradient, field_solutions, s_params = mixup_fn(
                eps_map, adj_src, gradient, field_solutions, s_params
            )
        if batch_idx == rand_idx:
            break

    for iter in range(10000):
        with amp.autocast("cuda", enabled=False):
            # forward
            output = model_fwd(  # now only suppose that the output is the gradient of the field
                eps_map,
                src_profiles["source_profile-wl-1.55-port-in_port_1-mode-1"]
                if model_fwd.train_field == "fwd"
                else adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"],
            )
            if isinstance(output, tuple):
                forward_Ez_field, forward_Ez_field_err_corr = output
            else:
                forward_Ez_field = output
                forward_field_err_corr = None

            forward_field, adjoint_source = cal_total_field_adj_src_from_fwd_field(
                Ez=forward_Ez_field,
                eps=eps_map,
                ht_ms=ht_m,
                et_ms=et_m,
                monitors=monitor_slices,
                pml_mask=model_fwd.pml_mask,
                from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                return_adj_src=False
                if (model_adj is None) or model_fwd.err_correction
                else True,
                sim=model_fwd.sim,
            )
            if adjoint_source is not None:
                adjoint_source = adjoint_source.detach()
            if model_fwd.err_correction:
                forward_field_err_corr, adjoint_source = (
                    cal_total_field_adj_src_from_fwd_field(
                        Ez=forward_Ez_field_err_corr,
                        eps=eps_map,
                        ht_ms=ht_m,
                        et_ms=et_m,
                        monitors=monitor_slices,
                        pml_mask=model_fwd.pml_mask,
                        from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                        return_adj_src=False if model_adj is None else True,
                        sim=model_fwd.sim,
                    )
                )
            # finish calculating the forward field
            if model_adj is not None:
                assert (
                    adjoint_source is not None
                ), "The adjoint source should be calculated"

                adjoint_source = adjoint_source * (
                    field_normalizer[
                        "field_adj_normalizer-wl-1.55-port-in_port_1-mode-1"
                    ].unsqueeze(1)
                )
                adjoint_output = model_adj(
                    eps_map,
                    adjoint_source,  # bs, H, W complex
                )

                if isinstance(adjoint_output, tuple):
                    adjoint_Ez_field, adjoint_Ez_field_err_corr = adjoint_output
                else:
                    adjoint_Ez_field = adjoint_output
                    adjoint_field_err_corr = None

                adjoint_field, _ = cal_total_field_adj_src_from_fwd_field(
                    Ez=adjoint_Ez_field,
                    eps=eps_map,
                    ht_ms=ht_m,
                    et_ms=et_m,
                    monitors=monitor_slices,
                    pml_mask=model_fwd.pml_mask,
                    from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                    return_adj_src=False,
                    sim=model_adj.sim,
                )
                if model_fwd.err_correction:
                    adjoint_field_err_corr, _ = cal_total_field_adj_src_from_fwd_field(
                        Ez=adjoint_Ez_field_err_corr,
                        eps=eps_map,
                        ht_ms=ht_m,
                        et_ms=et_m,
                        monitors=monitor_slices,
                        pml_mask=model_fwd.pml_mask,
                        from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                        return_adj_src=False,
                        sim=model_adj.sim,
                    )

            regression_loss = criterion(
                forward_field[:, -2:, ...],
                field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][
                    :, -2:, ...
                ]
                if model_fwd.train_field == "fwd"
                else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                    :, -2:, ...
                ],
                torch.ones_like(forward_field[:, -2:, ...]).to(device),
            )
            if model_adj is not None:
                regression_loss = (
                    regression_loss
                    + criterion(
                        adjoint_field[:, -2:, ...],
                        fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                            :, -2:, ...
                        ],
                        torch.ones_like(adjoint_field[:, -2:, ...]).to(device),
                    )
                ) / 2
            mse_meter.update(regression_loss.item())
            loss = regression_loss
            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                if name == "maxwell_residual_loss":
                    aux_loss = weight * aux_criterion(
                        Ez=forward_field,
                        # Ez=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                        source=src_profiles[
                            "source_profile-wl-1.55-port-in_port_1-mode-1"
                        ]
                        if model_fwd.train_field == "fwd"
                        else adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"],
                        As=As,
                        transpose_A=False if model_fwd.train_field == "fwd" else True,
                    )
                    if model_adj is not None:
                        aux_loss = (
                            aux_loss
                            + weight
                            * aux_criterion(
                                Ez=adjoint_field,
                                # Ez=fields_adj['fields_adj-wl-1.55-port-in_port_1-mode-1'],
                                # source=adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"],
                                source=adjoint_source,
                                As=As,
                                transpose_A=True
                                if model_adj.train_field == "adj"
                                else False,
                            )
                        ) / 2
                elif name == "grad_loss":
                    # there is no need to distinguish the forward and adjoint field here
                    # since the gradient must combine both forward and adjoint field
                    aux_loss = (
                        weight
                        * aux_criterion(
                            forward_fields=forward_field
                            if forward_field_err_corr is None
                            else forward_field_err_corr,
                            # forward_fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                            # backward_fields=field_solutions["field_solutions-wl-1.55-port-out_port_1-mode-1"][:, -2:, ...],
                            adjoint_fields=adjoint_field
                            if adjoint_field_err_corr is None
                            else adjoint_field_err_corr,
                            # adjoint_fields=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                            # backward_adjoint_fields = fields_adj['fields_adj-wl-1.55-port-in_port_1-mode-1'][:, -2:, ...],
                            target_gradient=gradient,
                            gradient_multiplier=field_normalizer,  # TODO the nomalizer should calculate from the forward field
                            # dr_mask=None,
                            dr_mask=design_region_mask,
                        )
                    )
                elif name == "s_param_loss":
                    # there is also no need to distinguish the forward and adjoint field here
                    # the s_param_loss is calculated based on the forward field and there is no label for the adjoint field
                    assert (
                        model_fwd.train_field == "fwd"
                    ), "The s_param_loss is only calculated based on the forward field"
                    aux_loss = (
                        weight
                        * aux_criterion(
                            fields=forward_field
                            if forward_field_err_corr is None
                            else forward_field_err_corr,
                            # fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                            ht_m=ht_m["ht_m-wl-1.55-port-out_port_1-mode-1"],
                            et_m=et_m["et_m-wl-1.55-port-out_port_1-mode-1"],
                            monitor_slices=monitor_slices,  # 'port_slice-out_port_1_x', 'port_slice-out_port_1_y'
                            target_SParam=s_params["s_params-fwd_trans-1.55-1"],
                        )
                    )
                elif name == "err_corr_Ez":
                    assert model_fwd.err_correction
                    aux_loss = weight * aux_criterion(
                        forward_field_err_corr[:, -2:, ...],
                        field_solutions[
                            "field_solutions-wl-1.55-port-in_port_1-mode-1"
                        ][:, -2:, ...]
                        if model_fwd.train_field == "fwd"
                        else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                            :, -2:, ...
                        ],
                        torch.ones_like(forward_field_err_corr[:, -2:, ...]).to(device),
                    )
                    if model_adj is not None:
                        assert model_adj.err_correction
                        aux_loss = (
                            aux_loss
                            + weight
                            * aux_criterion(
                                adjoint_field_err_corr[:, -2:, ...],
                                fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                    :, -2:, ...
                                ],
                                torch.ones_like(adjoint_field_err_corr[:, -2:, ...]).to(
                                    device
                                ),
                            )
                        ) / 2
                elif name == "err_corr_Hx":
                    assert model_fwd.err_correction
                    aux_loss = weight * aux_criterion(
                        forward_field_err_corr[:, :2, ...],
                        field_solutions[
                            "field_solutions-wl-1.55-port-in_port_1-mode-1"
                        ][:, :2, ...]
                        if model_fwd.train_field == "fwd"
                        else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                            :, :2, ...
                        ],
                        torch.ones_like(forward_field_err_corr[:, :2, ...]).to(device),
                    )
                    if model_adj is not None:
                        assert model_adj.err_correction
                        aux_loss = (
                            aux_loss
                            + weight
                            * aux_criterion(
                                adjoint_field_err_corr[:, :2, ...],
                                fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                    :, :2, ...
                                ],
                                torch.ones_like(adjoint_field_err_corr[:, :2, ...]).to(
                                    device
                                ),
                            )
                        ) / 2
                elif name == "err_corr_Hy":
                    assert model_fwd.err_correction
                    aux_loss = weight * aux_criterion(
                        forward_field_err_corr[:, 2:4, ...],
                        field_solutions[
                            "field_solutions-wl-1.55-port-in_port_1-mode-1"
                        ][:, 2:4, ...]
                        if model_fwd.train_field == "fwd"
                        else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                            :, 2:4, ...
                        ],
                        torch.ones_like(forward_field_err_corr[:, 2:4, ...]).to(device),
                    )
                    if model_adj is not None:
                        assert model_adj.err_correction
                        aux_loss = (
                            aux_loss
                            + weight
                            * aux_criterion(
                                adjoint_field_err_corr[:, 2:4, ...],
                                fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                    :, 2:4, ...
                                ],
                                torch.ones_like(adjoint_field_err_corr[:, 2:4, ...]).to(
                                    device
                                ),
                            )
                        ) / 2
                elif name == "Hx_loss":
                    aux_loss = weight * aux_criterion(
                        forward_field[:, :2, ...],
                        field_solutions[
                            "field_solutions-wl-1.55-port-in_port_1-mode-1"
                        ][:, :2, ...]
                        if model_fwd.train_field == "fwd"
                        else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                            :, :2, ...
                        ],
                        torch.ones_like(forward_field[:, :2, ...]).to(device),
                    )
                    if model_adj is not None:
                        aux_loss = (
                            aux_loss
                            + weight
                            * aux_criterion(
                                adjoint_field[:, :2, ...],
                                fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                    :, :2, ...
                                ],
                                torch.ones_like(adjoint_field[:, :2, ...]).to(device),
                            )
                        ) / 2
                elif name == "Hy_loss":
                    aux_loss = weight * aux_criterion(
                        forward_field[:, 2:4, ...],
                        field_solutions[
                            "field_solutions-wl-1.55-port-in_port_1-mode-1"
                        ][:, 2:4, ...]
                        if model_fwd.train_field == "fwd"
                        else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                            :, 2:4, ...
                        ],
                        torch.ones_like(forward_field[:, 2:4, ...]).to(device),
                    )
                    if model_adj is not None:
                        aux_loss = (
                            aux_loss
                            + weight
                            * aux_criterion(
                                adjoint_field[:, 2:4, ...],
                                fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                    :, 2:4, ...
                                ],
                                torch.ones_like(adjoint_field[:, 2:4, ...]).to(device),
                            )
                        ) / 2
                aux_meters[name].update(aux_loss.item())  # record the aux loss first
                loss = loss + aux_loss

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
            # quit()
    return None


def train(
    model_fwd: nn.Module,
    model_adj: nn.Module,
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
    lambda_: float = 1.0,  # Lagrange multiplier
    mu: float = 1.0,  # Penalty coefficient
    mu_growth: float = 10.0,  # Growth rate for penalty coefficient
    constraint_tol: float = 1e-4,  # Tolerance for residual
) -> None:
    torch.autograd.set_detect_anomaly(True)
    model_fwd.train()
    if model_adj is not None:
        model_adj.train()
    step = epoch * len(train_loader)

    mse_meter = AverageMeter("mse")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}

    data_counter = 0
    total_data = len(train_loader.dataset)
    for batch_idx, (
        eps_map,
        adj_srcs,
        gradient,
        field_solutions,
        s_params,
        src_profiles,
        fields_adj,
        field_normalizer,
        design_region_mask,
        ht_m,
        et_m,
        monitor_slices,
        As,
    ) in enumerate(train_loader):
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
        # for key, field in incident_field.items():
        #     incident_field[key] = field.to(device, non_blocking=True)
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
                    item_to_add = F.interpolate(
                        item_to_add,
                        size=size[i].item(),
                        mode="linear",
                        align_corners=True,
                    )
                    item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                    ht_list.append(
                        torch.view_as_complex(item_to_add).to(device, non_blocking=True)
                    )
                ht_m[key] = ht_list
        for key, et in et_m.items():
            if key.endswith("-origin_size"):
                continue
            else:
                size = et_m[key + "-origin_size"]
                et_list = []
                for i in range(size.shape[0]):
                    item_to_add = torch.view_as_real(et[i]).permute(1, 0).unsqueeze(0)
                    item_to_add = F.interpolate(
                        item_to_add,
                        size=size[i].item(),
                        mode="linear",
                        align_corners=True,
                    )
                    item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                    et_list.append(
                        torch.view_as_complex(item_to_add).to(device, non_blocking=True)
                    )
                et_m[key] = et_list
        for key, A in As.items():
            As[key] = A.to(device, non_blocking=True)

        data_counter += eps_map.shape[0]
        if mixup_fn is not None:
            eps_map, adj_src, gradient, field_solutions, s_params = mixup_fn(
                eps_map, adj_src, gradient, field_solutions, s_params
            )

        with amp.autocast("cuda", enabled=grad_scaler._enabled):
            # forward
            output = model_fwd(  # now only suppose that the output is the gradient of the field
                eps_map,
                src_profiles["source_profile-wl-1.55-port-in_port_1-mode-1"]
                if model_fwd.train_field == "fwd"
                else adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"],
            )
            if isinstance(output, tuple):
                forward_Ez_field, forward_Ez_field_err_corr = output
            else:
                forward_Ez_field = output
                forward_field_err_corr = None
            # calculate Hx and Hy from forward_Ez_field
            # -----sanity check-----
            # forward_Ez_field = field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...]
            # forward_Ez_field.requires_grad = True
            # -----check passed-----
            forward_field, adjoint_source = cal_total_field_adj_src_from_fwd_field(
                Ez=forward_Ez_field,
                eps=eps_map,
                ht_ms=ht_m,
                et_ms=et_m,
                monitors=monitor_slices,
                pml_mask=model_fwd.pml_mask,
                from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                return_adj_src=False
                if (model_adj is None) or model_fwd.err_correction
                else True,
                sim=model_fwd.sim,
            )
            if adjoint_source is not None:
                adjoint_source = adjoint_source.detach()
            if model_fwd.err_correction:
                forward_field_err_corr, adjoint_source = (
                    cal_total_field_adj_src_from_fwd_field(
                        Ez=forward_Ez_field_err_corr,
                        eps=eps_map,
                        ht_ms=ht_m,
                        et_ms=et_m,
                        monitors=monitor_slices,
                        pml_mask=model_fwd.pml_mask,
                        from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                        return_adj_src=False if model_adj is None else True,
                        sim=model_fwd.sim,
                    )
                )
            # finish calculating the forward field
            if model_adj is not None:
                assert (
                    adjoint_source is not None
                ), "The adjoint source should be calculated"

                adjoint_source = adjoint_source * (
                    field_normalizer[
                        "field_adj_normalizer-wl-1.55-port-in_port_1-mode-1"
                    ].unsqueeze(1)
                )
                # -----sanity check-----
                # difference = (adjoint_source - adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"]).to(torch.complex128)

                # error_energy = torch.norm(difference, p=2, dim=(-1, -2)).double()
                # field_energy = torch.norm(adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"].to(torch.complex128), p=2, dim=(-1, -2)).double()

                # print("this is the error energy: ", error_energy.mean())
                # print("this is the field energy: ", field_energy.mean())
                # print("this is the NL2norm: ", (error_energy / field_energy).mean())

                # print("this is the shape of the adjoint source: ", adjoint_source.shape, flush=True)
                # print("the following two stats should be the same")
                # print_stat(adjoint_source[0])
                # print_stat(adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"][0])
                # fig, ax = plt.subplots(2, 2, figsize=(10, 8))

                # # Plot the images
                # cal_adj_srouce_real = ax[0][0].imshow(adjoint_source[0].real.cpu().detach().numpy())
                # cal_adj_srouce_imag = ax[0][1].imshow(adjoint_source[0].imag.cpu().detach().numpy())
                # adj_src_real = ax[1][0].imshow(adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"][0].real.cpu().detach().numpy())
                # adj_src_imag = ax[1][1].imshow(adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"][0].imag.cpu().detach().numpy())

                # # Set individual subplot titles
                # ax[0][0].set_title("cal_adj_srouce_real")
                # ax[0][1].set_title("cal_adj_srouce_imag")
                # ax[1][0].set_title("adj_src_real")
                # ax[1][1].set_title("adj_src_imag")

                # # Add colorbars for each subplot
                # plt.colorbar(cal_adj_srouce_real, ax=ax[0][0])
                # plt.colorbar(cal_adj_srouce_imag, ax=ax[0][1])
                # plt.colorbar(adj_src_real, ax=ax[1][0])
                # plt.colorbar(adj_src_imag, ax=ax[1][1])

                # # Set a global title
                # fig.suptitle("Compare the adjoint source and the adjoint source from the adjoint field")

                # # Save the figure
                # plt.savefig("./figs/cmp_adj_src.png")
                # # compare the cosine similarity between the adjoint source and the adjoint source from the adjoint field
                # quit()
                # -----check passed-----
                adjoint_output = model_adj(
                    eps_map,
                    adjoint_source,  # bs, H, W complex
                )
                # if isinstance(adjoint_output, tuple):
                #     adjoint_field, adjoint_field_err_corr = adjoint_output
                # else:
                #     adjoint_field = adjoint_output
                #     adjoint_field_err_corr = None

                if isinstance(adjoint_output, tuple):
                    adjoint_Ez_field, adjoint_Ez_field_err_corr = adjoint_output
                else:
                    adjoint_Ez_field = adjoint_output
                    adjoint_field_err_corr = None

                adjoint_field, _ = cal_total_field_adj_src_from_fwd_field(
                    Ez=adjoint_Ez_field,
                    eps=eps_map,
                    ht_ms=ht_m,
                    et_ms=et_m,
                    monitors=monitor_slices,
                    pml_mask=model_fwd.pml_mask,
                    from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                    return_adj_src=False,
                    sim=model_adj.sim,
                )
                if model_fwd.err_correction:
                    adjoint_field_err_corr, _ = cal_total_field_adj_src_from_fwd_field(
                        Ez=adjoint_Ez_field_err_corr,
                        eps=eps_map,
                        ht_ms=ht_m,
                        et_ms=et_m,
                        monitors=monitor_slices,
                        pml_mask=model_fwd.pml_mask,
                        from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                        return_adj_src=False,
                        sim=model_adj.sim,
                    )

            regression_loss = criterion(
                forward_field[:, -2:, ...],
                field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][
                    :, -2:, ...
                ]
                if model_fwd.train_field == "fwd"
                else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                    :, -2:, ...
                ],
                torch.ones_like(forward_field[:, -2:, ...]).to(device),
            )
            if model_adj is not None:
                regression_loss = (
                    regression_loss
                    + criterion(
                        adjoint_field[:, -2:, ...],
                        fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                            :, -2:, ...
                        ],
                        torch.ones_like(adjoint_field[:, -2:, ...]).to(device),
                    )
                ) / 2
            mse_meter.update(regression_loss.item())
            loss = regression_loss
            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                if name == "maxwell_residual_loss":
                    aux_loss = weight * aux_criterion(
                        Ez=forward_field,
                        # Ez=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                        source=src_profiles[
                            "source_profile-wl-1.55-port-in_port_1-mode-1"
                        ]
                        if model_fwd.train_field == "fwd"
                        else adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"],
                        As=As,
                        transpose_A=False if model_fwd.train_field == "fwd" else True,
                    )
                    if model_adj is not None:
                        aux_loss = (
                            aux_loss
                            + weight
                            * aux_criterion(
                                Ez=adjoint_field,
                                # Ez=fields_adj['fields_adj-wl-1.55-port-in_port_1-mode-1'],
                                # source=adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"],
                                source=adjoint_source,
                                As=As,
                                transpose_A=True
                                if model_adj.train_field == "adj"
                                else False,
                            )
                        ) / 2
                elif name == "grad_loss":
                    # there is no need to distinguish the forward and adjoint field here
                    # since the gradient must combine both forward and adjoint field
                    aux_loss = (
                        weight
                        * aux_criterion(
                            forward_fields=forward_field
                            if forward_field_err_corr is None
                            else forward_field_err_corr,
                            # forward_fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                            # backward_fields=field_solutions["field_solutions-wl-1.55-port-out_port_1-mode-1"][:, -2:, ...],
                            adjoint_fields=adjoint_field
                            if adjoint_field_err_corr is None
                            else adjoint_field_err_corr,
                            # adjoint_fields=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                            # backward_adjoint_fields = fields_adj['fields_adj-wl-1.55-port-in_port_1-mode-1'][:, -2:, ...],
                            target_gradient=gradient,
                            gradient_multiplier=field_normalizer,  # TODO the nomalizer should calculate from the forward field
                            # dr_mask=None,
                            dr_mask=design_region_mask,
                        )
                    )
                elif name == "s_param_loss":
                    # there is also no need to distinguish the forward and adjoint field here
                    # the s_param_loss is calculated based on the forward field and there is no label for the adjoint field
                    assert (
                        model_fwd.train_field == "fwd"
                    ), "The s_param_loss is only calculated based on the forward field"
                    aux_loss = (
                        weight
                        * aux_criterion(
                            fields=forward_field
                            if forward_field_err_corr is None
                            else forward_field_err_corr,
                            # fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                            ht_m=ht_m["ht_m-wl-1.55-port-out_port_1-mode-1"],
                            et_m=et_m["et_m-wl-1.55-port-out_port_1-mode-1"],
                            monitor_slices=monitor_slices,  # 'port_slice-out_port_1_x', 'port_slice-out_port_1_y'
                            target_SParam=s_params["s_params-fwd_trans-1.55-1"],
                        )
                    )
                elif name == "err_corr_Ez":
                    assert model_fwd.err_correction
                    aux_loss = weight * aux_criterion(
                        forward_field_err_corr[:, -2:, ...],
                        field_solutions[
                            "field_solutions-wl-1.55-port-in_port_1-mode-1"
                        ][:, -2:, ...]
                        if model_fwd.train_field == "fwd"
                        else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                            :, -2:, ...
                        ],
                        torch.ones_like(forward_field_err_corr[:, -2:, ...]).to(device),
                    )
                    if model_adj is not None:
                        assert model_adj.err_correction
                        aux_loss = (
                            aux_loss
                            + weight
                            * aux_criterion(
                                adjoint_field_err_corr[:, -2:, ...],
                                fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                    :, -2:, ...
                                ],
                                torch.ones_like(adjoint_field_err_corr[:, -2:, ...]).to(
                                    device
                                ),
                            )
                        ) / 2
                elif name == "err_corr_Hx":
                    assert model_fwd.err_correction
                    aux_loss = weight * aux_criterion(
                        forward_field_err_corr[:, :2, ...],
                        field_solutions[
                            "field_solutions-wl-1.55-port-in_port_1-mode-1"
                        ][:, :2, ...]
                        if model_fwd.train_field == "fwd"
                        else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                            :, :2, ...
                        ],
                        torch.ones_like(forward_field_err_corr[:, :2, ...]).to(device),
                    )
                    if model_adj is not None:
                        assert model_adj.err_correction
                        aux_loss = (
                            aux_loss
                            + weight
                            * aux_criterion(
                                adjoint_field_err_corr[:, :2, ...],
                                fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                    :, :2, ...
                                ],
                                torch.ones_like(adjoint_field_err_corr[:, :2, ...]).to(
                                    device
                                ),
                            )
                        ) / 2
                elif name == "err_corr_Hy":
                    assert model_fwd.err_correction
                    aux_loss = weight * aux_criterion(
                        forward_field_err_corr[:, 2:4, ...],
                        field_solutions[
                            "field_solutions-wl-1.55-port-in_port_1-mode-1"
                        ][:, 2:4, ...]
                        if model_fwd.train_field == "fwd"
                        else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                            :, 2:4, ...
                        ],
                        torch.ones_like(forward_field_err_corr[:, 2:4, ...]).to(device),
                    )
                    if model_adj is not None:
                        assert model_adj.err_correction
                        aux_loss = (
                            aux_loss
                            + weight
                            * aux_criterion(
                                adjoint_field_err_corr[:, 2:4, ...],
                                fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                    :, 2:4, ...
                                ],
                                torch.ones_like(adjoint_field_err_corr[:, 2:4, ...]).to(
                                    device
                                ),
                            )
                        ) / 2
                elif name == "Hx_loss":
                    aux_loss = weight * aux_criterion(
                        forward_field[:, :2, ...],
                        field_solutions[
                            "field_solutions-wl-1.55-port-in_port_1-mode-1"
                        ][:, :2, ...]
                        if model_fwd.train_field == "fwd"
                        else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                            :, :2, ...
                        ],
                        torch.ones_like(forward_field[:, :2, ...]).to(device),
                    )
                    if model_adj is not None:
                        aux_loss = (
                            aux_loss
                            + weight
                            * aux_criterion(
                                adjoint_field[:, :2, ...],
                                fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                    :, :2, ...
                                ],
                                torch.ones_like(adjoint_field[:, :2, ...]).to(device),
                            )
                        ) / 2
                elif name == "Hy_loss":
                    aux_loss = weight * aux_criterion(
                        forward_field[:, 2:4, ...],
                        field_solutions[
                            "field_solutions-wl-1.55-port-in_port_1-mode-1"
                        ][:, 2:4, ...]
                        if model_fwd.train_field == "fwd"
                        else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                            :, 2:4, ...
                        ],
                        torch.ones_like(forward_field[:, 2:4, ...]).to(device),
                    )
                    if model_adj is not None:
                        aux_loss = (
                            aux_loss
                            + weight
                            * aux_criterion(
                                adjoint_field[:, 2:4, ...],
                                fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                    :, 2:4, ...
                                ],
                                torch.ones_like(adjoint_field[:, 2:4, ...]).to(device),
                            )
                        ) / 2
                aux_meters[name].update(aux_loss.item())  # record the aux loss first
                if (
                    lambda_ is not None and name == "maxwell_residual_loss"
                ):  # which means that we are using ALM
                    aux_loss = aux_loss * lambda_ + (mu / 2) * aux_loss**2
                loss = loss + aux_loss

        grad_scaler.scale(loss).backward()
        # for p in model.parameters():
        #     print(p.grad, flush=True)
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

    # ALM Updates
    if constraint_tol is not None:  # which means that we are using ALM
        if aux_meters["maxwell_residual_loss"].avg > constraint_tol:
            lambda_ += mu * aux_meters["maxwell_residual_loss"].avg
            mu *= mu_growth
            lg.info(f"Updated ALM Parameters - Lambda: {lambda_}, Mu: {mu}")

    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_train")
        if model_fwd.err_correction:
            plot_fields(
                fields=forward_field.clone().detach(),
                ground_truth=field_solutions[
                    "field_solutions-wl-1.55-port-in_port_1-mode-1"
                ]
                if model_fwd.train_field == "fwd"
                else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath + f"_{model_fwd.train_field}.png",
            )
            plot_fields(
                fields=forward_field_err_corr.clone().detach(),
                ground_truth=field_solutions[
                    "field_solutions-wl-1.55-port-in_port_1-mode-1"
                ]
                if model_fwd.train_field == "fwd"
                else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath + f"_{model_fwd.train_field}_err_corr.png",
            )
            if model_adj is not None:
                plot_fields(
                    fields=adjoint_field.clone().detach(),
                    ground_truth=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                    filepath=filepath + "_adj.png",
                )
                plot_fields(
                    fields=adjoint_field_err_corr.clone().detach(),
                    ground_truth=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                    filepath=filepath + "_adj_err_corr.png",
                )
        else:
            plot_fields(
                fields=forward_field.clone().detach(),
                ground_truth=field_solutions[
                    "field_solutions-wl-1.55-port-in_port_1-mode-1"
                ]
                if model_fwd.train_field == "fwd"
                else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath + f"_{model_fwd.train_field}.png",
            )
            if model_adj is not None:
                plot_fields(
                    fields=adjoint_field.clone().detach(),
                    ground_truth=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                    filepath=filepath + "_adj.png",
                )

    return lambda_, mu


def validate(
    model_fwd: nn.Module,
    model_adj: nn.Module,
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
    model_fwd.eval()
    if model_adj is not None:
        model_adj.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    log_meters = {name: AverageMeter(name) for name in log_criterions}
    with torch.no_grad(), DeterministicCtx(42):
        for batch_idx, (
            eps_map,
            adj_srcs,
            gradient,
            field_solutions,
            s_params,
            src_profiles,
            fields_adj,
            field_normalizer,
            design_region_mask,
            ht_m,
            et_m,
            monitor_slices,
            As,
        ) in enumerate(validation_loader):
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
            # for key, field in incident_field.items():
            #     incident_field[key] = field.to(device, non_blocking=True)
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
                        item_to_add = (
                            torch.view_as_real(ht[i]).permute(1, 0).unsqueeze(0)
                        )
                        item_to_add = F.interpolate(
                            item_to_add,
                            size=size[i].item(),
                            mode="linear",
                            align_corners=True,
                        )
                        item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                        ht_list.append(
                            torch.view_as_complex(item_to_add).to(
                                device, non_blocking=True
                            )
                        )
                    ht_m[key] = ht_list
            for key, et in et_m.items():
                if key.endswith("-origin_size"):
                    continue
                else:
                    size = et_m[key + "-origin_size"]
                    et_list = []
                    for i in range(size.shape[0]):
                        item_to_add = (
                            torch.view_as_real(et[i]).permute(1, 0).unsqueeze(0)
                        )
                        item_to_add = F.interpolate(
                            item_to_add,
                            size=size[i].item(),
                            mode="linear",
                            align_corners=True,
                        )
                        item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                        et_list.append(
                            torch.view_as_complex(item_to_add).to(
                                device, non_blocking=True
                            )
                        )
                    et_m[key] = et_list
            for key, A in As.items():
                As[key] = A.to(device, non_blocking=True)

            if mixup_fn is not None:
                eps_map, adj_src, gradient, field_solutions, s_params = mixup_fn(
                    eps_map, adj_src, gradient, field_solutions, s_params
                )

            with amp.autocast("cuda", enabled=False):
                # forward
                output = model_fwd(  # now only suppose that the output is the gradient of the field
                    eps_map,
                    src_profiles["source_profile-wl-1.55-port-in_port_1-mode-1"]
                    if model_fwd.train_field == "fwd"
                    else adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"],
                )
                if isinstance(output, tuple):
                    forward_Ez_field, forward_Ez_field_err_corr = output
                else:
                    forward_Ez_field = output
                    forward_field_err_corr = None
                with torch.enable_grad():
                    forward_field, adjoint_source = (
                        cal_total_field_adj_src_from_fwd_field(
                            Ez=forward_Ez_field,
                            eps=eps_map,
                            ht_ms=ht_m,
                            et_ms=et_m,
                            monitors=monitor_slices,
                            pml_mask=model_fwd.pml_mask,
                            from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                            return_adj_src=False
                            if (model_adj is None) or model_fwd.err_correction
                            else True,
                            sim=model_fwd.sim,
                        )
                    )
                    if adjoint_source is not None:
                        adjoint_source = adjoint_source.detach()
                if model_fwd.err_correction:
                    forward_field_err_corr, adjoint_source = (
                        cal_total_field_adj_src_from_fwd_field(
                            Ez=forward_Ez_field_err_corr,
                            eps=eps_map,
                            ht_ms=ht_m,
                            et_ms=et_m,
                            monitors=monitor_slices,
                            pml_mask=model_fwd.pml_mask,
                            from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                            return_adj_src=False if model_adj is None else True,
                            sim=model_fwd.sim,
                        )
                    )

                if model_adj is not None:
                    assert (
                        adjoint_source is not None
                    ), "The adjoint source should be calculated"

                    adjoint_source = adjoint_source * (
                        field_normalizer[
                            "field_adj_normalizer-wl-1.55-port-in_port_1-mode-1"
                        ].unsqueeze(1)
                    )

                    adjoint_output = model_adj(
                        eps_map,
                        adjoint_source,  # bs, H, W complex
                    )

                    if isinstance(adjoint_output, tuple):
                        adjoint_Ez_field, adjoint_Ez_field_err_corr = adjoint_output
                    else:
                        adjoint_Ez_field = adjoint_output
                        adjoint_field_err_corr = None

                    adjoint_field, _ = cal_total_field_adj_src_from_fwd_field(
                        Ez=adjoint_Ez_field,
                        eps=eps_map,
                        ht_ms=ht_m,
                        et_ms=et_m,
                        monitors=monitor_slices,
                        pml_mask=model_fwd.pml_mask,
                        from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                        return_adj_src=False,
                        sim=model_adj.sim,
                    )
                    if model_fwd.err_correction:
                        adjoint_field_err_corr, _ = (
                            cal_total_field_adj_src_from_fwd_field(
                                Ez=adjoint_Ez_field_err_corr,
                                eps=eps_map,
                                ht_ms=ht_m,
                                et_ms=et_m,
                                monitors=monitor_slices,
                                pml_mask=model_fwd.pml_mask,
                                from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                                return_adj_src=False,
                                sim=model_adj.sim,
                            )
                        )

                val_loss = criterion(
                    forward_field[:, -2:, ...],
                    field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][
                        :, -2:, ...
                    ]
                    if model_fwd.train_field == "fwd"
                    else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                        :, -2:, ...
                    ],
                    torch.ones_like(forward_field[:, -2:, ...]).to(device),
                )
                if model_adj is not None:
                    val_loss = (
                        val_loss
                        + criterion(
                            adjoint_field[:, -2:, ...],
                            fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                :, -2:, ...
                            ],
                            torch.ones_like(adjoint_field[:, -2:, ...]).to(device),
                        )
                    ) / 2
                mse_meter.update(val_loss.item())
                loss = val_loss
                for name, config in log_criterions.items():
                    log_criterion, weight = config
                    if name == "maxwell_residual_loss":
                        log_loss = weight * log_criterion(
                            Ez=forward_field,
                            # Ez=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                            source=src_profiles[
                                "source_profile-wl-1.55-port-in_port_1-mode-1"
                            ]
                            if model_fwd.train_field == "fwd"
                            else adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"],
                            As=As,
                            transpose_A=False
                            if model_fwd.train_field == "fwd"
                            else True,
                        )
                        if model_adj is not None:
                            log_loss = (
                                log_loss
                                + weight
                                * log_criterion(
                                    Ez=adjoint_field,
                                    # Ez=fields_adj['fields_adj-wl-1.55-port-in_port_1-mode-1'],
                                    # source=adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"],
                                    source=adjoint_source,
                                    As=As,
                                    transpose_A=True
                                    if model_adj.train_field == "adj"
                                    else False,
                                )
                            ) / 2
                    elif name == "grad_loss":
                        # there is no need to distinguish the forward and adjoint field here
                        # since the gradient must combine both forward and adjoint field
                        if model_adj is not None:
                            adj_field_cal_grad = (
                                adjoint_field
                                if adjoint_field_err_corr is None
                                else adjoint_field_err_corr
                            )
                        else:
                            adj_field_cal_grad = fields_adj[
                                "fields_adj-wl-1.55-port-in_port_1-mode-1"
                            ][:, -2:, ...]
                        log_loss = (
                            weight
                            * log_criterion(
                                forward_fields=forward_field
                                if forward_field_err_corr is None
                                else forward_field_err_corr,
                                # forward_fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                                # backward_fields=field_solutions["field_solutions-wl-1.55-port-out_port_1-mode-1"][:, -2:, ...],
                                adjoint_fields=adj_field_cal_grad,
                                # adjoint_fields=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                                # backward_adjoint_fields = fields_adj['fields_adj-wl-1.55-port-in_port_1-mode-1'][:, -2:, ...],
                                target_gradient=gradient,
                                gradient_multiplier=field_normalizer,  # TODO the nomalizer should calculate from the forward field
                                # dr_mask=None,
                                dr_mask=design_region_mask,
                            )
                        )
                    elif name == "s_param_loss":
                        # there is also no need to distinguish the forward and adjoint field here
                        # the s_param_loss is calculated based on the forward field and there is no label for the adjoint field
                        assert (
                            model_fwd.train_field == "fwd"
                        ), "The s_param_loss is only calculated based on the forward field"
                        log_loss = (
                            weight
                            * log_criterion(
                                fields=forward_field
                                if forward_field_err_corr is None
                                else forward_field_err_corr,
                                # fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                                ht_m=ht_m["ht_m-wl-1.55-port-out_port_1-mode-1"],
                                et_m=et_m["et_m-wl-1.55-port-out_port_1-mode-1"],
                                monitor_slices=monitor_slices,  # 'port_slice-out_port_1_x', 'port_slice-out_port_1_y'
                                target_SParam=s_params["s_params-fwd_trans-1.55-1"],
                            )
                        )
                    elif name == "err_corr_Ez":
                        assert model_fwd.err_correction
                        log_loss = weight * log_criterion(
                            forward_field_err_corr[:, -2:, ...],
                            field_solutions[
                                "field_solutions-wl-1.55-port-in_port_1-mode-1"
                            ][:, -2:, ...]
                            if model_fwd.train_field == "fwd"
                            else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                :, -2:, ...
                            ],
                            torch.ones_like(forward_field_err_corr[:, -2:, ...]).to(
                                device
                            ),
                        )
                        if model_adj is not None:
                            assert model_adj.err_correction
                            log_loss = (
                                log_loss
                                + weight
                                * log_criterion(
                                    adjoint_field_err_corr[:, -2:, ...],
                                    fields_adj[
                                        "fields_adj-wl-1.55-port-in_port_1-mode-1"
                                    ][:, -2:, ...],
                                    torch.ones_like(
                                        adjoint_field_err_corr[:, -2:, ...]
                                    ).to(device),
                                )
                            ) / 2
                    elif name == "err_corr_Hx":
                        assert model_fwd.err_correction
                        log_loss = weight * log_criterion(
                            forward_field_err_corr[:, :2, ...],
                            field_solutions[
                                "field_solutions-wl-1.55-port-in_port_1-mode-1"
                            ][:, :2, ...]
                            if model_fwd.train_field == "fwd"
                            else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                :, :2, ...
                            ],
                            torch.ones_like(forward_field_err_corr[:, :2, ...]).to(
                                device
                            ),
                        )
                        if model_adj is not None:
                            assert model_adj.err_correction
                            log_loss = (
                                log_loss
                                + weight
                                * log_criterion(
                                    adjoint_field_err_corr[:, :2, ...],
                                    fields_adj[
                                        "fields_adj-wl-1.55-port-in_port_1-mode-1"
                                    ][:, :2, ...],
                                    torch.ones_like(
                                        adjoint_field_err_corr[:, :2, ...]
                                    ).to(device),
                                )
                            ) / 2
                    elif name == "err_corr_Hy":
                        assert model_fwd.err_correction
                        log_loss = weight * log_criterion(
                            forward_field_err_corr[:, 2:4, ...],
                            field_solutions[
                                "field_solutions-wl-1.55-port-in_port_1-mode-1"
                            ][:, 2:4, ...]
                            if model_fwd.train_field == "fwd"
                            else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                :, 2:4, ...
                            ],
                            torch.ones_like(forward_field_err_corr[:, 2:4, ...]).to(
                                device
                            ),
                        )
                        if model_adj is not None:
                            assert model_adj.err_correction
                            log_loss = (
                                log_loss
                                + weight
                                * log_criterion(
                                    adjoint_field_err_corr[:, 2:4, ...],
                                    fields_adj[
                                        "fields_adj-wl-1.55-port-in_port_1-mode-1"
                                    ][:, 2:4, ...],
                                    torch.ones_like(
                                        adjoint_field_err_corr[:, 2:4, ...]
                                    ).to(device),
                                )
                            ) / 2
                    elif name == "Hx_loss":
                        log_loss = weight * log_criterion(
                            forward_field[:, :2, ...],
                            field_solutions[
                                "field_solutions-wl-1.55-port-in_port_1-mode-1"
                            ][:, :2, ...]
                            if model_fwd.train_field == "fwd"
                            else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                :, :2, ...
                            ],
                            torch.ones_like(forward_field[:, :2, ...]).to(device),
                        )
                        if model_adj is not None:
                            log_loss = (
                                log_loss
                                + weight
                                * log_criterion(
                                    adjoint_field[:, :2, ...],
                                    fields_adj[
                                        "fields_adj-wl-1.55-port-in_port_1-mode-1"
                                    ][:, :2, ...],
                                    torch.ones_like(adjoint_field[:, :2, ...]).to(
                                        device
                                    ),
                                )
                            ) / 2
                    elif name == "Hy_loss":
                        log_loss = weight * log_criterion(
                            forward_field[:, 2:4, ...],
                            field_solutions[
                                "field_solutions-wl-1.55-port-in_port_1-mode-1"
                            ][:, 2:4, ...]
                            if model_fwd.train_field == "fwd"
                            else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                :, 2:4, ...
                            ],
                            torch.ones_like(forward_field[:, 2:4, ...]).to(device),
                        )
                        if model_adj is not None:
                            log_loss = (
                                log_loss
                                + weight
                                * log_criterion(
                                    adjoint_field[:, 2:4, ...],
                                    fields_adj[
                                        "fields_adj-wl-1.55-port-in_port_1-mode-1"
                                    ][:, 2:4, ...],
                                    torch.ones_like(adjoint_field[:, 2:4, ...]).to(
                                        device
                                    ),
                                )
                            ) / 2
                    loss = loss + log_loss
                    log_meters[name].update(log_loss.item())
    if "err_corr_Ez" in log_criterions.keys():
        loss_to_append = log_meters["err_corr_Ez"].avg
        if "err_corr_Hx" in log_criterions.keys():
            assert (
                "err_corr_Hy" in log_criterions.keys()
            ), "H field loss must appear together"
            loss_to_append += (
                log_meters["err_corr_Hx"].avg + log_meters["err_corr_Hy"].avg
            )
    elif "Hy_loss" in log_criterions.keys():
        assert "Hx_loss" in log_criterions.keys(), "H field loss must appear together"
        loss_to_append = (
            log_meters["Hx_loss"].avg + log_meters["Hy_loss"].avg + mse_meter.avg
        )
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
        filepath = os.path.join(dir_path, f"epoch_{epoch}_val")
        if model_fwd.err_correction:
            plot_fields(
                fields=forward_field.clone().detach(),
                ground_truth=field_solutions[
                    "field_solutions-wl-1.55-port-in_port_1-mode-1"
                ]
                if model_fwd.train_field == "fwd"
                else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath + f"_{model_fwd.train_field}.png",
            )
            plot_fields(
                fields=forward_field_err_corr.clone().detach(),
                ground_truth=field_solutions[
                    "field_solutions-wl-1.55-port-in_port_1-mode-1"
                ]
                if model_fwd.train_field == "fwd"
                else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath + f"_{model_fwd.train_field}_err_corr.png",
            )
            if model_adj is not None:
                plot_fields(
                    fields=adjoint_field.clone().detach(),
                    ground_truth=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                    filepath=filepath + "_adj.png",
                )
                plot_fields(
                    fields=adjoint_field_err_corr.clone().detach(),
                    ground_truth=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                    filepath=filepath + "_adj_err_corr.png",
                )
        else:
            plot_fields(
                fields=forward_field.clone().detach(),
                ground_truth=field_solutions[
                    "field_solutions-wl-1.55-port-in_port_1-mode-1"
                ]
                if model_fwd.train_field == "fwd"
                else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath + f"_{model_fwd.train_field}.png",
            )
            if model_adj is not None:
                plot_fields(
                    fields=adjoint_field.clone().detach(),
                    ground_truth=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                    filepath=filepath + "_adj.png",
                )


def test(
    model_fwd: nn.Module,
    model_adj: nn.Module,
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
    model_fwd.eval()
    if model_adj is not None:
        model_adj.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    log_meters = {name: AverageMeter(name) for name in log_criterions}
    with torch.no_grad(), DeterministicCtx(42):
        for batch_idx, (
            eps_map,
            adj_srcs,
            gradient,
            field_solutions,
            s_params,
            src_profiles,
            fields_adj,
            field_normalizer,
            design_region_mask,
            ht_m,
            et_m,
            monitor_slices,
            As,
        ) in enumerate(test_loader):
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
            # for key, field in incident_field.items():
            #     incident_field[key] = field.to(device, non_blocking=True)
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
                        item_to_add = (
                            torch.view_as_real(ht[i]).permute(1, 0).unsqueeze(0)
                        )
                        item_to_add = F.interpolate(
                            item_to_add,
                            size=size[i].item(),
                            mode="linear",
                            align_corners=True,
                        )
                        item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                        ht_list.append(
                            torch.view_as_complex(item_to_add).to(
                                device, non_blocking=True
                            )
                        )
                    ht_m[key] = ht_list
            for key, et in et_m.items():
                if key.endswith("-origin_size"):
                    continue
                else:
                    size = et_m[key + "-origin_size"]
                    et_list = []
                    for i in range(size.shape[0]):
                        item_to_add = (
                            torch.view_as_real(et[i]).permute(1, 0).unsqueeze(0)
                        )
                        item_to_add = F.interpolate(
                            item_to_add,
                            size=size[i].item(),
                            mode="linear",
                            align_corners=True,
                        )
                        item_to_add = item_to_add.squeeze(0).permute(1, 0).contiguous()
                        et_list.append(
                            torch.view_as_complex(item_to_add).to(
                                device, non_blocking=True
                            )
                        )
                    et_m[key] = et_list
            for key, A in As.items():
                As[key] = A.to(device, non_blocking=True)

            if mixup_fn is not None:
                eps_map, adj_src, gradient, field_solutions, s_params = mixup_fn(
                    eps_map, adj_src, gradient, field_solutions, s_params
                )

            with amp.autocast("cuda", enabled=False):
                # forward
                output = model_fwd(  # now only suppose that the output is the gradient of the field
                    eps_map,
                    src_profiles["source_profile-wl-1.55-port-in_port_1-mode-1"]
                    if model_fwd.train_field == "fwd"
                    else adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"],
                )
                if isinstance(output, tuple):
                    forward_Ez_field, forward_Ez_field_err_corr = output
                else:
                    forward_Ez_field = output
                    forward_field_err_corr = None
                with torch.enable_grad():
                    forward_field, adjoint_source = (
                        cal_total_field_adj_src_from_fwd_field(
                            Ez=forward_Ez_field,
                            eps=eps_map,
                            ht_ms=ht_m,
                            et_ms=et_m,
                            monitors=monitor_slices,
                            pml_mask=model_fwd.pml_mask,
                            from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                            return_adj_src=False
                            if (model_adj is None) or model_fwd.err_correction
                            else True,
                            sim=model_fwd.sim,
                        )
                    )
                    if adjoint_source is not None:
                        adjoint_source = adjoint_source.detach()
                if model_fwd.err_correction:
                    forward_field_err_corr, adjoint_source = (
                        cal_total_field_adj_src_from_fwd_field(
                            Ez=forward_Ez_field_err_corr,
                            eps=eps_map,
                            ht_ms=ht_m,
                            et_ms=et_m,
                            monitors=monitor_slices,
                            pml_mask=model_fwd.pml_mask,
                            from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                            return_adj_src=False if model_adj is None else True,
                            sim=model_fwd.sim,
                        )
                    )

                if model_adj is not None:
                    assert (
                        adjoint_source is not None
                    ), "The adjoint source should be calculated"

                    adjoint_source = adjoint_source * (
                        field_normalizer[
                            "field_adj_normalizer-wl-1.55-port-in_port_1-mode-1"
                        ].unsqueeze(1)
                    )

                    adjoint_output = model_adj(
                        eps_map,
                        adjoint_source,  # bs, H, W complex
                    )

                    if isinstance(adjoint_output, tuple):
                        adjoint_Ez_field, adjoint_Ez_field_err_corr = adjoint_output
                    else:
                        adjoint_Ez_field = adjoint_output
                        adjoint_field_err_corr = None

                    adjoint_field, _ = cal_total_field_adj_src_from_fwd_field(
                        Ez=adjoint_Ez_field,
                        eps=eps_map,
                        ht_ms=ht_m,
                        et_ms=et_m,
                        monitors=monitor_slices,
                        pml_mask=model_fwd.pml_mask,
                        from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                        return_adj_src=False,
                        sim=model_adj.sim,
                    )
                    if model_fwd.err_correction:
                        adjoint_field_err_corr, _ = (
                            cal_total_field_adj_src_from_fwd_field(
                                Ez=adjoint_Ez_field_err_corr,
                                eps=eps_map,
                                ht_ms=ht_m,
                                et_ms=et_m,
                                monitors=monitor_slices,
                                pml_mask=model_fwd.pml_mask,
                                from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                                return_adj_src=False,
                                sim=model_adj.sim,
                            )
                        )

                val_loss = criterion(
                    forward_field[:, -2:, ...],
                    field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][
                        :, -2:, ...
                    ]
                    if model_fwd.train_field == "fwd"
                    else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                        :, -2:, ...
                    ],
                    torch.ones_like(forward_field[:, -2:, ...]).to(device),
                )
                if model_adj is not None:
                    val_loss = (
                        val_loss
                        + criterion(
                            adjoint_field[:, -2:, ...],
                            fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                :, -2:, ...
                            ],
                            torch.ones_like(adjoint_field[:, -2:, ...]).to(device),
                        )
                    ) / 2
                mse_meter.update(val_loss.item())
                loss = val_loss
                for name, config in log_criterions.items():
                    log_criterion, weight = config
                    if name == "maxwell_residual_loss":
                        log_loss = weight * log_criterion(
                            Ez=forward_field,
                            # Ez=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                            source=src_profiles[
                                "source_profile-wl-1.55-port-in_port_1-mode-1"
                            ]
                            if model_fwd.train_field == "fwd"
                            else adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"],
                            As=As,
                            transpose_A=False
                            if model_fwd.train_field == "fwd"
                            else True,
                        )
                        if model_adj is not None:
                            log_loss = (
                                log_loss
                                + weight
                                * log_criterion(
                                    Ez=adjoint_field,
                                    # Ez=fields_adj['fields_adj-wl-1.55-port-in_port_1-mode-1'],
                                    # source=adj_srcs["adj_src-wl-1.55-port-in_port_1-mode-1"],
                                    source=adjoint_source,
                                    As=As,
                                    transpose_A=True
                                    if model_adj.train_field == "adj"
                                    else False,
                                )
                            ) / 2
                    elif name == "grad_loss":
                        # there is no need to distinguish the forward and adjoint field here
                        # since the gradient must combine both forward and adjoint field
                        log_loss = (
                            weight
                            * log_criterion(
                                forward_fields=forward_field
                                if forward_field_err_corr is None
                                else forward_field_err_corr,
                                # forward_fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                                # backward_fields=field_solutions["field_solutions-wl-1.55-port-out_port_1-mode-1"][:, -2:, ...],
                                adjoint_fields=adjoint_field
                                if adjoint_field_err_corr is None
                                else adjoint_field_err_corr,
                                # adjoint_fields=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                                # backward_adjoint_fields = fields_adj['fields_adj-wl-1.55-port-in_port_1-mode-1'][:, -2:, ...],
                                target_gradient=gradient,
                                gradient_multiplier=field_normalizer,  # TODO the nomalizer should calculate from the forward field
                                # dr_mask=None,
                                dr_mask=design_region_mask,
                            )
                        )
                    elif name == "s_param_loss":
                        # there is also no need to distinguish the forward and adjoint field here
                        # the s_param_loss is calculated based on the forward field and there is no label for the adjoint field
                        assert (
                            model_fwd.train_field == "fwd"
                        ), "The s_param_loss is only calculated based on the forward field"
                        log_loss = (
                            weight
                            * log_criterion(
                                fields=forward_field
                                if forward_field_err_corr is None
                                else forward_field_err_corr,
                                # fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                                ht_m=ht_m["ht_m-wl-1.55-port-out_port_1-mode-1"],
                                et_m=et_m["et_m-wl-1.55-port-out_port_1-mode-1"],
                                monitor_slices=monitor_slices,  # 'port_slice-out_port_1_x', 'port_slice-out_port_1_y'
                                target_SParam=s_params["s_params-fwd_trans-1.55-1"],
                            )
                        )
                    elif name == "err_corr_Ez":
                        assert model_fwd.err_correction
                        log_loss = weight * log_criterion(
                            forward_field_err_corr[:, -2:, ...],
                            field_solutions[
                                "field_solutions-wl-1.55-port-in_port_1-mode-1"
                            ][:, -2:, ...]
                            if model_fwd.train_field == "fwd"
                            else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                :, -2:, ...
                            ],
                            torch.ones_like(forward_field_err_corr[:, -2:, ...]).to(
                                device
                            ),
                        )
                        if model_adj is not None:
                            assert model_adj.err_correction
                            log_loss = (
                                log_loss
                                + weight
                                * log_criterion(
                                    adjoint_field_err_corr[:, -2:, ...],
                                    fields_adj[
                                        "fields_adj-wl-1.55-port-in_port_1-mode-1"
                                    ][:, -2:, ...],
                                    torch.ones_like(
                                        adjoint_field_err_corr[:, -2:, ...]
                                    ).to(device),
                                )
                            ) / 2
                    elif name == "err_corr_Hx":
                        assert model_fwd.err_correction
                        log_loss = weight * log_criterion(
                            forward_field_err_corr[:, :2, ...],
                            field_solutions[
                                "field_solutions-wl-1.55-port-in_port_1-mode-1"
                            ][:, :2, ...]
                            if model_fwd.train_field == "fwd"
                            else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                :, :2, ...
                            ],
                            torch.ones_like(forward_field_err_corr[:, :2, ...]).to(
                                device
                            ),
                        )
                        if model_adj is not None:
                            assert model_adj.err_correction
                            log_loss = (
                                log_loss
                                + weight
                                * log_criterion(
                                    adjoint_field_err_corr[:, :2, ...],
                                    fields_adj[
                                        "fields_adj-wl-1.55-port-in_port_1-mode-1"
                                    ][:, :2, ...],
                                    torch.ones_like(
                                        adjoint_field_err_corr[:, :2, ...]
                                    ).to(device),
                                )
                            ) / 2
                    elif name == "err_corr_Hy":
                        assert model_fwd.err_correction
                        log_loss = weight * log_criterion(
                            forward_field_err_corr[:, 2:4, ...],
                            field_solutions[
                                "field_solutions-wl-1.55-port-in_port_1-mode-1"
                            ][:, 2:4, ...]
                            if model_fwd.train_field == "fwd"
                            else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                :, 2:4, ...
                            ],
                            torch.ones_like(forward_field_err_corr[:, 2:4, ...]).to(
                                device
                            ),
                        )
                        if model_adj is not None:
                            assert model_adj.err_correction
                            log_loss = (
                                log_loss
                                + weight
                                * log_criterion(
                                    adjoint_field_err_corr[:, 2:4, ...],
                                    fields_adj[
                                        "fields_adj-wl-1.55-port-in_port_1-mode-1"
                                    ][:, 2:4, ...],
                                    torch.ones_like(
                                        adjoint_field_err_corr[:, 2:4, ...]
                                    ).to(device),
                                )
                            ) / 2
                    elif name == "Hx_loss":
                        log_loss = weight * log_criterion(
                            forward_field[:, :2, ...],
                            field_solutions[
                                "field_solutions-wl-1.55-port-in_port_1-mode-1"
                            ][:, :2, ...]
                            if model_fwd.train_field == "fwd"
                            else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                :, :2, ...
                            ],
                            torch.ones_like(forward_field[:, :2, ...]).to(device),
                        )
                        if model_adj is not None:
                            log_loss = (
                                log_loss
                                + weight
                                * log_criterion(
                                    adjoint_field[:, :2, ...],
                                    fields_adj[
                                        "fields_adj-wl-1.55-port-in_port_1-mode-1"
                                    ][:, :2, ...],
                                    torch.ones_like(adjoint_field[:, :2, ...]).to(
                                        device
                                    ),
                                )
                            ) / 2
                    elif name == "Hy_loss":
                        log_loss = weight * log_criterion(
                            forward_field[:, 2:4, ...],
                            field_solutions[
                                "field_solutions-wl-1.55-port-in_port_1-mode-1"
                            ][:, 2:4, ...]
                            if model_fwd.train_field == "fwd"
                            else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][
                                :, 2:4, ...
                            ],
                            torch.ones_like(forward_field[:, 2:4, ...]).to(device),
                        )
                        if model_adj is not None:
                            log_loss = (
                                log_loss
                                + weight
                                * log_criterion(
                                    adjoint_field[:, 2:4, ...],
                                    fields_adj[
                                        "fields_adj-wl-1.55-port-in_port_1-mode-1"
                                    ][:, 2:4, ...],
                                    torch.ones_like(adjoint_field[:, 2:4, ...]).to(
                                        device
                                    ),
                                )
                            ) / 2
                    loss = loss + log_loss
                    log_meters[name].update(log_loss.item())
    if "err_corr_Ez" in log_criterions.keys():
        loss_to_append = log_meters["err_corr_Ez"].avg
        if "err_corr_Hx" in log_criterions.keys():
            assert (
                "err_corr_Hy" in log_criterions.keys()
            ), "H field loss must appear together"
            loss_to_append += (
                log_meters["err_corr_Hx"].avg + log_meters["err_corr_Hy"].avg
            )
    elif "Hy_loss" in log_criterions.keys():
        assert "Hx_loss" in log_criterions.keys(), "H field loss must appear together"
        loss_to_append = (
            log_meters["Hx_loss"].avg + log_meters["Hy_loss"].avg + mse_meter.avg
        )
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
        filepath = os.path.join(dir_path, f"epoch_{epoch}_test")
        if model_fwd.err_correction:
            plot_fields(
                fields=forward_field.clone().detach(),
                ground_truth=field_solutions[
                    "field_solutions-wl-1.55-port-in_port_1-mode-1"
                ]
                if model_fwd.train_field == "fwd"
                else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath + f"_{model_fwd.train_field}.png",
            )
            plot_fields(
                fields=forward_field_err_corr.clone().detach(),
                ground_truth=field_solutions[
                    "field_solutions-wl-1.55-port-in_port_1-mode-1"
                ]
                if model_fwd.train_field == "fwd"
                else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath + f"_{model_fwd.train_field}_err_corr.png",
            )
            if model_adj is not None:
                plot_fields(
                    fields=adjoint_field.clone().detach(),
                    ground_truth=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                    filepath=filepath + "_adj.png",
                )
                plot_fields(
                    fields=adjoint_field_err_corr.clone().detach(),
                    ground_truth=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                    filepath=filepath + "_adj_err_corr.png",
                )
        else:
            plot_fields(
                fields=forward_field.clone().detach(),
                ground_truth=field_solutions[
                    "field_solutions-wl-1.55-port-in_port_1-mode-1"
                ]
                if model_fwd.train_field == "fwd"
                else fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath + f"_{model_fwd.train_field}.png",
            )
            if model_adj is not None:
                plot_fields(
                    fields=adjoint_field.clone().detach(),
                    ground_truth=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                    filepath=filepath + "_adj.png",
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

    model_fwd = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
    )
    lg.info(model_fwd)
    if model_fwd.train_field == "adj":
        assert (
            not configs.run.include_adjoint_NN
        ), "when only adj field is trained, we should not include another adjoint NN"

    if configs.run.include_adjoint_NN:
        model_adj = builder.make_model(
            device,
            int(configs.run.random_state) if int(configs.run.deterministic) else None,
        )
        model_adj.train_field = "adj"
        lg.info(model_adj)
    else:
        model_adj = None

    train_loader, validation_loader, test_loader = builder.make_dataloader()

    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )
    test_criterion = builder.make_criterion(
        configs.test_criterion.name, configs.test_criterion
    ).to(device)

    optimizer = builder.make_optimizer(
        [p for p in model_fwd.parameters() if p.requires_grad]
        + (
            [p for p in model_adj.parameters() if p.requires_grad]
            if model_adj is not None
            else []
        ),
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    if configs.optimizer.ALM:
        assert (
            configs.aux_criterion.maxwell_residual_loss.weight > 0
        ), "ALM is only used when maxwell_residual_loss is used"
        lambda_ = configs.optimizer.ALM_lambda
        mu = configs.optimizer.ALM_mu
        mu_growth = configs.optimizer.ALM_mu_growth
        constraint_tol = configs.optimizer.ALM_constraint_tol
    else:
        lambda_ = None
        mu = None
        mu_growth = None
        constraint_tol = None
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
    saver_fwd = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=10,
        metric_name="err",
        format="{:.4f}",
    )
    if model_adj is not None:
        saver_adj = BestKModelSaver(
            k=int(configs.checkpoint.save_best_model_k),
            descend=False,
            truncate=10,
            metric_name="err",
            format="{:.4f}",
        )

    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of fwd NN parameters: {count_parameters(model_fwd)}")
    if model_adj is not None:
        lg.info(f"Number of adj NN parameters: {count_parameters(model_adj)}")

    model_name = f"{configs.model.name}"
    checkpoint_fwd = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"
    if model_adj is not None:
        checkpoint_adj = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}_adj.pt"

    lg.info(f"Current fwd NN checkpoint: {checkpoint_fwd}")
    if model_adj is not None:
        lg.info(f"Current adj NN checkpoint: {checkpoint_adj}")

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
            and len(configs.checkpoint.restore_checkpoint_fwd) > 0
            and len(configs.checkpoint.restore_checkpoint_adj) > 0
        ):
            load_model(
                model_fwd,
                configs.checkpoint.restore_checkpoint_fwd,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )
            if model_adj is not None:
                load_model(
                    model_adj,
                    configs.checkpoint.restore_checkpoint_adj,
                    ignore_size_mismatch=int(configs.checkpoint.no_linear),
                )
            lg.info("Validate resumed model...")
            test(
                model_fwd,
                model_adj,
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
            quit()
        for epoch in range(1, int(configs.run.n_epochs) + 1):
            # single_batch_check(
            #     model_fwd,
            #     model_adj,
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
            lambda_, mu = train(
                model_fwd,
                model_adj,
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
                lambda_=lambda_,  # Lagrange multiplier
                mu=mu,  # Penalty coefficient
                mu_growth=mu_growth,  # Growth rate for penalty coefficient
                constraint_tol=constraint_tol,  # Tolerance for residual
            )

            if validation_loader is not None:
                validate(
                    model_fwd,
                    model_adj,
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
                    model_fwd,
                    model_adj,
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
                saver_fwd.save_model(
                    model_fwd,
                    lossv[-1],
                    epoch=epoch,
                    path=checkpoint_fwd,
                    save_model=False,
                    print_msg=True,
                )
                if model_adj is not None:
                    saver_adj.save_model(
                        model_adj,
                        lossv[-1],
                        epoch=epoch,
                        path=checkpoint_adj,
                        save_model=False,
                        print_msg=True,
                    )
        wandb.finish()
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
