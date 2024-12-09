import argparse
import os
from typing import Callable, Dict, Iterable
import torch.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
from thirdparty.pyutility.pyutils.config import train_configs as configs
from thirdparty.pyutility.pyutils.general import AverageMeter, logger as lg
from thirdparty.pyutility.pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from thirdparty.pyutility.pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler
import torch.fft
from core.train import builder
from core.utils import DeterministicCtx
import wandb
import datetime
import random
from core.utils import plot_fields, cal_total_field_adj_src_from_fwd_field
from thirdparty.ceviche.ceviche.constants import *
from core.train.models.utils import from_Ez_to_Hx_Hy
import math

class PredTrainer(object):
    """Base class for a trainer used to train a field predictor."""

    def __init__(
            self,
            data_loaders, 
            model, 
            criterion,
            aux_criterion,
            log_criterion, 
            optimizer, 
            scheduler, 
            saver,
            grad_scaler,
            device, 
        ):
        self.data_loaders = data_loaders
        self.model = model
        self.criterion = criterion
        self.aux_criterion = aux_criterion
        self.log_criterion = log_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.saver = saver
        self.grad_scaler = grad_scaler
        self.device = device

        self.lossv = []

    def train(
            self,
            data_loader,
            task,
            epoch,
        ):
        assert task.lower() in ["train", "val", "test"], f"Invalid task {task}"
        self.set_model_status(task)
        main_criterion_meter, aux_criterion_meter = self.build_meters(task)

        data_counter = 0
        total_data = len(data_loader.dataset)  # Total samples
        num_batches = len(data_loader)  # Number of batches

        iterator = iter(data_loader)
        local_step = 0
        while local_step < num_batches:
            try:
                data = next(iterator)
            except StopIteration:
                iterator = iter(data_loader)
                data = next(iterator)

            data = self.data_preprocess(data)
            if task.lower() != "train":
                with torch.no_grad():
                    output = self.forward(data)
            else:
                output = self.forward(data)
            loss = self.loss_calculation(
                output, 
                data, 
                task, 
                main_criterion_meter, 
                aux_criterion_meter,
            )
            if task.lower() == "train":
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()

            data_counter += data[list(data.keys())[0]].shape[0]

            if local_step % int(configs.run.log_interval) == 0 and task == "train":
                log = "{} Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} Regression Loss: {:.4e}".format(
                    task,
                    epoch,
                    data_counter,
                    total_data,
                    100.0 * data_counter / total_data,
                    loss.data.item(),
                    main_criterion_meter.avg,
                )
                for name, aux_meter in aux_criterion_meter.items():
                    log += f" {name}: {aux_meter.val:.4e}"
                lg.info(log)

            local_step += 1

        self.scheduler.step()
        error_summary = f"\n{task} Epoch {epoch} Regression Loss: {main_criterion_meter.avg:.4e}"
        for name, aux_meter in aux_criterion_meter.items():
            error_summary += f" {name}: {aux_meter.avg:.4e}"
        lg.info(error_summary)

        if task.lower() == "val":
            self.lossv.append(loss.data.item())

        if getattr(configs.plot, task, False) and (
            epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
        ):
            dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
            os.makedirs(dir_path, exist_ok=True)
            filepath = os.path.join(dir_path, f"epoch_{epoch}_{task}")
            self.result_visualization(data, output, filepath)

    def single_batch_check(self):
        task = "train"
        data_loader = self.data_loaders[task]
        self.set_model_status(task)
        main_criterion_meter, aux_criterion_meter = self.build_meters(task)

        num_batches = 100000

        iterator = iter(data_loader)
        data = next(iterator)
        data = self.data_preprocess(data)
        local_step = 0
        while local_step < num_batches:

            if task.lower() != "train":
                with torch.no_grad():
                    output = self.forward(data)
            else:
                output = self.forward(data)
            loss = self.loss_calculation(
                output, 
                data, 
                task, 
                main_criterion_meter, 
                aux_criterion_meter,
            )
            if task.lower() == "train":
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()


            if local_step % int(configs.run.log_interval) == 0 and task == "train":
                log = "{} Epoch: {} Loss: {:.4e} Regression Loss: {:.4e}".format(
                    "single_batch_check",
                    0,
                    loss.data.item(),
                    main_criterion_meter.avg,
                )
                for name, aux_meter in aux_criterion_meter.items():
                    log += f" {name}: {aux_meter.val:.4e}"
                lg.info(log)

            local_step += 1

        self.scheduler.step()
        lg.info(f"\nsingle batch check Epoch 0 Regression Loss: {main_criterion_meter.avg:.4e}")

        # if task.lower() == "val":
        #     self.lossv.append(loss.data.item())

        # if getattr(configs.plot, task, False) and (
        #     epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
        # ):
        #     dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        #     os.makedirs(dir_path, exist_ok=True)
        #     filepath = os.path.join(dir_path, f"epoch_{epoch}_{task}")
        #     self.result_visualization(data, output, filepath)

    def save_model(self, epoch, checkpoint_path):
        self.saver.save_model(
            self.model,
            self.lossv[-1],
            epoch=epoch,
            path=checkpoint_path,
            save_model=False,
            print_msg=True,
        )

    def data_preprocess(self, data):
        eps_map, adj_srcs, gradient, field_solutions, s_params, src_profiles, fields_adj, field_normalizer, design_region_mask, ht_m, et_m, monitor_slices, As = data
        eps_map = eps_map.to(self.device, non_blocking=True)
        gradient = gradient.to(self.device, non_blocking=True)
        for key, field in field_solutions.items():
            field = torch.view_as_real(field).permute(0, 1, 4, 2, 3)
            field = field.flatten(1, 2)
            field_solutions[key] = field.to(self.device, non_blocking=True)
        for key, s_param in s_params.items():
            s_params[key] = s_param.to(self.device, non_blocking=True)
        for key, adj_src in adj_srcs.items():
            adj_srcs[key] = adj_src.to(self.device, non_blocking=True)
        for key, src_profile in src_profiles.items():
            src_profiles[key] = src_profile.to(self.device, non_blocking=True)
        for key, field_adj in fields_adj.items():
            field_adj = torch.view_as_real(field_adj).permute(0, 1, 4, 2, 3)
            field_adj = field_adj.flatten(1, 2)
            fields_adj[key] = field_adj.to(self.device, non_blocking=True)
        for key, field_norm in field_normalizer.items():
            field_normalizer[key] = field_norm.to(self.device, non_blocking=True)
        for key, monitor_slice in monitor_slices.items():
            monitor_slices[key] = monitor_slice.to(self.device, non_blocking=True)
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
                    ht_list.append(torch.view_as_complex(item_to_add).to(self.device, non_blocking=True))
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
                    et_list.append(torch.view_as_complex(item_to_add).to(self.device, non_blocking=True))
                et_m[key] = et_list
        for key, A in As.items():
            As[key] = A.to(self.device, non_blocking=True)

        return_dict = {
            "eps_map": eps_map,
            "adj_srcs": adj_srcs,
            "gradient": gradient,
            "field_solutions": field_solutions,
            "s_params": s_params,
            "src_profiles": src_profiles,
            "fields_adj": fields_adj,
            "field_normalizer": field_normalizer,
            "design_region_mask": design_region_mask,
            "ht_m": ht_m,
            "et_m": et_m,
            "monitor_slices": monitor_slices,
            "As": As,
        }

        return return_dict

    def set_model_status(self, task):
        if task.lower() == "train":
            self.model.train()
        else:
            self.model.eval()

    def build_meters(self, task):
        main_criterion_meter = AverageMeter(configs.criterion.name)
        if task.lower() == "train":
            aux_criterion_meter = {name: AverageMeter(name) for name in self.aux_criterion}
        else:
            aux_criterion_meter = {name: AverageMeter(name) for name in self.log_criterion}

        return main_criterion_meter, aux_criterion_meter


    def forward(self, data):
        output = self.model(data)
        return output # the output has to be a dictionary in which the available keys must be 'forward_field' and 'adjoint_field' or others

    def loss_calculation(
            self, 
            output, 
            data, 
            task,
            crietrion_meter,
            aux_criterion_meter,
        ):
        assert 'forward_field' in list(output.keys()), "The output must contain the forward field"
        assert 'adjoint_field' in list(output.keys()), "The output must contain the adjoint field, even if the value is None"
        assert 'adjoint_source' in list(output.keys()), "The output must contain the adjoint source, ensure the value is None if the adjoint field is None"
        forward_field = output['forward_field']
        adjoint_field = output['adjoint_field']
        adjoint_source = output['adjoint_source']
        criterion = self.criterion
        if task.lower() == "train":
            aux_criterions = self.aux_criterion
        else:
            aux_criterions = self.log_criterion
        regression_loss = criterion(
                forward_field[:, -2:, ...], 
                data['field_solutions']["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                torch.ones_like(forward_field[:, -2:, ...]).to(self.device)
            )
        if adjoint_field is not None:
            regression_loss = (regression_loss + criterion(
                adjoint_field[:, -2:, ...],
                data['fields_adj']["fields_adj-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                torch.ones_like(adjoint_field[:, -2:, ...]).to(self.device)
            ))/2
        crietrion_meter.update(regression_loss.item())
        regression_loss = regression_loss * float(configs.criterion.weight)
        loss = regression_loss
        for name, config in aux_criterions.items():
            aux_criterion, weight = config
            if name == "maxwell_residual_loss":
                aux_loss = weight * aux_criterion(
                        Ez=forward_field, 
                        source=data['src_profiles']["source_profile-wl-1.55-port-in_port_1-mode-1"], 
                        As=data['As'],
                        transpose_A=False,
                    )
                if adjoint_field is not None:
                    aux_loss = (aux_loss + weight * aux_criterion(
                        Ez=adjoint_field, 
                        source=adjoint_source,
                        As=data['As'],
                        transpose_A=True,
                    ))/2
            elif name == "grad_loss":
                if adjoint_field is not None:
                    aux_loss = weight * aux_criterion(
                        forward_fields=forward_field,
                        # forward_fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                        # backward_fields=field_solutions["field_solutions-wl-1.55-port-out_port_1-mode-1"][:, -2:, ...],
                        adjoint_fields=adjoint_field,  
                        # adjoint_fields=fields_adj["fields_adj-wl-1.55-port-in_port_1-mode-1"][:, -2:, ...],
                        # backward_adjoint_fields = fields_adj['fields_adj-wl-1.55-port-in_port_1-mode-1'][:, -2:, ...],
                        target_gradient=data['gradient'],
                        gradient_multiplier=data['field_normalizer'], # TODO the nomalizer should calculate from the forward field
                        # dr_mask=None,
                        dr_mask=data['design_region_mask'],
                    )
                else:
                    raise ValueError("The adjoint field is None, the gradient loss cannot be calculated")
            elif name == "s_param_loss": 
                # there is also no need to distinguish the forward and adjoint field here
                # the s_param_loss is calculated based on the forward field and there is no label for the adjoint field
                aux_loss = weight * aux_criterion(
                    fields=forward_field, 
                    # fields=field_solutions["field_solutions-wl-1.55-port-in_port_1-mode-1"],
                    ht_m=data['ht_m']['ht_m-wl-1.55-port-out_port_1-mode-1'],
                    et_m=data['et_m']['et_m-wl-1.55-port-out_port_1-mode-1'],
                    monitor_slices=data['monitor_slices'], # 'port_slice-out_port_1_x', 'port_slice-out_port_1_y'
                    target_SParam=data['s_params']['s_params-fwd_trans-1.55-1'],
                )
            elif name == "Hx_loss":
                aux_loss = weight * aux_criterion(
                    forward_field[:, :2, ...],
                    data['field_solutions']["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, :2, ...],
                    torch.ones_like(forward_field[:, :2, ...]).to(self.device)
                )
                if adjoint_field is not None:
                    aux_loss = (aux_loss + weight * aux_criterion(
                        adjoint_field[:, :2, ...],
                        data['fields_adj']["fields_adj-wl-1.55-port-in_port_1-mode-1"][:, :2, ...],
                        torch.ones_like(adjoint_field[:, :2, ...]).to(self.device)
                    ))/2
            elif name == "Hy_loss":
                aux_loss = weight * aux_criterion(
                    forward_field[:, 2:4, ...],
                    data['field_solutions']["field_solutions-wl-1.55-port-in_port_1-mode-1"][:, 2:4, ...],
                    torch.ones_like(forward_field[:, 2:4, ...]).to(self.device)
                )
                if adjoint_field is not None:
                    aux_loss = (aux_loss + weight * aux_criterion(
                        adjoint_field[:, 2:4, ...],
                        data['fields_adj']["fields_adj-wl-1.55-port-in_port_1-mode-1"][:, 2:4, ...],
                        torch.ones_like(adjoint_field[:, 2:4, ...]).to(self.device)
                    ))/2
            aux_criterion_meter[name].update(aux_loss.item()) # record the aux loss first
            loss = loss + aux_loss

        return loss

    def result_visualization(self, data, output, filepath):
        forward_field = output['forward_field']
        adjoint_field = output['adjoint_field']
        plot_fields(
            fields=forward_field.clone().detach(),
            ground_truth=data['field_solutions']["field_solutions-wl-1.55-port-in_port_1-mode-1"],
            filepath=filepath + f"_fwd.png",
        )
        if adjoint_field is not None:
            plot_fields(
                fields=adjoint_field.clone().detach(),
                ground_truth=data['fields_adj']["fields_adj-wl-1.55-port-in_port_1-mode-1"],
                filepath=filepath + "_adj.png",
            )