import os

import torch
import torch.amp as amp
import torch.fft
import torch.nn.functional as F
from core.utils import train_configs as configs
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
from core.train import builder
from core.utils import DeterministicCtx, print_stat
import wandb
import datetime
import random
from core.utils import plot_fields, cal_total_field_adj_src_from_fwd_field
from thirdparty.ceviche.ceviche.constants import *
from core.train.models.utils import from_Ez_to_Hx_Hy
import math

def data_preprocess(data, device):
    input_slice, wavelength, mode, temp, eps_map, adj_src, gradient, fwd_field, s_params, src_profile, adj_field, field_adj_normalizer, design_region_mask, ht_m, et_m, monitor_slices, A, data_file_path = data

    eps_map = eps_map.to(device, non_blocking=True)
    gradient = gradient.to(device, non_blocking=True)
    
    fwd_field = torch.view_as_real(fwd_field).permute(0, 1, 4, 2, 3)
    fwd_field = fwd_field.flatten(1, 2)
    fwd_field = fwd_field.to(device, non_blocking=True) # (batch_size, 6, H, W)

    for key, s_param in s_params.items():
        s_params[key] = s_param.to(device, non_blocking=True)

    adj_src = adj_src.to(device, non_blocking=True)

    src_profile = src_profile.to(device, non_blocking=True)

    adj_field = torch.view_as_real(adj_field).permute(0, 1, 4, 2, 3)
    adj_field = adj_field.flatten(1, 2)
    adj_field = adj_field.to(device, non_blocking=True) # (batch_size, 6, H, W)

    field_adj_normalizer = field_adj_normalizer.to(device, non_blocking=True)

    for key, monitor_slice in monitor_slices.items():
        monitor_slices[key] = monitor_slice.to(device, non_blocking=True)

    for key, ht in ht_m.items():
        ht_m[key] = ht.to(device, non_blocking=True)
    for key, et in et_m.items():
        et_m[key] = et.to(device, non_blocking=True)
    for key, value in A.items():
        A[key] = value.to(device, non_blocking=True)

    opt_cfg_file_path = []
    for filepath in data_file_path:
        if "perturbed" in filepath:
            config_path = filepath.split("_opt_step")[0] + "_perturbed_" + filepath.split("_perturbed_")[1].split("-")[0] + ".yml"
            opt_cfg_file_path.append(config_path)
        else:
            opt_cfg_file_path.append(filepath.split("_opt_step")[0] + ".yml")

    return_dict = {
        "eps_map": eps_map,
        "adj_src": adj_src,
        "gradient": gradient,
        "fwd_field": fwd_field,
        "s_params": s_params,
        "src_profile": src_profile,
        "adj_field": adj_field,
        "field_normalizer": field_adj_normalizer,
        "design_region_mask": design_region_mask,
        "ht_m": ht_m,
        "et_m": et_m,
        "monitor_slices": monitor_slices,
        "A": A,
        "opt_cfg_file_path": opt_cfg_file_path,
        "input_slice": input_slice,
        "wavelength": wavelength.to(device, non_blocking=True),
        "mode": mode.to(device, non_blocking=True),
        "temp": temp.to(device, non_blocking=True),
    }

    return return_dict

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
            fp16 = False,
            n_sample=None,
        ):
        assert task.lower() in ["train", "val", "test"], f"Invalid task {task}"
        self.set_model_status(task)
        if task.lower() == "test":
            set_torch_deterministic(42)
        main_criterion_meter, aux_criterion_meter = self.build_meters(task)

        data_counter = 0
        total_data = len(data_loader.dataset)  # Total samples
        num_batches = len(data_loader)  # Number of batches
        if n_sample is not None:
            num_batches = min(int(n_sample / data_loader.batch_size), num_batches)
            total_data = num_batches * data_loader.batch_size

        iterator = iter(data_loader)
        local_step = 0
        while local_step < num_batches:
            try:
                data = next(iterator)
            except StopIteration:
                iterator = iter(data_loader)
                data = next(iterator)

            data = data_preprocess(data, self.device)
            with amp.autocast('cuda', enabled=self.grad_scaler._enabled):
                if task.lower() != "train":
                    with torch.no_grad():
                        output = self.forward(data, epoch)
                else:
                    output = self.forward(data, epoch)
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
        error_summary = (
            f"\n{task} Epoch {epoch} Regression Loss: {main_criterion_meter.avg:.4e}"
        )
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
        data = data_preprocess(data, self.device)
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
        lg.info(
            f"\nsingle batch check Epoch 0 Regression Loss: {main_criterion_meter.avg:.4e}"
        )

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

    def set_model_status(self, task):
        if task.lower() == "train":
            self.model.train()
        else:
            self.model.eval()

    def build_meters(self, task):
        main_criterion_meter = AverageMeter(configs.criterion.name)
        if task.lower() == "train":
            aux_criterion_meter = {
                name: AverageMeter(name) for name in self.aux_criterion
            }
        else:
            aux_criterion_meter = {
                name: AverageMeter(name) for name in self.log_criterion
            }

        return main_criterion_meter, aux_criterion_meter


    def forward(self, data, epoch):
        output = self.model(data, epoch)
        return output # the output has to be a dictionary in which the available keys must be 'forward_field' and 'adjoint_field' or others

    def loss_calculation(
            self, 
            output, 
            data, 
            task,
            crietrion_meter,
            aux_criterion_meter,
        ):
    # return_dict = {
    #     "eps_map": eps_map,
    #     "adj_src": adj_src,
    #     "gradient": gradient,
    #     "fwd_field": fwd_field,
    #     "s_params": s_params,
    #     "src_profile": src_profile,
    #     "adj_field": adj_field,
    #     "field_normalizer": field_adj_normalizer,
    #     "design_region_mask": design_region_mask,
    #     "ht_m": ht_m,
    #     "et_m": et_m,
    #     "monitor_slices": monitor_slices,
    #     "A": A,
    #     "opt_cfg_file_path": opt_cfg_file_path,
    #     "input_slice": input_slice,
    #     "wavelength": wavelength,
    #     "mode": mode,
    #     "temp": temp,
    # }
        # for forward prediction, the output must contain the forward field and this should be a dictionary: (wl, mode, temp, in_port_name, out_port_name) -> forward field
        assert 'forward_field' in list(output.keys()), "The output must contain the forward field"
        assert 'adjoint_field' in list(output.keys()), "The output must contain the adjoint field, even if the value is None"
        assert 'adjoint_source' in list(output.keys()), "The output must contain the adjoint source, ensure the value is None if the adjoint field is None"
        forward_field = output['forward_field']
        adjoint_field = output['adjoint_field']
        adjoint_source = output['adjoint_source']
        s_params = output.get('s_params', None) # if it is not none, means that we have an S-param head to directly predict the S-params
        criterion = self.criterion
        if task.lower() == "train":
            aux_criterions = self.aux_criterion
        else:
            aux_criterions = self.log_criterion
        regression_loss = criterion(
                forward_field[:, -2:, ...], 
                data['fwd_field'][:, -2:, ...],
                torch.ones_like(forward_field[:, -2:, ...]).to(self.device)
            )
        if adjoint_field is not None:
            adjoint_loss = criterion(
                    adjoint_field[:, -2:, ...], 
                    data['adj_field'][:, -2:, ...],
                    torch.ones_like(adjoint_field[:, -2:, ...]).to(self.device)
                )
            regression_loss = (regression_loss + adjoint_loss)/2
        crietrion_meter.update(regression_loss.item())
        regression_loss = regression_loss * float(configs.criterion.weight)
        loss = regression_loss
        for name, config in aux_criterions.items():
            aux_criterion, weight = config
            if name == "maxwell_residual_loss":
                aux_loss = weight * aux_criterion(
                        forward_field, 
                        # data['fwd_field'][:, -2:, ...],
                        data['src_profile'],
                        data['A'],
                        transpose_A=False,
                        wl=data['wavelength'],
                        field_normalizer=data['field_normalizer'],
                    ) 
                if adjoint_field is not None:
                    adjoint_loss = weight * aux_criterion(
                            adjoint_field, 
                            # data['adj_field'][:, -2:, ...], # label is normalized field
                            adjoint_source, # b_adj, not normalized to 1e-8
                            data['A'],
                            transpose_A=True,
                            wl=data['wavelength'],
                            field_normalizer=data['field_normalizer'],
                        )
                    aux_loss = (aux_loss + adjoint_loss)/2
                    # print("maxwell aux_loss: ", aux_loss, flush=True)
            elif name == "grad_loss":
                if adjoint_field is not None:
                    aux_loss = weight * aux_criterion(
                        forward_fields=forward_field,
                        adjoint_fields=adjoint_field,  
                        # forward_fields=data["fwd_field"][:, -2:, ...],
                        # adjoint_fields=data["adj_field"][:, -2:, ...],
                        target_gradient=data['gradient'],
                        gradient_multiplier=data['field_normalizer'],
                        dr_mask=data['design_region_mask'],
                        wl = data['wavelength'],
                    )
                    # print("grad aux_loss: ", aux_loss, flush=True)
                else:
                    raise ValueError("The adjoint field is None, the gradient loss cannot be calculated")
            elif name == "grad_similarity_loss":
                if adjoint_field is not None:
                    aux_loss = weight * aux_criterion(
                        # forward_fields=data["fwd_field"][:, -2:, ...],
                        # adjoint_fields=data["adj_field"][:, -2:, ...],
                        forward_fields=forward_field,
                        adjoint_fields=adjoint_field,  
                        target_gradient=data['gradient'],
                        dr_mask=data['design_region_mask'],
                    )
                    # print("grad similarity aux_loss: ", aux_loss, flush=True)
                else:
                    raise ValueError("The adjoint field is None, the gradient loss cannot be calculated")
            elif name == "s_param_loss": 
                aux_loss = weight * aux_criterion(
                        fields=forward_field, 
                        # fields=data['fwd_field'],
                        ht_m=data['ht_m'],
                        et_m=data['et_m'],
                        monitor_slices=data['monitor_slices'],
                        target_SParam=data['s_params'],
                        opt_cfg_file_path=data['opt_cfg_file_path'],
                        mode=data['mode'],
                        temp=data['temp'],
                        wl=data['wavelength'],
                        src_in_slice_name=data['input_slice'],
                    )
            elif name == "direct_s_param_loss":
                # in this loss function, we don't calculate the S-params from the forward field, 
                # we directly compare the S-params from the prediction and the GT S-params from the data
                assert s_params is not None, "The s_params should not be None"
                aux_loss = weight * aux_criterion(
                        s_params, 
                        data['s_params'],
                    )
            elif name == "Hx_loss":
                aux_loss = weight * aux_criterion(
                        forward_field[:, :2, ...],
                        data['fwd_field'][:, :2, ...],
                        torch.ones_like(forward_field[:, :2, ...]).to(self.device)
                    )
                if adjoint_field is not None:
                    adjoint_loss = weight * aux_criterion(
                            adjoint_field[:, :2, ...],
                            # data['adj_field'],
                            data['adj_field'][:, :2, ...],
                            torch.ones_like(adjoint_field[:, :2, ...]).to(self.device)
                        )
                    aux_loss = (aux_loss + adjoint_loss)/2
            elif name == "Hy_loss":
                aux_loss = weight * aux_criterion(
                        forward_field[:, 2:4, ...],
                        data['fwd_field'][:, 2:4, ...],
                        torch.ones_like(forward_field[:, 2:4, ...]).to(self.device)
                    )
                if adjoint_field is not None:
                    adjoint_loss = weight * aux_criterion(
                            adjoint_field[:, 2:4, ...],
                            # data['adj_field'],
                            data['adj_field'][:, 2:4, ...],
                            torch.ones_like(adjoint_field[:, 2:4, ...]).to(self.device)
                        )
                    aux_loss = (aux_loss + adjoint_loss)/2
            aux_criterion_meter[name].update(aux_loss.item()) # record the aux loss first
            loss = loss + aux_loss

        return loss

    def result_visualization(self, data, output, filepath):
        forward_field = output['forward_field']
        adjoint_field = output['adjoint_field']
        plot_fields(
            fields=forward_field.detach(),
            ground_truth=data['fwd_field'],
            filepath=filepath + f"-fwd.png",
        )
        if adjoint_field is not None:
            plot_fields(
                fields=adjoint_field.clone().detach(),
                ground_truth=data['adj_field'],
                filepath=filepath + f"-adj.png",
            )