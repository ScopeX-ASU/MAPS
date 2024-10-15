#!/usr/bin/env python
# coding=UTF-8
import argparse
import datetime
import os
from typing import List

import mlflow
import torch
import torch.cuda.amp as amp
import wandb
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
from pyutils.typing import Criterion, Optimizer, Scheduler

from core import builder
from core.models.layers import *


def train(
    model,
    optimizer: Optimizer,
    lr_scheduler: Scheduler,
    sharp_scheduler: Scheduler,
    resolution_scheduler: Scheduler,
    epoch: int,
    plot: bool = False,
    grad_scaler=None,
    lossv: List = [],
) -> None:
    torch.autograd.set_detect_anomaly(True)
    model.train()
    step = epoch
    # aux_meters = {name: AverageMeter(name) for name in aux_criterions}

    if plot and epoch == 1:
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        # filepath = os.path.join(dir_path, f"epoch_{epoch - 1}_train.png")
        with torch.no_grad():
            results = model(
                configs.sharp_scheduler.final_sharp
            )
            opt_obj = results["obj"]
            log = "Train Epoch: {} | Loss: {:.4e}".format(
                epoch - 1,
                opt_obj.item(),
            )
            log += f" FoM: {opt_obj.item()}"
            for key, value in results["breakdown"].items():
                log += f" {key}: {value['value']}"
            lg.info(log)

            model.plot(
                eps_map=model._eps_map,
                obj=results["breakdown"]["fwd_trans"]["value"],
                plot_filename=f"metacoupler_opt_step_{epoch-1}_fwd.png",
                field_key=("in_port_1", 1.55, 1),
                field_component="Ez",
                in_port_name = "in_port_1",
                exclude_port_names=["refl_port_2"],
            )
            if "coupler" in configs.model.name.lower():
                model.plot(
                    eps_map=model._eps_map,
                    obj=results["breakdown"]["bwd_trans"]["value"],
                    plot_filename=f"metacoupler_opt_step_{epoch-1}_bwd.png",
                    field_key=("out_port_1", 1.55, 1),
                    field_component="Ez",
                    in_port_name = "out_port_1",
                    exclude_port_names=["refl_port_1"],
                )

    with amp.autocast(enabled=grad_scaler._enabled):
        sharpness = sharp_scheduler.step()
        resolution = resolution_scheduler.step()
        results = model(sharpness)
        opt_obj = results["obj"]
        loss = (
            -opt_obj
        )  # it should be minus fom, so that the gradient descent on loss could be the gradient ascent on fom


    grad_scaler.scale(loss).backward(retain_graph=True)
    # # print the gradient of the parameters
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(name, param.grad, flush=True)
    #     else:
    #         print(name, "None", flush=True)
    grad_scaler.unscale_(optimizer)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    step += 1
    lossv.append(loss.item())

    log = "Train Epoch: {} | Loss: {:.4e}".format(
        epoch,
        loss.data.item(),
    )
    log += f" FoM: {opt_obj.item()}"
    for key, value in results["breakdown"].items():
        log += f" {key}: {value['value']}"
        wandb.log(
            {
                key: value["value"],
            },
        )
    lg.info(log)

    wandb.log(
        {
            "FoM_train": opt_obj.item(),
        },
    )
    lr_scheduler.step()
    wandb.log(
        {
            "epoch": epoch,
            "lr": get_learning_rate(optimizer),
        },
    )
    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        model.plot(
            eps_map=model._eps_map,
            obj=results["breakdown"]["fwd_trans"]["value"],
            plot_filename=f"metacoupler_opt_step_{epoch}_fwd.png",
            field_key=("in_port_1", 1.55, 1),
            field_component="Ez",
            in_port_name = "in_port_1",
            exclude_port_names=["refl_port_2"],
        )
        if "coupler" in configs.model.name:
            model.plot(
                eps_map=model._eps_map,
                obj=results["breakdown"]["bwd_trans"]["value"],
                plot_filename=f"metacoupler_opt_step_{epoch}_bwd.png",
                field_key=("out_port_1", 1.55, 1),
                field_component="Ez",
                in_port_name = "out_port_1",
                exclude_port_names=["refl_port_1"],
            )


def test(
    model,
    optimizer: Optimizer,
    sharp_scheduler: Scheduler,
    resolution_scheduler: Scheduler,
    epoch: int,
    aux_criterions: Criterion,
    lossv: List,
    plot: bool = False,
) -> None:
    # the model here is a photonic crystal device
    # first we will use the meep to calculate the transmission efficiency of the device
    # and then we use the adjoint method to calculate the gradient of the transmission efficiency w.r.t the permittivity
    # then we use the autograd to calculate parital permittivity over partial design variables
    # and then we use the optimizer to update the permittivity
    # the scheduler is used to update the learning rate and the temperature of the binarization
    # TODO finish the training frame of the photonic crystal device
    torch.autograd.set_detect_anomaly(True)
    model.eval()
    step = epoch
    distance_meter = AverageMeter("distance")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}

    with torch.no_grad():
        sharpness = sharp_scheduler.get_sharpness()
        resolution = resolution_scheduler.get_resolution()
        output = model(sharpness, resolution)
        eval_obj = output["eval_obj"]
        ref_power = output["reflection"]
        loss = (
            -eval_obj + ref_power
        )  # it should be minus fom, so that the gradient descent on loss could be the gradient ascent on fom
        # bu it turns out when minus fom, the fom drops dramatically, try add fom see the result
        aux_loss = None
        for name, config in aux_criterions.items():
            aux_criterion, weight = config
            if name == "curl_loss":
                aux_loss = weight * aux_criterion(output)
            elif name == "gap_loss":
                aux_loss = weight * aux_criterion(output)
            elif name == "aspect_ratio_loss":
                aux_loss = weight * aux_criterion(output)
            else:
                raise ValueError(f"auxiliary criterion {name} is not supported")
            loss = loss + aux_loss
            aux_meters[name].update(aux_loss.item())

    step += 1

    log = "Test Epoch: {} | Loss: {:.4e}".format(
        epoch,
        loss.data.item(),
    )
    log += (
        f" FoM: {eval_obj.item() if isinstance(eval_obj, torch.Tensor) else eval_obj}"
    )
    log += f" Ref: {ref_power.item()}"
    if aux_loss is not None:
        for name, meter in aux_meters.items():
            log += f" {name}: {meter.avg:.4e}"
    lg.info(log)

    mlflow.log_metrics({"train_loss": loss.item()}, step=step)
    wandb.log(
        {
            "FoM_test": eval_obj.item()
            if isinstance(eval_obj, torch.Tensor)
            else eval_obj,
            "Ref_test": ref_power.item(),
            "Aux_loss_Test": aux_loss.item() if aux_loss is not None else 0,
            "sharpness": sharpness,
            "global_step": step,
        },
    )
    # lg.info(f"Train Regression Loss: {avg_regression_loss:.4e}")
    wandb.log(
        {
            "epoch": epoch,
            "lr": get_learning_rate(optimizer),
        },
    )
    lossv.append(eval_obj)
    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_test.png")
        # plot the permittivity using the matplot lib
        # and save the figure to the file path
        if configs.model.adjoint_mode == "fdtd":
            ax = model.device.opt.sim.plot2D(plot_sources_flag=False)
            fig = ax.figure
            fig.savefig(filepath, dpi=300)
        else:
            raise ValueError(
                f"adjoint mode {configs.model.adjoint_mode} is not supported"
            )
        if (
            configs.plot.phase_profile
            and epoch % configs.plot.phase_intensity_profile_interval == 0
        ):
            model.output_phase_profile(
                path=filepath[:-4],
                resolution=resolution,
            )
        if (
            configs.plot.intensity_profile
            and epoch % configs.plot.phase_intensity_profile_interval == 0
        ):
            model.output_light_intensity(
                path=filepath[:-4],
                grating=output["grating"],
                resolution=resolution,
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
        operation_device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        operation_device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic(int(configs.run.random_state))
        
    device = builder.make_device(
        device=operation_device,
    )

    model = builder.make_model(
        device=operation_device,
        random_state=int(configs.run.random_state) if int(configs.run.deterministic) else None,
        optDevice=device,
    )
    lg.info(model)

    # ---------- these two criterion are not needed here ------------
    # criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
    #     device
    # )
    # aux_criterions = {
    #     name: [builder.make_criterion(name, cfg=config), float(config.weight)]
    #     for name, config in configs.aux_criterion.items()
    #     if float(config.weight) > 0
    # }
    # ---------------------------------------------------------------

    optimizer = builder.make_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    lr_scheduler = builder.make_scheduler(optimizer, config_file=configs.lr_scheduler)
    sharp_scheduler = builder.make_scheduler(
        optimizer, name="sharpness", config_file=configs.sharp_scheduler
    )
    res_scheduler = builder.make_scheduler(
        optimizer, name="resolution", config_file=configs.res_scheduler
    )

    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=10,
        metric_name="FoM",
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

    lossv = [0]
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

        for epoch in range(1, int(configs.run.n_epochs) + 1):
            train(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                sharp_scheduler=sharp_scheduler,
                resolution_scheduler=res_scheduler,
                epoch=epoch,
                plot=configs.plot.train,
                grad_scaler=grad_scaler,
                lossv=lossv,
            )
            # for now there is no need for the test since the res is fixed to the config value
            # no descrpancy between the train and test
            # test(
            #     model=model,
            #     optimizer=optimizer,
            #     sharp_scheduler=sharp_scheduler,
            #     resolution_scheduler=res_scheduler,
            #     epoch=epoch,
            #     aux_criterions=aux_criterions,
            #     lossv=lossv,
            #     plot=configs.plot.test,
            # )
            if epoch > int(configs.run.n_epochs) - 21:
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
