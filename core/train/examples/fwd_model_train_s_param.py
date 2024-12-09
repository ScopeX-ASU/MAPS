import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "/home/pingchua/projects/MAPS")
)
sys.path.insert(0, project_root)

import argparse

import torch
import torch.amp as amp
import torch.nn as nn
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    set_torch_deterministic,
)

from core.train import builder
from core.train.models.utils import from_Ez_to_Hx_Hy
from core.train.trainer import PredTrainer
from core.utils import cal_total_field_adj_src_from_fwd_field
from thirdparty.pyutility.pyutils.config import train_configs as configs
import numpy as np
import time

class fwd_predictor(nn.Module):
    def __init__(self, model_fwd):
        super(fwd_predictor, self).__init__()
        self.model_fwd = model_fwd

    def forward(self, data):
        eps = data["eps_map"]
        src = data["src_profiles"]["source_profile-wl-1.55-port-in_port_1-mode-1"]
        s_params = self.model_fwd(eps, src)

        return {
            "s_params": s_params,
        }

class s_param_trainer(PredTrainer):

    def loss_calculation(self, output, data, task, crietrion_meter, aux_criterion_meter):
        """
        reload the loss calculation function to only calculate the s-parameter loss
        """
        s_param_pred = output['s_params']
        criterion = self.criterion
        regression_loss = criterion(
                    s_param_pred,
                    data['s_params']['s_params-fwd_trans-1.55-1'],
                    torch.ones_like(s_param_pred).to(self.device)
                )
        crietrion_meter.update(regression_loss.item())
        regression_loss = regression_loss * float(configs.criterion.weight)
        loss = regression_loss
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
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
    for key, value in configs.aux_criterion.items():
        value.weight = 0
    for key, value in configs.log_criterion.items():
        value.weight = 0
    print("this is the config: \n", configs, flush=True)
    if int(configs.run.deterministic) == True:
        set_torch_deterministic(int(configs.run.random_state))
    model_fwd = builder.make_model(device=device, **configs.model_fwd)
    print("this is the model: \n", model_fwd, flush=True)

    model = fwd_predictor(model_fwd)
    train_loader, validation_loader, test_loader = builder.make_dataloader()
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )
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
    print("log criterions used to monitor performance: ", log_criterions, flush=True)

    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=10,
        metric_name="err",
        format="{:.4f}",
    )

    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of NN parameters: {count_parameters(model)}")

    model_name = "dual_predictor"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"
    lg.info(f"Current fwd NN checkpoint: {checkpoint}")

    trainer = s_param_trainer(
        data_loaders={
            "train": train_loader,
            "val": validation_loader,
            "test": test_loader,
        },
        model=model,
        criterion=criterion,
        aux_criterion=aux_criterions,
        log_criterion=log_criterions,
        optimizer=optimizer,
        scheduler=scheduler,
        saver=saver,
        grad_scaler=grad_scaler,
        device=device,
    )
    # trainer.single_batch_check()
    for epoch in range(1, int(configs.run.n_epochs) + 1):
        trainer.train(
            data_loader=train_loader,
            task="train",
            epoch=epoch,
        )
        trainer.train(
            data_loader=validation_loader,
            task="val",
            epoch=epoch,
        )
        if epoch > int(configs.run.n_epochs) - 21:
            trainer.train(
                data_loader=test_loader,
                task="test",
                epoch=epoch,
            )
            trainer.save_model(epoch=epoch, checkpoint_path=checkpoint)


if __name__ == "__main__":
    main()
