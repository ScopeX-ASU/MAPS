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
class TimerCtx:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

class fwd_predictor(nn.Module):
    def __init__(self, model_fwd):
        super(fwd_predictor, self).__init__()
        self.model_fwd = model_fwd

    def forward(self, data):
        eps = data["eps_map"]
        src = data["src_profiles"]["source_profile-wl-1.55-port-in_port_1-mode-1"]
        x_fwd = self.model_fwd(eps, src)
        with torch.no_grad():
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model_fwd(eps, src)
        torch.cuda.synchronize()
        time_per_run = []
        for _ in range(10):
            with torch.no_grad():
                with TimerCtx() as t:
                    for _ in range(3):
                        x_fwd = self.model_fwd(eps, src).detach()
                    torch.cuda.synchronize()
            inference_time = t.interval / 3
            time_per_run.append(inference_time)
        time_per_run = np.array(time_per_run)
        print(f"Average inference time: {time_per_run.mean()}s")
        quit()

        forward_field, _ = cal_total_field_adj_src_from_fwd_field(
            Ez=x_fwd,
            eps=eps,
            ht_ms=data["ht_m"],
            et_ms=data["et_m"],
            monitors=data["monitor_slices"],
            pml_mask=self.model_fwd.pml_mask,
            from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
            return_adj_src=False,
            sim=self.model_fwd.sim,
        )
        return {
            "forward_field": forward_field,
            "adjoint_field": None,
            "adjoint_source": None,
        }


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

    trainer = PredTrainer(
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
