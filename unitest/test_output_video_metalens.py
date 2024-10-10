import argparse
import datetime
import os
from typing import List

import torch
import torch.cuda.amp as amp
import torch.fft
import matplotlib.pyplot as plt
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

eps_sio2 = 1.44**2
eps_si = 3.48**2

if __name__ == "__main__":

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

    # Extract parameters from model.parameters()
    params_from_parameters = set(p for p in model.parameters() if p.requires_grad)

    # Extract parameters from model.named_parameters()
    params_from_named_parameters = set(p for name, p in model.named_parameters())

    # Check if the two sets are the same
    if params_from_parameters == params_from_named_parameters:
        lg.info("The sets of parameters from model.parameters() and model.named_parameters() are the same.")
    else:
        raise ValueError("The sets of parameters from model.parameters() and model.named_parameters() are different.")

    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )
    aux_criterions = {
        name: [builder.make_criterion(name, cfg=config), float(config.weight)]
        for name, config in configs.aux_criterion.items()
        if float(config.weight) > 0
    }


    optimizer = builder.make_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    lr_scheduler = builder.make_scheduler(optimizer, config_file=configs.lr_scheduler)
    sharp_scheduler = builder.make_scheduler(
        optimizer, name="sharpness", config_file=configs.sharp_scheduler
    )

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

    group = f"{datetime.date.today()}"
    configs.run.pid = os.getpid()

    lossv = [0]
    epoch = 0

    try:
        if (
            int(configs.checkpoint.resume)
            and len(configs.checkpoint.restore_checkpoint) > 0
        ):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )
        else:
            lg.info("No checkpoint to restore, output the initial model video")
        model.output_video(path=configs.plot.output_video_path + ".h5", sharpness=configs.sharp_scheduler.lr_max)

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")
