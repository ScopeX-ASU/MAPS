'''
this file should only include the following functions that used to
1. build the model
2. build the optimizer
3. build the scheduler
4. build the criterion
that will be used in training the model

As for the dataloader, I think we should put it in the datasets submodules
we only request the train_loader, valid_loader and test_loader in the training process
and the dataloader should be returned by the datasets submodules
'''

import random
from typing import Tuple

import torch
import torch.nn as nn
from core.utils import train_configs as configs
from pyutils.datasets import get_dataset
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.sam import SAM
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device
from mmengine.registry import MODELS
from .models import *
from .datasets import *

from .utils import (
    DAdaptAdam,
    DistanceLoss,
    NL2NormLoss,
    NormalizedMSELoss,
    TemperatureScheduler,
    SharpnessScheduler,
    ResolutionScheduler,
    maskedNL2NormLoss,
    maskedNMSELoss,
    fab_penalty_ls_curve,
    fab_penalty_ls_gap,
    AspectRatioLoss,
    MaxwellResidualLoss,
    GradientLoss,
    SParamLoss,
    ComplexL1Loss,
)

__all__ = [
    "make_dataloader",
    "make_model",
    "make_weight_optimizer",
    "make_arch_optimizer",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]

def make_dataloader(
    name: str = None,
    splits=["train", "valid", "test"],
    train_noise_cfg=None,
    out_frames=None,
) -> Tuple[DataLoader, DataLoader]:
    name = (name or configs.dataset.name).lower()
    if name == "fdfd":
        train_dataset, validation_dataset, test_dataset = (
            (
                FDFDDataset(
                    device_type=configs.dataset.device_type,
                    root=configs.dataset.root,
                    split=split,
                    test_ratio=configs.dataset.test_ratio,
                    train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                    processed_dir=configs.dataset.processed_dir,
                )
                if split in splits
                else None
            )
            for split in ["train", "valid", "test"]
        )
    else:
        train_dataset, test_dataset = get_dataset(
            name,
            configs.dataset.img_height,
            configs.dataset.img_width,
            dataset_dir=configs.dataset.root,
            transform=configs.dataset.transform,
        )
        validation_dataset = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.run.batch_size,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
        prefetch_factor=8,
        persistent_workers=True,
        shuffle=int(configs.dataset.shuffle),
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=configs.run.batch_size,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
        prefetch_factor=8,
        persistent_workers=True,
        shuffle=int(configs.dataset.shuffle),
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=configs.run.batch_size,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
        prefetch_factor=8,
        persistent_workers=True,
        shuffle=int(configs.dataset.shuffle),
    )

    return train_loader, validation_loader, test_loader

def make_model(device: Device, random_state: int = None, **kwargs) -> nn.Module:
    model = MODELS.build(kwargs).to(device)
    model.reset_parameters(random_state)
    return model

def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "dadaptadam":
        optimizer = DAdaptAdam(
            params,
            lr=configs.lr,
            betas=getattr(configs, "betas", (0.9, 0.999)),
            weight_decay=configs.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.5),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.001),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            params,
            lr=configs.lr,  # for now, only the lr is tunable, others arguments just use the default value
            line_search_fn=configs.line_search_fn,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(
    optimizer: Optimizer, name: str = None, config_file: dict = {}
) -> Scheduler:
    name = (name or config_file.name).lower()
    if (
        name == "temperature"
    ):  # this temperature scheduler is a cosine annealing scheduler
        scheduler = TemperatureScheduler(
            initial_T=float(configs.temp_scheduler.lr),
            final_T=float(configs.temp_scheduler.lr_min),
            total_steps=int(configs.run.n_epochs),
        )
    elif name == "resolution":
        scheduler = ResolutionScheduler(
            initial_res=int(configs.res_scheduler.init_res),
            final_res=int(configs.res_scheduler.final_res),
            total_steps=int(configs.run.n_epochs),
        )
    elif name == "sharpness":
        scheduler = SharpnessScheduler(
            initial_sharp=float(configs.sharp_scheduler.init_sharp),
            final_sharp=float(configs.sharp_scheduler.final_sharp),
            total_steps=int(configs.run.n_epochs),
        )
    elif name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(configs.run.n_epochs),
            eta_min=float(configs.lr_scheduler.lr_min),
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.scheduler.lr_min,
            warmup_steps=int(configs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=configs.scheduler.lr_gamma
        )
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None, cfg=None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    cfg = cfg or configs.criterion
    if name == "mse":
        criterion = nn.MSELoss()
    elif name == "nmse":
        criterion = NormalizedMSELoss()
    elif name == "cmae":
        criterion = ComplexL1Loss(norm=cfg.norm)
    elif name == "curl_loss":
        criterion = fab_penalty_ls_curve(alpha=cfg.weight, min_feature_size=0.02)
    elif name == "gap_loss":
        criterion = fab_penalty_ls_gap(beta=1, min_feature_size=0.02)
    elif name == "nl2norm":
        criterion = NL2NormLoss()
    elif name == "masknl2norm":
        criterion = maskedNL2NormLoss(
            weighted_frames=cfg.weighted_frames,
            weight=cfg.weight,
            if_spatial_mask=cfg.if_spatial_mask,
        )
    elif name == "masknmse":
        criterion = maskedNMSELoss(
            weighted_frames=cfg.weighted_frames,
            weight=cfg.weight,
            if_spatial_mask=cfg.if_spatial_mask,
        )
    elif name == "distanceloss":
        criterion = DistanceLoss(min_distance=cfg.min_distance)
    elif name == "aspect_ratio_loss":
        criterion = AspectRatioLoss(
            aspect_ratio=cfg.aspect_ratio,
        )
    elif "err_corr" in name or (name == "hx_loss") or (name == "hy_loss"):
        criterion = NL2NormLoss()
    elif name == "maxwell_residual_loss":
        criterion = MaxwellResidualLoss(
            # wl_cen=cfg.wl_cen,
            # wl_width=cfg.wl_width,
            # n_wl=cfg.n_wl,
            # using_ALM=cfg.using_ALM,
        )
    elif name == "grad_loss":
        criterion = GradientLoss()
    elif name == "s_param_loss":
        criterion = SParamLoss()
    else:
        raise NotImplementedError(name)
    return criterion