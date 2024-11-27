import random
from typing import Tuple

import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.sam import SAM
from pyutils.typing import Optimizer, Scheduler
from torch.types import Device

from core.models import *
from core.datasets import *

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
    "make_model",
    "make_weight_optimizer",
    "make_arch_optimizer",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]

def make_device(device: Device):
    device_to_opt = eval(configs.model.device_type)(
        sim_cfg=configs.model.sim_cfg,
        device=device,
    )
    return device_to_opt

def make_model(device: Device, random_state: int = None, **kwargs) -> nn.Module:
    if (
        "repara_phc_1x1" in configs.model.name.lower()
        and "eff_vg" not in configs.model.name.lower()
    ):
        model = eval(configs.model.name)(
            device_cfg=configs.model.device_cfg,
            sim_cfg=configs.model.sim_cfg,
            perturbation=configs.model.perturbation,
            num_rows_perside=configs.model.num_rows_perside,
            num_cols=configs.model.num_cols,
            adjoint_mode=configs.model.adjoint_mode,
            learnable_bdry=configs.model.learnable_bdry,
            df=configs.model.df,
            nf=configs.model.nf,
        )
    elif (
        "repara_phc_1x1" in configs.model.name.lower()
        and "eff_vg" in configs.model.name.lower()
    ):
        model = eval(configs.model.name)(
            coupling_region_cfg=configs.model.coupling_region_cfg,
            sim_cfg=configs.model.sim_cfg,
            superlattice_cfg=configs.model.superlattice_cfg,
            port_width=configs.model.port_width,
            port_len=configs.model.port_len,
            taper_width=configs.model.taper_width,
            taper_len=configs.model.taper_len,
            sy_coupling=configs.model.sy_coupling,
            adjoint_mode=configs.model.adjoint_mode,
            eps_bg=configs.model.eps_bg,
            eps_r=configs.model.eps_r,
            df=configs.model.df,
            nf=configs.model.nf,
            a=configs.model.a,
            r=configs.model.r,
            mfs=configs.model.mfs,
            binary_projection_threshold=configs.sharp_scheduler.sharp_threshold,
            binary_projection_method=configs.model.binary_projection_method,
            coupling_init=configs.model.coupling_init,
            opt_coupling_method=configs.model.opt_coupling_method,
            grad_mode=configs.model.grad_mode,
            cal_bd_mode=configs.model.cal_bd_mode,
            aux_out=True
            if configs.aux_criterion.curl_loss.weight > 0
            or configs.aux_criterion.gap_loss.weight > 0
            else False,
            device=device,
        ).to(device)
    elif "metalens" in configs.model.name.lower():
        model = eval(configs.model.name)(
            ridge_height_max=configs.model.ridge_height_max,
            sub_height=configs.model.sub_height,
            aperture=configs.model.aperture,
            f_min=configs.model.f_min,
            f_max=configs.model.f_max,
            eps_r=configs.model.eps_r,
            eps_bg=configs.model.eps_bg,
            sim_cfg=configs.model.sim_cfg,
            ls_cfg=configs.model.ls_cfg,
            mfs=configs.model.mfs,
            binary_projection_threshold=configs.model.binary_projection_threshold,
            build_method=configs.model.build_method,
            center_ridge=configs.model.center_ridge,
            max_num_ridges_single_side=configs.model.max_num_ridges_single_side,
            operation_device=device,
            aspect_ratio=configs.model.aspect_ratio,
            initial_point=configs.model.initial_point,
            if_constant_period=configs.model.if_constant_period,
            focal_constant=configs.model.focal_constant,
        ).to(device)
    elif configs.model.name.lower() == 'metacoupleroptimization':
        model = eval(configs.model.name)(
            device=kwargs["optDevice"],
            sim_cfg=configs.model.sim_cfg,
        ).to(device)
    elif "simplecnn" in configs.model.name.lower():
        model = eval(configs.model.name)().to(device)
    elif "neurolight" in configs.model.name.lower():
        model = eval(configs.model.name)(
            in_channels=configs.model.in_channels,
            out_channels=configs.model.out_channels,
            dim=configs.model.dim,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            mode_list=configs.model.mode_list,
            act_func=configs.model.act_func,
            domain_size=configs.model.domain_size,
            grid_step=configs.model.grid_step,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            device=device,
            conv_stem=configs.model.conv_stem,
            aug_path=configs.model.aug_path,
            ffn=configs.model.ffn,
            ffn_dwconv=configs.model.ffn_dwconv,
            **kwargs,
        ).to(device)
    elif "fno3d" in configs.model.name.lower():
        model = eval(configs.model.name)(
            train_field=configs.model.train_field,
            in_channels=configs.model.in_channels,
            out_channels=configs.model.out_channels,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            mode_list=configs.model.mode_list,
            act_func=configs.model.act_func,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            device=device,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            pos_encoding=configs.model.pos_encoding,
            with_cp=configs.model.with_cp,
            mode1=configs.model.mode1,
            mode2=configs.model.mode2,
            fourier_feature=configs.model.fourier_feature,
            mapping_size=configs.model.mapping_size,
            err_correction=configs.model.err_correction,
            fno_block_only=configs.model.fno_block_only,
        ).to(device)
    elif "ffno2d" in configs.model.name.lower():
        model = eval(configs.model.name)(
            in_channels=configs.model.in_channels,
            out_channels=configs.model.out_channels,
            dim=configs.model.dim,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            mode_list=configs.model.mode_list,
            act_func=configs.model.act_func,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            device=device,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            with_cp=False,
            conv_stem=configs.model.conv_stem,
            aug_path=configs.model.aug_path,
            ffn=configs.model.ffn,
            ffn_dwconv=configs.model.ffn_dwconv,
        ).to(device)
    else:
        raise NotImplementedError(f"Not supported model name: {configs.model.name}")
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
            wl_cen=cfg.wl_cen,
            wl_width=cfg.wl_width,
            n_wl=cfg.n_wl,
            using_ALM=cfg.using_ALM,
        )
    elif name == "grad_loss":
        criterion = GradientLoss()
    elif name == "s_param_loss":
        criterion = SParamLoss()
    else:
        raise NotImplementedError(name)
    return criterion
