"""Factories for MAPS inverse-design devices, optimizers, and schedulers.

This module translates the InvDes configuration tree into concrete runtime
objects.  It is distinct from the top-level ``core.builder`` training/ML
factory: optimizer names, scheduler names, configuration fields, and closure
requirements documented here belong to the physics-based inverse-design path.

Notes:
    Factory dispatch strings are configuration-facing API. Renaming a class or
    optimizer without updating experiment configurations changes user-visible
    behavior.
"""

import random
from typing import Tuple

import torch
import torch.nn as nn
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.sam import SAM
from pyutils.typing import Optimizer, Scheduler
from torch.types import Device

from core.invdes.models import *
from core.invdes.optimizer import (
    BASIC_NCG,
    Adahessian,
    Adam,
    Muon,
    NesterovAcceleratedGradientOptimizer,
)
from core.utils import (
    DAdaptAdam,
    ResolutionScheduler,
    SharpnessScheduler,
    TemperatureScheduler,
)

__all__ = [
    "make_model",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def make_device(device: Device, config=None):
    """Construct the physical device selected by an InvDes configuration.

    Args:
        device: PyTorch execution device passed to the device constructor, for
            example ``torch.device("cuda:0")``.
        config: Device/model configuration object. It must provide
            ``device_type`` and ``sim_cfg``. ``device_type`` is resolved by
            name from the device/model symbols imported into this module.

    Returns:
        The constructed device instance. Its concrete type is selected by
        ``config.device_type``.

    Example:
        A configuration containing ``device_type="Bending"`` and a valid
        ``sim_cfg`` constructs the same physical-device layer used by the
        Bending inverse-design workflow::

            dev = make_device(torch.device("cuda:0"), config=model_cfg)

    Notes:
        This factory constructs the physical device, not the learnable
        optimization model. Geometry rasterization, grids, ports, monitors,
        sources, and design-region masks are established by the selected
        device class.
    """
    device_to_opt = eval(config.device_type)(
        sim_cfg=config.sim_cfg,
        device=device,
    )
    return device_to_opt


def make_model(device: Device, random_state: int = None, **kwargs) -> nn.Module:
    """Legacy inverse-design model factory; currently disabled.

    Args:
        device: Intended PyTorch execution device for the model.
        random_state: Intended random seed/state selector. It is not consumed
            because the function currently exits immediately.
        **kwargs: Legacy model-construction arguments.

    Returns:
        This function does not currently return a model.

    Raises:
        NotImplementedError: Always, before the legacy dispatch code below is
            reached.

    Notes:
        The code after the leading ``raise`` is retained legacy dispatch logic
        and is unreachable in the current baseline. Do not document those
        branches as supported public construction paths until this function is
        intentionally re-enabled and tested.
    """
    raise NotImplementedError
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
            aux_out=(
                True
                if configs.aux_criterion.curl_loss.weight > 0
                or configs.aux_criterion.gap_loss.weight > 0
                else False
            ),
            device=device,
        ).to(device)
    else:
        raise NotImplementedError(f"Not supported model name: {configs.model.name}")
    return model


def make_optimizer(params, total_config=None) -> Optimizer:
    """Build the optimizer used to update inverse-design parameters.

    Args:
        params: Iterable of PyTorch parameters or optimizer parameter-group
            dictionaries, normally ``devOptimization.parameters()``. Parameter
            tensor shapes are defined by the selected design parameterization;
            this factory does not reshape, copy, or move them.
        total_config: InvDes configuration object containing an ``optimizer``
            section. The common fields are ``name``, ``lr``, and
            ``weight_decay``; optimizer-specific optional fields are read with
            the defaults shown in the implementation.

    Returns:
        Configured optimizer instance. Depending on ``optimizer.name`` this may
        be a PyTorch optimizer or a MAPS custom optimizer implementing the same
        step/gradient-update role.

    Raises:
        NotImplementedError: If ``optimizer.name`` is not one of the supported
            dispatch names.

    Example:
        The standard ``InvDesign`` path uses this factory directly::

            optimizer = make_optimizer(
                params=dev_optimization.parameters(),
                total_config=cfg,
            )

    Notes:
        Supported names in this baseline are ``sgd``, ``adam``, ``adamw``,
        ``dadaptadam``, ``sam_sgd``, ``sam_adam``, ``lbfgs``, ``nesterov``,
        ``adahessian``, ``muon``, and the nonlinear-CG variants ``cg_fr``,
        ``cg_prp``, ``cg_hs``, ``cg_cd``, ``cg_ls``, ``cg_dy``, ``cg_hz``,
        and ``cg_hs-dy``.

        Closure semantics differ by optimizer family. ``InvDesign`` supplies a
        closure returning ``-results["obj"]`` so minimization-oriented
        optimizers maximize the physical figure of merit.
    """
    config = total_config.optimizer
    name = config.name.lower()
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=getattr(config, "betas", (0.9, 0.999)),
            eps=getattr(config, "eps", 1e-8),
            init_v=getattr(config, "init_v", 0.0),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif name == "dadaptadam":
        optimizer = DAdaptAdam(
            params,
            lr=config.lr,
            betas=getattr(config, "betas", (0.9, 0.999)),
            weight_decay=config.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(config, "rho", 0.5),
            adaptive=getattr(config, "adaptive", True),
            lr=config.lr,
            weight_decay=config.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(config, "rho", 0.001),
            adaptive=getattr(config, "adaptive", True),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            params,
            lr=config.lr,  # for now, only the lr is tunable, others arguments just use the default value
            line_search_fn=getattr(config, "line_search_fn", None),
            max_iter=getattr(config, "max_iter", 4),
            max_eval=getattr(config, "max_eval", None),
            history_size=getattr(config, "history_size", 100),
            tolerance_grad=getattr(config, "tolerance_grad", 1e-7),
            tolerance_change=getattr(config, "tolerance_change", 1e-9),
        )
    elif name == "nesterov":
        optimizer = NesterovAcceleratedGradientOptimizer(
            params,
            lr=config.lr,
            alg=getattr(config, "alg", "bb"),
            constraint_fn=getattr(config, "constraint_fn", None),
            block_tile_size=getattr(
                config, "block_tile_size", getattr(config, "tile_size", None)
            ),
            block_blur_kernel_size=getattr(config, "block_blur_kernel_size", 3),
        )
    elif name == "adahessian":
        optimizer = Adahessian(
            params,
            lr=config.lr,
            betas=getattr(config, "betas", (0.9, 0.999)),
            eps=getattr(config, "eps", 1e-4),
            weight_decay=config.weight_decay,
            hessian_power=getattr(config, "hessian_power", 0.5),
        )
    elif name == "muon":
        optimizer = Muon(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=getattr(config, "momentum", 0.9),
            nesterov=getattr(config, "nesterov", True),
        )
    elif name.lower() == "cg_fr":
        optimizer = BASIC_NCG(
            params,
            method="FR",
            line_search="Strong_Wolfe",
            c1=getattr(config, "c1", 1e-4),
            c2=getattr(config, "c2", 0.5),
            lr=getattr(config, "lr", 0.2),
            max_ls=getattr(config, "max_ls", 5),
        )
    elif name.lower() == "cg_prp":
        optimizer = BASIC_NCG(
            params,
            method="PRP",
            line_search="Armijo",
            c1=getattr(config, "c1", 1e-4),
            c2=getattr(config, "c2", 0.9),
            lr=getattr(config, "lr", 10),
            rho=getattr(config, "rho", 0.5),
            restart_interval=getattr(config, "restart_interval", None),
        )
    elif name.lower() == "cg_hs":
        optimizer = BASIC_NCG(
            params,
            method="HS",
            line_search="Strong_Wolfe",
            c1=getattr(config, "c1", 1e-4),
            c2=getattr(config, "c2", 0.4),
            lr=getattr(config, "lr", 0.2),
            max_ls=getattr(config, "max_ls", 5),
        )
    elif name.lower() == "cg_cd":
        optimizer = BASIC_NCG(
            params,
            method="CD",
            line_search="Armijo",
            c1=getattr(config, "c1", 1e-4),
            c2=getattr(config, "c2", 0.9),
            lr=getattr(config, "lr", 1),
            rho=getattr(config, "rho", 0.5),
        )
    elif name.lower() == "cg_ls":
        optimizer = BASIC_NCG(
            params,
            method="LS",
            line_search="Armijo",
            c1=getattr(config, "c1", 1e-4),
            c2=getattr(config, "c2", 0.9),
            lr=getattr(config, "lr", 1),
            rho=getattr(config, "rho", 0.5),
        )
    elif name.lower() == "cg_dy":
        optimizer = BASIC_NCG(
            params,
            method="DY",
            line_search="Strong_Wolfe",
            c1=getattr(config, "c1", 1e-4),
            c2=getattr(config, "c2", 0.9),
            lr=getattr(config, "lr", 0.2),
            max_ls=getattr(config, "max_ls", 5),
        )
    elif name.lower() == "cg_hz":
        optimizer = BASIC_NCG(
            params,
            method="HZ",
            line_search="Strong_Wolfe",
            c1=getattr(config, "c1", 1e-4),
            c2=getattr(config, "c2", 0.9),
            lr=getattr(config, "lr", 0.2),
            max_ls=getattr(config, "max_ls", 5),
        )
    elif name.lower() == "cg_hs-dy":
        optimizer = BASIC_NCG(
            params,
            method="HS-DY",
            line_search="Armijo",
            c1=getattr(config, "c1", 1e-4),
            c2=getattr(config, "c2", 0.9),
            lr=getattr(config, "lr", 10),
            rho=getattr(config, "rho", 0.5),
        )

    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(
    optimizer: Optimizer, scheduler_type, config_total: dict = {}
) -> Scheduler:
    """Build an InvDes learning-rate or physical continuation scheduler.

    Args:
        optimizer: Optimizer associated with the run. PyTorch learning-rate
            schedulers update this object; physical schedulers such as
            sharpness, temperature, and resolution retain the argument only as
            part of the common factory interface.
        scheduler_type: Name of the configuration section whose ``name`` field
            selects the scheduler, typically ``"lr_scheduler"`` or
            ``"sharp_scheduler"``. The temperature and resolution branches
            additionally read ``temp_scheduler`` and ``res_scheduler`` fields.
        config_total: Complete InvDes configuration. ``run.n_epochs`` defines
            the schedule horizon in optimization steps/epochs.

    Returns:
        Scheduler instance selected by the configuration. Physical schedulers
        expose their corresponding state (for example sharpness) while PyTorch
        schedulers update optimizer learning rates.

    Raises:
        NotImplementedError: If the selected scheduler name is unsupported.

    Example:
        ``InvDesign`` constructs its continuation schedule as::

            sharp_scheduler = make_scheduler(
                optimizer,
                scheduler_type="sharp_scheduler",
                config_total=cfg,
            )
            sharpness = sharp_scheduler.get_sharpness()

    Notes:
        ``sharpness`` is a continuation/design-projection parameter, not a
        Maxwell-solver tolerance. Increasing it changes the parameterization
        toward a sharper/binary design according to the selected
        parameterization pipeline.
    """
    config = config_total
    name = getattr(config, scheduler_type).name.lower()
    if (
        name == "temperature"
    ):  # this temperature scheduler is a cosine annealing scheduler
        scheduler = TemperatureScheduler(
            initial_T=float(config.temp_scheduler.lr),
            final_T=float(config.temp_scheduler.lr_min),
            total_steps=int(config.run.n_epochs),
        )
    elif name == "resolution":
        scheduler = ResolutionScheduler(
            initial_res=int(config.res_scheduler.init_res),
            final_res=int(config.res_scheduler.final_res),
            total_steps=int(config.run.n_epochs),
        )
    elif name == "sharpness":
        scheduler = SharpnessScheduler(
            initial_sharp=float(config.sharp_scheduler.init_sharp),
            final_sharp=float(config.sharp_scheduler.final_sharp),
            total_steps=int(config.run.n_epochs),
            mode=getattr(config.sharp_scheduler, "mode", "cosine"),
        )
    elif name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.run.n_epochs),
            eta_min=float(config.lr_scheduler.lr_min),
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=config.run.n_epochs,
            max_lr=config.optimizer.lr,
            min_lr=config.scheduler.lr_min,
            warmup_steps=int(config.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.scheduler.lr_gamma
        )
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None, cfg=None) -> nn.Module:
    """Legacy criterion factory; currently disabled.

    Args:
        name: Intended criterion dispatch name.
        cfg: Intended criterion-specific configuration.

    Returns:
        The function currently returns the ``NotImplementedError`` class object
        immediately; it does not instantiate a loss module.

    Notes:
        This behavior differs from ``make_model``, which raises
        ``NotImplementedError``. The legacy criterion dispatch code below the
        immediate return is unreachable in the current baseline. This
        docstring records the implementation as-is and does not normalize that
        inconsistency.
    """
    return NotImplementedError
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
