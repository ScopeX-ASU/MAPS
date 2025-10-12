"""
Date: 2024-10-05 02:02:21
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-05 23:27:36
FilePath: /Metasurface-Opt/core/models/layers/parametrization/__init__.py
"""

import torch
from torch import nn

from .base_parametrization import BaseParametrization
from .levelset import LeveSetParameterization

_param_registry = {"levelset": LeveSetParameterization}

__all__ = ["parametrization_builder"]


def parametrization_builder(
    device, hr_device, sim_cfg, parametrization_cfgs, operation_device, **kwargs
):
    ### build multiple design regions as a dictionary of nn.Module
    param_dict = nn.ModuleDict()
    for region_name, param_cfg in parametrization_cfgs.items():
        method = param_cfg["method"]
        if method not in _param_registry:
            raise ValueError(f"Invalid parametrization method: {method}")
        param_dict[region_name] = _param_registry[method](
            device=device,
            hr_device=hr_device,
            sim_cfg=sim_cfg,
            region_name=region_name,
            cfgs=param_cfg,
            operation_device=operation_device,
            **kwargs,
        )
    return param_dict
