"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2026-05-02 23:10:14
LastEditors: Jiaqi Gu (jiaqigu@asu.edu)
LastEditTime: 2026-07-09 23:00:26
FilePath: /MAPS/core/invdes/models/layers/parametrization/__init__.py

Refactor note:
    The temporary module-level adapter below redirects the legacy
    ``base_parametrization._convert_resolution`` symbol to the canonical
    Torch-native materialization implementation. Existing call sites in the
    large legacy module therefore keep their public signatures and call order
    while the implementation moves to ``core.invdes.models.materialization``.

    Migration invariant:
        Nonlinear optimization density is converted to arithmetic material
        fractions before geometric averaging. The LR physical permittivity is
        encoded back into the configured optimization coordinate only for
        compatibility. No second LR parameterization is created.
"""

import torch
from torch import nn

from core.invdes.models.materialization.resolution import convert_resolution_tensor

from . import base_parametrization as _base_parametrization
from .base_parametrization import BaseParametrization
from .box import BoxParameterization
from .circle import CircleParameterization, CylinderParameterization
from .edgebox import EdgeBoxParameterization
from .levelset import LevelSetParameterization
from .material_projection import EffectiveMediumProjector
from .polygon import PolygonParameterization

# Compatibility adapter: BaseParametrization.density_transform resolves
# ``_convert_resolution`` from its defining module at call time. Redirect that
# symbol here so existing callers use the canonical materialization function
# without rewriting the large legacy base_parametrization.py file yet.
_base_parametrization._convert_resolution = convert_resolution_tensor

_param_registry = {
    "levelset": LevelSetParameterization,
    "box": BoxParameterization,
    "circle": CircleParameterization,
    "cylinder": CylinderParameterization,
    "edgebox": EdgeBoxParameterization,
    "polygon": PolygonParameterization,
}

__all__ = ["parametrization_builder", "EffectiveMediumProjector"]


def parametrization_builder(
    device, hr_device, sim_cfg, parametrization_cfgs, operation_device, **kwargs
):
    """Build the configured differentiable design-region parametrizations.

    Args:
        device: Native-resolution MAPS device containing simulation-grid design
            regions and fixed material state.
        hr_device: High-resolution MAPS device used for HR design realization
            and layout-oriented outputs.
        sim_cfg: Simulation configuration used by parametrization objects.
        parametrization_cfgs: Mapping from design-region names to parameterization
            configurations. Each configuration must contain a supported
            ``method`` entry.
        operation_device: Torch device used for learnable parameters and
            differentiable intermediate tensors.
        **kwargs: Additional options forwarded to concrete parameterization
            constructors.

    Returns:
        ``torch.nn.ModuleDict`` keyed by design-region name. Each value owns the
        single trainable parameter set for that region; HR and LR material maps
        are derived from those shared parameters.

    Raises:
        ValueError: If a requested parameterization method is not registered.
    """
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
