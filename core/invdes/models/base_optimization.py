"""
Date: 2024-10-04 18:49:06
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-03-03 23:08:53
FilePath: /MAPS/core/invdes/models/base_optimization.py
"""

import copy
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import gdsfactory as gf
import h5py
import numpy as np
import ryaml
import torch
import yaml
from autograd.numpy.numpy_boxes import ArrayBox
from gdsfactory.generic_tech import get_generic_pdk
from pyutils.config import Config
from pyutils.general import ensure_dir, logger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.types import Device

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
import matplotlib.pyplot as plt

from core.utils import print_stat
from thirdparty.ceviche.constants import C_0, MICRON_UNIT

from .layers.device_base import N_Ports
from .layers.fom_layer import SimulatedFoM
from .layers.objective import ObjectiveFunc
from .layers.parametrization import parametrization_builder
from .layers.parametrization.base_parametrization import _convert_resolution
from .layers.thermal_control import collect_unique_control_states
from .layers.utils import plot_eps_field, plot_eps_field_3d

sys.path.pop(0)

__all__ = [
    "DefaultSimulationConfig",
    "BaseOptimization",
    "DefaultOptimizationConfig",
]


class DefaultSimulationConfig(Config):
    def __init__(self):
        super().__init__()
        self.update(
            dict(
                solver="ceviche",
                binary_projection=dict(
                    fw_threshold=100,
                    bw_threshold=100,
                    mode="regular",
                ),
                border_width=[0, 0, 6, 6],
                PML=[1, 1],
                cell_size=None,
                resolution=50,
                wl_cen=1.55,
                wl_width=0,
                n_wl=1,
                plot_root="./figs/metacoupler",
            )
        )


def _sum_objectives(breakdowns):
    loss = 0
    for name, obj in breakdowns.items():
        if "smatrix" in name:
            continue
        loss = loss + obj["weight"] * obj["value"]
    extra_breakdown = {}
    return loss, extra_breakdown


def _blend_design_region(eps_map, region_mask, region_eps, region_weight=None):
    if region_weight is None:
        eps_map[region_mask] = region_eps
        return

    weight = torch.as_tensor(
        region_weight, dtype=region_eps.dtype, device=region_eps.device
    )
    if weight.numel() == 0:
        return
    if tuple(weight.shape) != tuple(region_eps.shape):
        raise ValueError(
            "Design-region coverage weight shape mismatch: "
            f"{tuple(weight.shape)} vs {tuple(region_eps.shape)}"
        )

    base_eps = eps_map[region_mask]
    eps_map[region_mask] = base_eps * (1 - weight) + region_eps * weight


def _coverage_weight_from_axes(axis_weights, exclude_axes=()):
    if axis_weights is None:
        return None
    keep_weights = []
    for axis, axis_weight in enumerate(axis_weights):
        if axis in exclude_axes:
            axis_weight = np.ones_like(axis_weight, dtype=np.float32)
        keep_weights.append(axis_weight)

    weight = np.ones(tuple(len(w) for w in keep_weights), dtype=np.float32)
    for axis, axis_weight in enumerate(keep_weights):
        shape = [1] * len(keep_weights)
        shape[axis] = axis_weight.size
        weight *= axis_weight.reshape(shape)
    return weight


def _clone_material_map(material_map, *, dtype, device):
    if material_map is None:
        return None
    return torch.as_tensor(material_map, dtype=dtype, device=device).clone()


def _coords_from_region_mask(coords, region_mask):
    return tuple(
        np.asarray(axis_coords[slc], dtype=np.float64)
        for axis_coords, slc in zip(coords, region_mask)
    )


def _fit_region_mask_to_shape(region_mask, map_shape):
    fitted = []
    for slc, axis_size in zip(region_mask, map_shape):
        start = max(0, int(slc.start))
        stop = min(int(slc.stop), int(axis_size))
        if stop < start:
            stop = start
        fitted.append(slice(start, stop))
    return tuple(fitted)


class DefaultOptimizationConfig(Config):
    def __init__(self):
        super().__init__()
        self.update(
            dict(
                design_region_param_cfgs=dict(),
                sim_cfg={
                    "solver": "ceviche",
                    "border_width": [
                        0,
                        0,
                        0,
                        0,
                    ],  # left, right, lower, upper, containing PML
                    "PML": [1, 1],  # left/right, lower/upper
                    "cell_size": None,
                    "resolution": 50,
                    "wl_cen": 1.55,
                    "wl_width": 0,
                    "n_wl": 1,
                    "plot_root": "./figs/default",
                },
                obj_cfgs=dict(
                    # fwd_trans=dict(
                    #     weight=1,
                    #     #### objective is evaluated at this port
                    #     in_port_name="in_port_1",
                    #     out_port_name="out_port_1",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         1,
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     type="eigenmode",
                    #     direction="y+",
                    # ), # should not be taken as default, the obj functions should be all customized
                    #### objective fusion function can be customized here in obj_cfgs
                    #### the default fusion function is _sum_objectives
                    #### customized fusion function should take breakdown as input
                    #### and return a tuple of (total_obj, extra_breakdown)
                    _fusion_func=_sum_objectives,
                ),
            )
        )


class BaseOptimization(nn.Module):
    def __init__(
        self,
        device: N_Ports,
        hr_device: N_Ports,
        design_region_param_cfgs: dict = dict(),
        sim_cfg: dict = dict(),
        obj_cfgs=dict(),
        operation_device: Device = torch.device("cuda:0"),
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.hr_device = hr_device
        self.operation_device = operation_device
        self._cfgs = DefaultOptimizationConfig()  ## default optimization config
        self._cfgs.update(
            dict(
                sim_cfg=sim_cfg,
                obj_cfgs=obj_cfgs,
                design_region_param_cfgs=design_region_param_cfgs,
            )
        )  ## update with user-defined config
        ## update all the attributes in the config to the class
        for name, cfg in self._cfgs.items():
            setattr(self, name, cfg)

        self.epsilon_map = torch.from_numpy(device.epsilon_map).to(
            self.operation_device
        )
        self.hr_eps_map = torch.from_numpy(hr_device.epsilon_map).to(
            self.operation_device
        )
        self.design_region_masks = device.design_region_masks
        self.design_region_mask_weights = getattr(
            device, "design_region_mask_weights", {}
        )
        self.design_region_axis_weights = getattr(
            device, "design_region_axis_weights", {}
        )
        self.verbose = verbose
        self._thermal_states = {}
        self._thermal_state = None
        self._runtime_thermal_material_maps = None

        self.build_parameters()

        ### need to generate source/monitors
        device.init_monitors(verbose=verbose)

        ### need to run normalization run
        device.norm_run(verbose=verbose)
        self.norm_run_profiles = (
            device.port_sources_dict
        )  # {input_slice_name: source_profiles 2d array, ...}

        ### pre-build objectives
        self.build_objective(
            port_profiles=self.device.port_sources_dict,
            port_slices=self.device.port_monitor_slices,
            port_slices_native=self.device.port_monitor_slices_native,
            port_slices_native_symmetry=self.device.port_monitor_slices_native_symmetry,
            port_slices_info=self.device.port_monitor_slices_info,
            grid_metadata=self.device.fdtdx_native_grid_metadata,
            cell_weights=self.device.fdtdx_native_cell_weights,
            epsilon_map=self.device.epsilon_map,
            obj_cfgs=self.obj_cfgs,
            solver=self.sim_cfg["solver"],
        )

    def _retain_grad_in_structure(self, value):
        if isinstance(value, Tensor):
            if value.requires_grad:
                value.retain_grad()
            return
        if isinstance(value, dict):
            for child in value.values():
                self._retain_grad_in_structure(child)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                self._retain_grad_in_structure(child)
            return

    def _snapshot_runtime_thermal_states(self):
        runtime_states = getattr(self.objective, "runtime_thermal_states", {})
        self._thermal_states = {
            control_key: dict(state) for control_key, state in runtime_states.items()
        }
        self._retain_grad_in_structure(self._thermal_states)
        self._thermal_state = None
        if len(self._thermal_states) == 1:
            self._thermal_state = next(iter(self._thermal_states.values()))

    @staticmethod
    def _format_control_key(control_key) -> str:
        if isinstance(control_key, tuple):
            pieces = [f"{name}={float(value):.4g}A" for name, value in control_key]
            return ", ".join(pieces) if pieces else "all currents = 0 A"
        return f"T={control_key}"

    def _resolve_map_grid_info(self, field_key):
        if field_key in {
            "epsilon",
            "electrical_conductivity",
            "optical_temperature",
            "thermo_optic_coeff",
        }:
            ## this is optical native grid
            return self.device.grid_info_dict["epsilon_map"]
        elif field_key in {
            "temperature",
            "conductivity",
            "heat_capacity",
        }:
            ## those are on thermal grid
            return self.device.grid_info_dict["conductivity_map"]
        else:
            raise ValueError(
                f"Unknown thermal field key '{field_key}' for grid info resolution"
            )

    def _resolve_thermal_plot_map(self, field_key, thermal_map_name: str | None):
        if thermal_map_name is None:
            return None
        control_key = field_key[-1]
        thermal_state = self._thermal_states.get(control_key, self._thermal_state)
        if thermal_state is None:
            return None
        return thermal_state.get(thermal_map_name)

    def _resolve_plot_eps_map(self, field_key):
        control_key = field_key[-1]
        thermal_state = self._thermal_states.get(control_key, self._thermal_state)
        if thermal_state is None:
            return self._eps_map
        return thermal_state.get("eps", self._eps_map)

    def reset_levelset_sdf(self):
        for name, design_region in self.design_region_param_dict.items():
            if hasattr(design_region, "reset_levelset_sdf"):
                print(f"Reset levelset to SDF for design region {name}.")
                design_region.reset_levelset_sdf()

    def reset_parameters(self):
        for design_region in self.design_region_param_dict.values():
            design_region.reset_parameters()

    def build_parameters(self):
        ### create design region parametrizations based on device and design_region_param_cfgs
        ## each design region has a name, and it is an nn.Module.
        ## its self.weights is a nn.ParameterDict which contains all its learnable parameters
        ## during initialization, it will build all parameters and run reset_parameters
        if self.verbose:
            logger.info("Start building design region parametrizations ...")
        self.design_region_param_dict = parametrization_builder(
            device=self.device,
            hr_device=self.hr_device,
            sim_cfg=self.sim_cfg,
            parametrization_cfgs=self.design_region_param_cfgs,
            operation_device=self.operation_device,
        )  ## nn.ModuleDict = {region_name: nn.Module, ...}

        self.objective_layer = SimulatedFoM(self.cal_obj_grad, self.sim_cfg["solver"])

    def _region_exclude_axes(
        self, region_name: str, target_device: N_Ports | None = None
    ):
        target_device = self.device if target_device is None else target_device
        region_mask = target_device.design_region_masks[region_name]
        exclude_axes = ()
        if len(region_mask) == 3:
            design_region = self.design_region_param_dict[region_name]
            dims = sorted(design_region.cfgs.get("dims", (0, 1)))
            exclude_axes = tuple(set(range(3)) - set(dims))
        return exclude_axes

    def _region_blend_weight(
        self,
        region_name: str,
        target_device: N_Ports | None = None,
        *,
        axis_weights_dict: dict | None = None,
        mask_weights_dict: dict | None = None,
    ):
        target_device = self.device if target_device is None else target_device
        exclude_axes = self._region_exclude_axes(
            region_name, target_device=target_device
        )
        if axis_weights_dict is None:
            axis_weights_dict = getattr(target_device, "design_region_axis_weights", {})
        if mask_weights_dict is None:
            mask_weights_dict = getattr(target_device, "design_region_mask_weights", {})
        axis_weights = axis_weights_dict.get(region_name)
        region_weight = _coverage_weight_from_axes(axis_weights, exclude_axes)
        if region_weight is None:
            region_weight = mask_weights_dict.get(region_name)
        return region_weight

    def _apply_region_property_map(
        self,
        base_map: Tensor | None,
        region_name: str,
        region_values: Tensor | None,
        *,
        target_device: N_Ports | None = None,
        region_masks: dict | None = None,
        resize_property_name: str | None = None,
    ) -> Tensor | None:
        if base_map is None or region_values is None:
            return base_map
        target_device = self.device if target_device is None else target_device
        region_masks = (
            target_device.design_region_masks if region_masks is None else region_masks
        )
        region_mask = _fit_region_mask_to_shape(
            region_masks[region_name], base_map.shape
        )
        use_thermal_region_masks = region_masks is getattr(
            target_device, "thermal_design_region_masks", None
        ) and bool(getattr(target_device, "thermal_design_region_masks", {}))
        target_shape = self._region_shape_from_mask(region_mask)
        if tuple(region_values.shape) != tuple(target_shape):
            if use_thermal_region_masks and target_device.thermal_coords is not None:
                region_values = self._resize_region_tensor_for_thermal_property(
                    region_name,
                    region_values,
                    target_device=target_device,
                    target_region_mask=region_mask,
                    property_name=resize_property_name,
                )
            else:
                region_values = self._resize_region_tensor_for_property(
                    region_values,
                    target_shape,
                    property_name=resize_property_name,
                )
        region_weight = self._region_blend_weight(
            region_name,
            target_device=target_device,
            axis_weights_dict=(
                getattr(target_device, "thermal_design_region_axis_weights", {})
                if use_thermal_region_masks
                else None
            ),
            mask_weights_dict=(
                getattr(target_device, "thermal_design_region_mask_weights", {})
                if use_thermal_region_masks
                else None
            ),
        )
        if region_weight is not None and tuple(region_weight.shape) != tuple(
            target_shape
        ):
            region_weight = (
                self._resize_region_tensor_average(
                    torch.as_tensor(region_weight, dtype=region_values.dtype),
                    target_shape,
                )
                .detach()
                .cpu()
                .numpy()
            )
        _blend_design_region(base_map, region_mask, region_values, region_weight)
        return base_map

    def _apply_region_material_maps(
        self,
        base_maps: dict[str, Tensor | None],
        region_material_dict: dict[str, dict],
        *,
        target_device: N_Ports | None = None,
        property_names: tuple[str, ...] | None = None,
        region_masks: dict | None = None,
    ) -> dict[str, Tensor | None]:
        target_device = self.device if target_device is None else target_device
        property_names = property_names or tuple(base_maps.keys())
        for region_name, region_maps in region_material_dict.items():
            active_region_masks = (
                target_device.design_region_masks
                if region_masks is None
                else region_masks
            )
            if region_name not in active_region_masks:
                continue
            for property_name in property_names:
                if property_name not in base_maps:
                    continue
                self._apply_region_property_map(
                    base_maps[property_name],
                    region_name,
                    region_maps.get(property_name),
                    target_device=target_device,
                    region_masks=active_region_masks,
                    resize_property_name=property_name,
                )
        return base_maps

    @staticmethod
    def _region_shape_from_mask(region_mask):
        return tuple(slc.stop - slc.start for slc in region_mask)

    @staticmethod
    def _normalize_sample_coords(
        target_coords: np.ndarray, source_coords: np.ndarray
    ) -> np.ndarray:
        source_coords = np.asarray(source_coords, dtype=np.float64)
        target_coords = np.asarray(target_coords, dtype=np.float64)
        if source_coords.size <= 1:
            return np.zeros_like(target_coords, dtype=np.float64)
        lo = float(source_coords[0])
        hi = float(source_coords[-1])
        if np.isclose(hi, lo):
            return np.zeros_like(target_coords, dtype=np.float64)
        return 2.0 * (target_coords - lo) / (hi - lo) - 1.0

    @classmethod
    def _resample_region_tensor_between_coords(
        cls,
        region_values: Tensor,
        *,
        source_coords: tuple[np.ndarray, ...],
        target_coords: tuple[np.ndarray, ...],
        property_name: str | None = None,
    ) -> Tensor:
        if tuple(region_values.shape) == tuple(len(axis) for axis in target_coords):
            return region_values

        mode = "bilinear" if region_values.ndim == 2 else "trilinear"
        sample_values = region_values
        if property_name == "conductivity":
            eps = torch.finfo(region_values.dtype).tiny
            sample_values = 1.0 / torch.clamp(region_values, min=eps)

        if region_values.ndim == 2:
            src_x, src_y = source_coords
            tgt_x, tgt_y = target_coords
            grid_x = cls._normalize_sample_coords(tgt_x, src_x)
            grid_y = cls._normalize_sample_coords(tgt_y, src_y)
            mesh_x, mesh_y = np.meshgrid(grid_x, grid_y, indexing="ij")
            grid = np.stack((mesh_x.T, mesh_y.T), axis=-1)
            grid = torch.as_tensor(
                grid[None],
                dtype=region_values.dtype,
                device=region_values.device,
            )
            sample_input = sample_values.permute(1, 0)[None, None]
            sampled = F.grid_sample(
                sample_input,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[0, 0].permute(1, 0)
        elif region_values.ndim == 3:
            src_x, src_y, src_z = source_coords
            tgt_x, tgt_y, tgt_z = target_coords
            grid_x = cls._normalize_sample_coords(tgt_x, src_x)
            grid_y = cls._normalize_sample_coords(tgt_y, src_y)
            grid_z = cls._normalize_sample_coords(tgt_z, src_z)
            mesh_x, mesh_y, mesh_z = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
            grid = np.stack((mesh_x, mesh_y, mesh_z), axis=-1).transpose(2, 1, 0, 3)
            grid = torch.as_tensor(
                grid[None],
                dtype=region_values.dtype,
                device=region_values.device,
            )
            sample_input = sample_values.permute(2, 1, 0)[None, None]
            sampled = F.grid_sample(
                sample_input,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[0, 0].permute(2, 1, 0)
        else:
            raise ValueError(f"Unsupported region tensor dim {region_values.ndim}")

        if property_name == "conductivity":
            eps = torch.finfo(sampled.dtype).tiny
            sampled = 1.0 / torch.clamp(sampled, min=eps)
        return sampled

    @staticmethod
    def _resize_region_tensor(
        region_values: Tensor, target_shape: tuple[int, ...]
    ) -> Tensor:
        if tuple(region_values.shape) == tuple(target_shape):
            return region_values
        if region_values.ndim == 2:
            resized = F.interpolate(
                region_values[None, None],
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )
            return resized[0, 0]
        if region_values.ndim == 3:
            resized = F.interpolate(
                region_values[None, None],
                size=target_shape,
                mode="trilinear",
                align_corners=False,
            )
            return resized[0, 0]
        raise ValueError(f"Unsupported region tensor dim {region_values.ndim}")

    @staticmethod
    def _resize_region_tensor_resistive(
        region_values: Tensor, target_shape: tuple[int, ...]
    ) -> Tensor:
        if tuple(region_values.shape) == tuple(target_shape):
            return region_values
        eps = torch.finfo(region_values.dtype).tiny
        reciprocal = 1.0 / torch.clamp(region_values, min=eps)
        if all(
            target <= current
            for target, current in zip(target_shape, tuple(region_values.shape))
        ):
            if region_values.ndim == 2:
                pooled = F.adaptive_avg_pool2d(reciprocal[None, None], target_shape)[
                    0, 0
                ]
            elif region_values.ndim == 3:
                pooled = F.adaptive_avg_pool3d(reciprocal[None, None], target_shape)[
                    0, 0
                ]
            else:
                raise ValueError(f"Unsupported region tensor dim {region_values.ndim}")
            return 1.0 / torch.clamp(pooled, min=eps)
        return 1.0 / torch.clamp(
            BaseOptimization._resize_region_tensor(reciprocal, target_shape),
            min=eps,
        )

    @staticmethod
    def _resize_region_tensor_average(
        region_values: Tensor, target_shape: tuple[int, ...]
    ) -> Tensor:
        if tuple(region_values.shape) == tuple(target_shape):
            return region_values
        if all(
            target <= current
            for target, current in zip(target_shape, tuple(region_values.shape))
        ):
            if region_values.ndim == 2:
                return F.adaptive_avg_pool2d(region_values[None, None], target_shape)[
                    0, 0
                ]
            if region_values.ndim == 3:
                return F.adaptive_avg_pool3d(region_values[None, None], target_shape)[
                    0, 0
                ]
            raise ValueError(f"Unsupported region tensor dim {region_values.ndim}")
        return BaseOptimization._resize_region_tensor(region_values, target_shape)

    @classmethod
    def _resize_region_tensor_for_property(
        cls,
        region_values: Tensor,
        target_shape: tuple[int, ...],
        *,
        property_name: str | None = None,
    ) -> Tensor:
        if property_name == "conductivity":
            return cls._resize_region_tensor_resistive(region_values, target_shape)
        if property_name in {"heat_capacity", "electrical_conductivity"}:
            return cls._resize_region_tensor_average(region_values, target_shape)
        return cls._resize_region_tensor(region_values, target_shape)

    def _resize_region_tensor_for_thermal_property(
        self,
        region_name: str,
        region_values: Tensor,
        *,
        target_device: N_Ports,
        target_region_mask,
        property_name: str | None = None,
    ) -> Tensor:
        source_region_mask = self.device.design_region_masks[region_name]
        source_coords = _coords_from_region_mask(self.device.coords, source_region_mask)
        target_coords = _coords_from_region_mask(
            target_device.thermal_coords, target_region_mask
        )
        return self._resample_region_tensor_between_coords(
            region_values,
            source_coords=source_coords,
            target_coords=target_coords,
            property_name=property_name,
        )

    def _build_runtime_thermal_material_maps(
        self, design_region_eps_dict: dict
    ) -> dict | None:
        if not getattr(self, "uses_current_control", False):
            return None
        include_heat_capacity = bool(
            getattr(self.device, "_heat_include_capacity_default", lambda: False)()
        )

        design_region_material_dict = {}
        for region_name, region_eps in design_region_eps_dict.items():
            design_region = self.design_region_param_dict[region_name]
            density = design_region.normalize_permittivity(region_eps)
            region_materials = {
                "conductivity": self.device.denormalize_design_region_conductivity(
                    density, region_name=region_name
                ),
                "thermo_optic_coeff": self.device.denormalize_design_region_thermo_optic_coeff(
                    density, region_name=region_name
                ),
            }
            if include_heat_capacity:
                region_materials["heat_capacity"] = (
                    self.device.denormalize_design_region_heat_capacity(
                        density, region_name=region_name
                    )
                )
            else:
                region_materials["heat_capacity"] = None
            design_region_material_dict[region_name] = region_materials

        return self._build_runtime_thermal_material_maps_from_regions(
            design_region_material_dict
        )

    def _build_runtime_thermal_material_maps_from_regions(
        self, design_region_material_dict: dict
    ) -> dict | None:
        required_props = ("conductivity", "thermo_optic_coeff")
        include_heat_capacity = bool(
            getattr(self.device, "_heat_include_capacity_default", lambda: False)()
        )
        optional_props = ["electrical_conductivity", "thermo_optic_coeff"]
        if include_heat_capacity:
            optional_props.insert(1, "heat_capacity")
        available_props = {
            prop
            for prop in ("conductivity", *optional_props)
            if any(
                region_maps.get(prop) is not None
                for region_maps in design_region_material_dict.values()
            )
        }
        if not available_props:
            return None
        if getattr(self, "uses_current_control", False) and not all(
            prop in available_props for prop in required_props
        ):
            return None
        if (
            any(
                prop in available_props
                for prop in (
                    "conductivity",
                    *(("heat_capacity",) if include_heat_capacity else ()),
                    "thermo_optic_coeff",
                )
            )
            and self.device.conductivity_map is None
        ):
            self.device.build_thermal_property_maps()
        if "electrical_conductivity" in available_props and not getattr(
            self.device, "_electrical_conductivity_map_built", False
        ):
            self.device.build_electrical_conductivity_map()

        conductivity_base = (
            _clone_material_map(
                self.device.conductivity_map,
                dtype=torch.float32,
                device=self.operation_device,
            )
            if "conductivity" in available_props
            else None
        )
        heat_capacity_base = (
            _clone_material_map(
                self.device.heat_capacity_map,
                dtype=torch.float32,
                device=self.operation_device,
            )
            if "heat_capacity" in available_props
            else None
        )
        thermo_optic_base = _clone_material_map(
            self.device.thermo_optic_coeff_map,
            dtype=torch.float32,
            device=self.operation_device,
        )
        if "electrical_conductivity" in available_props:
            if self.device.electrical_conductivity_map is not None:
                electrical_conductivity_base = _clone_material_map(
                    self.device.electrical_conductivity_map,
                    dtype=torch.float32,
                    device=self.operation_device,
                )
            else:
                template = conductivity_base
                if template is None:
                    template = torch.as_tensor(
                        np.real(self.device.conductivity_map),
                        dtype=torch.float32,
                        device=self.operation_device,
                    )
                electrical_conductivity_base = torch.zeros_like(template)
        else:
            electrical_conductivity_base = None

        runtime_maps = {
            "conductivity": conductivity_base,
            "electrical_conductivity": electrical_conductivity_base,
            "heat_capacity": heat_capacity_base,
            "thermo_optic_coeff": thermo_optic_base,
        }
        thermal_property_names = tuple(
            name
            for name in ("conductivity", "electrical_conductivity", "heat_capacity")
            if runtime_maps.get(name) is not None
        )
        if thermal_property_names:
            self._apply_region_material_maps(
                runtime_maps,
                design_region_material_dict,
                property_names=thermal_property_names,
                region_masks=(
                    self.device.thermal_design_region_masks
                    or self.device.design_region_masks
                ),
            )
        if runtime_maps.get("thermo_optic_coeff") is not None:
            self._apply_region_material_maps(
                runtime_maps,
                design_region_material_dict,
                property_names=("thermo_optic_coeff",),
                region_masks=self.device.design_region_masks,
            )
        return runtime_maps

    def build_device(
        self,
        sharpness: float = 1,
        ls_knots: dict = None,
    ):
        design_region_eps_dict = {}
        hr_design_region_eps_dict = {}
        design_region_material_dict = {}
        ### we need to fill in the permittivity of each design region to the whole device eps_map
        eps_map = self.epsilon_map.data.clone()  # why clone here?
        hr_eps_map = self.hr_eps_map.data.clone()
        eps_region_material_dict = {}
        hr_eps_region_material_dict = {}

        for region_name, design_region in self.design_region_param_dict.items():
            ## obtain each design region's denormalized permittivity only in the design region
            hr_region_mask = self.hr_device.design_region_masks[region_name]
            if ls_knots is None:
                hr_region_maps, region_maps = design_region(
                    sharpness,
                    hr_eps_map,
                    hr_region_mask,
                )
            else:
                hr_region_maps, region_maps = design_region(
                    sharpness,
                    hr_eps_map,
                    hr_region_mask,
                    **ls_knots[region_name],
                )
            design_region_eps_dict[region_name] = region_maps["permittivity"]
            hr_design_region_eps_dict[region_name] = hr_region_maps["permittivity"]
            design_region_material_dict[region_name] = region_maps
            eps_region_material_dict[region_name] = {
                "permittivity": region_maps["permittivity"]
            }
            hr_eps_region_material_dict[region_name] = {
                "permittivity": hr_region_maps["permittivity"]
            }

        self._apply_region_material_maps(
            {"permittivity": eps_map},
            eps_region_material_dict,
            property_names=("permittivity",),
        )
        self._apply_region_material_maps(
            {"permittivity": hr_eps_map},
            hr_eps_region_material_dict,
            target_device=self.hr_device,
            property_names=("permittivity",),
        )

        return (
            eps_map,
            design_region_eps_dict,
            hr_eps_map,
            hr_design_region_eps_dict,
            design_region_material_dict,
        )

    def build_objective(
        self,
        port_profiles: dict,
        port_slices: dict,
        port_slices_native: dict,
        port_slices_native_symmetry: dict,
        port_slices_info: dict,
        grid_metadata=None,
        cell_weights=None,
        epsilon_map=None,
        obj_cfgs=dict(
            fwd_trans=dict(
                weight=1,
                #### objective is evaluated at this port
                in_slice_name="in_slice_1",
                out_slice_name="out_slice_1",
                #### objective is evaluated at all points by sweeping the wavelength and modes
                wl=1.55,
                in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                out_modes=(
                    "Ez1",
                ),  # can evaluate on multiple output modes and get average transmission
                type="eigenmode",
                direction="x+",
            ),
        ),
        solver: str = "ceviche",
    ):

        ### create static forward computational graph from eps to J, no actual execution.
        sim_cfg = self.sim_cfg
        epsilon_map = (
            epsilon_map if epsilon_map is not None else self.device.epsilon_map
        )
        heat_source_names = tuple(getattr(self.device, "heat_source_cfgs", {}).keys())
        control_states, processed_obj_cfgs = collect_unique_control_states(
            obj_cfgs,
            heat_source_names=heat_source_names,
        )
        self.control_states = control_states
        self.control_states_by_key = {
            state["control_key"]: state for state in control_states
        }
        self.uses_current_control = any(
            state["mode"] == "currents" for state in control_states
        )
        obj_cfgs = processed_obj_cfgs
        self.obj_cfgs = obj_cfgs
        if self.uses_current_control and self.device.conductivity_map is None:
            self.device.build_thermal_property_maps()

        if "ceviche" in sim_cfg["solver"]:
            ############# 2D FDFD ######################

            ## let's verify for 2D simulation, input mode and output mode should have the same polarization
            ## and we also collect input polarizations
            in_pols = set()
            for name, obj_cfg in obj_cfgs.items():
                if isinstance(obj_cfg, dict):
                    if "in_mode" in obj_cfg:
                        in_pol = obj_cfg["in_mode"][:2]
                        in_pols.add(in_pol)
                        out_pols = [mode[:2] for mode in obj_cfg["out_modes"]]
                        assert all(
                            [in_pol == out_pol for out_pol in out_pols]
                        ), f"Input and output modes of {name} should have the same polarization"
            ## this is input source wavelength range, each wl needs to build a fdfd simulation
            ## IMPORTANT: when to create a new simulation instance?
            ## Any change that can affect matrix A, we need to create new simulation, e.g., (wl, pol, temp)
            ## why: because we need this sim instance to cache solver state to reuse the solver, requiring the shared matrix A.
            wl_cen, wl_width, n_wl = (
                sim_cfg["wl_cen"],
                sim_cfg["wl_width"],
                sim_cfg["n_wl"],
            )
            simulations = (
                {}
            )  # different polarization and wavelength requires different simulation instances

            native_grid_info = self.device.grid_info_dict.get("epsilon_map")
            native_boundaries = native_grid_info.get("boundaries")
            native_dxs = (
                np.diff(np.asarray(native_boundaries[0], dtype=float)) * MICRON_UNIT
            )
            native_dys = (
                np.diff(np.asarray(native_boundaries[1], dtype=float)) * MICRON_UNIT
            )

            for wl in np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl):
                for pol in in_pols:  # {Ez}, {Hz}, {Ez, Hz}
                    omega = 2 * np.pi * C_0 / (wl * MICRON_UNIT)
                    dl = self.device.grid_step * MICRON_UNIT
                    for control_state in control_states:
                        control_key = control_state["control_key"]
                        sim = self.device.create_simulation(
                            omega,
                            dl,
                            epsilon_map,
                            self.device.NPML,
                            solver,
                            pol=pol,
                            dxs=native_dxs,
                            dys=native_dys,
                        )
                        simulations[(wl, pol, control_key)] = sim
        elif "fdtdx" in sim_cfg["solver"]:
            ############## 3D FDTDX ##################
            simulations = {}
            ## different temperature and mode (not pol) requires a new simulation
            ## each wavelength group (wl_cen, wl_width, n_wl) will be grouped for broadband simulation
            ## in 3D simulation, input and output polarizations do not need to be the same.
            ## but in FDFD, it is mainly for simulator/solver reuse, e.g., matrix factorization
            ## but in FDTD, there is no sharing anyway, to keep the same convention, we still use pol as key.
            in_pols = set()
            for name, obj_cfg in obj_cfgs.items():
                if isinstance(obj_cfg, dict):
                    if "in_mode" in obj_cfg:
                        in_pol = obj_cfg["in_mode"][:2]
                        in_pols.add(in_pol)
                        if "out_modes" in obj_cfg:
                            out_pols = [mode[:2] for mode in obj_cfg["out_modes"]]
                        else:
                            out_pols = [in_pol]  # if not exist, then just use in_pol
            wl_cen, wl_width, n_wl = (
                sim_cfg["wl_cen"],
                sim_cfg["wl_width"],
                sim_cfg["n_wl"],
            )

            for pol in in_pols:
                for control_state in control_states:
                    control_key = control_state["control_key"]
                    sim = self.device.create_simulation_fdtdx(
                        epsilon_map,
                        wl_cen=wl_cen,
                        wl_width=wl_width,
                        n_wl=n_wl,
                        NPML=self.device.NPML,
                    )
                    simulations[((wl_cen, wl_width, n_wl), pol, control_key)] = sim

        self.objective = ObjectiveFunc(
            simulations=simulations,
            port_profiles=port_profiles,
            port_slices=port_slices,
            port_slices_native=port_slices_native,
            port_slices_native_symmetry=port_slices_native_symmetry,
            port_slices_info=port_slices_info,
            grid_step=self.device.grid_step,
            eps_bg=self.device.eps_bg,
            device=self.device,
            control_states=self.control_states_by_key,
            grid_metadata=grid_metadata,
            cell_weights=cell_weights,
            design_region_masks=self.device.design_region_masks,
            design_region_cfgs=self.device.design_region_cfgs,
        )

        obj_cfgs = copy.deepcopy(obj_cfgs)
        self.objective.add_objective(obj_cfgs)

        ### create static backward computational graph from J to eps, no actual execution.'
        ### only usedful for autograd, not for torch autodiff
        self.gradient_region = "global_region"
        if self.sim_cfg["solver"] == "ceviche":
            # self.objective.add_adj_objective(obj_cfgs)
            self.objective.build_jacobian()
            self.objective.build_adj_jacobian()

        return self.objective

    def _get_region_param_grad_plot_data(self, region_name: str):
        design_region = self.design_region_param_dict[region_name]
        if design_region is None:
            return None

        grad_tensor = None
        grad_name = None
        for name, param in design_region.named_parameters():
            if param.grad is None:
                continue
            tensor = param.grad.detach()
            if tensor.ndim == 2:
                grad_tensor = tensor
                grad_name = name.split(".")[-1]
                break
            if tensor.ndim > 2:
                squeezed = tensor.squeeze()
                if squeezed.ndim == 2:
                    grad_tensor = squeezed
                    grad_name = name.split(".")[-1]
                    break

        if grad_tensor is None:
            return None

        dims = sorted(design_region.cfgs.get("dims", (0, 1)))
        region_size = np.asarray(design_region.design_region_cfg["size"], dtype=float)
        if region_size.shape[0] <= max(dims):
            return None

        return dict(
            grad=grad_tensor,
            x_width=float(region_size[dims[0]]),
            y_height=float(region_size[dims[1]]),
            name=grad_name or "param",
        )

    def cal_obj_grad(
        self,
        adjoint_mode: str = "ceviche",
        need_item: str = "need_value",
        resolution: int = None,
        permittivity_list: List[Tensor] = None,
        custom_source: dict = None,
        *args,
    ):
        ## here permittivity_list is a list of tensors (no grad required, since it is from autograd.Function)
        if adjoint_mode == "ceviche":
            total_value = self._cal_obj_grad_ceviche(
                need_item, [p.cpu().numpy() for p in permittivity_list], *args
            )
        elif adjoint_mode == "ceviche_torch":
            total_value = self._cal_obj_grad_ceviche(
                need_item, permittivity_list, custom_source, *args
            )
        elif adjoint_mode == "fdtdx":
            total_value = self._cal_obj_grad_ceviche(
                need_item, permittivity_list, custom_source, *args
            )
        else:
            raise ValueError(f"Unsupported adjoint mode: {adjoint_mode}")

        return total_value

    def _cal_obj_grad_ceviche(
        self,
        need_item,
        permittivity_list: List[np.ndarray | Tensor],
        custom_source,
        *args,
    ):
        ## here permittivity_list is a list of tensors (no grad required, since it is from autograd.Function)
        permittivity = permittivity_list[0]

        if need_item == "need_value":
            total_value = self.objective(
                permittivity, custom_source=custom_source, mode="forward"
            )
        elif need_item == "need_gradient":
            ### this is explicitly called for autograd, not needed for torch autodiff
            raise NotImplementedError(
                "ceviche adjoint mode is deprecated, please use ceviche_torch"
            )
            total_value = self.objective(
                permittivity,
                self.device.epsilon_map.shape,
                mode="backward",
            )
            self.current_eps_grad = total_value

        else:
            raise NotImplementedError
        return total_value

    def plot(
        self,
        plot_filename,
        eps_map=None,
        obj=None,
        field_key: Tuple = ("in_slice_1", 1.55, 1, 300),
        field_component: str = "Ez",
        in_slice_name: str = "in_slice_1",
        exclude_slice_names: List[str] = [],
        thermal_map_name: str | None = None,
        show_delta_eps: bool | None = None,
        eps_grad: bool | Tensor = False,
        param_grad: bool | Tensor | dict = False,
    ):
        # print(
        #     "this is the keys of self.objective.solutions",
        #     list(self.objective.solutions.keys()),
        #     flush=True,
        # )
        Ez = self.objective.solutions[field_key][field_component]
        on_native_grid = not self.objective._interpolate_fields_to_export_grid
        extended_Ez = self.objective.total_farfield_region_solutions.get(
            field_key, {}
        ).get(field_component, None)
        if extended_Ez is not None and Ez.ndim == 3:
            Ez = torch.cat((Ez, extended_Ez), dim=0)
            x_shift_coord = extended_Ez.shape[0] * self.device.grid_step
            x_shift_idx = extended_Ez.shape[0]
        monitors = []
        for name, m in self.device.port_monitor_slices_export.items():
            if name in exclude_slice_names:
                continue
            if name == in_slice_name:
                color = "r"
            elif name.startswith("rad_"):
                color = "g"
            else:
                color = "b"
            # if isinstance(m, np.ndarray):
            #     m = torch.from_numpy(m).to(self.operation_device)
            #     if extended_Ez is not None:
            #         m = m.cpu().numpy()
            #     else:
            #         extended_m = torch.zeros_like(extended_Ez)
            #         m = torch.cat((m, extended_m), dim=0)
            monitors.append((m, color))
        base_eps_map = self._eps_map
        eps_map = (
            eps_map if eps_map is not None else self._resolve_plot_eps_map(field_key)
        )
        eps_grad_map = None
        if isinstance(eps_grad, bool):
            if eps_grad and isinstance(eps_map, Tensor) and eps_map.grad is not None:
                eps_grad_map = eps_map.grad
        elif eps_grad is not None:
            eps_grad_map = eps_grad
        if extended_Ez is not None and Ez.ndim == 3:
            extended_eps_map = (
                torch.ones_like(extended_Ez, dtype=torch.float64) * self.device.eps_bg
            )
            eps_map = torch.cat((eps_map, extended_eps_map), dim=0)
            base_eps_map = torch.cat((base_eps_map, extended_eps_map), dim=0)
            if eps_grad_map is not None:
                extended_eps_grad = torch.zeros_like(
                    extended_Ez, dtype=eps_grad_map.dtype
                )
                eps_grad_map = torch.cat((eps_grad_map, extended_eps_grad), dim=0)
        obj = obj if obj is not None else self._obj
        if isinstance(obj, Tensor):
            obj = obj.item()
        if isinstance(Ez, ArrayBox):
            Ez = Ez._value
        thermal_map = self._resolve_thermal_plot_map(field_key, thermal_map_name)
        heat_source_map = self._resolve_thermal_plot_map(field_key, "q_map")
        if extended_Ez is not None and Ez.ndim == 3 and heat_source_map is not None:
            if not isinstance(heat_source_map, Tensor):
                heat_source_map = torch.as_tensor(
                    heat_source_map,
                    dtype=torch.float32,
                    device=self.operation_device,
                )
            extended_heat_source = torch.zeros_like(
                extended_Ez,
                dtype=heat_source_map.dtype,
                device=heat_source_map.device,
            )
            heat_source_map = torch.cat((heat_source_map, extended_heat_source), dim=0)
        control_label = self._format_control_key(field_key[-1])

        def _region_plot_filepath(base_filepath: str, region_name: str) -> str:
            root, ext = os.path.splitext(base_filepath)
            return f"{root}_{region_name}{ext}"

        def _resolve_region_param_grad(region_name: str):
            if isinstance(param_grad, bool):
                if not param_grad:
                    return None
                return self._get_region_param_grad_plot_data(region_name)
            if isinstance(param_grad, dict):
                return param_grad.get(region_name, None)
            if isinstance(param_grad, Tensor):
                if len(self.device.design_region_cfgs) == 1:
                    only_region = next(iter(self.device.design_region_cfgs))
                    if region_name == only_region:
                        region_cfg = self.device.design_region_cfgs[region_name]
                        dims = sorted(
                            self.design_region_param_dict[region_name].cfgs.get(
                                "dims", (0, 1)
                            )
                        )
                        region_size = np.asarray(region_cfg["size"], dtype=float)
                        return dict(
                            grad=param_grad,
                            x_width=float(region_size[dims[0]]),
                            y_height=float(region_size[dims[1]]),
                            name="param",
                        )
                return None
            return None

        if Ez.ndim == 3:
            base_filepath = os.path.join(self.sim_cfg["plot_root"], plot_filename)
            if not self.device.design_region_cfgs:
                ## there is no design region, we just plot the field and eps map in the center
                design_region_center = np.array([0.0, 0.0, 0.0])
                design_region_cfgs = {"region_1": {"center": design_region_center}}
            else:
                design_region_cfgs = self.device.design_region_cfgs

            fields = self.objective.solutions[field_key]
            map_dict = {
                "eps_map": eps_map.data,
                "base_eps_map": base_eps_map.data,
                "thermal_map": thermal_map.data if thermal_map is not None else None,
                "heat_source_map": (
                    heat_source_map.data if heat_source_map is not None else None
                ),
                "eps_grad_map": eps_grad_map.data if eps_grad_map is not None else None,
            }
            map_dict.update(
                {
                    k: v.data
                    for k, v in fields.items()
                    if k.startswith(field_component[0])
                }
            )
            grid_dict = {
                "eps_map": self.device.grid_info_dict.get("epsilon_map"),
                "base_eps_map": self.device.grid_info_dict.get("epsilon_map"),
                "thermal_map": (
                    self._resolve_map_grid_info(thermal_map_name)
                    if thermal_map is not None
                    else None
                ),
                "heat_source_map": self.device.grid_info_dict.get("conductivity_map"),
                "eps_grad_map": self.device.grid_info_dict.get("epsilon_map"),
            }
            grid_dict.update(
                {
                    k: self.device.grid_info_dict.get("epsilon_map")
                    for k, v in fields.items()
                    if k.startswith(field_component[0])
                }
            )

            for map_name in map_dict:
                map_values = map_dict[map_name]
                src_grid = grid_dict[map_name]
                if map_values is not None and src_grid is not None:
                    map_dict[map_name] = self.device.resample_map_between_coords(
                        values=map_values,
                        src_coords=src_grid["coords"],
                        dst_coords=self.device.export_grid_metadata["coords"],
                    )
            eps_map = map_dict["eps_map"]
            base_eps_map = map_dict["base_eps_map"]
            thermal_map = map_dict["thermal_map"]
            heat_source_map = map_dict["heat_source_map"]
            eps_grad_map = map_dict["eps_grad_map"]
            fields = {
                k: map_dict[k].data for k in fields if k.startswith(field_component[0])
            }

            for region_name, region_cfg in design_region_cfgs.items():
                design_region_center = np.asarray(region_cfg["center"], dtype=float)
                region_param_grad = _resolve_region_param_grad(region_name)
                if design_region_center.shape[0] == 2:
                    design_region_center = np.append(design_region_center, 0.0)
                if extended_Ez is not None:
                    design_region_center = design_region_center.copy()
                    design_region_center[0] -= x_shift_coord

                plot_eps_field_3d(
                    fields,
                    field_component,
                    eps_map,
                    base_eps=base_eps_map,
                    show_delta_eps=show_delta_eps,
                    thermal_map=thermal_map,
                    heat_source_map=heat_source_map,
                    thermal_map_name=thermal_map_name,
                    eps_grad=eps_grad_map,
                    param_grad=(
                        None if region_param_grad is None else region_param_grad["grad"]
                    ),
                    param_x_width=(
                        None
                        if region_param_grad is None
                        else region_param_grad["x_width"]
                    ),
                    param_y_height=(
                        None
                        if region_param_grad is None
                        else region_param_grad["y_height"]
                    ),
                    param_name=(
                        "param"
                        if region_param_grad is None
                        else region_param_grad["name"]
                    ),
                    filepath=_region_plot_filepath(base_filepath, region_name),
                    monitors=monitors,
                    center=design_region_center,
                    x_width=self.device.cell_size[0]
                    + (extended_Ez.shape[0] if extended_Ez is not None else 0)
                    * self.device.grid_step,
                    y_height=self.device.cell_size[1],
                    z_depth=self.device.cell_size[2],
                    NPML=self.device.NPML_export,
                    title=(
                        f"|{field_component}|: {control_label}, "
                        f"Region: {region_name}, FoM: {obj:.3f}"
                    ),
                    field_stat="abs_real",
                    zoom_eps_factor=1,
                    zoom_eps_center=design_region_center,
                    x_shift_coord=x_shift_coord if extended_Ez is not None else 0,
                    x_shift_idx=x_shift_idx if extended_Ez is not None else 0,
                )
        else:
            # 2D FDFD fields may be retained on the native rectilinear grid.
            # Plotting uses export-grid monitor indices, so resample every
            # optical quantity used by the plot to that same grid first.
            native_grid = self.device.grid_info_dict.get("epsilon_map")
            export_grid = self.device.grid_info_dict.get("export_epsilon_map")
            if native_grid is not None and export_grid is not None:
                field_grid = native_grid if on_native_grid else export_grid
                if on_native_grid:
                    Ez = self.device.resample_map_between_coords(
                        Ez,
                        src_coords=field_grid["coords"],
                        dst_coords=export_grid["coords"],
                    )

                def _resample_plot_map(values, source_grid):
                    if values is None or source_grid is None:
                        return values
                    return self.device.resample_map_between_coords(
                        values,
                        src_coords=source_grid["coords"],
                        dst_coords=export_grid["coords"],
                    )

                eps_map = _resample_plot_map(eps_map, native_grid)
                base_eps_map = _resample_plot_map(base_eps_map, native_grid)
                eps_grad_map = _resample_plot_map(eps_grad_map, native_grid)
                thermal_map = _resample_plot_map(
                    thermal_map,
                    self.device.grid_info_dict.get("conductivity_map"),
                )
                heat_source_map = _resample_plot_map(
                    heat_source_map,
                    self.device.grid_info_dict.get("conductivity_map"),
                )

            if extended_Ez is not None:
                Ez = torch.cat((Ez, extended_Ez), dim=0)
                x_shift_coord = extended_Ez.shape[0] * self.device.grid_step
                x_shift_idx = extended_Ez.shape[0]
                extended_eps_map = (
                    torch.ones_like(extended_Ez, dtype=eps_map.dtype)
                    * self.device.eps_bg
                )
                eps_map = torch.cat((eps_map, extended_eps_map), dim=0)
                base_eps_map = torch.cat((base_eps_map, extended_eps_map), dim=0)
                if eps_grad_map is not None:
                    eps_grad_map = torch.cat(
                        (
                            eps_grad_map,
                            torch.zeros_like(extended_Ez, dtype=eps_grad_map.dtype),
                        ),
                        dim=0,
                    )
                if heat_source_map is not None:
                    heat_source_map = torch.cat(
                        (
                            heat_source_map,
                            torch.zeros_like(extended_Ez, dtype=heat_source_map.dtype),
                        ),
                        dim=0,
                    )

            region_param_grad = None
            if len(self.device.design_region_cfgs) == 1:
                only_region = next(iter(self.device.design_region_cfgs))
                region_param_grad = _resolve_region_param_grad(only_region)
            design_region_center = np.mean(
                np.array(
                    [cfg["center"] for cfg in self.device.design_region_cfgs.values()]
                ),
                axis=0,
            )
            if extended_Ez is not None:
                design_region_center = design_region_center - x_shift_coord
            plot_eps_field(
                Ez,
                field_component,
                eps_map,
                base_eps=base_eps_map,
                show_delta_eps=show_delta_eps,
                thermal_map=thermal_map,
                heat_source_map=heat_source_map,
                thermal_map_name=thermal_map_name,
                eps_grad=eps_grad_map,
                param_grad=(
                    None if region_param_grad is None else region_param_grad["grad"]
                ),
                param_x_width=(
                    None if region_param_grad is None else region_param_grad["x_width"]
                ),
                param_y_height=(
                    None if region_param_grad is None else region_param_grad["y_height"]
                ),
                param_name=(
                    "param" if region_param_grad is None else region_param_grad["name"]
                ),
                filepath=os.path.join(self.sim_cfg["plot_root"], plot_filename),
                monitors=monitors,
                x_width=self.device.cell_size[0]
                + (extended_Ez.shape[0] if extended_Ez is not None else 0)
                * self.device.grid_step,
                y_height=self.device.cell_size[1],
                NPML=self.device.NPML_export,
                title=f"|{field_component}|: {control_label}, FoM: {obj:.3f}",
                field_stat="abs_real",
                zoom_eps_factor=1,
                zoom_eps_center=design_region_center,
                x_shift_coord=x_shift_coord if extended_Ez is not None else 0,
                x_shift_idx=x_shift_idx if extended_Ez is not None else 0,
            )

    def dump_gds_files(self, filename):
        def _resolve_port_axis(port_cfg: dict) -> str:
            direction = str(port_cfg.get("direction", "")).lower()
            if direction.startswith("x"):
                return "x"
            if direction.startswith("y"):
                return "y"

            size = port_cfg.get("size", [0.0, 0.0])
            return "x" if float(size[0]) >= float(size[1]) else "y"

        def _infer_port_endpoint(
            port_cfg: dict,
            cell_size: Tuple[float, float],
            offset: Tuple[float, float],
        ) -> tuple[tuple[float, float], float, float, str]:
            axis = _resolve_port_axis(port_cfg)
            center = [float(v) for v in port_cfg["center"]]
            size = [float(v) for v in port_cfg["size"]]

            if axis == "x":
                left = center[0] - size[0] / 2
                right = center[0] + size[0] / 2
                left_gap = abs(left + cell_size[0] / 2)
                right_gap = abs(cell_size[0] / 2 - right)
                if left_gap <= right_gap:
                    print(center[1], cell_size[1] / 2, offset[1])
                    location = (
                        left + cell_size[0] / 2 - offset[0],
                        center[1] + cell_size[1] / 2 - offset[1],
                    )
                    orientation = 180.0
                else:
                    location = (
                        right + cell_size[0] / 2 - offset[0],
                        center[1] + cell_size[1] / 2 - offset[1],
                    )
                    orientation = 0.0
                width = size[1]
            else:
                bottom = center[1] - size[1] / 2
                top = center[1] + size[1] / 2
                bottom_gap = abs(bottom + cell_size[1] / 2)
                top_gap = abs(cell_size[1] / 2 - top)
                if bottom_gap <= top_gap:
                    location = (
                        center[0] + cell_size[0] / 2 - offset[0],
                        bottom + cell_size[1] / 2 - offset[1],
                    )
                    orientation = 270.0
                else:
                    location = (
                        center[0] + cell_size[0] / 2 - offset[0],
                        top + cell_size[1] / 2 - offset[1],
                    )
                    orientation = 90.0
                width = size[0]

            return location, float(width), orientation, axis

        def _device_to_gds_coords(
            center_um: Tuple[float, float], shape: Tuple[int, int], grid_step: float
        ) -> tuple[float, float]:
            # gdsfactory.read.from_np pads 2 pixels around the ndarray.
            x_idx = center_um[0] / grid_step + (shape[0] - 1) / 2
            y_idx = center_um[1] / grid_step + (shape[1] - 1) / 2
            return (float((x_idx + 2.0) * grid_step), float((y_idx + 2.0) * grid_step))

        def _component_lower_left_and_size_um(
            component,
        ) -> tuple[tuple[float, float], tuple[float, float]]:
            # Prefer dbbox() because it is in um and stable across recent gdsfactory versions.
            if hasattr(component, "dbbox"):
                dbbox = component.dbbox()
                left = float(dbbox.left)
                bottom = float(dbbox.bottom)
                width = float(dbbox.width())
                height = float(dbbox.height())
                return (left, bottom), (width, height)

            # Fallback to basic bbox properties if dbbox() is unavailable.
            if all(
                hasattr(component, attr) for attr in ("xmin", "ymin", "xmax", "ymax")
            ):
                xmin = float(component.xmin)
                ymin = float(component.ymin)
                xmax = float(component.xmax)
                ymax = float(component.ymax)
                return (xmin, ymin), (xmax - xmin, ymax - ymin)

            raise RuntimeError(
                "Cannot infer component bounding box; unsupported gdsfactory component API."
            )

        def upsample_to_1nm(
            img: np.ndarray,
            ORIG_PIXEL_SIZE_NM: float = 1,
            TARGET_PIXEL_SIZE_1NM: float = 1,
            method: str = "nearest",
        ) -> np.ndarray:
            if ORIG_PIXEL_SIZE_NM == TARGET_PIXEL_SIZE_1NM:
                print("image resolution is {ORIG_PIXEL_SIZE_NM}, skip upsamling")
                return img
            h, w = img.shape
            scale = ORIG_PIXEL_SIZE_NM / TARGET_PIXEL_SIZE_1NM

            new_w = int(round(w * scale))
            new_h = int(round(h * scale))

            print(f"[1] Upsample: {w}x{h} -> {new_w}x{new_h} pixels (1 nm/px)")
            interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
            }.get(method)
            if interpolation is None:
                raise ValueError(f"Unsupported upsampling method: {method}")
            img_up = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

            # Effective pixel size after rounding
            eff_px_size_x = ORIG_PIXEL_SIZE_NM * w / new_w
            eff_px_size_y = ORIG_PIXEL_SIZE_NM * h / new_h
            print(
                f"    Effective pixel size: x ≈ {eff_px_size_x:.4f} nm, y ≈ {eff_px_size_y:.4f} nm"
            )
            return img_up

        export_device = self.hr_device

        export_eps_map = copy.deepcopy(self._hr_eps_map)
        if isinstance(export_eps_map, Tensor) and torch.is_complex(export_eps_map):
            export_eps_map = export_eps_map.real
        elif isinstance(export_eps_map, np.ndarray) and np.iscomplexobj(export_eps_map):
            export_eps_map = export_eps_map.real
        design_region_mask_list = []
        for (
            design_region_name,
            design_region_mask,
        ) in export_device.design_region_masks.items():
            design_region_mask_list.append(design_region_mask)
        assert (
            len(design_region_mask_list) == 1
        ), "Only support one design region for now"
        design_region_mask = design_region_mask_list[0]

        if isinstance(export_eps_map, Tensor) or isinstance(export_eps_map, np.ndarray):
            max_permittivity = export_eps_map[design_region_mask].max().item()
            min_permittivity = export_eps_map[design_region_mask].min().item()
        elif isinstance(export_eps_map, ArrayBox):
            max_permittivity = export_eps_map[design_region_mask]._value.max()
            min_permittivity = export_eps_map[design_region_mask]._value.min()
        else:
            raise ValueError(f"Unknown type of eps_map: {type(export_eps_map)}")

        if export_eps_map.ndim == 3:
            export_eps_map = export_eps_map[:, :, export_eps_map.shape[2] // 2]
            design_region_mask = design_region_mask[:2]

        if isinstance(export_eps_map, Tensor):
            final_design_eps = export_eps_map.detach().cpu().numpy()
        elif isinstance(export_eps_map, np.ndarray):
            final_design_eps = export_eps_map
        elif isinstance(export_eps_map, ArrayBox):
            final_design_eps = export_eps_map._value
        else:
            raise ValueError(f"Unknown type of eps_map: {type(export_eps_map)}")
        plt.figure()
        plt.imshow(final_design_eps.T, cmap="Greys")
        plt.colorbar()
        plt.savefig(
            os.path.join(self.sim_cfg["plot_root"], "final_design_eps" + ".png"),
            dpi=300,
        )
        plt.close()

        nm_per_pixel = 1000 / export_device.resolution
        print(nm_per_pixel)
        # Normalize eps map to [0, 1] before upsampling.
        eps_span = max_permittivity - min_permittivity
        if eps_span <= 0:
            raise ValueError(
                f"Invalid permittivity range for normalization: "
                f"max={max_permittivity}, min={min_permittivity}"
            )
        final_design_eps = ((final_design_eps - min_permittivity) / eps_span).clip(
            0.0, 1.0
        )
        target_nm_per_pixel = 1.0

        ## Todo: first upsampling the full eps map using the nearest method, then replace design region part with the upsampling mask using the bilinear method
        if isinstance(design_region_mask, np.ndarray):
            if design_region_mask.shape != final_design_eps.shape:
                raise ValueError(
                    "design_region_mask shape mismatch: "
                    f"{design_region_mask.shape} vs {final_design_eps.shape}"
                )
            design_region_mask_dense = design_region_mask.astype(np.float32)
        elif hasattr(design_region_mask, "x") and hasattr(design_region_mask, "y"):
            design_region_mask_dense = np.zeros_like(final_design_eps, dtype=np.float32)
            design_region_mask_dense[design_region_mask.x, design_region_mask.y] = 1.0
        elif isinstance(design_region_mask, tuple) and len(design_region_mask) == 2:
            design_region_mask_dense = np.zeros_like(final_design_eps, dtype=np.float32)
            design_region_mask_dense[design_region_mask[0], design_region_mask[1]] = 1.0
        else:
            raise TypeError(
                f"Unsupported design_region_mask type: {type(design_region_mask)}"
            )

        design_region_mask = final_design_eps[design_region_mask]
        print("design_region_mask shape: ", design_region_mask.shape)
        # full_up_nearest = upsample_to_1nm(
        #     final_design_eps,
        #     nm_per_pixel,
        #     target_nm_per_pixel,
        #     method="nearest",
        # )
        # full_up_bilinear = upsample_to_1nm(
        #     final_design_eps,
        #     nm_per_pixel,
        #     target_nm_per_pixel,
        #     method="bilinear",
        # )
        # design_region_up = upsample_to_1nm(
        #     design_region_mask_dense,
        #     nm_per_pixel,
        #     target_nm_per_pixel,
        #     method="nearest",
        # ) >= 0.5

        design_region_up = upsample_to_1nm(
            design_region_mask,
            nm_per_pixel,
            target_nm_per_pixel,
            method="bilinear",
        )
        print("design_region_up shape: ", design_region_up.shape)
        # final_up = np.where(design_region_up, full_up_bilinear, 0)
        mask = (design_region_up >= 0.5).astype(float)

        PDK = get_generic_pdk()
        PDK.activate()

        from data.experiment.mask_to_gds import mask_to_gds

        mask_u8 = cv2.rotate(
            np.ascontiguousarray((mask >= 0.5).astype(np.uint8) * 255),
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mask_png_path = tmpdir_path / "eps_mask.png"
            mask_gds_path = tmpdir_path / "eps_mask.gds"
            if not cv2.imwrite(str(mask_png_path), mask_u8):
                raise RuntimeError(
                    f"Failed to write temporary mask image: {mask_png_path}"
                )
            mask_to_gds(
                img_path=str(mask_png_path),
                out_gds=str(mask_gds_path),
                layer=1,
                datatype=0,
                pixel_size_um=target_nm_per_pixel / 1000.0,
                threshold=127,
                invert=False,
                rotate_ccw_90=False,
            )
            # eps_component = gf.read.from_gdspaths([mask_gds_path])
            eps_component = gf.import_gds(mask_gds_path)
        # eps_component = gf.read.from_np(
        #     mask,
        #     nm_per_pixel=target_nm_per_pixel,
        #     threshold=0.5,
        # )
        (ll_x_um, ll_y_um), (size_x_um, size_y_um) = _component_lower_left_and_size_um(
            eps_component
        )
        print(
            f"eps_component lower-left (um): ({ll_x_um:.6f}, {ll_y_um:.6f}), "
            f"size (um): ({size_x_um:.6f}, {size_y_um:.6f})"
        )
        device_width_x, device_width_y = design_region_up.shape
        device_width_x, device_width_y = device_width_x / 1000, device_width_y / 1000
        dx = device_width_x - size_x_um
        dy = device_width_y - size_y_um
        # print(
        #     f"eps_component lower-left (nm): ({ll_x_um * 1e3:.3f}, {ll_y_um * 1e3:.3f}), "
        #     f"size (nm): ({size_x_um * 1e3:.3f}, {size_y_um * 1e3:.3f})"
        # )
        if ll_x_um != 0.0 or ll_y_um != 0.0:
            if hasattr(eps_component, "dmove"):
                eps_component.dmove((-ll_x_um, -ll_y_um))
            elif hasattr(eps_component, "move"):
                eps_component.move(origin=(ll_x_um, ll_y_um), destination=(0.0, 0.0))
            else:
                raise RuntimeError(
                    "Cannot move component to origin; unsupported gdsfactory component API."
                )

        cell_size = (
            float(export_device.cell_size[0]),
            float(export_device.cell_size[1]),
        )
        grid_step = float(target_nm_per_pixel / 1000.0)
        offset = (self.sim_cfg["border_width"][0], self.sim_cfg["border_width"][2])
        print("cell_size: ", cell_size)  ## zero point at the center
        port_metadata = {}
        # final_gds = gf.Component("final_gds")
        final_gds = gf.Component()
        left_port_offset = 0.0
        bottom_port_offset = 0.0
        for port_name, port_cfg in export_device.port_cfgs.items():
            port_location_um, width_um, orientation, axis = _infer_port_endpoint(
                port_cfg, cell_size, offset
            )
            print(port_location_um, width_um, orientation, axis)
            if axis == "x":
                port_length = port_cfg["size"][0] - device_width_x / 2
            else:
                port_length = port_cfg["size"][1] - device_width_y / 2
            if orientation == 180.0:
                left_port_offset = max(left_port_offset, float(port_length))
            elif orientation == 270.0:
                bottom_port_offset = max(bottom_port_offset, float(port_length))
            # gds_center = _device_to_gds_coords(
            #     port_location_um, mask.shape, grid_step
            # )
            x0, y0 = float(port_location_um[0]), float(port_location_um[1])
            width_um = float(width_um)
            if axis == "x":
                rect_w, rect_h = port_length, width_um
                cx = x0 + (
                    port_length / 2.0 if orientation == 180.0 else -port_length / 2.0
                )
                cy = y0
            else:
                rect_w, rect_h = width_um, port_length
                cx = x0
                cy = y0 + (
                    port_length / 2.0 if orientation == 270.0 else -port_length / 2.0
                )
            half_w = rect_w / 2.0
            half_h = rect_h / 2.0
            final_gds.add_polygon(
                [
                    (cx - half_w, cy - half_h),
                    (cx + half_w, cy - half_h),
                    (cx + half_w, cy + half_h),
                    (cx - half_w, cy + half_h),
                ],
                layer=(1, 0),
            )

            final_gds.add_port(
                name=port_name,
                center=port_location_um,
                width=width_um,
                orientation=orientation,
                layer=(1, 0),
                port_type="optical",
            )
            port_metadata[port_name] = dict(
                center_um=[float(v) for v in port_cfg["center"]],
                size_um=[float(v) for v in port_cfg["size"]],
                inferred_port_location_um=[float(v) for v in port_location_um],
                # inferred_port_location_gds_um=[float(v) for v in gds_center],
                inferred_axis=axis,
                inferred_orientation_deg=float(orientation),
                inferred_width_um=float(width_um),
            )
        # eps_component.info["port_metadata"] = port_metadata
        # eps_component.info["nm_per_pixel"] = float(target_nm_per_pixel)
        # eps_component.info["source_nm_per_pixel"] = float(nm_per_pixel)
        # smatrix_cfg = self.obj_cfgs.get("smatrix")
        # if isinstance(smatrix_cfg, dict):
        #     eps_component.info["smatrix"] = copy.deepcopy(smatrix_cfg)
        final_gds.info["port_metadata"] = port_metadata
        final_gds.info["nm_per_pixel"] = float(target_nm_per_pixel)
        final_gds.info["source_nm_per_pixel"] = float(nm_per_pixel)
        smatrix_cfg = self.obj_cfgs.get("smatrix")
        smatrix_breakdown = getattr(self.objective, "breakdown", {}).get("smatrix")
        if (
            isinstance(smatrix_cfg, dict)
            and isinstance(smatrix_breakdown, dict)
            and "value" in smatrix_breakdown
        ):

            def _serialize_smatrix_info(value):
                if isinstance(value, Tensor):
                    value = value.detach().cpu().numpy()
                elif isinstance(value, ArrayBox):
                    value = value._value

                if isinstance(value, np.ndarray):
                    return [_serialize_smatrix_info(v) for v in value.tolist()]
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, (list, tuple)):
                    return [_serialize_smatrix_info(v) for v in value]
                if isinstance(value, dict):
                    return {
                        key: _serialize_smatrix_info(item)
                        for key, item in value.items()
                    }
                return value

            final_gds.info["smatrix"] = {
                "value": _serialize_smatrix_info(smatrix_breakdown["value"]),
                "in_mode": _serialize_smatrix_info(smatrix_cfg.get("in_mode")),
                "out_mode": _serialize_smatrix_info(smatrix_cfg.get("out_modes")),
                "temp": _serialize_smatrix_info(smatrix_cfg.get("temp")),
                "wl": _serialize_smatrix_info(smatrix_cfg.get("wl")),
            }
        print(port_metadata)

        comp_ref = final_gds.add_ref(eps_component)
        offset_x = left_port_offset
        offset_y = bottom_port_offset + dy
        if hasattr(comp_ref, "dmove"):
            comp_ref.dmove((offset_x, offset_y))
        elif hasattr(comp_ref, "move"):
            comp_ref.move(origin=(0.0, 0.0), destination=(offset_x, offset_y))
        else:
            raise RuntimeError(
                "Cannot move component reference; unsupported gdsfactory reference API."
            )
        # Write the GDS file
        final_gds.write_gds(gdspath=os.path.join(self.sim_cfg["plot_root"], filename))

    def dump_data(
        self,
        filename_h5,
        filename_yml,
        step,
        *,
        use_high_res_eps: bool = True,
        binarize_eps: bool = True,
    ):
        """
        switch to another different dump_data function
        for multiple times of shining the source
        before, we store them into one single h5 file with different keys to access them
        now, we want to store them into different h5 files just like in NeurOLight where the separate h5 files according to the input port

        the only difference should be before the gradient stored is the total gradient calculated from the two forward simulations

        now we need to seperate the gradient into two parts, one for each forward simulation and store them into different h5 files
        """
        # print("grad fn of self._eps_map", self._eps_map.grad_fn)
        # print("grad of self._eps_map", self._eps_map.grad)
        complex_type = [torch.complex64, torch.complex32, torch.complex128]
        filename_base = filename_h5[:-3]
        dir_path = os.path.dirname(filename_h5)
        ensure_dir(dir_path)
        with torch.no_grad():
            eps_tensor = self._hr_eps_map if use_high_res_eps else self._eps_map
            if isinstance(eps_tensor, Tensor):
                eps_tensor = eps_tensor.detach().clone()
            elif isinstance(eps_tensor, np.ndarray):
                eps_tensor = torch.from_numpy(np.array(eps_tensor, copy=True))
            elif isinstance(eps_tensor, ArrayBox):
                eps_tensor = torch.from_numpy(np.array(eps_tensor._value, copy=True))
            else:
                raise TypeError(f"Unsupported eps map type: {type(eps_tensor)}")

            if binarize_eps:
                eps_bg_global = getattr(self.device, "eps_bg", None)
                if eps_bg_global is None:
                    raise AttributeError(
                        "Device is missing attribute 'eps_bg' required for binarization."
                    )
                eps_bg_global = float(
                    eps_bg_global.item()
                    if isinstance(eps_bg_global, torch.Tensor)
                    else eps_bg_global
                )

                eps_high_candidates = []
                for region_cfg in self.device.design_region_cfgs.values():
                    eps_eff = region_cfg.get("eps_eff", region_cfg.get("eps", None))
                    if eps_eff is None:
                        raise KeyError(
                            "'eps_eff' or 'eps' not found in design region configuration."
                        )
                    eps_high_candidates.append(
                        float(
                            eps_eff.item()
                            if isinstance(eps_eff, torch.Tensor)
                            else eps_eff
                        )
                    )

                if not eps_high_candidates:
                    raise ValueError(
                        "Unable to determine high permittivity value for binarization."
                    )

                eps_high_global = max(eps_high_candidates)
                global_threshold = 0.5 * (eps_high_global + eps_bg_global)
                hi_global = torch.tensor(
                    eps_high_global, dtype=eps_tensor.dtype, device=eps_tensor.device
                )
                lo_global = torch.tensor(
                    eps_bg_global, dtype=eps_tensor.dtype, device=eps_tensor.device
                )
                eps_tensor = torch.where(
                    eps_tensor >= global_threshold, hi_global, lo_global
                )

            eps_dump = eps_tensor.detach().cpu().numpy()
            adj_srcs, fields_adj, field_adj_normalizer = (
                self.objective.obtain_adj_srcs()
            )
            gradients = self.objective.read_gradient()
            # the for loop shoul according to the keys of the solutions
            for (
                SliceName,
                WaveLen,
                SrcMode,
                Temperture,
            ), fields in self.objective.solutions.items():
                filename = (
                    filename_base + f"-{SliceName}-{WaveLen}-{SrcMode}-{Temperture}.h5"
                )
                with h5py.File(filename, "w") as f:
                    # eps
                    f.create_dataset("eps_map", data=eps_dump)  # 2d numpy array
                    # eps_unique_count = np.unique(eps_dump).size
                    # f.attrs["eps_map_unique_count"] = int(eps_unique_count)
                    # print(f"eps_map unique values: {eps_unique_count}")
                    # all the slices
                    for slice_name, slice in self.device.port_monitor_slices.items():
                        if isinstance(slice, np.ndarray):
                            f.create_dataset(f"port_slice-{slice_name}", data=slice)
                        else:
                            f.create_dataset(f"port_slice-{slice_name}_x", data=slice.x)
                            f.create_dataset(f"port_slice-{slice_name}_y", data=slice.y)
                    # only the source I care
                    for slice_name, source_profile in self.norm_run_profiles.items():
                        for key in list(source_profile.keys()):
                            if isinstance(key, str):
                                continue
                            wl, mode = key
                            profile = source_profile[key]
                            if isinstance(profile[0], np.ndarray):
                                src_mode = profile[0].astype(np.complex64)
                                ht_m = profile[1].astype(np.complex64)
                                et_m = profile[2].astype(np.complex64)
                            if isinstance(profile[0], Tensor):
                                if profile[0].dtype in complex_type:
                                    profile[0] = profile[0].to(torch.complex64)
                                    profile[1] = profile[1].to(torch.complex64)
                                    profile[2] = profile[2].to(torch.complex64)
                                src_mode = profile[0].detach().cpu().numpy()
                                ht_m = profile[1].detach().cpu().numpy()
                                et_m = profile[2].detach().cpu().numpy()
                            if isinstance(profile[0], ArrayBox):
                                src_mode = profile[0]._value
                                ht_m = profile[1]._value
                                et_m = profile[2]._value
                            if (
                                slice_name == SliceName
                                and wl == WaveLen
                                and mode == SrcMode
                            ):
                                f.create_dataset(
                                    f"source_profile",
                                    data=src_mode,
                                )
                            f.create_dataset(
                                f"ht_m-wl-{wl}-slice-{slice_name}-mode-{mode}",
                                data=ht_m,
                            )
                            f.create_dataset(
                                f"et_m-wl-{wl}-slice-{slice_name}-mode-{mode}",
                                data=et_m,
                            )
                    fields = self.objective.solutions[
                        (SliceName, WaveLen, SrcMode, Temperture)
                    ]
                    store_fields = {}
                    for key, field in fields.items():
                        if isinstance(fields[key], Tensor):
                            if fields[key].dtype in complex_type:
                                fields[key] = fields[key].to(torch.complex64)
                            store_fields[key] = fields[key].detach().cpu().numpy()
                        if isinstance(fields[key], ArrayBox):
                            store_fields[key] = fields[key]._value
                    store_fields = np.stack(
                        (store_fields["Hx"], store_fields["Hy"], store_fields["Ez"]),
                        axis=0,
                    )
                    f.create_dataset(
                        f"field_solutions",
                        data=store_fields,
                    )  # 3d numpy array
                    # only the A matrix I care
                    A = self.objective.As[(WaveLen, Temperture)]
                    Alist = []
                    for item in A:
                        if isinstance(item, Tensor):
                            Alist.append(item.detach().cpu().numpy())
                        elif isinstance(item, ArrayBox):
                            Alist.append(item._value)
                        elif isinstance(item, np.ndarray):
                            Alist.append(item)
                        else:
                            raise ValueError(
                                f"A is not a tensor, arraybox or numpy array, the type is {type(item)}"
                            )
                    f.create_dataset(f"A-entries_a", data=Alist[0])
                    f.create_dataset(f"A-indices_a", data=Alist[1])
                    # save all the s_params
                    for (
                        input_slice_name,
                        slice_name,
                        obj_type,
                        wl,
                        in_mode,
                        temp,
                    ), s_params in self.objective.s_params.items():
                        # if wl != WaveLen or temp != Temperture or input_slice_name != SliceName or in_mode != SrcMode:
                        #     continue
                        # the obj_type is a string, if it is an integer, it implys the eigenmode type and the value is the mode index
                        store_s_params = {}
                        for key, s_param in s_params.items():
                            if isinstance(s_param, Tensor):
                                if s_param.dtype in complex_type:
                                    s_param = s_param.to(torch.complex64)
                                store_s_params[key] = s_param.detach().cpu().numpy()
                            if isinstance(s_param, ArrayBox):
                                store_s_params = s_param._value
                        if "s_p" in store_s_params.keys():
                            store_s_params = np.stack(
                                (store_s_params["s_p"], store_s_params["s_m"]), axis=0
                            )
                        else:
                            store_s_params = store_s_params["s"]
                        f.create_dataset(
                            f"s_params-obj_slice_name-{slice_name}-type-{obj_type}-in_slice_name-{input_slice_name}-wl-{wl}-in_mode-{in_mode}-temp-{temp}",
                            data=store_s_params,
                        )  # 3d numpy array
                    # only the adj_src I care
                    adj_src = adj_srcs[(WaveLen, SrcMode[:2], Temperture)]
                    J_adj = adj_src[(SliceName, SrcMode, Temperture)]
                    J_adj = J_adj.reshape(self.epsilon_map.shape)
                    if isinstance(J_adj, Tensor):
                        if J_adj.dtype in complex_type:
                            J_adj = J_adj.to(torch.complex64)
                        J_adj = J_adj.detach().cpu().numpy()
                    if isinstance(J_adj, ArrayBox):
                        J_adj = J_adj._value
                    f.create_dataset(f"adj_src", data=J_adj)
                    # only the fields_adj I care
                    field = fields_adj[(WaveLen, SrcMode[:2], Temperture)][
                        (SliceName, SrcMode, Temperture)
                    ]
                    store_fields = {}
                    for components_key, component in field.items():
                        if isinstance(component, Tensor):
                            if component.dtype in complex_type:
                                component = component.to(torch.complex64)
                            store_fields[components_key] = (
                                component.detach().cpu().numpy()
                            )
                        if isinstance(component, ArrayBox):
                            store_fields[components_key] = component._value
                    store_fields = np.stack(
                        (
                            store_fields["Hx"],
                            store_fields["Hy"],
                            store_fields["Ez"],
                        ),
                        axis=0,
                    )
                    f.create_dataset(
                        f"fields_adj",
                        data=store_fields,
                    )  # 3d numpy array
                    # only the field_adj_normalizer I care
                    normalizer = field_adj_normalizer[
                        (WaveLen, SrcMode[:2], Temperture)
                    ][(SliceName, SrcMode, Temperture)]
                    if isinstance(normalizer, Tensor):
                        if normalizer.dtype in complex_type:
                            normalizer = normalizer.to(torch.complex64)
                        normalizer = normalizer.detach().cpu().numpy()
                    if isinstance(normalizer, ArrayBox):
                        normalizer = normalizer._value
                    f.create_dataset(
                        f"field_adj_normalizer",
                        data=normalizer,
                    )  # 2d numpy array
                    # all the design region mask
                    for (
                        design_region_name,
                        design_region_mask,
                    ) in self.design_region_masks.items():
                        f.create_dataset(
                            f"design_region_mask-{design_region_name}_x_start",
                            data=design_region_mask.x.start,
                        )
                        f.create_dataset(
                            f"design_region_mask-{design_region_name}_x_stop",
                            data=design_region_mask.x.stop,
                        )
                        f.create_dataset(
                            f"design_region_mask-{design_region_name}_y_start",
                            data=design_region_mask.y.start,
                        )
                        f.create_dataset(
                            f"design_region_mask-{design_region_name}_y_stop",
                            data=design_region_mask.y.stop,
                        )
                    # store the total gradient
                    f.create_dataset(
                        "total_gradient", data=self._eps_map.grad.detach().cpu().numpy()
                    )
                    # only the gradient I care
                    # not the total gradient, but the gradient from this specific forward simulation
                    if isinstance(
                        gradients[(WaveLen, SrcMode[:2], Temperture)][
                            (SliceName, SrcMode, Temperture)
                        ],
                        torch.Tensor,
                    ):
                        grad = (
                            gradients[(WaveLen, SrcMode[:2], Temperture)][
                                (SliceName, SrcMode, Temperture)
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    f.create_dataset("gradient", data=grad)
                    # we don't store the breakdown of the objective for now since we don't need to plot the distribution
                    # # for bending, we still save the fom:
                    # for name, item in self.objective.breakdown.items(): # store the breakdown of the objective
                    #     f.create_dataset(f"breakdown_{name}_weight", data=item["weight"])
                    #     f.create_dataset(f"breakdown_{name}_value", data=float(item["value"].item()))
        # in the following code, we just store the config files so we don't need to change them
        # Check if the file exists using os.path.exists
        if os.path.exists(filename_yml):
            # File exists, read its content
            with open(filename_yml, "r") as f:
                existing_data = (
                    ryaml.load(f) or {}
                )  # Load existing data or use an empty dict if file is empty
        else:
            # File does not exist, start with an empty dictionary
            existing_data = {}
            existing_data.update(self._cfgs.dict())
            existing_data["port_cfgs"] = self.device.port_cfgs
            existing_data["design_region_cfgs"] = self.device.design_region_cfgs
            existing_data["obj_cfgs"]["_fusion_func"] = existing_data["obj_cfgs"][
                "_fusion_func"
            ].__name__
            for key, value in existing_data["obj_cfgs"].items():
                if isinstance(value, dict):
                    value["out_modes"] = list(value["out_modes"])

        # Update the existing data with the new data
        opt_step = step
        existing_data[f"sharpness_{opt_step}"] = self.current_sharpness
        existing_data[f"parameters_{opt_step}"] = {
            name: param.clone().detach().cpu().numpy().tolist()
            for name, param in self.named_parameters()
        }

        # Write the data to the file
        with open(filename_yml, "w") as f:
            yaml.dump(existing_data, f)

    def get_design_region_eps_dict(self):
        design_region_eps_dict = {}
        for key, design_region in self._design_region_eps_dict.items():
            design_region_eps_dict[key] = design_region.clone().detach()
        return design_region_eps_dict

    def switch_solver(self, neural_solver, numerical_solver, use_autodiff=False):
        self.objective.switch_solver(neural_solver, numerical_solver, use_autodiff)

    def forward(
        self,
        sharpness: float = 1,
        ls_knots: dict = None,
        custom_source: dict = None,
    ):
        # eps_map, design_region_eps_dict = self.build_device(sharpness)
        self.current_sharpness = sharpness
        (
            eps_map,
            design_region_eps_dict,
            hr_eps_map,
            hr_design_region_eps_dict,
            design_region_material_dict,
        ) = self.build_device(sharpness, ls_knots)
        self._design_region_eps_dict = design_region_eps_dict
        self._runtime_thermal_material_maps = (
            self._build_runtime_thermal_material_maps_from_regions(
                design_region_material_dict
            )
        )
        self._retain_grad_in_structure(self._runtime_thermal_material_maps)
        self.objective.runtime_material_maps = self._runtime_thermal_material_maps
        ## need to create objective layer during forward, because all Simulations need to know the latest permittivity_list

        self._eps_map = eps_map
        if self._eps_map.requires_grad:
            self._eps_map.retain_grad()

        self._hr_eps_map = hr_eps_map
        obj = self.objective_layer(
            [eps_map], custom_source=custom_source
        )  # loss = loss_function(output, target)
        self._snapshot_runtime_thermal_states()
        self._obj = obj
        results = {"obj": obj, "breakdown": self.objective.breakdown}
        ## return design region epsilons and the final epsilon map for other penalty loss calculation
        results.update(design_region_eps_dict)
        results.update({"eps_map": eps_map})

        return results

    def evaluation(self, image_path, raw_data=None, read_S_parameter=False):
        """Evaluate a layout supplied as a PNG image or an HDF5 eps_map."""
        if len(self.device.design_region_masks) != 1:
            raise NotImplementedError(
                "Image-based evaluation currently supports a single design region."
            )

        true_s_matrix = None
        if read_S_parameter:
            with h5py.File(raw_data, "r") as handle:
                # Capture every S-parameter dataset stored in the HDF5 file.
                raw_sparams = {}
                sparam_entries = {}
                max_out_idx = 0
                max_in_idx = 0
                for name, dataset in handle.items():
                    if not name.startswith("s_params"):
                        continue
                    data = np.array(dataset)
                    if data.size == 0:
                        continue
                    value = complex(data.flat[0])
                    raw_sparams[name] = value
                    out_match = re.search(r"out_slice_(\d+)", name)
                    in_match = re.search(r"in_slice_(\d+)", name)
                    if not (out_match and in_match):
                        continue
                    out_idx = int(out_match.group(1))
                    in_idx = int(in_match.group(1))
                    max_out_idx = max(max_out_idx, out_idx)
                    max_in_idx = max(max_in_idx, in_idx)
                    # sparam_entries[(out_idx, in_idx)] = value
                    sparam_entries[(in_idx, out_idx)] = value

                if sparam_entries:
                    true_s_matrix = np.full(
                        (max_in_idx, max_out_idx), np.nan + 0j, dtype=np.complex128
                    )
                    for (in_idx, out_idx), value in sparam_entries.items():
                        true_s_matrix[in_idx - 1, out_idx - 1] = value

        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        region_name, region_mask = next(iter(self.device.design_region_masks.items()))
        region_cfg = self.device.design_region_cfgs[region_name]
        eps_lo = torch.tensor(
            float(region_cfg["eps_bg"]),
            dtype=self.epsilon_map.dtype,
            device=self.operation_device,
        )
        eps_hi = torch.tensor(
            float(region_cfg["eps"]),
            dtype=self.epsilon_map.dtype,
            device=self.operation_device,
        )

        from PIL import Image

        img = Image.open(image_path).convert("L")
        density = np.asarray(img, dtype=np.float32) / 255.0
        density_tensor = torch.from_numpy(density).to(
            device=self.operation_device, dtype=self.epsilon_map.dtype
        )
        full_map = eps_lo + density_tensor * (eps_hi - eps_lo)

        hr_eps_fullmap = full_map
        # target_size = (
        #     region_mask.x.stop - region_mask.x.start,
        #     region_mask.y.stop - region_mask.y.start,
        # )
        # if min(target_size) <= 0:
        #     raise ValueError(
        #         f"Invalid target size derived from region mask: {target_size}"
        #     )

        # print(hr_eps_fullmap.shape)
        # print(target_size)
        target_size_full = (self.epsilon_map.shape[-2], self.epsilon_map.shape[-1])

        eps_span = eps_hi - eps_lo
        if torch.allclose(eps_span, torch.zeros_like(eps_span)):
            raise ValueError(
                "High and low permittivity values are identical; cannot normalize."
            )
        normalized = ((hr_eps_fullmap - eps_lo) / eps_span).clamp(0.0, 1.0)
        # hr_region_mask = self.hr_device.design_region_masks[region_name]
        # normalized = ((hr_eps_fullmap - eps_lo) / eps_span).clamp(0.0, 1.0)[
        #     hr_region_mask
        # ]
        src_res = int(self.hr_device.sim_cfg["resolution"])
        tar_res = int(round(1000 / src_res)) * src_res
        hr_size = [
            max(1, int(round(dim * tar_res / src_res))) for dim in normalized.shape[-2:]
        ]
        hr_size = [
            max(1, int(round(size / tgt) * tgt)) if tgt != 0 else size
            for size, tgt in zip(hr_size, target_size_full)
        ]

        # print(normalized.shape)
        # print(hr_size)
        normalized = _convert_resolution(
            normalized,
            intplt_mode="nearest",
            target_size=hr_size,
        )
        normalized = _convert_resolution(
            normalized,
            subpixel_smoothing=True,
            eps_r=float(eps_hi.detach().cpu().item()),
            eps_bg=float(eps_lo.detach().cpu().item()),
            target_size=target_size_full,
        )
        low_res_region = eps_lo + normalized * eps_span
        # print(low_res_region.shape)
        eps_map = self.epsilon_map.clone()
        # eps_map[region_mask] = low_res_region
        # eps_map[region_mask] = low_res_region[region_mask]
        eps_map = low_res_region
        self._eps_map = eps_map
        with torch.no_grad():
            obj = self.objective_layer([eps_map])

        results = {"obj": obj, "breakdown": self.objective.breakdown}
        if true_s_matrix is not None:
            results["true_s_matrix"] = true_s_matrix
        return results

    def simulation(self, image_path, save_mask=False):
        """Evaluate a layout supplied as a PNG image or an HDF5 eps_map."""
        if len(self.device.design_region_masks) != 1:
            raise NotImplementedError(
                "Image-based evaluation currently supports a single design region."
            )

        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        region_name, region_mask = next(iter(self.device.design_region_masks.items()))
        region_cfg = self.device.design_region_cfgs[region_name]
        eps_lo = torch.tensor(
            float(region_cfg["eps_bg"]),
            dtype=self.epsilon_map.dtype,
            device=self.operation_device,
        )
        eps_hi = torch.tensor(
            float(region_cfg["eps"]),
            dtype=self.epsilon_map.dtype,
            device=self.operation_device,
        )

        img = Image.open(image_path).convert("L")
        density = np.asarray(img, dtype=np.float32) / 255.0
        density_tensor = torch.from_numpy(density).to(
            device=self.operation_device, dtype=self.epsilon_map.dtype
        )
        full_map = eps_lo + density_tensor * (eps_hi - eps_lo)

        hr_eps_fullmap = full_map
        target_size = (
            region_mask.x.stop - region_mask.x.start,
            region_mask.y.stop - region_mask.y.start,
        )
        if min(target_size) <= 0:
            raise ValueError(
                f"Invalid target size derived from region mask: {target_size}"
            )

        # print(target_size)
        eps_span = eps_hi - eps_lo
        if torch.allclose(eps_span, torch.zeros_like(eps_span)):
            raise ValueError(
                "High and low permittivity values are identical; cannot normalize."
            )

        normalized = ((hr_eps_fullmap - eps_lo) / eps_span).clamp(0.0, 1.0)
        if save_mask:
            hr_region_mask = self.hr_device.design_region_masks[region_name]
            hr_eps_map = self.hr_eps_map.clone()
            hr_eps_map[hr_region_mask] = eps_lo + normalized * eps_span
            self.hr_eps_map = hr_eps_map
            self._hr_eps_map = hr_eps_map
        src_res = int(self.hr_device.sim_cfg["resolution"])
        tar_res = int(round(1000 / src_res)) * src_res
        hr_size = [
            max(1, int(round(dim * tar_res / src_res))) for dim in normalized.shape[-2:]
        ]
        # print(hr_size)
        hr_size = [
            max(1, int(round(size / tgt) * tgt)) if tgt != 0 else size
            for size, tgt in zip(hr_size, target_size)
        ]
        # print(self.hr_eps_map[hr_region_mask].shape)
        # print(normalized.shape)
        # print(hr_size)
        # self.hr_eps_map[hr_region_mask] = normalized

        normalized = _convert_resolution(
            normalized,
            intplt_mode="nearest",
            target_size=hr_size,
        )
        # print(normalized.shape)
        normalized = _convert_resolution(
            normalized,
            subpixel_smoothing=True,
            eps_r=float(eps_hi.detach().cpu().item()),
            eps_bg=float(eps_lo.detach().cpu().item()),
            target_size=target_size,
        )
        # print(normalized.shape)
        low_res_region = eps_lo + normalized * eps_span
        # print("low_res_region:", low_res_region.shape)

        eps_map = self.epsilon_map.clone()
        eps_map[region_mask] = low_res_region
        self.epsilon_map[region_mask] = low_res_region
        self._eps_map = eps_map

        with torch.no_grad():
            obj = self.objective_layer([eps_map])

        results = {"obj": obj, "breakdown": self.objective.breakdown}
        return results
