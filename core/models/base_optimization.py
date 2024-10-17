"""
Date: 2024-10-04 18:49:06
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-06 00:38:40
FilePath: /Metasurface-Opt/core/models/base_optimization.py
"""

import copy
import os
from typing import List, Tuple

import numpy as np
import torch
from ceviche.constants import C_0
from pyutils.config import Config
from pyutils.general import logger
from torch import Tensor, nn
from torch.types import Device
from autograd.numpy.numpy_boxes import ArrayBox
from .layers.device_base import N_Ports
from .layers.fom_layer import SimulatedFoM
from .layers.parametrization import parametrization_builder
from .layers.utils import ObjectiveFunc, plot_eps_field
from .utils import nparray_as_real
import h5py
from copy import deepcopy

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
        loss = loss + obj["weight"] * obj["value"]
    extra_breakdown = {}
    return loss, extra_breakdown


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
                    fwd_trans=dict(
                        weight=1,
                        #### objective is evaluated at this port
                        in_port_name="in_port_1",
                        out_port_name="out_port_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="eigenmode",
                        direction="x+",
                    ),
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

        self.build_parameters()

        ### need to generate source/monitors
        device.init_monitors()

        ### need to run normalization run
        device.norm_run()
        self.norm_run_profiles = device.port_sources_dict # {input_slice_name: source_profiles 2d array, ...}

        ### pre-build objectives
        self.build_objective(
            port_profiles=self.device.port_sources_dict,
            port_slices=self.device.port_monitor_slices,
            epsilon_map=self.device.epsilon_map,
            obj_cfgs=self.obj_cfgs,
            solver=self.sim_cfg["solver"],
        )

    def reset_parameters(self):
        for design_region in self.design_region_param_dict.values():
            design_region.reset_parameters()

    def build_parameters(self):
        ### create design region parametrizations based on device and design_region_param_cfgs
        ## each design region has a name, and it is an nn.Module.
        ## its self.weights is a nn.ParameterDict which contains all its learnable parameters
        ## during initialization, it will build all parameters and run reset_parameters
        logger.info("Start building design region parametrizations ...")
        self.design_region_param_dict = parametrization_builder(
            device=self.device,
            hr_device=self.hr_device,
            sim_cfg=self.sim_cfg,
            parametrization_cfgs=self.design_region_param_cfgs,
        )  ## nn.ModuleDict = {region_name: nn.Module, ...}

        self.objective_layer = SimulatedFoM(self.cal_obj_grad, self.sim_cfg["solver"])

    def build_device(
        self,
        sharpness: float = 1,
    ):
        design_region_eps_dict = {}
        hr_design_region_eps_dict = {}
        for region_name, design_region in self.design_region_param_dict.items():
            ## obtain each design region's denormalized permittivity only in the design region
            hr_region, region = design_region(sharpness)
            design_region_eps_dict[region_name] = region
            hr_design_region_eps_dict[region_name] = hr_region

        ### then we need to fill in the permittivity of each design region to the whole device eps_map
        eps_map = self.epsilon_map.data.clone()
        hr_eps_map = self.hr_eps_map

        for region_name, design_region_eps in design_region_eps_dict.items():
            region_mask = self.design_region_masks[region_name]
            eps_map[region_mask] = design_region_eps
            hr_region_mask = self.hr_device.design_region_masks[region_name]
            hr_eps_map[hr_region_mask] = hr_design_region_eps_dict[region_name]

        return eps_map, design_region_eps_dict, hr_eps_map, hr_design_region_eps_dict

    def build_objective(
        self,
        port_profiles: dict,
        port_slices: dict,
        epsilon_map=None,
        obj_cfgs=dict(
            fwd_trans=dict(
                weight=1,
                #### objective is evaluated at this port
                in_port_name="in_port_1",
                out_port_name="out_port_1",
                #### objective is evaluated at all points by sweeping the wavelength and modes
                in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                out_modes=(
                    1,
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
        ## this is input source wavelength range, each wl needs to build a fdfd simulation
        wl_cen, wl_width, n_wl = sim_cfg["wl_cen"], sim_cfg["wl_width"], sim_cfg["n_wl"]
        simulations = {}
        for wl in np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl):
            omega = 2 * np.pi * C_0 / (wl * 1e-6)
            dl = self.device.grid_step * 1e-6
            sim = self.device.create_simulation(
                omega, dl, epsilon_map, self.device.NPML, solver
            )
            simulations[wl] = sim

        self.objective = ObjectiveFunc(
            simulations=simulations,
            port_profiles=port_profiles,
            port_slices=port_slices,
            grid_step=self.device.grid_step,
        )

        obj_cfgs = copy.deepcopy(obj_cfgs)
        self.objective.add_objective(obj_cfgs)

        ### create static backward computational graph from J to eps, no actual execution.'
        ### only usedful for autograd, not for torch autodiff
        self.gradient_region = "global_region"
        if self.sim_cfg["solver"] == "ceviche":
            self.objective.add_adj_objective(obj_cfgs)
            self.objective.build_jacobian()
            self.objective.build_adj_jacobian()

        return self.objective

    def cal_obj_grad(
        self,
        adjoint_mode: str = "ceviche",
        need_item: str = "need_value",
        resolution: int = None,
        permittivity_list: List[Tensor] = None,
        *args,
    ):
        ## here permittivity_list is a list of tensors (no grad required, since it is from autograd.Function)
        if adjoint_mode == "ceviche":
            total_value = self._cal_obj_grad_ceviche(
                need_item, [p.cpu().numpy() for p in permittivity_list], *args
            )
        elif adjoint_mode == "ceviche_torch":
            total_value = self._cal_obj_grad_ceviche(
                need_item, permittivity_list, *args
            )
        else:
            raise ValueError(f"Unsupported adjoint mode: {adjoint_mode}")

        return total_value

    def _cal_obj_grad_ceviche(
        self, need_item, permittivity_list: List[np.ndarray | Tensor], *args
    ):
        ## here permittivity_list is a list of tensors (no grad required, since it is from autograd.Function)
        permittivity = permittivity_list[0]

        if need_item == "need_value":
            total_value = self.objective(
                permittivity, mode="forward"
            ) 
        elif need_item == "need_gradient":
            ### this is explicitly called for autograd, not needed for torch autodiff
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
        field_key: Tuple = ("in_port_1", 1.55, 1),
        field_component: str = "Ez",
        in_port_name: str = "in_port_1",
        exclude_port_names: List[str] = [],
    ):
        Ez = self.objective.solutions[field_key][field_component]
        monitors = []
        for name, m in self.device.port_monitor_slices.items():
            if name in exclude_port_names:
                continue
            if name == in_port_name:
                color = "r"
            elif name.startswith("rad_"):
                color = "g"
            else:
                color = "b"
            monitors.append((m, color))
        eps_map = eps_map if eps_map is not None else self._eps_map
        obj = obj if obj is not None else self._obj
        if isinstance(obj, Tensor):
            obj = obj.item()
        if isinstance(Ez, ArrayBox):
            Ez = Ez._value
        plot_eps_field(
            Ez,
            eps_map.detach().cpu().numpy(),
            filepath=os.path.join(self.sim_cfg["plot_root"], plot_filename),
            monitors=monitors,
            x_width=self.device.cell_size[0],
            y_height=self.device.cell_size[1],
            NPML=self.device.NPML,
            title=f"|{field_component}|^2: {field_key}, FoM: {obj:.3f}",
            zoom_eps_factor=2,
        )

    def dump_data(self, filename):
        '''
        data needed to be dumped:
            1. eps_map (denormalized), downsample to different resolution
            2. E field, H field, corrresponding to different resolution eps_map
            3. Source_profile
            4. Scattering matrix
            5. gradient 
        '''
        # print("grad fn of self._eps_map", self._eps_map.grad_fn)
        # print("grad of self._eps_map", self._eps_map.grad)
        with torch.no_grad():
            with h5py.File(filename, 'w') as f:
                f.create_dataset('eps_map', data=self._eps_map.detach().cpu().numpy()) # 2d numpy array
                for port_name, source_profile in self.norm_run_profiles.items():
                    for (wl, mode), profile in source_profile.items():
                        if isinstance(profile[0], Tensor):
                            src_mode = profile[0].detach().cpu().numpy()
                        if isinstance(profile[0], ArrayBox):
                            src_mode = profile[0]._value
                        f.create_dataset(f'source_profile-wl-{wl}-port-{port_name}-mode-{mode}', data=src_mode)
                for (port_name, wl, mode), fields in self.objective.solutions.items():
                    store_fields = {}
                    for key, field in fields.items():
                        if isinstance(fields[key], Tensor):
                            store_fields[key] = fields[key].detach().cpu().numpy()
                        if isinstance(fields[key], ArrayBox):
                            store_fields[key] = fields[key]._value
                    store_fields = np.stack((store_fields["Hx"], store_fields["Hy"], store_fields["Ez"]), axis=0)
                    f.create_dataset(f'field_solutions-wl-{wl}-port-{port_name}-mode-{mode}', data=store_fields) # 3d numpy array
                for (port_name, wl, out_mode), s_params in self.objective.s_params.items():
                    store_s_params = {}
                    for key, s_param in s_params.items():
                        if isinstance(s_param, Tensor):
                            store_s_params[key] = s_param.detach().cpu().numpy()
                        if isinstance(s_param, ArrayBox):
                            store_s_params = s_param._value
                    store_s_params = np.stack((store_s_params["s_p"], store_s_params["s_m"]), axis=0)
                    f.create_dataset(f's_params-{port_name}-{wl}-{out_mode}', data=store_s_params) # 3d numpy array
                adj_srcs, fields_adj, field_adj_normalizer = self.objective.obtain_adj_srcs()
                for wl, adj_src in adj_srcs.items():
                    for (port_name, mode), b_adj in adj_src.items():
                        b_adj = b_adj.reshape(self.epsilon_map.shape)
                        if isinstance(b_adj, Tensor):
                            b_adj = b_adj.detach().cpu().numpy()
                        if isinstance(b_adj, ArrayBox):
                            b_adj = b_adj._value
                        f.create_dataset(f'adj_src-wl-{wl}-port-{port_name}-mode-{mode}', data=b_adj)
                for wl, fields in fields_adj.items():
                    for (port_name, mode), field in fields.items():
                        store_fields = {}
                        for components_key, component in field.items():
                            if isinstance(component, Tensor):
                                store_fields[components_key] = component.detach().cpu().numpy()
                            if isinstance(component, ArrayBox):
                                store_fields[components_key] = component._value
                        store_fields = np.stack((store_fields["Hx"], store_fields["Hy"], store_fields["Ez"]), axis=0)
                        f.create_dataset(f'fields_adj-wl-{wl}-port-{port_name}-mode-{mode}', data=store_fields) # 3d numpy array
                for wl, field_normalizer in field_adj_normalizer.items():
                    for (port_name, mode), normalizer in field_normalizer.items():
                        if isinstance(normalizer, Tensor):
                            normalizer = normalizer.detach().cpu().numpy()
                        if isinstance(normalizer, ArrayBox):
                            normalizer = normalizer._value
                        f.create_dataset(f'field_adj_normalizer-wl-{wl}-port-{port_name}-mode-{mode}', data=normalizer) # 2d numpy array
                if hasattr(self, 'current_eps_grad'):
                    if isinstance(self.current_eps_grad, ArrayBox):
                        self.current_eps_grad = self.current_eps_grad._value
                    f.create_dataset('gradient', data=self.current_eps_grad) # 2d numpy array
                else:
                    f.create_dataset('gradient', data=self._eps_map.grad.detach().cpu().numpy())
                for design_region_name, design_region_mask in self.design_region_masks.items(): 
                    f.create_dataset(f'design_region_mask-{design_region_name}_x_start', data=design_region_mask.x.start)
                    f.create_dataset(f'design_region_mask-{design_region_name}_x_stop', data=design_region_mask.x.stop)
                    f.create_dataset(f'design_region_mask-{design_region_name}_y_start', data=design_region_mask.y.start)
                    f.create_dataset(f'design_region_mask-{design_region_name}_y_stop', data=design_region_mask.y.stop)
            quit()

    def forward(
        self,
        sharpness: float = 1,
    ):
        # eps_map, design_region_eps_dict = self.build_device(sharpness)
        eps_map, design_region_eps_dict, hr_eps_map, hr_design_region_eps_dict = (
            self.build_device(sharpness)
        )
        ## need to create objective layer during forward, because all Simulations need to know the latest permittivity_list

        self._eps_map = eps_map
        self._eps_map.retain_grad()
        self._hr_eps_map = hr_eps_map
        obj = self.objective_layer([eps_map])
        self._obj = obj
        results = {"obj": obj, "breakdown": self.objective.breakdown}
        ## return design region epsilons and the final epsilon map for other penalty loss calculation
        results.update(design_region_eps_dict)
        results.update({"eps_map": eps_map})

        return results
