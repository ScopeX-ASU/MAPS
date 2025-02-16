"""
Date: 2024-10-04 18:47:39
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-16 16:43:56
FilePath: /MAPS/core/invdes/models/metamirror.py
"""

import torch

from .base_optimization import BaseOptimization, DefaultOptimizationConfig


class DefaultConfig(DefaultOptimizationConfig):
    def __init__(self):
        super().__init__()
        self.update(
            dict(
                design_region_param_cfgs=dict(
                    design_region_1=dict(
                        method="levelset",
                        rho_resolution=[50, 0],
                        interpolation="bilinear",
                        transform=[],
                        init_method="random",
                        binary_projection=dict(
                            fw_threshold=100,
                            bw_threshold=100,
                            mode="regular",
                        ),
                    )
                ),
                sim_cfg=dict(
                    solver="ceviche",
                    binary_projection=dict(
                        fw_threshold=100,
                        bw_threshold=100,
                        mode="regular",
                    ),
                    border_width=[0, 1.5, 0, 1.5],
                    PML=[1, 1],
                    cell_size=None,
                    resolution=100,
                    wl_cen=1.55,
                    wl_width=0,
                    n_wl=1,
                    plot_root="./figs/metamirror",
                ),
                obj_cfgs=dict(
                    fwd_trans=dict(
                        weight=1,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="out_slice_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="eigenmode",
                        direction="y-",
                        # type="flux",
                        # direction="y",
                    ),
                    refl_trans=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="refl_slice_1",
                        wl=[1.55],
                        temp=[300],
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux_minus_src",
                        direction="x",
                    ),
                    rad_trans_xp=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_monitor_xp",
                        wl=[1.55],
                        temp=[300],
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    rad_trans_xm=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_monitor_xm",
                        wl=[1.55],
                        temp=[300],
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    rad_trans_yp=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_monitor_yp",
                        wl=[1.55],
                        temp=[300],
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                    rad_trans_ym=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_monitor_ym",
                        wl=[1.55],
                        temp=[300],
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                ),
            )
        )


class MetaMirrorOptimization(BaseOptimization):
    def __init__(
        self,
        device,
        hr_device,
        design_region_param_cfgs=dict(),
        sim_cfg: dict = dict(),
        obj_cfgs=dict(),
        operation_device=torch.device("cuda:0"),
    ):  
        design_region_param_cfgs_copy = design_region_param_cfgs.copy()
        design_region_param_cfgs = dict()
        for region_name in device.design_region_cfgs.keys():
            design_region_param_cfgs[region_name] = dict(
                method=design_region_param_cfgs_copy.get("method", "levelset"),
                rho_resolution=design_region_param_cfgs_copy.get("rho_resolution", [50, 0]),
                interpolation=design_region_param_cfgs_copy.get("interpolation", "bilinear"),
                transform=design_region_param_cfgs_copy.get("transform", []),
                init_method=design_region_param_cfgs_copy.get("init_method", "random"),
                binary_projection=design_region_param_cfgs_copy.get(
                    "binary_projection",
                    dict(
                        fw_threshold=100,
                        bw_threshold=100,
                        mode="regular",
                    ),
                ),
            )
            
        cfgs = DefaultConfig()  ## this is default configurations
        ## here we accept new configurations and update the default configurations
        cfgs.update(
            dict(
                design_region_param_cfgs=design_region_param_cfgs,
                sim_cfg=sim_cfg,
                obj_cfgs=obj_cfgs,
            )
        )

        super().__init__(
            device=device,
            hr_device=hr_device,
            design_region_param_cfgs=cfgs.design_region_param_cfgs,
            sim_cfg=cfgs.sim_cfg,
            obj_cfgs=cfgs.obj_cfgs,
            operation_device=operation_device,
        )
