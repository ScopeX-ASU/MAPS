"""
Date: 2024-10-04 18:47:39
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-07 04:18:59
FilePath: /Metasurface-Opt/core/models/metamirror.py
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
                        rho_resolution=[20, 20],
                        transform=[dict(type="mirror_symmetry", dims=[1])],
                        init_method="rectangle",
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
                    border_width=[0, 0, 1.5, 1.5],
                    PML=[1, 1],
                    cell_size=None,
                    resolution=100,
                    wl_cen=1.55,
                    wl_width=0,
                    n_wl=1,
                    plot_root="./figs/isolator",
                ),
                obj_cfgs=dict(
                    fwd_trans=dict(
                        weight=1,
                        #### objective is evaluated at this port
                        in_port_name="in_port_1",
                        out_port_name="out_port_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            3,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="eigenmode",
                        direction="x+",
                    ),
                    bwd_trans=dict(
                        weight=-5,
                        #### objective is evaluated at this port
                        in_port_name="out_port_1",
                        out_port_name="in_port_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    fwd_refl_trans=dict(
                        weight=-1,
                        #### objective is evaluated at this port
                        in_port_name="in_port_1",
                        out_port_name="refl_port_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux_minus_src",
                        direction="x",
                    ),
                    bwd_refl_trans=dict(
                        weight=-1,
                        #### objective is evaluated at this port
                        in_port_name="out_port_1",
                        out_port_name="refl_port_2",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux_minus_src",
                        direction="x",
                    ),
                    fwd_rad_trans_xp=dict(
                        weight=-2,
                        #### objective is evaluated at this port
                        in_port_name="in_port_1",
                        out_port_name="rad_monitor_xp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    fwd_rad_trans_xm=dict(
                        weight=-2,
                        #### objective is evaluated at this port
                        in_port_name="in_port_1",
                        out_port_name="rad_monitor_xm",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    fwd_rad_trans_yp=dict(
                        weight=-2,
                        #### objective is evaluated at this port
                        in_port_name="in_port_1",
                        out_port_name="rad_monitor_yp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                    fwd_rad_trans_ym=dict(
                        weight=-2,
                        #### objective is evaluated at this port
                        in_port_name="in_port_1",
                        out_port_name="rad_monitor_ym",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                    bwd_rad_trans_xp=dict(
                        weight=2,
                        #### objective is evaluated at this port
                        in_port_name="out_port_1",
                        out_port_name="rad_monitor_xp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    bwd_rad_trans_xm=dict(
                        weight=2,
                        #### objective is evaluated at this port
                        in_port_name="out_port_1",
                        out_port_name="rad_monitor_xm",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    bwd_rad_trans_yp=dict(
                        weight=2,
                        #### objective is evaluated at this port
                        in_port_name="out_port_1",
                        out_port_name="rad_monitor_yp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                    bwd_rad_trans_ym=dict(
                        weight=2,
                        #### objective is evaluated at this port
                        in_port_name="out_port_1",
                        out_port_name="rad_monitor_ym",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                ),
            )
        )


class IsolatorOptimization(BaseOptimization):
    def __init__(
        self,
        device,
        design_region_param_cfgs=dict(),
        sim_cfg: dict = dict(),
        obj_cfgs=dict(),
        operation_device=torch.device("cuda:0"),
    ):
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
            design_region_param_cfgs=cfgs.design_region_param_cfgs,
            sim_cfg=cfgs.sim_cfg,
            obj_cfgs=cfgs.obj_cfgs,
            operation_device=operation_device,
        )
