"""
Date: 2024-10-04 18:47:39
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-03-13 21:16:23
FilePath: /MAPS/core/invdes/models/optical_diode.py
"""

import torch

from .base_optimization import BaseOptimization, DefaultOptimizationConfig


class DefaultConfig(DefaultOptimizationConfig):
    def __init__(self):
        super().__init__()
        self.update(
            dict(
                design_region_param_cfgs=dict(),
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
                    # need to clarify the difference between the slice name and the port name
                    # all the objective should be evaluated at the slice not the port
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
                            "Ez3",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="eigenmode",
                        direction="x+",
                    ),
                    bwd_trans=dict(
                        weight=-5,
                        #### objective is evaluated at this port
                        in_slice_name="out_slice_1",
                        out_slice_name="in_slice_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    fwd_refl_trans=dict(
                        weight=-1,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="refl_slice_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux_minus_src",
                        direction="x",
                    ),
                    bwd_refl_trans=dict(
                        weight=-1,
                        #### objective is evaluated at this port
                        in_slice_name="out_slice_1",
                        out_slice_name="refl_slice_2",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux_minus_src",
                        direction="x",
                    ),
                    fwd_rad_trans_xp=dict(
                        weight=-2,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_slice_xp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    fwd_rad_trans_xm=dict(
                        weight=-2,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_slice_xm",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    fwd_rad_trans_yp=dict(
                        weight=-2,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_slice_yp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                    fwd_rad_trans_ym=dict(
                        weight=-2,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_slice_ym",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                    bwd_rad_trans_xp=dict(
                        weight=2,
                        #### objective is evaluated at this port
                        in_slice_name="out_slice_1",
                        out_slice_name="rad_slice_xp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    bwd_rad_trans_xm=dict(
                        weight=-0.1,
                        #### objective is evaluated at this port
                        in_slice_name="out_slice_1",
                        out_slice_name="rad_slice_xm",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    bwd_rad_trans_yp=dict(
                        weight=2,
                        #### objective is evaluated at this port
                        in_slice_name="out_slice_1",
                        out_slice_name="rad_slice_yp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                    bwd_rad_trans_ym=dict(
                        weight=2,
                        #### objective is evaluated at this port
                        in_slice_name="out_slice_1",
                        out_slice_name="rad_slice_ym",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
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


class OpticalDiodeOptimization(BaseOptimization):
    def __init__(
        self,
        device,
        hr_device,
        design_region_param_cfgs=dict(),
        sim_cfg: dict = dict(),
        obj_cfgs=dict(),
        operation_device=torch.device("cuda:0"),
    ):

        design_region_param_cfgs = dict()
        for region_name in device.design_region_cfgs.keys():
            design_region_param_cfgs[region_name] = dict(
                method="levelset",
                rho_resolution=[25, 25],
                transform=[
                    dict(type="mirror_symmetry", dims=[1]),
                    dict(
                        type="blur",
                        mfs=0.1,
                        resolutions=[hr_device.resolution, hr_device.resolution],
                        dim="xy",
                    ),
                    dict(type="binarize"),
                ],
                init_method="random",
                denorm_mode="linear_eps",
                interpolation="bilinear",
                binary_projection=dict(
                    fw_threshold=100,
                    bw_threshold=100,
                    mode="regular",
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
