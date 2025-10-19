import torch

from core.invdes.models.base_optimization import (
    BaseOptimization,
    DefaultOptimizationConfig,
)


class DefaultConfig(DefaultOptimizationConfig):
    def __init__(self):
        super().__init__()
        self.update(
            dict(
                design_region_param_cfgs=dict(
                    design_region_1=dict(
                        method="levelset",
                        rho_resolution=[20, 20],
                        # transform=[dict(type="mirror_symmetry", dims=[1])],
                        transform=[
                            dict(
                                type="blur", mfs=0.1, resolutions=[310, 310], dim="xy"
                            ),
                            dict(type="binarize"),
                        ],  # there is no symmetry in this design region
                        init_method="random",
                        binary_projection=dict(
                            fw_threshold=100,
                            bw_threshold=100,
                            mode="regular",
                        ),
                    )
                ),
                sim_cfg=dict(
                    solver="ceviche_torch",
                    binary_projection=dict(
                        fw_threshold=100,
                        bw_threshold=100,
                        mode="regular",
                    ),
                    border_width=[0, 0, 1.5, 1.5],
                    PML=[0.5, 0.5],
                    cell_size=None,
                    resolution=100,
                    wl_cen=1.55,
                    wl_width=0,
                    n_wl=1,
                    plot_root="./figs/mdm",
                ),
                obj_cfgs=dict(
                    mode1_trans=dict(
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
                        direction="x+",
                    ),
                    mode1_trans_p2=dict(
                        weight=-1,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="out_slice_2",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x+",
                    ),
                    mode2_trans=dict(
                        weight=1,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="out_slice_2",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez2",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez2",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="eigenmode",
                        direction="x+",
                    ),
                    mode2_trans_p1=dict(
                        weight=-1,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="out_slice_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez2",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez2",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x+",
                    ),
                    mode1_refl_trans=dict(
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
                    mode2_refl_trans=dict(
                        weight=-1,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="refl_slice_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez2",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez2",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux_minus_src",
                        direction="x",
                    ),
                    mode1_rad_trans_xp=dict(
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
                    mode1_rad_trans_xm=dict(
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
                    mode1_rad_trans_yp=dict(
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
                    mode1_rad_trans_ym=dict(
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
                    mode2_rad_trans_xp=dict(
                        weight=-2,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_slice_xp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez2",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez2",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    mode2_rad_trans_xm=dict(
                        weight=-2,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_slice_xm",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez2",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez2",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    mode2_rad_trans_yp=dict(
                        weight=-2,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_slice_yp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez2",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez2",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                    mode2_rad_trans_ym=dict(
                        weight=-2,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_slice_ym",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Ez2",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Ez2",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                ),
            )
        )


class MDMOptimization(BaseOptimization):
    def __init__(
        self,
        device,
        hr_device,
        design_region_param_cfgs=dict(),
        sim_cfg: dict = dict(),
        obj_cfgs=dict(),
        operation_device=torch.device("cuda:0"),
    ):
        _design_region_cfgs = design_region_param_cfgs
        design_region_param_cfgs = dict()
        for region_name in device.design_region_cfgs.keys():
            design_region_param_cfgs[region_name] = dict(
                method="levelset",
                rho_resolution=[25, 25],
                transform=[
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
            if region_name in _design_region_cfgs:
                design_region_param_cfgs[region_name].update(
                    _design_region_cfgs[region_name]
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
