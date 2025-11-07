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
                        fw_threshold=180,
                        bw_threshold=180,
                        mode="regular",
                    ),
                    border_width=[0, 0, 4, 4],
                    PML=[0.8, 0.8],
                    cell_size=None,
                    resolution=50,
                    wl_cen=1.55,
                    wl_width=0,
                    n_wl=1,
                    plot_root="./figs/mmi",
                ),
                obj_cfgs=dict(
                    smatrix=dict(
                        weight=1,
                        #### objective is evaluated at this port
                        in_slice_name=["in_slice_1", "in_slice_2", "in_slice_3"],
                        out_slice_name=["out_slice_1", "out_slice_2", "out_slice_3"],
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        wl=[1.55],
                        temp=[300],
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="smatrix",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
                        direction=["x+", "x+", "x+"],
                    ),
                    # fwd_trans=dict(
                    #     weight=1,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="out_slice_1",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     wl=[1.55],
                    #     temp=[300],
                    #     out_modes=(
                    #         "Ez1",
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
                    #     direction="x+",
                    # ),
                    # refl_trans=dict(
                    #     weight=-1,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="refl_slice_1",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     wl=[1.55],
                    #     temp=[300],
                    #     out_modes=(
                    #         "Ez1",
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     type="flux_minus_src",
                    #     direction="x",
                    # ),
                    # rad_trans_xp=dict(
                    #     weight=-0.1,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="rad_monitor_xp",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     wl=[1.55],
                    #     temp=[300],
                    #     in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         "Ez1",
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     type="flux",
                    #     direction="x",
                    # ),
                    # rad_trans_xm=dict(
                    #     weight=-0.1,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="rad_monitor_xm",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     wl=[1.55],
                    #     temp=[300],
                    #     in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         "Ez1",
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     type="flux",
                    #     direction="x",
                    # ),
                    # rad_trans_yp=dict(
                    #     weight=-0.1,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="rad_monitor_yp",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     wl=[1.55],
                    #     temp=[300],
                    #     in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         "Ez1",
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     type="flux",
                    #     direction="y",
                    # ),
                    # rad_trans_ym=dict(
                    #     weight=-0.1,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="rad_monitor_ym",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     wl=[1.55],
                    #     temp=[300],
                    #     in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         "Ez1",
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     type="flux",
                    #     direction="y",
                    # ),
                ),
            )
        )


class MMIOptimization(BaseOptimization):
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
                    # dict(type="mirror_symmetry", dims=[1]),
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
        # for i in range(1, device.num_outports + 1):
        #     cfgs.obj_cfgs[f"fwd_trans_{i}"] = dict(
        #         weight=1,
        #         #### objective is evaluated at this port
        #         in_slice_name="in_slice_1",
        #         out_slice_name=f"out_slice_{i}",
        #         #### objective is evaluated at all points by sweeping the wavelength and modes
        #         in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
        #         wl=[1.55],
        #         temp=[300],
        #         out_modes=(
        #             "Ez1",
        #         ),  # can evaluate on multiple output modes and get average transmission
        #         type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
        #         direction="x+",
        #     )
        #     cfgs.obj_cfgs[f"phase_{i}"] = dict(
        #         weight=0,
        #         #### objective is evaluated at this port
        #         in_slice_name="in_slice_1",
        #         out_slice_name=f"out_slice_{i}",
        #         #### objective is evaluated at all points by sweeping the wavelength and modes
        #         in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
        #         wl=[1.55],
        #         temp=[300],
        #         out_modes=(
        #             "Ez1",
        #         ),  # can evaluate on multiple output modes and get average transmission
        #         type="phase",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
        #         direction="x+",
        #     )
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
