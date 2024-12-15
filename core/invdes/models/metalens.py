import torch

from .base_optimization import BaseOptimization, DefaultOptimizationConfig
from pyutils.config import Config


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
                    border_width=[0, 0, 1.5, 1.5],
                    PML=[0.8, 0.8],
                    cell_size=None,
                    resolution=50,
                    wl_cen=0.832,
                    wl_width=0,
                    n_wl=1,
                    plot_root="./figs/metalens",
                ),
                obj_cfgs=dict(
                    fwd_trans=dict(
                        weight=1,
                        #### objective is evaluated at this port
                        in_port_name="in_port_1",
                        out_port_name="farfield_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        temp=[300],
                        wl=[0.832],
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(1,),
                        type="flux_near2far",
                        direction="x+",
                    ),
                    # fwd_trans_2=dict(
                    #     weight=1,
                    #     #### objective is evaluated at this port
                    #     in_port_name="in_port_1",
                    #     out_port_name="farfield_2",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.832],
                    #     in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(1,),
                    #     type="flux_near2far",
                    #     direction="x+",
                    # ),
                    # fwd_trans_3=dict(
                    #     weight=1,
                    #     #### objective is evaluated at this port
                    #     in_port_name="in_port_1",
                    #     out_port_name="farfield_3",
                    #     temp=[300],
                    #     wl=[0.832],
                    #     in_mode=1,
                    #     out_modes=(1,),
                    #     type="flux_near2far",
                    #     direction="x+",
                    # ),
                    # fwd_trans_4=dict(
                    #     weight=1,
                    #     in_port_name="in_port_1",
                    #     out_port_name="farfield_4",
                    #     temp=[300],
                    #     wl=[0.832],
                    #     in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(1,),
                    #     type="flux_near2far",
                    #     direction="x+",
                    # ),
                    # fwd_trans_5=dict(
                    #     weight=1,
                    #     #### objective is evaluated at this port
                    #     in_port_name="in_port_1",
                    #     out_port_name="farfield_5",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.832],
                    #     in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(1,),
                    #     type="flux_near2far",
                    #     direction="x+",
                    # ),
                    fwd_refl_trans=dict(
                        weight=-0.1,
                        #### objective is evaluated at this port
                        in_port_name="in_port_1",
                        out_port_name="refl_port_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        temp=[300],
                        wl=[0.832],
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux_minus_src",
                        direction="x",
                    ),
                    rad_trans_yp=dict(
                        weight=-0.2,
                        #### objective is evaluated at this port
                        in_port_name="in_port_1",
                        out_port_name="rad_monitor_yp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        temp=[300],
                        wl=[0.832],
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        # type="flux_near2far",
                        type="flux",
                        direction="y",
                    ),
                    rad_trans_ym=dict(
                        weight=-0.2,
                        #### objective is evaluated at this port
                        in_port_name="in_port_1",
                        out_port_name="rad_monitor_ym",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        temp=[300],
                        wl=[0.832],
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        # type="flux_near2far",
                        type="flux",
                        direction="y",
                    ),
                    # rad_trans_xp=dict(
                    #     weight=0,
                    #     #### objective is evaluated at this port
                    #     in_port_name="in_port_1",
                    #     out_port_name="rad_monitor_xp",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.832],
                    #     in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         1,
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     # type="flux_near2far",
                    #     type="flux",
                    #     direction="x",
                    # ),
                    # rad_trans_xp_minus=dict(
                    #     weight=-0.2,
                    #    #### objective is evaluated at this port
                    #     in_port_name="in_port_1",
                    #     out_port_name="rad_monitor_xp_minus",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp = [300],
                    #     wl=[0.832],
                    #     in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         1,
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     # type="flux_near2far",
                    #     type="flux",
                    #     direction="x",
                    # ),
                    fwd_intensity_shape=dict(
                        weight=0.2,
                        #### objective is evaluated at this port
                        in_port_name="in_port_1",
                        out_port_name="farfield_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode=1,  # only one source mode is supported, cannot input multiple modes at the same time
                        wl=[0.832],
                        temp=[300],
                        out_modes=(
                            1,
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="intensity_shape_near2far",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
                        shape_type="gaussian",
                        shape_cfg=dict(
                            width=0.85, # sigma, 2.355 * sigma = FWHM = 2 um for gaussian, sigma = 0.85 um
                        ),
                        direction="x+",
                    ),
                ),
            )
        )


class MetaLensOptimization(BaseOptimization):
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
                rho_resolution=[0, 10],
                # transform=[dict(type="mirror_symmetry", dims=[1]), dict(type="blur", mfs=0.1, resolutions=[310, 310])],
                transform=[dict(type="mirror_symmetry", dims=[1])],
                init_method="grating_1d",
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
