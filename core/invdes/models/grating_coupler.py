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
                    border_width=[0, 1.5, 0.6, 0.6, 0.6, 0.6],
                    PML=[0.2, 0.2, 0.2],
                    cell_size=None,
                    resolution=50,
                    wl_cen=1.55,
                    wl_width=0,
                    n_wl=1,
                    plot_root="./figs/grating_coupler",
                ),
                obj_cfgs=dict(
                    fwd_trans=dict(
                        weight=1,
                        #### objective is evaluated at this port
                        in_slice_name="out_slice_1",
                        out_slice_name="in_slice_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        wl=[1.55],
                        temp=[300],
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
                        direction="x-",
                    ),
                ),
            )
        )


class GratingCouplerOptimization(BaseOptimization):
    def __init__(
        self,
        device,
        hr_device,
        design_region_param_cfgs=dict(),
        sim_cfg: dict = dict(),
        obj_cfgs=dict(),
        operation_device=torch.device("cuda:0"),
    ):
        if not design_region_param_cfgs:
            design_region_param_cfgs = dict()
            for region_name in device.design_region_cfgs.keys():
                design_region_param_cfgs[region_name] = dict(
                    method="levelset",
                    rho_resolution=[25, 25],
                    transform=[
                        dict(type="mirror_symmetry", dims=[1]),
                        dict(
                            type="blur",
                            mfs=0.05,
                            resolutions=[hr_device.resolution, hr_device.resolution],
                            dim="xy",
                        ),
                        dict(type="binarize"),
                    ],
                    init_method="ones",
                    # init_method="random",
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
        override_obj = obj_cfgs.pop(
            "override", False
        )  # remove this key to avoid confusion later
        if override_obj:
            print("Override default obj_cfgs with the provided obj_cfgs", flush=True)
            cfgs.obj_cfgs = {}
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
