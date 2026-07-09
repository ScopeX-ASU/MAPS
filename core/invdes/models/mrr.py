import torch

from core.invdes.models.base_optimization import (
    BaseOptimization,
    DefaultOptimizationConfig,
)


class DefaultConfig(DefaultOptimizationConfig):
    def __init__(self):
        super().__init__()
        off_currents = [dict(heater_1=0.0)]
        on_currents = [dict(heater_1=7.5e-3)]
        self.update(
            dict(
                design_region_param_cfgs=dict(),
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
                    plot_root="./figs/tdm",
                ),
                obj_cfgs=dict(
                    fwd_trans=dict(
                        weight=1,
                        in_slice_name="in_slice_1",
                        out_slice_name="through_slice",
                        wl=[1.55],
                        in_mode="Hz1",
                        out_modes=("Hz1",),
                        type="eigenmode",
                        direction="x+",
                    ),
                ),
            )
        )


class MRROptimization(BaseOptimization):
    def __init__(
        self,
        device,
        hr_device,
        design_region_param_cfgs=dict(),
        sim_cfg: dict = dict(),
        obj_cfgs=dict(),
        operation_device=torch.device("cuda:0"),
    ):
        provided_design_region_param_cfgs = dict(design_region_param_cfgs)
        design_region_param_cfgs = dict()
        for region_name in device.design_region_cfgs.keys():
            design_region_param_cfgs[region_name] = dict(
                method="levelset",
                rho_resolution=[100, 100],
                transform=[
                    dict(
                        type="blur",
                        mfs=0.08,
                        resolutions=[hr_device.resolution, hr_device.resolution],
                        dim="xy",
                    ),
                    dict(type="binarize"),
                ],  # there is no symmetry in this design region
                init_method="ones",
                # init_method="random",
                denorm_mode="linear_eps",
                interpolation="gaussian_linear",
                binary_projection=dict(
                    fw_threshold=100,
                    bw_threshold=100,
                    # mode="ste",
                    mode="regular",
                ),
            )
            if region_name in provided_design_region_param_cfgs:
                design_region_param_cfgs[region_name].update(
                    provided_design_region_param_cfgs[region_name]
                )
        cfgs = DefaultConfig()  ## this is default configurations
        override_obj = obj_cfgs.pop(
            "override", False
        )  # remove this key to avoid confusion later
        if override_obj:
            print("Override default obj_cfgs with the provided obj_cfgs", flush=True)
            cfgs.obj_cfgs = {}
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
