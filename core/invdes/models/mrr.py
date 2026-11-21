import torch

from core.invdes.models.base_optimization import (
    BaseOptimization,
    DefaultOptimizationConfig,
)


class DefaultConfig(DefaultOptimizationConfig):
    """Default Ceviche/FDFD and through-port objective configuration for MRR."""

    def __init__(self):
        """Initialize the MRR default configuration tree."""
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
    """Inverse-design optimization model for a microring resonator.

    Args:
        device: Simulation-resolution MRR device and design-region definitions.
        hr_device: Higher-resolution device used by the parameterization.
        design_region_param_cfgs: Optional per-region overrides. The built-in
            level-set parameterization is first constructed for each region,
            then matching supplied regions are updated with user values.
        sim_cfg: Overrides for the default ``ceviche_torch`` simulation settings.
        obj_cfgs: Objective overrides. ``override=True`` clears the default
            objective mapping before applying supplied terms.
        operation_device: PyTorch device for optimization tensors.

    Notes:
        The default design uses ``rho_resolution=[100,100]``, blur with
        ``mfs=0.08`` in ``xy``, binarization, ``init_method="ones"``,
        Gaussian-linear interpolation, and binary projection thresholds of 100.
        The active objective is forward eigenmode transmission from
        ``in_slice_1`` to ``through_slice`` using mode ``Hz1`` at 1.55 um.
        ``off_currents`` and ``on_currents`` are defined in the default config
        for heater-related workflows but are not consumed by this constructor.
    """

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
