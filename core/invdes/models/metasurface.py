"""Optimization model for 3-D diffractive metasurface image formation."""

import math

import torch

from .base_optimization import BaseOptimization, DefaultOptimizationConfig

__all__ = ["DiffractiveMetasurfaceOptimization"]


class DefaultConfig(DefaultOptimizationConfig):
    """Default configuration for diffractive metasurface optimization."""

    def __init__(self):
        """Initialize the default metasurface configuration tree."""
        super().__init__()
        self.update(
            dict(
                design_region_param_cfgs=dict(),
                sim_cfg=dict(
                    solver="fdtdx",
                    max_time=5e-13,
                    border_width=[0, 0, 0, 0, 0, 0.8 * 1.55 + 0.3],
                    PML=[0.0, 0.0, 0.8 * 1.55],
                    cell_size=None,
                    resolution=50,
                    wl_cen=1.55,
                    wl_width=0,
                    n_wl=1,
                    plot_root="./figs/metasurface_3d",
                ),
                obj_cfgs=dict(
                    image_match=dict(
                        weight=1.0,
                        type="target_intensity_angular_spectrum",
                        in_slice_name="in_slice_1",
                        out_slice_name="nearfield_projection",
                        in_mode="Hz1",
                        direction="z+",
                        wl=[1.55],
                        temp=[300],
                        target_intensity=None,
                        field_components=("Ex", "Ey", "Ez"),
                        projection_distance_um=6.1 - 0.3,
                        evanescent="decay",
                    ),
                ),
            )
        )


class DiffractiveMetasurfaceOptimization(BaseOptimization):
    """Optimize a 2-D dielectric mask through periodic field projection.

    Args:
        device: Simulation-resolution :class:`DiffractiveMetasurface` device.
        hr_device: Device used to define fabrication transforms and export maps.
        target_intensity: Nonnegative 2-D target image in x/y array order.
        design_region_param_cfgs: Optional MAPS design parameterization settings.
        sim_cfg: FDTD-X and wavelength configuration.
        obj_cfgs: Optional objective overrides.
        operation_device: Device for optimization tensors.
    """

    def __init__(
        self,
        device,
        hr_device,
        design_region_param_cfgs=None,
        sim_cfg=None,
        obj_cfgs=None,
        operation_device=torch.device("cuda:0"),
    ):

        if not design_region_param_cfgs:
            grid_cfg = device.sim_cfg.get("optical_grid", {}).get("grid_x", {})
            steps_per_wavelength = float(
                grid_cfg.get(
                    "min_steps_per_wvl",
                    device.resolution * float(device.sim_cfg["wl_cen"]),
                )
            )
            density_resolution = (
                steps_per_wavelength
                * math.sqrt(float(device.eps_design))
                / (2 * float(device.sim_cfg["wl_cen"]))
            )
            design_region_param_cfgs = dict()
            for region_name in device.design_region_cfgs.keys():
                design_region_param_cfgs[region_name] = {
                    "method": "levelset",
                    "rho_resolution": [density_resolution, density_resolution],
                    "dims": (0, 1),
                    "reference_plane": "middle",
                    "transform": [
                        {"type": "binarize"},
                    ],
                    # "init_method": "random_0.5",
                    "init_method": "constant_0",
                    "denorm_mode": "linear_eps",
                    "interpolation": "gaussian_linear",
                    "binary_projection": {
                        "fw_threshold": 100,
                        "bw_threshold": 100,
                        "mode": "regular",
                    },
                }
        cfgs = DefaultConfig()  ## this is default configurations
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
