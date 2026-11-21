import torch

from .base_optimization import BaseOptimization, DefaultOptimizationConfig


class DefaultConfig(DefaultOptimizationConfig):
    """Default configuration for the Bending inverse-design model.

    The configuration defines the default FDFD solver settings and objective
    terms used by :class:`BendingOptimization`, including forward eigenmode
    transmission, reflection, radiation, and optional S-matrix metrics.
    """

    def __init__(self):
        """Initialize the default simulation, parameterization, and objective configuration."""
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
                    plot_root="./figs/bending",
                ),
                obj_cfgs=dict(
                    fwd_trans=dict(
                        weight=1,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="out_slice_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        wl=[1.55],
                        temp=[300],
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
                        # type="flux",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
                        direction="y+",
                    ),
                    refl_trans=dict(
                        weight=-0.1,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="refl_slice_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        wl=[1.55],
                        temp=[300],
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
                    rad_trans_xm=dict(
                        weight=0,
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
                    rad_trans_yp=dict(
                        weight=0,
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
                    rad_trans_ym=dict(
                        weight=0,
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
                    smatrix=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name=["in_slice_1"],
                        # in_slice_name=["in_slice_1"],
                        # out_slice_name=["out_slice_1"],
                        out_slice_name=["out_slice_1", "refl_slice_1"],
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode="Ez1",  # only one source mode is supported, cannot input multiple modes at the same time
                        wl=[1.55],  #
                        temp=[300],
                        out_modes=(
                            "Ez1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="smatrix",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
                        direction=["y+", "x-"],
                        # direction=["x+"],
                    ),
                ),
            )
        )


class BendingOptimization(BaseOptimization):
    """Inverse-design optimization model for a Bending device.

    Args:
        device: Simulation-resolution device containing geometry, grids, ports,
            monitors, and design-region definitions.
        hr_device: Higher-resolution device used by the design parameterization
            and fabrication-oriented transforms.
        design_region_param_cfgs: Mapping from design-region names to
            parameterization configuration. If empty, a default level-set
            configuration is generated for each device design region.
        sim_cfg: Overrides for the default simulation configuration.
        obj_cfgs: Overrides for the default objective configuration. The
            special ``override`` key clears the default objective mapping before
            applying the supplied entries.
        operation_device: PyTorch device on which optimization tensors are held.

    Notes:
        The generated default parameterization uses a ``[50, 50]`` level-set
        variable grid, transpose symmetry, Gaussian/linear interpolation,
        ``mfs=0.08`` blur in ``xy``, and binary projection. The high-resolution
        design device and the simulation-resolution device may have different
        raster resolutions.
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
        if not design_region_param_cfgs:
            design_region_param_cfgs = dict()
            for region_name in device.design_region_cfgs.keys():
                design_region_param_cfgs[region_name] = dict(
                    method="levelset",
                    rho_resolution=[50, 50],
                    transform=[
                        dict(type="transpose_symmetry", rot_k=3),
                        dict(
                            type="blur",
                            mfs=0.08,
                            resolutions=[hr_device.resolution, hr_device.resolution],
                            dim="xy",
                        ),
                        # dict(type="binarize"),
                        # dict(type="litho", res=310),
                        dict(type="binarize"),
                    ],
                    init_method="random",
                    denorm_mode="linear_eps",
                    interpolation="gaussian_linear",
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
