import warnings
from functools import partial
from typing import Tuple

import torch
from pyutils.general import logger

from core.utils import material_fn_dict

from .device_base import N_Ports

__all__ = ["Crossing"]


class Crossing(N_Ports):
    """Define a four-port waveguide crossing device with optional 3-D geometry.

    Args:
        material_r1 (str | Any): Primary waveguide material key or direct value. Defaults to ``"Si_eff"``.
        material_r2 (str | Any): Design-region background material key or direct value. Defaults to ``"SiO2"``.
        thickness_r1 (float): Effective primary-material thickness for ``*_eff`` models. Defaults to 0.22.
        thickness_r2 (float): Effective secondary-material thickness. Defaults to 0.
        material_bg (str | Any): Simulation background material key or direct value. Defaults to ``"SiO2"``.
        sim_cfg (dict): Optical simulation configuration. Defaults to the built-in configuration.
        box_size (tuple[float, float] | Sequence[float]): Crossing design-region size in micrometers;
            in 3-D the implementation also reads ``box_size[2]`` for monitor-height calculations.
            Defaults to ``(1.5, 1.5)``.
        port_len (tuple[float, float]): x- and y-directed port lengths in micrometers. Defaults to ``(1.8, 1.8)``.
        port_width (tuple[float, float]): x- and y-directed port widths in micrometers. Defaults to ``(0.48, 0.48)``.
        port_height (tuple[float, float]): z extents for x- and y-directed ports in micrometers. Defaults to ``(0, 0)``.
        port_sidewall_angle (float): Port sidewall angle in degrees. Defaults to 0.0.
        device (torch.device): Device used by the underlying ``N_Ports`` state. Defaults to ``torch.device("cuda:0")``.
        is_3d (bool): Whether to include the z dimension in ports/design-region calculations. Defaults to False.
        mode (str): Mode used by normalization-source construction. Defaults to ``"Ez1"``.
        verbose (bool): Whether to emit initialization/monitor diagnostics. Defaults to True.
    """

    def __init__(
        self,
        material_r1: str = "Si_eff",  # waveguide material
        material_r2: str = "SiO2",  # waveguide material
        thickness_r1: float = 0.22,  # waveguide thickness
        thickness_r2: float = 0,  # cladding/design background thickness
        material_bg: str = "SiO2",  # background material
        sim_cfg: dict = {
            "border_width": [
                0,
                1.8,
                1.8,
                0,
            ],  # left, right, lower, upper, containing PML
            "PML": [0.5, 0.5],  # left/right, lower/upper
            "cell_size": None,
            "resolution": 50,
            "wl_cen": 1.55,
            "wl_width": 0,
            "n_wl": 1,
        },
        box_size: Tuple[float] = (1.5, 1.5),
        port_len: Tuple[float] = (1.8, 1.8),
        port_width: Tuple[float] = (0.48, 0.48),
        port_height: Tuple[float] = (0, 0),
        port_sidewall_angle: float = 0.0,
        device: torch.device = torch.device("cuda:0"),
        is_3d: bool = False,
        mode: str = "Ez1",
        verbose: bool = True,
    ):
        """Initialize materials, four orthogonal ports, and the crossing design region.

        Args:
            material_r1 (str | Any): Primary material key or direct value. Defaults to ``"Si_eff"``.
            material_r2 (str | Any): Design-region background key or direct value. Defaults to ``"SiO2"``.
            thickness_r1 (float): Primary effective thickness. Defaults to 0.22.
            thickness_r2 (float): Secondary effective thickness. Defaults to 0.
            material_bg (str | Any): Simulation background key or direct value. Defaults to ``"SiO2"``.
            sim_cfg (dict): Optical simulation configuration. Defaults to the built-in configuration.
            box_size (tuple[float, float] | Sequence[float]): Crossing-region size in micrometers.
                Defaults to ``(1.5, 1.5)``.
            port_len (tuple[float, float]): x/y port lengths in micrometers. Defaults to ``(1.8, 1.8)``.
            port_width (tuple[float, float]): x/y port widths in micrometers. Defaults to ``(0.48, 0.48)``.
            port_height (tuple[float, float]): x/y port z extents in micrometers. Defaults to ``(0, 0)``.
            port_sidewall_angle (float): Port sidewall angle in degrees. Defaults to 0.0.
            device (torch.device): Underlying device state device. Defaults to ``torch.device("cuda:0")``.
            is_3d (bool): Whether to construct 3-D geometry. Defaults to False.
            mode (str): Normalization/source mode. Defaults to ``"Ez1"``.
            verbose (bool): Whether to log initialization information. Defaults to True.
        """
        # ----------------------------------
        # |                                |
        # |                                |
        # |                                |
        # |[1]                             |
        # |                                |
        # |                                |
        # |              [0]               |
        # ----------------------------------
        self.is_3d = is_3d
        self.mode = mode
        self.box_size = box_size
        if box_size[0] != box_size[1]:
            warnings.warn(
                "Crossing region width and length are not equal, this is not a square crossing region."
            )
        wl_cen = sim_cfg["wl_cen"]
        if isinstance(material_r1, str):
            eps_r1_fn = material_fn_dict[material_r1]
            if "_eff" in material_r1:
                eps_r1_fn = partial(eps_r1_fn, thickness=thickness_r1)
        else:
            eps_r1_fn = lambda wl: material_r1

        if isinstance(material_r2, str):
            eps_r2_fn = material_fn_dict[material_r2]
            if "_eff" in material_r2:
                eps_r2_fn = partial(eps_r2_fn, thickness=thickness_r2)
        else:
            eps_r2_fn = lambda wl: material_r2

        if isinstance(material_bg, str):
            eps_bg_fn = material_fn_dict[material_bg]
            if "_eff" in material_bg:
                eps_bg_fn = partial(eps_bg_fn, thickness=thickness_r2)
        else:
            eps_bg_fn = lambda wl: material_bg

        box_size = list(box_size)
        grid_step = 1 / sim_cfg["resolution"]

        port_cfgs = dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-(port_len[0] + box_size[0]) / 2 + grid_step / 2, 0]
                + ([0] if is_3d else []),
                size=(
                    [port_len[0] + grid_step, port_width[0]]
                    + ([port_height[0]] if is_3d else [])
                ),
                eps=eps_r1_fn(wl_cen),
                port_sidewall_angle=port_sidewall_angle,
            ),
            out_port_1=dict(
                type="box",
                direction="x",
                center=[(port_len[0] + box_size[0]) / 2 - grid_step / 2, 0]
                + ([0] if is_3d else []),
                size=(
                    [port_len[0] + grid_step, port_width[0]]
                    + ([port_height[0]] if is_3d else [])
                ),
                eps=eps_r1_fn(wl_cen),
                port_sidewall_angle=port_sidewall_angle,
            ),
            top_port=dict(
                type="box",
                direction="y",
                center=[0.0, (port_len[1] + box_size[1]) / 2 - grid_step / 2]
                + ([0] if is_3d else []),
                size=(
                    [port_width[1], port_len[1] + grid_step]
                    + ([port_height[1]] if is_3d else [])
                ),
                eps=eps_r1_fn(wl_cen),
                port_sidewall_angle=port_sidewall_angle,
            ),
            bot_port=dict(
                type="box",
                direction="y",
                center=[0.0, -(port_len[1] + box_size[1]) / 2 + grid_step / 2]
                + ([0] if is_3d else []),
                size=(
                    [port_width[1], port_len[1] + grid_step]
                    + ([port_height[1]] if is_3d else [])
                ),
                eps=eps_r1_fn(wl_cen),
                port_sidewall_angle=port_sidewall_angle,
            ),
        )

        geometry_cfgs = dict()

        design_region_cfgs = dict()
        design_region_cfgs["crossing_region"] = dict(
            type="box",
            center=[0, 0] + ([0] if is_3d else []),
            size=box_size,
            eps=eps_r1_fn(wl_cen),
            eps_bg=eps_r2_fn(wl_cen),
        )

        super().__init__(
            eps_bg=eps_bg_fn(wl_cen),
            sim_cfg=sim_cfg,
            port_cfgs=port_cfgs,
            geometry_cfgs=geometry_cfgs,
            design_region_cfgs=design_region_cfgs,
            device=device,
            verbose=verbose,
        )

    def init_monitors(self, verbose: bool = True):
        """Build source/reflection/output/cross-port monitor slices and a 2-D radiation monitor.

        Args:
            verbose (bool): Whether to log monitor/source construction. Defaults to True.

        Returns:
            tuple: In 2-D, ``(src_slice, out_slice, refl_slice, top_slice, bot_slice, radiation_monitor)``.
            In 3-D, ``(src_slice, out_slice, refl_slice, top_slice, bot_slice)`` without a radiation monitor.
        """
        rel_width = 4
        pml_x, pml_y = self.sim_cfg["PML"][:2]
        offset_x, offset_y = 0.2 + pml_x, 0.2 + pml_y
        port_len_x = self.port_cfgs["in_port_1"]["size"][0]
        port_len_y = self.port_cfgs["top_port"]["size"][1]
        rel_height = None
        if self.is_3d:
            rel_height = (
                (
                    self.box_size[2]
                    + self.sim_cfg["border_width"][4]
                    + self.sim_cfg["border_width"][5]
                    - self.sim_cfg["PML"][-1] * 2
                )
                / self.box_size[2]
                if self.is_3d
                else None
            )
        if verbose:
            logger.info("Start generating sources and monitors ...")
        src_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_slice_1",
            rel_loc=offset_x / port_len_x,
            rel_width=rel_width,
            rel_height=rel_height,
        )
        refl_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="refl_slice_1",
            rel_loc=(offset_x + 0.05) / port_len_x,
            rel_width=rel_width,
            rel_height=rel_height,
        )
        out_slice = self.build_port_monitor_slice(
            port_name="out_port_1",
            slice_name="out_slice_1",
            rel_loc=1 - offset_x / port_len_x,
            rel_width=rel_width,
            rel_height=rel_height,
        )
        top_slice = self.build_port_monitor_slice(
            port_name="top_port",
            slice_name="top_slice",
            rel_loc=1 - offset_y / port_len_y,
            rel_width=rel_width,
            rel_height=rel_height,
        )
        bot_slice = self.build_port_monitor_slice(
            port_name="bot_port",
            slice_name="bot_slice",
            rel_loc=offset_y / port_len_y,
            rel_width=rel_width,
            rel_height=rel_height,
        )
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        if not self.is_3d:
            radiation_monitor = self.build_radiation_monitor(monitor_name="rad_slice")
            return (
                src_slice,
                out_slice,
                refl_slice,
                top_slice,
                bot_slice,
                radiation_monitor,
            )
        return src_slice, out_slice, refl_slice, top_slice, bot_slice

    def norm_run(self, verbose: bool = True):
        """Build normalization profiles for all four crossing ports.

        Args:
            verbose (bool): Whether to log normalization-run construction. Defaults to True.

        Returns:
            tuple[dict, dict, dict, dict, dict]: Normalization profile mappings for the
            input, reflection, output, top, and bottom port/slice paths, respectively.
            All profiles use ``self.mode`` and the configured solver; only the input profile
            sets ``require_sim=True``.
        """
        if verbose:
            logger.info("Start normalization run ...")
        # norm_run_sim_cfg = copy.deepcopy(self.sim_cfg)
        # norm_run_sim_cfg["numerical_solver"] = "solve_direct"
        norm_source_profiles = self.build_norm_sources(
            source_modes=(self.mode,),
            input_port_name="in_port_1",
            input_slice_name="in_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=True,
        )

        norm_refl_profiles = self.build_norm_sources(
            source_modes=(self.mode,),
            input_port_name="in_port_1",
            input_slice_name="refl_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )
        norm_monitor_profiles = self.build_norm_sources(
            source_modes=(self.mode,),
            input_port_name="out_port_1",
            input_slice_name="out_slice_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )
        norm_top_profiles = self.build_norm_sources(
            source_modes=(self.mode,),
            input_port_name="top_port",
            input_slice_name="top_slice",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )
        norm_bot_profiles = self.build_norm_sources(
            source_modes=(self.mode,),
            input_port_name="bot_port",
            input_slice_name="bot_slice",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            plot=True,
            require_sim=False,
        )
        return (
            norm_source_profiles,
            norm_refl_profiles,
            norm_monitor_profiles,
            norm_top_profiles,
            norm_bot_profiles,
        )
