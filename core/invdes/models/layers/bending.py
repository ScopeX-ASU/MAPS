import time
import warnings
from functools import partial
from typing import Tuple

import torch
from pyutils.general import logger

from core.utils import material_fn_dict

from .device_base import N_Ports

__all__ = ["Bending"]


class Bending(N_Ports):
    """Define a 90-degree waveguide-bending inverse-design device with 2-D/3-D support.

    Args:
        material_r1 (str | Any): Primary waveguide material key or direct value. Defaults to ``"Si_eff"``.
        material_r2 (str | Any): Design-region/background material key or direct value. Defaults to ``"SiO2"``.
        thickness_r1 (float): Effective primary-material thickness for ``*_eff`` models. Defaults to 0.22.
        thickness_r2 (float): Effective secondary/background thickness for ``*_eff`` models. Defaults to 0.0.
        material_bg (str | Any): Simulation background material key or direct value. Defaults to ``"SiO2"``.
        sim_cfg (dict): Optical simulation configuration. Defaults to the built-in configuration.
        bending_region_size (tuple[float, float]): Design-region ``(x, y)`` size in micrometers. Defaults to ``(1.5, 1.5)``.
        port_len (tuple[float, float]): Input and output port lengths in micrometers. Defaults to ``(1.8, 1.8)``.
        port_width (tuple[float, float]): Input and output transverse widths in micrometers. Defaults to ``(0.48, 0.48)``.
        port_height (tuple[float, float]): Input and output z extents for 3-D ports in micrometers. Defaults to ``(0, 0)``.
        LER_A (float): Line-edge-roughness amplitude passed to the port configurations. Defaults to 0.0.
        LER_Lc (float): Line-edge-roughness correlation length. Defaults to 0.0.
        sidewall_angle (float): Port sidewall angle in degrees. Defaults to 0.0.
        device (torch.device): Device used by the underlying ``N_Ports`` state. Defaults to ``torch.device("cuda:0")``.
        port_offset (float): Lateral offset of the input/output ports from the device center. Defaults to 0.0.
        is_3d (bool): Whether to include the z dimension in port and design-region geometry. Defaults to False.
        mode (str): Source/normalization mode identifier. Defaults to ``"Ez1"``.
        verbose (bool): Whether to emit initialization diagnostics. Defaults to True.
    """

    def __init__(
        self,
        material_r1: str = "Si_eff",  # waveguide material
        material_r2: str = "SiO2",  # waveguide material
        thickness_r1: float = 0.22,  # waveguide thickness
        thickness_r2: float = 0.0,  # waveguide thickness
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
        bending_region_size: Tuple[float] = (1.5, 1.5),
        port_len: Tuple[float] = (1.8, 1.8),
        port_width: Tuple[float] = (0.48, 0.48),
        port_height: Tuple[float] = (0, 0),
        LER_A: float = 0.0,  # line edge roughness amplitude
        LER_Lc: float = 0.0,  # line edge roughness correlation
        sidewall_angle: float = 0.0,  # sidewall angle in degrees
        device: torch.device = torch.device("cuda:0"),
        port_offset: float = 0.0,  # offset of the port from the center of the device
        is_3d: bool = False,
        mode: str = "Ez1",
        verbose: bool = True,  # whether to print the device information
    ):
        """Initialize materials, orthogonal ports, and the bending design region.

        Args:
            material_r1 (str | Any): Primary material key or direct value. Defaults to ``"Si_eff"``.
            material_r2 (str | Any): Design-region background key or direct value. Defaults to ``"SiO2"``.
            thickness_r1 (float): Primary effective thickness. Defaults to 0.22.
            thickness_r2 (float): Secondary effective thickness. Defaults to 0.0.
            material_bg (str | Any): Simulation background key or direct value. Defaults to ``"SiO2"``.
            sim_cfg (dict): Optical simulation configuration. Defaults to the built-in configuration.
            bending_region_size (tuple[float, float]): Design-region size in micrometers. Defaults to ``(1.5, 1.5)``.
            port_len (tuple[float, float]): Input/output port lengths in micrometers. Defaults to ``(1.8, 1.8)``.
            port_width (tuple[float, float]): Input/output widths in micrometers. Defaults to ``(0.48, 0.48)``.
            port_height (tuple[float, float]): Input/output z extents in micrometers. Defaults to ``(0, 0)``.
            LER_A (float): Line-edge-roughness amplitude. Defaults to 0.0.
            LER_Lc (float): Line-edge-roughness correlation length. Defaults to 0.0.
            sidewall_angle (float): Port sidewall angle in degrees. Defaults to 0.0.
            device (torch.device): Underlying device state device. Defaults to ``torch.device("cuda:0")``.
            port_offset (float): Port lateral offset from the design center. Defaults to 0.0.
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
        if bending_region_size[0] != bending_region_size[1]:
            warnings.warn(
                "Bending region width and length are not equal, this is not a square bending region."
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

        box_size = list(bending_region_size)
        grid_step = 1 / sim_cfg["resolution"]
        port_cfgs = dict(
            in_port_1=dict(
                type="box",
                direction="x",
                center=[-(port_len[0] + box_size[0]) / 2 + grid_step / 2, -port_offset]
                + ([0] if is_3d else []),
                size=(
                    [port_len[0] + grid_step, port_width[0]]
                    + ([port_height[0]] if is_3d else [])
                ),
                eps=eps_r1_fn(wl_cen),
                roughness=dict(amplitude=LER_A, correlation_length=LER_Lc),
                sidewall_angle=sidewall_angle,
            ),
            out_port_1=dict(
                type="box",
                direction="y",
                center=[port_offset, (port_len[1] + box_size[1]) / 2 - grid_step / 2]
                + ([0] if is_3d else []),
                size=(
                    [port_width[1], port_len[1] + grid_step]
                    + ([port_height[1]] if is_3d else [])
                ),
                eps=eps_r1_fn(wl_cen),
                roughness=dict(amplitude=LER_A, correlation_length=LER_Lc),
                sidewall_angle=sidewall_angle,
            ),
        )

        geometry_cfgs = dict()

        design_region_cfgs = dict()
        design_region_cfgs["bending_region"] = dict(
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
        """Build input, reflection, and output port monitors for the bending device.

        Args:
            verbose (bool): Whether to log monitor/source construction. Defaults to True.

        Returns:
            tuple: In 2-D, returns ``(src_slice, out_slice, refl_slice, radiation_monitor)``.
            In 3-D, returns ``(src_slice, out_slice, refl_slice)`` without a radiation monitor.
        """
        rel_width = 6
        pml = self.sim_cfg["PML"][0]
        offset = 0.2 + pml
        port_len = self.port_cfgs["in_port_1"]["size"][0]

        rel_height = None
        if self.is_3d:
            rel_height = (
                self.port_cfgs["in_port_1"]["size"][2]
                + self.sim_cfg["border_width"][4]
                + self.sim_cfg["border_width"][5]
                - self.sim_cfg["PML"][-2]
                - self.sim_cfg["PML"][-1]
                - 0.1
            ) / self.port_cfgs["in_port_1"]["size"][2]
        if verbose:
            logger.info("Start generating sources and monitors ...")
        src_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="in_slice_1",
            rel_loc=offset / port_len,
            rel_width=rel_width,
            rel_height=rel_height,
        )
        refl_slice = self.build_port_monitor_slice(
            port_name="in_port_1",
            slice_name="refl_slice_1",
            rel_loc=(offset + 0.2) / port_len,
            rel_width=rel_width,
            rel_height=rel_height,
        )
        out_slice = self.build_port_monitor_slice(
            port_name="out_port_1",
            slice_name="out_slice_1",
            rel_loc=1 - offset / port_len,
            rel_width=rel_width,
            rel_height=rel_height,
        )
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)
        if not self.is_3d:
            radiation_monitor = self.build_radiation_monitor(monitor_name="rad_slice")
            return src_slice, out_slice, refl_slice, radiation_monitor
        return src_slice, out_slice, refl_slice

    def norm_run(self, verbose: bool = True):
        """Build normalization source profiles for forward, reflection, and output paths.

        Args:
            verbose (bool): Whether to log the normalization runtime. Defaults to True.

        Returns:
            tuple[dict, dict, dict]: Forward/source, reflection, and output normalization
            profile mappings. All three use ``self.mode``; the first requires simulation
            while the latter two set ``require_sim=False`` and reuse the configured solver.
        """
        start_time = time.time()
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
        if verbose:
            logger.info(
                "Done normalization run in {:.4f} seconds.".format(
                    time.time() - start_time
                )
            )
        return norm_source_profiles, norm_refl_profiles, norm_monitor_profiles
