"""Three-dimensional diffractive metasurface device geometry."""

from __future__ import annotations

from core.invdes.models.layers.device_base import N_Ports

__all__ = ["DiffractiveMetasurface"]


class DiffractiveMetasurface(N_Ports):
    """Finite 3-D dielectric phase mask with normal-incidence z ports.

    Args:
        sim_cfg: MAPS simulation configuration. When ``cell_size`` is unset,
            the base device infers the asymmetric physical bounds and recenters
            the complete geometry once.
        aperture: Square design-region side length in micrometers.
        thickness: Design layer thickness in micrometers.
        substrate_thickness: Finite substrate thickness below the layer.
        source_distance: Source-plane distance below the layer in micrometers.
        image_distance: Output-plane distance above the layer in micrometers.
        nearfield_distance: Projection-monitor distance above the layer in
            micrometers.
        eps_design: High permittivity endpoint for the design layer.
        eps_substrate: Substrate relative permittivity.
        eps_background: Ambient relative permittivity.
        in_mode: Input field polarization/mode for the metasurface.
        device: Device on which simulation arrays are allocated.

    Notes:
        The finite substrate is a MAPS/FDTD-X approximation to the reference
        notebook's semi-infinite substrate. A native FDTD-X TFSF plane source
        illuminates one periodic supercell at normal incidence.
    """

    def __init__(
        self,
        sim_cfg: dict,
        aperture: float = 20.0,
        thickness: float = 1.54,
        substrate_thickness: float = 3.0,
        source_distance: float = 1.5,
        image_distance: float = 6.1,
        nearfield_distance: float = 0.3,
        eps_design: float = 2.0,
        eps_substrate: float = 2.0,
        eps_background: float = 1.0,
        in_mode: str = "Hz2",
        device="cuda:0",
    ):
        if (
            min(
                aperture,
                thickness,
                substrate_thickness,
                source_distance,
                image_distance,
                nearfield_distance,
            )
            <= 0
        ):
            raise ValueError(
                "Device dimensions and source/image distances must be positive"
            )
        if source_distance >= substrate_thickness:
            raise ValueError("source_distance must be smaller than substrate_thickness")
        if nearfield_distance >= image_distance:
            raise ValueError("nearfield_distance must be smaller than image_distance")
        output_port_length = nearfield_distance
        self.in_mode = in_mode
        grid_step = 1 / sim_cfg.get("resolution", 20)
        port_cfgs = {
            "in_port_1": {
                "type": "box",
                "direction": "z+",
                "center": [
                    0.0,
                    0.0,
                    -thickness / 2 - substrate_thickness / 2 + grid_step / 2,
                ],
                "size": [aperture, aperture, substrate_thickness + grid_step],
                "eps": eps_substrate,
            },
            "out_port_1": {
                "type": "box",
                "direction": "z+",
                "center": [0.0, 0.0, thickness / 2 + output_port_length / 2],
                "size": [aperture, aperture, output_port_length],
                "eps": eps_background,
            },
        }
        geometry_cfgs = {}
        design_region_cfgs = {
            "design_region": {
                "type": "box",
                "center": [0.0, 0.0, 0.0],
                "size": [aperture, aperture, thickness],
                "eps": eps_design,
                "eps_bg": eps_background,
            }
        }
        sim_cfg = dict(sim_cfg)
        super().__init__(
            eps_bg=eps_background,
            port_cfgs=port_cfgs,
            geometry_cfgs=geometry_cfgs,
            design_region_cfgs=design_region_cfgs,
            sim_cfg=sim_cfg,
            device=device,
        )
        self.aperture = aperture
        self.thickness = thickness
        self.substrate_thickness = substrate_thickness
        self.source_distance = source_distance
        self.image_distance = image_distance
        self.nearfield_distance = nearfield_distance
        self.eps_design = eps_design
        self.eps_substrate = eps_substrate

    def init_monitors(self, verbose: bool = True):
        """Register the source and 300 nm projection-monitor planes."""
        del verbose
        source_rel_loc = (
            self.substrate_thickness - self.source_distance
        ) / self.substrate_thickness
        self.build_port_monitor_slice(
            "in_port_1",
            "in_slice_1",
            rel_loc=source_rel_loc,
            rel_width=1.0,
            rel_height=1.0,
            direction="z+",
        )
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=1)

        refl_rel_loc = (
            self.substrate_thickness - self.source_distance + 0.2
        ) / self.substrate_thickness
        self.build_port_monitor_slice(
            "in_port_1",
            "refl_slice_1",
            rel_loc=refl_rel_loc,
            rel_width=1.0,
            rel_height=1.0,
            direction="z+",
        )
        self.build_near2far_slice(
            slice_name="nearfield_projection",
            center=(
                self.port_cfgs["out_port_1"]["center"][0],
                self.port_cfgs["out_port_1"]["center"][1],
                self.port_cfgs["out_port_1"]["center"][2]
                + self.port_cfgs["out_port_1"]["size"][2] / 2,
            ),
            size=(self.aperture, self.aperture, 0.0),
            direction="z+",
        )
        return (
            self.port_monitor_slices["in_slice_1"],
            self.port_monitor_slices["nearfield_projection"],
        )

    def norm_run(self, verbose: bool = True):
        """Build MAPS native TFSF plane-wave source and reference profiles."""
        del verbose
        common = dict(
            source_modes=(self.in_mode,),
            input_port_name="in_port_1",
            wl_cen=self.sim_cfg["wl_cen"],
            wl_width=self.sim_cfg["wl_width"],
            n_wl=self.sim_cfg["n_wl"],
            solver=self.sim_cfg["solver"],
            source_type="plane_wave",
            plot=False,
        )
        source = self.build_norm_sources(
            input_slice_name="in_slice_1", require_sim=True, **common
        )
        reflection = self.build_norm_sources(
            input_slice_name="refl_slice_1", require_sim=False, **common
        )
        return source, reflection
