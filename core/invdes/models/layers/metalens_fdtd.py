import math
import os
import sys
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import meep as mp
import meep.adjoint as mpa
import numpy as np
import torch
from autograd import numpy as npa
from IPython.display import Video
from pyutils.general import ensure_dir

# Determine the path to the directory containing device.py
device_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../data/fdtd")
)

# Add the directory to sys.path
sys.path.insert(0, device_dir)
from device import Device

sys.path.pop(0)

matplotlib.rcParams["text.usetex"] = False
eps_sio2 = 1.44**2
eps_si = 3.48**2
air = 1**2
eps_glass = 1.5**2

DEBUG = False
if not DEBUG:
    mp.verbosity(0)

__all__ = ["metalens_fdtd"]


class metalens_fdtd(Device):
    def __init__(
        self,
        aperture,
        ridge_height,
        sub_height,
        f_min,
        f_max,
        border_width: Tuple[float, float] = [1, 1],  # um, [x, y]
        eps_r: float = eps_si,  # relative refractive index
        eps_bg: float = eps_sio2,  # background refractive index
        PML: Tuple[int, int] = (2, 2),  # um, [x, y]
        focal_constant: float = 1.0,
    ):
        """

        the plot should look like this:
        #######################
        #  |    |  |       |  #
        #  |    |  |   |   |  #
        #  |    |  |       |  #
        #  |    |  |       |  #
        #  |    |  |   |   |  #
        #  |    |  |       |  #
        #######################
        input monitor
        substrate
        ridge
        near2far monitor
        """

        device_cfg = dict(
            aperture=aperture,
            ridge_height=ridge_height,
            sub_height=sub_height,
            f_min=f_min,
            f_max=f_max,
            border_width=border_width,
            eps_r=eps_r,
            eps_bg=eps_bg,
        )
        super().__init__(**device_cfg)

        self.aperture = aperture
        self.ridge_height = ridge_height
        self.sub_height = sub_height
        self.border_width = border_width
        self.eps_r = eps_r
        self.eps_bg = eps_bg
        self.PML = PML
        self.focal_constant = focal_constant

        self.update_device_config("Metalens", device_cfg)

        self.size = [ridge_height + sub_height, aperture]
        self.box_size = [ridge_height, aperture]

        self.sx = self.PML[0] * 2 + self.size[0] + self.border_width[0] * 2
        self.sy = self.PML[1] * 2 + self.size[1] + self.border_width[1] * 2

        self.f_min = (
            f_min
            - self.sx / 2
            + self.border_width[0]
            + self.PML[0]
            + self.ridge_height
            + self.sub_height
        )
        self.f_max = (
            f_max
            - self.sx / 2
            + self.border_width[0]
            + self.PML[0]
            + self.ridge_height
            + self.sub_height
        )

        self.sub_center_x = (
            -self.sx / 2 + self.PML[0] + self.border_width[0] + self.sub_height / 2
        )
        self.ridge_center_x = (
            -self.sx / 2
            + self.border_width[0]
            + self.ridge_height / 2
            + self.PML[0]
            + self.sub_height
        )

        sub_x = sub_height + self.border_width[0] + self.PML[0]
        substrate = mp.Block(
            size=mp.Vector3(sub_x, self.aperture, 0),
            center=mp.Vector3(-self.sx / 2 + sub_x / 2, 0),
            material=mp.Medium(epsilon=self.eps_r),
        )

        self.substrate = substrate

    def add_source(
        self,
        src_type: str = "GaussianSource",
        wl_cen=1.55,
        wl_width: float = 0.1,
        alpha: float = 0.5,
    ):
        fcen = 1 / wl_cen  # pulse center frequency
        ## alpha from 1/3 to 1/2
        fwidth = (
            3 * alpha * (1 / (wl_cen - wl_width / 2) - 1 / (wl_cen + wl_width / 2))
        )  # pulse frequency width
        self.fcen = fcen
        self.fwidth = fwidth
        self.src_type = src_type
        self.fwhm_min = (
            self.focal_constant
            * np.tan(np.arcsin(1.22 * (1 / self.fcen) / self.aperture))
            * self.f_min
        )
        self.fwhm_max = (
            self.focal_constant
            * np.tan(np.arcsin(1.22 * (1 / self.fcen) / self.aperture))
            * self.f_max
        )
        if src_type == "GaussianSource":
            src_fn = mp.GaussianSource
        elif src_type == "ContinuousSource":
            src_fn = mp.ContinuousSource
        else:
            raise NotImplementedError

        src_center = (
            self.sub_center_x - 0.5,
            0,
        )  # put the source at the center of the substrate
        src_size = (0, self.aperture, 0)
        if src_type == "GaussianSource":
            self.sources.append(
                mp.EigenModeSource(
                    src=src_fn(fcen, fwidth=fwidth),
                    center=mp.Vector3(*src_center),
                    size=src_size,
                    eig_match_freq=True,
                    # eig_parity=mp.ODD_Z + mp.EVEN_Y, # TODO think about the parity, for now, assume no parity
                )
            )
        elif src_type == "ContinuousSource":
            self.sources.append(
                mp.Source(
                    src=src_fn(fcen),
                    component=mp.Ez,
                    center=mp.Vector3(*src_center),
                    size=src_size,
                )
            )
        else:
            raise NotImplementedError

        self.add_source_config(
            dict(
                src_type=src_type,
                src_center=src_center,
                src_size=src_size,
                eig_match_freq=True,
                # eig_parity=mp.ODD_Z + mp.EVEN_Y, # TODO think about the parity, for now, assume no parity
                wl_cen=wl_cen,
                wl_width=wl_width,
                alpha=alpha,
            )
        )

    def update_permittivity(self, permittivity: torch.Tensor):
        permittivity = permittivity.detach().cpu().numpy()
        design_region_size = mp.Vector3(self.box_size[0], self.box_size[1], 0)
        medium1 = mp.Medium(epsilon=self.config.device.cfg.eps_bg)
        medium2 = mp.Medium(epsilon=self.config.device.cfg.eps_r)
        design_variables = mp.MaterialGrid(
            mp.Vector3(permittivity.shape[0], permittivity.shape[1]),
            medium1,
            medium2,
            weights=permittivity,
            grid_type="U_MEAN",
        )
        self.design_region = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(
                center=mp.Vector3(self.ridge_center_x, 0), size=design_region_size
            ),
        )
        self.geometry = [self.substrate] + [
            mp.Block(
                center=self.design_region.center,
                size=self.design_region.size,
                material=design_variables,
            )
        ]

    def create_simulation(
        self,
        resolution: int = 10,  # pixels / um
        record_interval: float = 0.3,  # timeunits, change it to 0.4 to match the time interval = 0.3 in mrr simulation
        store_fields=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        until: float = None,  # timesteps
        stop_when_decay: bool = False,
        sim_size: tuple = None,
        displacement: Tuple[float, float, float] = (0, 0, 0),
        if_complex: bool = False,
    ):
        boundary = [
            mp.PML(self.PML[0], direction=mp.X),
            mp.PML(self.PML[1], direction=mp.Y),
        ]

        # cell_size = (self.sx, self.sy, 0)
        cell_size = sim_size if sim_size is not None else (self.sx, self.sy, 0)
        geometry = []
        for shape in self.geometry:
            geometry.append(shape.shift(mp.Vector3(*displacement)))

        sources = []
        for source in self.sources:
            if self.src_type == "GaussianSource":
                sources.append(
                    mp.EigenModeSource(
                        src=source.src,
                        center=source.center + mp.Vector3(*displacement),
                        size=source.size,
                        eig_match_freq=True,
                        # eig_parity=mp.ODD_Z + mp.EVEN_Y, # TODO think about the parity, for now, assume no parity
                    )
                )
            elif self.src_type == "ContinuousSource":
                sources.append(
                    mp.Source(
                        src=source.src,
                        component=mp.Ez,
                        center=source.center + mp.Vector3(*displacement),
                        size=source.size,
                    )
                )
            else:
                raise NotImplementedError
        sim = mp.Simulation(
            resolution=resolution,
            cell_size=mp.Vector3(*cell_size),
            boundary_layers=boundary,
            geometry=geometry,
            sources=sources,
            default_material=mp.Medium(epsilon=self.config.device.cfg.eps_bg),
            force_all_components=True,
            force_complex_fields=if_complex,
        )
        self.update_simulation_config(
            dict(
                resolution=resolution,
                border_width=self.border_width,
                PML=self.PML,
                cell_size=cell_size,
                record_interval=record_interval,
                store_fields=store_fields,
                until=until,
                stop_when_decay=stop_when_decay,
            )
        )
        return sim

    def obtain_incident_light_power(
        self,
        sim: mp.Simulation,
        nf: int = 1,
    ):
        flux_box = sim.add_flux(
            self.fcen,
            self.fwidth,
            nf,
            mp.FluxRegion(
                center=mp.Vector3(self.sub_center_x, 0),
                size=mp.Vector3(0, self.aperture),
            ),
        )
        sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-3))
        return mp.get_fluxes(flux_box)  # this should be a list nf elements

    def output_light_intensity(
        self, sim: mp.Simulation, ff_res: int = 10, path: str = "./"
    ):
        n2f_obj = sim.add_near2far(
            self.fcen,
            0,
            1,
            mp.Near2FarRegion(
                center=mp.Vector3(self.ridge_center_x + self.ridge_height / 2 + 0.5, 0),
                size=mp.Vector3(y=self.aperture + 2 * self.PML[1]),
            ),
        )
        sim.run(until_after_sources=500)
        f_min_field = (
            abs(
                sim.get_farfields(
                    n2f_obj,
                    ff_res,
                    center=mp.Vector3(self.f_min),
                    size=mp.Vector3(y=1.5 * self.fwhm_min),
                )["Ez"]
            )
            ** 2
        )
        f_max_filed = (
            abs(
                sim.get_farfields(
                    n2f_obj,
                    ff_res,
                    center=mp.Vector3(self.f_max),
                    size=mp.Vector3(y=1.5 * self.fwhm_max),
                )["Ez"]
            )
            ** 2
        )
        x_axis_min = np.linspace(
            -0.75 * self.fwhm_min, +0.75 * self.fwhm_min, len(f_min_field)
        )
        x_axis_max = np.linspace(
            -0.75 * self.fwhm_max, +0.75 * self.fwhm_max, len(f_max_filed)
        )
        plt.figure()
        plt.plot(x_axis_min, f_min_field, label="f_min")
        plt.xlabel("x (um)")
        plt.ylabel("Light intensity |Ez|^2")
        plt.legend()
        plt.savefig(path + "_farfield_f_min.png", dpi=1000)
        plt.close()

        plt.figure()
        plt.plot(x_axis_max, f_max_filed, label="f_max")
        plt.xlabel("x (um)")
        plt.ylabel("Light intensity |Ez|^2")
        plt.legend()
        plt.savefig(path + "_farfield_f_max.png", dpi=1000)
        plt.close()

    def obtain_two_plane_eff(
        self,
        incident_power: list,
        sim: mp.Simulation,
        nf: int = 1,
        npts: int = 100,
    ):
        assert (
            len(incident_power) == nf
        ), "The length of incident_power should be equal to nf"
        nearfield_monitor = sim.add_near2far(
            self.fcen,
            self.fwidth,
            nf,
            mp.Near2FarRegion(
                center=mp.Vector3(self.ridge_center_x + self.ridge_height / 2 + 0.5, 0),
                size=mp.Vector3(0, self.aperture),
                weight=+1,
            ),
            mp.Near2FarRegion(
                center=mp.Vector3(
                    self.ridge_center_x + self.ridge_height / 2 + 0.25,
                    self.aperture / 2,
                ),
                size=mp.Vector3(0.5, 0),
                weight=+1,
            ),
            mp.Near2FarRegion(
                center=mp.Vector3(
                    self.ridge_center_x + self.ridge_height / 2 + 0.25,
                    -self.aperture / 2,
                ),
                size=mp.Vector3(0.5, 0),
                weight=-1,
            ),
        )
        sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-3))

        res_ff_min = npts / self.fwhm_min
        res_ff_max = npts / self.fwhm_max

        far_flux_box_min = nearfield_monitor.flux(
            mp.X,
            mp.Volume(center=mp.Vector3(self.f_min), size=mp.Vector3(y=self.fwhm_min)),
            res_ff_min,
        )
        far_flux_box_max = nearfield_monitor.flux(
            mp.X,
            mp.Volume(center=mp.Vector3(self.f_max), size=mp.Vector3(y=self.fwhm_max)),
            res_ff_max,
        )
        far_flux_box_min = np.array(far_flux_box_min)
        far_flux_box_max = np.array(far_flux_box_max)

        incident_power = np.array(incident_power)

        print(
            "this is the eff for fmin monitor: ",
            np.mean((far_flux_box_min) / (incident_power)),
            flush=True,
        )
        print(
            "this is the eff for fmax monitor: ",
            np.mean((far_flux_box_max) / (incident_power)),
            flush=True,
        )

        return float(
            np.mean((far_flux_box_min + far_flux_box_max) / (2 * incident_power))
        )

    def create_objective_ref(
        self,
    ):
        TE_ref = mpa.EigenmodeCoefficient(
            self.sim,
            mp.Volume(
                center=mp.Vector3(self.sub_center_x - 1, 0, 0),
                size=mp.Vector3(y=self.aperture),
            ),
            mode=1,
        )

        self.ob_list = [TE_ref]

    def create_objective_two_box(self, num_samples):
        samples_min_vertical = np.linspace(-self.f_min, self.f_min, num_samples)
        samples_max_vertical = np.linspace(-self.f_max, self.f_max, num_samples)

        samples_min_horizontal = np.linspace(
            0, self.f_min, round((num_samples - 1) // 2) + 1
        )[:-1]
        samples_max_horizontal = np.linspace(
            0, self.f_max, round((num_samples - 1) // 2) + 1
        )[:-1]

        self.focal_mask_min = np.abs(samples_min_vertical) <= self.fwhm_min / 2
        self.focal_mask_max = np.abs(samples_max_vertical) <= self.fwhm_max / 2

        farfields_min_vertical = []
        for i in range(len(samples_min_vertical)):
            farfields_min_vertical.append(
                mp.Vector3(
                    self.f_min,
                    samples_min_vertical[i],
                    0,
                )
            )
        farfields_min_horizontal = []
        for i in range(len(samples_min_horizontal)):
            farfields_min_horizontal.append(
                mp.Vector3(
                    samples_min_horizontal[i],
                    self.f_min,
                    0,
                )
            )
            farfields_min_horizontal.append(
                mp.Vector3(
                    samples_min_horizontal[i],
                    -self.f_min,
                    0,
                )
            )
        farfields_max_vertical = []
        for i in range(len(samples_max_vertical)):
            farfields_max_vertical.append(
                mp.Vector3(
                    self.f_max,
                    samples_max_vertical[i],
                    0,
                )
            )
        farfields_max_horizontal = []
        for i in range(len(samples_max_horizontal)):
            farfields_max_horizontal.append(
                mp.Vector3(
                    samples_max_horizontal[i],
                    self.f_max,
                    0,
                )
            )
            farfields_max_horizontal.append(
                mp.Vector3(
                    samples_max_horizontal[i],
                    -self.f_max,
                    0,
                )
            )
        NearRegions = [
            mp.Near2FarRegion(
                center=mp.Vector3(self.ridge_center_x + self.ridge_height / 2 + 0.5, 0),
                size=mp.Vector3(0, self.aperture),
                weight=+1,
            ),
            mp.Near2FarRegion(
                center=mp.Vector3(
                    self.ridge_center_x + self.ridge_height / 2 + 0.25,
                    self.aperture / 2,
                ),
                size=mp.Vector3(0.5, 0),
                weight=+1,
            ),
            mp.Near2FarRegion(
                center=mp.Vector3(
                    self.ridge_center_x + self.ridge_height / 2 + 0.25,
                    -self.aperture / 2,
                ),
                size=mp.Vector3(0.5, 0),
                weight=-1,
            ),
        ]
        FarFields_fmin_vertical = mpa.Near2FarFields(
            self.sim, NearRegions, farfields_min_vertical
        )
        FarFields_fmax_vertical = mpa.Near2FarFields(
            self.sim, NearRegions, farfields_max_vertical
        )
        FarFields_fmin_horizontal = mpa.Near2FarFields(
            self.sim, NearRegions, farfields_min_horizontal
        )
        FarFields_fmax_horizontal = mpa.Near2FarFields(
            self.sim, NearRegions, farfields_max_horizontal
        )

        self.ob_list = [
            FarFields_fmin_vertical,
            FarFields_fmax_vertical,
            FarFields_fmin_horizontal,
            FarFields_fmax_horizontal,
        ]

    def create_objective_two_dome(self, num_samples):
        angle = np.linspace(-0.5 * math.pi, 0.5 * math.pi, num_samples)
        focal_mask = np.abs(angle) <= 1.5 * 1.22 * (1 / self.fcen) / self.aperture
        angle_focal = angle[focal_mask]
        self.focal_mask = focal_mask

        farfields_min_all = []
        for i in range(num_samples):
            farfields_min_all.append(
                mp.Vector3(
                    self.f_min * math.cos(angle[i]),
                    self.f_min * math.sin(angle[i]),
                    0,
                )
            )
        farfields_max_all = []
        for i in range(num_samples):
            farfields_max_all.append(
                mp.Vector3(
                    self.f_max * math.cos(angle[i]),
                    self.f_max * math.sin(angle[i]),
                    0,
                )
            )
        NearRegions = [
            mp.Near2FarRegion(
                center=mp.Vector3(self.ridge_center_x + self.ridge_height / 2 + 0.5, 0),
                size=mp.Vector3(0, self.aperture),
                weight=+1,
            ),
            mp.Near2FarRegion(
                center=mp.Vector3(
                    self.ridge_center_x + self.ridge_height / 2 + 0.25,
                    self.aperture / 2,
                ),
                size=mp.Vector3(0.5, 0),
                weight=+1,
            ),
            mp.Near2FarRegion(
                center=mp.Vector3(
                    self.ridge_center_x + self.ridge_height / 2 + 0.25,
                    -self.aperture / 2,
                ),
                size=mp.Vector3(0.5, 0),
                weight=-1,
            ),
        ]
        FarFields_fmin_all = mpa.Near2FarFields(
            self.sim, NearRegions, farfields_min_all
        )
        FarFields_fmax_all = mpa.Near2FarFields(
            self.sim, NearRegions, farfields_max_all
        )
        self.ob_list = [FarFields_fmin_all, FarFields_fmax_all]

    def create_objective_two_plane(self, num_samples):
        sample_y_incident = np.linspace(
            -self.aperture / 2, self.aperture / 2, num_samples
        )
        farfields_incident = []
        for i in range(len(sample_y_incident)):
            farfields_incident.append(
                mp.Vector3(self.sub_center_x + 0.5, sample_y_incident[i], 0)
            )
        NearRegions_incident = [
            mp.Near2FarRegion(
                center=mp.Vector3(self.sub_center_x, 0),
                size=mp.Vector3(0, self.aperture),
                weight=+1,
            )
        ]
        te_in = mpa.Near2FarFields(self.sim, NearRegions_incident, farfields_incident)

        sample_y_min = np.linspace(-self.fwhm_min / 2, self.fwhm_min / 2, num_samples)
        sample_y_max = np.linspace(-self.fwhm_max / 2, self.fwhm_max / 2, num_samples)

        farfields = []
        for i in range(len(sample_y_min)):
            farfields.append(mp.Vector3(self.f_min, sample_y_min[i], 0))
            farfields.append(mp.Vector3(self.f_max, sample_y_max[i], 0))

        # farfields = [mp.Vector3(self.f_min, 0, 0), mp.Vector3(self.f_max, 0, 0)]

        NearRegions = [
            mp.Near2FarRegion(
                center=mp.Vector3(self.ridge_center_x + self.ridge_height / 2 + 0.5, 0),
                size=mp.Vector3(0, self.aperture),
                weight=+1,
            )
        ]
        FarFields = mpa.Near2FarFields(self.sim, NearRegions, farfields)
        self.ob_list = [FarFields, te_in]

    def J_ref(self, TE_ref):
        ref_power = npa.abs(TE_ref) ** 2
        return ref_power

    def J_two_box(
        self,
        FarFields_fmin_vertical,
        FarFields_fmax_vertical,
        FarFields_fmin_horizontal,
        FarFields_fmax_horizontal,
    ):
        E_min_focal = FarFields_fmin_vertical[self.focal_mask_min, :, :3]
        H_min_focal = FarFields_fmin_vertical[self.focal_mask_min, :, 3:]
        E_max_focal = FarFields_fmax_vertical[self.focal_mask_max, :, :3]
        H_max_focal = FarFields_fmax_vertical[self.focal_mask_max, :, 3:]
        Px_min_focal = npa.real(
            npa.conj(E_min_focal[:, :, 1]) * H_min_focal[:, :, 2]
            - npa.conj(E_min_focal[:, :, 2]) * H_min_focal[:, :, 1]
        )
        Py_min_focal = npa.real(
            npa.conj(E_min_focal[:, :, 2]) * H_min_focal[:, :, 0]
            - npa.conj(E_min_focal[:, :, 0]) * H_min_focal[:, :, 2]
        )
        Pr_min_focal = npa.sqrt((Px_min_focal) ** 2 + (Py_min_focal) ** 2)
        Px_max_focal = npa.real(
            npa.conj(E_max_focal[:, :, 1]) * H_max_focal[:, :, 2]
            - npa.conj(E_max_focal[:, :, 2]) * H_max_focal[:, :, 1]
        )
        Py_max_focal = npa.real(
            npa.conj(E_max_focal[:, :, 2]) * H_max_focal[:, :, 0]
            - npa.conj(E_max_focal[:, :, 0]) * H_max_focal[:, :, 2]
        )
        Pr_max_focal = npa.sqrt((Px_max_focal) ** 2 + (Py_max_focal) ** 2)
        focal_power_min = (
            npa.mean(Pr_min_focal, axis=0)
            * 3
            * npa.tan(npa.arcsin(1.22 * (1 / self.fcen) / self.aperture))
            * self.f_min
        )
        focal_power_max = (
            npa.mean(Pr_max_focal, axis=0)
            * 3
            * npa.tan(npa.arcsin(1.22 * (1 / self.fcen) / self.aperture))
            * self.f_max
        )

        E_min_vertical_defocus = FarFields_fmin_vertical[~self.focal_mask_min, :, :3]
        H_min_vertical_defocus = FarFields_fmin_vertical[~self.focal_mask_min, :, 3:]
        E_max_vertical_defocus = FarFields_fmax_vertical[~self.focal_mask_max, :, :3]
        H_max_vertical_defocus = FarFields_fmax_vertical[~self.focal_mask_max, :, 3:]
        E_min_horizontal = FarFields_fmin_horizontal[:, :, :3]
        H_min_horizontal = FarFields_fmin_horizontal[:, :, 3:]
        E_max_horizontal = FarFields_fmax_horizontal[:, :, :3]
        H_max_horizontal = FarFields_fmax_horizontal[:, :, 3:]
        E_min_defocus = npa.concatenate(
            (E_min_vertical_defocus, E_min_horizontal), axis=0
        )
        H_min_defocus = npa.concatenate(
            (H_min_vertical_defocus, H_min_horizontal), axis=0
        )
        E_max_defocus = npa.concatenate(
            (E_max_vertical_defocus, E_max_horizontal), axis=0
        )
        H_max_defocus = npa.concatenate(
            (H_max_vertical_defocus, H_max_horizontal), axis=0
        )
        Px_min_defocus = npa.real(
            npa.conj(E_min_defocus[:, :, 1]) * H_min_defocus[:, :, 2]
            - npa.conj(E_min_defocus[:, :, 2]) * H_min_defocus[:, :, 1]
        )
        Py_min_defocus = npa.real(
            npa.conj(E_min_defocus[:, :, 2]) * H_min_defocus[:, :, 0]
            - npa.conj(E_min_defocus[:, :, 0]) * H_min_defocus[:, :, 2]
        )
        Pr_min_defocus = npa.sqrt((Px_min_defocus) ** 2 + (Py_min_defocus) ** 2)
        Px_max_defocus = npa.real(
            npa.conj(E_max_defocus[:, :, 1]) * H_max_defocus[:, :, 2]
            - npa.conj(E_max_defocus[:, :, 2]) * H_max_defocus[:, :, 1]
        )
        Py_max_defocus = npa.real(
            npa.conj(E_max_defocus[:, :, 2]) * H_max_defocus[:, :, 0]
            - npa.conj(E_max_defocus[:, :, 0]) * H_max_defocus[:, :, 2]
        )
        Pr_max_defocus = npa.sqrt((Px_max_defocus) ** 2 + (Py_max_defocus) ** 2)
        defocus_power_min = npa.mean(Pr_min_defocus, axis=0) * (
            4 * self.f_min - self.fwhm_min
        )
        defocus_power_max = npa.mean(Pr_max_defocus, axis=0) * (
            4 * self.f_max - self.fwhm_max
        )

        if isinstance(focal_power_min, npa.numpy_boxes.ArrayBox):
            focal_power_min = np.array(focal_power_min._value)
        if isinstance(focal_power_max, npa.numpy_boxes.ArrayBox):
            focal_power_max = np.array(focal_power_max._value)
        total_power_min = (
            defocus_power_min + focal_power_min
        )  # cut the gradient for the focus region in denominator
        total_power_max = (
            defocus_power_max + focal_power_max
        )  # cut the gradient for the focus region in denominator

        efficiency = (
            (focal_power_min / total_power_min) + (focal_power_max / total_power_max)
        ) / 2

        return npa.mean(efficiency)

    def J_two_plane(self, FF, te_in):
        # incident_power = npa.sum(npa.absolute(te_in) ** 2)
        incident_power = npa.mean(npa.abs(te_in[:, :, 2]) ** 2) * self.aperture
        f_min_power = npa.mean(npa.abs(FF[::2, :, 2]) ** 2) * self.fwhm_min
        f_max_power = npa.mean(npa.abs(FF[1::2, :, 2]) ** 2) * self.fwhm_max
        efficiency = (f_min_power + f_max_power) / (2 * incident_power)
        return efficiency

    def J_two_dome(
        self,
        FarFields_fmin_all,
        FarFields_fmax_all,
    ):
        # incident_power = npa.sum(npa.absolute(te_in) ** 2)
        E_min = FarFields_fmin_all[:, :, :3]
        H_min = FarFields_fmin_all[:, :, 3:]
        E_max = FarFields_fmax_all[:, :, :3]
        H_max = FarFields_fmax_all[:, :, 3:]
        Px_min = npa.real(
            npa.conj(E_min[:, :, 1]) * H_min[:, :, 2]
            - npa.conj(E_min[:, :, 2]) * H_min[:, :, 1]
        )
        Py_min = npa.real(
            npa.conj(E_min[:, :, 2]) * H_min[:, :, 0]
            - npa.conj(E_min[:, :, 0]) * H_min[:, :, 2]
        )
        Pr_min = npa.sqrt((Px_min) ** 2 + (Py_min) ** 2)
        Px_max = npa.real(
            npa.conj(E_max[:, :, 1]) * H_max[:, :, 2]
            - npa.conj(E_max[:, :, 2]) * H_max[:, :, 1]
        )
        Py_max = npa.real(
            npa.conj(E_max[:, :, 2]) * H_max[:, :, 0]
            - npa.conj(E_max[:, :, 0]) * H_max[:, :, 2]
        )
        Pr_max = npa.sqrt((Px_max) ** 2 + (Py_max) ** 2)
        total_power_min = npa.mean(Pr_min, axis=0) * npa.pi
        total_power_max = npa.mean(Pr_max, axis=0) * npa.pi
        focal_power_min = (
            npa.mean(Pr_min[self.focal_mask, :], axis=0)
            * 3
            * 1.22
            * (1 / self.fcen)
            / self.aperture
        )
        focal_power_max = (
            npa.mean(Pr_max[self.focal_mask, :], axis=0)
            * 3
            * 1.22
            * (1 / self.fcen)
            / self.aperture
        )
        efficiency = (
            (focal_power_min / total_power_min) + (focal_power_max / total_power_max)
        ) / 2
        efficiency = npa.mean(efficiency)
        return efficiency

    def create_optimzation_ref(self, df: float = 0, nf: int = 1):
        self.opt = mpa.OptimizationProblem(
            simulation=self.sim,
            objective_functions=[self.J_ref],
            objective_arguments=self.ob_list,
            design_regions=self.design_region,
            fcen=self.fcen,
            df=df,
            nf=nf,
            decay_by=1e-3,
        )

    def create_optimzation_two_box(self, df: float = 0, nf: int = 1):
        self.opt = mpa.OptimizationProblem(
            simulation=self.sim,
            objective_functions=[self.J_two_box],
            objective_arguments=self.ob_list,
            design_regions=self.design_region,
            fcen=self.fcen,
            df=df,
            nf=nf,
            decay_by=1e-3,
        )

    def create_optimzation_two_plane(self, df: float = 0, nf: int = 1):
        self.opt = mpa.OptimizationProblem(
            simulation=self.sim,
            objective_functions=[self.J_two_plane],
            objective_arguments=self.ob_list,
            design_regions=self.design_region,
            fcen=self.fcen,
            df=df,
            nf=nf,
            decay_by=1e-3,
        )

    def create_optimzation_two_dome(self, df: float = 0, nf: int = 1):
        self.opt = mpa.OptimizationProblem(
            simulation=self.sim,
            objective_functions=[self.J_two_dome],
            objective_arguments=self.ob_list,
            design_regions=self.design_region,
            fcen=self.fcen,
            df=df,
            nf=nf,
            decay_by=1e-3,
        )

    def obtain_objective_and_gradient(self, need_item=None):
        if need_item == "need_value":
            result, _ = self.opt(need_value=True, need_gradient=False)
        elif need_item == "need_gradient":
            _, result = self.opt(need_value=False, need_gradient=True)
        else:
            raise NotImplementedError
        return result

    def run_sim(
        self,
        sim: mp.Simulation = None,
        filepath: str = None,
        export_video: bool = False,
        mode: str = "fdtd",
        tol: float = -8.0,
    ):
        sim = sim if sim is not None else self.sim
        nonpml_vol = mp.Volume(
            mp.Vector3(),
            size=mp.Vector3(
                sim.cell_size[0] - 2 * self.PML[0], sim.cell_size[1] - 2 * self.PML[1]
            ),
        )
        if mode == "fdtd":
            fwhm_min = 3 * 1.22 * (1 / self.fcen) / self.aperture * self.f_min
            dummy_monitor = mp.FluxRegion(
                center=mp.Vector3(self.f_min - (self.f_min + 20) / 2, 0, 0),
                size=mp.Vector3(0, fwhm_min, 0),
            )
            sim.add_flux(self.fcen, self.fwidth, 1, dummy_monitor)
            dft_obj = sim.add_dft_fields([mp.Ez], self.fcen, 0, 1, where=nonpml_vol)
            stop_when_decay = self.config.simulation.stop_when_decay
            output = dict(
                eps=None,
                Ex=[],
                Ey=[],
                Ez=[],
                Hx=[],
                Hy=[],
                Hz=[],
            )
            store_fields = self.config.simulation.store_fields

            def record_fields(sim):
                for field in store_fields:
                    if field == "Ex":
                        data = sim.get_efield_x()
                    elif field == "Ey":
                        data = sim.get_efield_y()
                    elif field == "Ez":
                        data = sim.get_efield_z()
                    elif field == "Hx":
                        data = sim.get_hfield_x()
                    elif field == "Hy":
                        data = sim.get_hfield_y()
                    elif field == "Hz":
                        data = sim.get_hfield_z()
                    output[field].append(data)

            at_every = [record_fields]
            if export_video:
                f = plt.figure(dpi=150)
                Animate = mp.Animate2D(
                    fields=mp.Ez, f=f, realtime=False, normalize=True
                )
                at_every.append(Animate)

            if stop_when_decay:
                # monitor_cen = list(self.out_port_centers[0]) + [0]
                monitor_cen = [self.f_min - (self.f_min + 20) / 2] + [0]

                sim.run(
                    mp.at_every(self.config.simulation.record_interval, *at_every),
                    until_after_sources=mp.stop_when_fields_decayed(
                        50, mp.Ez, monitor_cen, 1e-3
                    ),
                )
            else:
                print("running simulation")
                sim.run(
                    mp.at_every(self.config.simulation.record_interval, *at_every),
                    until=self.config.simulation.until,
                )
            ensure_dir(os.path.dirname(filepath))

            if export_video:
                filename = filepath[:-3] + ".mp4"
                Animate.to_mp4(20, filename)
                Video(filename)

            eps_data = sim.get_array(vol=nonpml_vol, component=mp.Dielectric)
            ez_data = np.real(sim.get_dft_array(dft_obj, mp.Ez, 0))

            plt.figure()
            plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
            plt.imshow(
                ez_data.transpose(), interpolation="spline36", cmap="RdBu", alpha=0.9
            )
            plt.axis("off")
            plt.savefig(filepath + "_dft_field.png", dpi=300)
        elif mode == "fdfd":
            tol = np.power(10, tol)
            sim.init_sim()
            sim.solve_cw(tol, 10000, 10)
            fdfd_result = sim.get_array(vol=nonpml_vol, component=mp.Ez)
            fdfd_result = np.real(fdfd_result)
            eps_data = sim.get_array(vol=nonpml_vol, component=mp.Dielectric)
            plt.figure()
            plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
            plt.imshow(
                fdfd_result.transpose(),
                interpolation="spline36",
                cmap="RdBu",
                alpha=0.9,
            )
            plt.axis("off")
            plt.savefig(filepath + "_solve_cw.png", dpi=300)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        str = f"Metaline-aper: {self.aperture}("
        str += f"size = {self.box_size[0]} um x {self.box_size[1]} um)"
        return str


def test():
    permittivity = torch.rand(61, 401)
    device = metalens_fdtd(
        aperture=20,
        ridge_height=3,
        sub_height=2,
        f_min=60,
        f_max=180,
        border_width=[2, 0],  # um, [x, y]
        eps_r=3.48**2,  # relative refractive index
        eps_bg=1,  # background refractive index
        PML=(2, 2),  # um, [x, y]
    )
    device.update_permittivity(permittivity)
    device.add_source(wl_cen=0.85)
    fdtd_sim = device.create_simulation(
        resolution=20,  # pixels / um
    )
    device.create_objective(501)
    device.create_optimzation(df=0, nf=1)
    f0 = device.obtain_objective_and_gradient("need_value")
    grad = device.obtain_objective_and_gradient("need_gradient")
    if isinstance(grad, np.ndarray):
        grad = torch.tensor(grad)
    if torch.isnan(grad).any():
        print("NaN detected in output")

    print(f0, grad)


if __name__ == "__main__":
    test()
