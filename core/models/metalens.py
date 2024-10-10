import math
import os
from functools import lru_cache

import h5py
import matplotlib.pyplot as plt
import meep as mp
import numpy
import torch
import torch.nn.functional as F
from torch import nn
from torch.types import Device

from .layers import (
    ClipLayer,
    HeavisideProjection,
    InsensitivePeriod,
    SimulatedFoM,
    SmoothRidge,
    heightProjection,
)
from .layers.metalens_fdtd import metalens_fdtd
from .layers.utils import get_eps_1d

eps_sio2 = 1.44**2
eps_si = 3.48**2, # 3.648 @850nm
air = 1**2
eps_glass = 1.5**2

mp.verbosity(0)


class Metalens(nn.Module):
    def __init__(
        self,
        ridge_height_max: int,
        sub_height: int,
        aperture: float,
        f_min: float,
        f_max: float,
        eps_r: float,
        eps_bg: float,
        sim_cfg: dict,
        ls_cfg: dict,
        mfs: float,
        binary_projection_threshold: float = 0.05,
        build_method: str = "level_set",
        center_ridge: bool = False,  # by default, the centere should be a center based on the example shown from Prof. Yao, need to check this with them
        max_num_ridges_single_side: int = 10,
        operation_device: Device = torch.device("cuda:0"),
        aspect_ratio: float = 0.1,
        initial_point: str = None,
        if_constant_period: bool = True,
        subpx_smoothing_res: int = 1000,
        focal_constant: float = 1,
    ):
        """
        Args:
            ridge_height_max: int
                the upper bound of the height of the ridge
            sub_height: int
                height of the substrate
            aperture: float
                aperture of the device
            f_min: float
                minimum focal position
            f_max: float
                maximum focal position
            n_ridge: float
                refractive index of the ridge
            eps_r: float
                relative permittivity of the device
            eps_bg: float
                relative permittivity of the background
            mfs: float
                minimum feature size
            sim_cfg: dict
                simulation configuration
            ls_cfg: dict
                level set configuration
            binary_projection_threshold: float
                threshold for the binary projection layer from which we by pass the gradient
                default is 0.05 which is we thought is a good binarization

        Returns:
            None
        """
        super().__init__()
        self.ridge_height_max = (
            ridge_height_max
            if not isinstance(ridge_height_max, str)
            else float(ridge_height_max)
        )
        self.sub_height = (
            sub_height if not isinstance(sub_height, str) else float(sub_height)
        )
        self.aperture = aperture if not isinstance(aperture, str) else float(aperture)
        self.f_min = f_min if not isinstance(f_min, str) else float(f_min)
        self.f_max = f_max if not isinstance(f_max, str) else float(f_max)
        self.eps_r = eps_r if not isinstance(eps_r, str) else eval(eps_r)
        self.eps_bg = eps_bg if not isinstance(eps_bg, str) else eval(eps_bg)
        self.mfs = mfs if not isinstance(mfs, str) else float(mfs)
        self.operation_device = operation_device
        self.aspect_ratio = aspect_ratio
        self.initial_point = (
            os.path.join(os.path.dirname(__file__), initial_point)
            if initial_point is not None
            else None
        )
        self.if_constant_period = if_constant_period
        self.subpx_smoothing_res = subpx_smoothing_res
        self.focal_constant = focal_constant
        for key in sim_cfg.keys():
            if key == "adjoint_mode":
                continue
            if isinstance(sim_cfg[key], str):
                sim_cfg[key] = eval(sim_cfg[key])
        for key in ls_cfg.keys():
            if isinstance(ls_cfg[key], str):
                ls_cfg[key] = eval(ls_cfg[key])
        self.sim_cfg = sim_cfg
        self.ls_cfg = ls_cfg
        # print("this is the type of self.mfs", type(self.mfs), self.mfs)
        # print("this is the type of self.sim_cfg['resolution']", type(self.sim_cfg['resolution']), self.sim_cfg['resolution'])
        self.mfs_px = int(self.mfs * self.sim_cfg["resolution"])
        # print("this is the self.mfs_px", self.mfs_px, flush=True)
        # print("this is the self.mfs", self.mfs, flush=True)
        # print("this is the self.sim_cfg['resolution']", self.sim_cfg['resolution'], flush=True)
        # quit()
        if self.mfs_px % 2 == 0:
            self.mfs_px += 1  # make sure it is odd
        self.binary_projection_threshold = (
            binary_projection_threshold
            if not isinstance(binary_projection_threshold, str)
            else float(binary_projection_threshold)
        )
        self.center_ridge = center_ridge
        self.build_method = build_method
        self.max_num_ridges_single_side = max_num_ridges_single_side

        self.init_params(build_method)
        self.build_layers()
        self.incident_light_power = self.incident_light_norm_run(
            resolution=self.sim_cfg["resolution"]
        )

    @lru_cache(maxsize=2)
    def incident_light_norm_run(self, resolution: int = 20):
        device = metalens_fdtd(
            aperture=self.aperture,
            ridge_height=self.ridge_height_max,
            sub_height=self.sub_height,
            f_min=self.f_min,
            f_max=self.f_max,
            border_width=self.sim_cfg["border_width"],  # um, [x, y]
            eps_r=self.eps_r,  # relative refractive index
            eps_bg=self.eps_r,  # background refractive index
            PML=self.sim_cfg["PML"],  # um, [x, y]
            focal_constant=self.focal_constant,
        )
        device.add_source(wl_cen=self.sim_cfg["wl_cen"])
        fdtd_sim = device.create_simulation(
            resolution=resolution,  # pixels / um
        )
        device.sim = fdtd_sim
        return device.obtain_incident_light_power(fdtd_sim, self.sim_cfg["nf"])

    def init_params(self, build_method: str = "level_set"):
        if build_method == "level_set":
            self._init_params_level_set()
        elif build_method == "periodic_ridge":
            self._init_params_periodic_ridge()
        else:
            raise ValueError(f"Unsupported build method: {build_method}")

    def _init_params_level_set(self):
        ls_grid_size = 1 / self.sim_cfg["resolution"]
        down_sample_rate = self.ls_cfg["down_sample_rate"]
        self.rho_size = ls_grid_size * down_sample_rate

        # Number of points on the parameter grid (rho) and level set grid (phi)
        self.ny_rho = int(self.aperture / self.rho_size / 2) + 1
        self.ny_phi = int(self.aperture / ls_grid_size / 2) + 1

        # xy coordinates of the parameter and level set grids.
        self.y_rho = torch.linspace(-self.aperture / 4, self.aperture / 4, self.ny_rho)
        self.y_phi = torch.linspace(-self.aperture / 4, self.aperture / 4, self.ny_phi)

        top_metalens = torch.randn(
            int(self.aperture * self.sim_cfg["resolution"] / 2) + 1,
        )[:: self.ls_cfg["down_sample_rate"]].flatten()
        self.top_metalens = nn.Parameter(top_metalens)

    def _init_params_periodic_ridge(self):
        # here we initialize the parameters with a good design from Prof. Yao
        if self.if_constant_period:
            self.period = torch.tensor([0.3]).to(self.operation_device)
        else:
            self.period = nn.Parameter(
                torch.tensor([0.3])
            )  # the inital period of the ridge
        self.ridge_height = nn.Parameter(
            torch.tensor([0.75])
        )  # the six here is to make margin for the opt of width, if set to 2, there is not room for the width and gap to meet the requirement
        width = (
            torch.ones(self.max_num_ridges_single_side)
            * self.ridge_height
            * self.aspect_ratio
            * 2
        )
        if self.initial_point is not None:
            with h5py.File(self.initial_point, "r") as f:
                widths = f["Si_width"][: len(f["Si_width"]) // 2 + 1, 0]
                if isinstance(widths, numpy.ndarray):
                    widths = torch.tensor(widths)
                    widths = widths.flip(0)
                width[: len(f["Si_width"]) // 2 + 1] = widths * 1e6
        self.width = nn.Parameter(width)  # the widths of the ridge

    def update_ridges_num(self):
        self.ridges_number = int(self.aperture / 2 / self.period)
        if self.ridges_number > self.max_num_ridges_single_side:
            self.ridges_number = self.max_num_ridges_single_side

    def build_layers(self):
        self.binary_projection = HeavisideProjection(self.binary_projection_threshold)
        self.eff_layer = SimulatedFoM(self.cal_obj_grad, self.sim_cfg["adjoint_mode"])
        self.reflection_layer = SimulatedFoM(self.cal_obj_grad, "reflection")
        self.height_projection = heightProjection(
            threshold=self.binary_projection_threshold, height_max=self.ridge_height_max
        )
        if self.build_method == "periodic_ridge":
            self.smooth_ridge = SmoothRidge(self.aperture / 2)
            self.clip_width = ClipLayer("both")
            self.clip_period = ClipLayer("lower_limit")
            self.insensitive_period = InsensitivePeriod()
        # Construct the triangular blurring kernel
        blurring_kernel = 1 - torch.abs(torch.linspace(-1, 1, steps=self.mfs_px)).to(
            self.operation_device
        )
        # Normalize the kernel so that it sums to 1
        blurring_kernel = blurring_kernel / blurring_kernel.sum()
        self.blurring_kernel = blurring_kernel.unsqueeze(0).unsqueeze(0).float()

    def subpixel_smoothing(self, grating, resolution):
        grating = self.eps_bg + (self.eps_r - self.eps_bg) * grating
        grating = 1 / grating
        if self.subpx_smoothing_res % resolution == 0:
            avg_pool_kernel_size = int(self.subpx_smoothing_res // resolution) + 1
            avg_pool_stride = int(self.subpx_smoothing_res // resolution)
            pad_size = avg_pool_kernel_size // 2
            grating = F.pad(grating, (pad_size, pad_size), mode="constant")
            grating = F.avg_pool1d(
                grating.unsqueeze(0).unsqueeze(0),
                kernel_size=avg_pool_kernel_size,
                stride=avg_pool_stride,
            ).squeeze()
            assert (
                grating.shape[0] == self.aperture * resolution + 1
            ), "The resolution of the grating is not correct"
        else:
            # have to use bilinear interpolation here
            target_length = self.aperture * resolution + 1
            grating = F.interpolate(
                grating.unsqueeze(0).unsqueeze(0), size=target_length, mode="linear"
            ).squeeze()
        grating = 1 / grating
        grating = (grating - self.eps_bg) / (self.eps_r - self.eps_bg)
        return grating

    def blurring(self, x):
        """
        Blurring operation using the Conv1D layer

        Args:
            x: torch.Tensor
                input tensor

        Returns:
            output: torch.Tensor
                output tensor
        """
        # print("this is the self.blurring_kernel", self.blurring_kernel, flush=True)
        x = torch.nn.functional.conv1d(
            x.unsqueeze(0).unsqueeze(0), self.blurring_kernel, padding=self.mfs_px // 2
        ).squeeze()
        return x

    def build_gratings(self, sharpness: float = 0.1, resolution: int = 20):
        """
        Build the gratings using the top_metalens parameter

        Returns:
            gratings: torch.Tensor
                the gratings built using the top_metalens parameter
        """
        if self.build_method == "level_set":
            top_gratings = self._build_gratings_level_set(sharpness=sharpness)
        elif self.build_method == "periodic_ridge":
            top_gratings = self._build_gratings_periodic_ridge(
                sharpness=sharpness, resolution=resolution
            )
        else:
            raise ValueError(f"Unsupported build method: {self.build_method}")

        grating = torch.cat((top_gratings, top_gratings.flip(0)[1:]), dim=0)

        return grating

    def _build_gratings_level_set(self, sharpness: float = 0.1):
        gratings = get_eps_1d(
            design_param=self.top_metalens,
            x_rho=self.y_rho,
            x_phi=self.y_phi,
            rho_size=self.rho_size,
            nx_rho=self.ny_rho,
            nx_phi=self.ny_phi,
            plot_levelset=False,
            sharpness=sharpness,
        )
        return gratings

    def _build_gratings_periodic_ridge(
        self, sharpness: float = 0.1, resolution: int = 20
    ):
        period = self.clip_period(
            self.period, lower_limit=self.ridge_height * self.aspect_ratio * 2
        )
        width_min = self.ridge_height * self.aspect_ratio
        width_max = period
        widths = self.clip_width(
            self.width, upper_limit=width_max, lower_limit=width_min
        )
        self.real_width = widths
        self.real_period = period

        ridges = []
        x_coord = torch.linspace(
            -self.aperture / 2, 0, round(self.aperture / 2 * resolution) + 1
        ).to(self.operation_device)
        print("this is the ridges_number", self.ridges_number, flush=True)
        for i in range(self.ridges_number):
            left_mask = x_coord < -period * i - widths[i] / 2
            middle_mask = (x_coord >= -period * i - widths[i] / 2) & (
                x_coord <= -period * i + widths[i] / 2
            )
            right_mask = x_coord > -period * i + widths[i] / 2
            ridge_left = self.smooth_ridge(
                x_coord[left_mask],
                self.insensitive_period(-period, i),
                widths[i],
                sharpness,
                "left",
            )
            ridge_middle = self.smooth_ridge(
                x_coord[middle_mask],
                self.insensitive_period(-period, i),
                widths[i],
                sharpness,
                "middle",
            )
            ridge_right = self.smooth_ridge(
                x_coord[right_mask],
                self.insensitive_period(-period, i),
                widths[i],
                sharpness,
                "right",
            )
            ridge = torch.cat((ridge_left, ridge_middle, ridge_right), dim=0)

            ridges.append(ridge)
        gratings = torch.stack(ridges, dim=0)
        gratings = 0.01 * torch.logsumexp(gratings / 0.01, dim=0)

        return gratings

    def build_height_mask(self, sharpness: float = 1, resolution: int = 20):
        """
        Build the height mask for the metalens, so that
            1. the height is now a design varibale
            2. the height is smoothed for better gradient propagation

        Returns:
            height_mask: torch.Tensor
                the height mask for the metalens
        """
        height_mask = torch.linspace(
            0, self.ridge_height_max, self.ridge_height_max * resolution + 1
        ).to(self.operation_device)
        height_mask = (
            torch.tanh(sharpness * (self.ridge_height - height_mask)) / 2 + 0.5
        )
        return height_mask

    def cal_obj_grad(self, mode, need_item, resolution, *args):
        if mode == "fdtd":
            result = self._cal_obj_grad_fdtd(need_item, resolution, *args)
        elif mode == "reflection":
            result = self._cal_obj_grad_reflection(need_item, resolution, *args)
        else:
            raise ValueError(f"Unsupported adjoint mode: {mode}")

        return result

    def _cal_obj_grad_reflection(self, need_item, resolution, *args):
        permittivity_tuple = args
        assert (
            len(permittivity_tuple) == 1
        ), "metalens model only supports one permittivity tensor"
        if need_item == "need_value":
            self.device_ref = metalens_fdtd(
                aperture=self.aperture,
                ridge_height=self.ridge_height_max,
                sub_height=self.sub_height,
                f_min=self.f_min,
                f_max=self.f_max,
                border_width=self.sim_cfg["border_width"],  # um, [x, y]
                eps_r=self.eps_r,  # relative refractive index
                eps_bg=self.eps_bg,  # background refractive index
                PML=self.sim_cfg["PML"],  # um, [x, y]
                focal_constant=self.focal_constant,
            )
            self.device_ref.update_permittivity(permittivity_tuple[0])
            self.device_ref.add_source(wl_cen=self.sim_cfg["wl_cen"])
            self.fdtd_sim = self.device_ref.create_simulation(
                resolution=resolution,  # pixels / um
            )
            self.device_ref.sim = self.fdtd_sim
            self.device_ref.create_objective_ref()
            self.device_ref.create_optimzation_ref(df=0, nf=1)
            result = self.device_ref.obtain_objective_and_gradient(need_item=need_item)
        elif need_item == "need_gradient":
            result = self.device_ref.obtain_objective_and_gradient(need_item=need_item)
        else:
            raise NotImplementedError
        return result

    def _cal_obj_grad_fdtd(self, need_item, resolution, *args):
        permittivity_tuple = args
        assert (
            len(permittivity_tuple) == 1
        ), "metalens model only supports one permittivity tensor"
        if need_item == "need_value":
            self.device = metalens_fdtd(
                aperture=self.aperture,
                ridge_height=self.ridge_height_max,
                sub_height=self.sub_height,
                f_min=self.f_min,
                f_max=self.f_max,
                border_width=self.sim_cfg["border_width"],  # um, [x, y]
                eps_r=self.eps_r,  # relative refractive index
                eps_bg=self.eps_bg,  # background refractive index
                PML=self.sim_cfg["PML"],  # um, [x, y]
                focal_constant=self.focal_constant,
            )
            self.device.update_permittivity(permittivity_tuple[0])
            self.device.add_source(wl_cen=self.sim_cfg["wl_cen"])
            self.fdtd_sim = self.device.create_simulation(
                resolution=resolution,  # pixels / um
            )
            self.device.sim = self.fdtd_sim
            self.device.create_objective_two_box(1001)
            self.device.create_optimzation_two_box(df=0, nf=1)
            result = self.device.obtain_objective_and_gradient(need_item=need_item)
        elif need_item == "need_gradient":
            result = self.device.obtain_objective_and_gradient(need_item=need_item)
        else:
            raise NotImplementedError
        return result

    def output_video(self, path: str, sharpness: float = 1, resolution: int = 20):
        """
        Output the video of the metalens

        Args:
            path: str
                path to save the video
            sharpness: float
                sharpness of the binary projection

        returns:
            None
        """
        self.update_ridges_num()
        grating = self.build_gratings(sharpness=sharpness, resolution=resolution)
        height_mask = self.build_height_mask(sharpness=sharpness, resolution=resolution)
        # grating = self.blurring(grating)
        grating = grating.unsqueeze(0).repeat(self.ridge_height_max * resolution + 1, 1)
        grating = grating * height_mask.unsqueeze(1)
        grating = self.binary_projection(
            grating, beta=torch.tensor(sharpness), eta=torch.tensor(0.5)
        )

        device = metalens_fdtd(
            aperture=self.aperture,
            ridge_height=self.ridge_height_max,
            sub_height=self.sub_height,
            f_min=self.f_min,
            f_max=self.f_max,
            border_width=self.sim_cfg["border_width"],  # um, [x, y]
            eps_r=self.eps_r,  # relative refractive index
            eps_bg=self.eps_bg,  # background refractive index
            PML=self.sim_cfg["PML"],  # um, [x, y]
            focal_constant=self.focal_constant,
        )
        device.update_permittivity(grating)
        device.add_source(wl_cen=self.sim_cfg["wl_cen"])
        size_x = device.sx + device.f_min + 20
        fdtd_sim = device.create_simulation(
            resolution=resolution,  # pixels / um
            sim_size=(
                size_x,
                device.sy,
                0,
            ),
            displacement=(
                -(device.f_min + 20) / 2,
                0,
                0,
            ),
            until=300,
        )
        device.run_sim(
            fdtd_sim,
            path + "_video.mp4",
            True,
        )

    def output_phase_profile(self, path, resolution: int = 20):
        symmetries = [mp.Mirror(mp.Y)]
        k_point = mp.Vector3(1, 0, 0)
        wl_cen = self.sim_cfg["wl_cen"]
        wl_width = 0.1
        fcen = 1 / wl_cen
        alpha = 0.5
        fwidth = 3 * alpha * (1 / (wl_cen - wl_width / 2) - 1 / (wl_cen + wl_width / 2))
        dpml = self.sim_cfg["PML"][0]
        dpad = self.sim_cfg["border_width"][0]
        sx = 2 * (dpml + dpad) + self.sub_height + self.ridge_height
        sx = float(sx)
        sub_center_x = -sx / 2 + dpml + dpad + self.sub_height / 2
        ridge_center_x = sub_center_x + self.sub_height / 2 + self.ridge_height / 2
        src_pt = mp.Vector3(sub_center_x - 0.5, 0)
        mon_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad)
        phase_profile = []
        pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]
        sub_height = float(self.sub_height)
        ridge_height = float(self.ridge_height)
        real_width = [float(self.real_width[i]) for i in range(len(self.real_width))]
        for i in range(len(self.real_width)):
            geometry = [
                mp.Block(
                    material=mp.Medium(index=math.sqrt(self.eps_r)),
                    size=mp.Vector3(dpml + dpad + sub_height, mp.inf, mp.inf),
                    center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dpad + sub_height)),
                )
            ]
            sy = float(self.period)
            cell_size = mp.Vector3(sx, sy, 0)
            sources = [
                mp.Source(
                    mp.GaussianSource(fcen, fwidth=fwidth),
                    component=mp.Ez,
                    center=src_pt,
                    size=mp.Vector3(y=sy),
                )
            ]
            geometry.append(
                mp.Block(
                    material=mp.Medium(index=math.sqrt(self.eps_r)),
                    size=mp.Vector3(ridge_height, real_width[i], mp.inf),
                    center=mp.Vector3(ridge_center_x),
                )
            )
            sim = mp.Simulation(
                resolution=resolution,
                cell_size=cell_size,
                boundary_layers=pml_layers,
                geometry=geometry,
                k_point=k_point,
                sources=sources,
                symmetries=symmetries,
            )

            flux_obj = sim.add_flux(
                fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy))
            )
            sim.run(until_after_sources=200)
            # freqs = mp.get_eigenmode_freqs(flux_obj)
            res = sim.get_eigenmode_coefficients(
                flux_obj, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y
            )
            coeffs = res.alpha

            mode_phase = numpy.angle(coeffs[0, 0, 0])
            if mode_phase > 0:
                mode_phase -= 2 * numpy.pi
            phase_profile.append(float(mode_phase))

        phase_profile = torch.tensor(phase_profile)
        phase_profile = torch.cat((phase_profile.flip(0)[1:], phase_profile), dim=0)
        x_axis = torch.linspace(
            -self.aperture / 2, self.aperture / 2, len(phase_profile)
        )
        # plot the phase profile to path
        plt.figure()
        plt.plot(x_axis, phase_profile)
        plt.xlabel("x (um)")
        plt.ylabel("Phase (rad)")
        plt.title("Phase Profile")
        plt.savefig(path + "_phase_profile.png", dpi=300)
        plt.close()

    def output_light_intensity(self, path, grating, resolution: int = 20):
        # the resolution must align between the grating and the simulation
        device = metalens_fdtd(
            aperture=self.aperture,
            ridge_height=self.ridge_height_max,
            sub_height=self.sub_height,
            f_min=self.f_min,
            f_max=self.f_max,
            border_width=self.sim_cfg["border_width"],  # um, [x, y]
            eps_r=self.eps_r,  # relative refractive index
            eps_bg=self.eps_bg,  # background refractive index
            PML=self.sim_cfg["PML"],  # um, [x, y]
            focal_constant=self.focal_constant,
        )
        device.update_permittivity(grating)
        device.add_source(wl_cen=self.sim_cfg["wl_cen"])
        fdtd_sim = device.create_simulation(
            resolution=resolution,  # pixels / um
        )
        device.sim = fdtd_sim
        device.output_light_intensity(fdtd_sim, 10, path)

    def two_monitor_eff(self, grating, resolution: int = 20):
        # the resolution must align between the grating and the simulation
        device = metalens_fdtd(
            aperture=self.aperture,
            ridge_height=self.ridge_height_max,
            sub_height=self.sub_height,
            f_min=self.f_min,
            f_max=self.f_max,
            border_width=self.sim_cfg["border_width"],  # um, [x, y]
            eps_r=self.eps_r,  # relative refractive index
            eps_bg=self.eps_bg,  # background refractive index
            PML=self.sim_cfg["PML"],  # um, [x, y]
            focal_constant=self.focal_constant,
        )
        device.update_permittivity(grating)
        device.add_source(wl_cen=self.sim_cfg["wl_cen"])
        fdtd_sim = device.create_simulation(
            resolution=resolution,  # pixels / um
        )
        device.sim = fdtd_sim
        return device.obtain_two_plane_eff(
            self.incident_light_power, fdtd_sim, self.sim_cfg["nf"], 1001
        )

    def build_metalense(self, sharpness: float = 1, resolution: int = 20):
        """
        Build the metalens using the current parameters

        Args:
            sharpness: float
                sharpness of the binary projection

        Returns:
            metalens: torch.Tensor
                the metalens built using the current parameters
        """
        self.update_ridges_num()
        grating = self.build_gratings(
            sharpness=sharpness, resolution=self.subpx_smoothing_res
        )
        grating = self.binary_projection(
            grating, beta=torch.tensor(sharpness), eta=torch.tensor(0.5)
        )
        grating = self.subpixel_smoothing(grating, resolution)
        # height_mask = self.build_height_mask(sharpness=sharpness, resolution=resolution)
        # grating = self.blurring(grating)
        grating = grating.unsqueeze(0).repeat(self.ridge_height_max * resolution + 1, 1)
        # grating = grating*height_mask.unsqueeze(1)
        grating = self.height_projection(
            self.ridge_height, grating, sharpness, resolution
        )

        return grating

    def evaluate_metalense(
        self, gratings: torch.Tensor, resolution, targets: list = None
    ):
        result = dict()
        for target in targets:
            if target == "opt_obj":
                eff = self.eff_layer(resolution, gratings)
                ref = self.reflection_layer(resolution, gratings)
                result["opt_obj"] = eff
                result["reflection"] = ref
            elif target == "eval_obj":
                self.incident_light_power = self.incident_light_norm_run(
                    resolution=resolution
                )
                eff = self.two_monitor_eff(gratings, resolution)
                ref = self.reflection_layer(resolution, gratings)
                result["eval_obj"] = eff
                result["reflection"] = ref
            else:
                raise ValueError(f"Unsupported target: {target}")
        return result

    def frequency_domain_sim(self, path, sharpness: float = 20, resolution: int = 20):
        self.update_ridges_num()
        grating = self.build_gratings(sharpness=sharpness, resolution=resolution)
        height_mask = self.build_height_mask(sharpness=sharpness, resolution=resolution)
        grating = grating.unsqueeze(0).repeat(self.ridge_height_max * resolution + 1, 1)
        grating = grating * height_mask.unsqueeze(1)
        grating = self.binary_projection(
            grating, beta=torch.tensor(sharpness), eta=torch.tensor(0.5)
        )
        # the resolution must align between the grating and the simulation
        device = metalens_fdtd(
            aperture=self.aperture,
            ridge_height=self.ridge_height_max,
            sub_height=self.sub_height,
            f_min=self.f_min,
            f_max=self.f_max,
            border_width=self.sim_cfg["border_width"],  # um, [x, y]
            eps_r=self.eps_r,  # relative refractive index
            eps_bg=self.eps_bg,  # background refractive index
            PML=self.sim_cfg["PML"],  # um, [x, y]
            focal_constant=self.focal_constant,
        )
        device.update_permittivity(grating)
        device.add_source(
            wl_cen=self.sim_cfg["wl_cen"],
            src_type="ContinuousSource",
        )
        size_x = device.sx + device.f_min + 20
        fdtd_sim = device.create_simulation(
            resolution=resolution,  # pixels / um
            sim_size=(
                size_x,
                device.sy,
                0,
            ),
            displacement=(
                -(device.f_min + 20) / 2,
                0,
                0,
            ),
            until=300,
            if_complex=True,
        )
        device.run_sim(
            sim=fdtd_sim,
            filepath=path,
            export_video=True,
            mode="fdfd",
            tol=float(-8),
        )

    def forward(self, sharpness: float = 1, resolution: int = 20):
        """
        Forward pass of the model

        Args:
            x: torch.Tensor
                input tensor

        Returns:
            output: torch.Tensor
                output tensor
        """
        gratings = self.build_metalense(sharpness, resolution)
        if self.training:
            targets = ["opt_obj"]
        else:
            targets = ["eval_obj"]
        fom = self.evaluate_metalense(gratings, resolution, targets=targets)

        result = dict()
        result.update(fom)
        result.update(
            {
                "grating": gratings,
                "width": self.real_width,
                "aspect_ratio": self.aspect_ratio,
                "period": self.real_period,
                "height": self.ridge_height,
            }
        )

        return result
