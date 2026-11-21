"""Optimize an image-forming dielectric metasurface with MAPS FDTD-X.

This example follows Tidy3D's ``Autograd13Metasurface`` physical problem: a
normally illuminated permittivity-2 phase mask forms a requested intensity
image 6.1 wavelengths above the device. MAPS uses FDTD-X's native uniform TFSF
plane source and Gaussian design interpolation plus projection instead of
Tidy3D's conic density filter.
The FDTD domain ends shortly above a tangential-field monitor placed 300 nm
above the mask. A differentiable nonparaxial angular-spectrum transform then
propagates that periodic x/y supercell to the virtual image plane.

Set ``MAPS_METASURFACE_SMOKE=1`` for a one-step GPU check, or
``MAPS_METASURFACE_FULL_SCALE=1`` for the reference 20-wavelength aperture and
35 optimization steps. Outputs are managed under ``runs/metasurface_3d``.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pyutils.config import Config
from pyutils.general import logger

from core.invdes.invdesign import InvDesign, clamp_parameters
from core.invdes.models import DiffractiveMetasurfaceOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import DiffractiveMetasurface
from core.invdes.run_management import OptimizationRunManager
from core.utils import set_torch_deterministic

sys.path.pop(0)


def make_maps_target(size: int = 192) -> np.ndarray:
    """Create a self-contained grayscale MAPS wordmark target in [0, 1]."""
    image = Image.new("L", (size, size), 0)
    # PIL's built-in bitmap font avoids a repository or machine-local asset.
    small = Image.new("L", (32, 16), 0)
    ImageDraw.Draw(small).text((1, 2), "MAPS", fill=255, font=ImageFont.load_default())
    small = small.resize((size - 24, size // 3), Image.Resampling.NEAREST)
    image.paste(small, ((size - small.width) // 2, (size - small.height) // 2))
    image = image.filter(ImageFilter.GaussianBlur(radius=max(1, size / 96)))
    target = np.asarray(image, dtype=np.float32) / 255.0
    # MAPS field planes use x/y axis order; image arrays use row/column (y/x).
    return np.flip(target.T, axis=1).copy()


def save_metasurface_diagnostics(target, nearfield, measured, density, scores, path):
    """Save near/projected fields, image error, mask, and score history."""
    target = np.asarray(target)
    measured = np.asarray(measured)
    density = np.asarray(density)
    nearfield = np.asarray(nearfield)
    fig, axes = plt.subplots(1, 6, figsize=(18, 3), constrained_layout=True)
    for axis, data, title in zip(
        axes,
        (target, nearfield, measured, np.abs(target - measured), density),
        (
            "target",
            "near intensity (z=0.3 um)",
            "projected intensity",
            "absolute error",
            "design density",
        ),
    ):
        axis.imshow(data.T, origin="lower", cmap="magma", vmin=0)
        axis.set_title(title)
        axis.set_axis_off()
    axes[-1].plot(np.arange(len(scores)), scores, marker="o", markersize=2)
    axes[-1].set_xlabel("iteration")
    axes[-1].set_ylabel("image score")
    axes[-1].set_title("optimization")
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device(f"cuda:{gpu_id}")
    set_torch_deterministic(541)

    full_scale = os.environ.get("MAPS_METASURFACE_FULL_SCALE") == "1"
    smoke = os.environ.get("MAPS_METASURFACE_SMOKE") == "1"
    wl_cen = 1.0 if full_scale else 1.55
    n_wl = 1
    wl_width = 0.0
    eps_design = 2.0
    aperture = 20.0 if full_scale else (4.0 if smoke else 8.0)
    source_distance = 0.5
    image_distance = 6.1 * wl_cen
    nearfield_distance = 0.3
    thickness = 4 / (2 * np.pi / wl_cen * (np.sqrt(eps_design) - 1))
    substrate_thickness = 2.0
    in_mode = "Hz2"
    resolution = 20 if full_scale else (8 if smoke else 18)
    grid_steps = 15 if full_scale else (8 if smoke else 15)
    epochs = 35 if full_scale else (1 if smoke else 20)
    learning_rate = 0.1 if full_scale else 0.15
    sidewall_angle = 0.0
    exp_name = "metasurface_3d_full" if full_scale else "metasurface_3d_demo"
    resume_checkpoint = os.environ.get("MAPS_RESUME_CHECKPOINT")

    run_manager = OptimizationRunManager(
        root="./runs",
        experiment="metasurface_3d",
        tag=exp_name,
        resume=resume_checkpoint is not None,
        resume_from=resume_checkpoint,
    )
    run_manager.attach_logger(logger)

    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="fdtdx",
            resolution=resolution,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
            max_time=1e-13 if smoke else 5e-13,
            border_width=[0, 0, 0, 0, 0, 0.8 * wl_cen + 0.3],
            # Zero transverse PML selects periodic x/y boundaries in FDTD-X.
            # This matches the reference and the FFT periodic-supercell model.
            PML=[0.0, 0.0, 0.8 * wl_cen],
            optical_grid={
                "mode": "fdtdx_rectilinear_nonuniform",
                "wavelength": wl_cen,
                "grid_x": {"type": "auto", "min_steps_per_wvl": grid_steps},
                "grid_y": {"type": "auto", "min_steps_per_wvl": grid_steps},
                "grid_z": {"type": "auto", "min_steps_per_wvl": grid_steps},
            },
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )
    run_manager.bind_simulation_config(sim_cfg)
    device = DiffractiveMetasurface(
        sim_cfg=sim_cfg,
        aperture=aperture,
        thickness=thickness,
        substrate_thickness=substrate_thickness,
        source_distance=source_distance,
        image_distance=image_distance,
        nearfield_distance=nearfield_distance,
        eps_design=eps_design,
        in_mode=in_mode,
        device=operation_device,
    )
    hr_device = device.copy(resolution=resolution)
    target = make_maps_target()

    wls = np.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl)
    obj_cfgs = dict(
        image_match=dict(
            weight=1.0,
            type="target_intensity_angular_spectrum",
            in_slice_name="in_slice_1",
            out_slice_name="nearfield_projection",
            in_mode=in_mode,
            direction="z+",
            wl=wls,
            temp=[300],
            target_intensity=target,
            field_components=("Ex", "Ey", "Ez"),
            projection_distance_um=image_distance - nearfield_distance,
            evanescent="decay",
        ),
    )

    design_region_param_cfgs = dict()
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[50, 50],
            sigma=1 / 50,
            transform=[
                # dict(
                #     type="blur",
                #     mfs=0.05,
                #     resolutions=[hr_device.resolution, hr_device.resolution],
                #     dim="xy",
                # ),
                dict(type="binarize"),
            ],
            init_method="constant_0.0",
            denorm_mode="linear_eps",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
            dims=(0, 1),
            reference_plane="middle",
            extrude_angle=sidewall_angle,
        )

    opt = DiffractiveMetasurfaceOptimization(
        device=device,
        hr_device=hr_device,
        design_region_param_cfgs=design_region_param_cfgs,
        obj_cfgs=obj_cfgs,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    score_history = []

    def record_diagnostics(current_invdesign):
        """Record the scalar score and periodic image diagnostics."""
        objective = opt.objective.Js["image_match"]["fn"]
        score = current_invdesign.closure.results["breakdown"]["image_match"]["value"]
        score_history.append(float(score.detach().cpu()))
        interval = 1
        if current_invdesign.global_step % interval != 0:
            return
        measured = objective.last_intensity
        nearfield = objective.last_nearfield_intensity
        target_grid = objective.last_target
        eps = opt._hr_eps_map.detach().cpu().numpy().real
        density = (eps - 1.0) / (eps_design - 1.0)
        density = density[:, :, density.shape[2] // 2]
        save_metasurface_diagnostics(
            target_grid.cpu().numpy(),
            nearfield.cpu().numpy(),
            measured.cpu().numpy(),
            density,
            score_history,
            run_manager.path(
                "plots", f"image_match_step_{current_invdesign.global_step:04d}.png"
            ),
        )

    invdesign = InvDesign(
        devOptimization=opt,
        run_manager=run_manager,
        run=Config(n_epochs=epochs),
        optimizer=Config(name="Adam", lr=learning_rate, weight_decay=0),
        lr_scheduler=Config(name="cosine", lr_min=learning_rate * 0.2),
        sharp_scheduler=Config(
            mode="cosine", name="sharpness", init_sharp=5, final_sharp=32
        ),
        plot_cfgs=Config(
            plot=True,
            interval=1,
            plot_name=f"{exp_name}",
            objs=["image_match"],
            field_keys=[("in_slice_1", wl_cen, in_mode, 300)],
            in_slice_names=["in_slice_1"],
            exclude_port_names=[],
            field_component="Ey",
            eps_grad=True,
            param_grad=True,
            monitor_plots=[
                dict(
                    objective="image_match",
                    fields=("nearfield", "farfield"),
                    components=("E", "Ey", "Ez"),
                    quantities=("intensity", "phase"),
                    phase_reference="Ey",
                )
            ],
        ),
        checkpoint_cfgs=Config(
            save_model=True,
            ckpt_name=exp_name,
            interval=max(1, epochs),
            save_latest=True,
            load_ckpt=resume_checkpoint,
            resume=resume_checkpoint is not None,
            dump_tidy3d=False,
            dump_gds=False,
        ),
        after_step_callbacks=[clamp_parameters, record_diagnostics],
    )
    invdesign.optimize()

    objective = opt.objective.Js["image_match"]["fn"]
    measured = objective.last_intensity
    nearfield = objective.last_nearfield_intensity
    eps = opt._hr_eps_map.detach().cpu().numpy()
    density = (eps.real - 1.0) / (eps_design - 1.0)
    density = density[:, :, density.shape[2] // 2]
    target_grid = objective.last_target
    save_metasurface_diagnostics(
        target_grid.cpu().numpy() if target_grid is not None else target,
        nearfield.cpu().numpy() if nearfield is not None else np.zeros_like(target),
        measured.cpu().numpy() if measured is not None else np.zeros_like(target),
        density,
        score_history,
        run_manager.path("plots", "image_match.png"),
    )
