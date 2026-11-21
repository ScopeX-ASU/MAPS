"""Stage-1 digital-circle inverse design for a partially etched silicon Y-branch.

The design region contains a fixed 20 x 20 lattice of circular footprints with
130 nm center-to-center pitch. Stage 1 optimizes only the per-circle material
values; circle-to-radius EMT transfer is intentionally left for a later stage.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

import torch
from pyutils.config import Config
from pyutils.general import logger

from core.invdes.invdesign import InvDesign, clamp_material_parameters
from core.invdes.models import MMIOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MMI
from core.invdes.run_management import OptimizationRunManager
from core.utils import material_fn_dict, set_torch_deterministic

sys.path.pop(0)


def init_circles_045(weight_dict, param_cfg, region_cfg):
    """Initialize a regular circle lattice with 45 nm radius.

    Args:
        weight_dict: Parameter mapping containing the ``"circles"`` geometry module.
        param_cfg: Circle parameterization configuration. ``material_init`` is
            used when material optimization is enabled.
        region_cfg: Physical design-region configuration containing ``center`` and
            ``size`` in micrometers.

    Returns:
        None; circle centers, radii, and optional material values are initialized
        in-place.

    Raises:
        ValueError: If the requested radius is not smaller than the lattice pitch.
    """
    circles = weight_dict["circles"]
    radius = 0.045
    n_circles = circles.centers.shape[:2]
    region_center = region_cfg["center"]
    region_size = region_cfg["size"]
    pitch = (
        float(region_size[0]) / n_circles[0],
        float(region_size[1]) / n_circles[1],
    )
    if radius <= 0.0 or 2.0 * radius > min(pitch):
        raise ValueError(
            f"Initial circle diameter must be in (0, {min(pitch):.6g}] um; "
            f"got diameter {2.0 * radius}"
        )

    x_centers = torch.linspace(
        float(region_center[0]) - float(region_size[0]) / 2 + pitch[0] / 2,
        float(region_center[0]) + float(region_size[0]) / 2 - pitch[0] / 2,
        steps=n_circles[0],
        device=circles.centers.device,
        dtype=circles.centers.dtype,
    )
    y_centers = torch.linspace(
        float(region_center[1]) - float(region_size[1]) / 2 + pitch[1] / 2,
        float(region_center[1]) + float(region_size[1]) / 2 - pitch[1] / 2,
        steps=n_circles[1],
        device=circles.centers.device,
        dtype=circles.centers.dtype,
    )

    with torch.no_grad():
        xx, yy = torch.meshgrid(x_centers, y_centers, indexing="ij")
        circles.centers[..., 0].copy_(xx)
        circles.centers[..., 1].copy_(yy)
        circles.radius.fill_(radius)
        if circles.material_rho is not None:
            material_init = float(
                param_cfg.get(
                    "material_init",
                    circles.geometry_cfgs.get("material_init", 0.5),
                )
            )
            circles.material_rho.fill_(material_init)


def build_stage1_y_branch():
    """Build the Stage-1 digital-circle Y-branch optimization.

    Returns:
        InvDesign: Configured optimization object. Calling ``optimize()`` runs
            material-only optimization with 45 nm initial circle radii.
    """
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device(f"cuda:{gpu_id}")
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(541)

    wavelength = 1.55
    silicon_thickness = 0.22
    etch_depth = 0.14
    residual_silicon_thickness = silicon_thickness - etch_depth
    design_size = 2.6
    pixel_pitch = 0.13
    pixel_count = 20
    if design_size != pixel_pitch * pixel_count:
        raise ValueError("design_size must equal pixel_pitch * pixel_count")

    resolution = 100
    port_length = 1.5
    port_width = 0.5
    polarization = "Hz"
    sidewall_angle = 0.0
    experiment_name = "digital_y_branch_3d_si_partial_etch_circle_stage1_hz"
    work_root = "./runs"
    resume_checkpoint = os.environ.get("MAPS_RESUME_CHECKPOINT")

    run_manager = OptimizationRunManager(
        root=work_root,
        experiment="digital_y_branch_circle_3d",
        tag=experiment_name,
        resume=resume_checkpoint is not None,
        resume_from=resume_checkpoint,
    )
    run_manager.attach_logger(logger)

    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="fdtdx",
            subpixel=True,
            symmetry=(0, -1, 0),
            max_time=3e-13,
            border_width=[0, 0, 1.5, 1.5, 1.2, 1.2],
            resolution=resolution,
            optical_grid={
                "mode": "fdtdx_rectilinear_nonuniform",
                "wavelength": wavelength,
                "grid_x": {"type": "auto", "min_steps_per_wvl": 25},
                "grid_y": {"type": "auto", "min_steps_per_wvl": 25},
                "grid_z": {"type": "auto", "min_steps_per_wvl": 15},
            },
            PML=[wavelength / 2, wavelength / 2, wavelength / 2 - 0.2],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wavelength,
            wl_width=0.0,
            n_wl=1,
        )
    )
    run_manager.bind_simulation_config(sim_cfg)

    eps_full_silicon = material_fn_dict["Si_eff"](
        wavelength, thickness=silicon_thickness
    )
    eps_oxide = material_fn_dict["SiO2"](wavelength)
    residual_silicon_fraction = residual_silicon_thickness / silicon_thickness
    eps_partial_etch = (
        residual_silicon_fraction * eps_full_silicon
        + (1.0 - residual_silicon_fraction) * eps_oxide
    )

    device = MMI(
        material_ports=("Si", "Si"),
        material_buried="SiO2",
        material_cladding="SiO2",
        region_material_r1_layers=("Si",),
        region_material_r2_layers=(eps_partial_etch,),
        is_design_region=(True,),
        thickness_layers=(silicon_thickness,),
        etch_depth_layers=(silicon_thickness,),
        gap_layers=(0.0,),
        box_size_layers=((design_size, design_size, silicon_thickness),),
        sim_cfg=sim_cfg,
        box_size=(design_size, design_size, silicon_thickness),
        port_height=(silicon_thickness, silicon_thickness),
        port_len=(port_length, port_length),
        port_width=(port_width, port_width),
        num_inports=1,
        num_outports=2,
        port_box_margin=0.5,
        rel_port_width=3.3,
        output_sine_bend_length=0.0,
        output_sine_bend_expand=0.0,
        sidewall_angle=sidewall_angle,
        pol=polarization,
        is_3d=True,
        device=operation_device,
    )
    hr_device = device.copy(resolution=resolution, LER_A=0.0, LER_Lc=0.0)

    def fom_func(breakdown):
        """Maximize normalized transmission into the upper output branch."""
        input_power = breakdown["input_trans"]["value"].detach()
        transmission = breakdown["fwd_trans_p1"]["value"] / input_power
        fom = breakdown["fwd_trans_p1"]["weight"] * transmission.mean()
        breakdown["fwd_trans_p1"]["value"] = transmission.mean().item()
        breakdown["fwd_trans_p2"] = {
            "value": transmission.mean().item(),
            "weight": 1,
        }

        # transmission = breakdown["fwd_trans_p2"]["value"] / input_power
        # fom += breakdown["fwd_trans_p2"]["weight"] * transmission.mean()
        # breakdown["fwd_trans_p2"]["value"] = transmission.mean().item()

        for key, objective in breakdown.items():
            if "energy_constraint" in key:
                fom = fom + objective["weight"] * objective["value"]
        return fom, {}

    obj_cfgs = dict(
        fwd_trans_p1=dict(
            weight=1,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_2",
            in_mode=f"{polarization}1",
            wl=[wavelength],
            temp=[300],
            out_modes=(f"{polarization}1",),
            type="eigenmode",
            direction="x+",
            requires_grad=True,
            requires_adjoint=True,
        ),
        input_trans=dict(
            weight=1,
            in_slice_name="in_slice_1",
            out_slice_name="refl_slice_1",
            in_mode=f"{polarization}1",
            wl=[wavelength],
            temp=[300],
            out_modes=(f"{polarization}1",),
            type="eigenmode",
            direction="x+",
            requires_grad=False,
        ),
        energy_constraint=dict(
            weight=-0,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            wl=[wavelength],
            temp=[300],
            in_mode=f"{polarization}1",
            out_modes=(f"{polarization}1",),
            type="energy_constraint",
            requires_adjoint=False,
        ),
        _fusion_func=fom_func,
        override=True,
    )

    design_region_param_cfgs = {
        region_name: dict(
            method="circle",
            geometry_cfgs=dict(
                batch_dims=(pixel_count, pixel_count),
                size_dim=1,
                material_optimization=True,
                material_init=0.5,
                phi_resolution=(100, 100),
                border=0.0,
                material_background_value=1.0,
                gpu_subpixel=True,
                gpu_circle_sides=128,
            ),
            transform=[dict(type="mirror_symmetry", dims=[1])],
            init_method=init_circles_045,
            denorm_mode="linear_eps",
            dims=(0, 1),
            reference_plane="middle",
            extrude_angle=sidewall_angle,
        )
        for region_name in device.design_region_cfgs
    }

    optimization = MMIOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
        design_region_param_cfgs=design_region_param_cfgs,
        obj_cfgs=obj_cfgs,
    ).to(operation_device)

    for design_region in optimization.design_region_param_dict.values():
        if design_region.cfgs.get("method") != "circle":
            raise ValueError(
                "Stage-1 digital Y-branch requires circle parameterization"
            )
        design_region.set_optimization_stage("material")

    return InvDesign(
        devOptimization=optimization,
        run_manager=run_manager,
        optimizer=Config(name="Adam", lr=0.2, weight_decay=0),
        lr_scheduler=Config(name="cosine", lr_min=0.1),
        sharp_scheduler=Config(
            mode="cosine", name="sharpness", init_sharp=1, final_sharp=1
        ),
        run=Config(n_epochs=100),
        plot_cfgs=Config(
            plot=True,
            interval=5,
            objs=["fwd_trans_p1"],
            plot_name=experiment_name,
            field_keys=[("in_slice_1", wavelength, f"{polarization}1", 300)],
            in_slice_names=["in_slice_1"],
            filename_suffixes=["s1"],
            field_component=polarization,
            exclude_port_names=[],
            eps_grad=True,
            param_grad=True,
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=experiment_name,
            dump_gds=True,
            gds_name=experiment_name,
            upsample_eps_to_1nm=True,
            load_ckpt=resume_checkpoint,
            resume=resume_checkpoint is not None,
        ),
        after_step_callbacks=[clamp_material_parameters],
    )


if __name__ == "__main__":
    build_stage1_y_branch().optimize()
