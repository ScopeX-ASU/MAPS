"""Stage-1 digital-circle inverse design for a partially etched silicon Y-branch.

The design region contains a fixed 20 x 20 lattice of circular footprints with
130 nm center-to-center pitch. Stage 1 optimizes only the per-circle material
values. The ``prepare_stage2_y_branch`` helper performs the EMT
material-to-radius transfer and enables radius optimization for the next stage.
"""

import os
import sys
from functools import partial

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


def init_circles(weight_dict, param_cfg, region_cfg, init_radius=0.045):
    """Initialize a regular circle lattice with 45 nm radius.

    Args:
        weight_dict: Parameter mapping containing the ``"circles"`` geometry module.
        param_cfg: Circle parameterization configuration. ``material_init`` is
            used when material optimization is enabled. The ``init_radius`` argument
            specifies the initial circle radius.
        region_cfg: Physical design-region configuration containing ``center`` and
            ``size`` in micrometers.

    Returns:
        None; circle centers, radii, and optional material values are initialized
        in-place.

    Raises:
        ValueError: If the requested radius is not smaller than the lattice pitch.
    """
    circles = weight_dict["circles"]
    radius = init_radius
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


def clamp_circle_radii(invdes):
    """Clamp circle radii to nonnegative, non-overlapping lattice values.

    Args:
        invdes: ``InvDesign`` instance containing ``Circles`` modules.

    Returns:
        None; radii are modified in-place under ``torch.no_grad()``.
    """
    with torch.no_grad():
        for module in invdes.devOptimization.modules():
            radius = getattr(module, "radius", None)
            max_radius_fn = getattr(module, "_max_radius", None)
            if isinstance(radius, torch.Tensor) and callable(max_radius_fn):
                radius.clamp_(min=0.0, max=float(max_radius_fn()))


def prepare_stage2_y_branch(invdes, lr=0.003, n_epochs=15):
    """Convert Stage-1 circle materials into binary etched radii.

    Args:
        invdes: Stage-1 ``InvDesign`` instance. Its current circle material
            values are transferred in-place before shape optimization begins.

    Returns:
        InvDesign: The same object, configured for radius-only Stage-2
        optimization with fixed circle centers.

    Notes:
        The design-region background is full silicon and each circle is the
        lower-index partially etched material. Therefore the inclusion fill
        fraction is ``1 - material_rho``; this polarity is passed explicitly to
        the effective-medium projector.
    """
    wavelength = 1.55
    silicon_thickness = 0.22
    etch_depth = 0.22
    residual_silicon_thickness = silicon_thickness - etch_depth
    eps_full_silicon = material_fn_dict["Si_eff"](
        wavelength, thickness=silicon_thickness
    )
    eps_oxide = material_fn_dict["SiO2"](wavelength)
    residual_fraction = residual_silicon_thickness / silicon_thickness
    eps_partial_etch = (
        residual_fraction * eps_full_silicon + (1.0 - residual_fraction) * eps_oxide
    )

    for design_region in invdes.devOptimization.design_region_param_dict.values():
        if design_region.cfgs.get("method") != "circle":
            raise ValueError("Stage 2 requires circle parameterization")
        design_region.transfer_material_to_shape(
            eps_host=eps_full_silicon,
            eps_inclusion=eps_partial_etch,
            eps_low=eps_partial_etch,
            eps_high=eps_full_silicon,
            beta=1,
        )
        design_region.set_optimization_stage("shape")

    experiment_name = invdes._cfg.plot_cfgs.plot_name
    stage2_name = (
        experiment_name.replace("stage1", "stage2")
        if "stage1" in experiment_name
        else f"{experiment_name}_stage2"
    )
    polarization = invdes._cfg.plot_cfgs.field_component
    stage2_run_manager = OptimizationRunManager(
        root=invdes.run_manager.root if invdes.run_manager is not None else "./runs",
        experiment=(
            invdes.run_manager.experiment
            if invdes.run_manager is not None
            else "digital_y_branch_circle"
        ),
        tag=stage2_name,
    )
    stage2_run_manager.attach_logger(logger)
    stage2_run_manager.bind_simulation_config(invdes.devOptimization.sim_cfg)
    invdes = InvDesign(
        devOptimization=invdes.devOptimization,
        run_manager=stage2_run_manager,
        optimizer=Config(name="Adam", lr=lr, weight_decay=0),
        lr_scheduler=Config(name="cosine", lr_min=lr),
        sharp_scheduler=Config(
            mode="cosine", name="sharpness", init_sharp=1, final_sharp=1
        ),
        run=Config(start_epoch=invdes.run.n_epochs, n_epochs=n_epochs),
        plot_cfgs=Config(
            plot=True,
            interval=1,
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
        ),
        after_step_callbacks=[clamp_circle_radii],
    )
    return invdes


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
    etch_depth = 0.22
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
    experiment_name = "digital_y_branch_si_partial_etch_circle_stage1_hz"
    work_root = "./runs"
    resume_checkpoint = os.environ.get("MAPS_RESUME_CHECKPOINT")

    run_manager = OptimizationRunManager(
        root=work_root,
        experiment="digital_y_branch_circle",
        tag=experiment_name,
        resume=resume_checkpoint is not None,
        resume_from=resume_checkpoint,
    )
    run_manager.attach_logger(logger)

    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            subpixel=True,
            symmetry=(0, -1, 0),
            max_time=3e-13,
            border_width=[0, 0, 1.5, 1.5],
            resolution=resolution,
            PML=[wavelength / 2, wavelength / 2],
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
        material_ports=("Si_eff", "Si_eff"),
        material_cladding="SiO2",
        region_material_r1_layers=("Si_eff",),
        region_material_r2_layers=(eps_partial_etch,),
        is_design_region=(True,),
        thickness_r1=silicon_thickness,
        thickness_r2=residual_silicon_thickness,
        sim_cfg=sim_cfg,
        box_size=(design_size, design_size),
        port_len=(port_length, port_length),
        port_width=(port_width, port_width),
        num_inports=1,
        num_outports=2,
        port_box_margin=0.5,
        rel_port_width=3.3,
        output_sine_bend_length=0.0,
        output_sine_bend_expand=0.0,
        pol=polarization,
        is_3d=False,
        device=operation_device,
    )
    hr_device = device.copy(resolution=resolution, LER_A=0.0, LER_Lc=0.0)

    def fom_func(breakdown):
        """Maximize normalized transmission into both output branches."""
        input_power = breakdown["input_trans"]["value"].detach()
        fom = 0.0
        for key in ("fwd_trans_p1", "fwd_trans_p2"):
            transmission = breakdown[key]["value"] / input_power
            fom = fom + breakdown[key]["weight"] * transmission.mean()
            breakdown[key]["value"] = transmission.mean().item()
        for key, objective in breakdown.items():
            if "energy_constraint" in key:
                fom = fom + objective["weight"] * objective["value"]
        return fom, {}

    obj_cfgs = dict(
        fwd_trans_p1=dict(
            weight=1,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            in_mode=f"{polarization}1",
            wl=[wavelength],
            temp=[300],
            out_modes=(f"{polarization}1",),
            type="eigenmode",
            direction="x+",
            requires_grad=True,
        ),
        fwd_trans_p2=dict(
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
                # EMT projection can leave numerically tiny circles. Remove them
                # before constructing Tidy3D PolySlabs instead of allowing a
                # nearly collapsed polygon to reach Pydantic validation.
                min_radius_um=0.03,  # 30nm due to fabrication limitation
                max_radius_um=0.045,
                small_circle_policy="snap_by_area",  # smaller than half smallest area goes to 0, otherwise clamp to min_area
            ),
            transform=[dict(type="mirror_symmetry", dims=[1])],
            init_method=partial(init_circles, init_radius=0.045),
            denorm_mode="linear_eps",
            dims=(0, 1),
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
        optimizer=Config(name="Adam", lr=0.1, weight_decay=0),
        lr_scheduler=Config(name="cosine", lr_min=0.1),
        sharp_scheduler=Config(
            mode="cosine", name="sharpness", init_sharp=1, final_sharp=1
        ),
        run=Config(start_epoch=0, n_epochs=15),
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
    print("Running Stage-1 digital-circle permittivity optimization...")
    invdes = build_stage1_y_branch()
    invdes.optimize()
    print("\nRunning Stage-2 radius optimization...")
    invdes = prepare_stage2_y_branch(invdes, lr=0.003, n_epochs=15)
    invdes.optimize()
