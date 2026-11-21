"""Stage-1 digital-box inverse design for a partially etched silicon Y-branch.

The design region is represented by a fixed 20 x 20 array of 130 nm-pitch
boxes. This example intentionally stops after material optimization; EMT
material-to-shape transfer is a separate stage and is not invoked here.
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


def init_small_boxes(weight_dict, param_cfg, region_cfg):
    """Initialize a centered digital box lattice with smaller boxes.

    Args:
        weight_dict: Parameter mapping containing the ``"boxes"`` geometry module.
        param_cfg: Digital-box parameterization configuration. The optional
            ``material_init`` value initializes ``material_rho`` when present.
        region_cfg: Physical design-region configuration containing ``center`` and
            ``size`` in micrometers.

    Returns:
        None; box centers, sizes, and optional material values are initialized in-place.

    Notes:
        Change ``box_size`` below to adjust the initial box width. The current
        value is 80 nm, smaller than the 130 nm lattice pitch.
    """
    boxes = weight_dict["boxes"]
    box_size = 0.08
    n_boxes = boxes.centers.shape[:2]
    region_center = region_cfg["center"]
    region_size = region_cfg["size"]
    pitch = (
        float(region_size[0]) / n_boxes[0],
        float(region_size[1]) / n_boxes[1],
    )
    if box_size <= 0.0 or box_size > min(pitch):
        raise ValueError(
            f"Initial box size must be in (0, {min(pitch):.6g}] um; got {box_size}"
        )

    x_centers = torch.linspace(
        float(region_center[0]) - float(region_size[0]) / 2 + pitch[0] / 2,
        float(region_center[0]) + float(region_size[0]) / 2 - pitch[0] / 2,
        steps=n_boxes[0],
        device=boxes.centers.device,
        dtype=boxes.centers.dtype,
    )
    y_centers = torch.linspace(
        float(region_center[1]) - float(region_size[1]) / 2 + pitch[1] / 2,
        float(region_center[1]) + float(region_size[1]) / 2 - pitch[1] / 2,
        steps=n_boxes[1],
        device=boxes.centers.device,
        dtype=boxes.centers.dtype,
    )

    with torch.no_grad():
        xx, yy = torch.meshgrid(x_centers, y_centers, indexing="ij")
        boxes.centers[..., 0].copy_(xx)
        boxes.centers[..., 1].copy_(yy)
        boxes.sizes.fill_(box_size)
        if boxes.material_rho is not None:
            material_init = float(
                param_cfg.get(
                    "material_init",
                    boxes.geometry_cfgs.get("material_init", 0.5),
                )
            )
            boxes.material_rho.fill_(material_init)


def build_stage1_y_branch():
    """Build the Stage-1 digital-box Y-branch optimization.

    Returns:
        InvDesign: Configured optimization object. The caller can invoke
            ``optimize()`` to run material-only optimization.
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
    branch_separation = 0.2
    num_inports = 1
    num_outports = 2
    polarization = "Hz"
    sidewall_angle = 0.0
    experiment_name = "digital_y_branch_3d_si_partial_etch_stage1_hz"
    work_root = "./runs"
    resume_checkpoint = os.environ.get("MAPS_RESUME_CHECKPOINT")

    run_manager = OptimizationRunManager(
        root=work_root,
        experiment="digital_y_branch_3d",
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
                "grid_x": {"type": "auto", "min_steps_per_wvl": 15},
                "grid_y": {"type": "auto", "min_steps_per_wvl": 15},
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

    # MMI is a scalar-material 2.5-D template. Represent the residual 80 nm
    # silicon after the 140 nm etch by a vertical effective-permittivity mix;
    # Si_eff_eps intentionally rejects thicknesses below 150 nm, so the 80 nm
    # endpoint cannot be passed to that helper directly.
    eps_full_silicon = material_fn_dict["Si_eff"](
        wavelength, thickness=silicon_thickness
    )
    eps_oxide = material_fn_dict["SiO2"](wavelength)
    residual_silicon_fraction = residual_silicon_thickness / silicon_thickness
    eps_partial_etch = (
        residual_silicon_fraction * eps_full_silicon
        + (1.0 - residual_silicon_fraction) * eps_oxide
    )

    # The MMI template supplies full-etch ports and a design-region endpoint
    # pair. The layer r2 material is the 80 nm residual-Si effective material
    # left beneath a 140 nm partial etch.
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
        num_inports=num_inports,
        num_outports=num_outports,
        port_box_margin=0.5,
        rel_port_width=3.3,
        output_sine_bend_length=0.0,
        output_sine_bend_expand=0.0,
        sidewall_angle=sidewall_angle,
        pol=polarization,
        is_3d=True,
        device=operation_device,
    )
    hr_device = device.copy(
        resolution=resolution,
        LER_A=0.0,
        LER_Lc=0.0,
    )

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
            requires_adjoint=False,
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
            method="box",
            geometry_cfgs=dict(
                batch_dims=(pixel_count, pixel_count),
                size_dim=2,
                material_optimization=True,
                material_init=0.5,
                # Geometry is rasterized on this uniform internal grid before
                # being sampled onto the solver grid.
                phi_resolution=(100, 100),
                # The box lattice fills the 2.6 um region at 130 nm pitch.
                border=0.0,
                # Full silicon is the fixed host; boxes represent partially
                # etched cells through their independently optimized rho.
                material_background_value=1.0,
                # Use the CUDA area/volume-average rasterizer for the forward
                # density map. The custom box backward remains the optimizer
                # gradient approximation.
                gpu_subpixel=True,
            ),
            transform=[
                dict(type="mirror_symmetry", dims=[1]),
            ],
            init_method=init_small_boxes,
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

    # Stage 1: only material_rho is trainable. No EMT projection or shape
    # transfer is performed in this example.
    for design_region in optimization.design_region_param_dict.values():
        if design_region.cfgs.get("method") != "box":
            raise ValueError("Stage-1 digital Y-branch requires box parameterization")
        design_region.set_optimization_stage("material")

    return InvDesign(
        devOptimization=optimization,
        run_manager=run_manager,
        optimizer=Config(name="Adam", lr=0.1, weight_decay=0),
        lr_scheduler=Config(name="cosine", lr_min=0.1),
        sharp_scheduler=Config(
            mode="cosine",
            name="sharpness",
            init_sharp=1,
            final_sharp=1,
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
