"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
"""

import math
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
import torch
from pyutils.config import Config
from pyutils.general import logger

from core.invdes.invdesign import InvDesign
from core.invdes.models import MMIOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MMI
from core.invdes.run_management import OptimizationRunManager
from core.utils import set_torch_deterministic

sys.path.pop(0)


def force_y_branch_edge_endpoints(invdesign):
    """Anchor the optimized y-edge to the input and output waveguides."""
    optimization = invdesign.devOptimization
    for design_region in optimization.design_region_param_dict.values():
        if design_region.cfgs.get("method") != "edgebox":
            continue

        geometry = design_region.weights["edgebox"]
        for side, spline in geometry.edge_modules.items():
            if not side.startswith("y"):
                continue

            side_sign = 1.0 if side.endswith("+") else -1.0
            input_port = optimization.device.port_cfgs["in_port_1"]
            output_sine_bends = [
                cfg
                for name, cfg in optimization.device.geometry_cfgs.items()
                if name.startswith("out_sine_bend_")
            ]
            if not output_sine_bends:
                raise ValueError(
                    "The y-branch must define at least one output sine bend"
                )

            output_sine_bend = max(
                output_sine_bends,
                key=lambda cfg: float(cfg["start"][1]) * side_sign,
            )
            target_boundaries = (
                float(input_port["center"][1])
                + side_sign * float(input_port["size"][1]) / 2.0,
                float(output_sine_bend["start"][1])
                + side_sign * float(output_sine_bend["width"]) / 2.0,
            )
            nominal_boundary = geometry._nominal_edge_base(side)
            target_offsets = tuple(
                boundary - nominal_boundary for boundary in target_boundaries
            )

            if spline.constraint is None:
                raw_targets = target_offsets
            else:
                low, high = spline.constraint
                raw_targets = []
                for offset in target_offsets:
                    if offset < 0.0:
                        if low >= 0.0:
                            raise ValueError(
                                f"Constraint for {side} cannot represent offset {offset}."
                            )
                        raw_targets.append(
                            -math.atanh(max(min(offset / low, 1.0 - 1e-7), 0.0))
                        )
                    else:
                        if high <= 0.0:
                            raise ValueError(
                                f"Constraint for {side} cannot represent offset {offset}."
                            )
                        raw_targets.append(
                            math.atanh(max(min(offset / high, 1.0 - 1e-7), 0.0))
                        )
                raw_targets = tuple(raw_targets)

            with torch.no_grad():
                spline.raw_ctrl.data[0] = raw_targets[0]
                spline.raw_ctrl.data[-1] = raw_targets[1]


def before_y_branch_step(invdesign, feed_dict):
    force_y_branch_edge_endpoints(invdesign)
    return feed_dict


if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    # mmi_region_size = (5, 5)
    # mmi_region_size = (15, 10)
    # mmi_region_size = (10, 10)
    thickness = 0.22
    mmi_region_size = (2, 2, thickness)
    port_len = 1.5
    wl_cen = 1.55
    wl_width = 0.0
    n_wl = 1
    branch_sep = 0.2

    input_port_width = 0.5
    output_port_width = 0.5
    num_inports = 1
    num_outports = 2
    resolution = 50
    pol = "Hz"
    sidewall_angle = 0.0

    material_r1 = "Si"
    exp_name = (
        f"y_branch_3d_{material_r1}_{mmi_region_size[0]}x{mmi_region_size[1]}_new2"
    )
    work_root = "./runs"
    resume_checkpoint = os.environ.get("MAPS_RESUME_CHECKPOINT")

    run_manager = OptimizationRunManager(
        root=work_root,
        experiment="y_branch_3d",
        tag=exp_name,
        resume=resume_checkpoint is not None,
        resume_from=resume_checkpoint,
    )
    run_manager.attach_logger(logger)

    sim_cfg.update(
        dict(
            solver="fdtdx",
            subpixel=True,
            symmetry=(0, -1, 0),  # PEC on y direction.
            max_time=3e-13,
            border_width=[
                0,
                0,
                1.5,
                1.5,
                1.2,
                1.2,
            ],  # left, right, lower, upper, containing PML
            resolution=resolution,
            optical_grid={
                "mode": "fdtdx_rectilinear_nonuniform",
                "wavelength": wl_cen,
                "grid_x": {"type": "auto", "min_steps_per_wvl": 15},
                "grid_y": {"type": "auto", "min_steps_per_wvl": 15},
                "grid_z": {"type": "auto", "min_steps_per_wvl": 15},
            },
            PML=[wl_cen / 2, wl_cen / 2, wl_cen / 2 - 0.2],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
        )
    )
    run_manager.bind_simulation_config(sim_cfg)

    def fom_func(breakdown):
        ## maximization fom
        fom = 0

        # due to symmetry, we only need to evaluate the upper port.
        trans = (
            breakdown[f"fwd_trans_p1"]["value"]
            / (breakdown["input_trans"]["value"].detach())
        ).mean()
        fom = fom + breakdown[f"fwd_trans_p1"]["weight"] * trans
        breakdown[f"fwd_trans_p1"]["value"] = trans.item()

        breakdown[f"fwd_trans_p2"] = {"value": trans.item(), "weight": 1}

        # trans = (
        #     breakdown[f"fwd_trans_p2"]["value"]
        #     / (breakdown["input_trans"]["value"].detach())
        # ).mean()
        # fom = fom + breakdown[f"fwd_trans_p2"]["weight"] * trans
        # breakdown[f"fwd_trans_p2"]["value"] = trans.item()

        for key, obj in breakdown.items():
            if "energy_constraint" in key:
                fom += obj["weight"] * obj["value"]

        return fom, {}

    obj_cfgs = dict(
        fwd_trans_p1=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_2",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=[wl_cen],  #
            temp=[300],
            out_modes=(
                f"{pol}1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction="x+",
            requires_grad=True,
        ),
        # fwd_trans_p2=dict(
        #     weight=1,
        #     #### objective is evaluated at this port
        #     in_slice_name="in_slice_1",
        #     out_slice_name="out_slice_1",
        #     #### objective is evaluated at all points by sweeping the wavelength and modes
        #     in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
        #     wl=[wl_cen],  #
        #     temp=[300],
        #     out_modes=(
        #         f"{pol}1",
        #     ),  # can evaluate on multiple output modes and get average transmission
        #     type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
        #     direction="x+",
        #     requires_grad=True,
        # ),
        input_trans=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="refl_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=[wl_cen],  #
            temp=[300],
            out_modes=(
                f"{pol}1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="flux_minus_src",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction="x",
            requires_grad=False,
        ),
        energy_constraint=dict(
            # weight=-1e4, # suitable with levelset
            weight=-1e3,  # suitable with edgebox
            # weight=0,  # suitable with edgebox
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=[wl_cen],
            temp=[300],
            in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                f"{pol}1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="energy_constraint",
            requires_adjoint=False,
        ),
        _fusion_func=fom_func,
        override=True,
    )

    device = MMI(
        material_ports=(material_r1, material_r1),
        material_buried="SiO2",
        material_cladding="SiO2",
        region_material_r1_layers=(material_r1,),
        region_material_r2_layers=("SiO2",),
        is_design_region=(True,),
        thickness_layers=(thickness,),
        etch_depth_layers=(thickness,),
        gap_layers=(0.0,),
        box_size_layers=(mmi_region_size,),
        sim_cfg=sim_cfg,
        box_size=mmi_region_size,
        port_height=(thickness, thickness),
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        num_inports=num_inports,
        num_outports=num_outports,
        device=operation_device,
        port_box_margin=(
            mmi_region_size[1]
            - output_port_width * num_outports
            - branch_sep * (num_outports - 1)
        )
        / 2,
        rel_port_width=4.5,
        output_sine_bend_length=4,
        output_sine_bend_expand=0.55,
        # LER_A=0.005,  # line edge roughness amplitude
        # LER_Lc=0.05,  # line edge roughness correlation
        sidewall_angle=sidewall_angle,
        LER_A=0.0,  # line edge roughness amplitude
        LER_Lc=0.0,  # line edge roughness correlation
        pol=pol,
        is_3d=True,
    )

    # hr_device should not have roughness
    hr_device = device.copy(resolution=resolution, LER_A=0.0, LER_Lc=0.0)
    print(device)

    design_region_param_cfgs = {}
    for region_name in device.design_region_cfgs.keys():
        #     design_region_param_cfgs[region_name] = dict(
        #         method="levelset",
        #         rho_resolution=[resolution, resolution],
        #         sigma=2 / resolution,
        #         transform=[
        #             dict(type="mirror_symmetry", dims=[1]),
        #             # dict(
        #             #     type="fft",
        #             #     mfs=0.1,
        #             #     # mfs=0.2,
        #             #     resolutions=[hr_device.resolution, hr_device.resolution],
        #             #     dim="xy",
        #             # ),
        #             dict(type="binarize"),
        #         ],
        #         # init_method="random",
        #         init_method="constant_0.1",
        #         denorm_mode="linear_eps",
        #         interpolation="gaussian_linear",
        #         binary_projection=dict(
        #             fw_threshold=100,
        #             bw_threshold=100,
        #             mode="regular",
        #         ),
        #         dims=(0, 1),
        #         reference_plane="middle",
        #         extrude_angle=sidewall_angle,
        #     )

        design_region_param_cfgs[region_name] = dict(
            method="edgebox",
            geometry_cfgs=dict(
                enabled_edges=("y-", "y+"),
                pair_symmetry="mirror",
                box_center=(0.0, 0.0),
                box_center_unit="relative",
                box_size=(1.0, 0.5),
                box_size_unit="relative",  # relative or um
                rho_resolution={"y-": 5, "y+": 5},
                default_rho_resolution=5,
                phi_resolution=50,
                degree=3,
                border_band_um=0.02,
                # constraints={},
                # constraints={"y+": (-0.9, 1.0)},
            ),
            transform=[
                dict(type="mirror_symmetry", dims=[1]),
                # dict(
                #     type="blur",
                #     mfs=0.001,
                #     resolutions=[hr_device.resolution, hr_device.resolution],
                #     dim="xy",
                #     ste=True,  # no blur only in backward grad, no blur in forward
                # ),
                dict(type="binarize"),
                # dict(
                #     type="add_roughness", A=0.012, Lc=0.05, seed=None
                # ),  # None means random for each iteration
            ],
            # init_method="random",
            init_method="zeros",
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

    opt = MMIOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
        design_region_param_cfgs=design_region_param_cfgs,
        obj_cfgs=obj_cfgs,
    ).to(operation_device)
    invdesign = InvDesign(
        devOptimization=opt,
        run_manager=run_manager,
        optimizer=Config(
            name="Adam",
            # lr=1e-2,
            lr=1,
            # lr=0.2,
            # lr=1,
            init_v=1e-7,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=0.2,
        ),
        sharp_scheduler=Config(
            mode="cosine",
            name="sharpness",
            init_sharp=1,  # no blur in forward, thus no need for sharpness
            final_sharp=1,
        ),
        run=Config(
            n_epochs=100,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=1,
            objs=["fwd_trans_p1"],
            plot_name=f"{exp_name}",
            field_keys=[
                (f"in_slice_{i + 1}", wl_cen, f"{pol}1", 300)
                for i in range(num_inports)
            ],
            in_slice_names=[f"in_slice_{i + 1}" for i in range(num_inports)],
            filename_suffixes=[f"s{i + 1}" for i in range(num_inports)],
            field_component="Ey",
            exclude_port_names=[],
            eps_grad=True,
            param_grad=True,
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
            upsample_eps_to_1nm=True,
            load_ckpt=resume_checkpoint,
            resume=resume_checkpoint is not None,
        ),
        after_step_callbacks=[],
        before_step_callbacks=[before_y_branch_step],
    )
    force_y_branch_edge_endpoints(invdesign)
    invdesign.optimize()
