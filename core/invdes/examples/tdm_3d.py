"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
import matplotlib.pyplot as plt
import torch
from pyutils.config import Config

from core.invdes.invdesign import InvDesign
from core.invdes.models import TDMOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import TDM
from core.invdes.models.layers.thermal_control import canonicalize_currents
from core.utils import set_torch_deterministic

sys.path.pop(0)


def plot_device_maps(
    device,
    plot_root,
    map_names=("epsilon", "conductivity", "thermo_optic_coeff"),
    filename="device_maps.jpg",
    cmap="Greys",
    value_mode="real",
    dpi=300,
    overlay_monitors=True,
    overlay_pml=True,
    overlay_heat_sources=True,
):
    """
    Plot selected device maps in rows.

    Rows:
        each map in `map_names`

    Columns:
        0: z = 0
        1: y = 0
        2: x = 0
        3: z = device.z_heater

    Parameters
    ----------
    device:
        Device object with plot_property(...) method.
    plot_root:
        Directory to save the figure.
    map_names:
        List/tuple of property names to plot, e.g.
        ["epsilon_map", "conductivity_map", "thermo_optic_coeff_map"].
    filename:
        Output image filename.
    """

    os.makedirs(plot_root, exist_ok=True)

    map_names = list(map_names)
    n_rows = len(map_names)
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 6 * n_rows),
        squeeze=False,
    )

    slices = [
        dict(z=0, title_suffix="z=0"),
        dict(y=0, title_suffix="y=0"),
        dict(x=0, title_suffix="x=0"),
        dict(z=device.z_heater, title_suffix=f"heater z={device.z_heater:.3f}"),
    ]

    for row, map_name in enumerate(map_names):
        for col, slice_info in enumerate(slices):
            ax = axes[row, col]

            title_suffix = slice_info.pop("title_suffix")
            slice_kwargs = dict(slice_info)

            device.plot_property(
                map_name,
                cmap=cmap,
                value_mode=value_mode,
                ax=ax,
                overlay_monitors=overlay_monitors,
                overlay_pml=overlay_pml,
                overlay_heat_sources=overlay_heat_sources,
                title=f"{map_name} slice at {title_suffix}",
                **slice_kwargs,
            )

            # Restore title_suffix because pop mutates the dict.
            slice_info["title_suffix"] = title_suffix

    plt.tight_layout()

    save_path = os.path.join(plot_root, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return save_path


def generate_tdm_3d(gpu_id, mfs):
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    sim_cfg = DefaultSimulationConfig()

    thickness = 0.22
    region_size = (8.0, 3.0, thickness)
    port_len = 1.5

    input_port_width = 0.5
    output_port_width = 0.5
    heater_gap = 0.8
    heater_thickness = 0.14
    heater_length = 0.5 * region_size[0]
    heater_width = 0.5 * region_size[1]
    heater_on_current = 2e-2
    mode = "Hz1"
    wl_cen = 1.55
    resolution = 40
    exp_name = f"tdm_3d_tin_top_{region_size[0]}x{region_size[1]}"

    # jax (reduced_free_dof): forward 6s, adjoint 4s
    # pydiso (reduced_free_dof): forward 154s, adjoint 3.63s
    sim_cfg.update(
        dict(
            solver="fdtdx",
            border_width=[0, 0, 1.5, 1.5, 0.9, 1.8],
            resolution=resolution,
            max_time=2e-13,
            optical_grid={
                "mode": "fdtdx_rectilinear_nonuniform",
                "wavelength": wl_cen,
                "grid_x": {"type": "auto", "min_steps_per_wvl": 15},
                "grid_y": {"type": "auto", "min_steps_per_wvl": 15},
                "grid_z": {"type": "auto", "min_steps_per_wvl": 15},
            },
            plot_root=f"./figs/{exp_name}",
            PML=[0.4, 0.4, 0.4],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=0,
            n_wl=1,
            heat={
                "backend": "jax",
                # "backend": "pydiso",
                "grid_spec": {
                    "wavelength": 1.55,
                    "grid_x": {"type": "auto", "min_steps_per_wvl": 12},
                    "grid_y": {"type": "auto", "min_steps_per_wvl": 12},
                    "grid_z": {"type": "auto", "min_steps_per_wvl": 20},
                },
                "padding": {
                    # "distance": [4.0, 4.0, 4.0, 4.0, 0.6, 0.0],
                    "distance": [2.0, 2.0, 3.0, 1.0, 0.6, 0.0],
                    # "distance": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "mode": "replicate",
                    "source_mode": "constant",
                    "scheme": "graded",
                    "growth_rate": 1.07,
                    "max_scale": 3.0,
                },
                "dirichlet_bc": {"zmin": 300.0},
                "neumann_bc": {
                    "xmin": 0.0,
                    "xmax": 0.0,
                    "ymin": 0.0,
                    "ymax": 0.0,
                    "zmax": 0.0,
                },
                "build_source_maps": True,
                "combine_sources": True,
                "use_tidy3d": True,
                "solver_options": {
                    "dirichlet_mode": "reduced_free_dof",
                    "pydiso_solver": {
                        "matrix_type": "real_symmetric_positive_definite",
                        "reuse_factorization": True,
                    },
                    "jax_solver": {
                        "method": "cg",
                        "precond": True,
                        "solve_dtype": "float64",
                        "check_residual": True,
                        "residual_rtol": 1e-6,
                        "residual_atol": 1e-8,
                    },
                },
                "adjoint_solver_options": {
                    "dirichlet_mode": "reduced_free_dof",
                    "pydiso_solver": {
                        "matrix_type": "real_symmetric_positive_definite",
                        "reuse_factorization": True,
                    },
                    "jax_solver": {
                        "method": "cg",
                        "precond": True,
                        "solve_dtype": "float64",
                        "check_residual": True,
                        "residual_rtol": 1e-6,
                        "residual_atol": 1e-8,
                    },
                },
                "requires_temp_grad": False,
            },
        )
    )

    heat_source_names = ("heater_1",)
    off_control_key = canonicalize_currents(
        {"heater_1": 0.0}, heat_source_names=heat_source_names
    )
    on_control_key = canonicalize_currents(
        {"heater_1": heater_on_current}, heat_source_names=heat_source_names
    )

    def fom_func(breakdown):
        off_input = breakdown["off_input_trans"]["value"].detach()
        on_input = breakdown["on_input_trans"]["value"].detach()
        breakdown["off_trans"]["value"] = breakdown["off_trans"]["value"] / off_input
        breakdown["on_trans"]["value"] = breakdown["on_trans"]["value"] / on_input
        breakdown["off_cross_talk"]["value"] = (
            breakdown["off_cross_talk"]["value"] / off_input
        )
        breakdown["on_cross_talk"]["value"] = (
            breakdown["on_cross_talk"]["value"] / on_input
        )

        fom = 0
        for key, obj in breakdown.items():
            if key in {"off_input_trans", "on_input_trans"}:
                continue
            if "cross_talk" in key:
                fom = fom + (obj["value"] - 0) ** 4 * obj["weight"]
            else:
                fom = fom - (obj["value"] - 1) ** 4 * obj["weight"]

        # product = breakdown["off_trans"]["value"] * breakdown["on_trans"]["value"] * 5
        fom = fom  # + product
        return fom, {}

    obj_cfgs = dict(
        off_trans=dict(
            weight=1,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            wl=[wl_cen],
            currents=[{"heater_1": 0.0}],
            in_mode=mode,
            out_modes=(mode,),
            type="eigenmode",
            direction="x+",
        ),
        off_cross_talk=dict(
            weight=-0.2,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_2",
            wl=[wl_cen],
            currents=[{"heater_1": 0.0}],
            in_mode=mode,
            out_modes=(mode,),
            type="flux",
            direction="x+",
        ),
        on_trans=dict(
            weight=1,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_2",
            wl=[wl_cen],
            currents=[{"heater_1": heater_on_current}],
            in_mode=mode,
            out_modes=(mode,),
            type="eigenmode",
            direction="x+",
        ),
        on_cross_talk=dict(
            weight=-0.2,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            wl=[wl_cen],
            currents=[{"heater_1": heater_on_current}],
            in_mode=mode,
            out_modes=(mode,),
            type="flux",
            direction="x+",
        ),
        off_input_trans=dict(
            weight=1,
            in_slice_name="in_slice_1",
            out_slice_name="refl_slice_1",
            wl=[wl_cen],
            currents=[{"heater_1": 0.0}],
            in_mode=mode,
            out_modes=(mode,),
            type="eigenmode",
            direction="x+",
            requires_grad=False,
        ),
        on_input_trans=dict(
            weight=1,
            in_slice_name="in_slice_1",
            out_slice_name="refl_slice_1",
            wl=[wl_cen],
            currents=[{"heater_1": heater_on_current}],
            in_mode=mode,
            out_modes=(mode,),
            type="eigenmode",
            direction="x+",
            requires_grad=False,
        ),
        _fusion_func=fom_func,
        override=True,
    )

    device = TDM(
        material_r1="Si",
        sim_cfg=sim_cfg,
        box_size=region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        port_height=(thickness, thickness),
        heater_gap=heater_gap,
        heater_height=heater_thickness,
        heater_length=heater_length,
        heater_width=heater_width,
        heater_center=[0, -1.0],
        heater_on_current=heater_on_current,
        input_port_offset=-1.0,
        is_3d=True,
        mode=mode,
        port_box_margin=0.2,
        device=operation_device,
    )

    hr_device = device.copy(resolution=resolution)
    print(device)

    design_region_param_cfgs = dict()
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[resolution, resolution],
            sigma=1 / resolution,
            transform=[
                # dict(
                #     type="blur",
                #     mfs=mfs,
                #     resolutions=[hr_device.resolution, hr_device.resolution],
                #     dim="xy",
                # ),
                dict(type="binarize"),
            ],
            init_method="constant_0.4",
            denorm_mode="inverse_eps",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
            dims=(0, 1),
            extrude_direction="-",
            extrude_angle=90.0,
        )

    opt = TDMOptimization(
        device=device,
        hr_device=hr_device,
        design_region_param_cfgs=design_region_param_cfgs,
        sim_cfg=sim_cfg,
        obj_cfgs=obj_cfgs,
        operation_device=operation_device,
    ).to(operation_device)

    plot_device_maps(
        device,
        sim_cfg["plot_root"],
        map_names=[
            "epsilon",
            "conductivity",
            # "thermo_optic_coeff_map",
        ],
    )

    invdesign = InvDesign(
        devOptimization=opt,
        run=Config(
            n_epochs=100,
        ),
        optimizer=dict(
            name="Adam",
            lr=2e-1,
            weight_decay=0,
            init_v=1e-10,
        ),
        sharp_scheduler=Config(
            mode="cosine",
            name="sharpness",
            init_sharp=4,
            final_sharp=20,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=1,
            plot_name=f"{exp_name}",
            objs=["off_trans", "on_trans"],
            field_keys=[
                ("in_slice_1", wl_cen, mode, off_control_key),
                ("in_slice_1", wl_cen, mode, on_control_key),
            ],
            in_slice_names=["in_slice_1", "in_slice_1"],
            exclude_slice_names=[[], []],
            field_component="Ey",
            thermal_map_names=["optical_temperature", "optical_temperature"],
            eps_grad=True,
            show_delta_eps=True,
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
            upsample_eps_to_1nm=True,
        ),
    )
    invdesign.optimize()


if __name__ == "__main__":
    gpu_id = 0
    for mfs in [0.15]:
        generate_tdm_3d(gpu_id, mfs)
