"""
Date: 2025-01-04 20:49:15
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-16 14:31:07
FilePath: /MAPS/core/invdes/examples/bending.py
"""

"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
"""

import os
import sys
from functools import partial

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
import numpy as np
import scipy.integrate as integrate
import torch
from pyutils.config import Config
from pyutils.general import logger
from pyutils.torch_train import load_model
from scipy.optimize import fsolve

from core.invdes.invdesign import InvDesign
from core.invdes.models import BendingOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import Bending
from core.invdes.run_management import OptimizationRunManager
from core.utils import interface_field_penalty, set_torch_deterministic

sys.path.pop(0)


def init_func_euler_polygon(
    weight_dict,
    param_cfg,
    region_cfg,
    bending_region_size,
    bending_width,
    port_offset,
    A,
):
    poly = weight_dict["polygon"]
    n_vertices = poly.n_vertices
    if n_vertices % 2 != 0:
        raise ValueError("Euler polygon initialization requires an even vertex count")

    R_eff = port_offset + bending_region_size[0] / 2
    A = A * R_eff

    def euler_curve(L_max, num_points=50):
        lengths = np.linspace(0.0, L_max, num_points)
        x = np.array(
            [
                integrate.quad(lambda theta: A * np.cos(theta**2 / 2.0), 0.0, L / A)[0]
                for L in lengths
            ]
        )
        y = np.array(
            [
                integrate.quad(lambda theta: A * np.sin(theta**2 / 2.0), 0.0, L / A)[0]
                for L in lengths
            ]
        )
        return x, y

    def residual(L_arr):
        L_max = float(np.atleast_1d(L_arr)[0])
        x, y = euler_curve(L_max)
        k = -(x[-1] - x[-2]) / (y[-1] - y[-2])
        xp, yp = x[-1], y[-1]
        radius = np.sqrt(
            ((R_eff + k * xp - yp) / (k + 1.0) - xp) ** 2
            + (-(R_eff + k * xp - yp) / (k + 1.0) + R_eff - yp) ** 2
        )
        return radius - A**2 / L_max

    solution = fsolve(residual, x0=np.array([R_eff / 2.0]), xtol=1e-10)
    L_max = float(solution[0])
    if abs(residual(solution)) > 1e-6:
        raise ValueError("Failed to solve Euler bend initialization")

    x1, y1 = euler_curve(L_max)
    xp, yp = x1[-1], y1[-1]
    R_min = A**2 / L_max

    x3 = np.flipud(R_eff - y1)
    y3 = np.flipud(R_eff - x1)

    def circle(var):
        a, b = var
        return np.array(
            [
                (xp - a) ** 2 + (yp - b) ** 2 - R_min**2,
                (R_eff - yp - a) ** 2 + (R_eff - xp - b) ** 2 - R_min**2,
            ]
        )

    a, b = fsolve(circle, (0.0, R_eff))
    x2 = np.linspace(xp + 0.01, R_eff - yp - 0.01, 50)
    y2 = -np.sqrt(np.maximum(R_min**2 - (x2 - a) ** 2, 0.0)) + b

    region_center = np.asarray(region_cfg["center"][:2], dtype=np.float64)
    input_center = region_center + np.array(
        [-bending_region_size[0] / 2, port_offset], dtype=np.float64
    )
    output_center = region_center + np.array(
        [port_offset, -bending_region_size[1] / 2], dtype=np.float64
    )
    centerline = np.stack(
        [
            np.concatenate([x1, x2, x3]),
            -np.concatenate([y1, y2, y3]),
        ],
        axis=1,
    )
    centerline += input_center
    centerline[0] = input_center
    centerline[-1] = output_center

    # Sample the centerline directly and offset it along its local normal.
    # This avoids rasterizing the design region during parameter initialization.
    centerline_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    centerline_s = np.concatenate([[0.0], np.cumsum(centerline_lengths)])
    sample_s = np.linspace(0.0, centerline_s[-1], n_vertices // 2)
    sampled = np.column_stack(
        [
            np.interp(sample_s, centerline_s, centerline[:, 0]),
            np.interp(sample_s, centerline_s, centerline[:, 1]),
        ]
    )
    tangent = np.gradient(sampled, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(min=1e-12)
    tangent[0] = np.array([1.0, 0.0])
    tangent[-1] = np.array([0.0, -1.0])
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    half_width = bending_width / 2.0
    vertices = np.concatenate(
        [sampled + half_width * normal, (sampled - half_width * normal)[::-1]],
        axis=0,
    )

    center = torch.as_tensor(
        region_cfg["center"][:2],
        device=poly.raw_vertices.device,
        dtype=poly.raw_vertices.dtype,
    )
    physical_vertices = torch.as_tensor(
        vertices, device=poly.raw_vertices.device, dtype=poly.raw_vertices.dtype
    )
    relative_vertices = physical_vertices - center
    with torch.no_grad():
        poly.raw_vertices.data.copy_(
            relative_vertices.unsqueeze(0).expand_as(poly.raw_vertices)
        )
    poly.clip_vertices_()


if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    thickness = 0.30
    bending_region_size = (20, 20)
    port_len = 1.5
    port_offset = 8

    input_port_width = 0.8
    output_port_width = 0.8

    exp_name = "bending_SiN_polygon"
    work_root = "./runs"
    resume_checkpoint = os.environ.get("MAPS_RESUME_CHECKPOINT")
    mode = "Hz1"
    resolution = 50
    wl_cen = 1.310
    wl_width = 0.00
    n_wl = 1
    sidewall_angle = 0

    run_manager = OptimizationRunManager(
        root=work_root,
        experiment="bending_SiN_polygon",
        tag=exp_name,
        resume=resume_checkpoint is not None,
        resume_from=resume_checkpoint,
    )
    run_manager.attach_logger(logger)

    # wl_width = 0.00
    # n_wl = 1

    wls = torch.linspace(wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl).tolist()

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[
                0,
                1,
                1,
                0,
            ],
            subpixel=True,
            resolution=resolution,
            # optical_grid={
            #     "mode": "fdtdx_rectilinear_nonuniform",
            #     "wavelength": wl_cen,
            #     "grid_x": {"type": "auto", "min_steps_per_wvl": 15},
            #     "grid_y": {"type": "auto", "min_steps_per_wvl": 15},
            #     "grid_z": {"type": "auto", "min_steps_per_wvl": 15},
            # },
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
        )
    )
    run_manager.bind_simulation_config(sim_cfg)

    device = Bending(
        material_r1=1.9809**2,
        material_r2=1.4519**2,
        material_bg=1.4519**2,
        thickness_r1=thickness,
        thickness_r2=thickness,
        sim_cfg=sim_cfg,
        bending_region_size=bending_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        port_height=(thickness, thickness),
        LER_A=0.0,
        LER_Lc=0.0,
        sidewall_angle=sidewall_angle,
        device=operation_device,
        port_offset=port_offset,
        is_3d=False,
        mode=mode,
    )

    # hr_device = device.copy(resolution=310)
    hr_device = device.copy(resolution=50, optical_grid=None, LER_A=0.0, LER_Lc=0.0)
    print(device)

    def init_func_circular(weight_dict, param_cfg, region_cfg):
        weight = weight_dict["ls_knots"]
        X, Y = weight.shape
        device = weight.device

        # x is the first dimension, y is the second dimension
        x = torch.linspace(
            -bending_region_size[0] / 2, bending_region_size[0] / 2, X, device=device
        )
        y = torch.linspace(
            -bending_region_size[1] / 2, bending_region_size[1] / 2, Y, device=device
        )
        xx, yy = torch.meshgrid(x, y, indexing="ij")  # shape: [X, Y]

        # Bend center and width
        cx, cy = -bending_region_size[0] / 2, -bending_region_size[1] / 2
        half_w = input_port_width / 2.0
        R = port_offset + bending_region_size[0] / 2

        r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # Circular bend band
        arc = torch.abs(r - R) <= half_w

        seed = arc

        weight.data.fill_(-0.3)
        weight.data[seed] = 0.3

    design_region_param_cfgs = dict()
    n_polygon_vertices = 32
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="polygon",
            geometry_cfgs=dict(
                batch_dims=(1,),
                n_vertices=n_polygon_vertices,
                border_band_um=0.1,
                boundary_interpolation="catmull_rom",
                samples_per_segment=4,
                fixed_vertices={
                    0: (
                        -bending_region_size[0] / 2,
                        port_offset + input_port_width / 2,
                    ),
                    n_polygon_vertices // 2
                    - 1: (
                        port_offset + output_port_width / 2,
                        -bending_region_size[1] / 2,
                    ),
                    n_polygon_vertices
                    // 2: (
                        port_offset - output_port_width / 2,
                        -bending_region_size[1] / 2,
                    ),
                    n_polygon_vertices
                    - 1: (
                        -bending_region_size[0] / 2,
                        port_offset - input_port_width / 2,
                    ),
                },
            ),
            transform=[
                dict(type="transpose_symmetry", rot_k=3),
                dict(
                    type="blur",
                    mfs=0.1,
                    resolutions=[hr_device.resolution, hr_device.resolution],
                    dim="xy",
                    ste=True,
                ),
            ],
            init_method=partial(
                init_func_euler_polygon,
                A=0.5,
                port_offset=port_offset,
                bending_region_size=bending_region_size,
                bending_width=input_port_width,
            ),
            denorm_mode="linear_eps",
        )

    obj_cfgs = dict(
        fwd_trans=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,  # broadband monitor
            temp=[300],
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            # type="flux",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction="y+",
        ),
        ## input power monitor is needed to normalize the transmission, as fdtd didn't run normalization run.
        input_trans=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="refl_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
            wl=wls,
            temp=[300],
            out_modes=(
                mode,
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            # type="flux",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction="x+",
            requires_grad=False,  # only used for normalization, no adjoint run is needed.
        ),
        override=True,
    )

    def fom_func(breakdown):
        trans = breakdown["fwd_trans"]["value"]
        spectrum = {}
        fom = 0
        for wl, s in zip(wls, trans):
            spectrum[f"|s21|^2 at {wl:.2f}"] = {"weight": 1, "value": s.item()}
            fom = fom + s
        # penalty = breakdown["energy_penalty"]["value"]
        # fom = fom + penalty * breakdown["energy_penalty"]["weight"]
        spectrum["avg|s21|^2"] = {"weight": 1, "value": trans.mean().item()}
        return fom, spectrum

    obj_cfgs["_fusion_func"] = fom_func

    opt = BendingOptimization(
        device=device,
        hr_device=hr_device,
        design_region_param_cfgs=design_region_param_cfgs,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
        obj_cfgs=obj_cfgs,
    ).to(operation_device)

    def call_back(invdesign):
        for region_name in device.design_region_cfgs.keys():
            rho = invdesign.devOptimization.design_region_param_dict[region_name]
            with torch.no_grad():
                rho.weights["polygon"].clip_vertices_()

    invdesign = InvDesign(
        devOptimization=opt,
        run_manager=run_manager,
        optimizer=dict(
            name="Adam",
            lr=0.05,
            weight_decay=0,
        ),
        run=Config(
            start_epoch=0,
            n_epochs=50,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=0.01,
        ),
        sharp_scheduler=Config(
            mode="linear",
            name="sharpness",
            init_sharp=1,
            # final_sharp=20,
            final_sharp=1,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=1,
            plot_name=f"{exp_name}",
            objs=["avg|s21|^2"],
            field_keys=[("in_slice_1", wl_cen, mode, 300)],
            in_slice_names=["in_slice_1"],
            exclude_slice_names=[],
            field_component="Hz",
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
        after_step_callbacks=[call_back],
    )
    invdesign.optimize()
