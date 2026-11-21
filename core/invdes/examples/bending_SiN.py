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


def init_func_euler(
    weight_dict,
    param_cfg,
    region_cfg,
    bending_region_size,
    bending_width,
    port_offset,
    A,
):

    weight = weight_dict["ls_knots"]
    X, Y = weight.shape
    device = weight.device

    # -----------------------------------
    # Geometry in your current layout
    # ls_knots is treated as [x_dim, y_dim]
    # -----------------------------------

    # Start/end points of the Euler bend centerline in physical coordinates
    # input from left, output to top
    start_point = np.array(
        [-bending_region_size[0] / 2, port_offset], dtype=np.float64
    )  # (-10, -8)
    end_point = np.array(
        [port_offset, -bending_region_size[1] / 2], dtype=np.float64
    )  # (8, 10)

    dx = float(end_point[0] - start_point[0])
    dy = float(end_point[1] - start_point[1])

    # if abs(dx - dy) > 1e-6:
    #     raise ValueError(
    #         f"Euler bend initializer expects a 90-degree bend with equal x/y span, got dx={dx}, dy={dy}."
    #     )

    R_eff = port_offset + bending_region_size[0] / 2
    A = A * R_eff

    def euler_curve(L_max: float, A: float, num_points: int = 200):
        Ls = np.linspace(0.0, L_max, num_points)
        x1 = np.zeros_like(Ls)
        y1 = np.zeros_like(Ls)
        for i, L in enumerate(Ls):
            y1[i], _ = integrate.quad(lambda th: A * np.sin(th**2 / 2.0), 0.0, L / A)
            x1[i], _ = integrate.quad(lambda th: A * np.cos(th**2 / 2.0), 0.0, L / A)
        return x1, y1

    def solve_L_max(R_eff: float, A: float, num_points: int = 200, tol: float = 1e-10):
        def residual(L_arr):
            L_max = float(np.atleast_1d(L_arr)[0])
            x1, y1 = euler_curve(L_max, A, num_points=num_points)

            # local tangent slope at the end of the first clothoid
            k = -(x1[-1] - x1[-2]) / (y1[-1] - y1[-2])
            xp, yp = x1[-1], y1[-1]

            # same matching condition used in the Tidy3D notebook
            R = np.sqrt(
                ((R_eff + k * xp - yp) / (k + 1.0) - xp) ** 2
                + (-(R_eff + k * xp - yp) / (k + 1.0) + R_eff - yp) ** 2
            )
            return R - (A**2 / L_max)

        sol = fsolve(residual, x0=np.array([R_eff / 2.0], dtype=np.float64), xtol=tol)
        L_max = float(sol[0])
        err = abs(float(residual(np.array([L_max]))))
        if err > 1e-6:
            raise ValueError(
                f"Failed to solve Euler bend L_max accurately (residual={err})."
            )
        return L_max

    def polyline_tube_mask(xx, yy, pts, half_w):
        """
        Compute a tube mask around a polyline by min distance to line segments.
        xx, yy: 2D numpy arrays of the same shape
        pts:    [N, 2] polyline vertices
        """
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
        pts = np.asarray(pts, dtype=np.float64)

        p0 = pts[:-1]
        p1 = pts[1:]
        seg = p1 - p0
        seg_len2 = np.sum(seg * seg, axis=1)
        seg_len2 = np.maximum(seg_len2, 1e-15)

        min_d2 = np.full((grid.shape[0],), np.inf, dtype=np.float64)

        # Segment chunking to keep memory modest
        chunk = 32
        for s in range(0, len(p0), chunk):
            a = p0[s : s + chunk]  # [C, 2]
            b = seg[s : s + chunk]  # [C, 2]
            l2 = seg_len2[s : s + chunk]  # [C]

            # grid[:, None, :] - a[None, :, :] -> [P, C, 2]
            ap = grid[:, None, :] - a[None, :, :]
            t = np.sum(ap * b[None, :, :], axis=-1) / l2[None, :]
            t = np.clip(t, 0.0, 1.0)

            proj = a[None, :, :] + t[..., None] * b[None, :, :]
            d2 = np.sum((grid[:, None, :] - proj) ** 2, axis=-1)
            min_d2 = np.minimum(min_d2, d2.min(axis=1))

        return min_d2.reshape(xx.shape) <= half_w**2

    # -----------------------------------
    # Canonical 90-degree Euler bend:
    # (0,0) -> (R_eff, R_eff)
    # -----------------------------------
    L_max = solve_L_max(R_eff, A)
    x1, y1 = euler_curve(L_max, A, num_points=200)

    xp, yp = x1[-1], y1[-1]
    R_min = A**2 / L_max

    # Mirror the first clothoid to get the second one
    x3 = np.flipud(R_eff - y1)
    y3 = np.flipud(R_eff - x1)

    # Solve circular connector center
    def circle(var):
        a, b = var
        f = np.empty(2, dtype=np.float64)
        f[0] = (xp - a) ** 2 + (yp - b) ** 2 - R_min**2
        f[1] = (R_eff - yp - a) ** 2 + (R_eff - xp - b) ** 2 - R_min**2
        return f

    a, b = fsolve(circle, (0.0, R_eff))

    x2 = np.linspace(xp + 0.01, R_eff - yp - 0.01, 120)
    y2 = -np.sqrt(np.maximum(R_min**2 - (x2 - a) ** 2, 0.0)) + b

    x_euler = np.concatenate([x1, x2, x3])
    y_euler = np.concatenate([y1, y2, y3])

    # -----------------------------------
    # Place the bend using start/end points
    # -----------------------------------

    y_euler = -y_euler

    # Translate so the start coincides with start_point
    x_euler += start_point[0]
    y_euler += start_point[1]

    curve_pts = np.stack([x_euler, y_euler], axis=1)

    # -----------------------------------
    # Build the seed mask on the ls_knots grid
    # -----------------------------------
    x = np.linspace(-bending_region_size[0] / 2, bending_region_size[0] / 2, X)
    y = np.linspace(-bending_region_size[1] / 2, bending_region_size[1] / 2, Y)
    xx, yy = np.meshgrid(x, y, indexing="ij")  # [X, Y]

    half_w = bending_width / 2.0
    seed = polyline_tube_mask(xx, yy, curve_pts, half_w)

    # -----------------------------------
    # Initialize ls_knots
    # -----------------------------------
    with torch.no_grad():
        weight.data.fill_(-0.3)
        weight.data[torch.as_tensor(seed, device=device)] = 0.3


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

    exp_name = "bending_SiN"
    work_root = "./runs"
    resume_checkpoint = os.environ.get("MAPS_RESUME_CHECKPOINT")
    mode = "Hz1"
    resolution = 50
    wl_cen = 1.310
    wl_width = 0.06
    n_wl = 3
    sidewall_angle = 0

    run_manager = OptimizationRunManager(
        root=work_root,
        experiment="bending_SiN",
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
            border_width=[0, 1, 1, 0],
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
        material_r1=1.70560**2,
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
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[25, 25],
            sigma=2 / 25,  # 2 * levelset knot grid step
            transform=[
                dict(type="transpose_symmetry", rot_k=3),
                # dict(
                #     type="blur",
                #     mfs=0.05,
                #     resolutions=[hr_device.resolution, hr_device.resolution],
                #     dim="xy",
                # ),
                dict(type="binarize"),
            ],
            # init_method="bending_0.5",
            # init_method="constant_0.2",
            # init_method=init_func_circular,
            init_method=partial(
                init_func_euler,
                A=0.5,
                port_offset=port_offset,
                bending_region_size=bending_region_size,
                bending_width=input_port_width + 0.1,
            ),
            # init_method="constant_0.5",
            # denorm_mode="linear_eps",
            denorm_mode="inverse_eps",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
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
        # interface_penalty=dict(
        #     weight=-0.000,
        #     #### objective is evaluated at this port
        #     in_slice_name="in_slice_1",
        #     #### objective is evaluated at all points by sweeping the wavelength and modes
        #     in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
        #     wl=wls,
        #     temp=[300],
        #     type="interface_penalty",
        #     interface_weights=(1, 1, 0),
        #     requires_grad=True,  # grad is needed, to avoid it from being detached.
        #     requires_adjoint=False,  # grad is not related to fields, thus no adjoint run is needed.
        # ),
        # interface_penalty=dict(
        #     weight=-2,
        #     #### objective is evaluated at this port
        #     in_slice_name="in_slice_1",
        #     #### objective is evaluated at all points by sweeping the wavelength and modes
        #     in_mode=mode,  # only one source mode is supported, cannot input multiple modes at the same time
        #     wl=wls,
        #     temp=[300],
        #     type="structure_simplify",
        #     energy_threshold=0.1,
        #     requires_grad=True,  # grad is needed, to avoid it from being detached.
        #     requires_adjoint=False,  # grad is not related to fields, thus no adjoint run is needed.
        # ),
        energy_penalty=dict(
            weight=-0,
            # weight=-5000,
            #### objective is evaluated at this port
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=[wl_cen],
            temp=[300],
            in_mode=f"{mode}",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                f"{mode}",
            ),  # can evaluate on multiple output modes and get average transmission
            type="energy_constraint",
            requires_grad=True,
            requires_adjoint=False,
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
        fom = fom / n_wl
        penalty = breakdown["energy_penalty"]["value"]
        fom = fom + penalty * breakdown["energy_penalty"]["weight"]
        spectrum["avg|s21|^2"] = {"weight": 1, "value": trans.mean().item()}
        return fom, spectrum | {
            "energy_penalty": {
                "weight": breakdown["energy_penalty"]["weight"],
                "value": penalty.item(),
            }
        }

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
                grad = rho.weights["ls_knots"].grad.abs()
                threshold = 0.001 * grad.max()
                rho.weights["ls_knots"].data = torch.where(
                    grad < threshold,
                    torch.clamp(rho.weights["ls_knots"].data - 0.2, -0.5, 0.5),
                    rho.weights["ls_knots"].data,
                )

    invdesign = InvDesign(
        devOptimization=opt,
        run_manager=run_manager,
        optimizer=dict(
            name="Adam",
            lr=0.2,
            weight_decay=0,
        ),
        run=Config(
            start_epoch=0,
            n_epochs=50,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=0.1,
        ),
        sharp_scheduler=Config(
            mode="linear",
            name="sharpness",
            init_sharp=128,
            # final_sharp=20,
            final_sharp=128,
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
            save_model=True,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
            upsample_eps_to_1nm=True,
            load_ckpt=resume_checkpoint,
            resume=resume_checkpoint is not None,
        ),
        # after_step_callbacks=[call_back],
    )
    invdesign.optimize()
