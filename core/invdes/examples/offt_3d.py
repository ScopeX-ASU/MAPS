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
import torch
from autograd.numpy import array as npa
from pyutils.config import Config
from pyutils.general import logger

from core.invdes.invdesign import InvDesign
from core.invdes.models import MMIOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import MMI
from core.invdes.run_management import OptimizationRunManager
from core.utils import set_torch_deterministic

sys.path.pop(0)


def bit_reversal_indices(n: int, device=None):
    """
    Generate bit-reversal permutation indices for FFT.

    Args:
        n: FFT size (must be a power of 2)

    Returns:
        torch.LongTensor of shape (n,)
    """
    if n & (n - 1):
        raise ValueError("n must be a power of 2")

    m = (n - 1).bit_length()

    idx = torch.arange(n, dtype=torch.long, device=device)
    rev = torch.zeros_like(idx)

    for _ in range(m):
        rev = (rev << 1) | (idx & 1)
        idx >>= 1

    return rev


if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    thickness = 0.22
    mmi_region_size = (5, 2.5, thickness)
    port_box_margin = 0.2

    # mmi_region_size = (15, 10)
    # mmi_region_size = (10, 10)
    # mmi_region_size = (16, 16)

    # mmi_region_size = (8, 8, thickness)
    # port_box_margin = 2
    port_len = 2.0

    input_port_width = 0.5
    output_port_width = 0.5
    num_inports = 2
    num_outports = 2
    resolution = 50
    wl_cen = 1.55
    wl_width = 0.00
    n_wl = 1
    pol = "Hz"
    sidewall_angle = 0.0
    material_r1 = "Si"
    exp_name = f"offt_3d_opt_{material_r1}_{num_inports}x{num_outports}_{mmi_region_size[0]}x{mmi_region_size[1]}_perm"
    work_root = "./runs"
    resume_checkpoint = os.environ.get("MAPS_RESUME_CHECKPOINT")

    run_manager = OptimizationRunManager(
        root=work_root,
        experiment="offt_3d",
        tag=exp_name,
        resume=resume_checkpoint is not None,
        resume_from=resume_checkpoint,
    )
    run_manager.attach_logger(logger)

    sim_cfg.update(
        dict(
            solver="fdtdx",
            symmetry=(0, 0, 0),
            subpixel=True,
            max_time=2e-13,
            optical_grid={
                "mode": "fdtdx_rectilinear_nonuniform",
                "wavelength": wl_cen,
                "grid_x": {"type": "auto", "min_steps_per_wvl": 12},
                "grid_y": {"type": "auto", "min_steps_per_wvl": 12},
            },
            border_width=[
                0,
                0,
                1,
                1,
                1.2,
                1.2,
            ],  # left, right, lower, upper, containing PML
            resolution=resolution,
            PML=[wl_cen / 2, wl_cen / 2, wl_cen / 2],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=wl_cen,
            wl_width=wl_width,
            n_wl=n_wl,
        )
    )
    run_manager.bind_simulation_config(sim_cfg)

    ## get target DFT transfer matrix
    target_unitary_smatrix = torch.fft.fft(
        torch.eye(num_outports, dtype=torch.complex64, device=operation_device),
        dim=1,
        norm="ortho",
    )  # * (0.485 / 0.5)

    # target_transmission = torch.tensor(
    #     [0.5, 0.5], dtype=torch.float32, device=operation_device
    # )

    # num_ports = max(num_inports, num_outports)
    # in_port_idx = torch.arange(1, num_inports + 1, device=operation_device)
    # out_port_idx = torch.arange(1, num_outports + 1, device=operation_device)
    # in_port_idx_f = in_port_idx.to(torch.float32)[None, :]
    # out_port_idx_f = out_port_idx.to(torch.float32)[:, None]
    # parity = 1 - 2 * ((out_port_idx[:, None] + in_port_idx[None, :]) & 1)
    # parity_f = parity.to(torch.float32)
    # phase_term = ((out_port_idx_f - 0.5) - parity_f * (in_port_idx_f - 0.5)) ** 2 * (
    #     torch.pi / (4 * num_ports)
    # )
    # prefactor = torch.exp(
    #     1j
    #     * torch.tensor(3 * torch.pi / 4, dtype=torch.float32, device=operation_device)
    # ) / (num_ports**0.5)
    # target_unitary_smatrix = (
    #     parity_f.to(torch.complex64) * prefactor * torch.exp(-1j * phase_term)
    # ).contiguous()

    ## butterfly bit reversal index permutation
    indices = bit_reversal_indices(num_outports, device=operation_device)
    target_unitary_smatrix = target_unitary_smatrix[indices][:, indices]
    target_norm = torch.norm(target_unitary_smatrix)
    print(f"target unitary smatrix:\n{target_unitary_smatrix}")
    print(target_unitary_smatrix.shape)

    ## 4-point DFT
    """
    target DFT matrix
    target unitary smatrix:
      ([[ 0.5000+0.0000j,  0.5000+0.0000j,  0.5000+0.0000j,  0.5000+0.0000j],
        [ 0.5000+0.0000j,  0.0000-0.5000j, -0.5000+0.0000j,  0.0000+0.5000j],
        [ 0.5000+0.0000j, -0.5000+0.0000j,  0.5000+0.0000j, -0.5000+0.0000j],
        [ 0.5000+0.0000j,  0.0000+0.5000j, -0.5000+0.0000j,  0.0000-0.5000j]],
       device='cuda:0')
       """

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        penalty = 0
        for key, obj in breakdown.items():
            if key == "smatrix":
                s_matrix = obj["value"]
                trans = torch.abs(s_matrix) ** 2
                if s_matrix.shape == (num_inports, num_outports):
                    s_matrix = s_matrix.transpose(0, 1)
                elif s_matrix.shape != (num_outports, num_inports):
                    if s_matrix.numel() != num_outports * num_inports:
                        raise ValueError(
                            f"Unexpected smatrix shape {tuple(s_matrix.shape)}"
                        )
                    s_matrix = s_matrix.reshape(num_inports, num_outports).transpose(
                        0, 1
                    )
                ## need to normalize to input s_param
                input_sparams = torch.stack(
                    [
                        breakdown[f"input_sparam_{i+1}"]["value"].squeeze().data
                        for i in range(num_inports)
                    ],
                    dim=0,
                )[..., None]
                s_matrix = s_matrix / input_sparams
                obj["value"] = s_matrix
                fom += (
                    1
                    - torch.norm(s_matrix - target_unitary_smatrix) ** 2
                    / target_norm**2
                )
                # fom += (1 - (s_matrix.abs() - target_unitary_smatrix.abs()).square().mean()) / 2
                # fom += (1 - (s_matrix.angle() - target_unitary_smatrix.angle()).square().mean()) / 4 / torch.pi

                # trans = torch.abs(s_matrix) ** 2
            if "energy_constraint" in key:
                penalty += obj["weight"] * obj["value"]
        results = {"smatrix_err": {"weight": 0, "value": 1 - fom.item()}}
        fom = fom + penalty
        results.update(
            {
                f"trans{i + 1}": {"weight": 0, "value": trans[i]}
                for i in range(num_outports)
            }
        )
        return fom, results

    # def fom_func(breakdown):
    #     ## maximization fom
    #     for key, obj in breakdown.items():
    #         if key == "smatrix":
    #             s_matrix = obj["value"]  # shape (num_outports, num_inports)
    #             trans = torch.abs(s_matrix) ** 2
    #             fom = -torch.sum((trans - target_transmission) ** 2)
    #     return fom, {
    #         "trans1": {"weight": 0, "value": trans[0]},
    #         "trans2": {"weight": 0, "value": trans[1]},
    #     }

    obj_cfgs = dict(
        smatrix=dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name=[f"in_slice_{i + 1}" for i in range(num_inports)],
            out_slice_name=[f"out_slice_{i + 1}" for i in range(num_outports)],
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=[wl_cen],  #
            temp=[300],
            out_modes=(
                f"{pol}1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="smatrix",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            direction=["x+"] * num_inports,
        ),
        _fusion_func=fom_func,
        override=True,
    )

    for i in range(num_inports):
        obj_cfgs[f"input_sparam_{i+1}"] = dict(
            weight=1,
            #### objective is evaluated at this port
            in_slice_name=f"in_slice_{i + 1}",
            out_slice_name=f"refl_slice_{i + 1}",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
            wl=[wl_cen],  #
            temp=[300],
            out_modes=(
                f"{pol}1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
            energy=False,
            direction="x+",
            requires_grad=False,
            requires_adjoint=False,
        )
        obj_cfgs[f"energy_constraint_{i + 1}"] = dict(
            weight=-1e3,
            #### objective is evaluated at this port
            in_slice_name=f"in_slice_{i + 1}",
            out_slice_name="out_slice_1",
            #### objective is evaluated at all points by sweeping the wavelength and modes
            wl=[wl_cen],
            temp=[300],
            in_mode=f"{pol}1",  # only one source mode is supported, cannot input multiple modes at the same time
            out_modes=(
                f"{pol}1",
            ),  # can evaluate on multiple output modes and get average transmission
            type="energy_constraint",
            requires_grad=True,
            requires_adjoint=False,
        )

    # obj_cfgs = dict(_fusion_func=fom_func)

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
        port_box_margin=port_box_margin,
        sidewall_angle=sidewall_angle,
        is_3d=True,
        pol=pol,
    )

    hr_device = device.copy(resolution=resolution)
    print(device)

    design_region_param_cfgs = {}
    for region_name in device.design_region_cfgs.keys():
        design_region_param_cfgs[region_name] = dict(
            method="levelset",
            rho_resolution=[resolution, resolution],
            sigma=2 / resolution,
            transform=[
                ## DFT matrix is symmetric
                dict(type="mirror_symmetry", dims=[0]),
                # dict(
                #     type="blur",
                #     mfs=0.05,
                #     resolutions=[hr_device.resolution, hr_device.resolution],
                #     dim="xy",
                # ),
                dict(type="binarize"),
            ],
            init_method="random_0.1",
            # init_method="constant_0.0",
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
            lr=0.3,
            init_v=1e-3,
            # name="lbfgs",
            # line_search_fn="strong_wolfe",
            # lr=1e-1,
            # weight_decay=0,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=0.1,
        ),
        sharp_scheduler=Config(
            mode="cosine",
            name="sharpness",
            init_sharp=4,
            final_sharp=32,
        ),
        run=Config(
            n_epochs=100,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=1,
            plot_name=f"{exp_name}",
            objs=[f"trans{i + 1}" for i in range(num_outports)],
            field_keys=[
                (f"in_slice_{i + 1}", wl_cen, f"{pol}1", 300)
                for i in range(num_inports)
            ],
            in_slice_names=[f"in_slice_{i + 1}" for i in range(num_inports)],
            filename_suffixes=[f"s{i + 1}" for i in range(num_inports)],
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
    )
    invdesign.optimize()
