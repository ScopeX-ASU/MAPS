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

    # mmi_region_size = (5, 2.1)
    # port_box_margin = 0.2

    # mmi_region_size = (15, 10)
    # mmi_region_size = (10, 10)
    # mmi_region_size = (16, 16)
    mmi_region_size = (8, 8)
    port_box_margin = 2
    port_len = 2.0

    input_port_width = 0.5
    output_port_width = 0.5
    num_inports = 4
    num_outports = 4
    resolution = 50
    wl_cen = 1.55
    wl_width = 0.00
    n_wl = 1
    pol = "Hz"
    material_r1 = "Si_eff"
    exp_name = f"offt_opt_{material_r1}_{num_inports}x{num_outports}_{mmi_region_size[0]}x{mmi_region_size[1]}_perm"
    work_root = "./runs"
    resume_checkpoint = os.environ.get("MAPS_RESUME_CHECKPOINT")

    run_manager = OptimizationRunManager(
        root=work_root,
        experiment="offt",
        tag=exp_name,
        resume=resume_checkpoint is not None,
        resume_from=resume_checkpoint,
    )
    run_manager.attach_logger(logger)

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            symmetry=(-1, -1, 0),
            subpixel=True,
            optical_grid={
                "mode": "fdtdx_rectilinear_nonuniform",
                "wavelength": wl_cen,
                "grid_x": {"type": "auto", "min_steps_per_wvl": 20},
                "grid_y": {"type": "auto", "min_steps_per_wvl": 20},
            },
            border_width=[0, 0, 1, 1],  # left, right, lower, upper, containing PML
            resolution=resolution,
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
    """ e.g., target unitary smatrix:
    [[-0.1627-0.6450j,  0.6381-0.2135j, -0.0702-0.0481j],
     [-0.5426-0.0271j, -0.0188+0.5920j, -0.5033+0.0567j],
     [-0.2311+0.3339j,  0.2916+0.1178j,  0.2743-0.7506j]]

    trained matrix, 0.0581 relative L2 norm error
    [[-0.1511-0.6146j,  0.6196-0.2099j, -0.0684-0.0495j,]
     [-0.4896-0.0319j, -0.0191+0.5528j, -0.4856+0.0568j,]
     [-0.2111+0.2967j,  0.2755+0.1021j,  0.2669-0.7242j]
    """

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
                ## remove global phase
                # s_matrix = s_matrix * torch.exp(
                #     -1j * torch.angle(s_matrix[0][0]))
                fom += (
                    1
                    - torch.norm(s_matrix - target_unitary_smatrix) ** 2
                    / target_norm**2
                )
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
            # in_slice_name=["in_slice_1"],
            # out_slice_name=["out_slice_1"],
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
            # direction=["x+"],
        ),
        _fusion_func=fom_func,
        override=True,
    )

    for i in range(num_inports):
        obj_cfgs[f"energy_constraint_{i + 1}"] = dict(
            weight=-0.5,
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
        )

    # obj_cfgs = dict(_fusion_func=fom_func)

    device = MMI(
        material_ports=(material_r1, material_r1),
        material_cladding="SiO2",
        region_material_r1_layers=(material_r1,),
        region_material_r2_layers=("SiO2",),
        is_design_region=(True,),
        sim_cfg=sim_cfg,
        box_size=mmi_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        num_inports=num_inports,
        num_outports=num_outports,
        device=operation_device,
        port_box_margin=port_box_margin,
        is_3d=False,
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
                # dict(type="mirror_symmetry", dims=[0]),
                # dict(
                #     type="blur",
                #     mfs=0.05,
                #     resolutions=[hr_device.resolution, hr_device.resolution],
                #     dim="xy",
                # ),
                dict(type="binarize"),
            ],
            # init_method="random",
            init_method="constant_0.1",
            denorm_mode="linear_eps",
            interpolation="gaussian_linear",
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
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
            lr=1e-2,
            # init_v=1e-7,
            # name="lbfgs",
            # line_search_fn="strong_wolfe",
            # lr=1e-1,
            # weight_decay=0,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=1e-2,
        ),
        sharp_scheduler=Config(
            mode="cosine",
            name="sharpness",
            init_sharp=4,
            final_sharp=128,
        ),
        run=Config(
            n_epochs=100,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=10,
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
