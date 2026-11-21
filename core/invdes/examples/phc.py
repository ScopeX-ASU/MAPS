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
from pyutils.config import Config
from pyutils.general import logger

from core.invdes.invdesign import InvDesign
from core.invdes.models import PhCOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import PhC
from core.invdes.run_management import OptimizationRunManager
from core.utils import set_torch_deterministic

sys.path.pop(0)
if __name__ == "__main__":
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    phc_region_size = (4, 4)
    port_len = 1.8

    input_port_width = 0.52
    output_port_width = 0.52
    exp_name = "phc_opt"
    work_root = "./runs"
    resume_checkpoint = os.environ.get("MAPS_RESUME_CHECKPOINT")

    run_manager = OptimizationRunManager(
        root=work_root,
        experiment="phc",
        tag=exp_name,
        resume=resume_checkpoint is not None,
        resume_from=resume_checkpoint,
    )
    run_manager.attach_logger(logger)

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, 0, 1, 1],  # left, right, lower, upper, containing PML
            resolution=100,
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
            wl_cen=1.55,
            wl_width=0.1,
            n_wl=11,
        )
    )
    run_manager.bind_simulation_config(sim_cfg)

    def fom_func(breakdown):
        ## maximization fom
        fom = 0
        for key, obj in breakdown.items():
            if "wl1_trans" in key:
                # fom = fom + obj["value"] * obj["weight"]
                pass
        # maximize the forward transmission and minimize the standard deviation of the s-parameters
        return fom, {}

    obj_cfgs = dict(_fusion_func=fom_func)

    device = PhC(
        material_r1="Si_eff",
        material_r2="Air",
        sim_cfg=sim_cfg,
        box_size=phc_region_size,
        port_len=(port_len, port_len),
        port_width=(input_port_width, output_port_width),
        device=operation_device,
        r0=0.10,
    )

    hr_device = device.copy(resolution=200)
    print(device)
    opt = PhCOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
        obj_cfgs=obj_cfgs,
    ).to(operation_device)
    invdesign = InvDesign(
        devOptimization=opt,
        run_manager=run_manager,
        run=Config(
            n_epochs=1,
        ),
        plot_cfgs=Config(
            plot=True,
            interval=5,
            plot_name=f"{exp_name}",
            objs=["wl1_trans"],
            field_keys=[
                ("in_slice_1", 1.55, "Hz1", 300),
            ],
            in_slice_names=["in_slice_1"],
            filename_suffixes=["s1"],
            exclude_port_names=[],
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=f"{exp_name}",
            dump_gds=True,
            gds_name=f"{exp_name}",
            load_ckpt=resume_checkpoint,
            resume=resume_checkpoint is not None,
        ),
    )
    invdesign.optimize()
