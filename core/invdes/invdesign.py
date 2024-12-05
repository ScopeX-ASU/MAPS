'''
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
'''
import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "/home/pingchua/projects/MAPS"))
sys.path.insert(0, project_root)
import torch
from pyutils.typing import Criterion, Optimizer, Scheduler
from core.invdes import builder
from pyutils.config import Config
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import Bending
from core.invdes.models import (
    BendingOptimization,
)
from core.utils import set_torch_deterministic

class InvDesign:
    '''
    default_cfgs is to set the default configurations 
    including optimizer, lr_scheduler, sharp_scheduler etc.
    '''
    default_cfgs = Config(
        devOptimization=None,
        optimizer=Config(
            name="Adam",
            lr=2e-2,
            weight_decay=0,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=2e-4,
        ),
        sharp_scheduler=Config(
            name="sharpness",
            init_sharp=1,
            final_sharp=256,
        ),
        run=Config(
            n_epochs=100,
        ),
    )

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.load_cfgs(**kwargs)
        assert self.devOptimization is not None, "devOptimization must be provided"
        # make optimizer and scheduler
        self.optimizer = builder.make_optimizer(
            params=self.devOptimization.parameters(),
            total_config=self._cfg,
        )
        self.lr_scheduler = builder.make_scheduler(
            optimizer=self.optimizer,
            scheduler_type='lr_scheduler',
            config_total=self._cfg,
        )
        self.sharp_scheduler = builder.make_scheduler(
            optimizer=self.optimizer,
            scheduler_type='sharp_scheduler',
            config_total=self._cfg,
        )


    def load_cfgs(self, **cfgs):
        # Start with default configurations
        self.__dict__.update(self.default_cfgs)
        # Update with provided configurations
        self.__dict__.update(cfgs)
        # Save the updated configurations
        self.default_cfgs.update(cfgs)
        self._cfg = self.default_cfgs

    def optimize(
        self, 
    ):
        for i in range(self._cfg.run.n_epochs):
            # train the model
            self.optimizer.zero_grad()
            sharpness = self.sharp_scheduler.get_sharpness()
            # forward pass
            results = self.devOptimization.forward(sharpness=sharpness)
            print(f"Step {i}:", end=" ")
            for k, obj in results["breakdown"].items():
                print(f"{k}: {obj['value']:.3f}", end=", ")
            print()
            # backward pass
            (-results["obj"]).backward()
            # update the weights
            self.optimizer.step()
            # update the learning rate
            self.lr_scheduler.step()
            # update the sharpness
            self.sharp_scheduler.step()

if __name__ == "__main__":
    gpu_id = 1
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41+500))
    # first we need to instantiate the a optimization object
    sim_cfg = DefaultSimulationConfig()

    bending_region_size = (1.6, 1.6)
    port_len = 1.8

    input_port_width = 0.48
    output_port_width = 0.48

    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            border_width=[0, port_len, port_len, 0],
            resolution=50,
            plot_root=f"./figs/test_mfs_bending_{500}",
            PML=[0.5, 0.5],
            neural_solver=None,
            numerical_solver="solve_direct",
            use_autodiff=False,
        )
    )

    device = Bending(
        sim_cfg=sim_cfg, 
        bending_region_size=bending_region_size,
        port_len=(port_len, port_len),
        port_width=(
            input_port_width,
            output_port_width
        ), 
        device=operation_device
    )

    hr_device = device.copy(resolution=310)
    print(device)
    opt = BendingOptimization(device=device, hr_device=hr_device, sim_cfg=sim_cfg, operation_device=operation_device).to(operation_device)
    invdesign = InvDesign(
        devOptimization=opt
    )
    invdesign.optimize()