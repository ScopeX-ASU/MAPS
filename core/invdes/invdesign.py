"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
"""

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../MAPS"))
sys.path.insert(0, project_root)
from concurrent.futures import ThreadPoolExecutor

import torch
from pyutils.config import Config
from pyutils.general import logger

from core.invdes import builder
from core.invdes.models import (
    BendingOptimization,
)
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import Bending
from core.utils import set_torch_deterministic
from pyutils.torch_train import BestKModelSaver

class InvDesign:
    """
    default_cfgs is to set the default configurations
    including optimizer, lr_scheduler, sharp_scheduler etc.
    """

    default_cfgs = Config(
        devOptimization=None,
        optimizer=Config(
            name="Adam",
            lr=1e-2,
            # name="lbfgs",
            # line_search_fn="strong_wolfe",
            # lr=1e-2,
            weight_decay=0,
        ),
        lr_scheduler=Config(
            name="cosine",
            lr_min=2e-4,
        ),
        sharp_scheduler=Config(
            mode="cosine",
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
            scheduler_type="lr_scheduler",
            config_total=self._cfg,
        )
        self.sharp_scheduler = builder.make_scheduler(
            optimizer=self.optimizer,
            scheduler_type="sharp_scheduler",
            config_total=self._cfg,
        )
        # self.plot_thread = ThreadPoolExecutor(2)
        self.saver = BestKModelSaver(
            k=1,
            descend=False,
            truncate=10,
            metric_name="err",
            format="{:.4f}",
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
        plot=False,
        plot_filename=None,
        objs=[],
        field_keys=[],
        in_slice_names=[],
        exclude_slice_names=[],
        dump_gds=False,
        save_model=False,
        field_component=None,
        ckpt_name=None,
        verbose: bool=True,
    ):
        if plot:
            assert plot_filename is not None, "plot_filename must be provided"
            assert len(objs) > 0, "objs must be provided"
            assert len(field_keys) > 0, "field_keys must be provided"
            assert len(in_slice_names) > 0, "in_port_names must be provided"
            if len(exclude_slice_names) == 0:
                exclude_slice_names = [[]] * len(objs)

        class Closure(object):
            def __init__(
                self,
                optimizer,  # optimizer
                devOptimization,  # device optimization model,
            ):
                self.results = None
                self.optimizer = optimizer
                self.devOptimization = devOptimization
                self.sharpness = 1

            def __call__(self):
                # clear grad here
                self.optimizer.zero_grad()
                # forward pass
                results = self.devOptimization.forward(sharpness=self.sharpness)

                # need backward to compute grad
                (-results["obj"]).backward()

                # store any results for plot/log
                self.results = results

                ## return the loss for gradient descent
                return -results["obj"]

        closure = Closure(
            optimizer=self.optimizer,
            devOptimization=self.devOptimization,
        )

        for i in range(self._cfg.run.n_epochs):
            sharpness = self.sharp_scheduler.get_sharpness()
            closure.sharpness = sharpness

            self.optimizer.step(closure)
            results = closure.results
            self.results = results # record this result

            log = f"Step {i:3d} (sharp: {sharpness:.1f}) "
            log += ", ".join(
                [f"{k}: {obj['value']:.3f}" for k, obj in results["breakdown"].items()]
            )
            if i == self._cfg.run.n_epochs - 1 and save_model:
                if plot_filename.endswith(".png"):
                    plot_filename = plot_filename[:-4]
                if ckpt_name is not None:
                    ckpt_name = plot_filename
                self.save_model(results["obj"].item(), f"./checkpoint/{ckpt_name}.pt")
            
            if verbose:
                logger.info(log)
            # update the learning rate
            self.lr_scheduler.step()
            # update the sharpness
            self.sharp_scheduler.step()

            if plot:
                if plot_filename.endswith(".png"):
                    plot_filename = plot_filename[:-4]
                for j in range(len(objs)):
                    # (port_name, wl, mode, temp), extract pol from mode, e.g., Ez1 -> Ez
                    pol = field_keys[j][2][:2]
                    self.devOptimization.plot(
                        eps_map=self.devOptimization._eps_map,
                        obj=results["breakdown"][objs[j]]["value"],
                        plot_filename=plot_filename + f"_{i}" + f"_{objs[j]}.jpg",
                        field_key=field_keys[j],
                        # field_component=pol,
                        field_component=field_component if field_component is not None else pol,
                        in_slice_name=in_slice_names[j],
                        exclude_slice_names=exclude_slice_names[j],
                    )
                    # self.plot_thread.submit(
                    #     self.devOptimization.plot,
                    #     eps_map=self.devOptimization._eps_map,
                    #     obj=results["breakdown"][objs[j]]["value"],
                    #     plot_filename=plot_filename + f"_{i}" + f"_{objs[j]}.jpg",
                    #     # field_key=("in_port_1", 1.55, 1),
                    #     field_key=field_keys[j],
                    #     field_component=pol,
                    #     # in_port_name="in_port_1",
                    #     in_slice_name=in_slice_names[j],
                    #     # exclude_port_names=["refl_port_2"],
                    #     exclude_slice_names=exclude_slice_names[j],
                    # )

        if dump_gds:
            if plot_filename.endswith(".png"):
                plot_filename = plot_filename[:-4]
            self.devOptimization.dump_gds_files(plot_filename + ".gds")

    def save_model(self, fom, path):
        self.saver.save_model(
            self.devOptimization,
            fom,
            epoch=self._cfg.run.n_epochs,
            path=path,
            save_model=False,
            print_msg=True,
        )


if __name__ == "__main__":
    gpu_id = 1
    torch.cuda.set_device(gpu_id)
    operation_device = torch.device("cuda:" + str(gpu_id))
    torch.backends.cudnn.benchmark = True
    set_torch_deterministic(int(41 + 500))
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
            resolution=100,
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
        port_width=(input_port_width, output_port_width),
        device=operation_device,
    )

    hr_device = device.copy(resolution=310)
    print(device)
    opt = BendingOptimization(
        device=device,
        hr_device=hr_device,
        sim_cfg=sim_cfg,
        operation_device=operation_device,
    ).to(operation_device)
    invdesign = InvDesign(devOptimization=opt)
    invdesign.optimize()
