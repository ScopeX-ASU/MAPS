"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
"""

import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import torch
from pyutils.config import Config
from pyutils.general import logger

from core.invdes import builder
from core.invdes.models import BendingOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import Bending
from core.utils import BestKModelSaver, set_torch_deterministic


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
        plot_cfgs=Config(
            plot=False,
            interval=5,
            plot_name=None,
            objs=[],
            field_keys=[],
            in_slice_names=[],
            exclude_slice_names=[],
            field_component=None,
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            dump_gds=False,
            save_best_model_k=1,
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
        self.plot_thread = None  # ThreadPoolExecutor(2)
        self.saver = BestKModelSaver(
            k=int(self._cfg.checkpoint_cfgs.save_best_model_k),
            descend=True,
            truncate=10,
            metric_name="FoM",
            format="{:.4f}",
        )

        ## closure is a function that will be called by the optimizer
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

        self.closure = Closure(
            optimizer=self.optimizer,
            devOptimization=self.devOptimization,
        )

        self.global_step = 0
        self._log = ""

    def load_cfgs(self, **cfgs):
        # Start with default configurations
        self.__dict__.update(self.default_cfgs)
        # Update with provided configurations
        self.__dict__.update(cfgs)
        # Save the updated configurations
        self.default_cfgs.update(cfgs)
        self._cfg = self.default_cfgs

        ## check cfgs
        plot_cfgs = self._cfg.plot_cfgs
        if plot_cfgs.plot:
            assert (
                plot_cfgs.plot_name is not None
            ), "plot_name (filename) must be provided if plot"
            assert len(plot_cfgs.objs) > 0, "objs must be provided"
            assert len(plot_cfgs.field_keys) > 0, "field_keys must be provided"
            plot_cfgs.field_keys = [tuple(fk) for fk in plot_cfgs.field_keys]
            assert len(plot_cfgs.in_slice_names) > 0, "in_port_names must be provided"
            if len(plot_cfgs.exclude_slice_names) == 0:
                plot_cfgs.exclude_slice_names = [[]] * len(plot_cfgs.objs)

    def _before_step_callbacks(self, feed_dict) -> Dict[str, Any]:
        return feed_dict

    def before_step(self) -> Dict[str, Any]:
        self._log = ""  # reset log
        sharpness = self.sharp_scheduler.get_sharpness()
        feed_dict = dict(
            sharpness=sharpness,
        )
        feed_dict = self._before_step_callbacks(feed_dict)
        return feed_dict

    def run_step(self, feed_dict: Dict[str, Any] = {}):
        sharpness = feed_dict["sharpness"]
        self.closure.sharpness = sharpness

        self.optimizer.step(self.closure)
        results = self.closure.results
        self.results = results  # record this result
        return results

    def _plot_callback(self, output_dict: Dict[str, Any] = {}) -> None:
        i = self.global_step
        plot_cfgs = self._cfg.plot_cfgs
        if plot_cfgs.plot and (
            i % plot_cfgs.interval == 0 or i == self._cfg.run.n_epochs - 1
        ):
            plot_filename = plot_cfgs.plot_name
            plot_filename_suffixes = plot_cfgs.get(
                "filename_suffixes", [""] * len(plot_cfgs.objs)
            )
            if plot_filename.endswith(".png"):
                plot_filename = plot_filename[:-4]
            for j in range(len(plot_cfgs.objs)):
                # (port_name, wl, mode, temp), extract pol from mode, e.g., Ez1 -> Ez
                pol = plot_cfgs.field_keys[j][2][:2]
                suffix = plot_filename_suffixes[j]
                if suffix != "":
                    suffix = "_" + suffix
                plot_kwargs = dict(
                    eps_map=self.devOptimization._eps_map,
                    obj=output_dict["breakdown"][plot_cfgs.objs[j]]["value"],
                    plot_filename=plot_filename
                    + f"_{i}"
                    + f"_{plot_cfgs.objs[j]}{suffix}.jpg",
                    field_key=plot_cfgs.field_keys[j],
                    field_component=(
                        plot_cfgs.field_component
                        if plot_cfgs.field_component is not None
                        else pol
                    ),
                    in_slice_name=plot_cfgs.in_slice_names[j],
                    exclude_slice_names=plot_cfgs.exclude_slice_names[j],
                )
                if not hasattr(self, "plot_thread") or self.plot_thread is None:
                    self.devOptimization.plot(
                        **plot_kwargs,
                    )
                else:
                    self.plot_thread.submit(self.devOptimization.plot, **plot_kwargs)

    def _save_model_callback(self, output_dict: Dict[str, Any] = {}) -> None:
        try:
            if self._cfg.checkpoint_cfgs.save_model:
                model_name = self.devOptimization.__class__.__name__
                ckpt_path = os.path.join(
                    self._cfg.plot_cfgs.root,
                    self._cfg.plot_cfgs.dir_name,
                    "checkpoint",
                    f"{model_name}_{self._cfg.checkpoint_cfgs.model_comment}.pt",
                )
                saved_path, del_path = self.save_model(
                    output_dict["obj"].item(), ckpt_path
                )
                try:
                    if self._cfg.checkpoint_cfgs.dump_gds and saved_path is not None:
                        gds_path = os.path.join(
                            "checkpoint", os.path.basename(saved_path)[:-3] + ".gds"
                        )
                        self.devOptimization.dump_gds_files(gds_path)
                        gds_path = os.path.join(
                            self._cfg.plot_cfgs.root,
                            self._cfg.plot_cfgs.dir_name,
                            gds_path,
                        )
                        print(f"[I] GDS dumped to {gds_path}", flush=True)

                        if del_path is not None:
                            try:
                                del_path = os.path.join(
                                    self._cfg.plot_cfgs.root,
                                    self._cfg.plot_cfgs.dir_name,
                                    "checkpoint",
                                    os.path.basename(del_path)[:-3] + ".gds",
                                )
                                os.remove(del_path)
                                print(f"[I] GDS {del_path} is removed", flush=True)
                            except Exception as e:
                                print(
                                    f"[E] Model {del_path} failed to be removed",
                                    flush=True,
                                )
                                traceback.print_exc(e)
                except Exception as e:
                    logger.error("dump gds failed")
                    traceback.print_exc()
        except Exception as e:
            logger.error("save model failed")
            traceback.print_exc()

    def after_step(self, output_dict: Dict[str, Any] = {}) -> None:
        # update the learning rate
        self.lr_scheduler.step()
        # update the sharpness
        self.sharp_scheduler.step()

        ## plot
        self._plot_callback(output_dict)

        ## save model and dump gds
        self._save_model_callback(output_dict)

    def after_epoch(self, output_dict: Dict[str, Any] = {}) -> None:
        pass

    def optimize(
        self,
        verbose: bool = True,
    ):
        for i in range(self._cfg.run.n_epochs):
            self.global_step = i
            feed_dict = self.before_step()
            results = self.run_step(feed_dict)
            self.after_step(results)

            log = f"Step {i:3d} (sharp: {feed_dict['sharpness']:.1f}) "
            log += ", ".join(
                [
                    (
                        f"{k}: {obj['value'].data}"
                        if isinstance(obj["value"], torch.Tensor)
                        and obj["value"].numel() > 1
                        else f"{k}: {obj['value']:.4f}"
                    )
                    for k, obj in results["breakdown"].items()
                ]
            )
            log += self._log
            if verbose:
                logger.info(log)
        self.after_epoch(results)

    def save_model(self, fom, path):
        saved_path, del_path = self.saver.save_model(
            self.devOptimization,
            fom,
            epoch=self.global_step,
            path=path,
            save_model=False,
            print_msg=True,
        )
        return saved_path, del_path


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
