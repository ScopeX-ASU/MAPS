"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
"""

import gc
import os
import sys
import traceback
from typing import Any, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../MAPS"))
sys.path.insert(0, project_root)
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from pyutils.config import Config
from pyutils.general import logger
from pyutils.torch_train import BestKModelSaver

from core.invdes import builder
from core.invdes.models import BendingOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import Bending
from core.utils import set_torch_deterministic

warnings.filterwarnings("ignore", category=FutureWarning)


def normalize_grad(g, mode="p95", eps=1e-12):
    if mode == "abs_mean":
        scale = g.abs().mean()
        return g / scale.clamp_min(eps)

    if mode == "rms":
        scale = g.square().mean().sqrt()
        return g / scale.clamp_min(eps)

    if mode == "p95":
        scale = torch.quantile(g.abs().flatten(), 0.95)
        return torch.clamp(g / scale.clamp_min(eps), -1.0, 1.0)

    raise ValueError(mode)


def clamp_parameters(invdes, min_val=-0.5, max_val=0.5):
    with torch.no_grad():
        for param in invdes.devOptimization.parameters():
            param.data.clamp_(min_val, max_val)


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
            start_epoch=0,
            n_epochs=100,
            norm_grad=None,
        ),
        plot_cfgs=Config(
            plot=False,
            interval=5,
            plot_name=None,
            objs=[],
            field_keys=[],
            in_slice_names=[],
            exclude_slice_names=[],
            thermal_map_names=[],
            field_component=None,
            eps_grad=False,
            param_grad=False,
        ),
        debug_cfgs=Config(
            log_cuda_memory=False,
            empty_cache=False,
            clear_step_results=True,
            synchronize_cuda=False,
        ),
        checkpoint_cfgs=Config(
            save_model=False,
            ckpt_name=None,
            interval=None,
            dump_gds=False,
            gds_name=None,
            dump_eps=False,
            eps_name=None,
            upsample_eps_to_1nm=False,
            load_ckpt=None,
            resume=False,
        ),
        after_step_callbacks=[clamp_parameters],
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
            k=1,
            descend=False,
            truncate=10,
            metric_name="err",
            format="{:.4f}",
        )
        self.after_step_callbacks = self._cfg.after_step_callbacks

        ## closure is a function that will be called by the optimizer
        class Closure(object):
            def __init__(
                self,
                optimizer,  # optimizer
                devOptimization,  # device optimization model,
                cfg=None,
            ):
                self.results = None
                self.optimizer = optimizer
                self.devOptimization = devOptimization
                self._cfg = cfg
                self.sharpness = 1

            def __call__(self):
                # clear grad here
                self.optimizer.zero_grad(set_to_none=False)
                # forward pass
                results = self.devOptimization.forward(sharpness=self.sharpness)

                # need backward to compute grad
                (-results["obj"]).backward()

                if self._cfg["run"].norm_grad is not None:
                    for p in self.devOptimization.parameters():
                        if p.grad is not None:
                            p.grad.copy_(
                                normalize_grad(p.grad, mode=self._cfg["run"].norm_grad)
                            )

                # store any results for plot/log
                self.results = results

                ## return the loss for gradient descent
                return -results["obj"]

        self.closure = Closure(
            optimizer=self.optimizer,
            devOptimization=self.devOptimization,
            cfg=self._cfg,
        )

        self.global_step = self._cfg.run.start_epoch
        self._log = ""

        if (
            self._cfg.checkpoint_cfgs.resume
            and self._cfg.checkpoint_cfgs.load_ckpt is not None
        ):
            self.load_model(self._cfg.checkpoint_cfgs.load_ckpt)

    def _cuda_memory_stats(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        if getattr(self._cfg.debug_cfgs, "synchronize_cuda", False):
            torch.cuda.synchronize()
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        }

    def _log_cuda_memory(self, tag: str) -> None:
        if not getattr(self._cfg.debug_cfgs, "log_cuda_memory", False):
            return
        stats = self._cuda_memory_stats()
        if not stats:
            return
        logger.info(
            f"[cuda_mem:{tag}] "
            f"allocated={stats['allocated_mb']:.1f}MB "
            f"reserved={stats['reserved_mb']:.1f}MB "
            f"max_allocated={stats['max_allocated_mb']:.1f}MB "
            f"max_reserved={stats['max_reserved_mb']:.1f}MB"
        )

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
            assert len(plot_cfgs.in_slice_names) > 0, "in_port_names must be provided"
            if len(plot_cfgs.exclude_slice_names) == 0:
                plot_cfgs.exclude_slice_names = [[]] * len(plot_cfgs.objs)
            if len(plot_cfgs.thermal_map_names) == 0:
                plot_cfgs.thermal_map_names = [None] * len(plot_cfgs.objs)

        ckpt_cfgs = self._cfg.checkpoint_cfgs
        if ckpt_cfgs.save_model:
            assert (
                ckpt_cfgs.ckpt_name is not None
            ), "ckpt_name must be provided if save model"
        if ckpt_cfgs.dump_gds:
            assert (
                ckpt_cfgs.gds_name is not None
            ), "gds_name must be provided if dump gds"

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

        self._log_cuda_memory("before_step")

        self.optimizer.step(self.closure)
        # clip parameter to -0.5, 0.5
        # with torch.no_grad():
        #     for param in self.devOptimization.parameters():
        #         param.clamp_(-0.5, 0.5)

        if self.after_step_callbacks is not None:
            for callback in self.after_step_callbacks:
                callback(self)
        # print(self.devOptimization._eps_map.grad.norm(), self.devOptimization.parameters().__next__().grad.norm())
        results = self.closure.results
        self.results = results  # record this result
        self._log_cuda_memory("after_step")
        return results

    def after_step(self, output_dict: Dict[str, Any] = {}) -> None:
        # update the learning rate
        self.lr_scheduler.step()
        # update the sharpness
        self.sharp_scheduler.step()

        ## plot
        i = self.global_step
        plot_cfgs = self._cfg.plot_cfgs
        if plot_cfgs.plot and (
            i % plot_cfgs.interval == 0
            or i == self._cfg.run.start_epoch + self._cfg.run.n_epochs - 1
        ):
            plot_filename = plot_cfgs.plot_name
            plot_filename_suffixes = plot_cfgs.get(
                "filename_suffixes", [""] * len(plot_cfgs.objs)
            )
            thermal_map_names = plot_cfgs.get(
                "thermal_map_names", [None] * len(plot_cfgs.objs)
            )
            if len(thermal_map_names) < len(plot_cfgs.objs):
                thermal_map_names = thermal_map_names + [None] * (
                    len(plot_cfgs.objs) - len(thermal_map_names)
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
                    obj=output_dict["breakdown"][plot_cfgs.objs[j]]["value"],
                    plot_filename=plot_filename
                    + f"_{i}"
                    + f"_{plot_cfgs.objs[j]}{suffix}.png",
                    field_key=plot_cfgs.field_keys[j],
                    # field_component=pol,
                    field_component=(
                        plot_cfgs.field_component
                        if plot_cfgs.field_component is not None
                        else pol
                    ),
                    in_slice_name=plot_cfgs.in_slice_names[j],
                    exclude_slice_names=plot_cfgs.exclude_slice_names[j],
                    thermal_map_name=thermal_map_names[j],
                    show_delta_eps=plot_cfgs.get("show_delta_eps", None),
                    eps_grad=plot_cfgs.get("eps_grad", False),
                    param_grad=plot_cfgs.get("param_grad", False),
                )
                if not hasattr(self, "plot_thread") or self.plot_thread is None:
                    self.devOptimization.plot(
                        **plot_kwargs,
                    )
                else:
                    self.plot_thread.submit(self.devOptimization.plot, **plot_kwargs)

        if getattr(self._cfg.debug_cfgs, "clear_step_results", True):
            self.closure.results = None
            self.results = None
            if hasattr(self.devOptimization, "_eps_map") and isinstance(
                getattr(self.devOptimization, "_eps_map"), torch.Tensor
            ):
                self.devOptimization._eps_map.grad = None
            gc.collect()
            if torch.cuda.is_available() and getattr(
                self._cfg.debug_cfgs, "empty_cache", False
            ):
                torch.cuda.empty_cache()
        self._log_cuda_memory("after_clear")

    def after_epoch(self, output_dict: Dict[str, Any] = {}) -> None:
        ## save model
        i = self.global_step
        try:
            if self._cfg.checkpoint_cfgs.save_model:
                ckpt_name = self._cfg.checkpoint_cfgs.ckpt_name
                if not ckpt_name.endswith(".pt"):
                    ckpt_name += f"_epoch-{i}.pt"
                else:
                    ckpt_name = ckpt_name[:-3] + f"_epoch-{i}.pt"
                path = os.path.join(self.devOptimization.sim_cfg.plot_root, ckpt_name)
                self.save_model(output_dict["obj"].item(), path)
                logger.info(f"save model to {path}")
        except Exception as e:
            logger.error("save model failed")
            traceback.print_exc()

        ## dump the full high resolution eps map into image
        try:
            if getattr(self._cfg.checkpoint_cfgs, "dump_eps", True):  ## hardcode
                self._dump_eps_image(self._cfg.checkpoint_cfgs, i)
        except Exception:
            logger.error("dump high-res eps map failed")
            traceback.print_exc()

        ## dump gds
        try:
            if self._cfg.checkpoint_cfgs.dump_gds:
                self.devOptimization.dump_gds_files(
                    self._cfg.checkpoint_cfgs.gds_name + ".gds"
                )
        except Exception as e:
            logger.error("dump gds failed")
            traceback.print_exc()

    def dump_intermediate_artifacts(self) -> None:
        ckpt_cfgs = self._cfg.checkpoint_cfgs
        interval = getattr(ckpt_cfgs, "interval", None)
        if interval is None:
            interval = getattr(self._cfg.plot_cfgs, "interval", None)
        if not interval or interval <= 0:
            return

        i = self.global_step
        if (
            i == self._cfg.run.start_epoch + self._cfg.run.n_epochs - 1
            or i % interval != 0
        ):
            return

        plot_root = self.devOptimization.sim_cfg.plot_root

        try:
            if getattr(ckpt_cfgs, "dump_eps", True):
                self._dump_eps_image(ckpt_cfgs, i)
        except Exception:
            logger.error("dump intermediate high-res eps map failed")
            traceback.print_exc()

        try:
            if ckpt_cfgs.dump_gds:
                self.devOptimization.dump_gds_files(
                    f"{ckpt_cfgs.gds_name}_epoch-{i}.gds"
                )
        except Exception:
            logger.error("dump intermediate gds failed")
            traceback.print_exc()

    def optimize(
        self,
        verbose: bool = True,
        dump_intermediate_result: bool = False,
    ):
        for i in range(
            self._cfg.run.start_epoch,
            self._cfg.run.start_epoch + self._cfg.run.n_epochs,
        ):
            self.global_step = i
            feed_dict = self.before_step()
            results = self.run_step(feed_dict)
            self.after_step(results)
            if dump_intermediate_result:
                self.dump_intermediate_artifacts()

            log = f"Step {i:3d} (sharp: {feed_dict['sharpness']:.1f}) "
            log += ", ".join(
                [
                    (
                        (
                            f"{k}: {obj['value'].data}"
                            if obj["value"].numel() > 1
                            else f"{k}: {obj['value'].item():.3e}"
                        )
                        if isinstance(obj["value"], torch.Tensor)
                        and obj["value"].numel() >= 1
                        else f"{k}: {obj['value']:.3e}"
                    )
                    for k, obj in results["breakdown"].items()
                ]
            )
            total_grads = torch.concat(
                [
                    param.grad.view(-1)
                    for param in self.devOptimization.parameters()
                    if param.requires_grad and param.grad is not None
                ],
                dim=0,
            )
            grad_norm = torch.norm(total_grads).item()
            log += f", grad_norm: {grad_norm:.3e}"
            log += self._log
            if verbose:
                logger.info(log)
        self.after_epoch(results)

    def load_model(self, path=None):
        state_dict = torch.load(path)
        self.devOptimization.load_state_dict(state_dict["state_dict"])
        logger.info(f"load checkpoint from {path}")

    def save_model(self, fom, path):
        self.saver.save_model(
            self.devOptimization,
            fom,
            epoch=self._cfg.run.start_epoch + self._cfg.run.n_epochs - 1,
            path=path,
            save_model=False,
            print_msg=True,
        )

    def _get_eps_dump_map(self, ckpt_cfgs):
        eps_map = getattr(self.devOptimization, "hr_eps_map", None)
        if eps_map is None:
            raise ValueError("high resolution eps map is not available")
        if isinstance(eps_map, torch.Tensor):
            eps_map = eps_map.detach().cpu().numpy()
        eps = np.asarray(eps_map, dtype=np.float64)
        if eps.size == 0:
            raise ValueError("eps map is empty")

        hr_device = self.devOptimization.hr_device
        hr_design_region_masks = hr_device.design_region_masks
        if len(hr_design_region_masks) != 1:
            raise ValueError("Only support one design region for now")

        if not getattr(ckpt_cfgs, "upsample_eps_to_1nm", False):
            return eps, hr_design_region_masks

        target_resolution = 1000
        if int(round(hr_device.sim_cfg["resolution"])) == target_resolution:
            return eps, hr_design_region_masks

        region_name, src_mask = next(iter(hr_design_region_masks.items()))
        export_device = getattr(self, "_cached_eps_dump_1nm_device", None)
        if (
            export_device is None
            or int(round(export_device.sim_cfg["resolution"])) != target_resolution
        ):
            export_device = hr_device.copy(resolution=target_resolution)
            self._cached_eps_dump_1nm_device = export_device

        target_mask = export_device.design_region_masks[region_name]
        target_size = (
            target_mask.x.stop - target_mask.x.start,
            target_mask.y.stop - target_mask.y.start,
        )
        if min(target_size) <= 0:
            raise ValueError(f"Invalid upsampled design region size: {target_size}")

        design_region_eps = (
            torch.as_tensor(eps[src_mask], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        upsampled_design_region_eps = (
            torch.nn.functional.interpolate(
                design_region_eps,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        export_eps = np.asarray(export_device.epsilon_map, dtype=np.float64).copy()
        export_eps[target_mask] = upsampled_design_region_eps
        return export_eps, export_device.design_region_masks

    def _dump_eps_image(self, ckpt_cfgs, epoch: int):
        name = (
            ckpt_cfgs.eps_name or ckpt_cfgs.gds_name or ckpt_cfgs.ckpt_name or "eps_map"
        )
        plot_root = self.devOptimization.sim_cfg.plot_root
        filename = os.path.join(plot_root, f"{name}_hr_eps_epoch-{epoch}.png")
        eps, design_region_masks = self._get_eps_dump_map(ckpt_cfgs)
        design_region_eps = eps[next(iter(design_region_masks.values()))]
        eps_min = float(np.nanmin(design_region_eps))
        eps_max = float(np.nanmax(design_region_eps))
        if not np.isfinite(eps_min) or not np.isfinite(eps_max):
            raise ValueError("eps map contains NaN/Inf")
        if eps_max == eps_min:
            bw = np.zeros_like(eps, dtype=np.uint8)
        else:
            threshold = 0.5 * (eps_min + eps_max)
            bw = (eps >= threshold).astype(np.uint8) * 255
        Image.fromarray(bw, mode="L").save(filename)


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
