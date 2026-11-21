"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
"""

import gc
import os
import random
import sys
import tempfile
import traceback
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
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
from core.invdes.run_management import OptimizationRunManager
from core.utils import set_torch_deterministic

warnings.filterwarnings("ignore", category=FutureWarning)


def normalize_grad(g, mode="p95", eps=1e-12):
    """
    Normalize a gradient tensor according to a selectable scale statistic.

    Args:
        g: PyTorch gradient tensor; shape, dtype, and device are preserved.
        mode: ``abs_mean``, ``rms``, or ``p95``. ``p95`` clips the result to ``[-1, 1]``.
        eps: Positive lower bound for the normalization scale.

    Returns:
        Tensor with the same shape, dtype, and device as ``g``.

    Example:
        ``normalize_grad(torch.tensor([-2.0, 0.0, 4.0]), mode="rms")`` returns shape ``[3]``.
    """
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
    """
    Clamp all learnable inverse-design parameters in-place.

    Args:
        invdes: ``InvDesign`` instance whose ``devOptimization`` owns the learnable parameters.
        min_val: Inclusive lower bound.
        max_val: Inclusive upper bound.

    Returns:
        None; parameters are modified in-place under ``torch.no_grad()``.
    """
    with torch.no_grad():
        for param in invdes.devOptimization.parameters():
            param.data.clamp_(min_val, max_val)


def clamp_material_parameters(invdes, min_val=0.0, max_val=1.0):
    """Clamp normalized digital-material parameters after an optimizer step.

    Args:
        invdes: ``InvDesign`` instance whose device-optimization model owns the
            material parameters. Modules exposing a ``material_rho`` tensor are
            clamped; unrelated geometry and model parameters are unchanged.
        min_val: Inclusive lower bound. Defaults to ``0.0``.
        max_val: Inclusive upper bound. Defaults to ``1.0``.

    Returns:
        None; material parameters are modified in-place under ``torch.no_grad()``.

    Raises:
        ValueError: If ``min_val`` is greater than ``max_val``.
    """
    if min_val > max_val:
        raise ValueError("min_val must not exceed max_val")

    with torch.no_grad():
        for module in invdes.devOptimization.modules():
            material_rho = getattr(module, "material_rho", None)
            if isinstance(material_rho, torch.Tensor):
                material_rho.clamp_(min_val, max_val)

    # for name, parameter in invdes.devOptimization.named_parameters():
    #     if "material_rho" in name:
    #         grad_max = (
    #             parameter.grad.abs().max().item()
    #             if parameter.grad is not None
    #             else None
    #         )
    #         print(
    #             name,
    #             parameter.min().item(),
    #             parameter.max().item(),
    #             grad_max,
    #         )


class InvDesign:
    """
    Orchestrate iterative photonic inverse-design optimization.

    Args:
        *args: Reserved for compatibility.
        **kwargs: Configuration overrides including ``devOptimization``, ``optimizer``, ``lr_scheduler``, ``sharp_scheduler``, ``run``, ``plot_cfgs``, ``debug_cfgs``, ``checkpoint_cfgs``, and lifecycle callbacks.

    Optimization convention:
        ``devOptimization.forward()`` returns a result dictionary containing scalar ``results["obj"]``. The optimizer minimizes ``-results["obj"]``.

    Example:
        ``InvDesign(devOptimization=model, run={"n_epochs": 10}).optimize()``.
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
            backward=True,
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
            monitor_plots=[],
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
            save_latest=True,
            latest_name=None,
            save_on_interrupt=True,
            dump_gds=False,
            gds_name=None,
            dump_eps=False,
            eps_name=None,
            upsample_eps_to_1nm=False,
            dump_tidy3d=False,
            tidy3d_name=None,
            load_ckpt=None,
            resume=False,
        ),
        run_management=Config(
            enabled=False,
            root="./runs",
            experiment="optimization",
            run_id=None,
            tag=None,
        ),
        after_step_callbacks=[clamp_parameters],
        before_step_callbacks=[],
    )

    def __init__(
        self,
        *args,
        run_manager: Optional[OptimizationRunManager] = None,
        **kwargs,
    ) -> None:
        """Initialize an inverse-design run.

        Args:
            *args: Reserved positional arguments retained for constructor
                compatibility.
            run_manager: Optional run manager created before device/model
                construction. Supplying it lets initialization-time plots and
                logs use the same managed run directories as optimization.
            **kwargs: Named configuration values forwarded to
                :meth:`load_cfgs`.
        """
        super().__init__()
        self.load_cfgs(**kwargs)
        assert self.devOptimization is not None, "devOptimization must be provided"
        self.run_manager = run_manager
        run_cfg = self._cfg.run_management
        if self.run_manager is not None:
            # Keep the resolved configuration truthful without requiring the
            # caller to duplicate settings already owned by the early manager.
            run_cfg.enabled = True
            run_cfg.root = str(self.run_manager.root)
            run_cfg.experiment = self.run_manager.experiment
            run_cfg.run_id = self.run_manager.run_id
            run_cfg.tag = self.run_manager.tag
        if self.run_manager is None and getattr(run_cfg, "enabled", False):
            resume_run = bool(getattr(self._cfg.checkpoint_cfgs, "resume", False))
            self.run_manager = OptimizationRunManager(
                root=run_cfg.root,
                experiment=run_cfg.experiment,
                run_id=getattr(run_cfg, "run_id", None),
                tag=getattr(run_cfg, "tag", None),
                resume=resume_run,
                resume_from=(
                    self._cfg.checkpoint_cfgs.load_ckpt if resume_run else None
                ),
            )
        if self.run_manager is not None:
            self.run_manager.attach_logger(logger)
            logger.info(f"managed optimization run: {self.run_manager.run_dir}")
            self.run_manager.bind_simulation_config(self.devOptimization.sim_cfg)
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
        self.before_step_callbacks = self._cfg.before_step_callbacks

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

                ## Backward is optional for forward-only normalization runs.
                if self._cfg["run"].get("backward", True):
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
        self._resume_step = self._cfg.run.start_epoch
        self._last_completed_step = self._cfg.run.start_epoch - 1
        self._last_fom = None
        self._log = ""

        if self.run_manager is not None:
            resolved_config = {
                "invdesign": {
                    key: self._plain_config(value)
                    for key, value in self._cfg.items()
                    if key != "devOptimization"
                },
                "dev_optimization": {
                    "class": (
                        f"{self.devOptimization.__class__.__module__}."
                        f"{self.devOptimization.__class__.__qualname__}"
                    ),
                    "config": self._plain_config(
                        getattr(self.devOptimization, "_cfg", {})
                    ),
                },
            }
            self.run_manager.write_resolved_config(resolved_config)

        if (
            self._cfg.checkpoint_cfgs.resume
            and self._cfg.checkpoint_cfgs.load_ckpt is not None
        ):
            self.load_model(self._cfg.checkpoint_cfgs.load_ckpt)

    def _artifact_path(self, category: str, filename: str) -> str:
        """Resolve an artifact path through the run manager or legacy root."""
        if self.run_manager is not None:
            return self.run_manager.path(category, filename)
        return os.path.join(self.devOptimization.sim_cfg.plot_root, filename)

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
        """
        Merge user configuration overrides into the active InvDesign configuration.

        Args:
            **cfgs: Top-level configuration overrides merged into ``default_cfgs``.

        Raises:
            AssertionError: If plotting/checkpoint configuration is enabled without required fields.
        """
        active_cfgs = deepcopy(self.default_cfgs)
        active_cfgs.update(cfgs)
        self.__dict__.update(active_cfgs)
        self._cfg = active_cfgs

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
        for callback in self.before_step_callbacks:
            feed_dict = callback(self, feed_dict)
        return feed_dict

    def before_step(self) -> Dict[str, Any]:
        """
        Prepare per-step optimization controls and execute pre-step callbacks.

        Returns:
            Dictionary containing at least ``sharpness`` from the sharpness scheduler; callbacks may add or modify entries before ``run_step()``.
        """
        self._log = ""  # reset log
        sharpness = self.sharp_scheduler.get_sharpness()
        feed_dict = dict(
            sharpness=sharpness,
        )
        feed_dict = self._before_step_callbacks(feed_dict)
        return feed_dict

    def run_step(self, feed_dict: Dict[str, Any] = {}):
        """
        Execute one optimizer step using the configured inverse-design closure.

        Args:
            feed_dict: Per-step control dictionary from ``before_step()``; the current implementation requires ``sharpness``.

        Returns:
            Result dictionary produced by ``devOptimization.forward()`` and stored as ``self.results``.
        """
        sharpness = feed_dict["sharpness"]
        self.closure.sharpness = sharpness
        variation = getattr(self.devOptimization, "variation", None)
        if variation is not None and variation.sampler.enabled:
            variation.step = self.global_step

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
        """
        Advance schedulers, optionally plot fields/objectives, and release per-step state.

        Args:
            output_dict: Result dictionary from the completed step.

        Side Effects:
            Advances learning-rate and sharpness schedulers, may generate plots, clears cached result tensors and gradients, and may empty the CUDA cache.
        """
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
                    objective_name=plot_cfgs.objs[j],
                    monitor_plots=plot_cfgs.get("monitor_plots", []),
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

    def after_epoch(self, output_dict: Optional[Dict[str, Any]] = None) -> None:
        """Write final optimization artifacts.

        Args:
            output_dict: Final step output. When omitted, state-derived
                artifacts are still regenerated, but the objective-ranked
                model checkpoint is skipped. This supports repairing missing
                artifacts from an already-completed resumable checkpoint
                without rerunning an optimization step.
        """
        output_dict = output_dict or {}
        ## save model
        i = self.global_step
        if not output_dict:
            # A full checkpoint stores parameters, not derived material maps.
            # Rebuild them once so completed-run recovery can also regenerate
            # GDS and epsilon artifacts without a forward/FDTD solve.
            build_device = getattr(self.devOptimization, "build_device", None)
            if callable(build_device) and any(
                (
                    getattr(self._cfg.checkpoint_cfgs, "dump_eps", True),
                    getattr(self._cfg.checkpoint_cfgs, "dump_gds", False),
                    getattr(self._cfg.checkpoint_cfgs, "dump_tidy3d", False),
                )
            ):
                rebuilt = build_device(sharpness=float(self.closure.sharpness))
                if len(rebuilt) < 3:
                    raise ValueError(
                        "build_device() must return a high-resolution map at index 2"
                    )
                self.devOptimization._hr_eps_map = rebuilt[2]
        try:
            if self._cfg.checkpoint_cfgs.save_model and "obj" in output_dict:
                ckpt_name = self._cfg.checkpoint_cfgs.ckpt_name
                if not ckpt_name.endswith(".pt"):
                    ckpt_name += f"_epoch-{i}.pt"
                else:
                    ckpt_name = ckpt_name[:-3] + f"_epoch-{i}.pt"
                path = self._artifact_path("checkpoints", ckpt_name)
                saved_path = self.save_model(output_dict["obj"].item(), path)
                if self.run_manager is not None and saved_path is not None:
                    self.run_manager.register_artifact(
                        "best_checkpoint", saved_path, checksum=True
                    )
                logger.info(f"save model to {saved_path or path}")
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
                gds_path = self._artifact_path(
                    "layout", self._cfg.checkpoint_cfgs.gds_name + ".gds"
                )
                self.devOptimization.dump_gds_files(gds_path)
                if self.run_manager is not None:
                    self.run_manager.register_artifact(
                        "final_gds", gds_path, checksum=True
                    )
        except Exception as e:
            logger.error("dump gds failed")
            traceback.print_exc()

        ## dump a self-contained Tidy3D geometry artifact
        try:
            if getattr(self._cfg.checkpoint_cfgs, "dump_tidy3d", False):
                path = self._dump_tidy3d_geometry(self._cfg.checkpoint_cfgs)
                logger.info(f"save Tidy3D geometry to {path}")
        except Exception:
            logger.error("dump Tidy3D geometry failed")
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

        try:
            if getattr(ckpt_cfgs, "dump_eps", True):
                self._dump_eps_image(ckpt_cfgs, i)
        except Exception:
            logger.error("dump intermediate high-res eps map failed")
            traceback.print_exc()

        try:
            if ckpt_cfgs.dump_gds:
                self.devOptimization.dump_gds_files(
                    self._artifact_path("layout", f"{ckpt_cfgs.gds_name}_epoch-{i}.gds")
                )
        except Exception:
            logger.error("dump intermediate gds failed")
            traceback.print_exc()

    def _checkpoint_base_name(self) -> str:
        """Return the configured checkpoint stem without a ``.pt`` suffix."""
        ckpt_cfgs = self._cfg.checkpoint_cfgs
        latest_name = getattr(ckpt_cfgs, "latest_name", None)
        name = latest_name or ckpt_cfgs.ckpt_name
        if name.endswith(".pt"):
            name = name[:-3]
        return name

    def _dump_tidy3d_geometry(self, ckpt_cfgs) -> str:
        """Materialize final parameters and save a Tidy3D geometry artifact.

        Args:
            ckpt_cfgs: Checkpoint configuration containing optional
                ``tidy3d_name`` and the other artifact-name fallbacks.

        Returns:
            Absolute path returned by
            ``devOptimization.save_tidy3d_geometry``.

        Notes:
            The map is rebuilt because the most recently cached forward map
            predates the optimizer update. The last sharpness supplied to the
            closure is used rather than the already-advanced scheduler value.
        """
        build_device = getattr(self.devOptimization, "build_device", None)
        save_geometry = getattr(self.devOptimization, "save_tidy3d_geometry", None)
        if not callable(build_device) or not callable(save_geometry):
            raise TypeError(
                "dump_tidy3d requires devOptimization.build_device() and "
                "devOptimization.save_tidy3d_geometry()"
            )

        sharpness = float(self.closure.sharpness)
        final_hr_eps_map = build_device(sharpness=sharpness)[2]
        artifact_stem = getattr(ckpt_cfgs, "tidy3d_name", None)
        if artifact_stem is None:
            base_name = (
                getattr(ckpt_cfgs, "ckpt_name", None)
                or getattr(ckpt_cfgs, "gds_name", None)
                or "final"
            )
            artifact_stem = f"{os.path.splitext(base_name)[0]}_tidy3d_geometry"
        for suffix in (".hdf5", ".h5"):
            if artifact_stem.lower().endswith(suffix):
                artifact_stem = artifact_stem[: -len(suffix)]
                break
        name = f"{artifact_stem}.h5"
        path = self._artifact_path("geometry", name)
        saved_path = save_geometry(
            path,
            eps_map=final_hr_eps_map,
            pixel_exact=False,
        )
        if self.run_manager is not None:
            self.run_manager.register_artifact(
                "final_tidy3d_geometry", saved_path, checksum=True
            )
        return saved_path

    def _save_latest_checkpoint_if_due(self) -> None:
        """Atomically update the resumable latest checkpoint when due."""
        ckpt_cfgs = self._cfg.checkpoint_cfgs
        interval = getattr(ckpt_cfgs, "interval", None)
        if (
            not ckpt_cfgs.save_model
            or not getattr(ckpt_cfgs, "save_latest", True)
            or not interval
            or interval <= 0
        ):
            return

        final_step = self._cfg.run.start_epoch + self._cfg.run.n_epochs - 1
        if self.global_step % interval != 0 and self.global_step != final_step:
            return

        path = self._artifact_path(
            "checkpoints", f"{self._checkpoint_base_name()}_latest.pt"
        )
        self.save_checkpoint(path, fom=self._last_fom)
        if self.run_manager is not None:
            self.run_manager.register_artifact("latest_checkpoint", path)
        logger.info(f"save latest checkpoint to {path}")

    def optimize(
        self,
        verbose: bool = True,
        dump_intermediate_result: bool = False,
    ):
        """
        Run the configured inverse-design optimization loop.

        Args:
            verbose: Whether to log objective breakdowns and gradient norm information.
            dump_intermediate_result: Whether to export configured intermediate artifacts.

        Lifecycle:
            Each iteration executes ``before_step() -> run_step() -> after_step()``; ``after_epoch()`` runs after the requested number of iterations.

        Returns:
            None. Final state remains in ``self`` and ``devOptimization``.
        """
        loop_start = max(int(self._cfg.run.start_epoch), int(self._resume_step))
        loop_end = int(self._cfg.run.start_epoch) + int(self._cfg.run.n_epochs)
        results = None
        step_in_progress = False
        try:
            for i in range(loop_start, loop_end):
                self.global_step = i
                step_in_progress = True
                feed_dict = self.before_step()
                results = self.run_step(feed_dict)
                self.after_step(results)
                self._last_completed_step = i
                self._last_fom = float(results["obj"].detach().cpu().item())
                self._resume_step = i + 1
                step_in_progress = False
                self._save_latest_checkpoint_if_due()
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
                gradients = [
                    param.grad.view(-1)
                    for param in self.devOptimization.parameters()
                    if param.requires_grad and param.grad is not None
                ]
                grad_norm = (
                    torch.norm(torch.cat(gradients)).item() if gradients else 0.0
                )
                log += f", grad_norm: {grad_norm:.3e}"
                log += self._log
                if self.run_manager is not None:
                    self.run_manager.append_metrics(
                        {
                            "step": i,
                            "fom": self._last_fom,
                            "sharpness": float(feed_dict["sharpness"]),
                            "learning_rates": [
                                float(group["lr"])
                                for group in self.optimizer.param_groups
                            ],
                            "grad_norm": grad_norm,
                            "breakdown": {
                                key: self._plain_config(item.get("value"))
                                for key, item in results["breakdown"].items()
                            },
                        }
                    )
                    if not verbose:
                        self.run_manager.log(log)
                if verbose:
                    logger.info(log)
        except (Exception, KeyboardInterrupt) as error:
            if (
                self._cfg.checkpoint_cfgs.save_model
                and getattr(self._cfg.checkpoint_cfgs, "save_on_interrupt", True)
                and not step_in_progress
                and self._last_completed_step >= self._cfg.run.start_epoch
            ):
                path = self._artifact_path(
                    "checkpoints",
                    f"{self._checkpoint_base_name()}_interrupted.pt",
                )
                try:
                    self.save_checkpoint(path, fom=self._last_fom)
                    if self.run_manager is not None:
                        self.run_manager.register_artifact(
                            "interrupted_checkpoint", path, checksum=True
                        )
                    logger.info(f"save interrupted checkpoint to {path}")
                except Exception:
                    logger.error("save interrupted checkpoint failed")
                    traceback.print_exc()
            if self.run_manager is not None:
                status = (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                )
                self.run_manager.finish(
                    status,
                    last_completed_step=self._last_completed_step,
                    fom=self._last_fom,
                    error=repr(error),
                )
            raise

        # A completed checkpoint has no remaining loop iterations. Rebuild its
        # final state-derived artifacts anyway so a failed or deleted export
        # can be repaired without another electromagnetic solve.
        self.after_epoch(results)
        if self.run_manager is not None:
            self.run_manager.finish(
                "completed",
                last_completed_step=self._last_completed_step,
                fom=self._last_fom,
            )

    @staticmethod
    def _component_state(component):
        """Return checkpoint state for a component when it exposes any."""
        component_class = (
            f"{component.__class__.__module__}.{component.__class__.__qualname__}"
        )
        state_dict = getattr(component, "state_dict", None)
        if callable(state_dict):
            return {
                "class": component_class,
                "kind": "state_dict",
                "state": state_dict(),
            }
        if hasattr(component, "__dict__"):
            state = {
                key: value
                for key, value in vars(component).items()
                if isinstance(value, (bool, int, float, str, type(None)))
            }
            if state:
                return {
                    "class": component_class,
                    "kind": "attributes",
                    "state": state,
                }
        return None

    @staticmethod
    def _restore_component_state(component, saved_state) -> None:
        """Restore state produced by :meth:`_component_state`."""
        if saved_state is None:
            return
        component_class = (
            f"{component.__class__.__module__}.{component.__class__.__qualname__}"
        )
        saved_class = saved_state.get("class")
        if saved_class is not None and saved_class != component_class:
            raise ValueError(
                f"Checkpoint component type mismatch: saved {saved_class}, "
                f"constructed {component_class}"
            )
        if saved_state["kind"] == "state_dict":
            component.load_state_dict(saved_state["state"])
        elif saved_state["kind"] == "attributes":
            for key, value in saved_state["state"].items():
                setattr(component, key, value)
        else:
            raise ValueError(f"Unknown component state kind: {saved_state['kind']}")

    @staticmethod
    def _plain_config(value):
        """Convert configuration values to portable checkpoint metadata."""
        if isinstance(value, Mapping):
            return {
                str(key): InvDesign._plain_config(item) for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [InvDesign._plain_config(item) for item in value]
        if isinstance(value, (bool, int, float, str, type(None))):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if callable(value):
            module = getattr(value, "__module__", "")
            name = getattr(value, "__qualname__", repr(value))
            return f"{module}.{name}".lstrip(".")
        return repr(value)

    @staticmethod
    def _rng_state() -> Dict[str, Any]:
        """Capture Python, NumPy, CPU, and available CUDA random states."""
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: Dict[str, Any]) -> None:
        """Restore random states captured by :meth:`_rng_state`."""
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def checkpoint_state_dict(self, fom=None) -> Dict[str, Any]:
        """Build the complete state required to continue an optimization.

        Args:
            fom: Optional latest scalar figure of merit.

        Returns:
            Versioned checkpoint mapping containing model, optimizer,
            schedulers, progress, random generators, configuration metadata,
            and variation state when the model exposes it.
        """
        variation = getattr(self.devOptimization, "variation", None)
        config_sections = {
            name: self._plain_config(getattr(self._cfg, name))
            for name in (
                "optimizer",
                "lr_scheduler",
                "sharp_scheduler",
                "run",
                "checkpoint_cfgs",
                "before_step_callbacks",
                "after_step_callbacks",
            )
            if hasattr(self._cfg, name)
        }
        return {
            "checkpoint_schema": "maps.invdesign.checkpoint/v1",
            "state_dict": self.devOptimization.state_dict(),
            "optimizer": self._component_state(self.optimizer),
            "lr_scheduler": self._component_state(self.lr_scheduler),
            "sharp_scheduler": self._component_state(self.sharp_scheduler),
            "variation": (
                self._component_state(variation) if variation is not None else None
            ),
            "progress": {
                "global_step": int(self.global_step),
                "last_completed_step": int(self._last_completed_step),
                "next_step": int(self._resume_step),
                "fom": None if fom is None else float(fom),
            },
            "rng_state": self._rng_state(),
            "config": config_sections,
        }

    def save_checkpoint(self, path, fom=None) -> str:
        """Atomically save a full resumable optimization checkpoint.

        Args:
            path: Destination ``.pt`` file.
            fom: Optional latest scalar figure of merit.

        Returns:
            The destination path.
        """
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{os.path.basename(path)}.",
                suffix=".tmp",
                dir=directory,
                delete=False,
            ) as temporary_file:
                temporary_path = temporary_file.name
            torch.save(self.checkpoint_state_dict(fom=fom), temporary_path)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)
        return path

    def load_model(self, path=None, restore_rng: bool = True):
        """
        Load a previously saved inverse-design checkpoint.

        Args:
            path: Full or legacy model-only checkpoint path.
            restore_rng: Whether to restore Python, NumPy, PyTorch CPU, and
                available CUDA random-generator states from a full checkpoint.

        Returns:
            None; ``devOptimization`` is restored in-place.
        """
        checkpoint = torch.load(path, weights_only=False)
        self.devOptimization.load_state_dict(checkpoint["state_dict"])
        if checkpoint.get("checkpoint_schema") == "maps.invdesign.checkpoint/v1":
            self._restore_component_state(self.optimizer, checkpoint.get("optimizer"))
            self._restore_component_state(
                self.lr_scheduler, checkpoint.get("lr_scheduler")
            )
            self._restore_component_state(
                self.sharp_scheduler, checkpoint.get("sharp_scheduler")
            )
            variation = getattr(self.devOptimization, "variation", None)
            if variation is not None:
                self._restore_component_state(variation, checkpoint.get("variation"))

            progress = checkpoint["progress"]
            self.global_step = int(progress["global_step"])
            self._last_completed_step = int(progress["last_completed_step"])
            self._resume_step = int(progress["next_step"])
            self._last_fom = progress.get("fom")
            if restore_rng and "rng_state" in checkpoint:
                self._restore_rng_state(checkpoint["rng_state"])
            logger.info(
                f"resume full checkpoint from {path} at step {self._resume_step}"
            )
        else:
            logger.info(f"load legacy model-only checkpoint from {path}")

    def save_model(self, fom, path):
        """
        Save the current inverse-design model with ``BestKModelSaver``.

        Args:
            fom: Scalar figure-of-merit used for checkpoint ranking.
            path: Destination checkpoint path.

        Returns:
            Actual checkpoint path returned by ``BestKModelSaver``, or
            ``None`` when the current FoM is not retained.
        """
        return self.saver.save_model(
            self.devOptimization,
            fom,
            epoch=self._cfg.run.start_epoch + self._cfg.run.n_epochs - 1,
            path=path,
            other_params=self.checkpoint_state_dict(fom=fom),
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
        filename = self._artifact_path("design", f"{name}_hr_eps_epoch-{epoch}.png")
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
        if self.run_manager is not None:
            self.run_manager.register_artifact("final_eps_image", filename)
        return filename


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
