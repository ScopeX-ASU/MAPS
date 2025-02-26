"""
this is a wrapper for the invdes module
we call use InvDesign.optimize() to optimize the inventory design
basically, this should be like the training logic like in train_NN.py
"""

import os
import sys
from copy import deepcopy
from typing import Callable

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../MAPS"))
sys.path.insert(0, project_root)
from concurrent.futures import ThreadPoolExecutor

import optuna
from pyutils.config import Config
from pyutils.general import logger
from pyutils.torch_train import BestKModelSaver
from tqdm import trange
__all__ = ["AutoTune"]


class AutoTune(object):
    """
    default_cfgs is to set the default configurations
    including optimizer, lr_scheduler, sharp_scheduler etc.
    """

    default_cfgs = Config(
        # sampler="CmaEsSampler",
        sampler="BoTorchSampler",
        params_cfgs=dict(
            design_region_size=dict(
                type="float",
                low=3,
                high=7,
                step=1,
                log=False,
            )
        ),
        run=Config(
            n_epochs=10,
        ),
    )

    def __init__(
        self,
        eval_obj_fn: Callable,  # given params, return objective
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.load_cfgs(**kwargs)
        self.eval_obj_fn = eval_obj_fn

        self.plot_thread = ThreadPoolExecutor(2)
        self.saver = BestKModelSaver(
            k=1,
            descend=False,
            truncate=10,
            metric_name="err",
            format="{:.4f}",
        )
        self.study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner()
        )
        self.distributions = self.define_distribution(self._cfg.params_cfgs)

    def define_distribution(self, params_cfgs):
        distributions = {}
        for key, param_cfg in params_cfgs.items():
            param_cfg = deepcopy(param_cfg)
            p_type = param_cfg.pop("type")
            if p_type == "float":
                distributions[key] = optuna.distributions.FloatDistribution(**param_cfg)
            elif p_type == "int":
                distributions[key] = optuna.distributions.IntDistribution(
                    key, **param_cfg
                )
            elif p_type == "categorical":
                distributions[key] = optuna.distributions.CategoricalDistribution(
                    key, **param_cfg
                )
            else:
                raise ValueError(f"Unknown parameter type: {p_type}")
        return distributions

    def objective(self, iter: int, trial):
        ### Step 1: obtain the parameters from the trial
        params = {key: trial.params[key] for key in self._cfg.params_cfgs}

        ### Step 2: calculate the objective via inverse design
        # this one need to be customized, we need objective and invdes object
        obj, invdes = self.eval_obj_fn(iter, params)
        return obj, invdes

    def load_cfgs(self, **cfgs):
        # Start with default configurations
        self.__dict__.update(self.default_cfgs)
        # Update with provided configurations
        self.__dict__.update(cfgs)
        # Save the updated configurations
        self.default_cfgs.update(cfgs)
        self._cfg = self.default_cfgs

    def search(
        self,
        progress_bar: bool=True,
    ):
        print(self.distributions)
        for i in trange(self._cfg.run.n_epochs, desc="Autotune", disable=not progress_bar, colour='green'):
            trial = self.study.ask(self.distributions)
            obj, invdes = self.objective(i, trial)
            self.study.tell(trial, obj)
            log = f"{'#' * 50}\n"
            log += f"Autotune Step {i:3d} objective: {obj:.4f} best obj: {self.study.best_trial.value:.4f} best: {self.study.best_trial.params}"
            log += f"\n{'#' * 50}\n"
            logger.warning(log)

    def save_model(self, invdes, fom, path):
        self.saver.save_model(
            invdes.devOptimization,
            fom,
            epoch=self._cfg.run.n_epochs,
            path=path,
            save_model=False,
            print_msg=True,
        )
