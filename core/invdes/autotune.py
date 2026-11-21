"""Auto-tuning wrapper for inverse-design optimization.

``AutoTune`` exposes an Optuna-driven search loop around a user-supplied inverse-design
evaluation callback.
"""

import os
import pprint
import sys
from copy import deepcopy
from typing import Callable, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../MAPS"))
sys.path.insert(0, project_root)
from concurrent.futures import ThreadPoolExecutor

import optuna
from pyutils.config import Config
from pyutils.general import ensure_dir, get_logger
from pyutils.torch_train import BestKModelSaver
from tqdm import trange

__all__ = ["AutoTune"]


class AutoTune(object):
    """Manage Optuna-based hyperparameter search around an inverse-design objective.

    Args:
        eval_obj_fn (Callable): Callback invoked as ``eval_obj_fn(iter, params)`` and
            expected to return ``(objective_value, invdes)`` for a sampled parameter set.
        *args: Reserved positional arguments retained for compatibility.
        opt_direction (str): Optuna study direction, typically ``"maximize"`` or
            ``"minimize"``. Defaults to ``"maximize"``.
        log_path (str): Optional path for the AutoTune log file. Defaults to ``""``.
        **kwargs: Configuration overrides merged into ``default_cfgs``.

    Attributes:
        study (optuna.study.Study): Optuna study used to sample and record trials.
        distributions (dict): Optuna distribution objects derived from ``params_cfgs``.
        trial_cache (dict): Cache mapping sampled parameter tuples to previously computed
            objective values so duplicate trials can reuse results.
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
        opt_direction: str = "maximize",
        log_path: str = "",
        **kwargs,
    ) -> None:
        """Initialize the Optuna study, parameter distributions, cache, and logger.

        Args:
            eval_obj_fn (Callable): Objective callback used for each trial.
            *args: Reserved positional arguments retained for compatibility.
            opt_direction (str): Optuna study direction. Defaults to ``"maximize"``.
            log_path (str): Optional output log path. Defaults to ``""``.
            **kwargs: Configuration overrides merged into ``default_cfgs``.

        Side Effects:
            Creates an Optuna study, an executor used by legacy plotting paths, a
            single-model saver, and an in-memory duplicate-trial cache.
        """
        super().__init__()
        self.load_cfgs(**kwargs)
        self.eval_obj_fn = eval_obj_fn
        self.opt_direction = opt_direction
        self.log_path = log_path
        if log_path:
            ensure_dir(os.path.dirname(log_path))
            if os.path.exists(log_path):
                with open(log_path, "w") as f:
                    f.write("")
        self.logger = get_logger(log_path=log_path, name="AutoTune")

        self.plot_thread = ThreadPoolExecutor(2)
        self.saver = BestKModelSaver(
            k=1,
            descend=False,
            truncate=10,
            metric_name="err",
            format="{:.4f}",
        )
        self.study = optuna.create_study(
            direction=opt_direction, pruner=optuna.pruners.MedianPruner()
        )
        self.distributions = self.define_distribution(self._cfg.params_cfgs)
        self.init_cache()

    def define_distribution(self, params_cfgs):
        """Convert parameter configuration mappings into Optuna distributions.

        Args:
            params_cfgs (dict[str, dict]): Mapping from parameter name to a configuration
                containing a ``type`` key and the keyword arguments required by the
                corresponding Optuna distribution.

        Returns:
            dict[str, optuna.distributions.BaseDistribution]: Distribution object per
            configured parameter.

        Raises:
            ValueError: If a configuration uses an unsupported ``type`` value.
        """
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

    def add_init_guesses(self, guess_list: List[dict]):
        """Queue one or more initial parameter guesses in the Optuna study.

        Args:
            guess_list (dict | list[dict]): Parameter assignments to enqueue. A single
                dictionary is converted to a one-element list.

        Raises:
            AssertionError: If any supplied parameter name is absent from
                ``self.distributions``.

        Side Effects:
            Adds the validated guesses to the Optuna study's pending trial queue.
        """
        ## study.enqueue_trial({"x": 5})
        if isinstance(guess_list, dict):
            guess_list = [guess_list]
        for guess in guess_list:
            check = [(key, key in self.distributions) for key in guess]
            assert all(
                state[1] for state in check
            ), f"Guess contains invalid parameters {list(filter(lambda x: not x[1], check))}"
            self.study.enqueue_trial(guess)

    def init_cache(self):
        """Initialize the in-memory duplicate-trial objective cache.

        Returns:
            None. ``self.trial_cache`` is reset to an empty dictionary.
        """
        # Initialize the cache for duplicate trials
        self.trial_cache = {}  # params_tuple: value

    def check_duplicate(self, trial):
        """Return a cached objective value for a trial's parameter tuple, if present.

        Args:
            trial (optuna.trial.Trial): Trial whose sampled parameter values are checked.

        Returns:
            float | None: Cached objective value, or ``None`` when the parameter tuple
            has not been evaluated previously.
        """
        # Check if the trial is a duplicate
        params_tuple = tuple(trial.params.values())
        # self.logger.info(f"{params_tuple}")
        # self.logger.info(f"{self.trial_cache}")
        return self.trial_cache.get(params_tuple, None)

    def sample_unique_trial(self, iter: int, max_try: int = 3):
        """Ask Optuna for a trial and retry when the sampled parameters duplicate a cache entry.

        Args:
            iter (int): Current outer search iteration, used only for diagnostic logging.
            max_try (int): Maximum number of duplicate resampling attempts. Defaults to 3.

        Returns:
            tuple[optuna.trial.Trial, float | None]: The sampled trial and either ``None``
            for a new parameter tuple or the cached objective value after exhausting retries.
        """
        # Check if the trial is a duplicate
        ### check if the trial is a duplicate
        trial = self.study.ask(self.distributions)
        for _ in range(max_try):
            value = self.check_duplicate(trial)
            if value is not None:
                trial = self.study.ask(self.distributions)
            else:
                return trial, None
        else:
            self.logger.warning(
                f"Trial {trial.number} at Step {iter} is a duplicate, use cached obj value {value:>4f}."
            )
            return trial, value

    def objective(self, iter: int, trial):
        """Evaluate one Optuna trial through the configured inverse-design callback and cache it.

        Args:
            iter (int): Current search iteration passed to ``eval_obj_fn``.
            trial (optuna.trial.Trial): Optuna trial containing the sampled parameters.

        Returns:
            tuple[float, object]: Objective value and the inverse-design object returned by
            ``eval_obj_fn``. The objective value is also stored in ``self.trial_cache``.
        """
        ### Step 1: obtain the parameters from the trial
        params = {key: trial.params[key] for key in self._cfg.params_cfgs}

        ### Step 2: calculate the objective via inverse design
        # this one need to be customized, we need objective and invdes object
        obj, invdes = self.eval_obj_fn(iter, params)

        ### Step 3: save the objective value in the cache
        self.trial_cache[tuple(trial.params.values())] = obj
        return obj, invdes

    def load_cfgs(self, **cfgs):
        """Merge configuration overrides into the active AutoTune configuration.

        Args:
            **cfgs (dict): Top-level configuration overrides. These are merged into
                ``default_cfgs`` and become available through ``self._cfg``.

        Returns:
            None. ``self._cfg`` references the updated configuration object.
        """
        # Start with default configurations
        self.__dict__.update(self.default_cfgs)
        # Update with provided configurations
        self.__dict__.update(cfgs)
        # Save the updated configurations
        self.default_cfgs.update(cfgs)
        self._cfg = self.default_cfgs

    def report_topk(self, study=None, k: int = 3, iter: int = 0):
        """Format the current top-k completed trial results for logging.

        Args:
            study (optuna.study.Study | None): Study to inspect. Defaults to None,
                which uses ``self.study``.
            k (int): Number of best completed trials to report. Defaults to 3.
            iter (int): Current outer search iteration included in the report header.
                Defaults to 0.

        Returns:
            str: Human-readable report containing objective values and sampled parameters.
        """
        study = study or self.study
        trials = study.trials
        trials = sorted(
            [t for t in trials if t.value is not None],
            key=lambda t: t.value,
            reverse=self.opt_direction == "maximize",
        )[:k]
        log = f"Autotune Top {k} Report (Step {iter})\n"
        for i, trial in enumerate(trials):
            log += f"\tTop {i + 1} (Step {trial.user_attrs['iter']}): Obj: {trial.value:.4f}, Params: {trial.params} \n"
        return log

    def search(
        self,
        progress_bar: bool = True,
        report_topk: int = 3,
        max_resample_try: int = 10,
    ):
        """Run the configured Optuna search loop for the configured number of epochs.

        Args:
            progress_bar (bool): Whether to display the ``tqdm`` progress bar. Defaults to True.
            report_topk (int): Number of top trials included in each log report. Defaults to 3.
            max_resample_try (int): Maximum duplicate-trial resampling attempts per iteration.
                Defaults to 10.

        Returns:
            None. The completed trials are stored in ``self.study`` and duplicate objectives
            remain available in ``self.trial_cache``.
        """
        self.logger.warning("Autotune is searching the following variables:")
        pprint.pprint(self.distributions)
        for i in trange(
            self._cfg.run.n_epochs,
            desc="Autotune",
            disable=not progress_bar,
            colour="green",
        ):
            trial, cached_obj = self.sample_unique_trial(
                iter=i, max_try=max_resample_try
            )
            trial.set_user_attr("iter", i)
            log = f"Autotune Step {i:3d} trying params: {trial.params}....."
            self.logger.info(log)
            if cached_obj is None:
                obj, invdes = self.objective(i, trial)
            else:
                obj = cached_obj
                invdes = None
            self.study.tell(trial, obj)
            best_trial = self.study.best_trial
            log = f"\n{'#' * 100}\n"
            log += f"Autotune Step {i:3d} objective: {obj:.4f} best obj (Step {best_trial.user_attrs['iter']}): {best_trial.value:.4f} best: {best_trial.params}\n"
            log += self.report_topk(study=self.study, k=report_topk, iter=i)
            log += f"\n{'#' * 100}\n"
            self.logger.warning(log)

    def save_model(self, invdes, fom, path):
        """Save the best inverse-design model through the configured model saver.

        Args:
            invdes (object): Inverse-design object exposing ``devOptimization``.
            fom (float): Objective/figure-of-merit value used by the saver.
            path (str): Destination checkpoint directory or file path accepted by
                ``BestKModelSaver``.

        Returns:
            None. The saver is invoked with the configured optimization epoch count.
        """
        self.saver.save_model(
            invdes.devOptimization,
            fom,
            epoch=self._cfg.run.n_epochs,
            path=path,
            save_model=False,
            print_msg=True,
        )
