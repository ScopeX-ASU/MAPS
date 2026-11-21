"""Physics-independent named scenario sampling with a private reproducible RNG."""

import secrets
from copy import deepcopy

import numpy as np
from scipy import stats
from scipy.stats import qmc


def _configuration_signature(value):
    """Describe sampling settings without object addresses or RNG state.

    Args:
        value: Nested configuration value, including programmatic adapters.

    Returns:
        Comparable structure suitable for checkpoint compatibility validation.
    """
    if isinstance(value, dict):
        return {k: _configuration_signature(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_configuration_signature(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if hasattr(value, "dist") and hasattr(value.dist, "name"):
        return [
            value.dist.name,
            _configuration_signature(value.args),
            _configuration_signature(value.kwds),
        ]
    if hasattr(value, "mean") and hasattr(value, "cov"):
        return [
            type(value).__name__,
            _configuration_signature(value.mean),
            _configuration_signature(value.cov),
        ]
    if type(value).__module__.startswith("torch.distributions"):
        return [type(value).__name__, _configuration_signature(vars(value))]
    if callable(value):
        return [
            getattr(value, "__module__", type(value).__module__),
            getattr(value, "__qualname__", type(value).__qualname__),
        ]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


class VariationSampler:
    """Sample deviations without touching model-initialization random state.

    Args:
        config: Dictionary with enabled, seed, num_samples, resample_every,
            strategy, distributions, and optional explicit scenarios/weights.
            Distribution entries contain params and native library arguments.
            Uniform entries accept intuitive ``min``/``max`` bounds for SciPy
            and NumPy; existing backend-native bounds remain supported. For
            Gaussian entries, min/max truncate the distribution in parameter
            units, without clipping draws to the endpoints.

    Attributes:
        seed: Actual seed, including the recorded entropy seed when unspecified.
        enabled: Whether scenarios should be evaluated.

    Notes:
        Call ``get(step)`` with a logical optimizer step, never a closure count.
        Checkpoints require the same configuration and callable implementations.
    """

    def __init__(self, config):
        """Validate controls and initialize a private random stream.

        Args:
            config: Sampling settings described by the class interface.

        Raises:
            ValueError: If counts or independent parameter names are invalid.
        """
        self.config = deepcopy(dict(config))
        self.enabled = bool(self.config.get("enabled", False))
        self.num_samples = int(self.config.get("num_samples", 1))
        self.interval = int(self.config.get("resample_every", 1))
        if self.num_samples < 1 or self.interval < 1:
            raise ValueError("num_samples and resample_every must be positive")
        self.strategy = self.config.get("strategy", "iid")
        self.frozen = self.strategy == "fixed"
        self.set_seed(self.config.get("seed"))
        names = [p for d in self.config.get("distributions", []) for p in d["params"]]
        if len(names) != len(set(names)) or any(
            not isinstance(p, str) or not p for p in names
        ):
            raise ValueError("Distribution parameter names must be nonempty and unique")
        self.names = names

    def set_seed(self, seed):
        """Reset the independent random stream and discard cached scenarios.

        Args:
            seed: Nonnegative integer, or None to use system entropy.
        """
        self.seed = secrets.randbits(63) if seed is None else int(seed)
        self.config["seed"] = seed
        self.rng = np.random.default_rng(self.seed)
        self._batch = None
        self._bucket = None

    def randomize_seed(self):
        """Reset with an entropy-generated seed and return the recorded seed.

        Returns:
            Integer seed suitable for reproducing the new stream.
        """
        self.set_seed(None)
        return self.seed

    def freeze(self):
        """Reuse the next or currently cached batch indefinitely."""
        self.frozen = True

    def unfreeze(self):
        """Resume refreshing batches at logical-step intervals."""
        self.frozen = False

    def _draw(self, spec, n, unit=None):
        """Draw one native distribution block.

        Args:
            spec: Distribution entry, including ordered parameter names.
            n: Number of scenarios.
            unit: Optional unit-cube samples for inverse-CDF strategies.

        Returns:
            Finite array with shape (n, len(params)).
        """
        dim = len(spec["params"])
        backend = spec.get("backend", "scipy")
        kind = spec.get("type", "norm")
        kwargs = {
            k: v
            for k, v in spec.items()
            if k
            not in {
                "params",
                "backend",
                "type",
                "distribution",
                "components",
                "weights",
            }
        }
        if "min" in kwargs or "max" in kwargs:
            if "min" not in kwargs or "max" not in kwargs:
                raise ValueError(f"{kind} bounds require both min and max")
            if kind == "uniform" and (
                spec.get("distribution") is not None
                or {"loc", "scale", "low", "high"} & kwargs.keys()
            ):
                raise ValueError(
                    "Uniform min/max cannot be combined with native bounds or a frozen distribution"
                )
            lower = np.asarray(kwargs.pop("min"), dtype=float)
            upper = np.asarray(kwargs.pop("max"), dtype=float)
            if (
                lower.shape not in {(), (dim,)}
                or upper.shape not in {(), (dim,)}
                or not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
                or not np.all(upper > lower)
            ):
                raise ValueError(
                    f"{kind} min/max must be finite scalar or per-parameter bounds with max > min"
                )
            if kind == "uniform" and backend in {"scipy", "numpy"}:
                if backend == "scipy":
                    kwargs.update(loc=lower, scale=upper - lower)
                else:
                    kwargs.update(low=lower, high=upper)
            elif (backend, kind) in {
                ("scipy", "norm"),
                ("numpy", "normal"),
            } and spec.get("distribution") is None:
                loc = np.asarray(kwargs.pop("loc", 0.0), dtype=float)
                scale = np.asarray(kwargs.pop("scale", 1.0), dtype=float)
                if (
                    not np.isfinite(loc).all()
                    or not np.isfinite(scale).all()
                    or np.any(scale <= 0)
                ):
                    raise ValueError(
                        "Truncated Gaussian loc/scale must be finite with scale > 0"
                    )
                distribution = stats.truncnorm(
                    (lower - loc) / scale,
                    (upper - loc) / scale,
                    loc=loc,
                    scale=scale,
                    **kwargs,
                )
                values = (
                    distribution.ppf(unit)
                    if unit is not None
                    else distribution.rvs(size=(n, dim), random_state=self.rng)
                )
                values = np.asarray(values, dtype=float).reshape(n, dim)
                if not np.isfinite(values).all():
                    raise ValueError("Scenarios must contain finite values")
                return values
            else:
                if unit is not None:
                    raise ValueError(
                        "General truncated distributions support IID sampling only"
                    )
                # Rejection conditions the joint distribution on the box and
                # preserves correlations; clipping would add endpoint masses.
                base = {k: v for k, v in spec.items() if k not in {"min", "max"}}
                accepted = []
                remaining = n
                for _ in range(1024):
                    candidates = self._draw(base, max(remaining * 2, 32))
                    candidates = candidates[
                        np.all((candidates >= lower) & (candidates <= upper), axis=1)
                    ][:remaining]
                    accepted.append(candidates)
                    remaining -= len(candidates)
                    if not remaining:
                        return np.concatenate(accepted)
                raise ValueError(
                    "Truncation rejection budget exhausted; widen bounds or use an inverse-CDF adapter"
                )
        if kind == "mixture":
            if unit is not None:
                raise ValueError("Mixtures support IID sampling only")
            components = spec["components"]
            weights = np.asarray(spec["weights"], dtype=float)
            if (
                weights.shape != (len(components),)
                or np.any(weights < 0)
                or not np.isfinite(weights).all()
                or weights.sum() <= 0
            ):
                raise ValueError("Invalid mixture weights")
            choices = self.rng.choice(len(components), n, p=weights / weights.sum())
            values = np.empty((n, dim))
            for i, component in enumerate(components):
                count = int(np.sum(choices == i))
                if count:
                    values[choices == i] = self._draw(
                        dict(component, params=spec["params"]), count
                    )
        elif backend == "scipy":
            allowed = {"norm", "uniform", "truncnorm", "lognorm", "multivariate_normal"}
            if kind not in allowed:
                raise ValueError(f"Unsupported scipy distribution: {kind}")
            distribution = spec.get("distribution")
            if distribution is None:
                if kind == "multivariate_normal":
                    mean = np.asarray(kwargs.get("mean"))
                    cov = np.asarray(kwargs.get("cov"))
                    if (
                        mean.shape != (dim,)
                        or cov.shape != (dim, dim)
                        or not np.allclose(cov, cov.T)
                    ):
                        raise ValueError(
                            "Joint mean/covariance dimension or symmetry mismatch"
                        )
                distribution = getattr(stats, kind)(**kwargs)
            if unit is not None:
                if not hasattr(distribution, "ppf") or kind == "multivariate_normal":
                    raise ValueError("Unit-cube strategies require scalar inverse CDFs")
                values = distribution.ppf(unit)
            else:
                is_joint = (
                    kind == "multivariate_normal" or getattr(distribution, "dim", 1) > 1
                )
                size = n if is_joint else (n, dim)
                values = distribution.rvs(size=size, random_state=self.rng)
        elif backend == "numpy":
            if unit is not None or kind not in {
                "normal",
                "uniform",
                "multivariate_normal",
            }:
                raise ValueError("Unsupported NumPy distribution/strategy")
            if kind == "multivariate_normal":
                mean = np.asarray(kwargs.get("mean"))
                cov = np.asarray(kwargs.get("cov"))
                if (
                    mean.shape != (dim,)
                    or cov.shape != (dim, dim)
                    or not np.isfinite(cov).all()
                    or not np.allclose(cov, cov.T)
                    or np.linalg.eigvalsh(cov).min() < 0
                ):
                    raise ValueError(
                        "Joint covariance must be symmetric positive semidefinite with matching dimensions"
                    )
                kwargs["check_valid"] = "raise"
            size = n if kind == "multivariate_normal" else (n, dim)
            values = getattr(self.rng, kind)(size=size, **kwargs)
        elif backend == "custom":
            if unit is not None:
                raise ValueError(
                    "Custom distributions require IID or a custom strategy"
                )
            values = spec["distribution"](self.rng, n)
        elif backend == "torch":
            if unit is not None:
                raise ValueError("Torch distributions support IID sampling only")
            import torch

            distribution = spec["distribution"]
            if any(
                isinstance(v, torch.Tensor) and v.device.type != "cpu"
                for v in vars(distribution).values()
            ):
                raise ValueError("Torch sampling adapters require CPU distributions")
            # fork_rng restores the process-wide CPU RNG after the adapter call.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(
                    int(self.rng.integers(0, 2**63 - 1))
                )
                values = distribution.sample((n,)).cpu().numpy()
        else:
            raise ValueError(f"Unknown distribution backend: {backend}")
        values = np.asarray(values, dtype=float)
        if values.size != n * dim:
            raise ValueError("Sample dimension does not match parameter names")
        values = values.reshape(n, dim)
        if not np.isfinite(values).all():
            raise ValueError("Scenarios must contain finite values")
        return values

    def support_bounds(self):
        """Return theoretical parameter support, without drawing samples.

        Returns:
            Mapping from names to inclusive (min, max) bounds. Unbounded or
            opaque distributions have infinite bounds. Explicit scenarios use
            their extrema including zero for missing parameters.

        Notes:
            Finite bounds allow geometry recorders to cover every permitted
            sample. Bounds describe deviations, not absolute geometry lengths.
        """
        explicit = self.config.get("scenarios")
        if explicit is not None:
            names = {name for scenario in explicit for name in scenario}
            return {
                name: (
                    min(s.get(name, 0.0) for s in explicit),
                    max(s.get(name, 0.0) for s in explicit),
                )
                for name in names
            }

        def bounds(spec):
            """Resolve an adapter's support in parameter order.

            Args:
                spec: Distribution configuration with ordered parameter names.

            Returns:
                Lower and upper arrays with one value per parameter.
            """
            dim = len(spec["params"])
            if "min" in spec or "max" in spec:
                lower, upper = np.broadcast_to(spec["min"], (dim,)), np.broadcast_to(
                    spec["max"], (dim,)
                )
            elif spec.get("type") == "mixture":
                active = [
                    bounds(dict(c, params=spec["params"]))
                    for c, w in zip(spec["components"], spec["weights"])
                    if w > 0
                ]
                lower = np.min([b[0] for b in active], axis=0)
                upper = np.max([b[1] for b in active], axis=0)
            elif (
                spec.get("backend", "scipy") == "scipy"
                and spec.get("type", "norm") != "multivariate_normal"
            ):
                distribution = spec.get("distribution")
                if distribution is None:
                    kwargs = {
                        k: v
                        for k, v in spec.items()
                        if k not in {"params", "backend", "type"}
                    }
                    distribution = getattr(stats, spec.get("type", "norm"))(**kwargs)
                if hasattr(distribution, "support"):
                    lower, upper = distribution.support()
                else:
                    lower, upper = -np.inf, np.inf
            elif spec.get("backend") == "numpy" and spec.get("type") == "uniform":
                lower, upper = spec.get("low", 0.0), spec.get("high", 1.0)
            else:
                lower, upper = -np.inf, np.inf
            return np.broadcast_to(lower, (dim,)), np.broadcast_to(upper, (dim,))

        result = {}
        for spec in self.config.get("distributions", []):
            lower, upper = bounds(spec)
            result.update(
                {
                    name: (float(lo), float(hi))
                    for name, lo, hi in zip(spec["params"], lower, upper)
                }
            )
        return result

    def sample(self, n=None):
        """Generate a fresh batch without changing the logical-step cache.

        Args:
            n: Optional positive sample count; defaults to num_samples.

        Returns:
            Dictionary with scenarios (list of named deviations) and weights.
            Disabled sampling returns a single empty nominal scenario.

        Raises:
            ValueError: For unsupported strategy/distribution combinations.
        """
        if not self.enabled:
            return {"scenarios": [{}], "weights": [1.0]}
        n = self.num_samples if n is None else int(n)
        if n < 1:
            raise ValueError("Sample count must be positive")
        explicit = self.config.get("scenarios")
        if explicit is not None:
            scenarios = deepcopy(explicit)
        elif callable(self.strategy):
            scenarios = self.strategy(self.rng, n)
        else:
            unit = None
            strategy = "iid" if self.strategy == "fixed" else self.strategy
            if strategy in {"sobol", "halton", "latin_hypercube", "antithetic"}:
                if not self.names:
                    raise ValueError("Unit-cube strategies need distributions")
                if strategy == "antithetic":
                    half = self.rng.random(((n + 1) // 2, len(self.names)))
                    unit = np.concatenate((half, 1 - half), axis=0)[:n]
                else:
                    engine = {
                        "sobol": qmc.Sobol,
                        "halton": qmc.Halton,
                        "latin_hypercube": qmc.LatinHypercube,
                    }[strategy]
                    # SciPy spawns child SeedSequences when given a Generator;
                    # that spawn counter is not part of bit_generator.state.
                    # Draw an explicit engine seed so the saved RNG is sufficient.
                    engine_seed = int(self.rng.integers(0, 2**32 - 1))
                    unit = engine(d=len(self.names), seed=engine_seed).random(n)
                unit = np.clip(unit, np.finfo(float).eps, 1 - np.finfo(float).eps)
            elif strategy != "iid":
                raise ValueError(f"Unknown sampling strategy: {strategy}")
            scenarios = [{} for _ in range(n)]
            column = 0
            for spec in self.config.get("distributions", []):
                dim = len(spec["params"])
                block = self._draw(
                    spec, n, None if unit is None else unit[:, column : column + dim]
                )
                column += dim
                for row, scenario in enumerate(scenarios):
                    scenario.update(zip(spec["params"], block[row].tolist()))
        if not scenarios or any(
            not np.isfinite(list(s.values())).all() for s in scenarios
        ):
            raise ValueError("Scenarios must be nonempty and finite")
        if any(not isinstance(k, str) or not k for s in scenarios for k in s):
            raise ValueError("Scenario parameter names must be nonempty strings")
        try:
            scenarios = [{k: float(v) for k, v in s.items()} for s in scenarios]
        except (TypeError, ValueError) as error:
            raise ValueError("Scenario values must be scalars") from error
        weights = np.asarray(
            self.config.get("weights", np.ones(len(scenarios))), dtype=float
        )
        if (
            weights.shape != (len(scenarios),)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError("Invalid scenario weights")
        return {"scenarios": scenarios, "weights": (weights / weights.sum()).tolist()}

    def get(self, step):
        """Return the cached batch for a logical step interval.

        Args:
            step: Nonnegative logical optimization-step index.

        Returns:
            Defensive copy of the current scenario batch.
        """
        if int(step) != step or step < 0:
            raise ValueError("step must be a nonnegative integer")
        bucket = int(step) // self.interval
        if self._batch is not None and self._bucket is None:
            self._bucket = bucket
        if self._batch is None or (not self.frozen and bucket != self._bucket):
            self._batch = self.sample()
            self._bucket = bucket
        return deepcopy(self._batch)

    def resample(self):
        """Replace the cached batch, including when frozen.

        Returns:
            Defensive copy of the new batch.
        """
        self._batch = self.sample()
        return deepcopy(self._batch)

    def state_dict(self):
        """Export RNG, seed, cache, interval index, and freeze state.

        Returns:
            Plain serializable dictionary for exact continuation.
        """
        return deepcopy(
            dict(
                seed=self.seed,
                rng=self.rng.bit_generator.state,
                batch=self._batch,
                bucket=self._bucket,
                frozen=self.frozen,
                config_signature=_configuration_signature(self.config),
            )
        )

    def load_state_dict(self, state):
        """Restore a checkpoint created with this sampling configuration.

        Args:
            state: Dictionary returned by state_dict.

        Raises:
            ValueError: If the saved non-callable configuration differs.
        """
        if state["config_signature"] != _configuration_signature(self.config):
            raise ValueError("Sampler checkpoint configuration mismatch")
        self.seed = state["seed"]
        self.rng.bit_generator.state = deepcopy(state["rng"])
        self._batch = deepcopy(state["batch"])
        self._bucket = state["bucket"]
        self.frozen = state["frozen"]
