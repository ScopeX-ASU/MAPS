"""Weighted maximization objectives and sequential, first-order scenario evaluation."""

import time
from contextlib import contextmanager
from copy import deepcopy

import torch

from .recording import sparse_recording_context
from .sampling import VariationSampler
from .transforms import MapsGeometryTransform


def aggregate_objectives(
    values, weights=None, mode="mean", penalty=1.0, temperature=0.1
):
    """Aggregate scalar objectives under the MAPS maximization convention.

    Args:
        values: Real tensor of shape (K,) containing scenario objectives.
        weights: Optional nonnegative K-vector of scenario probabilities.
        mode: mean, mean_std, minimum, smooth_min, or callable(values, weights).
        penalty: Nonnegative standard-deviation penalty for mean_std.
        temperature: Positive smooth-min temperature in objective units.

    Returns:
        Scalar tensor differentiable with respect to scenario objectives.

    Raises:
        ValueError: For invalid shapes, weights, or aggregation settings.
    """
    if (
        values.ndim != 1
        or not values.numel()
        or torch.is_complex(values)
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("Objectives must be a finite real nonempty vector")
    w = (
        torch.ones_like(values)
        if weights is None
        else torch.as_tensor(weights, dtype=values.dtype, device=values.device)
    )
    if (
        w.shape != values.shape
        or not bool(torch.isfinite(w).all())
        or bool((w < 0).any())
        or not bool(w.sum() > 0)
    ):
        raise ValueError(
            "Scenario weights must be finite, nonnegative, and sum to a positive value"
        )
    w = w / w.sum()
    if callable(mode):
        result = mode(values, w)
        if result.ndim != 0:
            raise ValueError("Custom aggregation must return a scalar tensor")
        return result
    mean = (w * values).sum()
    if mode == "mean":
        return mean
    if mode == "mean_std":
        if penalty < 0:
            raise ValueError("Standard deviation penalty must be nonnegative")
        variance = (w * (values - mean).square()).sum()
        # At zero population variance choose the zero subgradient, avoiding
        # sqrt(0)'s undefined derivative and preserving the exact mean value.
        std = variance.clamp_min(torch.finfo(values.dtype).tiny).sqrt()
        std = torch.where(variance > 0, std, torch.zeros_like(std))
        return mean - penalty * std
    active = w > 0
    if mode == "minimum":
        return values[active].min()
    if mode == "smooth_min":
        if temperature <= 0:
            raise ValueError("Smooth-min temperature must be positive")
        return -temperature * torch.logsumexp(
            w[active].log() - values[active] / temperature, dim=0
        )
    raise ValueError(f"Unknown aggregation mode: {mode}")


def sequential_objective(
    materials,
    evaluate,
    weights=None,
    aggregation=None,
    timings=None,
    gradient_callback=None,
):
    """Finish each solver backward before evaluating the next scenario.

    Args:
        materials: Iterable of differentiable material tensors for shared design.
        evaluate: Callable(material_leaf) returning a scalar objective tensor.
        weights: Optional scenario weights.
        aggregation: Keyword arguments for aggregate_objectives.
        timings: Optional dictionary accumulating host material-backward seconds.
        gradient_callback: Optional callable(surrogate, leaf, gradient), invoked
            after each immediate material backward. May attach diagnostic hooks
            to the scalar surrogate without rerunning the solver.

    Returns:
        Tuple of robust scalar and detached scenario objective vector.

    Notes:
        Each solver sees an independent leaf. Its material gradient is computed
        immediately, then connected to the design by a first-order surrogate.
        This protects mutable adjoint state and avoids retaining K solver graphs.
        Backward through the returned scalar computes the exact first derivative;
        solver Hessians and differentiation with respect to variations are not
        supported. Existing design transforms retain their autograd graph.
    """
    objectives = []
    differentiating = torch.is_grad_enabled()
    for material in materials:
        leaf = material.detach().requires_grad_(
            differentiating and material.requires_grad
        )
        value = evaluate(leaf)
        if value.numel() != 1:
            raise ValueError("Scenario objective must be scalar")
        value = value.reshape(())
        if leaf.requires_grad and value.requires_grad:
            before = time.perf_counter()
            (gradient,) = torch.autograd.grad(value, leaf)
            if timings is not None:
                timings["material_backward"] = (
                    timings.get("material_backward", 0.0) + time.perf_counter() - before
                )
            # For complex material tensors PyTorch uses the real inner product.
            correction = (
                (material - material.detach()).conj() * gradient.detach()
            ).real.sum()
            value = value.detach() + correction
            if gradient_callback is not None:
                gradient_callback(value, leaf, gradient.detach())
        objectives.append(value)
    values = torch.stack(objectives)
    return aggregate_objectives(values, weights, **(aggregation or {})), values.detach()


def _preserve_runtime_eps_gradients(value, leaf, gradient, states):
    """Populate scenario plotting gradients during outer loss backward.

    Args:
        value: Scalar scenario surrogate used by the robust aggregator.
        leaf: Independent permittivity tensor given to the scenario solver.
        gradient: Detached objective derivative with respect to leaf.
        states: Runtime material states whose eps tensors feed field plotting.

    Notes:
        Nonleaf runtime tensors must retain gradients before the immediate
        solver backward. The outer cotangent supplies the caller's loss sign
        and scenario aggregation coefficient. Diagnostics accumulate just like
        ordinary retained tensor gradients and keep no live solver graph.
    """
    recorded = []
    seen = set()
    for state in states.values():
        eps = state.get("eps")
        if (
            not isinstance(eps, torch.Tensor)
            or not eps.requires_grad
            or id(eps) in seen
        ):
            continue
        seen.add(id(eps))
        raw = gradient if eps is leaf else eps.grad
        if raw is None:
            continue
        recorded.append((eps, raw.detach()))
        # Immediate backward differentiated +objective, not the optimizer loss.
        eps.grad = None
    if not recorded:
        return

    def retain(cotangent):
        """Apply the actual outer loss cotangent to saved material derivatives.

        Args:
            cotangent: Scalar loss derivative with respect to scenario value.

        Returns:
            Unchanged cotangent for the existing design backward path.
        """
        for eps, raw in recorded:
            contribution = raw * cotangent.detach()
            eps.grad = contribution if eps.grad is None else eps.grad + contribution
        return cotangent

    value.register_hook(retain)


class RobustForward:
    """Opt-in controller preserving BaseOptimization's public result contract.

    Args:
        config: VariationSampler settings plus optional aggregation dictionary.
        materializer: Optional callable(nominal_build_tuple, scenario) returning
            an optical permittivity tensor. Defaults to MapsGeometryTransform.
        source_context: Optional callable(model, scenario) returning a context
            manager for supported source changes; defaults to native Gaussian.

    Attributes:
        sampler: Dedicated reproducible scenario sampler.
        step: Current logical optimization step, controlled by InvDesign.
    """

    def __init__(self, config, materializer=None, source_context=None):
        """Initialize sampling controls without building a device or solver.

        Args:
            config: Sampling settings and optional aggregation arguments.
            materializer: Optional geometry adapter; otherwise created lazily.
            source_context: Optional context-manager factory for source updates.
        """
        self.sampler = VariationSampler(config)
        self.aggregation = dict(config.get("aggregation", {}))
        self.materializer = materializer
        self.source_context = source_context or gaussian_source_context
        self.step = 0

    def state_dict(self):
        """Export sampler and logical step for checkpoint continuation.

        Returns:
            Plain checkpoint dictionary; adapters must be configured on restore.
        """
        return {"sampler": self.sampler.state_dict(), "step": self.step}

    def load_state_dict(self, state):
        """Restore sampling sequence and current logical step.

        Args:
            state: Previously exported controller state.
        """
        self.sampler.load_state_dict(state["sampler"])
        self.step = state["step"]

    def forward(self, model, sharpness, ls_knots=None, custom_source=None):
        """Build the nominal design once and evaluate its sampled variations.

        Args:
            model: BaseOptimization model whose nominal forward path is bypassed.
            sharpness: Current design-projection sharpness.
            ls_knots: Optional nominal level-set arguments.
            custom_source: Existing source override; incompatible with native
                sampled Gaussian source changes.

        Returns:
            Existing result dictionary plus detached scenario diagnostics/timings.
        """
        control_states = getattr(model.objective, "control_states", {})
        if (
            getattr(model.device, "active_region_cfgs", {})
            or getattr(model.device, "active_region_masks", {})
            or any(
                state.get("mode") != "legacy_temp" for state in control_states.values()
            )
        ):
            raise NotImplementedError(
                "Robust coupled thermal/control-state solves require a custom integration"
            )
        start = time.perf_counter()
        nominal = model.build_device(sharpness, ls_knots)
        built = time.perf_counter()
        if self.materializer is None:
            self.materializer = MapsGeometryTransform(model)
        model.current_sharpness = sharpness
        model._design_region_eps_dict = nominal[1]
        model._eps_map, model._hr_eps_map = nominal[0], nominal[2]
        model._runtime_thermal_material_maps = (
            model._build_runtime_thermal_material_maps_from_regions(nominal[4])
        )
        model.objective.runtime_material_maps = model._runtime_thermal_material_maps
        batch = self.sampler.get(self.step)
        sampled = time.perf_counter()
        breakdowns = []
        material_seconds = 0.0
        solve_seconds = 0.0

        def materials():
            nonlocal material_seconds
            for scenario in batch["scenarios"]:
                before = time.perf_counter()
                geometry = {
                    k: v for k, v in scenario.items() if not k.startswith("source.")
                }
                material = self.materializer(nominal, geometry)
                material_seconds += time.perf_counter() - before
                yield material

        scenarios = iter(batch["scenarios"])

        def evaluate(material):
            nonlocal solve_seconds
            scenario = next(scenarios)
            if custom_source is not None and any(
                k.startswith("source.") for k in scenario
            ):
                raise ValueError(
                    "custom_source cannot be combined with sampled native sources"
                )
            before = time.perf_counter()
            value = model.objective_layer([material], custom_source=custom_source)
            breakdowns.append(
                {
                    k: {
                        **item,
                        "value": (
                            item["value"].detach()
                            if isinstance(item["value"], torch.Tensor)
                            else item["value"]
                        ),
                    }
                    for k, item in model.objective.breakdown.items()
                }
            )
            solve_seconds += time.perf_counter() - before
            if len(breakdowns) == len(batch["scenarios"]):
                # Plotting keeps the last scenario's fields/material. Retain its
                # runtime material derivatives before the immediate adjoint.
                model._snapshot_runtime_thermal_states()
            return value

        def preserve_plot_gradient(value, leaf, gradient):
            """Keep only the last scenario's material plotting diagnostics.

            Args:
                value: Scenario surrogate to receive an outer-backward hook.
                leaf: Independent scenario permittivity.
                gradient: Detached objective material derivative.
            """
            if len(breakdowns) == len(batch["scenarios"]):
                _preserve_runtime_eps_gradients(
                    value,
                    leaf,
                    gradient,
                    getattr(model.objective, "runtime_thermal_states", {}),
                )

        # Keep sources in place through the immediate adjoint, not just forward.
        # The generator context surrounds yield until sequential_objective has
        # consumed the value and completed its material derivative.
        def contextual_materials():
            for material, scenario in zip(materials(), batch["scenarios"]):
                with self.source_context(model, scenario):
                    yield material

        timings = {}
        with sparse_recording_context(model, self.sampler, self.materializer):
            obj, values = sequential_objective(
                contextual_materials(),
                evaluate,
                batch["weights"],
                self.aggregation,
                timings,
                gradient_callback=preserve_plot_gradient,
            )
        model._obj = obj
        model._snapshot_runtime_thermal_states()
        breakdown = {}
        for name in breakdowns[0]:
            items = [b[name] for b in breakdowns]
            breakdown[name] = {
                **items[0],
                "value": sum(
                    w * item["value"] for w, item in zip(batch["weights"], items)
                ),
            }
        result = dict(obj=obj, breakdown=breakdown, eps_map=nominal[0], **nominal[1])
        result.update(
            scenario_objectives=values,
            scenarios=deepcopy(batch),
            scenario_breakdowns=breakdowns,
            variation_timings={
                "nominal_build": built - start,
                "sampling_and_setup": sampled - built,
                "materialization": material_seconds,
                "forward": solve_seconds,
                **timings,
                "total": time.perf_counter() - start,
            },
        )
        return result


@contextmanager
def gaussian_source_context(model, scenario):
    """Temporarily replace native Gaussian profiles, restoring them on failure.

    Args:
        model: MAPS optimization model with objective and simulation profiles.
        scenario: Names ``source.<slice>.dx/dy/waist_radius/waist_distance`` in
            micrometers. Only z-propagating GaussianPlaneSourceTidy3d is supported.

    Yields:
        None while per-scenario normalized sources are active.

    Notes:
        Native Poynting-flux normalization is recomputed during source placement.
        The injection plane and monitors remain fixed. Source changes may incur
        backend compilation; this adapter does not promise runtime-array reuse.
    """
    changes = {k: v for k, v in scenario.items() if k.startswith("source.")}
    if not changes:
        yield
        return
    patches = []
    try:
        profiles = model.objective.port_profiles
        grouped = {}
        for key, delta in changes.items():
            prefix, slice_name, parameter = key.split(".")
            if (
                parameter not in {"dx", "dy", "waist_radius", "waist_distance"}
                or slice_name not in profiles
            ):
                raise ValueError(f"Unsupported source variation: {key}")
            if (
                parameter in {"dx", "dy"}
                and delta
                and model.device.sim_cfg.get("symmetry", (0, 0, 0))[
                    {"dx": 0, "dy": 1}[parameter]
                ]
            ):
                raise ValueError(
                    "Off-center beams require disabling symmetry on the offset axis"
                )
            grouped.setdefault(slice_name, {})[parameter] = delta
        dictionaries = [profiles, model.device.port_sources_dict]
        dictionaries.extend(
            sim.port_sources_dict
            for sim in model.objective.sims.values()
            if hasattr(sim, "port_sources_dict")
        )
        unique = {id(d): d for d in dictionaries}.values()
        for slice_name, parameters in grouped.items():
            updated = {}
            for key, profile in profiles[slice_name].items():
                source = profile[0]
                if (
                    type(source).__name__ != "GaussianPlaneSourceTidy3d"
                    or source.partial_grid_shape[2] != 1
                ):
                    raise NotImplementedError(
                        "Source adapter requires native z-propagating Gaussian beams"
                    )
                for parameter in ("waist_radius", "waist_distance"):
                    if parameter in parameters:
                        new_value = (
                            getattr(source, parameter) + parameters[parameter] * 1e-6
                        )
                        if parameter == "waist_radius" and new_value <= 0:
                            raise ValueError(
                                "Gaussian waist radius must remain positive"
                            )
                        source = source.aset(parameter, new_value)
                if parameters.get("dx", 0) or parameters.get("dy", 0):
                    source = source.aset(
                        "beam_center_offset",
                        (
                            parameters.get("dx", 0) * 1e-6,
                            parameters.get("dy", 0) * 1e-6,
                        ),
                    )
                updated[key] = (source, *profile[1:])
            for dictionary in unique:
                if slice_name in dictionary:
                    patches.append((dictionary, slice_name, dictionary[slice_name]))
                    dictionary[slice_name] = updated
        yield
    finally:
        for dictionary, key, original in reversed(patches):
            dictionary[key] = original
