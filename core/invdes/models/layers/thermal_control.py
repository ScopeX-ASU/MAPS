from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

_CURRENT_ROUND_DIGITS = 15


def _round_scalar(value) -> float:
    """Round a scalar control value to the canonical current-state precision.

    Args:
        value (float | int | Any): Scalar value convertible to ``float``.

    Returns:
        float: Value rounded to 15 decimal places.
    """
    return float(round(float(value), _CURRENT_ROUND_DIGITS))


def canonicalize_currents(
    currents: dict[str, float] | None,
    heat_source_names: Iterable[str] = (),
) -> tuple[tuple[str, float], ...]:
    """Canonicalize a heat-source current mapping into a deterministic control key.

    Args:
        currents (dict[str, float] | None): Mapping from heat-source name to current
            value. Missing sources are assigned zero current. Defaults to None.
        heat_source_names (Iterable[str]): Source names that must be present in the
            canonical state even when absent from ``currents``. Defaults to ``()``.

    Returns:
        tuple[tuple[str, float], ...]: Sorted ``(source_name, rounded_current)``
        pairs suitable for hashing, deduplication, and objective-control lookup.
        Every name in either input mapping/set appears exactly once.
    """
    dense = OrderedDict()
    for name in sorted(set(heat_source_names) | set((currents or {}).keys())):
        dense[name] = 0.0
    if currents:
        for name, value in currents.items():
            dense[str(name)] = _round_scalar(value)
    return tuple((name, _round_scalar(value)) for name, value in dense.items())


def currents_key_to_dict(
    control_key: tuple[tuple[str, float], ...],
) -> dict[str, float]:
    """Convert a canonical current-state key back into a mutable mapping.

    Args:
        control_key (tuple[tuple[str, float], ...]): Canonical source/current pairs
            produced by ``canonicalize_currents``.

    Returns:
        dict[str, float]: Mapping from heat-source name to floating-point current.
    """
    return {name: float(value) for name, value in control_key}


def is_zero_currents_key(control_key) -> bool:
    """Test whether every current in a canonical control key is exactly zero.

    Args:
        control_key (Iterable[tuple[str, float]]): Heat-source name/current pairs.

    Returns:
        bool: ``True`` when every current has zero magnitude after conversion to
        ``float``; otherwise ``False``.
    """
    return all(abs(float(value)) == 0.0 for _, value in control_key)


def normalize_objective_control_state(
    cfg: dict,
    heat_source_names: Iterable[str] = (),
) -> list[dict]:
    """Normalize objective control configuration into explicit control-state records.

    Args:
        cfg (dict): Objective configuration. The current implementation recognizes
            ``"currents"`` and legacy ``"temp"`` entries; when neither is present,
            it creates an all-zero current state.
        heat_source_names (Iterable[str]): Names of heat sources used to densify
            current mappings. Defaults to ``()``.

    Returns:
        list[dict]: One or more normalized state dictionaries. Current-driven states
        contain ``mode=\"currents\"``, a tuple-valued ``control_key``, and a ``currents``
        mapping. Legacy temperature states contain ``mode=\"legacy_temp\"``, a scalar
        ``control_key``, and ``temp``.
    """
    if "currents" in cfg:
        raw_states = cfg["currents"]
        if isinstance(raw_states, dict):
            raw_states = [raw_states]
        states = []
        for raw_state in raw_states:
            key = canonicalize_currents(raw_state, heat_source_names)
            states.append(
                {
                    "mode": "currents",
                    "control_key": key,
                    "currents": currents_key_to_dict(key),
                }
            )
        return states

    if "temp" in cfg:
        temps = cfg["temp"]
        if not isinstance(temps, (list, tuple)):
            temps = [temps]
        return [
            {
                "mode": "legacy_temp",
                "control_key": _round_scalar(temp),
                "temp": _round_scalar(temp),
            }
            for temp in temps
        ]

    key = canonicalize_currents({}, heat_source_names)
    return [
        {
            "mode": "currents",
            "control_key": key,
            "currents": currents_key_to_dict(key),
        }
    ]


def collect_unique_control_states(
    obj_cfgs: dict,
    heat_source_names: Iterable[str] = (),
) -> tuple[list[dict], dict]:
    """Collect unique normalized control states across objective configurations.

    Args:
        obj_cfgs (dict): Mapping from objective name to objective configuration.
            Non-dictionary values are preserved unchanged.
        heat_source_names (Iterable[str]): Heat-source names used when normalizing
            current-based objective states. Defaults to ``()``.

    Returns:
        tuple[list[dict], dict]: Ordered unique control-state records and a processed
        objective-configuration mapping. Each dictionary-valued objective receives
        ``control_states`` and ``control_keys`` entries.
    """
    ordered_states = []
    processed_cfgs = {}
    seen_keys = set()

    for name, cfg in obj_cfgs.items():
        if not isinstance(cfg, dict):
            processed_cfgs[name] = cfg
            continue
        states = normalize_objective_control_state(cfg, heat_source_names)
        cfg_copy = dict(cfg)
        cfg_copy["control_states"] = states
        cfg_copy["control_keys"] = [state["control_key"] for state in states]
        processed_cfgs[name] = cfg_copy
        for state in states:
            key = state["control_key"]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ordered_states.append(state)

    return ordered_states, processed_cfgs
