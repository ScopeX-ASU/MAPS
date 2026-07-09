from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

_CURRENT_ROUND_DIGITS = 15


def _round_scalar(value) -> float:
    return float(round(float(value), _CURRENT_ROUND_DIGITS))


def canonicalize_currents(
    currents: dict[str, float] | None,
    heat_source_names: Iterable[str] = (),
) -> tuple[tuple[str, float], ...]:
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
    return {name: float(value) for name, value in control_key}


def is_zero_currents_key(control_key) -> bool:
    return all(abs(float(value)) == 0.0 for _, value in control_key)


def normalize_objective_control_state(
    cfg: dict,
    heat_source_names: Iterable[str] = (),
) -> list[dict]:
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
