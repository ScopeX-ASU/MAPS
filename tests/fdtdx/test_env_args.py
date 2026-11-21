"""FDTD environment defaults and parsing contracts; no solver/GPU imports."""

import pytest

from core.env_args import ENV_ARGS, describe_env_args, get_env_arg

pytestmark = pytest.mark.unit


# Baseline captured from solver.py, fdtd.py, and device_base.py before migration.
DEFAULTS = {
    "MAPS_NORMALIZE_PHASOR_BY_SOURCE_SPECTRUM": True,
    "MAPS_FDTDX_ADJOINT_DEBUG": False,
    "MAPS_FDTDX_ADJOINT_DEBUG_DIR": "figs/fdtdx_adjoint_debug",
    "MAPS_ENABLE_BROADBAND_ADJOINT": False,
    "MAPS_BROADBAND_ADJOINT_METHOD": "auto",
    "MAPS_BROADBAND_ADJOINT_PROFILE_ORDER": 3,
    "MAPS_BROADBAND_ADJOINT_PROFILE_RTOL": 1e-3,
    "MAPS_BROADBAND_ADJOINT_MIN_OVERLAP": 0.5,
    "MAPS_BROADBAND_ADJOINT_MAX_COEFFICIENT_NORM": 100.0,
    "MAPS_SPARSE_PHASOR": False,
    "MAPS_SPARSE_PHASOR_VISUALIZATION_PLANES": True,
    "MAPS_COMPARE_SPARSE_DENSE": False,
    "MAPS_WRITE_SPARSE_DENSE_PLOTS": False,
    "MAPS_SPARSE_ADJOINT_SOURCE": False,
    "MAPS_ENABLE_PHASOR_H": True,
    "MAPS_PHASOR_H_NATIVE_SLAB_CELLS": 3,
    "MAPS_COMPARE_EXACT_INTERPOLATION": False,
}


@pytest.mark.parametrize("name, expected", DEFAULTS.items())
def test_existing_defaults_and_types_are_preserved(name, expected):
    actual = get_env_arg(name, environ={})
    assert actual == expected
    assert type(actual) is type(expected)


def test_device_monitor_h_default_is_preserved():
    assert get_env_arg("MAPS_ENABLE_PHASOR_H", context="device", environ={}) is False
    for context in ("device", "backend"):
        for value in ("0", "1"):
            assert get_env_arg(
                "MAPS_ENABLE_PHASOR_H",
                context=context,
                environ={"MAPS_ENABLE_PHASOR_H": value},
            ) is (value == "1")


@pytest.mark.parametrize(
    "name, raw, expected",
    [
        ("MAPS_BROADBAND_ADJOINT_METHOD", "CHEBYSHEV", "chebyshev"),
        ("MAPS_BROADBAND_ADJOINT_METHOD", "legacy", "legacy"),
        ("MAPS_BROADBAND_ADJOINT_PROFILE_ORDER", "20", 20),
        ("MAPS_BROADBAND_ADJOINT_PROFILE_RTOL", "0", 0.0),
        ("MAPS_BROADBAND_ADJOINT_PROFILE_RTOL", "1", 1.0),
        ("MAPS_BROADBAND_ADJOINT_MIN_OVERLAP", "1", 1.0),
        ("MAPS_FDTDX_ADJOINT_DEBUG_DIR", "custom plots", "custom plots"),
    ],
)
def test_explicit_values(name, raw, expected):
    assert get_env_arg(name, environ={name: raw}) == expected


@pytest.mark.parametrize(
    "name, raw",
    [
        ("MAPS_SPARSE_PHASOR", "true"),
        ("MAPS_ENABLE_BROADBAND_ADJOINT", "2"),
        ("MAPS_BROADBAND_ADJOINT_METHOD", "typo"),
        ("MAPS_BROADBAND_ADJOINT_PROFILE_ORDER", "-1"),
        ("MAPS_BROADBAND_ADJOINT_PROFILE_ORDER", "21"),
        ("MAPS_BROADBAND_ADJOINT_PROFILE_ORDER", "3.5"),
        ("MAPS_BROADBAND_ADJOINT_PROFILE_RTOL", "nan"),
        ("MAPS_BROADBAND_ADJOINT_PROFILE_RTOL", "inf"),
        ("MAPS_BROADBAND_ADJOINT_PROFILE_RTOL", "-1"),
        ("MAPS_BROADBAND_ADJOINT_MIN_OVERLAP", "1.1"),
        ("MAPS_BROADBAND_ADJOINT_MAX_COEFFICIENT_NORM", "-1"),
        ("MAPS_PHASOR_H_NATIVE_SLAB_CELLS", "0"),
        ("MAPS_FDTDX_ADJOINT_DEBUG_DIR", ""),
    ],
)
def test_invalid_values_report_name_and_domain(name, raw):
    with pytest.raises(ValueError) as exc:
        get_env_arg(name, environ={name: raw})
    assert name in str(exc.value)
    assert ENV_ARGS[name].allowed in str(exc.value)


def test_reads_are_not_cached(monkeypatch):
    name = "MAPS_SPARSE_PHASOR"
    monkeypatch.setenv(name, "0")
    assert get_env_arg(name) is False
    monkeypatch.setenv(name, "1")
    assert get_env_arg(name) is True


def test_help_covers_registry_and_defaults():
    assert set(DEFAULTS) == set(ENV_ARGS)
    help_text = describe_env_args()
    for name, spec in ENV_ARGS.items():
        assert name in help_text
        assert spec.default in help_text
        assert spec.allowed in help_text
        assert spec.description in help_text


def test_unknown_name_and_context_fail_explicitly():
    with pytest.raises(KeyError):
        get_env_arg("MAPS_TYPO", environ={})
    with pytest.raises(ValueError, match="context"):
        get_env_arg("MAPS_SPARSE_PHASOR", context="typo", environ={})
