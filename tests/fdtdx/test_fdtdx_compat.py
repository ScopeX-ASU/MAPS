"""Contract tests for the MAPS FDTD-X compatibility layer."""

from __future__ import annotations

from importlib import import_module

import pytest

fdtdx = pytest.importorskip("fdtdx")
compat = pytest.importorskip("thirdparty.fdtdx")


@pytest.mark.fdtdx
@pytest.mark.unit
def test_public_fdtdx_is_the_runtime_engine() -> None:
    """The compatibility package must wrap the installed public ``fdtdx`` module."""
    assert compat.fdtdx is fdtdx


@pytest.mark.fdtdx
@pytest.mark.unit
def test_maps_extensions_are_registered_on_public_fdtdx() -> None:
    """MAPS-required forward extensions are available through the public namespace."""
    assert fdtdx.place_objects is compat.place_objects
    assert (
        fdtdx.DetectorDecayAfterSourceCondition
        is compat.DetectorDecayAfterSourceCondition
    )
    assert fdtdx.GaussianPlaneSourceTidy3d is compat.GaussianPlaneSourceTidy3d


@pytest.mark.fdtdx
@pytest.mark.unit
def test_physical_power_overrides_are_registered_on_public_fdtdx() -> None:
    """Public FDTD-X entry points must resolve to MAPS' local overrides."""
    metrics = import_module("fdtdx.core.physics.metrics")
    modes = import_module("fdtdx.core.physics.modes")
    overrides = import_module("thirdparty.fdtdx.core.physics.power_normalization")

    assert metrics.compute_integrated_power is overrides.compute_integrated_power
    assert metrics.normalize_by_poynting_flux is overrides.normalize_by_poynting_flux
    assert modes.compute_mode is overrides.compute_mode
    assert (
        modes.compute_mode_symmetry_reduced is overrides.compute_mode_symmetry_reduced
    )
    assert fdtdx.compute_integrated_power is overrides.compute_integrated_power
    assert fdtdx.normalize_by_poynting_flux is overrides.normalize_by_poynting_flux


@pytest.mark.fdtdx
@pytest.mark.unit
def test_distributed_current_source_capability_is_explicit() -> None:
    """Adjoint-source availability is represented explicitly rather than implicitly."""
    assert isinstance(compat.FDTDX_ADJOINT_AVAILABLE, bool)
    assert compat.FDTDX_ADJOINT_AVAILABLE is (
        compat.DistributedCurrentSource is not None
    )

    if compat.FDTDX_ADJOINT_AVAILABLE:
        assert fdtdx.DistributedCurrentSource is compat.DistributedCurrentSource
    else:
        assert not hasattr(fdtdx, "DistributedCurrentSource")


@pytest.mark.fdtdx
@pytest.mark.unit
def test_maps_fdtdx_public_import_contract() -> None:
    """Protect the public FDTD-X import paths consumed by MAPS FDTD code."""
    expected_imports = {
        "fdtdx.config": ("SimulationConfig",),
        "fdtdx.core.grid": ("RectilinearGrid", "UniformGrid"),
        "fdtdx.core.physics.metrics": ("normalize_by_poynting_flux",),
        "fdtdx.fdtd.container": ("ArrayContainer", "ObjectContainer"),
        "fdtdx.objects.sources.profile": (
            "GaussianPulseProfile",
            "SingleFrequencyProfile",
        ),
    }

    for module_name, symbol_names in expected_imports.items():
        module = import_module(module_name)
        for symbol_name in symbol_names:
            symbol = getattr(module, symbol_name, None)
            assert (
                symbol is not None
            ), f"MAPS FDTD-X integration requires {module_name}.{symbol_name}"
