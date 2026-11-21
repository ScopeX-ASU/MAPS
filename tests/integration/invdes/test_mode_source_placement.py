"""Actual mode-source placement and propagation through the MAPS FDTD adapter."""

import numpy as np
import pytest
import torch

pytest.importorskip("fdtdx")

from core.fdtd.fdtd import fdtd3d
from thirdparty.fdtdx import fdtdx

pytestmark = [pytest.mark.integration, pytest.mark.fdtdx, pytest.mark.numerical]


def test_nonuniform_mode_source_and_recorder_use_requested_physical_aperture(
    monkeypatch,
):
    """Resolve a guided mode on the corrected aperture and propagate finite fields."""
    import importlib

    import jax.numpy as jnp

    module = importlib.import_module("core.fdtd.fdtd")
    monkeypatch.setattr(module, "_MAPS_SPARSE_PHASOR", True)
    edges_um = (
        np.linspace(-1.6, 1.6, 17),
        np.array(
            [
                -1.2,
                -1.1,
                -1.0,
                -0.8,
                -0.6,
                -0.4,
                -0.2,
                0.0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
                1.2,
            ]
        ),
        np.linspace(-1.2, 1.2, 13),
    )
    coords = tuple((axis[:-1] + axis[1:]) / 2 for axis in edges_um)
    shape = tuple(len(axis) for axis in coords)
    epsilon = torch.full(shape, 1.44**2)
    core = (np.abs(coords[1][:, None]) < 0.4) & (np.abs(coords[2][None, :]) < 0.2)
    epsilon[:, core] = 3.5**2
    native = dict(shape=shape, boundaries=edges_um, coords=coords, is_uniform=False)
    metadata = dict(native=native, export=native, native_equals_export=False)
    source = fdtdx.ModePlaneSource(
        name="input_mode",
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, 1.6e-6, 1.6e-6),
        partial_real_position=(-0.5e-6, 0.0, 0.0),
        wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6),
        direction="+",
        mode_index=0,
    )
    monitor = fdtdx.PhasorDetector(
        name="input_recorder",
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, 1.6e-6, 1.6e-6),
        partial_real_position=(-0.5e-6, 0.0, 0.0),
        wave_characters=(source.wave_character,),
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        exact_interpolation=True,
    )
    key = ((1.55, 0.0, 1), "Hz1")
    profiles = {
        "input": {key: (source,)},
        "input_field_monitor": {key + ("port:input",): (monitor,)},
    }
    simulation = fdtd3d(
        wl_cen=1.55,
        wl_width=0.0,
        n_wl=1,
        dL=0.2e-6,
        eps_r=epsilon,
        port_sources_dict=profiles,
        port_monitor_slices={"input": (slice(5, 6), slice(3, 11), slice(2, 10))},
        port_monitor_slices_info={
            "input": dict(
                center=(-0.5, 0.0, 0.0),
                size=(0.0, 1.6, 1.6),
                direction="x+",
            )
        },
        optical_grid_metadata=metadata,
        npml=((0, 0), (0, 0), (0, 0)),
        max_time=8e-15,
        device="cpu",
    )
    arrays, objects, config, rng, monitors, *_ = simulation.prepare_forward_simulation(
        "input", 1.55, 0.0, 1, "Hz1"
    )
    placed = objects[source.name]
    expected = ((5, 6), (3, 11), (2, 10))
    assert placed.grid_slice_tuple == expected
    assert objects[monitor.name].grid_slice_tuple == expected
    assert tuple(placed._E.shape) == (3, 1, 8, 8)
    assert tuple(placed._H.shape) == (3, 1, 8, 8)
    assert bool(jnp.all(jnp.isfinite(placed._E)))
    assert bool(jnp.all(jnp.isfinite(placed._H)))
    assert float(jnp.max(jnp.abs(placed._E))) > 0
    for axis in (1, 2):
        lower, upper = expected[axis]
        assert edges_um[axis][lower] == pytest.approx(-0.8)
        assert edges_um[axis][upper] == pytest.approx(0.8)
    _, propagated = fdtdx.run_fdtd(
        arrays=arrays, objects=objects, config=config, key=rng
    )
    assert bool(jnp.all(jnp.isfinite(propagated.fields.E)))
    assert bool(jnp.all(jnp.isfinite(propagated.fields.H)))
    assert float(jnp.max(jnp.abs(propagated.fields.E))) > 0
    assert profiles["input"][key][0] is source
    assert source.partial_grid_shape == (1, None, None)
