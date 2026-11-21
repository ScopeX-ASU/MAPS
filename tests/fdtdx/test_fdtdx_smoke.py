"""Small deterministic FDTDX end-to-end regression."""

from __future__ import annotations

import pytest

fdtdx = pytest.importorskip("fdtdx")
jax = pytest.importorskip("jax")
import jax.numpy as jnp


@pytest.mark.fdtdx
@pytest.mark.numerical
def test_small_periodic_fdtd_run_produces_finite_fields() -> None:
    """Run a tiny CPU FDTD simulation and validate the Yee fields."""
    key = jax.random.PRNGKey(7)
    config = fdtdx.SimulationConfig(
        time=4e-15,
        grid=fdtdx.UniformGrid(spacing=200e-9),
        dtype=jnp.float32,
        courant_factor=0.8,
        backend="cpu",
    )

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(2.4e-6, 2.4e-6, 2.4e-6),
        material=fdtdx.Material(permittivity=2.0),
    )
    boundary_config = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic")
    boundary_objects, boundary_constraints = fdtdx.boundary_objects_from_config(
        boundary_config,
        volume,
    )

    source = fdtdx.GaussianPlaneSource(
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(2.0e-6, 2.0e-6, None),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6),
        radius=0.8e-6,
        std=1 / 3,
        direction="-",
        elevation_angle=0.0,
    )

    constraints = [
        *boundary_constraints,
        source.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, 0),
        ),
    ]
    object_list = [volume, *boundary_objects.values(), source]

    key, subkey = jax.random.split(key)
    objects, arrays, _, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=constraints,
        key=subkey,
    )

    _, arrays = fdtdx.run_fdtd(
        arrays=arrays,
        objects=objects,
        config=config,
        key=subkey,
    )

    electric = arrays.fields.E
    magnetic = arrays.fields.H

    assert electric.shape[0] == 3
    assert magnetic.shape[0] == 3
    assert bool(jnp.all(jnp.isfinite(electric)))
    assert bool(jnp.all(jnp.isfinite(magnetic)))
    assert float(jnp.max(jnp.abs(electric))) > 0.0
    assert float(jnp.max(jnp.abs(magnetic))) > 0.0
