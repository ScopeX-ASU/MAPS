"""Fast straight-waveguide forward sanity check for the pinned public FDTD-X backend."""

from __future__ import annotations

import pytest

fdtdx = pytest.importorskip("fdtdx")
jax = pytest.importorskip("jax")
import jax.numpy as jnp


@pytest.mark.fdtdx
@pytest.mark.numerical
def test_straight_waveguide_mode_source_produces_finite_guided_fields() -> None:
    """Exercise material placement, mode insertion, and forward propagation in a tiny guide."""
    key = jax.random.PRNGKey(19)
    wavelength = 1.55e-6
    propagation_axis = 0
    transverse_axis = 1
    vertical_axis = 2

    config = fdtdx.SimulationConfig(
        time=8e-15,
        grid=fdtdx.UniformGrid(spacing=200e-9),
        dtype=jnp.float32,
        courant_factor=0.8,
        backend="cpu",
    )
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(3.2e-6, 2.4e-6, 2.4e-6),
        material=fdtdx.Material(permittivity=2.0),
    )

    boundary_config = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic")
    boundary_objects, constraints = fdtdx.boundary_objects_from_config(
        boundary_config, volume
    )

    guide_shape: list[float | None] = [None, None, None]
    guide_shape[transverse_axis] = 0.8e-6
    guide_shape[vertical_axis] = 0.4e-6
    waveguide = fdtdx.UniformMaterialObject(
        partial_real_shape=tuple(guide_shape),
        material=fdtdx.Material(permittivity=12.0),
    )
    constraints.extend(
        [
            waveguide.place_at_center(volume, axes=transverse_axis),
            waveguide.place_at_center(volume, axes=vertical_axis),
        ]
    )

    source_shape: list[int | None] = [None, None, None]
    source_shape[propagation_axis] = 1
    source = fdtdx.ModePlaneSource(
        partial_grid_shape=tuple(source_shape),
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength),
        direction="+",
        mode_index=0,
        filter_pol=None,
    )
    constraints.append(
        source.place_relative_to(
            waveguide,
            axes=(propagation_axis,),
            own_positions=(-1,),
            other_positions=(-1,),
            grid_margins=(3,),
        )
    )

    object_list = [volume, *boundary_objects.values(), waveguide, source]
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
