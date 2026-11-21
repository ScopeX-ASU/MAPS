"""Unit tests for orientation- and symmetry-invariant 3D surface power."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jnp = pytest.importorskip("jax.numpy")

# Register MAPS' overrides before importing numerical entry points from the
# installed public FDTD-X package.
pytest.importorskip("thirdparty.fdtdx")

from fdtdx.core.physics.metrics import (
    compute_integrated_power,
    normalize_by_poynting_flux,
)

from core.utils import (
    build_rectilinear_grid_metadata,
    get_eigenmode_coefficients_3d,
    get_flux_3d,
    unfold_phasor,
)

pytestmark = [pytest.mark.fdtdx, pytest.mark.unit]


def _metadata_from_boundaries(
    boundaries: tuple[np.ndarray, np.ndarray, np.ndarray],
):
    """Build rectilinear metadata whose coordinates are cell centers."""
    coords = tuple(0.5 * (axis[:-1] + axis[1:]) for axis in boundaries)
    return build_rectilinear_grid_metadata(coords=coords, boundaries=boundaries)


def _constant_cross_axis_fields(
    shape: tuple[int, int, int],
) -> tuple[torch.Tensor, ...]:
    """Return fields with unit time-averaged flux density along x and z."""
    zeros = torch.zeros(shape, dtype=torch.complex128)
    ones = torch.ones(shape, dtype=torch.complex128)
    twos = 2 * ones
    return ones, ones, zeros, zeros, twos, twos


def test_fdtdx_source_normalization_uses_absolute_cell_areas() -> None:
    """The installed FDTD-X dependency must normalize physical integrated power."""
    electric = jnp.zeros((3, 1, 2, 2), dtype=jnp.float32).at[1].set(1.0)
    magnetic = jnp.zeros((3, 1, 2, 2), dtype=jnp.float32).at[2].set(2.0)
    areas = jnp.asarray(
        [[[0.1e-12, 0.2e-12], [0.3e-12, 0.4e-12]]],
        dtype=jnp.float32,
    )

    electric, magnetic = normalize_by_poynting_flux(
        electric,
        magnetic,
        axis=0,
        area_weights=areas,
    )
    power = compute_integrated_power(
        electric,
        magnetic,
        axis=0,
        area_weights=areas,
    )

    assert float(power) == pytest.approx(1.0, rel=1e-5, abs=1e-6)


def test_surface_power_uses_one_area_scale_across_monitor_orientations() -> None:
    """Cross-axis powers must retain physical plane-area ratios."""
    metadata = _metadata_from_boundaries(
        (
            np.array([0.0, 0.2, 0.5]),
            np.array([0.0, 0.1, 0.5, 1.0]),
            np.array([0.0, 0.25, 1.0]),
        )
    )
    fields = _constant_cross_axis_fields((2, 3, 2))

    power_x = get_flux_3d(
        *fields,
        grid_step=1.0,
        monitor=(slice(0, 1), slice(None), slice(None)),
        direction="x",
        grid_metadata=metadata,
    )
    power_z = get_flux_3d(
        *fields,
        grid_step=1.0,
        monitor=(slice(None), slice(None), slice(0, 1)),
        direction="z",
        grid_metadata=metadata,
    )

    # The yz plane is 1 um^2 and the xy plane is 0.5 um^2, while both have
    # unit Poynting-flux density in SI units.
    torch.testing.assert_close(power_x, torch.tensor(1e-12, dtype=power_x.dtype))
    torch.testing.assert_close(power_z, torch.tensor(0.5e-12, dtype=power_z.dtype))


def test_mode_self_overlap_uses_physical_unit_power_normalization() -> None:
    """A physically unit-power mode must have unit self-overlap on any mesh."""
    metadata = _metadata_from_boundaries(
        (
            np.array([0.0, 0.4]),
            np.array([0.0, 0.2, 0.8, 2.0]),
            np.array([0.0, 0.25, 1.0]),
        )
    )
    # The plane area is 2 um^2, so scale both E and H by 1/sqrt(2e-12)
    # to obtain one watt under the physical-area quadrature.
    fields = tuple(
        field / np.sqrt(2e-12) for field in _constant_cross_axis_fields((1, 3, 2))
    )
    mode_e = torch.stack(fields[:3], dim=0)
    mode_h = torch.stack(fields[3:], dim=0)

    forward, backward = get_eigenmode_coefficients_3d(
        *fields,
        ht_m=mode_h,
        et_m=mode_e,
        monitor=(slice(0, 1), slice(None), slice(None)),
        energy=True,
        direction="x+",
        grid_metadata=metadata,
    )

    torch.testing.assert_close(forward, torch.tensor(1.0, dtype=forward.dtype))
    torch.testing.assert_close(backward, torch.tensor(0.0, dtype=backward.dtype))


def test_symmetry_unfolding_and_inverse_sqrt_scaling_preserve_power() -> None:
    """Unfolding a reduced mode and scaling fields by 1/sqrt(2) preserves power."""
    reduced_metadata = _metadata_from_boundaries(
        (
            np.array([0.0, 0.4]),
            np.array([0.0, 0.2, 0.5]),
            np.array([0.0, 0.3, 1.0]),
        )
    )
    full_metadata = _metadata_from_boundaries(
        (
            np.array([0.0, 0.4]),
            np.array([-0.5, -0.2, 0.0, 0.2, 0.5]),
            np.array([0.0, 0.3, 1.0]),
        )
    )
    reduced_fields = _constant_cross_axis_fields((1, 2, 2))
    reduced_power = get_flux_3d(
        *reduced_fields,
        grid_step=1.0,
        monitor=(slice(0, 1), slice(None), slice(None)),
        direction="x",
        grid_metadata=reduced_metadata,
    )

    components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    reduced_phasor = torch.stack(reduced_fields, dim=0).unsqueeze(0)
    monitor = SimpleNamespace(
        _grid_slice_tuple=((0, 1), (0, 2), (0, 2)),
        components=components,
        reduce_volume=False,
    )
    config = SimpleNamespace(has_symmetry=True, symmetry=(0, -1, 0))
    full_phasor = unfold_phasor(
        reduced_phasor,
        monitor=monitor,
        components=components,
        config=config,
    ) / np.sqrt(2.0)
    full_fields = tuple(full_phasor[0, component] for component in range(6))
    full_power = get_flux_3d(
        *full_fields,
        grid_step=1.0,
        monitor=(slice(0, 1), slice(None), slice(None)),
        direction="x",
        grid_metadata=full_metadata,
    )

    torch.testing.assert_close(full_power, reduced_power)


def test_uniform_fallback_uses_physical_grid_step_area() -> None:
    """Omitting rectilinear metadata must fall back to (grid_step * 1e-6)^2 m^2."""
    grid_step = (
        0.05  # um -> cell area = 2.5e-15 m^2 on a 2x3 plane -> total 1.5e-14 m^2
    )
    total_area_m2 = 6.0 * (grid_step * 1e-6) ** 2
    fields = tuple(
        field / np.sqrt(total_area_m2)
        for field in _constant_cross_axis_fields((1, 2, 3))
    )
    mode_e = torch.stack(fields[:3], dim=0)
    mode_h = torch.stack(fields[3:], dim=0)
    monitor = (slice(0, 1), slice(None), slice(None))

    flux = get_flux_3d(
        *fields,
        grid_step=grid_step,
        monitor=monitor,
        direction="x",
        grid_metadata=None,
        cell_weights=None,
    )
    power_p, power_m = get_eigenmode_coefficients_3d(
        *fields,
        ht_m=mode_h,
        et_m=mode_e,
        monitor=monitor,
        grid_step=grid_step,
        energy=True,
        direction="x+",
        grid_metadata=None,
        cell_weights=None,
    )
    amp_p, amp_m = get_eigenmode_coefficients_3d(
        *fields,
        ht_m=mode_h,
        et_m=mode_e,
        monitor=monitor,
        grid_step=grid_step,
        energy=False,
        direction="x+",
        grid_metadata=None,
        cell_weights=None,
    )

    torch.testing.assert_close(flux, torch.tensor(1.0, dtype=flux.dtype))
    torch.testing.assert_close(power_p, torch.tensor(1.0, dtype=power_p.dtype))
    torch.testing.assert_close(power_m, torch.tensor(0.0, dtype=power_m.dtype))
    torch.testing.assert_close(amp_p.abs(), torch.tensor(1.0, dtype=amp_p.real.dtype))
    assert amp_m == 0


def test_smatrix_objective_3d_respects_energy_flag() -> None:
    """3D SMatrixObjective must return modal power when energy=True and amplitude when False."""
    from core.invdes.models.layers.objective import SMatrixObjective

    metadata = _metadata_from_boundaries(
        (
            np.array([0.0, 0.2]),
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.0]),
        )
    )
    # Unit-power reference mode on a 1 um^2 yz plane; scale simulated fields by 0.5 in amplitude (0.25 in power).
    unit_fields = tuple(
        field / np.sqrt(1e-12) for field in _constant_cross_axis_fields((1, 2, 2))
    )
    mode_e = torch.stack(unit_fields[:3], dim=0).unsqueeze(0)
    mode_h = torch.stack(unit_fields[3:], dim=0).unsqueeze(0)
    sim_fields = {
        ("in_slice_1", 1.55, "Hz1", 300): {
            name: 0.5 * tensor
            for name, tensor in zip(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), unit_fields)
        }
    }
    port_profiles = {
        "in_slice_1": {((1.55, 0.0, 1), "Hz1"): [None, None, None, 1.0, True]},
        "out_slice_1": {(1.55, "Hz1"): [None, mode_h, mode_e, 1.0, False]},
    }
    port_slices = {"out_slice_1": (slice(0, 1), slice(None), slice(None))}
    sims = {((1.55, 0.0, 1), "Hz", 300): object()}

    s_amp = SMatrixObjective(
        sims=sims,
        s_params={},
        port_profiles=port_profiles,
        port_slices=port_slices,
        in_slice_names=("in_slice_1",),
        out_slice_names=("out_slice_1",),
        in_mode="Hz1",
        out_modes=("Hz1",),
        directions=("x+",),
        name="s_amp",
        target_wls=(1.55,),
        target_temps=(300,),
        grid_step=0.2,
        energy=False,
        grid_metadata=metadata,
    )(sim_fields)
    s_pow = SMatrixObjective(
        sims=sims,
        s_params={},
        port_profiles=port_profiles,
        port_slices=port_slices,
        in_slice_names=("in_slice_1",),
        out_slice_names=("out_slice_1",),
        in_mode="Hz1",
        out_modes=("Hz1",),
        directions=("x+",),
        name="s_pow",
        target_wls=(1.55,),
        target_temps=(300,),
        grid_step=0.2,
        energy=True,
        grid_metadata=metadata,
    )(sim_fields)

    torch.testing.assert_close(s_amp.abs(), torch.tensor([0.5], dtype=s_amp.real.dtype))
    torch.testing.assert_close(s_pow, torch.tensor([0.25], dtype=s_pow.dtype))


def test_fdtdx_source_builders_use_unit_default_amplitude_calibration() -> None:
    """Mode and Gaussian 3D source builders must launch unit amplitude factor by default."""
    from core.invdes.models.layers.device_base import N_Ports
    from core.utils import Slice3D

    dummy = SimpleNamespace(
        port_monitor_slices_info={
            "in_slice_1": {"size": [0.0, 2.0, 1.5], "center": [0.0, 0.0, 0.0]},
            "top_slice": {"size": [3.0, 3.0, 0.0], "center": [0.0, 0.0, 1.0]},
        },
        cell_extend=[[-2.0, 2.0], [-2.0, 2.0], [-1.5, 1.5]],
        sim_cfg={"symmetry": (0, -1, 0)},
    )
    dummy._check_object_symmetry = (
        lambda *args, **kwargs: N_Ports._check_object_symmetry(dummy, *args, **kwargs)
    )

    mode_profiles = N_Ports.insert_modes_fdtd3d(
        dummy,
        source_slice_name="in_slice_1",
        eps=None,
        slice=Slice3D(0, slice(0, 10), slice(0, 8)),
        wl_cen=1.55,
        wl_width=0.0,
        n_wl=1,
        source_modes=("Hz1",),
        direction="x+",
    )
    mode_src, _, _, norm_power = mode_profiles[((1.55, 0.0, 1), "Hz1")]
    assert norm_power == 1
    assert float(mode_src.static_amplitude_factor) == pytest.approx(1.0 / np.sqrt(2.0))

    gauss_profiles = N_Ports.insert_gaussian_beam_fdtd3d(
        dummy,
        source_slice_name="top_slice",
        eps=None,
        slice=Slice3D(slice(0, 10), slice(0, 10), 0),
        wl_cen=1.55,
        wl_width=0.0,
        n_wl=1,
        source_modes=("Hz1", "Hz2"),
        waist_radii=(1.0e-6, 1.2e-6),
        waist_distances=(0.0, -0.5e-6),
        direction="z-",
    )
    assert set(gauss_profiles.keys()) == {
        ((1.55, 0.0, 1), "Hz1"),
        ((1.55, 0.0, 1), "Hz2"),
    }
    for mode_label in ("Hz1", "Hz2"):
        gauss_src, _, _, gauss_norm = gauss_profiles[((1.55, 0.0, 1), mode_label)]
        assert gauss_norm == 1
        assert float(gauss_src.static_amplitude_factor) == pytest.approx(1.0)
