"""Identical-input complex-field checks against optional Tidy3D local projection.

No cloud simulation or external time-domain solver is required. Coordinates are
converted from meters to Tidy3D micrometers; E/H retain their common amplitude
scale. Trapezoidal quadrature, resampling, normal, and origin are matched.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
td = pytest.importorskip("tidy3d")

from core.nearfield import Surface, project_farfield, project_nearfield

pytestmark = [pytest.mark.integration, pytest.mark.numerical]


def _weights(x):
    """Return nodal trapezoidal weights independently of the projector."""
    widths = np.diff(x)
    return np.r_[widths[0] / 2, (widths[:-1] + widths[1:]) / 2, widths[-1] / 2]


def _cartesian(data, points, prefix):
    """Convert Tidy3D radial/polar/azimuthal fields to Cartesian arrays."""
    r = np.linalg.norm(points, axis=-1)
    theta = np.arccos(points[:, 2] / r)
    phi = np.arctan2(points[:, 1], points[:, 0])
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    basis = np.array(
        [[st * cp, st * sp, ct], [ct * cp, ct * sp, -st], [-sp, cp, np.zeros_like(sp)]]
    ).transpose(2, 0, 1)
    components = np.stack(
        [
            np.asarray(getattr(data, prefix + name)).reshape(-1)
            for name in ("r", "theta", "phi")
        ],
        axis=-1,
    )
    return np.einsum("mc,mcd->md", components, basis)


@pytest.mark.parametrize("far_field", [False, True])
@pytest.mark.parametrize("normal_sign", [-1, 1])
@pytest.mark.parametrize("eps_r", [1.0, 2.25])
def test_same_near_fields_match_all_six_complex_components(
    far_field, normal_sign, eps_r
):
    """Match phase, vector components, physical scale, and both normal signs."""
    if not hasattr(td.FieldProjector, "from_near_field_data"):
        pytest.skip("Installed Tidy3D lacks the independent raw FieldData constructor")
    axes_um = (
        np.linspace(-0.43, 0.57, 7),
        np.linspace(-0.36, 0.42, 9),
        np.array([0.13]),
    )
    x, y, z = np.meshgrid(*axes_um, indexing="ij")
    fields = []
    for component in range(6):
        # Independent smooth samples exercise longitudinal and both polarizations.
        values = ((component + 1) / 6) * np.exp(
            1j * ((component + 0.7) * x + (2.3 - component) * y)
        )
        fields.append(values if component < 3 else values / 377)
    frequency = 193.4e12
    monitor = td.FieldMonitor(
        center=(0.07, 0.03, 0.13),
        size=(1.0, 0.78, 0),
        freqs=[frequency],
        name="near",
        colocate=True,
    )
    coords = dict(zip(("x", "y", "z"), axes_um)) | {"f": [frequency]}
    data = td.FieldData(
        monitor=monitor,
        **{
            name: td.ScalarFieldDataArray(values[..., None], coords=coords)
            for name, values in zip(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), fields)
        },
    )
    xs, ys = (
        ([-43.0, 29.0], [-31.0, 11.0, 51.0])
        if far_field
        else ([-0.23, 0.19], [-0.17, 0.11, 0.31])
    )
    distance = 120.0 if far_field else 1.9
    projector, target = td.FieldProjector.from_near_field_data(
        data,
        name="projected",
        x=xs,
        y=ys,
        proj_axis=2,
        proj_distance=distance,
        normal_dir="+" if normal_sign > 0 else "-",
        medium=td.Medium(permittivity=eps_r),
        pts_per_wavelength=None,
        origin=(0, 0, 0),
        far_field_approx=far_field,
    )
    reference = projector.project_fields(target, verbose=False)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    points_um = np.stack((xx.ravel(), yy.ravel(), np.full(xx.size, distance)), axis=-1)
    positions = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=-1) * 1e-6
    weights = np.outer(_weights(axes_um[0]), _weights(axes_um[1])).ravel() * 1e-12
    surface = Surface(
        torch.tensor(positions),
        torch.tensor([0.0, 0.0, float(normal_sign)])
        .expand(len(positions), -1)
        .double(),
        torch.tensor(weights),
        torch.tensor(np.stack(fields[:3], axis=-1).reshape(1, -1, 3)),
        torch.tensor(np.stack(fields[3:], axis=-1).reshape(1, -1, 3)),
        torch.tensor([frequency], dtype=torch.float64),
    )
    operation = project_farfield if far_field else project_nearfield
    points_m = torch.tensor(points_um * 1e-6)
    if far_field:
        from core.nearfield.projector import C_0

        radius = torch.linalg.vector_norm(points_m, dim=-1)
        actual = operation(surface, points_m / radius[:, None], eps_r=eps_r)
        radial_factor = (
            torch.exp(1j * (2 * np.pi * frequency / C_0 * np.sqrt(eps_r)) * radius)
            / radius
        )
        actual = tuple(value * radial_factor[None, :, None] for value in actual)
    else:
        actual = operation(surface, points_m, eps_r=eps_r)
    for prefix, value in zip(("E", "H"), actual):
        expected = _cartesian(reference, points_um, prefix)
        relative = np.linalg.norm(
            value[0].detach().numpy() - expected
        ) / np.linalg.norm(expected)
        assert relative < 2e-5, (prefix, far_field, normal_sign, eps_r, relative)
