"""Tidy3D-compatible monitor and projection facade for MAPS near fields.

Geometry and observation coordinates use micrometers at this public boundary;
the numerical projection kernels use SI units internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .observations import direction_cosines, spherical_components
from .projector import C_0, Surface, concatenate_surfaces, project_farfield

_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_FACE_NAMES = ("x-", "x+", "y-", "y+", "z-", "z+")


def _triple(value, name, *, positive=False):
    result = tuple(float(v) for v in value)
    if len(result) != 3 or any(not torch.isfinite(torch.tensor(v)) for v in result):
        raise ValueError(f"{name} must contain three finite values")
    if positive and any(v < 0 for v in result):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _coords(value, name):
    tensor = torch.as_tensor(value)
    if tensor.is_complex() or not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    if tensor.ndim != 1 or tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be a nonempty finite coordinate vector")
    return tensor


def _trapezoid_weights(axis: Tensor) -> Tensor:
    """Return physical composite-trapezoid nodal weights."""
    if axis.numel() == 1:
        return torch.ones_like(axis)
    delta = axis[1:] - axis[:-1]
    weights = torch.empty_like(axis)
    weights[0], weights[-1] = delta[0] / 2, delta[-1] / 2
    if axis.numel() > 2:
        weights[1:-1] = (delta[:-1] + delta[1:]) / 2
    return weights


def _resample_axis(values: Tensor, old: Tensor, new: Tensor, dim: int) -> Tensor:
    """Low-pass by physical overlap on reduction; interpolate otherwise."""
    if old.numel() == 1 or torch.equal(old, new):
        return values
    if new.numel() < old.numel():

        def control_edges(axis):
            return torch.cat((axis[:1], (axis[:-1] + axis[1:]) / 2, axis[-1:]))

        source_edges, target_edges = control_edges(old), control_edges(new)
        overlap = (
            torch.minimum(target_edges[1:, None], source_edges[None, 1:])
            - torch.maximum(target_edges[:-1, None], source_edges[None, :-1])
        ).clamp_min(0)
        target_width = target_edges[1:] - target_edges[:-1]
        averaging = (overlap / target_width[:, None]).transpose(0, 1).to(values.dtype)
        return torch.matmul(values.movedim(dim, -1), averaging).movedim(-1, dim)
    upper = torch.searchsorted(old.contiguous(), new.contiguous()).clamp(
        1, old.numel() - 1
    )
    lower = upper - 1
    fraction = (new - old[lower]) / (old[upper] - old[lower])
    left, right = values.index_select(dim, lower), values.index_select(dim, upper)
    shape = [1] * values.ndim
    shape[dim] = new.numel()
    weight = fraction.reshape(shape)
    return left + (right - left) * weight


def _resample_surface(
    surface: Surface,
    monitor: FieldMonitor,
    pts_per_wavelength: int | None,
    eps_r: float | Tensor,
    mu_r: float | Tensor,
) -> Surface:
    """Resample a verified plane, rejecting point clouds without a full grid."""
    if pts_per_wavelength is None:
        return surface
    axis = monitor.size.index(0.0)
    transverse = tuple(i for i in range(3) if i != axis)
    positions = surface.positions_m
    coords = tuple(torch.unique(positions[:, i], sorted=True) for i in transverse)
    counts = (coords[0].numel(), coords[1].numel())
    if counts[0] * counts[1] != positions.shape[0]:
        raise NotImplementedError(
            f"monitor {monitor.name!r} samples are not a complete rectilinear plane; "
            "wavelength-based resampling requires every transverse coordinate pair"
        )
    # Integer lookup tolerates small coordinate roundoff while rejecting duplicate
    # or missing grid points. The lookup itself is discrete; gathered phasors
    # remain connected to autograd.
    indices = []
    for coordinate, values in zip(transverse, coords):
        index = torch.searchsorted(
            values.contiguous(), positions[:, coordinate].contiguous()
        )
        index = index.clamp_max(values.numel() - 1)
        tolerance = (
            torch.finfo(positions.dtype).eps
            * 16
            * torch.maximum(values.abs().max(), values.new_tensor(1.0))
        )
        if not torch.allclose(
            values[index], positions[:, coordinate], atol=float(tolerance), rtol=0
        ):
            raise NotImplementedError(
                f"monitor {monitor.name!r} does not form a rectilinear plane; "
                "wavelength-based resampling requires Cartesian-product coordinates"
            )
        indices.append(index)
    linear = indices[0] * counts[1] + indices[1]
    if torch.unique(linear).numel() != positions.shape[0]:
        raise NotImplementedError(
            f"monitor {monitor.name!r} has duplicate plane coordinates; "
            "wavelength-based resampling requires a unique rectilinear grid"
        )
    if bool((positions[:, axis] - positions[0, axis]).abs().max() > 1e-10):
        raise NotImplementedError(
            "surface samples are not planar on the declared monitor axis"
        )
    fields = torch.cat((surface.electric_fields, surface.magnetic_fields), dim=-1)
    grid = fields.new_zeros((fields.shape[0], counts[0], counts[1], 6))
    grid[:, indices[0], indices[1], :] = fields
    eps_value = torch.as_tensor(eps_r, device=positions.device)
    mu_value = torch.as_tensor(mu_r, device=positions.device)
    if eps_value.is_complex() or mu_value.is_complex():
        raise ValueError("eps_r and mu_r must be real for a lossless projection medium")
    medium = eps_value.to(positions.dtype) * mu_value.to(positions.dtype)
    if medium.numel() != 1 or not torch.isfinite(medium).all() or medium <= 0:
        raise ValueError(
            "eps_r and mu_r must define a positive finite homogeneous medium"
        )
    wavelength = 299792458.0 / (surface.frequencies_hz.max() * torch.sqrt(medium))
    target_step = wavelength / pts_per_wavelength
    new_coords = []
    for coordinate in coords:
        if coordinate.numel() == 1:
            new_coords.append(coordinate)
        else:
            intervals = torch.ceil(
                (coordinate[-1] - coordinate[0]) / target_step - 1e-10
            )
            count = max(
                2,
                int(intervals.item()) + 1,
            )
            new_coords.append(
                torch.linspace(
                    coordinate[0],
                    coordinate[-1],
                    count,
                    device=positions.device,
                    dtype=positions.dtype,
                )
            )
    grid = _resample_axis(grid, coords[0], new_coords[0], 1)
    grid = _resample_axis(grid, coords[1], new_coords[1], 2)
    mesh = torch.meshgrid(*new_coords, indexing="ij")
    new_positions = positions.new_zeros((mesh[0].numel(), 3))
    for i, coordinate in zip(transverse, mesh):
        new_positions[:, i] = coordinate.reshape(-1)
    new_positions[:, axis] = positions[:, axis].mean()
    u_weights, v_weights = (_trapezoid_weights(coord) for coord in new_coords)
    weights = (u_weights[:, None] * v_weights[None, :]).reshape(-1)
    # A one-node transverse dimension has no width encoded by coordinates;
    # preserve the supplied total physical area for that degenerate case.
    if any(coord.numel() == 1 for coord in new_coords):
        weights = weights * (surface.weights_m2.sum() / weights.sum())
    return Surface(
        new_positions,
        surface.normals[0].expand(new_positions.shape[0], 3),
        weights,
        grid[..., :3].reshape(surface.electric_fields.shape[0], -1, 3),
        grid[..., 3:].reshape(surface.magnetic_fields.shape[0], -1, 3),
        surface.frequencies_hz,
    )


@dataclass(frozen=True)
class FieldMonitor:
    """Describe a sampled MAPS near-field monitor in micrometers and hertz.

    Args:
        center: Monitor center in micrometers.
        size: Nonnegative extent in micrometers.
        freqs: Positive frequencies in hertz.
        name: Key used to resolve monitor data.
        fields: Recorded Cartesian field components.
        interval_space: Positive integer recording stride on x, y, z.
        colocate: Request colocated output. MAPS supports only ``True``.
        use_colocated_integration: Request integration on the colocated grid.
            MAPS supports only ``True`` for this facade.
    """

    center: Sequence[float] = (0.0, 0.0, 0.0)
    size: Sequence[float] = (0.0, 0.0, 0.0)
    freqs: Sequence[float] = ()
    name: str = ""
    fields: Sequence[str] = _FIELDS
    interval_space: Sequence[int] = (1, 1, 1)
    colocate: bool = True
    use_colocated_integration: bool = True

    def __post_init__(self):
        object.__setattr__(self, "center", _triple(self.center, "center"))
        object.__setattr__(self, "size", _triple(self.size, "size", positive=True))
        freq = tuple(float(x) for x in self.freqs)
        if not freq or any(x <= 0 or not torch.isfinite(torch.tensor(x)) for x in freq):
            raise ValueError("freqs must contain positive finite frequencies")
        if not self.name:
            raise ValueError("name is required")
        if not set(self.fields) <= set(_FIELDS) or len(set(self.fields)) != len(
            self.fields
        ):
            raise ValueError("fields must be a unique subset of Ex, Ey, Ez, Hx, Hy, Hz")
        stride = tuple(self.interval_space)
        if len(stride) != 3 or any(
            not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in stride
        ):
            raise ValueError("interval_space must contain three positive integers")
        if not self.colocate or not self.use_colocated_integration:
            raise ValueError(
                "MAPS facade requires colocate=True and use_colocated_integration=True"
            )
        object.__setattr__(self, "freqs", freq)
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "interval_space", stride)


@dataclass(frozen=True)
class FieldProjectionSurface:
    """Associate a planar field monitor with its outward normal direction.

    Args:
        monitor: Planar monitor recording the four tangential E/H components.
        normal_dir: Outward normal sign, either ``"+"`` or ``"-"`` along the
            monitor's zero-size axis.
    """

    monitor: FieldMonitor
    normal_dir: str

    @property
    def axis(self) -> int:
        """Return the global axis normal to the monitor plane (0, 1, or 2)."""
        return self.monitor.size.index(0.0)

    def __post_init__(self):
        if not isinstance(self.monitor, FieldMonitor):
            raise TypeError("monitor must be a FieldMonitor")
        if self.normal_dir not in ("+", "-"):
            raise ValueError("normal_dir must be '+' or '-'")
        if sum(v == 0 for v in self.monitor.size) != 1:
            raise ValueError(
                "a projection surface monitor must have exactly one zero size"
            )
        axis = self.monitor.size.index(0.0)
        tangential = (
            ("Ey", "Ez", "Hy", "Hz"),
            ("Ex", "Ez", "Hx", "Hz"),
            ("Ex", "Ey", "Hx", "Hy"),
        )[axis]
        if not set(tangential) <= set(self.monitor.fields):
            raise ValueError(
                f"monitor {self.monitor.name!r} must record tangential fields {tangential}"
            )


@dataclass(frozen=True)
class _ProjectionMonitor:
    center: Sequence[float]
    size: Sequence[float]
    freqs: Sequence[float]
    name: str
    normal_dir: str | None = None
    exclude_surfaces: Sequence[str] | None = None
    custom_origin: Sequence[float] | None = None
    far_field_approx: bool = True
    interval_space: Sequence[int] = (1, 1, 1)
    colocate: bool = True
    use_colocated_integration: bool = True
    window_size: Sequence[float] = (0.0, 0.0)
    medium: object | None = None

    def __post_init__(self):
        object.__setattr__(self, "center", _triple(self.center, "center"))
        object.__setattr__(self, "size", _triple(self.size, "size", positive=True))
        if not self.name:
            raise ValueError("name is required")
        frequencies = tuple(float(f) for f in self.freqs)
        if not frequencies or any(
            f <= 0 or not torch.isfinite(torch.tensor(f)) for f in frequencies
        ):
            raise ValueError("freqs must contain positive finite frequencies")
        object.__setattr__(self, "freqs", frequencies)
        if sum(v == 0 for v in self.size) not in (0, 1):
            raise ValueError("projection size must describe a plane or a closed box")
        if sum(v == 0 for v in self.size) == 1 and self.normal_dir not in ("+", "-"):
            raise ValueError("normal_dir is required for a planar projection monitor")
        invalid = set(self.exclude_surfaces or ()) - set(_FACE_NAMES)
        if invalid:
            raise ValueError(f"unknown excluded surfaces: {sorted(invalid)}")
        if self.custom_origin is not None:
            object.__setattr__(
                self, "custom_origin", _triple(self.custom_origin, "custom_origin")
            )
        stride = tuple(self.interval_space)
        if len(stride) != 3 or any(
            not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in stride
        ):
            raise ValueError("interval_space must contain three positive integers")
        object.__setattr__(self, "interval_space", stride)
        if not self.colocate or not self.use_colocated_integration:
            raise ValueError(
                "MAPS projection monitors require colocate=True and "
                "use_colocated_integration=True"
            )
        window = tuple(float(value) for value in self.window_size)
        if len(window) != 2 or any(value < 0 for value in window):
            raise ValueError("window_size must contain two nonnegative values")
        if any(value != 0 for value in window):
            raise NotImplementedError("nonzero projection window_size is not supported")
        object.__setattr__(self, "window_size", window)
        if self.medium is not None:
            raise NotImplementedError(
                "set FieldProjector eps_r and mu_r for medium selection"
            )

    @property
    def projection_surfaces(self):
        """Return monitor/normal pairs for each included plane or box face."""
        if sum(v == 0 for v in self.size) == 1:
            axis = self.size.index(0.0)
            fields = (
                ("Ey", "Ez", "Hy", "Hz"),
                ("Ex", "Ez", "Hx", "Hz"),
                ("Ex", "Ey", "Hx", "Hy"),
            )[axis]
            return (
                FieldProjectionSurface(
                    FieldMonitor(
                        self.center,
                        self.size,
                        self.freqs,
                        self.name,
                        fields=fields,
                        interval_space=self.interval_space,
                    ),
                    self.normal_dir,
                ),
            )
        output = []
        for axis in range(3):
            for sign in (-1, 1):
                face = f"{'xyz'[axis]}{'-' if sign < 0 else '+'}"
                if face in (self.exclude_surfaces or ()):
                    continue
                size = list(self.size)
                center = list(self.center)
                center[axis] += sign * size[axis] / 2
                size[axis] = 0.0
                tangential = (
                    ("Ey", "Ez", "Hy", "Hz"),
                    ("Ex", "Ez", "Hx", "Hz"),
                    ("Ex", "Ey", "Hx", "Hy"),
                )[axis]
                mon = FieldMonitor(
                    center,
                    size,
                    self.freqs,
                    f"{self.name}_{face}",
                    fields=tangential,
                    interval_space=self.interval_space,
                )
                output.append(FieldProjectionSurface(mon, "+" if sign > 0 else "-"))
        return tuple(output)


@dataclass(frozen=True)
class FieldProjectionAngleMonitor(_ProjectionMonitor):
    """Request fields on a spherical theta/phi grid.

    Args:
        theta: Polar angles from +z in radians.
        phi: Azimuthal angles from +x toward +y in radians.
        proj_distance: Radial distance from the projector origin in micrometers.
        Other arguments: Shared monitor geometry, frequencies, orientation,
            and projection options documented by ``FieldProjection*Monitor``.
    """

    theta: Sequence[float] = ()
    phi: Sequence[float] = ()
    proj_distance: float = 1e6

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "theta", _coords(self.theta, "theta"))
        object.__setattr__(self, "phi", _coords(self.phi, "phi"))
        if self.proj_distance <= 0:
            raise ValueError("proj_distance must be positive")


@dataclass(frozen=True)
class FieldProjectionCartesianMonitor(_ProjectionMonitor):
    """Request fields on a Cartesian plane in local micrometer coordinates.

    Args:
        proj_axis: Global axis normal to the output plane, 0, 1, or 2.
        x: First local transverse coordinate vector in micrometers.
        y: Second local transverse coordinate vector in micrometers.
        proj_distance: Signed displacement from the projector origin in
            micrometers.
        Other arguments: Shared monitor geometry, frequencies, orientation,
            and projection options documented by ``FieldProjection*Monitor``.
    """

    proj_axis: int = 2
    x: Sequence[float] = ()
    y: Sequence[float] = ()
    proj_distance: float = 1e6

    def __post_init__(self):
        super().__post_init__()
        if self.proj_axis not in (0, 1, 2):
            raise ValueError("proj_axis must be 0, 1, or 2")
        object.__setattr__(self, "x", _coords(self.x, "x"))
        object.__setattr__(self, "y", _coords(self.y, "y"))
        if self.proj_distance == 0:
            raise ValueError("proj_distance must be nonzero")


@dataclass(frozen=True)
class FieldProjectionKSpaceMonitor(_ProjectionMonitor):
    """Request propagating directions by transverse direction cosine.

    Args:
        proj_axis: Global axis normal to the direction-cosine plane, 0, 1, or 2.
        ux: First transverse direction-cosine vector.
        uy: Second transverse direction-cosine vector.
        proj_distance: Positive radial distance from the projector origin in
            micrometers.
        Other arguments: Shared monitor geometry, frequencies, orientation,
            and projection options documented by ``FieldProjection*Monitor``.
    """

    proj_axis: int = 2
    ux: Sequence[float] = ()
    uy: Sequence[float] = ()
    proj_distance: float = 1e6

    def __post_init__(self):
        super().__post_init__()
        if self.proj_axis not in (0, 1, 2):
            raise ValueError("proj_axis must be 0, 1, or 2")
        object.__setattr__(self, "ux", _coords(self.ux, "ux"))
        object.__setattr__(self, "uy", _coords(self.uy, "uy"))
        if self.proj_distance <= 0:
            raise ValueError("proj_distance must be positive")
        if (self.ux[:, None].square() + self.uy[None, :].square() > 1).any():
            raise ValueError(
                "ux squared plus uy squared must be at most one for propagating directions"
            )


@dataclass
class FieldProjectionData:
    """Torch-native spherical projection fields and explicit coordinates.

    The six field tensors have frequency last and preserve the source dtype,
    device, and autograd graph.

    Attributes:
        monitor: Projection monitor used to create the result.
        surfaces: Source monitor surfaces in projector order.
        coordinates: Explicit coordinates; spatial coordinates use micrometers
            and angular coordinates use radians.
        Er, Etheta, Ephi, Hr, Htheta, Hphi: Complex spherical components with
            frequency on the last axis.
    """

    monitor: _ProjectionMonitor
    surfaces: tuple[FieldProjectionSurface, ...]
    coordinates: Mapping[str, Tensor]
    Er: Tensor
    Etheta: Tensor
    Ephi: Tensor
    Hr: Tensor
    Htheta: Tensor
    Hphi: Tensor

    @property
    def projection_surfaces(self):
        """Return source surfaces (Tidy3D-compatible alias for ``surfaces``)."""
        return self.surfaces

    @property
    def power(self):
        """Return outward time-average radial Poynting flux in W/m²."""
        return (
            0.5 * (self.Etheta * self.Hphi.conj() - self.Ephi * self.Htheta.conj()).real
        )

    @property
    def directional_power(self):
        """Return radial power per solid angle in W/sr for angular monitors."""
        radial = self.coordinates["r"]
        if isinstance(self.monitor, FieldProjectionCartesianMonitor):
            raise ValueError(
                "directional_power is defined for angle and k-space monitors"
            )
        while radial.ndim < self.power.ndim:
            radial = radial.unsqueeze(-1)
        return self.power * (radial * 1e-6).square()

    @property
    def integrated_flux(self):
        """Integrate projected power over the sampled observation surface.

        Returns:
            Real per-frequency power in watts. Angular monitors use the
            spherical ``r² sin(theta) dtheta dphi`` measure, k-space monitors
            use the direction-cosine solid-angle Jacobian, and Cartesian
            monitors integrate normal Poynting flux over physical plane area.

        Raises:
            ValueError: If a Cartesian observation axis has fewer than two
                samples, so its physical quadrature cannot be determined.
        """
        if isinstance(self.monitor, FieldProjectionAngleMonitor):
            theta = self.monitor.theta.to(
                device=self.Er.device, dtype=self.Er.real.dtype
            )
            phi = self.monitor.phi.to(device=self.Er.device, dtype=self.Er.real.dtype)
            wt = _trapezoid_weights(theta)
            wp = _trapezoid_weights(phi)
            weights = torch.sin(theta).clamp_min(0)[:, None] * wt[:, None] * wp[None, :]
            return (self.directional_power * weights[None, ..., None]).sum(
                dim=(0, 1, 2)
            )
        if isinstance(self.monitor, FieldProjectionKSpaceMonitor):
            ux = self.monitor.ux.to(device=self.Er.device, dtype=self.Er.real.dtype)
            uy = self.monitor.uy.to(device=self.Er.device, dtype=self.Er.real.dtype)
            u, v = torch.meshgrid(ux, uy, indexing="ij")
            jacobian = torch.rsqrt((1 - u.square() - v.square()).clamp_min(1e-12))
            weights = (
                _trapezoid_weights(ux)[:, None]
                * _trapezoid_weights(uy)[None, :]
                * jacobian
            )
            return (self.directional_power * weights[:, :, None, None]).sum(
                dim=(0, 1, 2)
            )
        cartesian = self.cartesian_fields()
        axis = self.monitor.proj_axis
        electric = torch.stack(
            (cartesian["Ex"], cartesian["Ey"], cartesian["Ez"]), dim=-2
        )
        magnetic = torch.stack(
            (cartesian["Hx"], cartesian["Hy"], cartesian["Hz"]), dim=-2
        )
        flux = (
            0.5
            * torch.linalg.cross(electric, magnetic.conj(), dim=-2)[..., axis, :].real
        )
        flux = flux.squeeze(axis)
        x = self.monitor.x.to(device=flux.device, dtype=flux.dtype) * 1e-6
        y = self.monitor.y.to(device=flux.device, dtype=flux.dtype) * 1e-6
        if x.numel() < 2 or y.numel() < 2:
            raise ValueError(
                "Cartesian integrated_flux needs at least two samples on each transverse axis"
            )
        first = torch.trapezoid(flux, x=x, dim=0)
        orientation = 1.0 if self.monitor.proj_distance > 0 else -1.0
        return orientation * torch.trapezoid(first, x=y, dim=0)

    def cartesian_fields(self):
        """Convert spherical E/H values to Cartesian components.

        Returns:
            A mapping from ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, and ``Hz``
            to complex tensors with the same coordinate and frequency layout.
        """
        theta, phi = self.coordinates["theta"], self.coordinates["phi"]
        st, ct, sp, cp = (
            torch.sin(theta)[..., None],
            torch.cos(theta)[..., None],
            torch.sin(phi)[..., None],
            torch.cos(phi)[..., None],
        )
        er, et, ep = self.Er, self.Etheta, self.Ephi
        hr, ht, hp = self.Hr, self.Htheta, self.Hphi

        def convert(r, t, p):
            return (
                r * st * cp + t * ct * cp - p * sp,
                r * st * sp + t * ct * sp + p * cp,
                r * ct - t * st,
            )

        return {
            "Ex": convert(er, et, ep)[0],
            "Ey": convert(er, et, ep)[1],
            "Ez": convert(er, et, ep)[2],
            "Hx": convert(hr, ht, hp)[0],
            "Hy": convert(hr, ht, hp)[1],
            "Hz": convert(hr, ht, hp)[2],
        }

    def _cartesian_component(self, name: str) -> Tensor:
        """Resolve one Cartesian component for the convenience properties."""
        return self.cartesian_fields()[name]

    @property
    def Ex(self):
        """Return Cartesian x electric field."""
        return self._cartesian_component("Ex")

    @property
    def Ey(self):
        """Return Cartesian y electric field."""
        return self._cartesian_component("Ey")

    @property
    def Ez(self):
        """Return Cartesian z electric field."""
        return self._cartesian_component("Ez")

    @property
    def Hx(self):
        """Return Cartesian x magnetic field."""
        return self._cartesian_component("Hx")

    @property
    def Hy(self):
        """Return Cartesian y magnetic field."""
        return self._cartesian_component("Hy")

    @property
    def Hz(self):
        """Return Cartesian z magnetic field."""
        return self._cartesian_component("Hz")


@dataclass
class FieldProjector:
    """Project verified monitor fields to angle, Cartesian, or k-space grids.

    Args:
        sim_data: Mapping from monitor names to verified ``Surface`` samples.
            Raw solver outputs need explicit Yee metadata and are rejected.
        surfaces: Planar surface descriptions and outward normal signs.
        pts_per_wavelength: Optional current resampling density. ``None``
            preserves the verified input sampling unchanged.
        origin: Global phase origin in micrometers.
        eps_r: Positive lossless relative permittivity.
        mu_r: Positive lossless relative permeability.
        backend: Existing direct projection backend selection.
    """

    sim_data: Mapping[str, Surface]
    surfaces: Sequence[FieldProjectionSurface]
    pts_per_wavelength: int | None = 10
    origin: Sequence[float] | None = None
    eps_r: float | Tensor = 1.0
    mu_r: float | Tensor = 1.0
    backend: str = "auto"

    def __post_init__(self):
        if self.pts_per_wavelength is not None and (
            not isinstance(self.pts_per_wavelength, int)
            or isinstance(self.pts_per_wavelength, bool)
            or self.pts_per_wavelength <= 0
        ):
            raise ValueError("pts_per_wavelength must be a positive integer or None")
        self.surfaces = tuple(self.surfaces)
        if not self.surfaces:
            raise ValueError("at least one projection surface is required")
        if any(s.monitor.name not in self.sim_data for s in self.surfaces):
            missing = [
                s.monitor.name
                for s in self.surfaces
                if s.monitor.name not in self.sim_data
            ]
            raise KeyError(f"monitor data missing for {missing}")
        if any(
            not isinstance(self.sim_data[s.monitor.name], Surface)
            for s in self.surfaces
        ):
            raise ValueError(
                "Raw phasor data requires explicit component Yee coordinates and alignment metadata; pass verified Surface samples"
            )
        prepared = []
        for item in self.surfaces:
            surface = _resample_surface(
                self.sim_data[item.monitor.name],
                item.monitor,
                self.pts_per_wavelength,
                self.eps_r,
                self.mu_r,
            )
            axis = item.monitor.size.index(0.0)
            sign = 1 if item.normal_dir == "+" else -1
            # The facade's monitor normal is authoritative. Existing Surface
            # samples remain a low-level representation of the four measured
            # tangential components and their physical positions/weights.
            normals = torch.zeros_like(surface.normals)
            normals[:, axis] = sign
            prepared.append(
                Surface(
                    surface.positions_m,
                    normals,
                    surface.weights_m2,
                    surface.electric_fields,
                    surface.magnetic_fields,
                    surface.frequencies_hz,
                )
            )
        self._surface = concatenate_surfaces(prepared)
        if self.origin is None:
            self.origin = tuple(
                sum(s.monitor.center[i] for s in self.surfaces) / len(self.surfaces)
                for i in range(3)
            )
        if self.origin is not None:
            self.origin = _triple(self.origin, "origin")
        if self.backend not in ("auto", "direct_torch", "direct_triton"):
            raise ValueError(
                "backend must be 'auto', 'direct_torch', or 'direct_triton'"
            )
        self.origin = (
            torch.as_tensor(
                self.origin,
                device=self._surface.positions_m.device,
                dtype=self._surface.positions_m.dtype,
            )
            * 1e-6
        )

    @property
    def monitor_data(self):
        """Compatibility alias for ``sim_data``."""
        return self.sim_data

    @property
    def frequencies(self) -> Tensor:
        """Return the common source frequency grid in hertz."""
        return self._surface.frequencies_hz

    @classmethod
    def from_near_field_monitors(
        cls,
        sim_data,
        near_monitors,
        normal_dirs=None,
        *,
        pts_per_wavelength=10,
        origin=None,
        **kwargs,
    ):
        """Build a projector from monitor descriptions and outward signs.

        Args:
            sim_data: Mapping from monitor name to verified Surface samples.
            near_monitors: Ordered near-field monitor descriptions.
            normal_dirs: Outward signs corresponding to ``near_monitors``.
            pts_per_wavelength: Current resampling density, or ``None`` to
                preserve explicit samples.
            origin: Global phase origin in micrometers.
            **kwargs: Additional FieldProjector options such as medium and
                backend selection.

        Returns:
            A projector associated with all supplied monitors.

        Raises:
            ValueError: If monitor and normal counts differ or signs are invalid.
        """
        monitors = tuple(near_monitors)
        if normal_dirs is None:
            raise ValueError("normal_dirs is required")
        if len(monitors) != len(normal_dirs):
            raise ValueError("near_monitors and normal_dirs must have equal length")
        return cls(
            sim_data,
            tuple(FieldProjectionSurface(m, n) for m, n in zip(monitors, normal_dirs)),
            pts_per_wavelength,
            origin,
            **kwargs,
        )

    def project_fields(self, proj_monitor, verbose=True, freq_chunk_size=8):
        """Evaluate a supported projection monitor.

        Args:
            proj_monitor: Angle, Cartesian, or k-space output monitor.
            verbose: Accepted for Tidy3D call compatibility; currently unused.
            freq_chunk_size: Accepted for Tidy3D call compatibility; the
                underlying MAPS kernels currently handle frequency iteration.

        Returns:
            Torch-native spherical field data with frequency on the last axis.

        Raises:
            TypeError: If the projection monitor type is unsupported.
            ValueError: If its frequencies differ from the source frequency grid.
        """
        del verbose, freq_chunk_size
        if not isinstance(
            proj_monitor,
            (
                FieldProjectionAngleMonitor,
                FieldProjectionCartesianMonitor,
                FieldProjectionKSpaceMonitor,
            ),
        ):
            raise TypeError("unsupported projection monitor type")
        surface = self._surface
        device, dtype = surface.positions_m.device, surface.positions_m.dtype
        real_dtype = dtype
        requested_freqs = torch.as_tensor(
            proj_monitor.freqs, device=device, dtype=surface.frequencies_hz.dtype
        )
        if requested_freqs.shape != surface.frequencies_hz.shape or not torch.equal(
            requested_freqs, surface.frequencies_hz
        ):
            raise ValueError(
                "projection monitor freqs must match the source monitor frequency order"
            )
        origin = self.origin
        if proj_monitor.custom_origin is not None:
            monitor_origin = (
                torch.as_tensor(
                    proj_monitor.custom_origin, device=device, dtype=real_dtype
                )
                * 1e-6
            )
            if not torch.allclose(monitor_origin, origin):
                raise ValueError("custom_origin must match the FieldProjector origin")
        local_origin = origin
        if isinstance(proj_monitor, FieldProjectionAngleMonitor):
            theta = proj_monitor.theta.to(device=device, dtype=real_dtype)
            phi = proj_monitor.phi.to(device=device, dtype=real_dtype)
            th, ph = torch.meshgrid(theta, phi, indexing="ij")
            dirs = torch.stack(
                (
                    torch.sin(th) * torch.cos(ph),
                    torch.sin(th) * torch.sin(ph),
                    torch.cos(th),
                ),
                -1,
            ).reshape(-1, 3)
            radial = torch.full(
                (dirs.shape[0],),
                float(proj_monitor.proj_distance) * 1e-6,
                device=device,
                dtype=real_dtype,
            )
            coordinates = {
                "r": radial[:1] * 1e6,
                "theta": th[None],
                "phi": ph[None],
                "f": surface.frequencies_hz,
                "freqs": surface.frequencies_hz,
            }
        elif isinstance(proj_monitor, FieldProjectionKSpaceMonitor):
            ux, uy = (
                proj_monitor.ux.to(device=device, dtype=real_dtype),
                proj_monitor.uy.to(device=device, dtype=real_dtype),
            )
            u, v = torch.meshgrid(ux, uy, indexing="ij")
            dirs = direction_cosines(u, v, normal_axis=proj_monitor.proj_axis)
            dirs = dirs.reshape(-1, 3)
            radial = torch.full(
                (dirs.shape[0],),
                float(proj_monitor.proj_distance) * 1e-6,
                device=device,
                dtype=real_dtype,
            )
            th = torch.acos(dirs[:, 2]).reshape(u.shape)
            ph = torch.atan2(dirs[:, 1], dirs[:, 0]).reshape(u.shape)
            coordinates = {
                "ux": u,
                "uy": v,
                "r": radial[:1] * 1e6,
                "theta": th[None],
                "phi": ph[None],
                "f": surface.frequencies_hz,
                "freqs": surface.frequencies_hz,
            }
        else:
            axes = [a for a in range(3) if a != proj_monitor.proj_axis]
            x = proj_monitor.x.to(device=device, dtype=real_dtype) * 1e-6
            y = proj_monitor.y.to(device=device, dtype=real_dtype) * 1e-6
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            components = [None, None, None]
            components[axes[0]], components[axes[1]] = xx, yy
            coordinates_xyz = [c for c in components]
            coordinates_xyz[proj_monitor.proj_axis] = torch.full_like(
                xx, float(proj_monitor.proj_distance) * 1e-6
            )
            points = torch.stack(coordinates_xyz, -1).reshape(-1, 3) + local_origin
            rel = points - local_origin
            radial = torch.linalg.vector_norm(rel, dim=-1)
            dirs = rel / radial[:, None]
            th = torch.acos(dirs[:, 2]).reshape(xx.shape)
            ph = torch.atan2(dirs[:, 1], dirs[:, 0]).reshape(xx.shape)
            shaped = [None, None, None]
            shaped[axes[0]], shaped[axes[1]] = xx, yy
            shaped[proj_monitor.proj_axis] = torch.full_like(
                xx, float(proj_monitor.proj_distance) * 1e-6
            )
            # Axis coordinates are represented as broadcastable grids.
            coordinates = {
                "x": shaped[0].unsqueeze(proj_monitor.proj_axis) * 1e6,
                "y": shaped[1].unsqueeze(proj_monitor.proj_axis) * 1e6,
                "z": shaped[2].unsqueeze(proj_monitor.proj_axis) * 1e6,
                "theta": th.unsqueeze(proj_monitor.proj_axis),
                "phi": ph.unsqueeze(proj_monitor.proj_axis),
                "r": radial.reshape(xx.shape).unsqueeze(proj_monitor.proj_axis) * 1e6,
                "f": surface.frequencies_hz,
                "freqs": surface.frequencies_hz,
            }
        if proj_monitor.far_field_approx:
            if self.backend == "direct_triton":
                raise ValueError(
                    "direct_triton is an exact finite-distance backend; "
                    "use backend='auto' or 'direct_torch' for far_field_approx=True"
                )
            phase_surface = Surface(
                surface.positions_m - origin[None],
                surface.normals,
                surface.weights_m2,
                surface.electric_fields,
                surface.magnetic_fields,
                surface.frequencies_hz,
            )
            ee, hh = project_farfield(phase_surface, dirs, self.eps_r, self.mu_r)
            eps = torch.as_tensor(self.eps_r, device=device, dtype=real_dtype)
            mu = torch.as_tensor(self.mu_r, device=device, dtype=real_dtype)
            k = 2 * torch.pi * surface.frequencies_hz * torch.sqrt(eps * mu) / C_0
            phase = torch.exp(1j * k[:, None] * radial[None, :]) / radial[None, :]
            ee, hh = ee * phase[..., None], hh * phase[..., None]
        else:
            points = local_origin[None] + radial[:, None] * dirs
            from .dispatch import project

            ee, hh = project(
                surface, points, self.backend, eps_r=self.eps_r, mu_r=self.mu_r
            )
        es = (
            spherical_components(ee, dirs)
            .reshape((ee.shape[0], *th.shape, 3))
            .movedim(0, -1)
        )
        hs = (
            spherical_components(hh, dirs)
            .reshape((hh.shape[0], *th.shape, 3))
            .movedim(0, -1)
        )
        if isinstance(proj_monitor, FieldProjectionAngleMonitor):
            es, hs = es.unsqueeze(0), hs.unsqueeze(0)
        elif isinstance(proj_monitor, FieldProjectionKSpaceMonitor):
            es, hs = es.unsqueeze(2), hs.unsqueeze(2)
        else:
            axis = proj_monitor.proj_axis
            es, hs = es.unsqueeze(axis), hs.unsqueeze(axis)
        return FieldProjectionData(
            proj_monitor,
            self.surfaces,
            coordinates,
            *es.unbind(-2),
            *hs.unbind(-2),
        )
