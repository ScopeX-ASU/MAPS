"""Differentiable physical-coordinate transforms, independent of sampling."""

from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch


def translate(values, coords, offsets):
    """Translate a full-grid scalar field using rectilinear interpolation.

    Args:
        values: Real or complex tensor in MAPS (x, y[, z]) order.
        coords: Increasing cell-center vectors, in micrometers.
        offsets: Translation per axis in micrometers; positive moves features
            toward positive coordinates. Outside-domain samples are zero.

    Returns:
        Tensor with the same shape, dtype, device, and design gradient path.

    Notes:
        Supports nonuniform centers. Zero offsets return the input exactly.
        Full-grid inputs allow features to leave their original region box.
    """
    if len(coords) != values.ndim or len(offsets) != values.ndim:
        raise ValueError("Coordinate/offset rank must match the tensor")
    result = values
    for axis, (coordinates, offset) in enumerate(zip(coords, offsets)):
        c = torch.as_tensor(
            np.asarray(coordinates).copy(),
            dtype=values.real.dtype,
            device=values.device,
        )
        if (
            c.ndim != 1
            or c.numel() != values.shape[axis]
            or c.numel() < 2
            or not bool(torch.all(c[1:] > c[:-1]))
        ):
            raise ValueError(
                "Each coordinate axis must be increasing with at least two centers"
            )
        if not np.isfinite(offset):
            raise ValueError("Offsets must be finite")
        if offset == 0:
            continue
        # Extend by one zero-valued ghost center on each side, preserving
        # fractional-grid interpolation at the edge of the simulation domain.
        c_ext = torch.cat((2 * c[:1] - c[1:2], c, 2 * c[-1:] - c[-2:-1]))
        shape = list(result.shape)
        shape[axis] = 1
        zero = result.new_zeros(shape)
        padded = torch.cat((zero, result, zero), dim=axis)
        query = c - float(offset)
        upper = torch.searchsorted(c_ext, query).clamp(1, c_ext.numel() - 1)
        lower = upper - 1
        fraction = (query - c_ext[lower]) / (c_ext[upper] - c_ext[lower])
        shape = [1] * values.ndim
        shape[axis] = c.numel()
        fraction = fraction.reshape(shape)
        result = (
            padded.index_select(axis, lower) * (1 - fraction)
            + padded.index_select(axis, upper) * fraction
        )
        inside = ((query >= c_ext[0]) & (query <= c_ext[-1])).reshape(shape)
        result = torch.where(inside, result, 0)
    return result


def interval_coverage(boundaries, lower, upper, reference):
    """Compute exact fractional cell coverage of a physical interval.

    Args:
        boundaries: Increasing cell edges in micrometers, length N + 1.
        lower: Lower interface coordinate in micrometers.
        upper: Upper interface coordinate in micrometers, greater than lower.
        reference: Tensor supplying output dtype and device.

    Returns:
        Real tensor of N cell-volume fractions in [0, 1].
    """
    b = torch.as_tensor(
        np.asarray(boundaries).copy(),
        dtype=reference.real.dtype,
        device=reference.device,
    )
    if (
        b.ndim != 1
        or len(b) < 2
        or not bool(torch.all(b[1:] > b[:-1]))
        or not np.isfinite([lower, upper]).all()
        or upper <= lower
    ):
        raise ValueError("Invalid cell boundaries or physical interval")
    return (
        torch.minimum(b[1:], b.new_tensor(upper))
        - torch.maximum(b[:-1], b.new_tensor(lower))
    ).clamp_min(0) / (b[1:] - b[:-1])


class MapsGeometryTransform:
    """Materialize box/extruded MAPS patterns on the existing optical grid.

    Args:
        model: BaseOptimization instance, with initialized devices/parameters.
        cache_size: Maximum number of static material backgrounds to retain.

    Notes:
        Supported names are ``<region>.dx``, ``.dy``, ``.etch_depth``,
        ``.thickness`` (grating stacks), and ``.eps`` (relative-permittivity
        deviation). Grating thickness holds the stack bottom fixed and shifts
        upper layers; first-layer thickness also changes the access waveguide.
        Etch errors are positive for deeper etches, with the layer top fixed.
        Separately named boundary boxes (``<region>_box``) remain unchanged;
        no process coupling is inferred from geometry names.
        Static geometry is rasterized only on cache misses, using the existing
        rasterizer; simulation grid shapes are preserved. Sparse recorder
        envelopes are configured separately during robust evaluation.
    """

    def __init__(self, model, cache_size=4):
        """Retain the nominal model and initialize an empty background cache.

        Args:
            model: Initialized MAPS optimization model.
            cache_size: Positive maximum number of cached static backgrounds.

        Raises:
            ValueError: If cache_size is not positive.
        """
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.model = model
        self.cache_size = cache_size
        self._backgrounds = OrderedDict()

    def _configuration(self, scenario):
        """Resolve physical deviations into independent geometry dictionaries.

        Args:
            scenario: Named scalar deviations in micrometers or permittivity.

        Returns:
            Static geometry, design regions, and normalized deviation mapping.
        """
        device = self.model.device
        static = deepcopy(device._combined_geometry_cfgs())
        regions = deepcopy(device.design_region_cfgs)
        deviations = dict(scenario)
        if "silicon_thickness_error" in deviations:
            if "grating_layer_1.thickness" in deviations:
                raise ValueError("Specify only one first-layer thickness deviation")
            deviations["grating_layer_1.thickness"] = deviations.pop(
                "silicon_thickness_error"
            )
        for key, value in deviations.items():
            name, separator, parameter = key.rpartition(".")
            if (
                not separator
                or name not in regions
                or parameter not in {"dx", "dy", "etch_depth", "thickness", "eps"}
            ):
                raise ValueError(f"Unsupported geometry variation: {key}")
            if not np.isfinite(value):
                raise ValueError("Variation values must be finite")
        cumulative = 0.0
        for name, cfg in regions.items():
            if cfg["type"] != "box" or len(cfg["size"]) != 3:
                raise NotImplementedError(
                    "Built-in adapter requires 3D box design regions"
                )
            dt = deviations.get(f"{name}.thickness", 0.0)
            dd = deviations.get(f"{name}.etch_depth", 0.0)
            cfg["center"] = list(cfg["center"])
            cfg["size"] = list(cfg["size"])
            if dt or dd or cumulative:
                if not name.startswith("grating_layer_"):
                    raise NotImplementedError(
                        "Etch/thickness adapter currently supports grating stacks"
                    )
                index = name.removeprefix("grating_layer_")
                slab_name = f"fixed_slab_layer_{index}"
                top = cfg["center"][2] + cfg["size"][2] / 2
                old_depth = cfg["size"][2]
                slab = static.get(slab_name)
                bottom = (
                    top - old_depth
                    if slab is None
                    else slab["center"][2] - slab["size"][2] / 2
                )
                thickness = top - bottom + dt
                depth = old_depth + dd
                if thickness <= 0 or depth <= 0 or depth > thickness + 1e-12:
                    raise ValueError(
                        "Etch depth must be positive and cannot exceed layer thickness"
                    )
                top += cumulative + dt
                bottom += cumulative
                cfg["size"][2] = depth
                cfg["center"][2] = top - depth / 2
                remaining = thickness - depth
                if remaining > 1e-12:
                    size = [
                        cfg["size"][0],
                        cfg["size"][1],
                        remaining + device.grid_step,
                    ]
                    center = [cfg["center"][0], cfg["center"][1], bottom + size[2] / 2]
                    static[slab_name] = dict(
                        type="box", center=center, size=size, eps=cfg["eps"]
                    )
                else:
                    static.pop(slab_name, None)
                if index == "1" and dt:
                    for port_name, port in static.items():
                        if port_name.startswith("out_port_"):
                            port["center"] = list(port["center"])
                            port["size"] = list(port["size"])
                            port["center"][2] += dt / 2
                            port["size"][2] += dt
                cumulative += dt
            cfg["eps"] += deviations.get(f"{name}.eps", 0.0)
            if cfg["eps"] <= 0:
                raise ValueError("Relative permittivity must remain positive")
        return static, regions, deviations

    def recording_bounds(self, support):
        """Bound all design-dependent material cells for sparse recording.

        Args:
            support: Mapping of geometry deviation names to finite min/max
                pairs, in micrometers (or relative permittivity for eps).

        Returns:
            Mapping of region names to three native half-open cell bounds.

        Raises:
            ValueError: If geometry support is unbounded or crosses the domain.

        Notes:
            Stack thickness moves all upper layers. Lateral interpolation and
            native-grid resampling receive a conservative two-cell-width halo.
            Bounds include the nominal support and remain fixed across samples.
        """
        device = self.model.device
        self._configuration({key: 0.0 for key in support})
        support = dict(support)
        if "silicon_thickness_error" in support:
            support["grating_layer_1.thickness"] = support.pop(
                "silicon_thickness_error"
            )
        if any(
            not np.isfinite(pair).all() or pair[0] > pair[1]
            for pair in support.values()
        ):
            raise ValueError(
                "Sparse geometry recording requires finite distribution support; specify min/max truncation"
            )
        native = device.fdtdx_native_grid_metadata
        raster = device._get_raster_grid_info("epsilon_map")
        cumulative = np.zeros(2)
        result = {}
        for name, cfg in device.design_region_cfgs.items():
            dt = np.asarray(support.get(f"{name}.thickness", (0.0, 0.0)))
            cumulative += dt
            etch = support.get(f"{name}.etch_depth", (0.0, 0.0))
            bounds = []
            for axis, edges in enumerate(native["boundaries"]):
                edges = np.asarray(edges)
                lower = cfg["center"][axis] - cfg["size"][axis] / 2
                upper = cfg["center"][axis] + cfg["size"][axis] / 2
                if axis < 2:
                    delta = support.get(
                        f"{name}.{'dx' if axis == 0 else 'dy'}", (0.0, 0.0)
                    )
                    lower += min(0.0, delta[0])
                    upper += max(0.0, delta[1])
                else:
                    lower += min(0.0, cumulative[0] - etch[1])
                    upper += max(0.0, cumulative[1])
                if lower < edges[0] - 1e-12 or upper > edges[-1] + 1e-12:
                    raise ValueError(
                        "Variation support exceeds the simulation domain; tighten min/max or enlarge the domain"
                    )
                # A fractional source cell can extend beyond the physical box;
                # translating its interpolant adds one further source interval.
                halo = 2 * max(
                    np.max(np.diff(raster["boundaries"][axis])), np.max(np.diff(edges))
                )
                start = max(
                    0, int(np.searchsorted(edges, lower - halo, side="right") - 1)
                )
                stop = min(
                    len(edges) - 1,
                    int(np.searchsorted(edges, upper + halo, side="left")),
                )
                bounds.append((start, stop))
            result[name] = tuple(bounds)
        return result

    def _background(self, static, reference):
        """Rasterize static geometry on the existing grid with bounded caching.

        Args:
            static: Physical geometry configuration without learnable regions.
            reference: Tensor supplying dtype/device and expected grid shape.

        Returns:
            Independent static permittivity tensor.
        """
        key = repr(static)
        if key not in self._backgrounds:
            device = self.model.device
            use_tidy3d = any(
                type(g).__module__.startswith("tidy3d")
                for g in device.geometry.values()
            )
            geometry = {
                name: device._build_geometry_from_cfg(
                    cfg, value_key="eps", use_tidy3d=use_tidy3d
                )
                for name, cfg in static.items()
            }
            array, _ = device.rasterize_geometry_map(
                cell_size=device.cell_size,
                geometry=geometry,
                PML=device.sim_cfg["PML"],
                resolution=device.resolution,
                background_value=device.eps_bg,
                use_tidy3d=use_tidy3d,
                override_structures={},
                grid_info=device._get_raster_grid_info("epsilon_map"),
                record_grid=False,
                subpixel=device.sim_cfg.get("subpixel", True),
            )
            array = np.asarray(array)
            if np.iscomplexobj(array) and np.any(array.imag != 0):
                raise NotImplementedError(
                    "Built-in adapter requires lossless materials"
                )
            value = torch.as_tensor(
                array.real.copy(), dtype=reference.dtype, device=reference.device
            )
            if value.shape != reference.shape:
                raise ValueError("Static rasterizer changed the optical grid shape")
            self._backgrounds[key] = value
            if len(self._backgrounds) > self.cache_size:
                self._backgrounds.popitem(last=False)
        self._backgrounds.move_to_end(key)
        return self._backgrounds[key].clone()

    def __call__(self, nominal, scenario):
        """Apply one scenario without mutating the nominal model or geometry.

        Args:
            nominal: Five-item tuple returned by BaseOptimization.build_device.
            scenario: Mapping of physical deviations.

        Returns:
            Perturbed full optical permittivity tensor.

        Raises:
            NotImplementedError: For nonextruded patterns or custom support maps.
        """
        if not any(scenario.values()):
            self._configuration(scenario)  # Reject unknown names even at zero.
            return nominal[0]
        model = self.model
        device = model.device
        static, regions, deviations = self._configuration(scenario)
        symmetry = device.sim_cfg.get("symmetry", (0, 0, 0))
        for name in regions:
            for axis, parameter in enumerate(("dx", "dy")):
                if symmetry[axis] and deviations.get(f"{name}.{parameter}", 0):
                    raise ValueError(
                        "Lateral offsets along a symmetry axis require disabling symmetry reduction"
                    )
        eps = self._background(static, nominal[0])
        info = device._get_raster_grid_info("epsilon_map")
        coords, boundaries = info["coords"], info["boundaries"]
        for name, cfg in regions.items():
            maps = nominal[4][name]
            mask = device.design_region_masks[name]
            support_mask = maps.get("_region_mask")
            if (
                support_mask is not None and tuple(support_mask) != tuple(mask)
            ) or maps.get("_region_weight") is not None:
                raise NotImplementedError(
                    "Custom sidewall/support maps require a custom materializer"
                )
            values = maps["permittivity"]
            if values.ndim != 3 or tuple(values.shape) != tuple(eps[mask].shape):
                raise NotImplementedError(
                    "Adapter requires native-grid extruded region tensors"
                )
            old_cfg = device.design_region_cfgs[name]
            contrast = old_cfg["eps"] - old_cfg["eps_bg"]
            if contrast == 0:
                raise ValueError("Pattern materials must have distinct permittivities")
            parameters = getattr(model, "design_region_param_dict", {})
            parameterization = parameters[name] if name in parameters else None
            if parameterization is not None:
                if parameterization.cfgs.get("extrude_angle", 0) != 0 or sorted(
                    parameterization.cfgs.get("dims", (0, 1))
                ) != [0, 1]:
                    raise NotImplementedError(
                        "Built-in adapter requires vertical xy-pattern extrusions"
                    )
                if (
                    parameterization.cfgs.get("denorm_mode", "linear_eps")
                    != "linear_eps"
                ):
                    raise NotImplementedError(
                        "Built-in adapter requires linear_eps material blending"
                    )
            axis_weights = getattr(device, "design_region_axis_weights", {}).get(name)
            if axis_weights is not None and hasattr(model, "epsilon_map"):
                # Native extrusion has already blended fractional z cells with
                # fixed geometry. Invert one well-covered slice to recover the
                # shared pattern, retaining its established gradient surrogate.
                wz = np.asarray(axis_weights[2])
                index = int(np.argmax(wz))
                weight_z = float(wz[index])
                if weight_z <= 0:
                    raise ValueError("Nominal extrusion has no covered cells")
                base = model.epsilon_map[mask][..., index].detach()
                plane = (values[..., index] - base * (1 - weight_z)) / weight_z
            else:
                if not torch.allclose(values, values[..., :1].expand_as(values)):
                    raise NotImplementedError(
                        "Nonuniform extrusion requires nominal coverage metadata"
                    )
                plane = values[..., 0]
            density = (plane - old_cfg["eps_bg"]) / contrast
            planar_shape = eps.shape[:2]
            planar_eps = eps.new_zeros(planar_shape)
            planar_weight = eps.new_zeros(planar_shape)
            wx = interval_coverage(
                boundaries[0],
                cfg["center"][0] - cfg["size"][0] / 2,
                cfg["center"][0] + cfg["size"][0] / 2,
                eps,
            )
            wy = interval_coverage(
                boundaries[1],
                cfg["center"][1] - cfg["size"][1] / 2,
                cfg["center"][1] + cfg["size"][1] / 2,
                eps,
            )
            weight = wx[mask.x, None] * wy[None, mask.y]
            planar_eps[mask.x, mask.y] = (
                cfg["eps_bg"] + density * (cfg["eps"] - cfg["eps_bg"])
            ) * weight
            planar_weight[mask.x, mask.y] = weight
            offsets = (
                deviations.get(f"{name}.dx", 0.0),
                deviations.get(f"{name}.dy", 0.0),
            )
            if any(
                cfg["center"][axis] - cfg["size"][axis] / 2 + offsets[axis]
                < boundaries[axis][0]
                or cfg["center"][axis] + cfg["size"][axis] / 2 + offsets[axis]
                > boundaries[axis][-1]
                for axis in range(2)
            ):
                raise ValueError("Translated region exceeds the simulation domain")
            planar_eps = translate(planar_eps, coords[:2], offsets)
            planar_weight = translate(planar_weight, coords[:2], offsets)
            z_weight = interval_coverage(
                boundaries[2],
                cfg["center"][2] - cfg["size"][2] / 2,
                cfg["center"][2] + cfg["size"][2] / 2,
                eps,
            )
            if any(
                (
                    cfg["center"][i] - cfg["size"][i] / 2 < boundaries[i][0]
                    or cfg["center"][i] + cfg["size"][i] / 2 > boundaries[i][-1]
                )
                for i in range(3)
            ):
                raise ValueError("Perturbed region exceeds simulation domain")
            eps = (
                eps * (1 - planar_weight[..., None] * z_weight)
                + planar_eps[..., None] * z_weight
            )
        return eps
