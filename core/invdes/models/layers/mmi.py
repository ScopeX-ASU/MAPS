import copy
import warnings
from functools import partial
from typing import Any, Sequence, Tuple

import numpy as np
import tidy3d as td
import torch
from pyutils.general import logger

from core.utils import material_fn_dict

from .device_base import N_Ports

__all__ = ["MMI"]


def _expand_layer_values(
    value: Any,
    *,
    name: str,
    n_layers: int,
    default: Any,
) -> tuple[Any, ...]:
    """Normalize a scalar or one-element layer value to ``n_layers`` entries."""
    if value is None:
        value = default
    if isinstance(value, (str, bytes)) or np.isscalar(value):
        values = [value]
    else:
        values = list(value)
    if len(values) == 1 and n_layers > 1:
        values = values * n_layers
    if len(values) != n_layers:
        raise ValueError(f"{name} must have length 1 or {n_layers}, got {len(values)}.")
    return tuple(values)


def _layer_value_count(value: Any, *, box_size: bool = False) -> int:
    """Return the number of entries in a layer argument."""
    if value is None or isinstance(value, (str, bytes)) or np.isscalar(value):
        return 0 if value is None else 1
    values = list(value)
    if box_size and values and all(np.isscalar(item) for item in values):
        return 1
    return len(values)


def _normalize_layer_box_sizes(
    value: Sequence[Any] | None,
    *,
    name: str,
    n_layers: int,
    default: Sequence[float],
    is_3d: bool,
) -> tuple[tuple[float, ...], ...]:
    """Normalize per-layer footprint sizes while accepting one legacy size."""
    if value is None:
        values: list[Any] = [default]
    else:
        raw_values = list(value)
        if raw_values and all(np.isscalar(item) for item in raw_values):
            values = [raw_values]
        else:
            values = raw_values
    if len(values) == 1 and n_layers > 1:
        values = values * n_layers
    if len(values) != n_layers:
        raise ValueError(f"{name} must have length 1 or {n_layers}, got {len(values)}.")

    expected_lengths = (2, 3) if is_3d else (2,)
    normalized = []
    for index, layer_size in enumerate(values):
        layer_size = tuple(float(item) for item in layer_size)
        if len(layer_size) not in expected_lengths:
            expected = "2 or 3" if is_3d else "2"
            raise ValueError(
                f"{name}[{index}] must have {expected} entries, got {len(layer_size)}."
            )
        if any(size <= 0 for size in layer_size[:2]):
            raise ValueError(f"{name}[{index}] must have positive x/y sizes.")
        normalized.append(layer_size)
    return tuple(normalized)


class MMI(N_Ports):
    def __init__(
        self,
        material_r1: str = "Si_eff",  # waveguide material
        material_r2: str = "SiO2",  # waveguide material
        region_material_r1: str | None = None,  # region material 1
        region_material_r2: str | None = None,  # region material 2
        thickness_r1: float = 0.22,  # waveguide thickness
        thickness_r2: float = 0.0,  # waveguide thickness
        material_bg: str = "SiO2",  # background material
        sim_cfg: dict = {
            "border_width": [
                0,
                1.8,
                1.8,
                0,
            ],  # left, right, lower, upper, containing PML
            "PML": [0.5, 0.5],  # left/right, lower/upper
            "cell_size": None,
            "resolution": 50,
            "wl_cen": 1.55,
            "wl_width": 0,
            "n_wl": 1,
        },
        box_size: Tuple[float] = (1.5, 1.5),
        port_len: Tuple[float] = (1.8, 1.8),
        port_width: Tuple[float] = (0.48, 0.48),
        port_height: Tuple[float, float] = (0.22, 0.22),
        num_inports: int = 1,
        num_outports: int = 2,
        port_box_margin: float = 0.2,
        rel_port_width: float = 3,
        output_sine_bend_length: float = 0.0,  # if >0, this is the length of the sine_bend
        output_sine_bend_expand: float = 1.0,  # extra expansion from the starting points
        sidewall_angle: float = 0.0,  # signed sidewall angle in degrees; 0 is vertical
        LER_A: float = 0.0,  # line edge roughness amplitude
        LER_Lc: float = 0.0,  # line edge roughness
        is_3d: bool = False,
        pol: str = "Ez",
        device: torch.device = torch.device("cuda:0"),
        material_layers: Sequence[Any] | None = None,
        region_material_r1_layers: Sequence[Any] | None = None,
        region_material_r2_layers: Sequence[Any] | None = None,
        thickness_layers: Sequence[float] | None = None,
        etch_depth_layers: Sequence[float] | None = None,
        gap_layers: Sequence[float] | None = None,
        box_size_layers: Sequence[Sequence[float]] | None = None,
        layer_stack_z_offset: float = 0.0,
        material_ports: Sequence[Any] | None = None,
        material_buried: Any | None = None,
        material_cladding: Any | None = None,
        is_design_region: Sequence[bool] | bool | None = None,
    ):
        """Create a 2-D or 3-D multimode-interference device.

        The historical scalar arguments describe one design layer. Physical
        layer arguments such as ``thickness_layers`` or ``box_size_layers``
        build a vertical stack. A single r1/r2 layer-material entry can also be
        used without extrusion, preserving the legacy 2-D design footprint.
        Every physical layer consists of an optional fixed lower slab and an
        optional etched/design slab. Layer gaps are inserted before each layer,
        matching the convention used by :class:`GratingCoupler`.

        Args:
            port_height: Input and output port heights in micrometers. The
                output height also controls the output sine-bend slab extent.
            material_layers: High-index material for each stacked layer. A
                scalar or one-element sequence is broadcast.
            region_material_r1_layers: Upper/design-region material ``eps`` for
                each layer. Defaults to ``region_material_r1`` or
                ``material_layers``.
            region_material_r2_layers: Lower/design-region background ``eps_bg``
                for each layer. Defaults to ``region_material_r2`` or
                ``material_r2``.
            thickness_layers: Nonnegative physical thickness of each layer in
                micrometers. A zero-thickness entry preserves its layer index
                and gap but emits no geometry or design region.
            etch_depth_layers: Active/design thickness removed from the top of
                each layer. Must satisfy ``0 <= etch_depth <= thickness``.
            gap_layers: Vertical spacing before each layer in micrometers. A
                one-element sequence is broadcast.
            box_size_layers: Per-layer design-region footprint. Entries are
                ``(x, y)`` in 2-D or ``(x, y[, z])`` in 3-D; the z entry is
                accepted for convenience and the layer thickness controls the
                actual z extent.
            layer_stack_z_offset: Translation of the complete layer stack along
                z in micrometers. Ports remain centered at z = 0. This is
                independent of ``gap_layers``, which only specifies physical
                separation between adjacent layers.
            material_ports: Input and output port materials as ``(input,
                output)``. A scalar or one-element sequence is broadcast.
                Defaults to the legacy ``material_r1`` for both ports.
            material_buried: Material below the lowest port bottom plane in
                3-D. If omitted, no separate buried region is created.
            material_cladding: Material above the buried region and outside all
                explicit geometries. Defaults to the legacy ``material_bg``.
            is_design_region: Per-layer flags selecting whether each layer's
                etched/top portion is a design region. ``False`` makes that
                portion fixed geometry. A scalar or one-element sequence is
                broadcast. If omitted, the legacy behavior treats every
                positive etch depth as a design region.

        Notes:
            If all new layer arguments are omitted, the original single-layer
            construction is used, including the ``mmi_region`` name, centered
            footprint, and full-thickness design region.
        """
        physical_layer_args = (
            material_layers,
            thickness_layers,
            etch_depth_layers,
            gap_layers,
            box_size_layers,
        )
        role_layer_args = (
            region_material_r1_layers,
            region_material_r2_layers,
            is_design_region,
        )
        layer_args = physical_layer_args + role_layer_args
        layered = any(value is not None for value in physical_layer_args)
        role_layer_interface = any(value is not None for value in role_layer_args)
        if material_buried is not None and not is_3d:
            raise ValueError("material_buried is only supported in 3-D.")
        layer_stack_z_offset = float(layer_stack_z_offset)
        if not np.isfinite(layer_stack_z_offset):
            raise ValueError("layer_stack_z_offset must be finite.")
        if layer_stack_z_offset != 0.0 and not is_3d:
            raise ValueError("layer_stack_z_offset is only supported in 3-D.")
        if layer_stack_z_offset != 0.0 and not layered:
            raise ValueError(
                "layer_stack_z_offset requires the stacked-layer interface."
            )
        n_layers = max(
            [
                _layer_value_count(value, box_size=name == "box_size_layers")
                for name, value in zip(
                    (
                        "material_layers",
                        "thickness_layers",
                        "etch_depth_layers",
                        "gap_layers",
                        "box_size_layers",
                        "region_material_r1_layers",
                        "region_material_r2_layers",
                        "is_design_region",
                    ),
                    layer_args,
                )
            ]
            + [1]
        )
        if not layered and n_layers != 1:
            raise ValueError(
                "Multiple layer material/design entries require physical layer "
                "arguments such as thickness_layers and box_size_layers."
            )

        if is_3d:
            if len(box_size) != 3:
                raise ValueError("box_size must be a tuple of length 3 in 3-D.")
            if not -90.0 <= sidewall_angle <= 90.0:
                raise ValueError("sidewall_angle must be in the interval [-90, 90].")
        elif len(box_size) != 2:
            raise ValueError("box_size must be a tuple of length 2 in 2-D.")

        port_height = tuple(float(value) for value in port_height)
        if len(port_height) != 2:
            raise ValueError(
                f"port_height must contain input/output heights, got {port_height}."
            )
        if is_3d and any(value <= 0 for value in port_height):
            raise ValueError(f"3-D port heights must be positive, got {port_height}.")

        port_materials = _expand_layer_values(
            material_ports,
            name="material_ports",
            n_layers=2,
            default=material_r1,
        )
        cladding_material = (
            material_bg if material_cladding is None else material_cladding
        )

        layer_box_sizes = _normalize_layer_box_sizes(
            box_size_layers,
            name="box_size_layers",
            n_layers=n_layers,
            default=box_size,
            is_3d=is_3d,
        )
        if material_layers is None:
            # Preserve array-like legacy material values as one layer rather
            # than interpreting their components as layer entries.
            layer_materials = tuple(material_r1 for _ in range(n_layers))
        else:
            layer_materials = _expand_layer_values(
                material_layers,
                name="material_layers",
                n_layers=n_layers,
                default=material_r1,
            )
        if region_material_r1_layers is None:
            if region_material_r1 is None:
                layer_region_r1 = layer_materials
            else:
                layer_region_r1 = tuple(region_material_r1 for _ in range(n_layers))
        else:
            layer_region_r1 = _expand_layer_values(
                region_material_r1_layers,
                name="region_material_r1_layers",
                n_layers=n_layers,
                default=region_material_r1,
            )
        if material_layers is None and is_design_region is not None:
            # In the role-based interface, eps/r1 is also the material used
            # when a layer (or its lower slab) is emitted as fixed geometry.
            layer_materials = layer_region_r1
        if region_material_r2_layers is None:
            layer_region_r2 = tuple(
                material_r2 if region_material_r2 is None else region_material_r2
                for _ in range(n_layers)
            )
        else:
            layer_region_r2 = _expand_layer_values(
                region_material_r2_layers,
                name="region_material_r2_layers",
                n_layers=n_layers,
                default=material_r2,
            )
        layer_thicknesses = tuple(
            float(value)
            for value in _expand_layer_values(
                thickness_layers,
                name="thickness_layers",
                n_layers=n_layers,
                default=thickness_r1,
            )
        )
        layer_etch_depths = tuple(
            float(value)
            for value in _expand_layer_values(
                etch_depth_layers,
                name="etch_depth_layers",
                n_layers=n_layers,
                default=layer_thicknesses,
            )
        )
        layer_gaps = tuple(
            float(value)
            for value in _expand_layer_values(
                gap_layers,
                name="gap_layers",
                n_layers=n_layers,
                default=0.0,
            )
        )
        if is_design_region is None:
            layer_is_design = tuple(depth > 0 for depth in layer_etch_depths)
        else:
            layer_is_design = tuple(
                bool(value)
                for value in _expand_layer_values(
                    is_design_region,
                    name="is_design_region",
                    n_layers=n_layers,
                    default=False,
                )
            )
        for index, (thickness, etch_depth, gap) in enumerate(
            zip(layer_thicknesses, layer_etch_depths, layer_gaps)
        ):
            if thickness < 0:
                raise ValueError(f"thickness_layers[{index}] must be nonnegative.")
            if not 0 <= etch_depth <= thickness:
                raise ValueError(
                    f"etch_depth_layers[{index}]={etch_depth} must be between 0 and "
                    f"thickness_layers[{index}]={thickness}."
                )
            if gap < 0:
                raise ValueError(f"gap_layers[{index}] must be nonnegative.")
            if thickness > 0 and layer_is_design[index] and etch_depth == 0:
                raise ValueError(
                    f"is_design_region[{index}] is True, but "
                    f"etch_depth_layers[{index}] is zero."
                )

        self.num_inports = num_inports
        self.num_outports = num_outports
        wl_cen = sim_cfg["wl_cen"]
        self.rel_width = rel_port_width
        self.pol = pol
        self.is_3d = is_3d
        self.port_height = port_height
        self.box_size = (
            box_size
            if not layered
            else (
                layer_box_sizes[0]
                if not is_3d
                else (*layer_box_sizes[0][:2], layer_thicknesses[0])
            )
        )
        self.layer_box_sizes = layer_box_sizes
        self.layer_thicknesses = layer_thicknesses
        self.layer_etch_depths = layer_etch_depths
        self.layer_gaps = layer_gaps
        self.layer_is_design = layer_is_design
        self.is_design_region = layer_is_design
        self.layer_stack_z_offset = layer_stack_z_offset
        self.material_ports = port_materials
        self.material_buried = material_buried
        self.material_cladding = cladding_material

        def _get_eps_fn(material, thickness=None):
            if isinstance(material, str):
                eps_fn = material_fn_dict[material]
                if "_eff" in material:
                    eps_fn = partial(eps_fn, thickness=thickness)
            else:
                eps_fn = lambda wl: material
            return eps_fn

        eps_r1_fn = _get_eps_fn(material_r1, thickness=thickness_r1)

        eps_r2_fn = _get_eps_fn(material_r2, thickness=thickness_r2)

        eps_bg_fn = _get_eps_fn(cladding_material, thickness=thickness_r2)

        if region_material_r1 is None:
            eps_region_r1 = eps_r1_fn
        else:
            eps_region_r1 = _get_eps_fn(region_material_r1, thickness=thickness_r1)

        if region_material_r2 is None:
            eps_region_r2 = eps_r2_fn
        else:
            eps_region_r2 = _get_eps_fn(region_material_r2, thickness=thickness_r2)

        layer_material_fns = tuple(
            _get_eps_fn(material, thickness=thickness)
            for material, thickness in zip(layer_materials, layer_thicknesses)
        )
        layer_region_r1_fns = tuple(
            _get_eps_fn(material, thickness=thickness)
            for material, thickness in zip(layer_region_r1, layer_thicknesses)
        )
        layer_region_r2_fns = tuple(
            _get_eps_fn(material, thickness=thickness)
            for material, thickness in zip(layer_region_r2, layer_thicknesses)
        )
        port_material_fns = tuple(
            _get_eps_fn(material, thickness=height)
            for material, height in zip(port_materials, port_height)
        )
        buried_material_fn = (
            None
            if material_buried is None
            else _get_eps_fn(material_buried, thickness=thickness_r2)
        )

        port_cfgs = dict()

        grid_step = 1 / sim_cfg["resolution"]

        base_box_size = layer_box_sizes[0]
        base_thickness = layer_thicknesses[0]
        port_z_center = 0.0
        if layered:
            # The first/core layer is centered at the legacy vertical origin
            # before applying the independent stack translation. Added layers
            # (for example, a PCM film) are placed above it.
            current_bottom = -base_thickness / 2 + layer_stack_z_offset
            layer_bounds = []
            for thickness, gap in zip(layer_thicknesses, layer_gaps):
                layer_bottom = current_bottom + gap
                layer_top = layer_bottom + thickness
                layer_bounds.append(
                    (layer_bottom, layer_top, 0.5 * (layer_bottom + layer_top))
                )
                current_bottom = layer_top
        else:
            layer_bounds = [(-base_thickness / 2, base_thickness / 2, 0.0)]
        self.layer_bounds = tuple(layer_bounds)

        assert (
            base_box_size[1] - 2 * port_box_margin >= num_inports * port_width[0]
        ), "The base layer footprint is too narrow for the input ports."
        assert (
            base_box_size[1] - 2 * port_box_margin >= num_outports * port_width[1]
        ), "The base layer footprint is too narrow for the output ports."

        if num_inports == 1:
            inport_y_coords = np.array([0.0])
        else:
            inport_y_coords = np.linspace(
                -base_box_size[1] / 2 + port_width[0] / 2 + port_box_margin,
                base_box_size[1] / 2 - port_width[0] / 2 - port_box_margin,
                num_inports,
            )

        for i in range(1, num_inports + 1):
            port_cfgs[f"in_port_{i}"] = dict(
                type="box",
                direction="x",
                center=[
                    -(port_len[0] + base_box_size[0]) / 2 + grid_step / 2,
                    inport_y_coords[i - 1],
                ]
                + ([port_z_center] if is_3d else []),
                size=[port_len[0] + grid_step, port_width[0]]
                + ([port_height[0]] if is_3d else []),
                sidewall_angle=sidewall_angle,
                eps=port_material_fns[0](wl_cen),
                roughness=dict(amplitude=LER_A, correlation_length=LER_Lc),
            )
        geometry_cfgs = {}
        if output_sine_bend_length > 0:
            outport_y_start_coords = np.linspace(
                -base_box_size[1] / 2 + port_width[1] / 2 + port_box_margin,
                base_box_size[1] / 2 - port_width[1] / 2 - port_box_margin,
                num_outports,
            )
            outport_y_end_coords = np.linspace(
                -base_box_size[1] / 2
                + port_width[1] / 2
                + port_box_margin
                - output_sine_bend_expand,
                base_box_size[1] / 2
                - port_width[1] / 2
                - port_box_margin
                + output_sine_bend_expand,
                num_outports,
            )

            for i in range(1, num_outports + 1):
                geometry_cfgs[f"out_sine_bend_{i}"] = dict(
                    type="sine_bend",
                    direction="x",
                    start=[
                        base_box_size[0] / 2 - grid_step / 2,
                        outport_y_start_coords[i - 1],
                    ],
                    end=[
                        base_box_size[0] / 2 - grid_step / 2 + output_sine_bend_length,
                        outport_y_end_coords[i - 1],
                    ],
                    axis=2,
                    slab_bounds=(
                        (
                            port_z_center - port_height[1] / 2,
                            port_z_center + port_height[1] / 2,
                        )
                        if is_3d
                        else (-td.inf, td.inf)
                    ),
                    width=port_width[1],
                    sidewall_angle=sidewall_angle,
                    num_samples=100,
                    eps=port_material_fns[1](wl_cen),
                    roughness=dict(amplitude=LER_A, correlation_length=LER_Lc),
                )
        else:
            outport_y_end_coords = np.linspace(
                -base_box_size[1] / 2 + port_width[1] / 2 + port_box_margin,
                base_box_size[1] / 2 - port_width[1] / 2 - port_box_margin,
                num_outports,
            )
        for i in range(1, num_outports + 1):
            port_cfgs[f"out_port_{i}"] = dict(
                type="box",
                direction="x",
                center=[
                    base_box_size[0] / 2
                    + output_sine_bend_length
                    + port_len[1] / 2
                    - grid_step / 2,
                    outport_y_end_coords[i - 1],
                ]
                + ([port_z_center] if is_3d else []),
                size=[port_len[1] + grid_step, port_width[1]]
                + ([port_height[1]] if is_3d else []),
                sidewall_angle=sidewall_angle,
                eps=port_material_fns[1](wl_cen),
                roughness=dict(amplitude=LER_A, correlation_length=LER_Lc),
            )

        design_region_cfgs = {}
        if not layered:
            region_eps_fn = (
                layer_region_r1_fns[0] if role_layer_interface else eps_region_r1
            )
            region_eps_bg_fn = (
                layer_region_r2_fns[0] if role_layer_interface else eps_region_r2
            )
            region_cfg = dict(
                type="box",
                center=[0, 0] + ([0] if is_3d else []),
                size=box_size,
                eps=region_eps_fn(wl_cen),
            )
            if role_layer_interface and not layer_is_design[0]:
                geometry_cfgs["fixed_mmi_region"] = region_cfg
            else:
                region_cfg["eps_bg"] = region_eps_bg_fn(wl_cen)
                design_region_cfgs["mmi_region"] = region_cfg
        else:
            layer_entries = zip(
                layer_bounds,
                layer_box_sizes,
                layer_etch_depths,
                layer_is_design,
            )
            for index, (
                bounds,
                layer_size,
                etch_depth,
                layer_is_design_region,
            ) in enumerate(layer_entries, start=1):
                layer_bottom, layer_top, _ = bounds
                layer_material_fn = layer_material_fns[index - 1]
                if layer_thicknesses[index - 1] - etch_depth > 0:
                    fixed_thickness = layer_thicknesses[index - 1] - etch_depth
                    fixed_center = layer_bottom + fixed_thickness / 2
                    fixed_center_xyz = [0.0, fixed_center]
                    fixed_size = [layer_size[0], fixed_thickness]
                    if is_3d:
                        fixed_center_xyz = [0.0, 0.0, fixed_center]
                        fixed_size = [layer_size[0], layer_size[1], fixed_thickness]
                    geometry_cfgs[f"fixed_slab_layer_{index}"] = dict(
                        type="box",
                        center=fixed_center_xyz,
                        size=fixed_size,
                        eps=layer_material_fn(wl_cen),
                    )
                if etch_depth > 0:
                    top_slab_center = layer_top - etch_depth / 2
                    top_slab_center_xyz = [0.0, top_slab_center]
                    top_slab_size = [layer_size[0], etch_depth]
                    if is_3d:
                        top_slab_center_xyz = [0.0, 0.0, top_slab_center]
                        top_slab_size = [layer_size[0], layer_size[1], etch_depth]
                    if layer_is_design_region:
                        region_name = (
                            "mmi_region"
                            if n_layers == 1
                            else f"mmi_region_layer_{index}"
                        )
                        design_region_cfgs[region_name] = dict(
                            type="box",
                            center=top_slab_center_xyz,
                            size=top_slab_size,
                            eps=layer_region_r1_fns[index - 1](wl_cen),
                            eps_bg=layer_region_r2_fns[index - 1](wl_cen),
                        )
                    else:
                        geometry_cfgs[f"fixed_top_layer_{index}"] = dict(
                            type="box",
                            center=top_slab_center_xyz,
                            size=top_slab_size,
                            eps=layer_material_fn(wl_cen),
                        )

        if buried_material_fn is not None:
            buried_top = port_z_center - max(port_height) / 2
            configured_cell_size = sim_cfg.get("cell_size")
            if configured_cell_size not in (None, "None"):
                buried_bottom = -float(configured_cell_size[2]) / 2
            else:
                border_width = list(sim_cfg.get("border_width") or [])
                if len(border_width) == 6:
                    lower_z_border = float(border_width[4])
                elif len(border_width) == 3:
                    lower_z_border = float(border_width[2])
                elif len(border_width) in (0, 4):
                    lower_z_border = 0.0
                else:
                    raise ValueError(
                        "3-D border_width must contain 3 or 6 entries when "
                        "material_buried is specified."
                    )
                natural_bottom = min(
                    buried_top,
                    *(bounds[0] for bounds in layer_bounds),
                )
                buried_bottom = natural_bottom - lower_z_border
            buried_thickness = buried_top - buried_bottom
            if buried_thickness <= 0:
                raise ValueError(
                    "The buried-material top plane must lie above the simulation "
                    "cell bottom."
                )
            geometry_cfgs = {
                "buried_region": dict(
                    type="box",
                    center=[0.0, 0.0, (buried_bottom + buried_top) / 2],
                    size=[None, None, buried_thickness],
                    eps=buried_material_fn(wl_cen),
                    _exclude_from_bounds=True,
                ),
                **geometry_cfgs,
            }

        super().__init__(
            eps_bg=eps_bg_fn(wl_cen),
            sim_cfg=sim_cfg,
            port_cfgs=port_cfgs,
            geometry_cfgs=geometry_cfgs,
            design_region_cfgs=design_region_cfgs,
            device=device,
        )

    def init_monitors(self, verbose: bool = True):
        rel_width = self.rel_width
        if verbose:
            logger.info("Start generating sources and monitors ...")
        pml = self.sim_cfg["PML"][0]
        port_len = self.port_cfgs["in_port_1"]["size"][0]
        offset = 0.2 + pml

        if self.is_3d:
            monitor_height = (
                self.box_size[2]
                + self.sim_cfg["border_width"][-2]
                + self.sim_cfg["border_width"][-1]
                - self.sim_cfg["PML"][2] * 2
            )
            in_rel_height = monitor_height / self.port_height[0]
            out_rel_height = monitor_height / self.port_height[1]
        else:
            in_rel_height = None
            out_rel_height = None

        src_slices = [
            self.build_port_monitor_slice(
                port_name=f"in_port_{i}",
                slice_name=f"in_slice_{i}",
                rel_loc=offset / port_len,
                rel_width=rel_width,
                rel_height=in_rel_height,
                direction="x+",
            )
            for i in range(1, self.num_inports + 1)
        ]
        refl_slices = [
            self.build_port_monitor_slice(
                port_name=f"in_port_{i}",
                slice_name=f"refl_slice_{i}",
                rel_loc=(offset + 0.1) / port_len,
                rel_width=rel_width,
                rel_height=in_rel_height,
                direction="x+",
            )
            for i in range(1, self.num_inports + 1)
        ]

        out_slices = []
        for i in range(1, self.num_outports + 1):
            center = self.port_cfgs[f"out_port_{i}"]["center"]
            if (
                self.is_3d
                and "symmetry" in self.sim_cfg
                and self.sim_cfg["symmetry"] is not None
                and self.sim_cfg["symmetry"][1] != 0
                and center[1] < 0
            ):
                ## this will not appear in the reduced simulation space with symmetry
                logger.warning(
                    f"out_port_{i} is not in the simulation space with symmetry, skip building the monitor slice"
                )
            else:
                out_slices.append(
                    self.build_port_monitor_slice(
                        port_name=f"out_port_{i}",
                        slice_name=f"out_slice_{i}",
                        rel_loc=1 - offset / port_len,
                        rel_width=rel_width,
                        rel_height=out_rel_height,
                        direction="x+",
                    )
                )

        # out_slices = [
        #     self.build_port_monitor_slice(
        #         port_name=f"out_port_{i}",
        #         slice_name=f"out_slice_{i}",
        #         rel_loc=1 - offset / port_len,
        #         rel_width=rel_width,
        #         rel_height=rel_height,
        #         direction="x+",
        #     )
        #     for i in range(1, self.num_outports + 1)
        # ]
        self.ports_regions = self.build_port_region(self.port_cfgs, rel_width=rel_width)

        if not self.is_3d:
            radiation_monitor = self.build_radiation_monitor(monitor_name="rad_slice")
        else:
            radiation_monitor = None

        return src_slices, out_slices, refl_slices, radiation_monitor

    def norm_run(self, verbose: bool = True):
        if verbose:
            logger.info("Start normalization run ...")
        # norm_run_sim_cfg = copy.deepcopy(self.sim_cfg)
        # norm_run_sim_cfg["numerical_solver"] = "solve_direct"
        norm_source_profiles = [
            self.build_norm_sources(
                source_modes=(f"{self.pol}1",),
                input_port_name=f"in_port_{i}",
                input_slice_name=f"in_slice_{i}",
                wl_cen=self.sim_cfg["wl_cen"],
                wl_width=self.sim_cfg["wl_width"],
                n_wl=self.sim_cfg["n_wl"],
                solver=self.sim_cfg["solver"],
                plot=True,
                require_sim=True,
            )
            for i in range(1, self.num_inports + 1)
        ]

        norm_refl_profiles = [
            self.build_norm_sources(
                source_modes=(f"{self.pol}1",),
                input_port_name=f"in_port_{i}",
                input_slice_name=f"refl_slice_{i}",
                wl_cen=self.sim_cfg["wl_cen"],
                wl_width=self.sim_cfg["wl_width"],
                n_wl=self.sim_cfg["n_wl"],
                solver=self.sim_cfg["solver"],
                plot=True,
                require_sim=False,
            )
            for i in range(1, self.num_inports + 1)
        ]
        # norm_monitor_profiles = [
        #     self.build_norm_sources(
        #         source_modes=(f"{self.pol}1",),
        #         input_port_name=f"out_port_{i}",
        #         input_slice_name=f"out_slice_{i}",
        #         wl_cen=self.sim_cfg["wl_cen"],
        #         wl_width=self.sim_cfg["wl_width"],
        #         n_wl=self.sim_cfg["n_wl"],
        #         solver=self.sim_cfg["solver"],
        #         plot=True,
        #         require_sim=False,
        #     )
        #     for i in range(1, self.num_outports + 1)
        # ]

        norm_monitor_profiles = []
        for i in range(1, self.num_outports + 1):
            center = self.port_cfgs[f"out_port_{i}"]["center"]
            if (
                self.is_3d
                and "symmetry" in self.sim_cfg
                and self.sim_cfg["symmetry"] is not None
                and self.sim_cfg["symmetry"][1] != 0
                and center[1] < 0
            ):
                ## this will not appear in the reduced simulation space with symmetry
                logger.warning(
                    f"out_port_{i} is not in the simulation space with symmetry, skip norm run"
                )
            else:
                norm_monitor_profiles.append(
                    self.build_norm_sources(
                        source_modes=(f"{self.pol}1",),
                        input_port_name=f"out_port_{i}",
                        input_slice_name=f"out_slice_{i}",
                        wl_cen=self.sim_cfg["wl_cen"],
                        wl_width=self.sim_cfg["wl_width"],
                        n_wl=self.sim_cfg["n_wl"],
                        solver=self.sim_cfg["solver"],
                        plot=True,
                        require_sim=False,
                    )
                )

        return norm_source_profiles, norm_refl_profiles, norm_monitor_profiles
