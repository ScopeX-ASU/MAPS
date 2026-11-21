"""Compare dynamic grating etch/thickness maps with explicit physical boxes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.invdes.variation import MapsGeometryTransform
from core.utils import Slice3D

pytestmark = [pytest.mark.integration, pytest.mark.numerical]


def box_map(cfg, boundaries, reference):
    """Compute independent Cartesian box volume fractions from interval lengths."""
    weights = []
    for axis, edges in enumerate(boundaries):
        lo = cfg["center"][axis] - cfg["size"][axis] / 2
        hi = cfg["center"][axis] + cfg["size"][axis] / 2
        lengths = np.maximum(0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo))
        weights.append(torch.as_tensor(lengths / np.diff(edges), dtype=reference.dtype))
    return (
        weights[0][:, None, None]
        * weights[1][None, :, None]
        * weights[2][None, None, :]
    )


class BoxDevice:
    """Small physical box fixture with an independently calculated static map."""

    def __init__(self):
        self.grid_step = 0.1
        self.resolution = 10
        self.eps_bg = 2.0
        self.cell_size = (1.0, 1.0, 1.0)
        self.sim_cfg = {"PML": [0.0, 0.0, 0.0], "subpixel": True, "symmetry": (0, 0, 0)}
        self.geometry = {}
        self.boundaries = tuple(np.linspace(-0.5, 0.5, 11) for _ in range(3))
        self.coords = tuple((b[:-1] + b[1:]) / 2 for b in self.boundaries)
        self.design_region_cfgs = {
            "grating_layer_1": dict(
                type="box",
                center=[0.0, 0.0, -0.05],
                size=[0.4, 0.4, 0.2],
                eps=12.0,
                eps_bg=2.0,
            ),
            "grating_layer_2": dict(
                type="box",
                center=[0.0, 0.0, 0.3],
                size=[0.4, 0.4, 0.2],
                eps=4.0,
                eps_bg=2.0,
            ),
        }
        self.design_region_masks = {
            "grating_layer_1": Slice3D(slice(3, 7), slice(3, 7), slice(3, 6)),
            "grating_layer_2": Slice3D(slice(3, 7), slice(3, 7), slice(7, 9)),
        }
        self.static = {
            "fixed_slab_layer_1": dict(
                type="box", center=[0.0, 0.0, -0.15], size=[0.4, 0.4, 0.2], eps=12.0
            )
        }
        self.calls = 0

    def _combined_geometry_cfgs(self):
        return self.static

    def _get_raster_grid_info(self, name):
        return {"coords": self.coords, "boundaries": self.boundaries}

    def _build_geometry_from_cfg(self, cfg, **kwargs):
        return cfg

    def rasterize_geometry_map(self, **kwargs):
        self.calls += 1
        reference = torch.zeros((10, 10, 10), dtype=torch.float64)
        eps = torch.full_like(reference, self.eps_bg)
        for cfg in kwargs["geometry"].values():
            weight = box_map(cfg, self.boundaries, reference)
            eps = eps * (1 - weight) + cfg["eps"] * weight
        return eps.numpy(), None


def fixture():
    device = BoxDevice()
    model = SimpleNamespace(device=device)
    transform = MapsGeometryTransform(model)
    density = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    maps = {
        name: {
            "permittivity": torch.ones(
                (4, 4, mask.z.stop - mask.z.start), dtype=torch.float64
            )
            * (cfg["eps_bg"] + density * (cfg["eps"] - cfg["eps_bg"]))
        }
        for (name, cfg), mask in zip(
            device.design_region_cfgs.items(), device.design_region_masks.values()
        )
    }
    nominal = (torch.zeros((10, 10, 10), dtype=torch.float64), {}, None, {}, maps)
    return device, transform, nominal, density


@pytest.mark.parametrize(
    "scenario",
    [
        {"grating_layer_1.etch_depth": 0.05},
        {"grating_layer_1.etch_depth": -0.05},
        {"silicon_thickness_error": 0.05},
        {"grating_layer_1.dx": 0.1, "grating_layer_2.dy": -0.1},
    ],
)
def test_material_map_matches_explicit_reconstruction_and_preserves_design_gradient(
    scenario,
):
    device, transform, nominal, density = fixture()
    result = transform(nominal, scenario)
    # Independent physical references: hold bottom=-.25, top=.05 for etch;
    # thickness moves first top and the upper layer by +.05.
    depth = 0.2 + scenario.get("grating_layer_1.etch_depth", 0.0)
    dt = scenario.get("silicon_thickness_error", 0.0)
    top = 0.05 + dt
    remaining = top - depth - (-0.25)
    reference = torch.full_like(result, 2.0)
    slab = dict(
        center=[0.0, 0.0, -0.25 + (remaining + 0.1) / 2],
        size=[0.4, 0.4, remaining + 0.1],
    )
    w = box_map(slab, device.boundaries, result)
    reference = reference * (1 - w) + 12 * w
    for i, (name, original) in enumerate(device.design_region_cfgs.items()):
        cfg = dict(
            original, center=list(original["center"]), size=list(original["size"])
        )
        if i == 0:
            cfg["center"][2] = top - depth / 2
            cfg["size"][2] = depth
        else:
            cfg["center"][2] += dt
        cfg["center"][0] += scenario.get(f"{name}.dx", 0.0)
        cfg["center"][1] += scenario.get(f"{name}.dy", 0.0)
        w = box_map(cfg, device.boundaries, result)
        material = cfg["eps_bg"] + density * (cfg["eps"] - cfg["eps_bg"])
        reference = reference * (1 - w) + material * w
    torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)
    (actual_grad,) = torch.autograd.grad(result.sum(), density, retain_graph=True)
    (reference_grad,) = torch.autograd.grad(reference.sum(), density)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-12, atol=1e-12)
    original = device.design_region_cfgs["grating_layer_1"]
    assert original["size"][2] == 0.2 and original["center"][2] == -0.05
    transform(nominal, scenario)
    assert device.calls == 1


def test_zero_unknown_impossible_depth_and_support_validation():
    device, transform, nominal, _ = fixture()
    assert transform(nominal, {}) is nominal[0]
    with pytest.raises(ValueError):
        transform(nominal, {"unknown.dx": 0.0})
    with pytest.raises(ValueError):
        transform(nominal, {"grating_layer_1.etch_depth": 1.0})


def test_overlapping_layers_preserve_ordered_material_composition():
    device, transform, nominal, density = fixture()
    device.design_region_cfgs["grating_layer_2"]["center"][2] = -0.05
    device.design_region_masks["grating_layer_2"] = device.design_region_masks[
        "grating_layer_1"
    ]
    nominal[4]["grating_layer_2"]["permittivity"] = nominal[4]["grating_layer_2"][
        "permittivity"
    ][..., :1].expand(4, 4, 3)
    scenario = {"grating_layer_1.dx": 0.1, "grating_layer_2.dy": -0.1}
    result = transform(nominal, scenario)
    reference = torch.full_like(result, device.eps_bg)
    slab = device.static["fixed_slab_layer_1"]
    w = box_map(slab, device.boundaries, result)
    reference = reference * (1 - w) + slab["eps"] * w
    for name, original in device.design_region_cfgs.items():
        cfg = dict(original, center=list(original["center"]))
        cfg["center"][0] += scenario.get(f"{name}.dx", 0.0)
        cfg["center"][1] += scenario.get(f"{name}.dy", 0.0)
        w = box_map(cfg, device.boundaries, result)
        eps = cfg["eps_bg"] + density * (cfg["eps"] - cfg["eps_bg"])
        reference = reference * (1 - w) + eps * w
    torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_fixed_recording_envelope_contains_perturbed_design_gradient(nonuniform):
    """Every material cell with a design derivative lies inside the envelope."""
    device, transform, _, density = fixture()
    edges = 2 * np.linspace(0, 1, 21) ** (1.2 if nonuniform else 1.0) - 1
    device.boundaries = tuple(edges.copy() for _ in range(3))
    device.coords = tuple((b[:-1] + b[1:]) / 2 for b in device.boundaries)
    device.cell_size = (2.0, 2.0, 2.0)
    maps = {}
    for name, cfg in device.design_region_cfgs.items():
        slices = []
        for axis, boundaries in enumerate(device.boundaries):
            lo = cfg["center"][axis] - cfg["size"][axis] / 2
            hi = cfg["center"][axis] + cfg["size"][axis] / 2
            slices.append(
                slice(
                    int(np.searchsorted(boundaries, lo, side="right") - 1),
                    int(np.searchsorted(boundaries, hi, side="left")),
                )
            )
        mask = Slice3D(*slices)
        device.design_region_masks[name] = mask
        shape = tuple(s.stop - s.start for s in slices)
        maps[name] = {
            "permittivity": torch.ones(shape, dtype=torch.float64)
            * (cfg["eps_bg"] + density * (cfg["eps"] - cfg["eps_bg"]))
        }
    reference = torch.zeros((20, 20, 20), dtype=torch.float64)
    nominal = (reference, {}, None, {}, maps)

    def rasterize(**kwargs):
        """Rasterize static boxes on the expanded fixture domain.

        Args:
            **kwargs: Rasterizer arguments including the geometry mapping.

        Returns:
            NumPy permittivity map and absent auxiliary metadata.
        """
        eps = torch.full_like(reference, device.eps_bg)
        for cfg in kwargs["geometry"].values():
            weight = box_map(cfg, device.boundaries, reference)
            eps = eps * (1 - weight) + cfg["eps"] * weight
        return eps.numpy(), None

    device.rasterize_geometry_map = rasterize
    device.fdtdx_native_grid_metadata = dict(
        coords=device.coords, boundaries=device.boundaries
    )
    support = {
        "grating_layer_1.dx": (-0.11, 0.13),
        "grating_layer_1.dy": (-0.12, 0.11),
        "grating_layer_2.dx": (-0.10, 0.12),
        "grating_layer_2.dy": (-0.11, 0.10),
        "grating_layer_1.etch_depth": (-0.03, 0.04),
        "grating_layer_1.thickness": (-0.02, 0.03),
        "grating_layer_2.etch_depth": (-0.02, 0.0),
    }
    bounds = transform.recording_bounds(support)
    coverage = torch.zeros_like(nominal[0])
    for region in bounds.values():
        coverage[tuple(slice(lo, hi) for lo, hi in region)] = 1
    assert coverage.sum() < 0.75 * coverage.numel()
    generator = np.random.default_rng(73)
    for fraction in [0.0, 1.0, *generator.random(8)]:
        scenario = {
            name: lo + fraction * (hi - lo) for name, (lo, hi) in support.items()
        }
        material = transform(nominal, scenario)
        missing_derivative = torch.autograd.grad(
            (material * (1 - coverage)).sum(), density, retain_graph=True
        )[0]
        assert missing_derivative.item() == 0.0
        weights = torch.as_tensor(
            generator.random(material.shape), dtype=material.dtype
        )
        full = torch.autograd.grad(
            (material * weights).sum(), density, retain_graph=True
        )[0]
        sparse = torch.autograd.grad(
            (material * weights * coverage).sum(), density, retain_graph=True
        )[0]
        torch.testing.assert_close(sparse, full, rtol=0, atol=0)


def test_recording_rejects_unbounded_and_out_of_domain_geometry():
    device, transform, _, _ = fixture()
    device.fdtdx_native_grid_metadata = dict(
        coords=device.coords, boundaries=device.boundaries
    )
    with pytest.raises(ValueError, match="finite"):
        transform.recording_bounds({"grating_layer_1.dx": (-np.inf, np.inf)})
    with pytest.raises(ValueError, match="domain"):
        transform.recording_bounds({"grating_layer_1.dx": (-1.0, 1.0)})


def test_sparse_recording_context_restores_profiles_and_leaves_ports_fixed(monkeypatch):
    from core.invdes.variation import VariationSampler
    from core.invdes.variation.recording import sparse_recording_context

    class Detector:
        def aset(self, name, value):
            return SimpleNamespace(**{name: value})

    device, transform, _, _ = fixture()
    device.fdtdx_native_grid_metadata = dict(
        coords=device.coords, boundaries=device.boundaries
    )
    key = ((1.31, 0.0, 1), "Hz2", "grating_layer_1")
    port_key = ((1.31, 0.0, 1), "Hz2", "port:out_slice_1")
    value = [Detector(), None, None, 1, ((3, 7), (3, 7), (3, 6)), False]
    port_value = [Detector(), None, None, 1, ((0, 1), (3, 7), (3, 6)), False]
    profile = {key: value, port_key: port_value}
    device.port_sources_dict = {"in_slice_1_field_monitor": profile}
    sampler = VariationSampler(
        dict(enabled=True, scenarios=[{"grating_layer_1.dx": 0.1}])
    )
    monkeypatch.setenv("MAPS_SPARSE_PHASOR", "1")
    with pytest.raises(RuntimeError, match="injected"):
        with sparse_recording_context(transform.model, sampler, transform):
            assert profile[key][4][0][1] > value[4][0][1]
            assert profile[port_key] is port_value
            raise RuntimeError("injected")
    assert profile[key] is value
    monkeypatch.setenv("MAPS_SPARSE_PHASOR", "0")
    with sparse_recording_context(transform.model, sampler, transform):
        assert profile[key] is value


@pytest.mark.parametrize(
    "scenario",
    [
        {"grating_layer_1.etch_depth": 0.04},
        {"grating_layer_1.etch_depth": -0.03},
        {"grating_layer_1.thickness": 0.03, "grating_layer_1.etch_depth": 0.04},
    ],
)
def test_pattern_variation_leaves_separate_boundary_box_unchanged(scenario):
    """Pattern process errors cannot implicitly vary the second-process box."""
    device, transform, nominal, _ = fixture()
    device.static["grating_layer_1_box"] = dict(
        type="box", center=[0.0, 0.0, -0.1], size=[0.6, 0.6, 0.3], eps=12.0
    )
    static, regions, _ = transform._configuration(scenario)
    boundary = static["grating_layer_1_box"]
    thickness = 0.3
    assert boundary == device.static["grating_layer_1_box"]
    assert boundary["size"] == pytest.approx([0.6, 0.6, thickness])
    assert boundary["center"][2] - boundary["size"][2] / 2 == pytest.approx(-0.25)
    assert regions["grating_layer_1"]["size"][2] == pytest.approx(
        0.2 + scenario.get("grating_layer_1.etch_depth", 0.0)
    )
    material = transform(nominal, scenario)
    # Outside the pattern, integrate the cross-section's material volume rather
    # than thresholding partial cells. Column x=-0.25 is in the protected border.
    column = material[2, 4]
    integrated_thickness = (
        (column - device.eps_bg)
        / (12.0 - device.eps_bg)
        * torch.as_tensor(np.diff(device.boundaries[2]))
    ).sum()
    assert integrated_thickness.item() == pytest.approx(thickness, abs=1e-12)
    assert transform(nominal, {"grating_layer_1.etch_depth": 0.0}) is nominal[0]
    assert device.static["grating_layer_1_box"]["size"][2] == 0.3


@pytest.mark.parametrize(
    "scenario",
    [
        {"grating_layer_1.dx": 0.02, "grating_layer_1.dy": -0.03},
        {"grating_layer_1.eps": 0.5},
        {"grating_layer_1.thickness": 0.03},
        {"silicon_thickness_error": 0.03},
    ],
)
def test_pattern_variations_leave_all_boundary_boxes_fixed(scenario):
    """Neither local errors nor stack shifts imply a second lithography error."""
    from copy import deepcopy

    device, transform, _, _ = fixture()
    device.static["grating_layer_1_box"] = dict(
        type="box", center=[0, 0, -0.1], size=[0.6, 0.6, 0.3], eps=12.0
    )
    device.static["grating_layer_2_box"] = dict(
        type="box", center=[0, 0, 0.3], size=[0.6, 0.6, 0.2], eps=4.0
    )
    before = deepcopy(device.static)
    static, _, _ = transform._configuration(scenario)
    for name in ("grating_layer_1_box", "grating_layer_2_box"):
        assert static[name] == before[name]
    assert device.static == before
