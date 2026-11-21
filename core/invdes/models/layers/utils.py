import copy
import math
import os
from typing import Callable, List, Tuple

import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend
matplotlib.rcParams["text.usetex"] = False
import matplotlib.patches as patches
import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch
from autograd import numpy as npa
from pyutils.general import ensure_dir, logger
from scipy.ndimage import label, zoom

try:
    from spins.fdfd_solvers.waveguide_mode import solve_waveguide_mode
except ImportError:
    solve_waveguide_mode = None
from torch import Tensor

from core.utils import Slice, get_eigenmode_coefficients
from thirdparty.ceviche import constants
from thirdparty.ceviche.fdfd import compute_derivative_matrices
from thirdparty.ceviche.modes import filter_modes, normalize_modes

from .viz import abs as plot_abs
from .viz import real as plot_real

__all__ = [
    "get_grid",
    "apply_regions_gpu",
    "AdjointGradient",
    "differentiable_boundary",
    "BinaryProjection",
    "ApplyLowerLimit",
    "ApplyUpperLimit",
    "ApplyBothLimit",
    "HeavisideProjectionLayer",
    "heightProjectionLayer",
    "InsensitivePeriodLayer",
    "poynting_vector",
    "plot_eps_field",
    "plot_monitor_fields",
    "get_eigenmode_coefficients",
    "insert_mode",
    "get_temp_related_eps",
    "modulation_fn_dict",
    "apply_line_edge_roughness",
    "sample_correlated_roughness",
]


def plot_monitor_fields(
    fields: dict,
    *,
    filepath: str,
    title: str,
    components=("Ex", "Ey", "Ez"),
    quantities=("intensity", "amplitude", "phase"),
    phase_reference: str | None = None,
):
    """Plot complex vector fields sampled on a planar monitor.

    Args:
        fields: Mapping containing complex component arrays. Supported component
            keys are ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, and ``Hz``.
            Optional ``positions_m`` supplies physical planar coordinates, and
            ``spacing_um`` supplies a uniform two-axis sampling pitch and
            ``plane_axes`` names those axes, for example ``("x", "y")``.
        filepath: Output PNG path.
        title: Figure title.
        components: Ordered plot rows. A single component may be passed as a
            string. ``"E"`` and ``"H"`` request aggregate vector rows.
        quantities: Any of ``intensity``, ``amplitude``, ``real``, and ``phase``.
            Intensity and component magnitudes retain their raw values.
        phase_reference: Optional Cartesian component used for a vector-row
            phase or real-value panel. By default, the available component with
            the largest integrated intensity is used.

    Returns:
        Absolute or relative path written to ``filepath``.

    Raises:
        ValueError: If fields cannot be represented as a planar two-dimensional
            monitor or an unsupported quantity is requested.

    Notes:
        A vector field has no unique scalar phase. Therefore ``phase(E)`` and
        ``phase(H)`` use the documented ``phase_reference`` convention.
        Diagnostic tensors are expected to be detached by their owning
        objective before this function is called.
    """

    def _numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    component_data = {
        name: np.squeeze(_numpy(fields[name]))
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        if fields.get(name) is not None
    }
    if not component_data:
        raise ValueError("Monitor plot requires at least one vector-field component")

    if isinstance(components, str):
        components = (components,)
    else:
        components = tuple(components)
    component_groups = {
        "E": ("Ex", "Ey", "Ez"),
        "H": ("Hx", "Hy", "Hz"),
    }
    if isinstance(quantities, str):
        quantities = (quantities,)

    reference = next(iter(component_data.values()))
    positions = fields.get("positions_m")
    axis_labels = ("axis 0", "axis 1")
    extent = None
    if reference.ndim == 1 and positions is not None:
        positions_np = _numpy(positions)
        if positions_np.shape != (reference.size, 3):
            raise ValueError(
                "Flattened monitor positions must have shape (samples, 3), got "
                f"{positions_np.shape}"
            )
        spans = np.ptp(positions_np, axis=0)
        varying = np.flatnonzero(spans > max(float(spans.max()), 1.0) * 1e-9)
        if varying.size != 2:
            raise ValueError(
                "Flattened monitor plotting requires one planar surface; "
                f"found {varying.size} varying coordinate axes"
            )
        coords = [np.unique(positions_np[:, axis]) for axis in varying]
        if coords[0].size * coords[1].size != reference.size:
            raise ValueError("Monitor samples do not form a complete Cartesian plane")
        order = np.lexsort((positions_np[:, varying[1]], positions_np[:, varying[0]]))
        shape = (coords[0].size, coords[1].size)
        component_data = {
            name: values[order].reshape(shape)
            for name, values in component_data.items()
        }
        reference = next(iter(component_data.values()))
        axis_labels = tuple(f"{'xyz'[axis]} (um)" for axis in varying)
        extent = (
            float(coords[0][0] * 1e6),
            float(coords[0][-1] * 1e6),
            float(coords[1][0] * 1e6),
            float(coords[1][-1] * 1e6),
        )
    elif reference.ndim != 2:
        raise ValueError(
            f"Monitor components must be planar 2-D arrays, got {reference.shape}"
        )
    elif positions is not None:
        positions_np = _numpy(positions)
        if positions_np.shape == reference.shape + (3,):
            spans = np.ptp(positions_np.reshape(-1, 3), axis=0)
            varying = np.flatnonzero(spans > max(float(spans.max()), 1.0) * 1e-9)
            if varying.size == 2:
                axis_labels = tuple(f"{'xyz'[axis]} (um)" for axis in varying)
                extent = (
                    float(positions_np[..., varying[0]].min() * 1e6),
                    float(positions_np[..., varying[0]].max() * 1e6),
                    float(positions_np[..., varying[1]].min() * 1e6),
                    float(positions_np[..., varying[1]].max() * 1e6),
                )
    elif fields.get("spacing_um") is not None:
        spacing = tuple(float(value) for value in fields["spacing_um"])
        if len(spacing) != 2:
            raise ValueError("spacing_um must contain two transverse spacings")
        extent = (
            -0.5 * reference.shape[0] * spacing[0],
            0.5 * reference.shape[0] * spacing[0],
            -0.5 * reference.shape[1] * spacing[1],
            0.5 * reference.shape[1] * spacing[1],
        )
        plane_axes = fields.get("plane_axes", ("u", "v"))
        if isinstance(plane_axes, str) or len(plane_axes) != 2:
            raise ValueError("plane_axes must contain two coordinate names")
        axis_labels = tuple(f"{axis} (um)" for axis in plane_axes)

    requested_quantities = tuple(str(value).lower() for value in quantities)
    unsupported = set(requested_quantities) - {
        "intensity",
        "amplitude",
        "real",
        "phase",
    }
    if unsupported:
        raise ValueError(f"Unsupported monitor plot quantities: {sorted(unsupported)}")

    panels = []
    for component in components:
        family = [
            name
            for name in component_groups.get(component, (component,))
            if name in component_data
        ]
        if not family:
            continue
        is_vector = component in component_groups
        intensity = sum(np.abs(component_data[name]) ** 2 for name in family)
        if is_vector:
            if phase_reference is not None:
                if phase_reference not in family:
                    raise ValueError(
                        f"phase_reference {phase_reference!r} is unavailable for "
                        f"the {component} vector row; available: {family}"
                    )
                reference_name = phase_reference
            else:
                reference_name = max(
                    family,
                    key=lambda name: float(np.sum(np.abs(component_data[name]) ** 2)),
                )
        else:
            reference_name = component
        reference_values = component_data[reference_name]

        for quantity in requested_quantities:
            if quantity == "intensity":
                panels.append((intensity, f"|{component}|² (raw)", "magma", None, None))
            elif quantity == "amplitude":
                panels.append(
                    (np.sqrt(intensity), f"|{component}|", "magma", None, None)
                )
            elif quantity == "real":
                real_values = np.real(reference_values)
                limit = float(np.max(np.abs(real_values)))
                limit = 1.0 if np.isclose(limit, 0.0) else limit
                label = f"Re({component})"
                if is_vector:
                    label += f" [{reference_name} ref]"
                panels.append((real_values, label, "RdBu_r", -limit, limit))
            elif quantity == "phase":
                label = f"phase({component})"
                if is_vector:
                    label += f" [{reference_name} ref]"
                panels.append(
                    (
                        np.angle(reference_values),
                        label,
                        "twilight",
                        -np.pi,
                        np.pi,
                    )
                )
    if not panels:
        raise ValueError(
            "Requested monitor components and quantities produced no panels"
        )

    ncols = min(len(requested_quantities), len(panels))
    nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (values, panel_label, cmap, vmin, vmax) in zip(axes.flat, panels):
        image = axis.imshow(
            np.asarray(values).T,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(panel_label)
        axis.set_xlabel(axis_labels[0])
        axis.set_ylabel(axis_labels[1])
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        if not panel_label.startswith("phase("):
            formatter = matplotlib.ticker.ScalarFormatter(
                useMathText=True,
                useOffset=False,
            )
            formatter.set_powerlimits((-3, 4))
            colorbar.formatter = formatter
            colorbar.update_ticks()
    for axis in axes.flat[len(panels) :]:
        axis.set_visible(False)
    fig.suptitle(title)
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(filepath, dpi=180)
    plt.close(fig)
    return filepath


def sample_correlated_roughness(
    distances: np.ndarray,
    amplitude: float,
    correlation_length: float,
    seed: int | None = None,
) -> np.ndarray:
    """Sample a Gaussian line process with K(r)=A^2 exp(-2 r^2/Lc^2)."""
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("distances must be a non-empty one-dimensional array")
    if amplitude < 0 or correlation_length <= 0:
        raise ValueError(
            "amplitude must be non-negative and correlation_length positive"
        )

    separation = distances[:, None] - distances[None, :]
    covariance = amplitude**2 * np.exp(-2.0 * separation**2 / correlation_length**2)
    covariance.flat[:: distances.size + 1] += max(amplitude**2, 1.0) * 1e-12
    rng = np.random.default_rng(seed)
    return np.linalg.cholesky(covariance) @ rng.standard_normal(distances.size)


def _roughness_config(cfg: dict) -> dict | None:
    roughness = cfg.get("roughness")
    if roughness is None or roughness is False:
        return None
    if roughness is True:
        raise ValueError("roughness must contain amplitude and correlation_length")
    if not isinstance(roughness, dict):
        raise TypeError("roughness must be a mapping or None")
    return roughness


def _line_roughness_samples(lengths, roughness, seed_offset=0):
    amplitude = float(roughness["amplitude"])
    correlation_length = float(roughness["correlation_length"])
    seed = roughness.get("seed")
    seed = None if seed is None else int(seed) + seed_offset
    return sample_correlated_roughness(
        np.asarray(lengths), amplitude, correlation_length, seed
    )


def apply_line_edge_roughness(cfg: dict) -> dict:
    """Convert supported line-edge roughness configs into prism geometry."""
    roughness = _roughness_config(cfg)
    if roughness is None:
        return cfg
    if float(roughness.get("amplitude", 0.0)) == 0.0:
        return cfg

    geometry_type = cfg["type"]
    samples = int(roughness.get("num_samples", cfg.get("num_samples", 100)))
    if samples < 3:
        raise ValueError("roughness num_samples must be at least 3")

    if geometry_type == "box":
        center = np.asarray(cfg["center"], dtype=float)
        size = np.asarray(cfg["size"], dtype=float)
        direction = cfg.get("direction", "x").lower()[0]
        if direction not in ("x", "y"):
            raise ValueError("roughness for box geometry requires x or y direction")
        axis = 0 if direction == "x" else 1
        transverse = 1 - axis
        longitudinal = np.linspace(
            center[axis] - size[axis] / 2,
            center[axis] + size[axis] / 2,
            samples,
        )
        edge_distance = longitudinal - longitudinal[0]
        lower = _line_roughness_samples(edge_distance, roughness, 0)
        upper = _line_roughness_samples(edge_distance, roughness, 1)
        lower_coord = center[transverse] - size[transverse] / 2 + lower
        upper_coord = center[transverse] + size[transverse] / 2 + upper
        if direction == "x":
            vertices = [[x, y] for x, y in zip(longitudinal, lower_coord)]
            vertices += [[x, y] for x, y in zip(longitudinal[::-1], upper_coord[::-1])]
        else:
            vertices = [[x, y] for y, x in zip(longitudinal, lower_coord)]
            vertices += [[x, y] for y, x in zip(longitudinal[::-1], upper_coord[::-1])]
        result = copy.deepcopy(cfg)
        result.update(type="prism", vertices=vertices)
        result.pop("roughness", None)
        if len(size) >= 3:
            z_center = center[2]
            result["slab_bounds"] = (
                z_center - size[2] / 2,
                z_center + size[2] / 2,
            )
        return result

    if geometry_type != "sine_bend":
        raise ValueError(f"Line-edge roughness is not supported for {geometry_type!r}")

    start = np.asarray(cfg["start"], dtype=float)
    end = np.asarray(cfg["end"], dtype=float)
    direction = cfg.get("direction", "x").lower()
    if direction not in ("x", "y"):
        raise ValueError("roughness for sine_bend requires direction x or y")
    parameter = np.linspace(0.0, 1.0, samples)
    delta = end - start
    smooth = parameter - np.sin(2 * np.pi * parameter) / (2 * np.pi)
    if direction == "x":
        centerline = np.column_stack(
            (start[0] + parameter * delta[0], start[1] + smooth * delta[1])
        )
    else:
        centerline = np.column_stack(
            (start[0] + smooth * delta[0], start[1] + parameter * delta[1])
        )
    tangent = np.gradient(centerline, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    arc_length = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))]
    )
    lower = _line_roughness_samples(arc_length, roughness, 0)
    upper = _line_roughness_samples(arc_length, roughness, 1)
    half_width = float(cfg["width"]) / 2
    lower_edge = centerline + normal * (-half_width + lower[:, None])
    upper_edge = centerline + normal * (half_width + upper[:, None])
    result = copy.deepcopy(cfg)
    result.update(
        type="prism", vertices=lower_edge.tolist() + upper_edge[::-1].tolist()
    )
    result.pop("roughness", None)
    if "slab_bounds" not in result and len(start) >= 3:
        z_center = start[2]
        thickness = float(cfg.get("height", 0.0))
        result["slab_bounds"] = (
            z_center - thickness / 2,
            z_center + thickness / 2,
        )
    return result


def get_temp_related_eps(
    eps, temp, temp_0: float = 300, eps_r_0: float = 3.48**2, dn_dT=1.8e-4
):
    # and we treat the air as it is independent of the temperature
    eps_max = eps.max()
    eps_min = eps.min()
    eps = (eps - eps_min) / (eps_max - eps_min)  # (0, 1)
    n_si = math.sqrt(eps_r_0) + (temp - temp_0) * dn_dT
    eps = eps * (n_si**2 / eps_r_0)
    eps = eps * (eps_max - eps_min) + eps_min
    return eps


def temperature_modulation(
    eps: float, T: float, T0: float = 300, dn_dT: float = 1.8e-4
):
    return (math.sqrt(eps) + (T - T0) * dn_dT) ** 2


modulation_fn_dict = {
    "temperature": temperature_modulation,
}


def get_grid(shape, dl):
    # dl in um
    # computes the coordinates in the grid

    (Nx, Ny) = shape
    # if Ny % 2 == 0:
    #     Ny -= 1
    # coordinate vectors
    x_coord = np.linspace(-(Nx - 1) / 2 * dl, (Nx - 1) / 2 * dl, Nx)
    y_coord = np.linspace(-(Ny - 1) / 2 * dl, (Ny - 1) / 2 * dl, Ny)

    # x and y coordinate arrays
    xs, ys = np.meshgrid(x_coord, y_coord, indexing="ij")
    return (xs, ys)


def apply_regions_gpu(reg_list, xs, ys, eps_r_list, eps_bg, device="cuda"):
    # Convert inputs to tensors and move them to the GPU
    xs = torch.tensor(xs, device=device)
    ys = torch.tensor(ys, device=device)

    # Handle scalars to lists conversion
    if isinstance(eps_r_list, (int, float)):
        eps_r_list = [eps_r_list] * len(reg_list)
    if not isinstance(reg_list, (list, tuple)):
        reg_list = [reg_list]

    # Initialize permittivity tensor with background value
    eps_r = torch.full(xs.shape, eps_bg, device=device, dtype=torch.float32)

    # Convert region functions to a vectorized form using PyTorch operations
    for e, reg in zip(eps_r_list, reg_list):
        # Assume that reg is a lambda or function that can be applied to tensors
        material_mask = reg(xs, ys)  # This should return a boolean tensor
        # print("this is the dtype of the eps_r", eps_r.dtype)
        # print("this is the dtype of the e", e.dtype)
        eps_r[material_mask] = e

    return eps_r.cpu().numpy()


class AdjointGradient(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        obj_and_grad_fn: Callable,
        adjoint_mode: str,
        resolution: int,
        *permittivity_list: List[Tensor],
    ) -> Tensor:
        obj = obj_and_grad_fn(adjoint_mode, "need_value", resolution, permittivity_list)

        ctx.save_for_backward(*permittivity_list)
        ctx.save_adjoint_mode = adjoint_mode
        ctx.save_obj_and_grad_fn = obj_and_grad_fn
        ctx.save_resolution = resolution
        obj = torch.tensor(
            obj,
            device=permittivity_list[0].device,
            dtype=permittivity_list[0].dtype,
            requires_grad=True,
        )
        return obj

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        permittivity_list, adjoint_mode, obj_and_grad_fn, resolution = (
            ctx.saved_tensors,
            ctx.save_adjoint_mode,
            ctx.save_obj_and_grad_fn,
            ctx.save_resolution,
        )
        grad = obj_and_grad_fn(
            adjoint_mode, "need_gradient", resolution, permittivity_list
        )

        gradients = []
        if adjoint_mode == "reflection":
            if isinstance(grad, np.ndarray):  # make sure the gradient is torch tensor
                grad = (
                    torch.from_numpy(grad)
                    .to(permittivity_list[0].device)
                    .to(permittivity_list[0].dtype)
                )
            grad = grad.view_as(permittivity_list[0])
            gradients.append(grad_output * grad)
            return None, None, None, *gradients
        if adjoint_mode == "legume":
            if isinstance(grad, np.ndarray):  # make sure the gradient is torch tensor
                grad = (
                    torch.from_numpy(grad)
                    .to(permittivity_list[0].device)
                    .to(permittivity_list[0].dtype)
                )
            grad = grad.view_as(permittivity_list[0])
            gradients.append(grad_output * grad)
        else:
            if isinstance(
                grad, list
            ):  # which means that there are multiple design regions
                for i, g in enumerate(grad):
                    if isinstance(
                        g, np.ndarray
                    ):  # make sure the gradient is torch tensor
                        g = (
                            torch.from_numpy(g)
                            .to(permittivity_list[i].device)
                            .to(permittivity_list[i].dtype)
                        )

                    if (
                        len(g.shape) == 2
                    ):  # summarize the gradient along different frequencies
                        g = torch.sum(g, dim=-1)
                    g = g.view_as(permittivity_list[i])
                    gradients.append(grad_output * g)
            else:
                # there are two possibility:
                #   1. there is only one design region and the grad is a ndarray
                #   2. the mode is legume
                if isinstance(
                    grad, np.ndarray
                ):  # make sure the gradient is torch tensor
                    grad = (
                        torch.from_numpy(grad)
                        .to(permittivity_list[0].device)
                        .to(permittivity_list[0].dtype)
                    )

                # if len(grad.shape) == 2:  # summarize the gradient along different frquencies
                #     grad = torch.sum(grad, dim=-1)
                if adjoint_mode == "fdtd":
                    grad = grad.view_as(permittivity_list[0])
                elif adjoint_mode == "fdfd_angler":
                    Nx = int(grad.numel() // permittivity_list[0].shape[1])
                    grad = grad.view(Nx, permittivity_list[0].shape[1])
                elif "ceviche" in adjoint_mode:
                    if len(grad.shape) == 2:
                        Nx = round(grad.numel() // permittivity_list[0].shape[1])
                        grad = grad.view(Nx, permittivity_list[0].shape[1])
                        # print("this is the gradient in the custom function: ", grad)
                    elif len(grad.shape) == 3:
                        Nx = round(grad[0].numel() // permittivity_list[0].shape[1])
                        grad = grad.view(-1, Nx, permittivity_list[0].shape[1])
                else:
                    raise ValueError(f"mode {adjoint_mode} is not supported")
                gradients.append(grad_output * grad)
        return None, None, None, *gradients


class differentiable_boundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, total_length, T):
        ctx.save_for_backward(w)
        ctx.x = x
        ctx.total_length = total_length
        ctx.T = T
        w1 = total_length - w
        output = torch.where(
            x < -w / 2,
            1
            / (
                torch.exp(
                    -(((x + w / 2 + w1 / 2) ** 2 - (w1 / 2) ** 2) / T)
                    * (total_length / (3 * w1)) ** 2
                )
                + 1
            ),
            torch.where(
                x < w / 2,
                1
                / (
                    torch.exp(
                        ((x**2 - (w / 2) ** 2) / T) * (total_length / (3 * w)) ** 2
                    )
                    + 1
                ),
                1
                / (
                    torch.exp(
                        -(((x - w / 2 - w1 / 2) ** 2 - (w1 / 2) ** 2) / T)
                        * (total_length / (3 * w1)) ** 2
                    )
                    + 1
                ),
            ),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (w,) = ctx.saved_tensors
        x = ctx.x
        total_length = ctx.total_length
        T = ctx.T

        w1 = total_length - w

        # Precompute common expressions
        exp1 = torch.exp(
            -(((x + w / 2 + w1 / 2) ** 2 - (w1 / 2) ** 2) / T)
            * (total_length / (3 * w1)) ** 2
        )
        exp2 = torch.exp(((x**2 - (w / 2) ** 2) / T) * (total_length / (3 * w)) ** 2)
        exp3 = torch.exp(
            -(((x - w / 2 - w1 / 2) ** 2 - (w1 / 2) ** 2) / T)
            * (total_length / (3 * w1)) ** 2
        )

        denominator1 = (exp1 + 1) ** 2
        denominator2 = (exp2 + 1) ** 2
        denominator3 = (exp3 + 1) ** 2

        doutput_dw = torch.where(
            x < -w / 2,
            -exp1
            * (-2 * total_length**2 * (x + total_length / 2) ** 2)
            / (9 * w1**3 * T * denominator1),
            torch.where(
                x < w / 2,
                -exp2 * (-2 * total_length**2 * x**2) / (9 * w**3 * T * denominator2),
                -exp3
                * (-2 * total_length**2 * (x - total_length / 2) ** 2)
                / (9 * w1**3 * T * denominator3),
            ),
        )

        # not quite sure with the following code
        grad_w = (grad_output * doutput_dw).sum()

        return None, grad_w, None, None


class BinaryProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permittivity: Tensor, T_bny: float, T_threshold: float):
        ctx.T_bny = T_bny
        ctx.T_threshold = T_threshold
        ctx.save_for_backward(permittivity)
        result = (torch.tanh((0.5 - permittivity) / T_bny) + 1) / 2
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # if T_bny is larger than T_threshold, then use the automatic differentiation of the tanh function
        # if the T_bny is smaller than T_threshold, then use the gradient as if T_bny is T_threshold
        T_bny = ctx.T_bny
        T_threshold = ctx.T_threshold
        (permittivity,) = ctx.saved_tensors

        if T_bny > T_threshold:
            grad = (
                -grad_output
                * (1 - torch.tanh((0.5 - permittivity) / T_bny) ** 2)
                / T_bny
            )
        else:
            grad = (
                -grad_output
                * (1 - torch.tanh((0.5 - permittivity) / T_threshold) ** 2)
                / T_threshold
            )

        return grad, None, None


class LevelSetInterp1D(object):
    """This class implements the level set surface using Gaussian radial basis functions in 1D."""

    def __init__(
        self,
        x0: Tensor = None,  # 1D input coordinates
        z0: Tensor = None,  # Corresponding level set values
        sigma: float = None,  # Gaussian RBF standard deviation
    ):
        # Input data
        self.x0 = x0  # 1D coordinates
        self.z0 = z0  # Level set values
        self.sig = sigma  # Gaussian kernel width

        # Builds the level set interpolation model
        gauss_kernel = self.gaussian(self.x0, self.x0)
        self.model = torch.linalg.solve(
            gauss_kernel, self.z0
        )  # Solving gauss_kernel @ model = z0

    def gaussian(self, xi, xj):
        # Compute the Gaussian RBF kernel
        dist = torch.abs(xi.reshape(-1, 1) - xj.reshape(1, -1))
        return torch.exp(-(dist**2) / (2 * self.sig**2))

    def get_ls(self, x1):
        # Interpolate the level set function at new points x1
        gauss_matrix = self.gaussian(self.x0, x1)
        ls = gauss_matrix.T @ self.model
        return ls


def get_eps_1d(
    design_param,
    x_rho,
    x_phi,
    rho_size,
    nx_rho,
    nx_phi,
    plot_levelset=False,
    sharpness=0.1,
):
    """Returns the permittivities defined by the zero level set isocontour for a 1D case"""

    # Initialize the LevelSetInterp model for 1D case
    phi_model = LevelSetInterp1D(x0=x_rho, z0=design_param, sigma=rho_size)

    # Obtain the level set function phi
    phi = phi_model.get_ls(x1=x_phi)

    eps_phi = 0.5 * (torch.tanh(sharpness * phi) + 1)

    # Reshape the design parameters into a 1D array
    eps = torch.reshape(eps_phi, (nx_phi,))

    # Plot the level set surface if required
    if plot_levelset:
        rho = np.reshape(design_param, (nx_rho,))
        phi = np.reshape(phi, (nx_phi,))
        plot_level_set_1d(x0=x_rho, rho=rho, x1=x_phi, phi=phi)

    return eps


# Function to plot the level set in 1D
def plot_level_set_1d(x0, rho, x1, phi, path="./1D_Level_Set_Plot.png"):
    """
    Plots the level set for the 1D case.

    x0: array-like, coordinates corresponding to design parameters
    rho: array-like, design parameters
    x1: array-like, coordinates where phi is evaluated
    phi: array-like, level set values
    """

    fig, ax1 = plt.subplots(figsize=(12, 6), tight_layout=True)

    # Plot the design parameters as scatter plot
    ax1.scatter(x0, rho, color="black", label="Design Parameters")

    # Plot the level set function
    ax1.plot(x1, phi, color="blue", label="Level Set Function")

    # Highlight the zero level set
    ax1.axhline(0, color="red", linestyle="--", label="Zero Level Set")

    ax1.set_title("1D Level Set Plot")
    ax1.set_xlabel("x ($\mu m$)")
    ax1.set_ylabel("Value")
    ax1.legend()

    plt.savefig(path)


class ApplyLowerLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lower_limit):
        ctx.save_for_backward(x)
        ctx.lower_limit = lower_limit
        return torch.maximum(x, lower_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors
        lower_limit = ctx.lower_limit

        # Compute gradient
        # If x > lower_limit, propagate grad_output normally
        # If x <= lower_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
        )  # None for lower_limit since it does not require gradients


class ApplyUpperLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, upper_limit):
        ctx.save_for_backward(x)
        ctx.upper_limit = upper_limit
        return torch.minimum(x, upper_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors
        upper_limit = ctx.upper_limit

        # Compute gradient
        # If x > upper_limit, propagate grad_output normally
        # If x <= upper_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
        )  # None for upper_limit since it does not require gradients


class ApplyBothLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, upper_limit, lower_limit):
        ctx.save_for_backward(x)
        ctx.upper_limit = upper_limit
        ctx.lower_limit = lower_limit
        return torch.minimum(torch.maximum(x, lower_limit), upper_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors
        upper_limit = ctx.upper_limit
        lower_limit = ctx.lower_limit

        # Compute gradient
        # If x > upper_limit, propagate grad_output normally
        # If x <= upper_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
            None,
        )  # None for upper_limit and lower_limit since they do not require gradients


class HeavisideProjectionLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, eta, threshold):
        ctx.save_for_backward(x, beta, eta)
        ctx.threshold = threshold
        return torch.where(
            x < eta,
            torch.tensor(0, dtype=torch.float32).to(x.device),
            torch.tensor(1, dtype=torch.float32).to(x.device),
        )
        if beta < threshold:
            return (torch.tanh(threshold * eta) + torch.tanh(threshold * (x - eta))) / (
                torch.tanh(threshold * eta) + torch.tanh(threshold * (1 - eta))
            )
        else:
            return (torch.tanh(beta * eta) + torch.tanh(beta * (x - eta))) / (
                torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
            )

    @staticmethod
    def backward(ctx, grad_output):
        x, beta, eta = ctx.saved_tensors

        threshold = ctx.threshold

        grad = (
            grad_output
            * (beta * (1 - (torch.tanh(beta * (x - eta))) ** 2))
            / (torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta)))
        )

        return grad, None, None, None


class heightProjectionLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ridge_height, height_mask, sharpness, threshold):
        ctx.save_for_backward(ridge_height, height_mask)
        ctx.sharpness = sharpness
        return torch.where(
            height_mask < ridge_height,
            torch.tensor(1, dtype=torch.float32).to(ridge_height.device),
            torch.tensor(0, dtype=torch.float32).to(ridge_height.device),
        )
        if sharpness < threshold:
            return torch.tanh(threshold * (ridge_height - height_mask)) / 2 + 0.5
        else:
            return torch.tanh(sharpness * (ridge_height - height_mask)) / 2 + 0.5

    @staticmethod
    def backward(ctx, grad_output):
        ridge_height, height_mask = ctx.saved_tensors
        sharpness = ctx.sharpness

        grad = (
            grad_output
            * sharpness
            * (1 - (torch.tanh(sharpness * (ridge_height - height_mask))) ** 2)
            / 2
        )

        return grad, None, None, None


class InsensitivePeriodLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, i):
        ctx.save_for_backward(x)
        ctx.i = i
        return x * i

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        i = ctx.i
        grad = grad_output

        return grad, None


def poynting_vector(
    Hx, Hy, Ez, grid_step, monitor=None, direction="x+", autograd=False
):
    if autograd:
        conj = npa.conj
        real = npa.real
        sum = npa.sum
    else:
        conj = np.conj
        real = np.real
        sum = np.sum
    if isinstance(monitor, (Slice, np.ndarray)):
        Hx = Hx[monitor]
        Hy = Hy[monitor]
        Ez_conj = conj(Ez[monitor])

    if direction == "x+":
        P = sum(real(Ez_conj * Hy)) * (-grid_step)
    elif direction == "x-":
        P = sum(real(Ez_conj * Hy)) * grid_step
    elif direction == "y+":
        P = sum(real(Ez_conj * Hx)) * grid_step
    elif direction == "y-":
        P = sum(real(Ez_conj * Hx)) * (-grid_step)
    else:
        raise ValueError("Invalid direction")
    return P


def loc2ind(
    loc: Tuple[float, float],
    box_size: Tuple[float, float],
    box_shape: Tuple[float, float],
    clip: bool = True,
):
    ## take arbitrary dimensions and return the index of the location in the box (if clip), otherwise can be outside of box
    indices = []
    ## box is in the center of the space, center aligns with origin
    assert (
        len(loc) == len(box_size) == len(box_shape)
    ), "The dimensions of loc, box_size, and box_shape should be the same"
    loc = np.array(loc)
    box_size = np.array(box_size)
    box_shape = np.array(box_shape)
    indices = np.round((loc + box_size / 2) / box_size * box_shape).astype(np.int32)

    if clip:
        indices = np.clip(indices, 0, box_shape - 1)
    return indices


def slice_to_indices(slice_obj, torch_tensor: bool = False):
    if isinstance(slice_obj, slice):
        start = slice_obj.start if slice_obj.start is not None else 0
        stop = slice_obj.stop
        if stop is None:
            raise ValueError(f"Slice {slice_obj} must have a stop value.")
        if torch_tensor:
            return torch.arange(start, stop, dtype=torch.long)
        return np.arange(start, stop, dtype=np.int64)
    axis_array = np.asarray(slice_obj)
    if axis_array.ndim == 0:
        if torch_tensor:
            return torch.tensor([int(axis_array.item())], dtype=torch.long)
        else:
            return np.asarray([int(axis_array.item())], dtype=np.int64)
    indices = np.unique(axis_array.astype(np.int64).reshape(-1))
    if torch_tensor:
        return torch.from_numpy(indices).to(dtype=torch.long)
    else:
        return indices


def slice3d_to_indices(slice_obj, torch_tensor: bool = False):
    if (
        not hasattr(slice_obj, "x")
        or not hasattr(slice_obj, "y")
        or not hasattr(slice_obj, "z")
    ):
        xs = slice_to_indices(slice_obj[0], torch_tensor=torch_tensor)
        ys = slice_to_indices(slice_obj[1], torch_tensor=torch_tensor)
        zs = slice_to_indices(slice_obj[2], torch_tensor=torch_tensor)
    else:
        xs = slice_to_indices(slice_obj.x, torch_tensor=torch_tensor)
        ys = slice_to_indices(slice_obj.y, torch_tensor=torch_tensor)
        zs = slice_to_indices(slice_obj.z, torch_tensor=torch_tensor)
    return xs, ys, zs


def _symmetric_percentile_limit(values, percentile: float = 99.95) -> float:
    values = np.asarray(values)
    finite_abs = np.abs(values[np.isfinite(values)])
    if finite_abs.size == 0:
        return 1.0
    limit = float(np.percentile(finite_abs, percentile))
    return 1.0 if np.isclose(limit, 0.0) else limit


def plot_eps_field(
    field,
    component: str,
    eps,
    base_eps=None,
    show_delta_eps: bool | None = None,
    thermal_map=None,
    heat_source_map=None,
    thermal_map_name: str | None = None,
    eps_grad=None,
    param_grad=None,
    param_x_width=None,
    param_y_height=None,
    param_name: str = "param",
    edgebox_grad=None,
    monitors=[],
    filepath=None,
    zoom_eps_factor=1,
    zoom_eps_center=(0, 0),
    x_width=1,
    y_height=1,
    NPML=[0, 0],
    field_stat: str = "abs",  # "abs" or "real" or "abs_real"
    title: str = None,
    x_shift_coord: int = 0,
    x_shift_idx: int = 0,
    if_gif: bool = False,
):
    import matplotlib.ticker as mticker

    eps_grad_arr = None
    if isinstance(eps_grad, bool):
        if eps_grad and isinstance(eps, torch.Tensor) and eps.grad is not None:
            eps_grad_arr = eps.grad.detach().cpu().numpy()
    elif eps_grad is not None:
        eps_grad_arr = (
            eps_grad.detach().cpu().numpy()
            if isinstance(eps_grad, torch.Tensor)
            else np.asarray(eps_grad)
        )

    if isinstance(field, torch.Tensor):
        field = field.data.cpu().numpy()
    if isinstance(eps, torch.Tensor):
        eps = eps.detach().cpu().numpy()
    base_eps_arr = None
    if base_eps is not None:
        base_eps_arr = (
            base_eps.detach().cpu().numpy()
            if isinstance(base_eps, torch.Tensor)
            else np.asarray(base_eps)
        )
        if base_eps_arr.shape != eps.shape:
            raise ValueError(
                f"base_eps must have the same shape as eps, got {base_eps_arr.shape} and {eps.shape}"
            )
    thermal_map_arr = None
    if thermal_map is not None:
        thermal_map_arr = (
            thermal_map.detach().cpu().numpy()
            if isinstance(thermal_map, torch.Tensor)
            else np.asarray(thermal_map)
        )
    heat_source_map_arr = None
    if heat_source_map is not None:
        heat_source_map_arr = (
            heat_source_map.detach().cpu().numpy()
            if isinstance(heat_source_map, torch.Tensor)
            else np.asarray(heat_source_map)
        )
    if eps_grad_arr is not None and eps_grad_arr.shape != eps.shape:
        raise ValueError(
            f"eps_grad must have the same shape as eps, got {eps_grad_arr.shape} and {eps.shape}"
        )

    if filepath is not None:
        ensure_dir(os.path.dirname(filepath))

    if show_delta_eps is None:
        show_delta_eps = base_eps_arr is not None and not np.allclose(
            np.real(eps),
            np.real(base_eps_arr),
            atol=1e-12,
            rtol=1e-9,
        )
    else:
        show_delta_eps = bool(show_delta_eps) and base_eps_arr is not None

    field_stat = field_stat.lower().split("_")
    edgebox_edges = (
        edgebox_grad.get("edge_knots", []) if isinstance(edgebox_grad, dict) else []
    )
    polygon_vertices = (
        edgebox_grad.get("polygon_vertices") if isinstance(edgebox_grad, dict) else None
    )
    polygon_grad = (
        edgebox_grad.get("polygon_grad") if isinstance(edgebox_grad, dict) else None
    )
    if polygon_vertices is not None:
        polygon_vertices = np.asarray(polygon_vertices, dtype=float)
        polygon_grad = np.asarray(polygon_grad, dtype=float)
        if polygon_vertices.shape != polygon_grad.shape or polygon_vertices.ndim != 2:
            raise ValueError(
                "polygon_vertices and polygon_grad must both have shape [n_vertices, 2]"
            )
    n_rows = len(field_stat) + 1 + int(eps_grad_arr is not None)
    n_rows += int(thermal_map_arr is not None)
    n_rows += int(show_delta_eps)
    n_rows += len(edgebox_edges)
    polygon_row_span = 2 if polygon_vertices is not None else 0
    n_rows += int(polygon_vertices is not None)

    def _positive_float(name, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {value}")
        return value

    x_width = _positive_float("x_width", x_width)
    y_height = _positive_float("y_height", y_height)

    def _finite_minmax(a):
        a = np.asarray(a)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(np.min(finite)), float(np.max(finite))

    field_abs_vmin, field_abs_vmax = _finite_minmax(np.abs(field))
    field_real_min, field_real_max = _finite_minmax(np.real(field))
    field_real_lim = max(abs(field_real_min), abs(field_real_max))
    if np.isclose(field_real_lim, 0.0):
        field_real_lim = 1.0

    field_intensity_vmin, field_intensity_vmax = _finite_minmax(np.abs(field) ** 2)
    eps_vmin, eps_vmax = _finite_minmax(np.real(eps))
    if show_delta_eps:
        delta_eps_arr = np.real(eps) - np.real(base_eps_arr)
        delta_min, delta_max = _finite_minmax(delta_eps_arr)
        delta_lim = max(abs(delta_min), abs(delta_max))
        if np.isclose(delta_lim, 0.0):
            delta_lim = 1.0
    else:
        delta_eps_arr = None
    if thermal_map_arr is not None:
        thermal_vmin, thermal_vmax = _finite_minmax(np.real(thermal_map_arr))
        thermal_lim = max(abs(thermal_vmin), abs(thermal_vmax))
        if np.isclose(thermal_lim, 0.0):
            thermal_lim = 1.0

    if eps_grad_arr is not None:
        grad_lim = _symmetric_percentile_limit(np.real(eps_grad_arr))

    param_grad_arr = None
    if param_grad is not None:
        param_grad_arr = (
            param_grad.detach().cpu().numpy()
            if isinstance(param_grad, torch.Tensor)
            else np.asarray(param_grad)
        )
        if param_grad_arr.ndim != 2:
            raise ValueError(f"param_grad must be 2D, got shape {param_grad_arr.shape}")
        param_x_width = _positive_float(
            "param_x_width", x_width if param_x_width is None else param_x_width
        )
        param_y_height = _positive_float(
            "param_y_height", y_height if param_y_height is None else param_y_height
        )
        param_min, param_max = _finite_minmax(np.real(param_grad_arr))
        param_lim = max(abs(param_min), abs(param_max))
        if np.isclose(param_lim, 0.0):
            param_lim = 1.0

    left_margin = 1.00
    right_margin = 0.45
    bottom_margin = 1.00
    top_margin = 0.95 + (0.45 if title is not None else 0.0)
    row_gap = 0.70
    cbar_gap = 0.10
    cbar_width = 0.14

    target_fig_w = 7.2
    target_fig_h = 10.0

    fixed_w = left_margin + right_margin + cbar_gap + cbar_width
    fixed_h = bottom_margin + top_margin + (n_rows - 1) * row_gap

    avail_w = target_fig_w - fixed_w
    scale_w = avail_w / x_width if x_width > 0 else 1.0
    avail_h = target_fig_h - fixed_h
    scale_h = avail_h / (n_rows * y_height) if y_height > 0 else 1.0

    scale = min(scale_w, scale_h)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    panel_w = scale * x_width
    panel_h = scale * y_height
    min_row_band_h = 1.85 if eps_grad_arr is None else 2.10
    row_band_h = max(panel_h, min_row_band_h)

    fig_w = fixed_w + panel_w
    fig_h = fixed_h + n_rows * row_band_h
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    font_scale = np.clip(row_band_h / 2.2, 0.80, 1.20)
    panel_title_fs = 15 * font_scale
    label_fs = 13 * font_scale
    tick_fs = 11 * font_scale
    cbar_tick_fs = 9 * font_scale
    suptitle_fs = 18 * font_scale

    def _add_axes(row):
        y_band0 = fig_h - top_margin - (row + 1) * row_band_h - row * row_gap
        y0 = y_band0 + 0.5 * (row_band_h - panel_h)

        ax_rect = [
            left_margin / fig_w,
            y0 / fig_h,
            panel_w / fig_w,
            panel_h / fig_h,
        ]
        cax_rect = [
            (left_margin + panel_w + cbar_gap) / fig_w,
            y0 / fig_h,
            cbar_width / fig_w,
            panel_h / fig_h,
        ]
        return fig.add_axes(ax_rect), fig.add_axes(cax_rect)

    def _add_edgebox_axes(row):
        y_band0 = fig_h - top_margin - (row + 1) * row_band_h - row * row_gap
        y0 = y_band0 + 0.5 * (row_band_h - panel_h)
        content_width = fig_w - left_margin - right_margin
        axis_gap = 0.55
        axis_width = (content_width - 2 * axis_gap) / 3
        axes = []
        for col in range(3):
            x0 = left_margin + col * (axis_width + axis_gap)
            axes.append(
                fig.add_axes(
                    [
                        x0 / fig_w,
                        y0 / fig_h,
                        axis_width / fig_w,
                        panel_h / fig_h,
                    ]
                )
            )
        return axes

    def _add_polygon_axes(row):
        y_band0 = (
            fig_h - top_margin - (row + polygon_row_span) * row_band_h - row * row_gap
        )
        content_width = fig_w - left_margin - right_margin
        axis_gap = 1.25
        axis_height = row_band_h - 0.5 * axis_gap
        cbar_gap = 0.10
        cbar_width = 0.14
        axis_width = content_width - cbar_gap - cbar_width
        axes = []
        cb_axes = []
        for row_offset in range(2):
            y0 = y_band0 + (1 - row_offset) * (axis_height + axis_gap)
            axes.append(
                fig.add_axes(
                    [
                        left_margin / fig_w,
                        y0 / fig_h,
                        axis_width / fig_w,
                        axis_height / fig_h,
                    ]
                )
            )
            cb_axes.append(
                fig.add_axes(
                    [
                        (left_margin + axis_width + cbar_gap) / fig_w,
                        y0 / fig_h,
                        cbar_width / fig_w,
                        axis_height / fig_h,
                    ]
                )
            )
        return axes, cb_axes

    def _imshow_phys(axis, data_2d, extent, **kwargs):
        im = axis.imshow(
            np.asarray(data_2d).T,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
            **kwargs,
        )
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect("equal", adjustable="box")
        return im

    def _draw_npml(axis, shape_2d, npml_2d, x_phys, y_phys):
        n0, n1 = shape_2d
        pml0, pml1 = [max(0, int(v)) for v in npml_2d]
        pml0 = min(pml0, n0 // 2)
        pml1 = min(pml1, n1 // 2)
        if pml0 == 0 and pml1 == 0:
            return

        x_min, x_max = -x_phys / 2, x_phys / 2
        y_min, y_max = -y_phys / 2, y_phys / 2
        dx = x_phys / max(n0, 1)
        dy = y_phys / max(n1, 1)
        pml_x = pml0 * dx
        pml_y = pml1 * dy
        rect_kw = dict(facecolor="gray", alpha=0.40, edgecolor="none", zorder=5)

        if pml_x > 0:
            axis.add_patch(patches.Rectangle((x_min, y_min), pml_x, y_phys, **rect_kw))
            axis.add_patch(
                patches.Rectangle((x_max - pml_x, y_min), pml_x, y_phys, **rect_kw)
            )

        if pml_y > 0:
            inner_x0 = x_min + pml_x
            inner_w = max(x_phys - 2 * pml_x, 0.0)
            axis.add_patch(
                patches.Rectangle((inner_x0, y_min), inner_w, pml_y, **rect_kw)
            )
            axis.add_patch(
                patches.Rectangle((inner_x0, y_max - pml_y), inner_w, pml_y, **rect_kw)
            )

    def _draw_heat_source_frames(axis, source_2d, extent):
        if source_2d is None:
            return
        source_mask = np.asarray(np.abs(source_2d) > 0, dtype=bool)
        if source_mask.ndim != 2 or not np.any(source_mask):
            return

        component_labels, num_components = label(source_mask)
        n0, n1 = source_mask.shape
        x_min, x_max, y_min, y_max = extent
        dx = (x_max - x_min) / max(n0, 1)
        dy = (y_max - y_min) / max(n1, 1)
        frame_kw = dict(
            fill=False,
            edgecolor="purple",
            linestyle=":",
            linewidth=max(1.0, 1.6 * font_scale),
            alpha=0.40,
            zorder=22,
        )

        for component_idx in range(1, num_components + 1):
            xs, ys = np.nonzero(component_labels == component_idx)
            if xs.size == 0:
                continue
            left = x_min + float(xs.min()) * dx
            bottom = y_min + float(ys.min()) * dy
            width = float(xs.max() - xs.min() + 1) * dx
            height = float(ys.max() - ys.min() + 1) * dy
            axis.add_patch(patches.Rectangle((left, bottom), width, height, **frame_kw))

    def _format_axis(axis, xlabel, ylabel):
        axis.set_xlabel(xlabel, fontsize=label_fs, labelpad=2)
        axis.set_ylabel(ylabel, fontsize=label_fs, labelpad=2)

        nbins_x = 3 if panel_w < 1.5 else 4 if panel_w < 2.3 else 5
        nbins_y = 3 if panel_h < 1.2 else 4 if panel_h < 2.0 else 5

        axis.xaxis.set_major_locator(mticker.MaxNLocator(nbins=nbins_x))
        axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nbins_y))
        axis.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        axis.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        axis.tick_params(
            axis="both",
            labelsize=tick_fs,
            pad=1.5,
            length=2.5,
            width=0.6,
        )

    def _add_colorbar(im, cax):
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=cbar_tick_fs, pad=1.0, length=2.5, width=0.6)
        cb.locator = mticker.MaxNLocator(nbins=4)
        cb.formatter = mticker.ScalarFormatter(useMathText=True)
        cb.formatter.set_powerlimits((-2, 3))
        cb.update_ticks()
        cb.ax.yaxis.get_offset_text().set_size(max(cbar_tick_fs * 0.9, 7.0))
        return cb

    def _idx_to_phys(values, dim_len, dim_phys):
        values = np.asarray(values, dtype=float)
        if dim_len <= 1:
            out = np.zeros_like(values, dtype=float)
        else:
            out = (values / (dim_len - 1) - 0.5) * dim_phys
        out[np.isnan(values)] = np.nan
        return out

    def _insert_nan_gaps(vals):
        vals = np.asarray(vals, dtype=float)
        if vals.size <= 1:
            return vals
        vals = vals[np.isfinite(vals)]
        vals = np.unique(vals)
        vals.sort()
        if vals.size <= 1:
            return vals
        diffs = np.diff(vals)
        positive_diffs = diffs[diffs > 1e-12]
        step = np.median(positive_diffs) if positive_diffs.size > 0 else 1.0
        gap_threshold = max(1.5 * step, step + 1e-9)
        out = [vals[0]]
        for a, b in zip(vals[:-1], vals[1:]):
            if b - a > gap_threshold:
                out.append(np.nan)
            out.append(b)
        return np.asarray(out, dtype=float)

    def _plot_index_line(axis, x_idx, y_idx, color):
        x_phys = _idx_to_phys(x_idx, field.shape[0], x_width)
        y_phys = _idx_to_phys(y_idx, field.shape[1], y_height)
        axis.plot(
            x_phys,
            y_phys,
            color=color,
            alpha=0.4,
            linewidth=max(0.8, 1.2 * font_scale),
            zorder=20,
        )

    def _scatter_index_points(axis, x_idx, y_idx, color):
        x_phys = _idx_to_phys(x_idx, field.shape[0], x_width)
        y_phys = _idx_to_phys(y_idx, field.shape[1], y_height)
        axis.scatter(
            x_phys,
            y_phys,
            c=color,
            s=max(2.0, 4.0 * font_scale),
            alpha=0.4,
            linewidths=0,
            zorder=21,
        )

    def _draw_monitors(axis):
        if monitors is None or len(monitors) == 0:
            return
        for m in monitors:
            if not isinstance(m, (tuple, list)) or len(m) < 2:
                continue
            obj, color = m[0], m[1]
            if isinstance(obj, (tuple, list)) and len(obj) == 2:
                obj = Slice(x=obj[0], y=obj[1])
            if isinstance(obj, Slice):
                if isinstance(obj.x, int):
                    ys = _insert_nan_gaps(np.arange(obj.y.start, obj.y.stop))
                    xs = np.full_like(ys, float(obj.x), dtype=float)
                    xs[np.isnan(ys)] = np.nan
                    _plot_index_line(axis, xs, ys, color)
                elif isinstance(obj.y, int):
                    xs = _insert_nan_gaps(np.arange(obj.x.start, obj.x.stop))
                    ys = np.full_like(xs, float(obj.y), dtype=float)
                    ys[np.isnan(xs)] = np.nan
                    _plot_index_line(axis, xs, ys, color)
                elif isinstance(obj.x, np.ndarray) and isinstance(obj.y, np.ndarray):
                    # xs = obj.x[:, 0].astype(float)
                    # ys = obj.y[0].astype(float)
                    xs = obj.x[:].astype(float)
                    ys = obj.y[:].astype(float)
                    x_line = _insert_nan_gaps(xs)
                    y_line = _insert_nan_gaps(ys)
                    x_min = np.nanmin(xs)
                    x_max = np.nanmax(xs)
                    y_min = np.nanmin(ys)
                    y_max = np.nanmax(ys)
                    _plot_index_line(
                        axis, x_line, np.full_like(x_line, y_min, dtype=float), color
                    )
                    _plot_index_line(
                        axis, x_line, np.full_like(x_line, y_max, dtype=float), color
                    )
                    _plot_index_line(
                        axis, np.full_like(y_line, x_min, dtype=float), y_line, color
                    )
                    _plot_index_line(
                        axis, np.full_like(y_line, x_max, dtype=float), y_line, color
                    )
            elif isinstance(obj, np.ndarray):
                xs, ys = obj.nonzero()
                if len(xs) > 0:
                    _scatter_index_points(axis, xs, ys, color)

    if if_gif:
        fig_gif = plt.figure(
            figsize=(fig_w, max(3.0, top_margin + bottom_margin + panel_h))
        )
        gif_ax = fig_gif.add_axes(
            [
                left_margin / fig_w,
                bottom_margin / fig_gif.get_figheight(),
                panel_w / fig_w,
                panel_h / fig_gif.get_figheight(),
            ]
        )
        gif_extent = [-x_width / 2, x_width / 2, -y_height / 2, y_height / 2]
        _imshow_phys(gif_ax, np.real(field), gif_extent, cmap="RdBu_r")
        _imshow_phys(
            gif_ax,
            eps.astype(np.float64),
            gif_extent,
            cmap="Greys",
            vmin=eps_vmin,
            vmax=eps_vmax,
            alpha=0.18,
        )
        _draw_npml(gif_ax, field.shape, NPML, x_width, y_height)
        _format_axis(gif_ax, r"$x$ width ($\mu m$)", r"$y$ height ($\mu m$)")

    if if_gif:
        pass

    size = eps.shape
    ## center crop of eps of size of new_size
    ## find center pixel index based on zoom_eps_center
    patch_size = (x_width / zoom_eps_factor, y_height / zoom_eps_factor)
    if zoom_eps_factor > 1:
        ## move center to avoid exceeding the boundary
        zoom_eps_center = np.clip(
            zoom_eps_center,
            (-(x_width - patch_size[0]) / 2, -(y_height - patch_size[1]) / 2),
            ((x_width - patch_size[0]) / 2, (y_height - patch_size[1]) / 2),
        )
        zoom_eps_center_ind = np.round(
            loc2ind(zoom_eps_center, (x_width, y_height), size) * zoom_eps_factor
        ).astype(np.int32)

        eps = zoom(eps, zoom_eps_factor)
        eps = eps[
            zoom_eps_center_ind[0]
            - size[0] // 2 : zoom_eps_center_ind[0]
            + size[0] // 2,
            zoom_eps_center_ind[1]
            - size[1] // 2 : zoom_eps_center_ind[1]
            + size[1] // 2,
        ]
        if eps_grad_arr is not None:
            eps_grad_arr = zoom(eps_grad_arr, zoom_eps_factor)
            eps_grad_arr = eps_grad_arr[
                zoom_eps_center_ind[0]
                - size[0] // 2 : zoom_eps_center_ind[0]
                + size[0] // 2,
                zoom_eps_center_ind[1]
                - size[1] // 2 : zoom_eps_center_ind[1]
                + size[1] // 2,
            ]
        if thermal_map_arr is not None and thermal_map_arr.shape == eps.shape:
            thermal_map_arr = zoom(thermal_map_arr, zoom_eps_factor)
            thermal_map_arr = thermal_map_arr[
                zoom_eps_center_ind[0]
                - size[0] // 2 : zoom_eps_center_ind[0]
                + size[0] // 2,
                zoom_eps_center_ind[1]
                - size[1] // 2 : zoom_eps_center_ind[1]
                + size[1] // 2,
            ]
        if base_eps_arr is not None:
            base_eps_arr = zoom(base_eps_arr, zoom_eps_factor)
            base_eps_arr = base_eps_arr[
                zoom_eps_center_ind[0]
                - size[0] // 2 : zoom_eps_center_ind[0]
                + size[0] // 2,
                zoom_eps_center_ind[1]
                - size[1] // 2 : zoom_eps_center_ind[1]
                + size[1] // 2,
            ]
        if heat_source_map_arr is not None and heat_source_map_arr.shape == eps.shape:
            heat_source_map_arr = zoom(heat_source_map_arr, zoom_eps_factor)
            heat_source_map_arr = heat_source_map_arr[
                zoom_eps_center_ind[0]
                - size[0] // 2 : zoom_eps_center_ind[0]
                + size[0] // 2,
                zoom_eps_center_ind[1]
                - size[1] // 2 : zoom_eps_center_ind[1]
                + size[1] // 2,
            ]
    else:
        zoom_eps_center = (0, 0)  # force to be origin if not zoomed

    if show_delta_eps:
        delta_eps_arr = np.real(eps) - np.real(base_eps_arr)
        delta_min, delta_max = _finite_minmax(delta_eps_arr)
        delta_lim = max(abs(delta_min), abs(delta_max))
        if np.isclose(delta_lim, 0.0):
            delta_lim = 1.0

    field_extent = [-x_width / 2, x_width / 2, -y_height / 2, y_height / 2]
    eps_extent = [
        zoom_eps_center[0] - patch_size[0] / 2,
        zoom_eps_center[0] + patch_size[0] / 2,
        zoom_eps_center[1] - patch_size[1] / 2,
        zoom_eps_center[1] + patch_size[1] / 2,
    ]

    row_specs = []
    for stat in field_stat:
        if stat == "abs":
            row_specs.append(
                dict(
                    title=r"$|" + component + r"|$",
                    kind="abs",
                    cmap="magma",
                    vmin=field_abs_vmin,
                    vmax=field_abs_vmax,
                    extent=field_extent,
                    overlay_eps=True,
                    draw_npml=True,
                    draw_monitors=True,
                )
            )
        elif stat == "real":
            row_specs.append(
                dict(
                    title=r"$\mathrm{Re}(" + component + r")$",
                    kind="real",
                    cmap="RdBu_r",
                    vmin=-field_real_lim,
                    vmax=field_real_lim,
                    extent=field_extent,
                    overlay_eps=True,
                    draw_npml=True,
                    draw_monitors=True,
                )
            )
        elif stat == "intensity":
            row_specs.append(
                dict(
                    title=r"$|" + component + r"|^2$",
                    kind="intensity",
                    cmap="magma",
                    vmin=field_intensity_vmin,
                    vmax=field_intensity_vmax,
                    extent=field_extent,
                    overlay_eps=True,
                    draw_npml=True,
                    draw_monitors=True,
                )
            )
        else:
            raise ValueError(f"Unsupported field_stat entry: {stat}")

    row_specs.append(
        dict(
            title=r"$\epsilon$",
            kind="eps",
            cmap="Greys",
            vmin=eps_vmin,
            vmax=eps_vmax,
            extent=eps_extent,
            overlay_eps=False,
            draw_npml=False,
            draw_monitors=False,
        )
    )
    if show_delta_eps:
        row_specs.append(
            dict(
                title=r"$\Delta \epsilon$",
                kind="delta_eps",
                cmap="RdBu_r",
                vmin=-delta_lim,
                vmax=delta_lim,
                extent=eps_extent,
                overlay_eps=False,
                draw_npml=False,
                draw_monitors=False,
            )
        )
    if thermal_map_arr is not None:
        thermal_title = (thermal_map_name or "thermal_map").replace("_", " ")
        use_diverging = (
            "coeff" in thermal_title.lower()
            or "grad" in thermal_title.lower()
            or thermal_vmin < 0.0
        )
        row_specs.append(
            dict(
                title=thermal_title,
                kind="thermal",
                cmap="RdBu_r" if use_diverging else "coolwarm",
                vmin=(-thermal_lim if use_diverging else thermal_vmin),
                vmax=(thermal_lim if use_diverging else thermal_vmax),
                extent=(
                    eps_extent if thermal_map_arr.shape == eps.shape else field_extent
                ),
                overlay_eps=thermal_map_arr.shape == eps.shape,
                draw_npml=False,
                draw_monitors=False,
            )
        )
    if eps_grad_arr is not None:
        row_specs.append(
            dict(
                title=r"$dL/d\epsilon$",
                kind="eps_grad",
                cmap="RdBu_r",
                vmin=-grad_lim,
                vmax=grad_lim,
                extent=eps_extent,
                overlay_eps=True,
                draw_npml=False,
                draw_monitors=False,
            )
        )
    if param_grad_arr is not None and polygon_vertices is None:
        row_specs.append(
            dict(
                title=r"$dL/d$" + param_name,
                kind="param_grad",
                cmap="RdBu_r",
                vmin=-param_lim,
                vmax=param_lim,
                extent=[
                    -param_x_width / 2,
                    param_x_width / 2,
                    -param_y_height / 2,
                    param_y_height / 2,
                ],
                overlay_eps=False,
                draw_npml=False,
                draw_monitors=False,
            )
        )
    if polygon_vertices is not None:
        polygon_grad_lim = np.max(np.abs(polygon_grad))
        if not np.isfinite(polygon_grad_lim) or np.isclose(polygon_grad_lim, 0.0):
            polygon_grad_lim = 1.0
        row_specs.append(
            dict(
                title="polygon_vertex_grad",
                kind="polygon_vertex_grad",
                cmap="RdBu_r",
                vmin=-polygon_grad_lim,
                vmax=polygon_grad_lim,
                extent=eps_extent,
                overlay_eps=False,
                draw_npml=False,
                draw_monitors=False,
            )
        )

    axes = []
    for i, spec in enumerate(row_specs):
        if spec["kind"] == "polygon_vertex_grad":
            polygon_axes, polygon_caxes = _add_polygon_axes(i)
            for axis, cax, component, panel_title in zip(
                polygon_axes,
                polygon_caxes,
                (0, 1),
                (r"$dL/dx$", r"$dL/dy$"),
            ):
                values = polygon_grad[:, component]
                scatter = axis.scatter(
                    polygon_vertices[:, 0],
                    polygon_vertices[:, 1],
                    c=values,
                    cmap=spec["cmap"],
                    vmin=spec["vmin"],
                    vmax=spec["vmax"],
                    s=2,
                    edgecolors="black",
                    linewidths=0.1,
                    zorder=3,
                )
                axis.plot(
                    polygon_vertices[:, 0],
                    polygon_vertices[:, 1],
                    color="0.45",
                    linewidth=0.7,
                    alpha=0.75,
                    zorder=1,
                )
                axis.set_title(panel_title, fontsize=panel_title_fs)
                axis.set_xlim(spec["extent"][0], spec["extent"][1])
                axis.set_ylim(spec["extent"][2], spec["extent"][3])
                axis.set_aspect("equal", adjustable="box")
                _format_axis(axis, "x (um)", "y (um)")
                _add_colorbar(scatter, cax)
            axes.extend(polygon_axes)
            continue
        ax_i, cax_i = _add_axes(i)
        if spec["kind"] == "abs":
            img_data = np.abs(field)
        elif spec["kind"] == "real":
            img_data = np.real(field)
        elif spec["kind"] == "intensity":
            img_data = np.abs(field) ** 2
        elif spec["kind"] == "eps":
            img_data = np.real(eps).astype(np.float64)
        elif spec["kind"] == "delta_eps":
            img_data = delta_eps_arr.astype(np.float64)
        elif spec["kind"] == "thermal":
            img_data = np.real(thermal_map_arr).astype(np.float64)
        elif spec["kind"] == "param_grad":
            img_data = np.real(param_grad_arr).astype(np.float64)
        else:
            img_data = np.real(eps_grad_arr).astype(np.float64)

        im = _imshow_phys(
            ax_i,
            img_data,
            spec["extent"],
            cmap=spec["cmap"],
            vmin=spec["vmin"],
            vmax=spec["vmax"],
        )
        if spec["overlay_eps"]:
            overlay_extent = field_extent
            if spec["kind"] in {"eps_grad", "thermal", "delta_eps"}:
                overlay_extent = eps_extent
            _imshow_phys(
                ax_i,
                np.real(eps).astype(np.float64),
                overlay_extent,
                cmap="Greys",
                vmin=eps_vmin,
                vmax=eps_vmax,
                alpha=0.18,
            )
        if spec["draw_npml"]:
            _draw_npml(ax_i, field.shape, NPML, x_width, y_height)
        if spec["draw_monitors"]:
            _draw_monitors(ax_i)
        _draw_heat_source_frames(
            ax_i,
            heat_source_map_arr,
            spec["extent"],
        )

        _format_axis(ax_i, r"$x$ width ($\mu m$)", r"$y$ height ($\mu m$)")
        ax_i.set_title(spec["title"], fontsize=panel_title_fs, pad=4)
        _add_colorbar(im, cax_i)
        axes.append(ax_i)

    for edge_row, edge in enumerate(edgebox_edges):
        edge_axes = _add_edgebox_axes(len(row_specs) + polygon_row_span - 1 + edge_row)
        knot_index = np.arange(len(edge["raw"]))
        raw = np.asarray(edge["raw"])
        physical = np.asarray(edge["physical"])
        gradient = edge.get("grad")
        gradient = None if gradient is None else np.asarray(gradient)
        edge_title = edge.get("side", f"edge {edge_row}")

        edge_axes[0].plot(knot_index, raw, "o-", color="tab:blue")
        edge_axes[0].set_title(f"{edge_title}: raw knots", fontsize=panel_title_fs)
        edge_axes[0].set_ylabel("raw control", fontsize=label_fs)
        edge_axes[1].plot(knot_index, physical, "o-", color="tab:green")
        edge_axes[1].set_title(
            f"{edge_title}: physical offset", fontsize=panel_title_fs
        )
        edge_axes[1].set_ylabel("offset [um]", fontsize=label_fs)
        if gradient is None:
            edge_axes[2].text(
                0.5,
                0.5,
                "gradient unavailable",
                ha="center",
                va="center",
                transform=edge_axes[2].transAxes,
            )
        else:
            edge_axes[2].bar(knot_index, gradient, color="tab:red")
        edge_axes[2].set_title(f"{edge_title}: knot gradient", fontsize=panel_title_fs)
        edge_axes[2].set_ylabel("d objective / d raw", fontsize=label_fs)
        for axis in edge_axes:
            axis.set_xlabel("knot index", fontsize=label_fs)
            axis.tick_params(axis="both", labelsize=tick_fs)
            axis.grid(True, alpha=0.25)

    if title is not None:
        fig.suptitle(title, fontsize=suptitle_fs, y=1.0 - 0.10 / fig_h)

    area = field.shape[0] * field.shape[1]
    if area > 2000**2:
        dpi = 300
    else:
        dpi = 600
    if filepath is not None:
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    if if_gif:
        gif_filepath = filepath[:-4] + "_gif" + filepath[-4:]
        fig_gif.savefig(gif_filepath, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig_gif)


def plot_eps_field_3d(
    field: dict,  # all components {"Ex":..., "Ey":..., "Ez":..., "Hx":..., "Hy":..., "Hz":...}
    component: str,  # which component to plot, e.g. "Ez"
    eps,
    base_eps=None,
    show_delta_eps: bool | None = None,
    thermal_map=None,
    heat_source_map=None,
    thermal_map_name: str | None = None,
    eps_grad=None,
    param_grad=None,
    param_x_width=None,
    param_y_height=None,
    param_name: str = "param",
    edgebox_grad=None,
    monitors=[],
    filepath=None,
    center=None,
    zoom_eps_factor=1,
    zoom_eps_center=(0, 0, 0),
    x_width=1,
    y_height=1,
    z_depth=1,
    NPML=[0, 0, 0],
    field_stat: str = "abs_real",  # kept for compatibility
    title: str = None,
    x_shift_coord: int = 0,
    x_shift_idx: int = 0,
    if_gif: bool = False,
    precomputed_planes: list[dict] | None = None,
    grid_shape: tuple[int, int, int] | None = None,
):
    """
    Plot 3D field and epsilon.

    field: dict with components, e.g.
        {"Ex":..., "Ey":..., "Ez":..., "Hx":..., "Hy":..., "Hz":...}

    component:
        Which component to plot, e.g. "Ez".

    eps:
        [nx, ny, nz]

    monitors:
        Supports:
            (Slice3D_like_object, color)
            (3D_numpy_mask, color)

        Slice3D_like_object should have:
            .x, .y, .z

        Coordinates are assumed to be grid-index coordinates, consistent
        with your 2D monitor plotting code.

    center:
        Physical coordinate `(x, y, z)` in the device coordinate system for
        the three orthogonal slice planes. If omitted, the plot keeps the
        existing middle-plane behavior.

    precomputed_planes:
        Optional CPU/Torch plane dictionaries ordered as `xy`, `xz`, and `yz`.
        Each dictionary contains `field`, `total_field_abs`, `eps`, and
        `fixed_idx`, plus optional plot-map planes. When provided, full 3-D
        `field` and `eps` inputs are not required.

    grid_shape:
        Full export-grid `(nx, ny, nz)` shape used for monitor coordinates when
        `precomputed_planes` is provided.

    Layout:
        row 0: total vector field magnitude, e.g. |E| or |H|
        row 1: real(component), e.g. Re(Ez)
        row 2: epsilon

        col 0: x-y plane at z = middle
        col 1: x-z plane at y = middle
        col 2: y-z plane at x = middle

    IMPORTANT:
        A single global physical scale is used for all panels:
            panel_width  = scale * physical_x_extent
            panel_height = scale * physical_y_extent

        Therefore, if two panels both use x_width on the x-axis, they will
        have the same displayed width in the final figure.
    """

    import os

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------
    def _as_numpy(value):
        if value is None:
            return None
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    using_precomputed_planes = precomputed_planes is not None
    if using_precomputed_planes:
        if grid_shape is None or len(grid_shape) != 3:
            raise ValueError(
                "grid_shape with three entries is required for precomputed planes."
            )
        if len(precomputed_planes) != 3:
            raise ValueError(
                "precomputed_planes must contain the xy, xz, and yz planes."
            )
        nx, ny, nz = (int(size) for size in grid_shape)
        field = None
        total_field_abs = None
        eps = None
        base_eps_arr = None
        eps_grad_arr = None
        thermal_map_arr = None
        heat_source_map_arr = None
    else:
        fx, fy, fz = (
            field[f"{component[0]}x"].data,
            field[f"{component[0]}y"].data,
            field[f"{component[0]}z"].data,
        )
        field = field[component].data
        total_field_abs = (fx.abs() ** 2 + fy.abs() ** 2 + fz.abs() ** 2).sqrt()
        field = _as_numpy(field)
        total_field_abs = _as_numpy(total_field_abs)

        eps_grad_arr = None
        if isinstance(eps_grad, bool):
            if (
                eps_grad
                and torch is not None
                and isinstance(eps, torch.Tensor)
                and eps.grad is not None
            ):
                eps_grad_arr = eps.grad.detach().cpu().numpy()
        elif eps_grad is not None:
            eps_grad_arr = _as_numpy(eps_grad)

        eps = _as_numpy(eps)
        base_eps_arr = _as_numpy(base_eps)
        thermal_map_arr = _as_numpy(thermal_map)
        heat_source_map_arr = _as_numpy(heat_source_map)

        if field.ndim != 3:
            raise ValueError(
                f"Expected field to be 3D [nx, ny, nz], got shape {field.shape}"
            )
        if eps.ndim != 3:
            raise ValueError(
                f"Expected eps to be 3D [nx, ny, nz], got shape {eps.shape}"
            )
        if base_eps_arr is not None and base_eps_arr.shape != eps.shape:
            raise ValueError(
                "base_eps must have the same shape as eps, got "
                f"{base_eps_arr.shape} and {eps.shape}"
            )
        if field.shape != eps.shape:
            raise ValueError(
                f"field and eps must have the same shape, got {field.shape} and {eps.shape}"
            )
        if eps_grad_arr is not None and eps_grad_arr.shape != eps.shape:
            raise ValueError(
                "eps_grad must have the same shape as eps, got "
                f"{eps_grad_arr.shape} and {eps.shape}"
            )
        if total_field_abs.shape != field.shape:
            raise ValueError(
                f"total_field_abs and field must have the same shape, "
                f"got {total_field_abs.shape} and {field.shape}"
            )
        nx, ny, nz = field.shape
    if len(NPML) != 3:
        raise ValueError("For 3D plotting, NPML must be [NPML_x, NPML_y, NPML_z]")

    def _positive_float(name, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {value}")
        return value

    x_width = _positive_float("x_width", x_width)
    y_height = _positive_float("y_height", y_height)
    z_depth = _positive_float("z_depth", z_depth)

    if filepath is not None:
        outdir = os.path.dirname(filepath)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

    NPML_x, NPML_y, NPML_z = [int(v) for v in NPML]

    dim_sizes = {
        0: nx,
        1: ny,
        2: nz,
    }

    dim_phys = {
        0: x_width,
        1: y_height,
        2: z_depth,
    }

    def _slice_coord_um(idx, dim):
        n = dim_sizes[dim]
        L = dim_phys[dim]
        if n <= 1:
            return 0.0
        return (float(idx) / (n - 1) - 0.5) * L

    if using_precomputed_planes:
        x_mid = int(precomputed_planes[2]["fixed_idx"])
        y_mid = int(precomputed_planes[1]["fixed_idx"])
        z_mid = int(precomputed_planes[0]["fixed_idx"])
    elif center is None:
        x_mid = nx // 2
        y_mid = ny // 2
        z_mid = nz // 2

        if x_shift_idx != 0:
            x_mid = int(np.clip(x_mid + x_shift_idx, 0, nx - 1))
    else:
        center = np.asarray(center, dtype=float)
        if center.shape != (3,):
            raise ValueError(
                f"center must be a length-3 coordinate, got shape {center.shape}"
            )
        x_mid, y_mid, z_mid = loc2ind(
            center,
            (x_width, y_height, z_depth),
            (nx, ny, nz),
        ).tolist()

    plane_metadata = [
        {
            "name": f"x-y, z={_slice_coord_um(z_mid, 2):.3f}" + r"$\mu m$",
            "xlabel": r"$x$ width ($\mu m$)",
            "ylabel": r"$y$ height ($\mu m$)",
            "x_phys": x_width,
            "y_phys": y_height,
            "npml": [NPML_x, NPML_y],
            "h_dim": 0,
            "v_dim": 1,
            "fixed_dim": 2,
            "fixed_idx": z_mid,
        },
        {
            "name": f"x-z, y={_slice_coord_um(y_mid, 1):.3f}" + r"$\mu m$",
            "xlabel": r"$x$ width ($\mu m$)",
            "ylabel": r"$z$ depth ($\mu m$)",
            "x_phys": x_width,
            "y_phys": z_depth,
            "npml": [NPML_x, NPML_z],
            "h_dim": 0,
            "v_dim": 2,
            "fixed_dim": 1,
            "fixed_idx": y_mid,
        },
        {
            "name": f"y-z, x={_slice_coord_um(x_mid, 0):.3f}" + r"$\mu m$",
            "xlabel": r"$y$ height ($\mu m$)",
            "ylabel": r"$z$ depth ($\mu m$)",
            "x_phys": y_height,
            "y_phys": z_depth,
            "npml": [NPML_y, NPML_z],
            "h_dim": 1,
            "v_dim": 2,
            "fixed_dim": 0,
            "fixed_idx": x_mid,
        },
    ]

    if using_precomputed_planes:
        planes = []
        required_keys = ("field", "total_field_abs", "eps", "fixed_idx")
        for values, metadata in zip(precomputed_planes, plane_metadata):
            missing = [key for key in required_keys if key not in values]
            if missing:
                raise ValueError(f"Precomputed plot plane is missing keys: {missing}")
            plane = dict(metadata)
            plane.update(
                {
                    key: _as_numpy(values.get(key))
                    for key in (
                        "field",
                        "total_field_abs",
                        "eps",
                        "base_eps",
                        "eps_grad",
                        "thermal",
                        "heat_source",
                    )
                }
            )
            plane["fixed_idx"] = int(values["fixed_idx"])
            expected_shape = (dim_sizes[plane["h_dim"]], dim_sizes[plane["v_dim"]])
            for key in ("field", "total_field_abs", "eps"):
                if plane[key].shape != expected_shape:
                    raise ValueError(
                        f"Precomputed {key} plane has shape {plane[key].shape}; "
                        f"expected {expected_shape}."
                    )
            planes.append(plane)
        field_stat_values = np.concatenate(
            [np.asarray(plane["field"]).reshape(-1) for plane in planes]
        )
        total_field_stat_values = np.concatenate(
            [np.asarray(plane["total_field_abs"]).reshape(-1) for plane in planes]
        )
        eps_stat_values = np.concatenate(
            [np.asarray(plane["eps"]).reshape(-1) for plane in planes]
        )
        base_eps_values = [plane["base_eps"] for plane in planes]
        base_eps_stat_values = (
            None
            if any(value is None for value in base_eps_values)
            else np.concatenate(
                [np.asarray(value).reshape(-1) for value in base_eps_values]
            )
        )
        eps_grad_values = [plane["eps_grad"] for plane in planes]
        eps_grad_arr = (
            None
            if all(value is None for value in eps_grad_values)
            else np.concatenate(
                [
                    np.asarray(value).reshape(-1)
                    for value in eps_grad_values
                    if value is not None
                ]
            )
        )
        thermal_values = [plane["thermal"] for plane in planes]
        thermal_map_arr = (
            None
            if all(value is None for value in thermal_values)
            else np.concatenate(
                [
                    np.asarray(value).reshape(-1)
                    for value in thermal_values
                    if value is not None
                ]
            )
        )
        heat_source_map_arr = (
            None if all(plane["heat_source"] is None for plane in planes) else True
        )
    else:
        planes = plane_metadata
        dense_slices = (
            (slice(None), slice(None), z_mid),
            (slice(None), y_mid, slice(None)),
            (x_mid, slice(None), slice(None)),
        )
        for plane, plane_slice in zip(planes, dense_slices):
            plane.update(
                {
                    "field": field[plane_slice],
                    "total_field_abs": total_field_abs[plane_slice],
                    "eps": eps[plane_slice],
                    "base_eps": (
                        None if base_eps_arr is None else base_eps_arr[plane_slice]
                    ),
                    "eps_grad": (
                        None if eps_grad_arr is None else eps_grad_arr[plane_slice]
                    ),
                    "thermal": (
                        None
                        if thermal_map_arr is None
                        else thermal_map_arr[plane_slice]
                    ),
                    "heat_source": (
                        None
                        if heat_source_map_arr is None
                        else heat_source_map_arr[plane_slice]
                    ),
                }
            )
        field_stat_values = field
        total_field_stat_values = total_field_abs
        eps_stat_values = eps
        base_eps_stat_values = base_eps_arr

    # ------------------------------------------------------------------
    # Global color limits
    # ------------------------------------------------------------------
    def _finite_minmax(a):
        a = np.asarray(a)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return 0.0, 1.0
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        # if np.isclose(vmin, vmax):
        #     pad = 1.0 if np.isclose(vmin, 0.0) else 0.05 * abs(vmin)
        #     vmin -= pad
        #     vmax += pad
        return vmin, vmax

    # Since row 0 plots total_field_abs, its color scale should come from total_field_abs.
    abs_vmin, abs_vmax = _finite_minmax(total_field_stat_values)

    real_min, real_max = _finite_minmax(np.real(field_stat_values))
    real_lim = max(abs(real_min), abs(real_max))
    if np.isclose(real_lim, 0.0):
        real_lim = 1.0

    eps_vmin, eps_vmax = _finite_minmax(np.real(eps_stat_values))
    if show_delta_eps is None:
        show_delta_eps = base_eps_stat_values is not None and not np.allclose(
            np.real(eps_stat_values),
            np.real(base_eps_stat_values),
            atol=1e-12,
            rtol=1e-9,
        )
    else:
        show_delta_eps = bool(show_delta_eps) and base_eps_stat_values is not None
    if show_delta_eps:
        delta_eps_stat_values = np.real(eps_stat_values) - np.real(base_eps_stat_values)
        delta_min, delta_max = _finite_minmax(delta_eps_stat_values)
        delta_lim = max(abs(delta_min), abs(delta_max))
        if np.isclose(delta_lim, 0.0):
            delta_lim = 1.0
        for plane in planes:
            plane["delta_eps"] = np.real(plane["eps"]) - np.real(plane["base_eps"])
    else:
        for plane in planes:
            plane["delta_eps"] = None
    if thermal_map_arr is not None:
        thermal_vmin, thermal_vmax = _finite_minmax(np.real(thermal_map_arr))
        thermal_lim = max(abs(thermal_vmin), abs(thermal_vmax))
        if np.isclose(thermal_lim, 0.0):
            thermal_lim = 1.0
    # eps_vmin = 0.0
    if eps_grad_arr is not None:
        grad_lim = _symmetric_percentile_limit(np.real(eps_grad_arr))

    param_grad_arr = None
    if param_grad is not None:
        param_grad_arr = (
            param_grad.detach().cpu().numpy()
            if torch is not None and isinstance(param_grad, torch.Tensor)
            else np.asarray(param_grad)
        )
        if param_grad_arr.ndim != 2:
            raise ValueError(f"param_grad must be 2D, got shape {param_grad_arr.shape}")
        param_x_width = _positive_float(
            "param_x_width", x_width if param_x_width is None else param_x_width
        )
        param_y_height = _positive_float(
            "param_y_height", y_height if param_y_height is None else param_y_height
        )
        param_min, param_max = _finite_minmax(np.real(param_grad_arr))
        param_lim = max(abs(param_min), abs(param_max))
        if np.isclose(param_lim, 0.0):
            param_lim = 1.0

    edgebox_edges = (
        edgebox_grad.get("edge_knots", []) if isinstance(edgebox_grad, dict) else []
    )

    # ------------------------------------------------------------------
    # Figure geometry with ONE GLOBAL SCALE
    # ------------------------------------------------------------------
    panel_x_phys = np.array([p["x_phys"] for p in planes], dtype=float)
    panel_y_phys = np.array([p["y_phys"] for p in planes], dtype=float)

    # Use conservative margins because axes, double-line titles, and colorbars
    # are positioned manually in figure coordinates.
    left_margin = 1.00
    right_margin = 0.45
    bottom_margin = 1.00
    top_margin = 0.95 + (0.45 if title is not None else 0.0)

    col_gap = 1.0
    row_gap = 0.70
    cbar_gap = 0.10
    cbar_width = 0.14

    target_fig_w = 16.0
    target_fig_h = 12.0

    total_phys_w = np.sum(panel_x_phys)
    max_phys_h = np.max(panel_y_phys)

    fixed_w = left_margin + right_margin + 2 * col_gap + 3 * (cbar_gap + cbar_width)
    n_plot_rows = (
        3
        + int(show_delta_eps)
        + int(thermal_map_arr is not None)
        + int(eps_grad_arr is not None)
        + int(param_grad_arr is not None)
        + len(edgebox_edges)
    )
    fixed_h = bottom_margin + top_margin + (n_plot_rows - 1) * row_gap

    avail_w = target_fig_w - fixed_w
    scale_w = avail_w / total_phys_w if total_phys_w > 0 else 1.0

    avail_h = target_fig_h - fixed_h
    scale_h = avail_h / (n_plot_rows * max_phys_h) if max_phys_h > 0 else 1.0

    scale = min(scale_w, scale_h)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    panel_ws = scale * panel_x_phys
    panel_hs = scale * panel_y_phys
    # Keep a minimum row band height so smaller devices still leave room for
    # axis labels, a two-line title, and the extra gradient row.
    min_row_band_h = 1.85 if eps_grad_arr is None else 2.10
    row_band_h = max(float(np.max(panel_hs)), min_row_band_h)

    fig_w = fixed_w + np.sum(panel_ws)
    fig_h = fixed_h + n_plot_rows * row_band_h

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    # ------------------------------------------------------------------
    # Font sizes based on actual rendered scale
    # ------------------------------------------------------------------
    font_scale = np.clip(row_band_h / 2.2, 0.80, 1.20)

    panel_title_fs = 15 * font_scale
    label_fs = 13 * font_scale
    tick_fs = 11 * font_scale
    cbar_tick_fs = 9 * font_scale
    suptitle_fs = 18 * font_scale

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _add_axes(row, col):
        """
        Place axes manually.
        Shorter panels are vertically centered inside their row band.
        """
        x0 = left_margin
        for jj in range(col):
            x0 += panel_ws[jj] + cbar_gap + cbar_width + col_gap

        y_band0 = fig_h - top_margin - (row + 1) * row_band_h - row * row_gap
        y0 = y_band0 + 0.5 * (row_band_h - panel_hs[col])

        ax_rect = [
            x0 / fig_w,
            y0 / fig_h,
            panel_ws[col] / fig_w,
            panel_hs[col] / fig_h,
        ]

        cax_rect = [
            (x0 + panel_ws[col] + cbar_gap) / fig_w,
            y0 / fig_h,
            cbar_width / fig_w,
            panel_hs[col] / fig_h,
        ]

        ax = fig.add_axes(ax_rect)
        cax = fig.add_axes(cax_rect)
        return ax, cax

    def _add_full_row_axes(row, x_phys, y_phys):
        total_content_w = fig_w - left_margin - right_margin
        cbar_total_w = cbar_gap + cbar_width
        ax_avail_w = total_content_w - cbar_total_w
        desired_w = row_band_h * (x_phys / max(y_phys, 1e-12))
        ax_w = min(ax_avail_w, desired_w)

        y_band0 = fig_h - top_margin - (row + 1) * row_band_h - row * row_gap
        y0 = y_band0 + 0.5 * (row_band_h - row_band_h)
        x0 = left_margin + 0.5 * (ax_avail_w - ax_w)

        ax_rect = [x0 / fig_w, y0 / fig_h, ax_w / fig_w, row_band_h / fig_h]
        cax_rect = [
            (left_margin + ax_avail_w + cbar_gap) / fig_w,
            y0 / fig_h,
            cbar_width / fig_w,
            row_band_h / fig_h,
        ]
        return fig.add_axes(ax_rect), fig.add_axes(cax_rect)

    def _add_edgebox_axes(row):
        total_content_w = fig_w - left_margin - right_margin
        axis_gap = 0.55
        axis_width = (total_content_w - 2 * axis_gap) / 3
        y_band0 = fig_h - top_margin - (row + 1) * row_band_h - row * row_gap
        axes = []
        for col in range(3):
            x0 = left_margin + col * (axis_width + axis_gap)
            axes.append(
                fig.add_axes(
                    [
                        x0 / fig_w,
                        y_band0 / fig_h,
                        axis_width / fig_w,
                        row_band_h / fig_h,
                    ]
                )
            )
        return axes

    def _imshow_phys(axis, data_2d, x_phys, y_phys, **kwargs):
        """
        data_2d is assumed shape [n_horizontal, n_vertical].
        imshow expects [n_vertical, n_horizontal], so transpose.
        """
        extent = [-x_phys / 2, x_phys / 2, -y_phys / 2, y_phys / 2]

        im = axis.imshow(
            np.asarray(data_2d).T,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
            **kwargs,
        )

        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect("equal", adjustable="box")

        return im

    def _draw_npml(axis, shape_2d, npml_2d, x_phys, y_phys):
        n0, n1 = shape_2d
        pml0, pml1 = [max(0, int(v)) for v in npml_2d]

        pml0 = min(pml0, n0 // 2)
        pml1 = min(pml1, n1 // 2)

        if pml0 == 0 and pml1 == 0:
            return

        x_min, x_max = -x_phys / 2, x_phys / 2
        y_min, y_max = -y_phys / 2, y_phys / 2

        dx = x_phys / max(n0, 1)
        dy = y_phys / max(n1, 1)

        pml_x = pml0 * dx
        pml_y = pml1 * dy

        rect_kw = dict(facecolor="gray", alpha=0.40, edgecolor="none", zorder=5)

        if pml_x > 0:
            axis.add_patch(patches.Rectangle((x_min, y_min), pml_x, y_phys, **rect_kw))
            axis.add_patch(
                patches.Rectangle((x_max - pml_x, y_min), pml_x, y_phys, **rect_kw)
            )

        if pml_y > 0:
            inner_x0 = x_min + pml_x
            inner_w = max(x_phys - 2 * pml_x, 0.0)
            axis.add_patch(
                patches.Rectangle((inner_x0, y_min), inner_w, pml_y, **rect_kw)
            )
            axis.add_patch(
                patches.Rectangle((inner_x0, y_max - pml_y), inner_w, pml_y, **rect_kw)
            )

    def _draw_heat_source_frames(axis, source_2d, x_phys, y_phys):
        if source_2d is None:
            return
        source_mask = np.asarray(np.abs(source_2d) > 0, dtype=bool)
        if source_mask.ndim != 2 or not np.any(source_mask):
            return

        component_labels, num_components = label(source_mask)
        n0, n1 = source_mask.shape
        x_min, y_min = -x_phys / 2, -y_phys / 2
        dx = x_phys / max(n0, 1)
        dy = y_phys / max(n1, 1)
        frame_kw = dict(
            fill=False,
            edgecolor="purple",
            linestyle=":",
            linewidth=max(0.5, 1 * font_scale),
            alpha=0.40,
            zorder=22,
        )

        for component_idx in range(1, num_components + 1):
            hs, vs = np.nonzero(component_labels == component_idx)
            if hs.size == 0:
                continue
            left = x_min + float(hs.min()) * dx
            bottom = y_min + float(vs.min()) * dy
            width = float(hs.max() - hs.min() + 1) * dx
            height = float(vs.max() - vs.min() + 1) * dy
            axis.add_patch(patches.Rectangle((left, bottom), width, height, **frame_kw))

    def _format_axis(axis, xlabel, ylabel, axis_w_in, axis_h_in):
        axis.set_xlabel(xlabel, fontsize=label_fs, labelpad=2)
        axis.set_ylabel(ylabel, fontsize=label_fs, labelpad=2)

        nbins_x = 3 if axis_w_in < 1.5 else 4 if axis_w_in < 2.3 else 5
        nbins_y = 3 if axis_h_in < 1.2 else 4 if axis_h_in < 2.0 else 5

        axis.xaxis.set_major_locator(mticker.MaxNLocator(nbins=nbins_x))
        axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nbins_y))
        axis.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        axis.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

        axis.tick_params(
            axis="both",
            labelsize=tick_fs,
            pad=1.5,
            length=2.5,
            width=0.6,
        )

    def _add_colorbar(im, cax):
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=cbar_tick_fs, pad=1.0, length=2.5, width=0.6)
        cb.locator = mticker.MaxNLocator(nbins=4)
        cb.formatter = mticker.ScalarFormatter(useMathText=True)
        cb.formatter.set_powerlimits((-2, 3))
        cb.update_ticks()
        cb.ax.yaxis.get_offset_text().set_size(max(cbar_tick_fs * 0.9, 7.0))
        return cb

    # ------------------------------------------------------------------
    # Monitor helper functions
    # ------------------------------------------------------------------
    def _idx_to_phys(values, dim):
        """
        Convert grid-index coordinates to centered physical coordinates.

        index 0     -> -L / 2
        index n - 1 ->  L / 2
        """
        values = np.asarray(values, dtype=float)
        n = dim_sizes[dim]
        L = dim_phys[dim]

        if n <= 1:
            out = np.zeros_like(values, dtype=float)
        else:
            out = (values / (n - 1) - 0.5) * L

        out[np.isnan(values)] = np.nan
        return out

    def _coord_values(coord):
        """
        Return sorted unique finite coordinate values from scalar or array.
        """
        arr = np.asarray(coord, dtype=float)
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return np.array([], dtype=float)

        vals = np.unique(arr.astype(float))
        vals.sort()
        return vals

    def _coord_is_scalar(coord):
        vals = _coord_values(coord)

        if vals.size <= 1:
            return True

        return bool(np.allclose(vals, vals[0]))

    def _coord_contains_index(coord, idx, atol=1e-6):
        """
        Test whether a Slice3D coordinate includes the currently displayed
        fixed-plane index.
        """
        vals = _coord_values(coord)

        if vals.size == 0:
            return False

        return bool(np.any(np.isclose(vals, float(idx), atol=atol, rtol=0.0)))

    def _insert_nan_gaps(vals):
        """
        Insert NaNs into an index-coordinate line so matplotlib breaks
        disconnected monitor pieces.
        """
        vals = np.asarray(vals, dtype=float)

        if vals.size <= 1:
            return vals

        vals = vals[np.isfinite(vals)]
        vals = np.unique(vals)
        vals.sort()

        if vals.size <= 1:
            return vals

        diffs = np.diff(vals)
        positive_diffs = diffs[diffs > 1e-12]
        step = np.median(positive_diffs) if positive_diffs.size > 0 else 1.0
        gap_threshold = max(1.5 * step, step + 1e-9)

        out = [vals[0]]
        for a, b in zip(vals[:-1], vals[1:]):
            if b - a > gap_threshold:
                out.append(np.nan)
            out.append(b)

        return np.asarray(out, dtype=float)

    def _is_slice3d_like(obj):
        """
        Avoid importing Slice3D here; use duck typing instead.
        """
        return (hasattr(obj, "x") and hasattr(obj, "y") and hasattr(obj, "z")) or (
            len(obj) == 3
            and all(isinstance(o, (slice, np.ndarray, torch.Tensor)) for o in obj)
        )

    def _plot_index_line(axis, h_idx, v_idx, h_dim, v_dim, color):
        h_phys = _idx_to_phys(h_idx, h_dim)
        v_phys = _idx_to_phys(v_idx, v_dim)

        axis.plot(
            h_phys,
            v_phys,
            color=color,
            alpha=0.80,
            linewidth=max(0.8, 1.2 * font_scale),
            zorder=20,
        )

    def _scatter_index_points(axis, h_idx, v_idx, h_dim, v_dim, color):
        h_phys = _idx_to_phys(h_idx, h_dim)
        v_phys = _idx_to_phys(v_idx, v_dim)

        axis.scatter(
            h_phys,
            v_phys,
            c=color,
            s=max(2.0, 4.0 * font_scale),
            alpha=0.65,
            linewidths=0,
            zorder=21,
        )

    def _draw_slice3d_monitor(axis, m_slice, color, plane):
        """
        Project one Slice3D-like monitor onto one of the three 2D views.

        View definitions:
            x-y view: fixed z, draw x vs y
            x-z view: fixed y, draw x vs z
            y-z view: fixed x, draw y vs z

        A monitor is drawn only if it intersects the currently displayed
        fixed cross-section.
        """
        if hasattr(m_slice, "x"):
            coords = {
                0: m_slice.x,
                1: m_slice.y,
                2: m_slice.z,
            }
        else:
            coords = {
                0: m_slice[0],
                1: m_slice[1],
                2: m_slice[2],
            }

        h_dim = plane["h_dim"]
        v_dim = plane["v_dim"]
        fixed_dim = plane["fixed_dim"]
        fixed_idx = plane["fixed_idx"]

        if not _coord_contains_index(coords[fixed_dim], fixed_idx):
            return

        h_vals = _coord_values(coords[h_dim])
        v_vals = _coord_values(coords[v_dim])

        if h_vals.size == 0 or v_vals.size == 0:
            return

        h_scalar = _coord_is_scalar(coords[h_dim])
        v_scalar = _coord_is_scalar(coords[v_dim])

        if h_scalar and v_scalar:
            _scatter_index_points(
                axis,
                np.asarray([h_vals[0]]),
                np.asarray([v_vals[0]]),
                h_dim,
                v_dim,
                color,
            )

        elif h_scalar and not v_scalar:
            v_line = _insert_nan_gaps(v_vals)
            h_line = np.full_like(v_line, h_vals[0], dtype=float)
            h_line[np.isnan(v_line)] = np.nan

            _plot_index_line(axis, h_line, v_line, h_dim, v_dim, color)

        elif not h_scalar and v_scalar:
            h_line = _insert_nan_gaps(h_vals)
            v_line = np.full_like(h_line, v_vals[0], dtype=float)
            v_line[np.isnan(h_line)] = np.nan

            _plot_index_line(axis, h_line, v_line, h_dim, v_dim, color)

        else:
            # The projected monitor covers an area in this view.
            # Draw its boundary, matching the 2D box behavior.
            h_line = _insert_nan_gaps(h_vals)
            v_line = _insert_nan_gaps(v_vals)

            h_min = np.nanmin(h_vals)
            h_max = np.nanmax(h_vals)
            v_min = np.nanmin(v_vals)
            v_max = np.nanmax(v_vals)

            # bottom edge
            _plot_index_line(
                axis,
                h_line,
                np.full_like(h_line, v_min, dtype=float),
                h_dim,
                v_dim,
                color,
            )

            # top edge
            _plot_index_line(
                axis,
                h_line,
                np.full_like(h_line, v_max, dtype=float),
                h_dim,
                v_dim,
                color,
            )

            # left edge
            _plot_index_line(
                axis,
                np.full_like(v_line, h_min, dtype=float),
                v_line,
                h_dim,
                v_dim,
                color,
            )

            # right edge
            _plot_index_line(
                axis,
                np.full_like(v_line, h_max, dtype=float),
                v_line,
                h_dim,
                v_dim,
                color,
            )

    def _extract_mask_slice(mask, plane):
        """
        Extract a 2D mask slice corresponding to the current display plane.
        """
        mask = np.asarray(mask)

        h_dim = plane["h_dim"]
        v_dim = plane["v_dim"]
        fixed_dim = plane["fixed_dim"]
        fixed_idx = plane["fixed_idx"]

        if mask.ndim == 3:
            if mask.shape != (nx, ny, nz):
                return None

            if fixed_dim == 2:
                return mask[:, :, fixed_idx]
            if fixed_dim == 1:
                return mask[:, fixed_idx, :]
            if fixed_dim == 0:
                return mask[fixed_idx, :, :]

        if mask.ndim == 2:
            expected_shape = (dim_sizes[h_dim], dim_sizes[v_dim])
            if mask.shape == expected_shape:
                return mask

        return None

    def _draw_mask_monitor(axis, mask, color, plane):
        mask_2d = _extract_mask_slice(mask, plane)

        if mask_2d is None:
            return

        h_idx, v_idx = np.nonzero(mask_2d)

        if len(h_idx) == 0:
            return

        _scatter_index_points(
            axis,
            h_idx,
            v_idx,
            plane["h_dim"],
            plane["v_dim"],
            color,
        )

    def _draw_monitors(axis, plane):
        if monitors is None or len(monitors) == 0:
            return

        for m in monitors:
            if not isinstance(m, (tuple, list)) or len(m) < 2:
                continue

            obj, color = m[0], m[1]

            obj = slice3d_to_indices(obj)

            if _is_slice3d_like(obj):
                _draw_slice3d_monitor(axis, obj, color, plane)

            elif isinstance(obj, np.ndarray):
                _draw_mask_monitor(axis, obj, color, plane)

    row_specs = [
        {
            "title": r"$|" + component[0] + r"|$",
            "kind": "abs",
            "cmap": "magma",
            "vmin": abs_vmin,
            "vmax": abs_vmax,
        },
        {
            "title": r"$\mathrm{Re}(" + component + r")$",
            "kind": "real",
            "cmap": "RdBu_r",
            "vmin": -real_lim,
            "vmax": real_lim,
        },
        {
            "title": r"$\epsilon$",
            "kind": "eps",
            "cmap": "Greys",
            "vmin": eps_vmin,
            "vmax": eps_vmax,
        },
    ]
    if thermal_map_arr is not None:
        thermal_title = (thermal_map_name or "thermal_map").replace("_", " ")
        use_diverging = (
            "coeff" in thermal_title.lower()
            or "grad" in thermal_title.lower()
            or thermal_vmin < 0.0
        )
        row_specs.append(
            {
                "title": thermal_title,
                "kind": "thermal",
                "cmap": "RdBu_r" if use_diverging else "coolwarm",
                "vmin": (-thermal_lim if use_diverging else thermal_vmin),
                "vmax": (thermal_lim if use_diverging else thermal_vmax),
                "per_plane_limits": True,
                "use_diverging": use_diverging,
            }
        )
    if show_delta_eps:
        row_specs.append(
            {
                "title": r"$\Delta \epsilon$",
                "kind": "delta_eps",
                "cmap": "RdBu_r",
                "vmin": -delta_lim,
                "vmax": delta_lim,
            }
        )
    if eps_grad_arr is not None:
        row_specs.append(
            {
                "title": r"$dL/d\epsilon$",
                "kind": "eps_grad",
                "cmap": "RdBu_r",
                "vmin": -grad_lim,
                "vmax": grad_lim,
            }
        )

    # ------------------------------------------------------------------
    # Draw all panels
    # ------------------------------------------------------------------
    for j, plane in enumerate(planes):
        field_2d = plane["field"]
        total_field_abs_2d = plane["total_field_abs"]
        eps_2d = np.real(plane["eps"]).astype(np.float64)
        delta_eps_2d = plane["delta_eps"]
        eps_grad_2d = (
            None
            if plane["eps_grad"] is None
            else np.real(plane["eps_grad"]).astype(np.float64)
        )
        thermal_2d = (
            None
            if plane["thermal"] is None
            else np.real(plane["thermal"]).astype(np.float64)
        )
        heat_source_2d = plane["heat_source"]

        for i, spec in enumerate(row_specs):
            ax_i, cax_i = _add_axes(i, j)

            if spec["kind"] == "abs":
                img_data = total_field_abs_2d
            elif spec["kind"] == "real":
                img_data = np.real(field_2d)
            elif spec["kind"] == "delta_eps":
                img_data = delta_eps_2d
            elif spec["kind"] == "thermal":
                img_data = thermal_2d
            elif spec["kind"] == "eps_grad":
                img_data = eps_grad_2d
            else:
                img_data = eps_2d

            vmin = spec["vmin"]
            vmax = spec["vmax"]
            if spec["kind"] == "thermal" and spec.get("per_plane_limits", False):
                plane_min, plane_max = _finite_minmax(img_data)
                if spec.get("use_diverging", False):
                    plane_lim = max(abs(plane_min), abs(plane_max))
                    if np.isclose(plane_lim, 0.0):
                        plane_lim = 1.0
                    vmin, vmax = -plane_lim, plane_lim
                else:
                    if np.isclose(plane_min, plane_max):
                        pad = (
                            1.0 if np.isclose(plane_min, 0.0) else 0.05 * abs(plane_min)
                        )
                        plane_min -= pad
                        plane_max += pad
                    vmin, vmax = plane_min, plane_max

            im = _imshow_phys(
                ax_i,
                img_data,
                plane["x_phys"],
                plane["y_phys"],
                cmap=spec["cmap"],
                vmin=vmin,
                vmax=vmax,
            )

            # Overlay epsilon lightly on field and gradient plots, but keep
            # delta-epsilon panels as the pure difference field.
            if spec["kind"] in ("abs", "real", "eps_grad", "thermal"):
                _imshow_phys(
                    ax_i,
                    eps_2d,
                    plane["x_phys"],
                    plane["y_phys"],
                    cmap="Greys",
                    vmin=eps_vmin,
                    vmax=eps_vmax,
                    alpha=0.18,
                )

            _draw_npml(
                ax_i,
                field_2d.shape,
                plane["npml"],
                plane["x_phys"],
                plane["y_phys"],
            )

            # Monitor markup.
            _draw_monitors(ax_i, plane)
            _draw_heat_source_frames(
                ax_i,
                heat_source_2d,
                plane["x_phys"],
                plane["y_phys"],
            )

            _format_axis(
                ax_i,
                plane["xlabel"],
                plane["ylabel"],
                panel_ws[j],
                panel_hs[j],
            )

            ax_i.set_title(
                spec["title"] + "\n" + plane["name"],
                fontsize=panel_title_fs,
                pad=4,
            )

            _add_colorbar(im, cax_i)

    if param_grad_arr is not None:
        param_row = len(row_specs)
        ax_p, cax_p = _add_full_row_axes(param_row, param_x_width, param_y_height)
        im = _imshow_phys(
            ax_p,
            np.real(param_grad_arr).astype(np.float64),
            param_x_width,
            param_y_height,
            cmap="RdBu_r",
            vmin=-param_lim,
            vmax=param_lim,
        )
        _format_axis(
            ax_p,
            r"$x$ width ($\mu m$)",
            r"$y$ height ($\mu m$)",
            ax_p.get_position().width * fig_w,
            ax_p.get_position().height * fig_h,
        )
        ax_p.set_title(r"$dL/d$" + param_name, fontsize=panel_title_fs, pad=4)
        _add_colorbar(im, cax_p)

    edge_row = len(row_specs) + int(param_grad_arr is not None)
    for edge_index, edge in enumerate(edgebox_edges):
        edge_axes = _add_edgebox_axes(edge_row + edge_index)
        knot_index = np.arange(len(edge["raw"]))
        raw = np.asarray(edge["raw"])
        physical = np.asarray(edge["physical"])
        gradient = edge.get("grad")
        gradient = None if gradient is None else np.asarray(gradient)
        edge_title = edge.get("side", f"edge {edge_index}")

        edge_axes[0].plot(knot_index, raw, "o-", color="tab:blue")
        edge_axes[0].set_title(f"{edge_title}: raw knots", fontsize=panel_title_fs)
        edge_axes[0].set_ylabel("raw control", fontsize=label_fs)
        edge_axes[1].plot(knot_index, physical, "o-", color="tab:green")
        edge_axes[1].set_title(
            f"{edge_title}: physical offset", fontsize=panel_title_fs
        )
        edge_axes[1].set_ylabel("offset [um]", fontsize=label_fs)
        if gradient is None:
            edge_axes[2].text(
                0.5,
                0.5,
                "gradient unavailable",
                ha="center",
                va="center",
                transform=edge_axes[2].transAxes,
            )
        else:
            edge_axes[2].bar(knot_index, gradient, color="tab:red")
        edge_axes[2].set_title(f"{edge_title}: knot gradient", fontsize=panel_title_fs)
        edge_axes[2].set_ylabel("d objective / d raw", fontsize=label_fs)
        for axis in edge_axes:
            axis.set_xlabel("knot index", fontsize=label_fs)
            axis.tick_params(axis="both", labelsize=tick_fs)
            axis.grid(True, alpha=0.25)

    if title is not None:
        fig.suptitle(title, fontsize=suptitle_fs, y=1.0 - 0.10 / fig_h)

    volume = nx * ny * nz
    dpi = 150 if volume > 2000**3 else 300

    if filepath is not None:
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight", pad_inches=0.08)

    plt.close(fig)

    if if_gif:
        print("Warning: if_gif=True is not implemented for plot_eps_field_3d.")


def solver_eigs(A, Neigs, guess_value=1.0):
    """solves for `Neigs` eigenmodes of A
        A:            sparse linear operator describing modes
        Neigs:        number of eigenmodes to return
        guess_value:  estimate for the eigenvalues
    For more info, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
    """

    values, vectors = spl.eigs(A, k=Neigs, sigma=guess_value, v0=None, which="LM")

    return values, vectors


def insert_mode_spins(
    omega,
    dx,
    x,
    y,
    epsr,
    target=None,
    npml=0,
    m="Ez1",
    filtering=False,
):
    if isinstance(m, int):
        pol = "Ez"  # by default Ez mode
        logger.warning("The mode is not specified, by default, it is Ez mode")
    pol = m[0:2]
    m = int(m[2:])

    if target is None:
        target = np.zeros(epsr.shape, dtype=complex)
    epsr_cross = epsr[x, y]

    if len(x.shape) == 0:  # x direction slice
        direction = "x"
        xx = slice(x, x + 1)
        yy = y
        epsr_cross = epsr_cross[None, :]
    elif len(y.shape) == 0:  # y direction slice
        direction = "y"
        xx = x
        yy = slice(y, y + 1)
        epsr_cross = epsr_cross[:, None]

    # dxes = [dx * 1e6, dx * 1e6]  # dx_e and dx_h
    dxes = [dx, dx]  # dx_e and dx_h
    # args_2d = {
    #     "dxes": [
    #         [np.zeros(epsr_cross.shape[0]) + dx, np.zeros(epsr_cross.shape[1]) + dx]
    #         for dx in dxes
    #     ],  # [[dx_e, dy_e], [dx_h, dy_h]]
    #     "epsilon": np.concatenate([epsr_cross.flatten()] * 3, axis=0),
    #     "mu": np.concatenate([np.zeros_like(epsr_cross.flatten()) + 1] * 3, axis=0),
    #     "wavenumber_correction": True,
    # }
    if pol == "Ez":
        # from 1,2,3,4,5.... to 0, 2, 4, 6, 8
        m = (m - 1) * 2
    elif pol == "Hz":
        # from 1,2,3,4,5.... to 1, 3, 5, 7, 9
        m = m * 2 - 1
    # fields_2d = solve_waveguide_mode_2d(m, omega=omega / constants.C_0 / 1e6, **args_2d)

    # sim_params = {
    #         'omega': omega / constants.C_0 / 1e6,
    #         'axis': direction, # propagation direction
    #         'slices': (x, y),
    #         'mu': np.concatenate([np.zeros_like(epsr_cross.flatten()) + 1]*3, axis=0)
    #     }
    epsr = epsr[..., None]  # x,y,z
    sim_params = {
        # "omega": omega / constants.C_0 / 1e6,
        "omega": omega,
        "dxes": [
            [np.zeros(epsr.shape[0]) + dx, np.zeros(epsr.shape[1]) + dx, np.array([dx])]
            for dx in dxes
        ],  # [[dx_e, dy_e], [dx_h, dy_h]]
        "axis": 0 if direction == "x" else 1,  # propagation direction
        "slices": (xx, yy, slice(0, 1)),
        "polarity": 1,
        "mu": [np.zeros_like(epsr) + constants.MU_0] * 3,
    }
    slices = tuple(sim_params["slices"])

    epsr_x = (epsr + np.roll(epsr, shift=1, axis=0)) / 2
    epsr_y = (epsr + np.roll(epsr, shift=1, axis=1)) / 2
    fields_2d = solve_waveguide_mode(
        mode_number=m,
        epsilon=[
            epsr_x * constants.EPSILON_0,
            epsr_y * constants.EPSILON_0,
            epsr * constants.EPSILON_0,
        ],
        **sim_params,
    )
    if pol == "Ez":
        e = fields_2d["E"][2]  # Ez
        if direction == "x":
            h = fields_2d["H"][1]  # Hy
        elif direction == "y":
            h = fields_2d["H"][0]  # Hx
        target = e
        e = e[slices]
        h = h[slices]
    elif pol == "Hz":
        if direction == "x":
            e = fields_2d["E"][1]  # Ey
        elif direction == "y":
            e = fields_2d["E"][0]  # Ex
        h = fields_2d["H"][2]  # Hz
        target = h
        h = h[slices]
        e = e[slices]
        # es = {i: e[slices] for i, e in enumerate(fields_2d["E"])}
        # hs = {i: h[slices] for i, h in enumerate(fields_2d["H"])}
        # print(es)
        # print(hs)

    # import matplotlib.pyplot as plt
    # for m in range(0, 4):
    #     fields_2d = solve_waveguide_mode_2d(m, omega=omega / constants.C_0 / 1e6, **args_2d)
    #     print(fields_2d.keys())
    #     fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    #     for i, (key, field) in enumerate(fields_2d.items()):
    #         if key  == "E":
    #             for j, f in enumerate(field):
    #                 axes[0, j].plot(np.abs(f))
    #                 axes[0, j].set_title(f"{key} {j}")
    #         elif key == "H":
    #             for j, f in enumerate(field):
    #                 axes[1, j].plot(np.abs(f))
    #                 axes[1, j].set_title(f"{key} {j}")
    #     fig.savefig(f"fields_spins_mode-{m}.png", dpi=300)
    #     plt.close(fig)
    # exit(0)
    return h, e, 0, target


def get_modes(
    eps_cross,
    omega,
    dL,
    npml,
    m=1,
    filtering=True,
    eps_cross_xx=None,
    pol: str = "Ez",
    direction: str = "x",
    dxs=None,
    dys=None,
    weights=None,
):
    """Solve for the modes of a waveguide cross section
    ARGUMENTS
        eps_cross: the permittivity profile of the waveguide
        omega:     angular frequency of the modes
        dL:        grid size of the cross section in meters for the uniform case
        dxs, dys:  optional x/y cell-width arrays in meters. For direct
               get_modes calls, the supplied transverse widths must have
               length eps_cross.size and the other axis must have length 1.
        npml:      number of PML points on each side of the cross section
        m:         number of modes to solve for
        filtering: whether to filter out evanescent modes
    RETURNS
        vals:      array of effective indeces of the modes
        vectors:   array containing the corresponding mode profiles
    """

    k0 = omega / constants.C_0

    N = eps_cross.size

    matrices = compute_derivative_matrices(
        omega,
        (N, 1),
        [npml, 0],
        dL=dL,
        dxs=dxs,
        dys=dys,
    )

    Dxf, Dxb, Dyf, Dyb, Dzf, Dzb = matrices

    diag_eps_r = sp.spdiags(eps_cross.flatten(), [0], N, N)
    if pol == "Ez":
        # https://empossible.net/wp-content/uploads/2019/08/Lecture-4c-Finite-Difference-Analysis-of-Waveguides.pdf Slide 48 E mode
        A = diag_eps_r + Dxf.dot(Dxb) * (1 / k0) ** 2
    elif pol == "Hz":
        diag_eps_r_xx_inv = sp.spdiags(1 / eps_cross_xx.flatten(), [0], N, N)
        A = (
            diag_eps_r
            + diag_eps_r.dot(Dxf).dot(diag_eps_r_xx_inv).dot(Dxb) * (1 / k0) ** 2
        )

    # n_max = np.sqrt(np.max(eps_cross)) * 0.92
    n_max = np.sqrt(np.max(eps_cross))  # why 0.92???
    vals, vecs = solver_eigs(A, m, guess_value=n_max**2)

    if pol == "Hz":
        if direction[0] == "x":
            # vecs = np.roll(vecs, shift=1, axis=0)
            vecs = vecs
        elif direction[0] == "y":
            vecs = (vecs + np.roll(vecs, shift=1, axis=0)) / 2
        # vecs = np.roll(vecs, shift=1, axis=0)

    if filtering:
        filter_re = lambda vals: np.real(vals) > 0.0
        # filter_im = lambda vals: np.abs(np.imag(vals)) <= 1e-12
        filters = [filter_re]
        vals, vecs = filter_modes(vals, vecs, filters=filters)

    if vals.size == 0:
        raise BaseException("Could not find any eigenmodes for this waveguide")

    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size != vecs.shape[0]:
            raise ValueError(
                "Mode normalization weights must have one entry per cross-section point"
            )
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Mode normalization weights must be finite and positive")
        weights = weights / np.mean(weights)
    vecs = normalize_modes(vecs, weights=weights)

    return vals, vecs


def insert_mode(
    omega,
    dx,
    x,
    y,
    epsr,
    target=None,
    npml=0,
    m="Ez1",
    filtering=False,
    direction: str = "x",
    single_direction: bool = True,
    dxs=None,
    dys=None,
):
    """Solve for the modes in a cross section of epsr at the location defined by 'x' and 'y'

    The mode is inserted into the 'target' array if it is suppled, if the target array is not
    supplied, then a target array is created with the same shape as epsr, and the mode is
    inserted into it.
    """
    """
    https://github.com/fancompute/ceviche/blob/master/notes/FDFD_notes.pdf (Figure 2)
    Assume epsr is always at cell center [i+1/2, j+1/2],
    Then for Ez polarization, et_m is ez, which is at [i+1/2, j+1/2], ht_m is also at [i+1/2, j+1/2]. ht_m is colocated with et_m as an estimation.
    For Hz polarization,

    ## for nonuniform mode solving, make sure the grid is fine enough, an example, Hz for 500nm Si_eff waveguide, we use autogrid, min_per_wvl=25, 15 will get very bad mode.
    """
    # direction = direction[0]
    # from angler import Simulation
    if len(direction) == 1:
        single_direction = False

    if isinstance(m, int):
        pol = "Ez"  # by default Ez mode
        logger.warning("The mode is not specified, by default, it is Ez mode")
    pol = m[0:2]
    m = int(m[2:])
    # print(omega, epsr, dx, npml)

    # sim = Simulation(omega, epsr, dl=dx*1e6, NPML=[npml, npml], pol="Hz")
    # if len(x.shape) == 0:
    #     center = (x.item(), (y[0] + y[-1])//2)
    #     width = y[-1] - y[0]
    #     dir = "x"
    # else:
    #     center = ((x[0] + x[-1]) // 2, y.item())
    #     width = x[-1] - x[0]
    #     dir = "y"
    # # print(dir, center, width)
    # sim.add_mode(np.max(epsr)**0.5, dir, center=center, width=width, order=1)
    # sim.setup_modes()
    # fz_angler = sim.src[x, y]
    # print(fz_angler)

    if target is None:
        target = np.zeros(epsr.shape, dtype=complex)
    epsr_cross = epsr[x, y]

    # For a one-way source, the phase-shifted profile is placed in the cell
    # immediately before/after the source slice.  On a rectilinear grid the
    # distance between cell centers is the average of the two neighboring
    # cell widths, not a single global ``dx``.
    propagation_spacing = dx
    if single_direction and (dxs is not None or dys is not None):
        if dxs is None or dys is None:
            raise ValueError("dxs and dys must be provided together")
        dxs = np.asarray(dxs, dtype=float)
        dys = np.asarray(dys, dtype=float)
        if dxs.ndim != 1 or dys.ndim != 1:
            raise ValueError("dxs and dys must be one-dimensional arrays")

        if direction[0] == "x":
            propagation_index = x
            propagation_widths = dxs
        elif direction[0] == "y":
            propagation_index = y
            propagation_widths = dys
        else:
            raise ValueError(f"Invalid direction {direction}, should be 'x' or 'y'")

        if not isinstance(propagation_index, (int, np.integer)):
            raise ValueError(
                "The propagation-axis index must be scalar when single_direction is enabled"
            )
        if propagation_index < 0 or propagation_index >= propagation_widths.size:
            raise IndexError("The source slice is outside the propagation grid")

        offset = -1 if direction[1] == "+" else 1
        offset_index = propagation_index + offset
        if offset_index < 0 or offset_index >= propagation_widths.size:
            raise IndexError("The phase-offset cell is outside the propagation grid")
        propagation_spacing = 0.5 * (
            propagation_widths[propagation_index] + propagation_widths[offset_index]
        )
        if not np.isfinite(propagation_spacing) or propagation_spacing <= 0:
            raise ValueError("Propagation cell widths must be finite and positive")

    transverse_spacing = None
    if dxs is not None or dys is not None:
        if dxs is None or dys is None:
            raise ValueError("dxs and dys must be provided together")
        dxs = np.asarray(dxs, dtype=float)
        dys = np.asarray(dys, dtype=float)
        if dxs.ndim != 1 or dys.ndim != 1:
            raise ValueError("dxs and dys must be one-dimensional arrays")
        if direction[0] == "x":
            transverse_spacing = dys[y]
        elif direction[0] == "y":
            transverse_spacing = dxs[x]
        else:
            raise ValueError(f"Invalid direction {direction}, should be 'x' or 'y'")
        transverse_spacing = np.asarray(transverse_spacing, dtype=float).reshape(-1)
        if transverse_spacing.size != epsr_cross.size:
            raise ValueError(
                "Transverse spacing must have the same number of entries as epsr_cross"
            )
        if np.any(~np.isfinite(transverse_spacing)) or np.any(transverse_spacing <= 0):
            raise ValueError("Transverse spacing must contain finite positive values")

    if pol == "Hz":
        # if len(x.shape) == 0:  # x direction slice
        #     epsr_cross_xx = epsr_cross
        #     direction = "x"
        # elif len(y.shape) == 0:  # y direction slice
        #     epsr_cross_xx = (epsr_cross + np.roll(epsr_cross, shift=1)) / 2
        #     direction = "y"
        if direction[0] == "x":
            # epsr_cross_xx = epsr_cross
            epsr_previous = np.roll(epsr_cross, shift=1)
            if transverse_spacing is None:
                epsr_cross_xx = (epsr_cross + epsr_previous) / 2
            else:
                width = transverse_spacing
                previous_width = np.roll(width, shift=1)
                # Interpolate to the interface between the current cell and
                # its previous neighbour.  The value is weighted by the
                # opposite cell-center distance.
                epsr_cross_xx = (
                    width * epsr_previous + previous_width * epsr_cross
                ) / (width + previous_width)
            # epsr_cross_xx = (epsr_cross + np.roll(epsr_cross, shift=-1)) / 2
        elif direction[0] == "y":
            epsr_previous = np.roll(epsr_cross, shift=1)
            if transverse_spacing is None:
                epsr_cross_xx = (epsr_cross + epsr_previous) / 2
            else:
                width = transverse_spacing
                previous_width = np.roll(width, shift=1)
                epsr_cross_xx = (
                    width * epsr_previous + previous_width * epsr_cross
                ) / (width + previous_width)
        else:
            raise ValueError(f"Invalid direction {direction}, should be 'x' or 'y'")

    else:
        epsr_cross_xx = None
        # direction = "x" + direction[1:]

    ## see page 89 in https://empossible.net/wp-content/uploads/2019/08/Lecture-4f-FDFD-Extras.pdf
    ## E mode: -(Dxf @ Dxb + MU_0 * eps_0 * eps_r) Ez = gamma^2 Ez
    ## (Dxf @ Dxb / k0^2 + MU_0 * eps_0 / k0^2 * eps_r) Ez = (beta/k0)^2 Ez
    ## (Dxf @ Dxb / k0^2 + 1/omega^2 * eps_r) Ez = (beta/k0)^2 * Ez
    # gamma = j*k0*n_eff = j*beta
    ## beta = neff * k0 = neff * 2pi / lambda = neff * omega / c
    ## -(Dxf @ Dxb / k0^2 + eps_r) Ez = -beta*2 * Ez
    ## (Dxf @ Dxb + eps_r) Ez = beta*2 * Ez
    # Solves the eigenvalue problem:
    #    [ ∂²/∂x² / (k₀²) + εr ] E = (β²/k₀²) E
    #    [ ∂²/∂x² / (k₀²) + εr ] E = (β²/k₀²) E
    ## eigen value is effective index n_eff^2
    vals, fz = get_modes(
        epsr_cross,
        omega,
        dx,
        npml,
        m=m,
        filtering=filtering,
        eps_cross_xx=epsr_cross_xx,
        pol=pol,
        direction=direction,
        dxs=transverse_spacing,
        dys=(np.ones(1) if transverse_spacing is not None else None),
        weights=transverse_spacing,
    )

    # Compute transverse magnetic field as:
    #    H = β / (μ₀ ω) * E
    # where the β term originates from the spatial derivative in the propagation
    # direction.
    ## remove center phase
    if fz.shape[0] % 2 == 0:
        center_phase = np.exp(
            -1j
            * np.angle(
                (
                    fz[fz.shape[0] // 2 - 1 : fz.shape[0] // 2]
                    + fz[fz.shape[0] // 2 : fz.shape[0] // 2 + 1]
                )
                / 2
            )
        )
    else:
        center_phase = np.exp(
            -1j * np.angle(fz[fz.shape[0] // 2 : fz.shape[0] // 2 + 1])
        )
    fz = fz * center_phase

    ## for Ez pol, this e is Ez, h is tangential field, i.e., for x direction: hy, for y direction: hx
    k0 = omega / constants.C_0
    beta = np.real(np.sqrt(vals, dtype=complex)) * k0
    if pol == "Ez":
        e = fz
        h = beta / omega / constants.MU_0 * e
        mode_profile = np.atleast_2d(e)[:, m - 1].squeeze()
        target[x, y] = mode_profile

    elif pol == "Hz":
        h = fz
        e = h * omega * constants.MU_0 / beta
        mode_profile = np.atleast_2d(h)[:, m - 1].squeeze()
        target[x, y] = mode_profile

    if single_direction:
        neff = beta[m - 1] / (omega / constants.C_0)
        wl_cen = (
            (2 * np.pi * constants.C_0) / omega / neff
        )  # effective wavelength in waveguide in unit of meter
        mode_profile = mode_profile * np.exp(
            -1j * 2 * np.pi / wl_cen * propagation_spacing - 1j * np.pi
        )
        offset = -1 if direction[1] == "+" else 1
        if direction[0] == "x":
            target[x + offset, y] = mode_profile
        elif direction[0] == "y":
            target[x, y + offset] = mode_profile

    # import matplotlib.pyplot as plt

    # fig, axes = plt.subplots(1, 4, figsize=(10, 5))
    # im0 = axes[0].plot(
    #     np.real(epsr_cross),
    # )
    # axes[0].set_title("Real part of epsr_cross")
    # axes[1].plot(
    #     dxs if dxs is not None else np.full(epsr_cross.shape[0], dx), label="dxs"
    # )
    # axes[1].set_title("dxs, Propagation widths")
    # axes[2].plot(
    #     dys if dys is not None else np.full(epsr_cross.shape[1], dx), label="dys"
    # )
    # axes[2].set_title("dys, Transverse widths")
    # axes[3].plot(np.abs(mode_profile), label="Mode profile")
    # plt.legend()
    # plt.tight_layout()
    # print(x, y, epsr.shape)
    # print("Debug: [insert_modes] saved figures in 'insert_modes_debug.png'")
    # plt.savefig("insert_modes_debug.png")
    # exit(0)

    return h[:, m - 1], e[:, m - 1], beta, target
