"""CuPy RawKernel scalar subpixel rasterizer.

This computes geometric volume averages at rectilinear cell centers. It is not
Tidy3D's anisotropic Yee-face/Yee-edge subpixel model. Supported geometries are
``td.Box``, ``td.Cylinder``, and z-extruded ``td.PolySlab`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover
    cp = None
    _CUPY_IMPORT_ERROR = exc
else:
    _CUPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class _Geometry:
    kind: int
    vertices: np.ndarray
    zmin: float
    zmax: float
    eps: float
    priority: int
    bbox: tuple[float, float, float, float]
    axis: int = 2
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    axis_min: float = 0.0
    axis_max: float = 0.0


def _require_cupy():
    if cp is None:
        raise ImportError("gpu_subpixel requires CuPy") from _CUPY_IMPORT_ERROR


def _as_float_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 1D array of length >= 2")
    if not np.all(np.diff(array) > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return np.ascontiguousarray(array)


def _grid_boundaries(grid):
    try:
        values = grid.boundaries.to_list
    except AttributeError as exc:
        raise TypeError("grid must expose grid.boundaries.to_list") from exc
    if len(values) != 3:
        raise ValueError("Only 3D rectilinear grids are supported")
    return tuple(
        _as_float_array(value, f"grid boundary {axis}")
        for axis, value in zip("xyz", values)
    )


def _scalar_permittivity(structure) -> float:

    value = getattr(getattr(structure, "medium", None), "permittivity", None)
    if value is None:
        value = getattr(structure, "permittivity", None)
    if value is None:
        print(structure)
        raise ValueError("Each structure must have scalar medium.permittivity")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Structure permittivity must be finite")
    return value


def _finite_extent(center, size, axis: int, fallback):
    value = float(size[axis])
    if np.isinf(value):
        return fallback
    half = abs(value) / 2
    middle = float(center[axis])
    return middle - half, middle + half


def _geometry_records(structures: Iterable, grid_bounds) -> list[_Geometry]:
    x_bounds, y_bounds, z_bounds = grid_bounds
    records = []
    for order, structure in enumerate(structures):
        geometry = getattr(structure, "geometry", None)
        if geometry is None:
            raise TypeError("Each structure must expose a geometry")
        raw_priority = getattr(structure, "priority", None)
        priority = 0 if raw_priority is None else int(raw_priority)
        eps = _scalar_permittivity(structure)
        children = getattr(geometry, "geometries", None)
        if children is not None:
            for child_order, child in enumerate(children):
                proxy = type("StructureProxy", (), {})()
                proxy.geometry = child
                proxy.medium = structure.medium
                proxy.priority = priority
                records.extend(_geometry_records([proxy], grid_bounds))
            continue
        name = type(geometry).__name__
        if name == "Box":
            center = np.asarray(geometry.center, dtype=np.float64)
            size = np.asarray(geometry.size, dtype=np.float64)
            xmin, xmax = _finite_extent(center, size, 0, (x_bounds[0], x_bounds[-1]))
            ymin, ymax = _finite_extent(center, size, 1, (y_bounds[0], y_bounds[-1]))
            zmin, zmax = _finite_extent(center, size, 2, (z_bounds[0], z_bounds[-1]))
            records.append(
                _Geometry(
                    0,
                    np.empty((0, 2)),
                    zmin,
                    zmax,
                    eps,
                    priority * 1000000 + order,
                    (xmin, ymin, xmax, ymax),
                )
            )
        elif name == "PolySlab":
            if int(getattr(geometry, "axis", 2)) != 2:
                raise NotImplementedError(
                    "Only PolySlab objects extruded along z are supported"
                )
            vertices = np.asarray(geometry.vertices, dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
                continue
            zmin, zmax = map(float, geometry.slab_bounds)
            if zmax <= zmin:
                continue
            vertices = np.ascontiguousarray(vertices)
            sidewall_angle = float(getattr(geometry, "sidewall_angle", 0.0))
            if abs(sidewall_angle) > 1e-14:
                bottom = np.asarray(geometry.base_polygon, dtype=np.float64)
                top = np.asarray(geometry.top_polygon, dtype=np.float64)
                if bottom.shape != vertices.shape or top.shape != vertices.shape:
                    raise ValueError(
                        "Tidy3D PolySlab endpoint polygons must match vertices"
                    )
                vertices = np.ascontiguousarray(np.concatenate((bottom, top), axis=0))
                bbox_vertices = vertices
                kind = 2
            else:
                bbox_vertices = vertices
                kind = 1
            records.append(
                _Geometry(
                    kind,
                    vertices,
                    zmin,
                    zmax,
                    eps,
                    priority * 1000000 + order,
                    (
                        float(bbox_vertices[:, 0].min()),
                        float(bbox_vertices[:, 1].min()),
                        float(bbox_vertices[:, 0].max()),
                        float(bbox_vertices[:, 1].max()),
                    ),
                )
            )
        elif name == "Cylinder":
            axis = int(getattr(geometry, "axis", 2))
            if axis not in (0, 1, 2):
                raise ValueError(f"Cylinder axis must be 0, 1, or 2, got {axis}")
            center = tuple(float(value) for value in geometry.center)
            radius = float(geometry.radius)
            length = float(geometry.length)
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("Cylinder radius must be finite and positive")
            if np.isnan(length) or length <= 0:
                raise ValueError("Cylinder length must be positive")
            axis_bounds = (x_bounds, y_bounds, z_bounds)[axis]
            if np.isinf(length):
                axis_min, axis_max = float(axis_bounds[0]), float(axis_bounds[-1])
            else:
                axis_min = center[axis] - length / 2
                axis_max = center[axis] + length / 2
            mins = [center[index] - radius for index in range(3)]
            maxs = [center[index] + radius for index in range(3)]
            mins[axis], maxs[axis] = axis_min, axis_max
            records.append(
                _Geometry(
                    3,
                    np.empty((0, 2)),
                    mins[2],
                    maxs[2],
                    eps,
                    priority * 1000000 + order,
                    (mins[0], mins[1], maxs[0], maxs[1]),
                    axis=axis,
                    center=center,
                    radius=radius,
                    axis_min=axis_min,
                    axis_max=axis_max,
                )
            )
        else:
            raise NotImplementedError(f"Unsupported Tidy3D geometry: {name}")
    return sorted(records, key=lambda record: record.priority)


def _index_range(bounds: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    start = int(np.searchsorted(bounds, lo, side="right") - 1)
    stop = int(np.searchsorted(bounds, hi, side="left"))
    return max(0, start), min(bounds.size - 1, stop)


_KERNEL_TEMPLATE = r"""
extern "C" __device__ double circle_edge_area(
    double ax, double ay, double bx, double by, double radius) {
    double dx = bx - ax, dy = by - ay;
    double qa = dx * dx + dy * dy;
    double qb = 2.0 * (ax * dx + ay * dy);
    double qc = ax * ax + ay * ay - radius * radius;
    double cuts[4] = {0.0, 1.0, 1.0, 1.0};
    int ncuts = 2;
    if (qa > 1e-30) {
        double discriminant = qb * qb - 4.0 * qa * qc;
        if (discriminant > 0.0) {
            double root = sqrt(discriminant);
            double t0 = (-qb - root) / (2.0 * qa);
            double t1 = (-qb + root) / (2.0 * qa);
            if (t0 > 0.0 && t0 < 1.0) cuts[ncuts++] = t0;
            if (t1 > 0.0 && t1 < 1.0) cuts[ncuts++] = t1;
        }
    }
    for (int i = 1; i < ncuts; ++i) {
        double value = cuts[i];
        int j = i - 1;
        while (j >= 0 && cuts[j] > value) {
            cuts[j + 1] = cuts[j];
            --j;
        }
        cuts[j + 1] = value;
    }
    double area = 0.0;
    for (int i = 0; i + 1 < ncuts; ++i) {
        double ta = cuts[i], tb = cuts[i + 1];
        if (tb - ta <= 1e-15) continue;
        double px = ax + ta * dx, py = ay + ta * dy;
        double qx = ax + tb * dx, qy = ay + tb * dy;
        double tm = 0.5 * (ta + tb);
        double mx = ax + tm * dx, my = ay + tm * dy;
        double cross = px * qy - py * qx;
        if (mx * mx + my * my <= radius * radius) {
            area += 0.5 * cross;
        } else {
            double dot = px * qx + py * qy;
            area += 0.5 * radius * radius * atan2(cross, dot);
        }
    }
    return area;
}

extern "C" __device__ double circle_rect_area(
    double cx, double cy, double radius,
    double xmin, double xmax, double ymin, double ymax) {
    double x0 = xmin - cx, x1 = xmax - cx;
    double y0 = ymin - cy, y1 = ymax - cy;
    double area = 0.0;
    area += circle_edge_area(x0, y0, x1, y0, radius);
    area += circle_edge_area(x1, y0, x1, y1, radius);
    area += circle_edge_area(x1, y1, x0, y1, radius);
    area += circle_edge_area(x0, y1, x0, y0, radius);
    return fabs(area);
}

extern "C" __device__ float clip_area(
    const double *v, int start, int top_start, int n, double t,
    double xmin, double xmax, double ymin, double ymax) {
    double px[VERTEX_CAPACITY], py[VERTEX_CAPACITY];
    double qx[VERTEX_CAPACITY], qy[VERTEX_CAPACITY];
    for (int i = 0; i < n; ++i) {
        double bottom_x = v[2 * (start + i)], bottom_y = v[2 * (start + i) + 1];
        double top_x = v[2 * (top_start + i)], top_y = v[2 * (top_start + i) + 1];
        px[i] = bottom_x + t * (top_x - bottom_x);
        py[i] = bottom_y + t * (top_y - bottom_y);
    }
    double limits[4] = {xmin, xmax, ymin, ymax};
    for (int edge = 0; edge < 4; ++edge) {
        int out = 0;
        for (int i = 0; i < n; ++i) {
            int j = (i + n - 1) % n;
            double ax = px[j], ay = py[j], bx = px[i], by = py[i];
            bool ain = edge == 0 ? ax >= limits[0] : edge == 1 ? ax <= limits[1] : edge == 2 ? ay >= limits[2] : ay <= limits[3];
            bool bin = edge == 0 ? bx >= limits[0] : edge == 1 ? bx <= limits[1] : edge == 2 ? by >= limits[2] : by <= limits[3];
            if (ain != bin) {
                double da = edge < 2 ? bx - ax : by - ay;
                double target = limits[edge];
                double t = fabs(da) < 1e-30 ? 0.0 : (target - (edge < 2 ? ax : ay)) / da;
                qx[out] = ax + t * (bx - ax); qy[out++] = ay + t * (by - ay);
            }
            if (bin) { qx[out] = bx; qy[out++] = by; }
        }
        n = out;
        for (int i = 0; i < n; ++i) { px[i] = qx[i]; py[i] = qy[i]; }
        if (n == 0) return 0.0f;
    }
    double area = 0.0;
    for (int i = 0; i < n; ++i) { int j = (i + 1) % n; area += px[i] * py[j] - px[j] * py[i]; }
    return (float)fabs(0.5 * area);
}

extern "C" __global__ void rasterize(
    const double *xb, const double *yb, const double *zb, const int *kind,
    const double *vertices, const int *offset, const int *count,
    const double *zmin, const double *zmax, const double *material,
    const int *axis, const double *center, const double *radius,
    const double *axis_min, const double *axis_max,
    const int *ix0, const int *ix1, const int *iy0, const int *iy1,
    const int *iz0, const int *iz1, int ns, double background, float *out,
    float *fraction, int nx, int ny, int nz, double x0, double y0, double z0,
    double dx, double dy, double dz, int uniform) {
    int linear = blockDim.x * blockIdx.x + threadIdx.x;
    if (linear >= nx * ny * nz) return;
    int i = linear / (ny * nz), j = (linear / nz) % ny, k = linear % nz;
    double xa = uniform ? x0 + i * dx : xb[i], xa1 = uniform ? xa + dx : xb[i + 1];
    double ya = uniform ? y0 + j * dy : yb[j], ya1 = uniform ? ya + dy : yb[j + 1];
    double za = uniform ? z0 + k * dz : zb[k], za1 = uniform ? za + dz : zb[k + 1];
    double area_cell = (xa1 - xa) * (ya1 - ya), volume = area_cell * (za1 - za);
    double eps = background, fill = 0.0;
    for (int s = 0; s < ns; ++s) {
        if (i < ix0[s] || i >= ix1[s] || j < iy0[s] || j >= iy1[s] || k < iz0[s] || k >= iz1[s]) continue;
        double area;
        if (kind[s] == 0) {
            int base = 2 * offset[s];
            double w = fmax(0.0, fmin(xa1, vertices[base + 2]) - fmax(xa, vertices[base]));
            double h = fmax(0.0, fmin(ya1, vertices[base + 3]) - fmax(ya, vertices[base + 1]));
            area = w * h;
        } else if (kind[s] == 1) {
            area = clip_area(vertices, offset[s], offset[s], count[s], 0.0, xa, xa1, ya, ya1);
        } else if (kind[s] == 2) {
            double overlap_lo = fmax(za, zmin[s]);
            double overlap_hi = fmin(za1, zmax[s]);
            double midpoint = 0.5 * (overlap_lo + overlap_hi);
            double half_width = 0.5 * (overlap_hi - overlap_lo);
            const double gauss_x[5] = {-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459};
            const double gauss_w[5] = {0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851};
            double area_integral = 0.0;
            double z_span = zmax[s] - zmin[s];
            for (int q = 0; q < 5; ++q) {
                double sample_z = midpoint + half_width * gauss_x[q];
                double t = (sample_z - zmin[s]) / z_span;
                area_integral += gauss_w[q] * clip_area(vertices, offset[s], offset[s] + count[s], count[s], t, xa, xa1, ya, ya1);
            }
            area = half_width * area_integral;
        } else {
            int cylinder_axis = axis[s];
            double axial_overlap;
            if (cylinder_axis == 0) {
                area = circle_rect_area(
                    center[3 * s + 1], center[3 * s + 2], radius[s],
                    ya, ya1, za, za1);
                axial_overlap = fmax(
                    0.0, fmin(xa1, axis_max[s]) - fmax(xa, axis_min[s]));
            } else if (cylinder_axis == 1) {
                area = circle_rect_area(
                    center[3 * s], center[3 * s + 2], radius[s],
                    xa, xa1, za, za1);
                axial_overlap = fmax(
                    0.0, fmin(ya1, axis_max[s]) - fmax(ya, axis_min[s]));
            } else {
                area = circle_rect_area(
                    center[3 * s], center[3 * s + 1], radius[s],
                    xa, xa1, ya, ya1);
                axial_overlap = fmax(
                    0.0, fmin(za1, axis_max[s]) - fmax(za, axis_min[s]));
            }
            area *= axial_overlap;
        }
        double height = fmax(0.0, fmin(za1, zmax[s]) - fmax(za, zmin[s]));
        double occupied_volume = kind[s] == 2 || kind[s] == 3 ? area : area * height;
        double f = volume > 0.0 ? occupied_volume / volume : 0.0;
        f = fmin(1.0, fmax(0.0, f));
        eps += f * (material[s] - eps);
        fill += f * (1.0 - fill);
    }
    out[linear] = eps;
    if (fraction) fraction[linear] = fmin(1.0, fmax(0.0, fill));
}
"""


def _kernel_source(max_vertices: int, dtype) -> str:
    # Clipping can add at most four vertices, one for each rectangle side.
    capacity = max(8, int(max_vertices) + 4)
    scalar = "double" if dtype == cp.float64 else "float"
    return (
        _KERNEL_TEMPLATE.replace("VERTEX_CAPACITY", str(capacity))
        .replace("float *out", f"{scalar} *out")
        .replace("float *fraction", f"{scalar} *fraction")
    )


class GPUSubpixelRasterizer:
    """Reusable scalar volume-averaging rasterizer for one fixed 3D grid.

    Args:
        grid: Tidy3D grid exposing three rectilinear boundary arrays.
        dtype: CuPy ``float32`` or ``float64`` output dtype.

    Notes:
        Geometry preprocessing runs on the CPU. Rasterization runs on the
        active CUDA device with one thread per grid cell.
    """

    def __init__(self, grid, *, dtype=None):
        _require_cupy()
        self.bounds = _grid_boundaries(grid)
        self.shape = tuple(axis.size - 1 for axis in self.bounds)
        self.dtype = dtype or cp.float32
        if self.dtype not in (cp.float32, cp.float64):
            raise TypeError("dtype must be cupy.float32 or cupy.float64")
        self.uniform = all(
            np.allclose(np.diff(axis), np.diff(axis)[0]) for axis in self.bounds
        )
        self._kernel = None

    def rasterize(self, structures, background_eps=1.0, *, return_fraction=False):
        """Rasterize structures using ordered fractional-volume composition.

        Args:
            structures: Iterable of Tidy3D structures containing supported
                box, cylinder, or z-extruded polyslab geometries.
            background_eps: Scalar background relative permittivity.
            return_fraction: Whether to also return total occupied fraction.

        Returns:
            CuPy permittivity array, or ``(permittivity, fraction)`` when
            ``return_fraction`` is true. Arrays have shape ``(Nx, Ny, Nz)``.
        """
        records = _geometry_records(structures, self.bounds)
        if not records:
            output = cp.full(self.shape, background_eps, dtype=self.dtype)
            if return_fraction:
                return output, cp.zeros(self.shape, dtype=self.dtype)
            return output
        n = len(records)
        xb, yb, zb = self.bounds
        vertex_counts = [
            2 if record.kind == 0 else record.vertices.shape[0] for record in records
        ]
        offsets = np.cumsum([0, *vertex_counts[:-1]], dtype=np.int32)
        vertices = np.empty((sum(vertex_counts), 2), dtype=np.float64)
        kind = np.empty(n, dtype=np.int32)
        offset = offsets
        count = np.zeros(n, dtype=np.int32)
        zmin = np.empty(n, dtype=np.float64)
        zmax = np.empty(n, dtype=np.float64)
        material = np.empty(n, dtype=np.float64)
        axis = np.empty(n, dtype=np.int32)
        center = np.empty((n, 3), dtype=np.float64)
        radius = np.empty(n, dtype=np.float64)
        axis_min = np.empty(n, dtype=np.float64)
        axis_max = np.empty(n, dtype=np.float64)
        ranges = np.empty((n, 6), dtype=np.int32)
        for s, record in enumerate(records):
            kind[s] = record.kind
            zmin[s] = record.zmin
            zmax[s] = record.zmax
            material[s] = record.eps
            axis[s] = record.axis
            center[s] = record.center
            radius[s] = record.radius
            axis_min[s] = record.axis_min
            axis_max[s] = record.axis_max
            if record.kind == 0:
                vertices[offset[s] : offset[s] + 2] = (
                    (record.bbox[0], record.bbox[1]),
                    (record.bbox[2], record.bbox[3]),
                )
            else:
                vertices[offset[s] : offset[s] + record.vertices.shape[0]] = (
                    record.vertices
                )
                count[s] = (
                    record.vertices.shape[0]
                    if record.kind == 1
                    else record.vertices.shape[0] // 2
                )
            ranges[s] = (
                *_index_range(xb, record.bbox[0], record.bbox[2]),
                *_index_range(yb, record.bbox[1], record.bbox[3]),
                *_index_range(zb, record.zmin, record.zmax),
            )
        gpu = [
            cp.asarray(value)
            for value in (
                xb,
                yb,
                zb,
                kind,
                vertices,
                offset,
                count,
                zmin,
                zmax,
                material,
                axis,
                center,
                radius,
                axis_min,
                axis_max,
                ranges[:, 0],
                ranges[:, 1],
                ranges[:, 2],
                ranges[:, 3],
                ranges[:, 4],
                ranges[:, 5],
            )
        ]

        output = cp.empty(self.shape, dtype=self.dtype)
        fraction = cp.empty(self.shape, dtype=self.dtype) if return_fraction else None
        steps = [float(np.diff(axis)[0]) for axis in self.bounds]
        args = (
            *gpu,
            n,
            float(background_eps),
            output,
            fraction,
            *self.shape,
            xb[0],
            yb[0],
            zb[0],
            *steps,
            int(self.uniform),
        )
        max_vertices = (
            max(
                (
                    record.vertices.shape[0]
                    if record.kind == 1
                    else record.vertices.shape[0] // 2
                )
                for record in records
                if record.kind in (1, 2)
            )
            if any(record.kind in (1, 2) for record in records)
            else 0
        )
        if (
            self._kernel is None
            or getattr(self, "_kernel_capacity", None) != max_vertices
        ):
            self._kernel = cp.RawKernel(
                _kernel_source(max_vertices, self.dtype),
                "rasterize",
                options=("--std=c++11",),
            )
            self._kernel_capacity = max_vertices
        self._kernel((max(1, (output.size + 255) // 256),), (256,), args)
        return (output, fraction) if return_fraction else output


def gpu_subpixel_rasterize(
    structures, grid, background_eps=1.0, dtype=None, return_fraction=False
):
    """Rasterize Tidy3D structures into a CuPy scalar-permittivity array.

    Args:
        structures: Iterable of Tidy3D structures containing supported box,
            cylinder, or z-extruded polyslab geometries.
        grid: Existing three-dimensional Tidy3D rectilinear grid.
        background_eps: Scalar background relative permittivity.
        dtype: CuPy ``float32`` or ``float64`` output dtype.
        return_fraction: Whether to return the occupied-volume fraction beside
            the permittivity map.

    Returns:
        CuPy permittivity array, or ``(permittivity, fraction)`` when
        ``return_fraction`` is true.

    Notes:
        Cylinders may use any Cartesian extrusion axis. Circle-cell overlap is
        evaluated analytically; no polygon approximation is introduced.
    """
    return GPUSubpixelRasterizer(grid, dtype=dtype).rasterize(
        structures, background_eps, return_fraction=return_fraction
    )
