import time

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mesh import build_fixed_mesh, build_structured_mesh
from .problem import SteadyStateHeatJaxRuntime
from .solver import HeatSolveTorch
from .transfer import FixedMeshTransfer
from .utils import (
    ensure_positive_conductivity,
    infer_ndim,
    normalize_spacing,
    validate_k_map,
    validate_q_map,
)


class HeatSolver(nn.Module):
    """Torch-facing steady-state differentiable heat solver.

    Boundary conditions are specified on domain faces using keys
    ``xmin``, ``xmax``, ``ymin``, ``ymax`` and, in 3D, ``zmin``/``zmax``.
    Dirichlet and Neumann values are scalar in V1.
    """

    def __init__(
        self,
        grid_step,
        *,
        dimension=None,
        mesh_type="structured",
        fixed_mesh_points=None,
        fixed_mesh_cells=None,
        fixed_mesh_ele_type=None,
        fixed_mesh_grid_shape=None,
        fixed_mesh_grid_points=None,
        fixed_mesh_direct_cell_mapping=False,
        fixed_mesh_transfer_neighbors=8,
        dirichlet_bc=None,
        neumann_bc=None,
        backend="jax",
        solver_options=None,
        adjoint_solver_options=None,
    ):
        super().__init__()
        self.grid_step = grid_step
        self.dimension = dimension
        self.mesh_type = mesh_type
        self.fixed_mesh_points = fixed_mesh_points
        self.fixed_mesh_cells = fixed_mesh_cells
        self.fixed_mesh_ele_type = fixed_mesh_ele_type
        self.fixed_mesh_grid_shape = fixed_mesh_grid_shape
        self.fixed_mesh_grid_points = fixed_mesh_grid_points
        self.fixed_mesh_direct_cell_mapping = bool(fixed_mesh_direct_cell_mapping)
        self.fixed_mesh_transfer_neighbors = fixed_mesh_transfer_neighbors
        self.dirichlet_bc = dirichlet_bc
        self.neumann_bc = neumann_bc
        self.backend = str(backend).lower()
        self.solver_options = self._normalize_solver_options(
            self.backend,
            solver_options,
        )
        self.adjoint_solver_options = self._normalize_solver_options(
            self.backend,
            adjoint_solver_options,
        )
        self._linear_cache = {}
        self._fixed_mesh = None
        self._transfer = None
        self._bridge = HeatSolveTorch()
        if self.mesh_type == "fixed":
            self._initialize_fixed_mesh()

    @staticmethod
    def _default_solver_options(backend):
        ### pydiso is fast for smaller problem then iterative
        ### for relatively large problem, try jax.
        backend = backend.lower()
        if backend == "jax":
            return {
                "dirichlet_mode": "reduced_free_dof",
                "jax_solver": {
                    "precond": True,
                    "method": "cg",
                    "solve_dtype": "float64",
                    "jit": True,
                    "check_residual": True,
                    "residual_rtol": 1e-6,
                    "residual_atol": 1e-8,
                },
            }
        if backend == "pydiso":
            return {
                "dirichlet_mode": "reduced_free_dof",
                "pydiso_solver": {
                    "matrix_type": "real_symmetric_positive_definite",
                    "reuse_factorization": True,
                },
            }
        if backend == "scipy":
            return {"dirichlet_mode": "row_elimination", "spsolve_solver": {}}
        if backend == "petsc":
            return {
                "dirichlet_mode": "row_elimination",
                "petsc_solver": {"ksp_type": "bcgsl", "pc_type": "ilu"},
            }
        raise ValueError("Unsupported heat backend: %r" % (backend,))

    @classmethod
    def _normalize_solver_options(cls, backend, solver_options):
        backend = str(backend).lower()
        options = dict(solver_options or cls._default_solver_options(backend))
        dirichlet_mode = options.get(
            "dirichlet_mode",
            cls._default_solver_options(backend).get(
                "dirichlet_mode", "row_elimination"
            ),
        )
        normalized = {"dirichlet_mode": dirichlet_mode}

        backend_option_key = {
            "jax": "jax_solver",
            "pydiso": "pydiso_solver",
            "scipy": "spsolve_solver",
            "petsc": "petsc_solver",
        }.get(backend)
        if backend_option_key is None:
            raise ValueError("Unsupported heat backend: %r" % (backend,))

        backend_options = options.get(backend_option_key)
        if backend_options is None:
            default_options = cls._default_solver_options(backend)
            backend_options = default_options.get(backend_option_key, {})
        normalized[backend_option_key] = dict(backend_options)
        return normalized

    def clear_cache(self):
        seen_entries = set()
        for key in tuple(self._linear_cache.keys()):
            entry = self._linear_cache.pop(key, None)
            if entry is None:
                continue
            entry_id = id(entry)
            if entry_id in seen_entries:
                continue
            seen_entries.add(entry_id)
            if entry.get("solver") is not None:
                try:
                    entry["solver"].clear()
                except Exception:
                    pass

    def remesh(
        self,
        mesh_points=None,
        mesh_cells=None,
        ele_type=None,
        grid_shape=None,
        grid_points=None,
        direct_cell_mapping=None,
    ):
        if self.mesh_type != "fixed":
            raise ValueError("remesh is only available when mesh_type='fixed'.")
        if mesh_points is not None:
            self.fixed_mesh_points = mesh_points
        if mesh_cells is not None:
            self.fixed_mesh_cells = mesh_cells
        if ele_type is not None:
            self.fixed_mesh_ele_type = ele_type
        if grid_shape is not None:
            self.fixed_mesh_grid_shape = tuple(int(v) for v in grid_shape)
        if grid_points is not None:
            self.fixed_mesh_grid_points = grid_points
        if direct_cell_mapping is not None:
            self.fixed_mesh_direct_cell_mapping = bool(direct_cell_mapping)
        self.clear_cache()
        self._initialize_fixed_mesh()

    def _initialize_fixed_mesh(self):
        if self.fixed_mesh_points is None or self.fixed_mesh_cells is None:
            raise ValueError(
                "mesh_type='fixed' requires fixed_mesh_points and fixed_mesh_cells."
            )
        if self.fixed_mesh_grid_shape is None:
            raise ValueError(
                "mesh_type='fixed' requires fixed_mesh_grid_shape to define the design grid."
            )
        ndim = len(self.fixed_mesh_grid_shape)
        grid_step = normalize_spacing(self.grid_step, ndim)
        self._fixed_mesh = build_fixed_mesh(
            self.fixed_mesh_points,
            self.fixed_mesh_cells,
            ele_type=self.fixed_mesh_ele_type,
        )
        if self.fixed_mesh_direct_cell_mapping:
            expected_cells = int(torch.tensor(self.fixed_mesh_grid_shape).prod().item())
            num_cells = int(self._fixed_mesh.cells.shape[0])
            if num_cells != expected_cells:
                raise ValueError(
                    "fixed_mesh_direct_cell_mapping requires one FE cell per grid cell; "
                    f"got {num_cells} cells for grid shape {self.fixed_mesh_grid_shape}."
                )
            if self.fixed_mesh_grid_points is not None:
                grid_points = np.asarray(self.fixed_mesh_grid_points, dtype=np.float64)
                if grid_points.shape != self._fixed_mesh.cell_centers.shape:
                    raise ValueError(
                        "fixed_mesh_grid_points must match the fixed mesh cell-center array "
                        f"shape; got {grid_points.shape} vs {self._fixed_mesh.cell_centers.shape}."
                    )
                max_center_error = float(
                    np.max(np.abs(self._fixed_mesh.cell_centers - grid_points))
                )
                if max_center_error > 1e-8:
                    raise ValueError(
                        "fixed_mesh_direct_cell_mapping requires FE cell ordering to match "
                        "grid-point ordering, but cell centers differ from provided grid points "
                        f"by up to {max_center_error:.3e}."
                    )
            self._transfer = None
        else:
            self._transfer = FixedMeshTransfer(
                self.fixed_mesh_grid_shape,
                grid_step,
                self._fixed_mesh.cell_centers,
                k_neighbors=self.fixed_mesh_transfer_neighbors,
                grid_points=self.fixed_mesh_grid_points,
            )

    @staticmethod
    def _resize_grid_tensor(grid_tensor: Tensor, target_shape) -> Tensor:
        target_shape = tuple(int(v) for v in target_shape)
        if tuple(int(v) for v in grid_tensor.shape) == target_shape:
            return grid_tensor
        if grid_tensor.ndim == 2:
            resized = F.interpolate(
                grid_tensor[None, None],
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )
            return resized[0, 0]
        if grid_tensor.ndim == 3:
            resized = F.interpolate(
                grid_tensor[None, None],
                size=target_shape,
                mode="trilinear",
                align_corners=False,
            )
            return resized[0, 0]
        raise ValueError("Unsupported heat grid tensor dim %r." % (grid_tensor.ndim,))

    @staticmethod
    def _resize_positive_field_tensor(
        grid_tensor: Tensor,
        target_shape,
        *,
        mode_2d: str = "bilinear",
        mode_3d: str = "trilinear",
    ) -> Tensor:
        target_shape = tuple(int(v) for v in target_shape)
        input_shape = tuple(int(v) for v in grid_tensor.shape)
        if input_shape == target_shape:
            return grid_tensor

        eps = torch.finfo(grid_tensor.dtype).tiny
        reciprocal = 1.0 / torch.clamp(grid_tensor, min=eps)

        # For downsampling positive transport coefficients like conductivity, average
        # the reciprocal field so the coarse-grid representation preserves resistance
        # bottlenecks instead of artificially short-circuiting interfaces.
        if all(target <= current for target, current in zip(target_shape, input_shape)):
            if grid_tensor.ndim == 2:
                pooled = F.adaptive_avg_pool2d(reciprocal[None, None], target_shape)[
                    0, 0
                ]
            elif grid_tensor.ndim == 3:
                pooled = F.adaptive_avg_pool3d(reciprocal[None, None], target_shape)[
                    0, 0
                ]
            else:
                raise ValueError(
                    "Unsupported positive field tensor dim %r." % (grid_tensor.ndim,)
                )
            return 1.0 / torch.clamp(pooled, min=eps)

        # For upsampling, interpolate the reciprocal field and invert so interface
        # transitions remain resistance-preserving and conductivity stays positive.
        if grid_tensor.ndim == 2:
            resized = F.interpolate(
                reciprocal[None, None],
                size=target_shape,
                mode=mode_2d,
                align_corners=False,
            )[0, 0]
            return 1.0 / torch.clamp(resized, min=eps)
        if grid_tensor.ndim == 3:
            resized = F.interpolate(
                reciprocal[None, None],
                size=target_shape,
                mode=mode_3d,
                align_corners=False,
            )[0, 0]
            return 1.0 / torch.clamp(resized, min=eps)
        raise ValueError(
            "Unsupported positive field tensor dim %r." % (grid_tensor.ndim,)
        )

    @staticmethod
    def _resize_conductivity_tensor(grid_tensor: Tensor, target_shape) -> Tensor:
        return HeatSolver._resize_positive_field_tensor(
            grid_tensor,
            target_shape,
            mode_2d="bilinear",
            mode_3d="trilinear",
        )

    @staticmethod
    def _resize_density_tensor(grid_tensor: Tensor, target_shape) -> Tensor:
        target_shape = tuple(int(v) for v in target_shape)
        input_shape = tuple(int(v) for v in grid_tensor.shape)
        if input_shape == target_shape:
            return grid_tensor

        # q_map stores volumetric heat density, so when we downsample from the
        # optical grid to the coarser heat grid we need a conservative average,
        # not point-sampled interpolation. This keeps the integrated heat power
        # close to the source specified by geometry/current even for thin heaters
        # or non-integer pixel coverage.
        if all(target <= current for target, current in zip(target_shape, input_shape)):
            if grid_tensor.ndim == 2:
                return F.adaptive_avg_pool2d(grid_tensor[None, None], target_shape)[
                    0, 0
                ]
            if grid_tensor.ndim == 3:
                return F.adaptive_avg_pool3d(grid_tensor[None, None], target_shape)[
                    0, 0
                ]

        return HeatSolver._resize_grid_tensor(grid_tensor, target_shape)

    def _structured_solver_shape(
        self,
        input_shape,
        input_grid_step=None,
    ):
        input_shape = tuple(int(v) for v in input_shape)
        solver_grid_step = normalize_spacing(self.grid_step, len(input_shape))
        if input_grid_step is None:
            return input_shape
        input_grid_step = normalize_spacing(input_grid_step, len(input_shape))
        return tuple(
            max(1, int(round((n * dl_in) / dl_solver)))
            for n, dl_in, dl_solver in zip(
                input_shape, input_grid_step, solver_grid_step
            )
        )

    @staticmethod
    def _padding_pad_tuple(ndim: int, padding_cells) -> tuple[int, ...]:
        padding_cells = tuple(int(v) for v in padding_cells)
        if ndim == 2:
            return (
                padding_cells[2],
                padding_cells[3],
                padding_cells[0],
                padding_cells[1],
            )
        if ndim == 3:
            return (
                padding_cells[4],
                padding_cells[5],
                padding_cells[2],
                padding_cells[3],
                padding_cells[0],
                padding_cells[1],
            )
        raise ValueError(f"Unsupported heat padding dimension {ndim}")

    @classmethod
    def _pad_grid_tensor(
        cls,
        grid_tensor: Tensor,
        padding_cells,
        *,
        mode: str,
        value: float = 0.0,
    ) -> Tensor:
        # Heat and optical domains do not need to match physically. We pad only for the
        # heat solve so lateral insulating boundaries can sit several microns away from
        # the heater without forcing the optical simulation domain to grow as well.
        # After the solve, the temperature is cropped back to the original optical grid.
        padding_cells = tuple(int(v) for v in padding_cells)
        if not any(padding_cells):
            return grid_tensor
        pad_tuple = cls._padding_pad_tuple(grid_tensor.ndim, padding_cells)
        padded = grid_tensor[None, None]
        if mode == "constant":
            padded = F.pad(padded, pad_tuple, mode="constant", value=float(value))
        elif mode in {"replicate", "reflect"}:
            padded = F.pad(padded, pad_tuple, mode=mode)
        else:
            raise ValueError(f"Unsupported heat padding mode: {mode}")
        return padded[0, 0]

    @staticmethod
    def _crop_grid_tensor(grid_tensor: Tensor, padding_cells) -> Tensor:
        # Remove the artificial heat-only padding so the returned temperature map stays
        # aligned with the original optical-domain material maps.
        padding_cells = tuple(int(v) for v in padding_cells)
        if not any(padding_cells):
            return grid_tensor
        slices = []
        for axis in range(grid_tensor.ndim):
            start = padding_cells[2 * axis]
            stop_pad = padding_cells[2 * axis + 1]
            stop = -stop_pad if stop_pad > 0 else None
            slices.append(slice(start, stop))
        return grid_tensor[tuple(slices)]

    def forward(
        self,
        k_map,
        q_map=None,
        *,
        dirichlet_bc=None,
        neumann_bc=None,
        input_grid_step=None,
        padding_cells=None,
        padding_mode="replicate",
        source_padding_mode="constant",
        return_metadata=False,
    ):
        start_time = time.time()
        ndim = infer_ndim(
            k_map,
            (
                self.dimension
                if self.mesh_type == "structured"
                else ("%dd" % len(self.fixed_mesh_grid_shape))
            ),
        )
        validate_k_map(k_map, ndim=ndim)
        ensure_positive_conductivity(k_map)

        if q_map is None:
            q_map = torch.zeros_like(k_map)
        else:
            validate_q_map(q_map, k_map)

        input_shape = tuple(int(v) for v in k_map.shape)
        padding_cells = tuple(int(v) for v in (padding_cells or [0] * (2 * ndim)))
        solver_input_k = self._pad_grid_tensor(
            k_map,
            padding_cells,
            mode=str(padding_mode).lower(),
        )
        solver_input_q = self._pad_grid_tensor(
            q_map,
            padding_cells,
            mode=str(source_padding_mode).lower(),
            value=0.0,
        )

        grid_step = normalize_spacing(self.grid_step, ndim)
        solver_input_shape = tuple(int(v) for v in solver_input_k.shape)
        if self.mesh_type == "structured":
            solver_grid_shape = self._structured_solver_shape(
                solver_input_shape,
                input_grid_step=input_grid_step,
            )
            solver_grid_k = self._resize_conductivity_tensor(
                solver_input_k, solver_grid_shape
            )
            solver_grid_q = self._resize_density_tensor(
                solver_input_q, solver_grid_shape
            )
            mesh = build_structured_mesh(solver_grid_shape, grid_step)
            runtime_k = solver_grid_k
            runtime_q = solver_grid_q
        elif self.mesh_type == "fixed":
            solver_grid_shape = tuple(int(v) for v in self.fixed_mesh_grid_shape)
            if solver_input_shape != solver_grid_shape:
                solver_grid_k = self._resize_conductivity_tensor(
                    solver_input_k, solver_grid_shape
                )
                solver_grid_q = self._resize_density_tensor(
                    solver_input_q, solver_grid_shape
                )
            else:
                solver_grid_k = solver_input_k
                solver_grid_q = solver_input_q
            mesh = self._fixed_mesh
            if self.fixed_mesh_direct_cell_mapping:
                runtime_k = solver_grid_k.reshape(-1)
                runtime_q = solver_grid_q.reshape(-1)
            else:
                runtime_k = self._transfer.grid_to_mesh(solver_grid_k)
                runtime_q = self._transfer.grid_to_mesh(solver_grid_q)
        else:
            raise ValueError("Unsupported mesh_type: %r" % (self.mesh_type,))
        runtime = SteadyStateHeatJaxRuntime(
            mesh=mesh,
            dirichlet_bc=(
                dirichlet_bc if dirichlet_bc is not None else self.dirichlet_bc
            ),
            neumann_bc=neumann_bc if neumann_bc is not None else self.neumann_bc,
            solver_options=self.solver_options,
            adjoint_solver_options=self.adjoint_solver_options,
            linear_cache=self._linear_cache,
        )
        mesh_temperature = self._bridge(runtime_k, runtime_q, runtime)
        if self.mesh_type == "fixed":
            if self.fixed_mesh_direct_cell_mapping:
                temperature = mesh_temperature.reshape(solver_grid_shape)
            else:
                temperature = self._transfer.mesh_to_grid(mesh_temperature)
        else:
            temperature = mesh_temperature
        if tuple(int(v) for v in temperature.shape) != solver_input_shape:
            temperature = self._resize_grid_tensor(temperature, solver_input_shape)
        temperature = self._crop_grid_tensor(temperature, padding_cells)
        if tuple(int(v) for v in temperature.shape) != input_shape:
            temperature = self._resize_grid_tensor(temperature, input_shape)

        end_time = time.time()
        print(f"HEAT ({self.backend}) solver takes {end_time - start_time:.4f} seconds")
        if return_metadata:
            metadata = runtime.metadata()
            metadata["mesh_type"] = self.mesh_type
            metadata["input_grid_shape"] = input_shape
            metadata["padded_input_grid_shape"] = solver_input_shape
            metadata["solver_grid_shape"] = tuple(int(v) for v in solver_grid_shape)
            metadata["padding_cells"] = tuple(int(v) for v in padding_cells)
            if input_grid_step is not None:
                input_grid_step = normalize_spacing(input_grid_step, len(input_shape))
                metadata["physical_size"] = tuple(
                    float(n * dl) for n, dl in zip(input_shape, input_grid_step)
                )
            if self.mesh_type == "fixed":
                metadata["design_grid_shape"] = tuple(
                    int(v) for v in self.fixed_mesh_grid_shape
                )
                metadata["transfer_neighbors"] = int(self.fixed_mesh_transfer_neighbors)
                metadata["direct_cell_mapping"] = bool(
                    self.fixed_mesh_direct_cell_mapping
                )
            return temperature, metadata
        return temperature
