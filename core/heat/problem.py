from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax.flatten_util
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from .bc import normalize_dirichlet_bc, normalize_neumann_bc
from .mesh import StructuredMesh
from .utils import ensure_heat_dependencies

try:
    from core.fdfd.pydiso_solver import MKLPardisoSolver as PydisoSolver

    HAS_PYDISO = True
except Exception:
    PydisoSolver = None
    HAS_PYDISO = False

try:
    import logging

    from jax_fem import logger

    logger.setLevel(logging.WARNING)
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO
    from jax.scipy.sparse.linalg import bicgstab as jax_bicgstab
    from jax.scipy.sparse.linalg import cg as jax_cg
    from jax_fem.problem import Problem as _JaxFemProblem
except Exception as exc:  # pragma: no cover - exercised by import guards in tests
    jax = None
    jnp = None
    BCOO = None
    jax_cg = None
    jax_bicgstab = None
    _JaxFemProblem = object
    _HEAT_IMPORT_ERROR = exc
else:
    _HEAT_IMPORT_ERROR = None


class _CSRMatrixWrapper(object):
    def __init__(self, matrix):
        self.matrix = matrix.tocsr()

    def getValuesCSR(self):
        return self.matrix.indptr, self.matrix.indices, self.matrix.data

    def getSize(self):
        return self.matrix.shape

    def transpose(self):
        return _CSRMatrixWrapper(self.matrix.transpose().tocsr())

    def zeroRows(self, row_inds):
        row_inds = np.asarray(row_inds, dtype=np.int64).reshape(-1)
        lil = self.matrix.tolil(copy=True)
        for row in row_inds:
            lil.rows[int(row)] = []
            lil.data[int(row)] = []
            lil[int(row), int(row)] = 1.0
        self.matrix = lil.tocsr()


def _csr_to_scipy(matrix):
    indptr, indices, data = matrix.getValuesCSR()
    return sp.csr_matrix(
        (np.asarray(data), np.asarray(indices), np.asarray(indptr)),
        shape=matrix.getSize(),
    )


def _csr_matrices_equal(a, b):
    return (
        a.shape == b.shape
        and np.array_equal(a.indptr, b.indptr)
        and np.array_equal(a.indices, b.indices)
        and np.array_equal(a.data, b.data)
    )


def _csr_matrix_structure_equal(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and np.array_equal(a.indptr, b.indptr)
        and np.array_equal(a.indices, b.indices)
    )


def _iter_candidate_cache_keys(primary_key):
    seen = set()
    for key in (primary_key, "linear_fwd", "linear_adj"):
        if key is None or key in seen:
            continue
        seen.add(key)
        yield key


def _get_cached_warm_start(cache, key, size):
    if cache is None or key is None:
        return None
    entry = cache.get(key)
    if entry is None:
        return None
    value = entry.get("value")
    if value is None:
        return None
    value_np = np.asarray(value)
    if value_np.shape != (int(size),):
        return None
    return value_np


def _set_cached_warm_start(cache, key, value):
    if cache is None or key is None:
        return
    cache[key] = {
        "backend": "state",
        "value": np.array(np.asarray(value), copy=True).reshape(-1),
    }


def _csr_to_jax_bcoo(matrix):
    scipy_matrix = _csr_to_scipy(matrix)
    return BCOO.from_scipy_sparse(scipy_matrix).sort_indices(), scipy_matrix


def _normalize_dirichlet_mode(mode):
    normalized = str(mode or "row_elimination").strip().lower().replace("-", "_")
    aliases = {
        "row": "row_elimination",
        "row_only": "row_elimination",
        "row_elimination": "row_elimination",
        "reduced": "reduced_free_dof",
        "free_dof": "reduced_free_dof",
        "free_dofs": "reduced_free_dof",
        "reduced_free_dof": "reduced_free_dof",
        "reduced_free_dofs": "reduced_free_dof",
        "symmetric": "row_column_elimination",
        "row_column": "row_column_elimination",
        "row_column_elimination": "row_column_elimination",
        "symmetric_elimination": "row_column_elimination",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported heat Dirichlet elimination mode: {mode!r}")
    return aliases[normalized]


def _collect_dirichlet_data(problem):
    row_to_value = {}
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            row_inds = np.array(
                fe.node_inds_list[i] * fe.vec
                + fe.vec_inds_list[i]
                + problem.offset[ind],
                dtype=np.int64,
            ).reshape(-1)
            values = np.asarray(fe.vals_list[i], dtype=np.float64).reshape(-1)
            for row, value in zip(row_inds.tolist(), values.tolist()):
                previous = row_to_value.get(row)
                if previous is not None and not np.isclose(previous, value):
                    raise ValueError(
                        "Conflicting Dirichlet values detected for the same heat DOF."
                    )
                row_to_value[row] = float(value)

    if not row_to_value:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)

    rows = np.array(sorted(row_to_value), dtype=np.int64)
    values = np.array([row_to_value[int(row)] for row in rows], dtype=np.float64)
    return rows, values


def _build_raw_scipy_matrix(problem):
    return sp.csr_matrix(
        (np.asarray(problem.V), (problem.I, problem.J)),
        shape=(problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars),
    )


def _build_raw_triplets(problem):
    return (
        np.asarray(problem.I, dtype=np.int64),
        np.asarray(problem.J, dtype=np.int64),
        np.asarray(problem.V),
    )


def _get_reduced_dof_partition(problem):
    cached = getattr(problem, "_reduced_dof_partition", None)
    if cached is not None:
        return cached

    rows, _ = _collect_dirichlet_data(problem)
    num_dofs = int(problem.num_total_dofs_all_vars)
    constrained_mask = np.zeros(num_dofs, dtype=bool)
    constrained_mask[rows] = True
    free_rows = np.nonzero(~constrained_mask)[0].astype(np.int64, copy=False)

    free_id = np.full(num_dofs, -1, dtype=np.int64)
    free_id[free_rows] = np.arange(free_rows.size, dtype=np.int64)

    constrained_id = np.full(num_dofs, -1, dtype=np.int64)
    constrained_id[rows] = np.arange(rows.size, dtype=np.int64)

    cached = {
        "rows": rows,
        "constrained_mask": constrained_mask,
        "free_rows": free_rows,
        "free_mask": ~constrained_mask,
        "free_id": free_id,
        "constrained_id": constrained_id,
    }
    problem._reduced_dof_partition = cached
    return cached


def _apply_dirichlet_to_matrix(raw_matrix, dirichlet_rows, dirichlet_mode):
    dirichlet_mode = _normalize_dirichlet_mode(dirichlet_mode)
    if len(dirichlet_rows) == 0:
        return raw_matrix.tocsr()

    raw_matrix = raw_matrix.tocsr()
    lil = raw_matrix.tolil(copy=True)
    if dirichlet_mode == "row_elimination":
        lil[dirichlet_rows, :] = 0.0
        lil[dirichlet_rows, dirichlet_rows] = 1.0
        return lil.tocsr()

    constrained = np.zeros(raw_matrix.shape[0], dtype=bool)
    constrained[dirichlet_rows] = True
    coo = raw_matrix.tocoo(copy=False)
    keep_mask = (~constrained[coo.row]) & (~constrained[coo.col])
    diag_data = np.ones(len(dirichlet_rows), dtype=raw_matrix.dtype)
    matrix = sp.csr_matrix(
        (
            np.concatenate((coo.data[keep_mask], diag_data)),
            (
                np.concatenate((coo.row[keep_mask], dirichlet_rows)),
                np.concatenate((coo.col[keep_mask], dirichlet_rows)),
            ),
        ),
        shape=raw_matrix.shape,
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _build_scipy_matrix(problem, dirichlet_mode="row_elimination"):
    matrix = _build_raw_scipy_matrix(problem)
    dirichlet_rows, _ = _collect_dirichlet_data(problem)
    matrix = _apply_dirichlet_to_matrix(matrix, dirichlet_rows, dirichlet_mode)

    if hasattr(problem, "P_mat"):
        matrix = problem.P_mat.T @ (matrix @ problem.P_mat)

    return _CSRMatrixWrapper(matrix)


def _get_flatten_fn(fn_sol_list, problem):
    def fn_dofs(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        val_list = fn_sol_list(sol_list)
        return jax.flatten_util.ravel_pytree(val_list)[0]

    return fn_dofs


def _apply_bc_vec(res_vec, dofs, problem, scale=1.0):
    res_list = problem.unflatten_fn_sol_list(res_vec)
    sol_list = problem.unflatten_fn_sol_list(dofs)

    for ind, fe in enumerate(problem.fes):
        res = res_list[ind]
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]],
                unique_indices=True,
            )
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].add(
                -fe.vals_list[i] * scale
            )
        res_list[ind] = res

    return jax.flatten_util.ravel_pytree(res_list)[0]


def _apply_bc(res_fn, problem, scale=1.0):
    def res_fn_bc(dofs):
        return _apply_bc_vec(res_fn(dofs), dofs, problem, scale)

    return res_fn_bc


def _assign_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(fe.vals_list[i])
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def _copy_bc(dofs, problem):
    new_dofs = jnp.zeros_like(dofs)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    new_sol_list = problem.unflatten_fn_sol_list(new_dofs)

    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        new_sol = new_sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            new_sol = new_sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]]
            )
        new_sol_list[ind] = new_sol

    return jax.flatten_util.ravel_pytree(new_sol_list)[0]


def _dirichlet_increment(dofs, problem):
    rows, values = _collect_dirichlet_data(problem)
    if len(rows) == 0:
        return rows, values, np.empty((0,), dtype=np.float64)
    dofs_np = np.asarray(dofs)
    delta = values - dofs_np[rows]
    return rows, values, delta


def _prepare_symmetric_increment_system(problem, raw_matrix, raw_residual, dofs):
    rows, _, delta = _dirichlet_increment(dofs, problem)
    matrix = _apply_dirichlet_to_matrix(raw_matrix, rows, "row_column_elimination")
    rhs = -np.asarray(raw_residual, dtype=np.float64)
    if len(rows) > 0:
        delta_full = np.zeros(raw_matrix.shape[1], dtype=rhs.dtype)
        delta_full[rows] = delta
        rhs = rhs - raw_matrix @ delta_full
        rhs[rows] = delta
    return _CSRMatrixWrapper(matrix), jnp.asarray(rhs)


def _prepare_reduced_increment_system(problem, raw_matrix, raw_residual, dofs):
    del raw_matrix
    rows, _, delta = _dirichlet_increment(dofs, problem)
    reduced_matrix, free_rows = _build_reduced_system_matrix(problem)
    partition = _get_reduced_dof_partition(problem)
    free_mask = partition["free_mask"]
    free_id = partition["free_id"]
    constrained_id = partition["constrained_id"]

    rhs = -np.asarray(raw_residual, dtype=np.float64)[free_rows]
    if len(rows) > 0:
        I, J, V = _build_raw_triplets(problem)
        fc_mask = free_mask[I] & (~free_mask[J])
        if np.any(fc_mask):
            rhs = rhs - np.bincount(
                free_id[I[fc_mask]],
                weights=np.asarray(V[fc_mask], dtype=np.float64)
                * delta[constrained_id[J[fc_mask]]],
                minlength=free_rows.size,
            )

    return reduced_matrix, jnp.asarray(rhs), free_rows, rows, delta


def _build_reduced_system_matrix(problem):
    partition = _get_reduced_dof_partition(problem)
    free_rows = partition["free_rows"]
    free_mask = partition["free_mask"]
    free_id = partition["free_id"]

    I, J, V = _build_raw_triplets(problem)
    ff_mask = free_mask[I] & free_mask[J]
    reduced_matrix = sp.csr_matrix(
        (
            V[ff_mask],
            (free_id[I[ff_mask]], free_id[J[ff_mask]]),
        ),
        shape=(free_rows.size, free_rows.size),
    )
    reduced_matrix.sum_duplicates()
    reduced_matrix.sort_indices()
    return _CSRMatrixWrapper(reduced_matrix), free_rows


def _prepare_reduced_adjoint_system(problem, adjoint_rhs):
    reduced_matrix, free_rows = _build_reduced_system_matrix(problem)
    reduced_rhs = np.asarray(adjoint_rhs, dtype=np.float64)[free_rows]
    return reduced_matrix, jnp.asarray(reduced_rhs), free_rows


def _linear_spsolve(matrix, rhs):
    scipy_matrix = _csr_to_scipy(matrix)
    rhs_np = np.asarray(rhs)
    x = spl.spsolve(scipy_matrix, rhs_np)
    return jnp.asarray(x)


def _linear_jax_solve(
    matrix,
    rhs,
    x0=None,
    precond=True,
    method="bicgstab",
    tol=1e-10,
    atol=1e-10,
    maxiter=10000,
    solve_dtype="float64",
    jit=True,
    check_residual=True,
    residual_rtol=1e-7,
    residual_atol=1e-9,
    cache=None,
    cache_key="linear_fwd",
):
    scipy_matrix = _csr_to_scipy(matrix)
    solve_dtype = str(solve_dtype).lower()
    if solve_dtype in {"float64", "fp64", "double"}:
        np_dtype = np.float64
        jax_dtype = jnp.float64
    elif solve_dtype in {"float32", "fp32", "single"}:
        np_dtype = np.float32
        jax_dtype = jnp.float32
    else:
        raise ValueError(f"Unsupported JAX heat solve dtype: {solve_dtype!r}")

    scipy_matrix = scipy_matrix.astype(np_dtype)
    solve_signature = (
        str(method).lower(),
        bool(precond),
        float(tol),
        float(atol),
        int(maxiter),
        bool(jit),
    )
    cached_key = None
    cached_entry = None
    cached_jax_matrix = None
    cached_diagonal = None
    compiled_solvers = None
    if cache is not None:
        for candidate_key in _iter_candidate_cache_keys(cache_key):
            candidate_entry = cache.get(candidate_key)
            if candidate_entry is None or candidate_entry.get("backend") != "jax":
                continue
            cached_matrix = candidate_entry.get("matrix")
            if cached_matrix is None or not _csr_matrices_equal(
                cached_matrix, scipy_matrix
            ):
                continue
            cached_key = candidate_key
            cached_entry = candidate_entry
            cached_jax_matrix = candidate_entry.get("jax_matrix")
            cached_diagonal = candidate_entry.get("diagonal")
            compiled_solvers = candidate_entry.get("compiled_solvers")
            break

    if cached_jax_matrix is None:
        jax_matrix = BCOO.from_scipy_sparse(scipy_matrix).sort_indices()
        diagonal = jnp.asarray(scipy_matrix.diagonal())
        cache_entry = {
            "backend": "jax",
            "matrix": scipy_matrix.copy(),
            "jax_matrix": jax_matrix,
            "diagonal": diagonal,
            "compiled_solvers": {},
        }
        if cache is not None and cache_key is not None:
            cache[cache_key] = cache_entry
            if cached_key is not None and cached_key != cache_key:
                cache[cached_key] = cache_entry
    else:
        jax_matrix = cached_jax_matrix
        diagonal = cached_diagonal
        if cache is not None and cache_key is not None:
            cache[cache_key] = cached_entry

    active_entry = None if cache is None or cache_key is None else cache.get(cache_key)
    compiled_solvers = (
        {} if active_entry is None else active_entry.setdefault("compiled_solvers", {})
    )

    rhs_jax = jnp.asarray(rhs, dtype=jax_dtype)
    x0_jax = jnp.zeros_like(rhs_jax) if x0 is None else jnp.asarray(x0, dtype=jax_dtype)

    def preconditioner(vec):
        safe_diag = jnp.where(jnp.abs(diagonal) > 0, diagonal, 1.0)
        return vec / safe_diag

    method = str(method).lower()
    if method == "bicgstab":
        solve_fn = jax_bicgstab
    elif method == "cg":
        solve_fn = jax_cg
    else:
        raise ValueError(f"Unsupported JAX heat iterative method: {method!r}")

    compiled_solve = compiled_solvers.get(solve_signature)
    if compiled_solve is None:

        def solve_impl(matrix_arg, rhs_arg, x0_arg, diagonal_arg):
            def diag_preconditioner(vec):
                safe_diag = jnp.where(jnp.abs(diagonal_arg) > 0, diagonal_arg, 1.0)
                return vec / safe_diag

            preconditioner_fn = diag_preconditioner if precond else None
            return solve_fn(
                matrix_arg,
                rhs_arg,
                x0=x0_arg,
                M=preconditioner_fn,
                tol=tol,
                atol=atol,
                maxiter=maxiter,
            )

        compiled_solve = jax.jit(solve_impl) if jit else solve_impl
        compiled_solvers[solve_signature] = compiled_solve

    x, info = compiled_solve(jax_matrix, rhs_jax, x0_jax, diagonal)
    if info is not None:
        info_value = (
            np.asarray(info).item()
            if np.asarray(info).shape == ()
            else np.asarray(info)
        )
        if np.any(np.asarray(info_value) != 0):
            raise RuntimeError(f"JAX {method} did not converge for the heat solve.")

    if check_residual:
        residual = jax_matrix @ x - rhs_jax
        residual_norm = jnp.linalg.norm(residual)
        rhs_norm = jnp.linalg.norm(rhs_jax)
        allowed_residual = jnp.maximum(
            jnp.asarray(float(residual_atol), dtype=jax_dtype),
            jnp.asarray(float(residual_rtol), dtype=jax_dtype)
            * jnp.maximum(rhs_norm, jnp.asarray(1.0, dtype=jax_dtype)),
        )
        residual_norm_value = float(np.asarray(residual_norm))
        allowed_residual_value = float(np.asarray(allowed_residual))
        if (
            not np.isfinite(residual_norm_value)
            or residual_norm_value > allowed_residual_value
        ):
            raise RuntimeError(
                "JAX heat solve residual check failed: "
                f"||Ax-b||={residual_norm_value:.3e}, allowed<={allowed_residual_value:.3e}."
            )
    return x


def _linear_petsc_solve(matrix, rhs, ksp_type="bcgsl", pc_type="ilu"):
    from petsc4py import PETSc

    scipy_matrix = _csr_to_scipy(matrix)
    A = PETSc.Mat().createAIJ(
        size=scipy_matrix.shape,
        csr=(
            scipy_matrix.indptr.astype(PETSc.IntType, copy=False),
            scipy_matrix.indices.astype(PETSc.IntType, copy=False),
            scipy_matrix.data,
        ),
    )
    b = PETSc.Vec().createSeq(len(rhs))
    b.setValues(range(len(rhs)), np.asarray(rhs))
    x = PETSc.Vec().createSeq(len(rhs))
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    ksp.pc.setType(pc_type)
    ksp.solve(b, x)
    return jnp.asarray(x.getArray())


def _linear_pydiso_solve(matrix, rhs, matrix_type="real_nonsymmetric"):
    if not HAS_PYDISO:
        raise ImportError("pydiso backend requested, but pydiso is not available.")

    scipy_matrix = _csr_to_scipy(matrix)
    rhs_np = np.array(np.asarray(rhs), copy=True)
    if scipy_matrix.dtype != rhs_np.dtype:
        scipy_matrix = scipy_matrix.astype(rhs_np.dtype)
    solver = PydisoSolver(scipy_matrix, matrix_type=matrix_type, factor=True)
    x = solver.solve(rhs_np).reshape(-1)
    solver.clear()
    return jnp.asarray(x)


def _linear_pydiso_solve_cached(
    matrix,
    rhs,
    matrix_type="real_nonsymmetric",
    cache=None,
    cache_key="linear_fwd",
    reuse_factorization=True,
):
    if not HAS_PYDISO:
        raise ImportError("pydiso backend requested, but pydiso is not available.")

    scipy_matrix = _csr_to_scipy(matrix)
    rhs_np = np.array(np.asarray(rhs), copy=True)
    if scipy_matrix.dtype != rhs_np.dtype:
        scipy_matrix = scipy_matrix.astype(rhs_np.dtype)

    def _iter_candidate_cache_keys(primary_key):
        seen = set()
        for key in (primary_key, "linear_fwd", "linear_adj"):
            if key is None or key in seen:
                continue
            seen.add(key)
            yield key

    def _match_cached_entry(entry):
        if entry is None or entry.get("matrix_type") != matrix_type:
            return None, False
        cached_matrix = entry.get("matrix")
        if cached_matrix is None:
            return None, False
        structure_match_local = _csr_matrix_structure_equal(cached_matrix, scipy_matrix)
        if not structure_match_local:
            return None, False
        if _csr_matrices_equal(cached_matrix, scipy_matrix):
            return entry.get("solver"), True
        return entry.get("solver"), False

    cached_key = None
    cached_entry = None
    reusable_solver = None
    exact_matrix_match = False
    structure_match = False
    if reuse_factorization and cache is not None:
        for candidate_key in _iter_candidate_cache_keys(cache_key):
            candidate_entry = cache.get(candidate_key)
            candidate_solver, candidate_exact_match = _match_cached_entry(
                candidate_entry
            )
            if candidate_solver is None and not candidate_exact_match:
                continue
            cached_key = candidate_key
            cached_entry = candidate_entry
            reusable_solver = candidate_solver
            exact_matrix_match = candidate_exact_match
            structure_match = True
            if exact_matrix_match:
                break

    solver = None
    if exact_matrix_match:
        solver = reusable_solver

    if solver is None:
        if reuse_factorization and structure_match and reusable_solver is not None:
            solver = reusable_solver
            solver.refactor(scipy_matrix)
        else:
            if cached_entry is not None and cached_entry.get("solver") is not None:
                try:
                    cached_entry["solver"].clear()
                except Exception:
                    pass
            solver = PydisoSolver(scipy_matrix, matrix_type=matrix_type, factor=True)
        if cache is not None:
            cache[cache_key] = {
                "solver": solver,
                "matrix": scipy_matrix.copy(),
                "matrix_type": matrix_type,
            }
            if cached_key is not None and cached_key != cache_key:
                cache[cached_key] = cache[cache_key]
    elif cache is not None and cache_key is not None:
        cache[cache_key] = cached_entry

    x = solver.solve(rhs_np).reshape(-1)
    if not reuse_factorization:
        solver.clear()
    return jnp.asarray(x)


def _linear_solve(matrix, rhs, x0, solver_options, cache=None, cache_key=None):
    solver_options = solver_options or {}
    if "petsc_solver" in solver_options:
        petsc_options = solver_options["petsc_solver"]
        return _linear_petsc_solve(
            matrix,
            rhs,
            ksp_type=petsc_options.get("ksp_type", "bcgsl"),
            pc_type=petsc_options.get("pc_type", "ilu"),
        )
    if "pydiso_solver" in solver_options:
        pydiso_options = solver_options["pydiso_solver"]
        return _linear_pydiso_solve_cached(
            matrix,
            rhs,
            matrix_type=pydiso_options.get("matrix_type", "real_nonsymmetric"),
            cache=cache,
            cache_key=cache_key or "linear_fwd",
            reuse_factorization=pydiso_options.get("reuse_factorization", True),
        )
    if "spsolve_solver" in solver_options:
        return _linear_spsolve(matrix, rhs)

    jax_options = solver_options.get("jax_solver", {})
    return _linear_jax_solve(
        matrix,
        rhs,
        x0=x0,
        precond=jax_options.get("precond", True),
        method=jax_options.get("method", "bicgstab"),
        tol=jax_options.get("tol", 1e-10),
        atol=jax_options.get("atol", 1e-10),
        maxiter=jax_options.get("maxiter", 100000),
        solve_dtype=jax_options.get("solve_dtype", "float64"),
        jit=jax_options.get("jit", True),
        check_residual=jax_options.get("check_residual", True),
        residual_rtol=jax_options.get("residual_rtol", 1e-7),
        residual_atol=jax_options.get("residual_atol", 1e-9),
        cache=cache,
        cache_key=cache_key or "linear_fwd",
    )


def _prepare_forward_solver_options(solver_options):
    effective_options = dict(solver_options or {})
    dirichlet_mode = _normalize_dirichlet_mode(
        effective_options.get("dirichlet_mode", "row_elimination")
    )

    if dirichlet_mode in {"row_column_elimination", "reduced_free_dof"}:
        if "pydiso_solver" in effective_options:
            pydiso_options = dict(effective_options["pydiso_solver"])
            pydiso_options.setdefault("matrix_type", "real_symmetric_positive_definite")
            effective_options["pydiso_solver"] = pydiso_options
        if "jax_solver" in effective_options:
            jax_options = dict(effective_options["jax_solver"])
            jax_options.setdefault("method", "cg")
            jax_options.setdefault("jit", True)
            effective_options["jax_solver"] = jax_options

    return effective_options, dirichlet_mode


def _linear_incremental_solver(
    problem,
    raw_res_vec,
    bc_res_vec,
    raw_matrix,
    dofs,
    solver_options,
    cache=None,
):
    effective_solver_options, dirichlet_mode = _prepare_forward_solver_options(
        solver_options
    )
    if dirichlet_mode in {"row_column_elimination", "reduced_free_dof"}:
        A, b, free_rows, constrained_rows, delta = _prepare_reduced_increment_system(
            problem, raw_matrix, raw_res_vec, dofs
        )
        x0_full = _assign_bc(
            jnp.zeros(problem.num_total_dofs_all_vars), problem
        ) - _copy_bc(dofs, problem)
        x0 = np.asarray(x0_full)[free_rows]
        free_inc = _linear_solve(
            A,
            b,
            x0,
            effective_solver_options,
            cache=cache,
            cache_key="linear_fwd",
        )
        full_inc = np.zeros(
            problem.num_total_dofs_all_vars, dtype=np.asarray(free_inc).dtype
        )
        full_inc[free_rows] = np.asarray(free_inc)
        full_inc[constrained_rows] = delta
        return dofs + jnp.asarray(full_inc)
    else:
        A = _CSRMatrixWrapper(
            _apply_dirichlet_to_matrix(
                raw_matrix, _collect_dirichlet_data(problem)[0], dirichlet_mode
            )
        )
        b = -bc_res_vec

    x0_1 = _assign_bc(jnp.zeros(problem.num_total_dofs_all_vars), problem)
    x0_2 = _copy_bc(dofs, problem)
    x0 = x0_1 - x0_2
    inc = _linear_solve(
        A,
        b,
        x0,
        effective_solver_options,
        cache=cache,
        cache_key="linear_fwd",
    )
    return dofs + inc


def _local_solver(problem, solver_options=None, linear_cache=None):
    cached_dofs = _get_cached_warm_start(
        linear_cache,
        "warm_start_fwd",
        problem.num_total_dofs_all_vars,
    )
    if cached_dofs is None:
        dofs = jnp.zeros(problem.num_total_dofs_all_vars)
    else:
        dofs = _assign_bc(jnp.asarray(cached_dofs), problem)
    tol = 1e-6
    rel_tol = 1e-8

    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        raw_res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        bc_res_vec = _apply_bc_vec(raw_res_vec, dofs, problem)
        raw_matrix = _build_raw_scipy_matrix(problem)
        return raw_res_vec, bc_res_vec, raw_matrix

    raw_res_vec, bc_res_vec, raw_matrix = newton_update_helper(dofs)
    res_val = jnp.linalg.norm(bc_res_vec)
    res_val_initial = jnp.where(res_val == 0, 1.0, res_val)
    rel_res_val = res_val / res_val_initial

    max_iters = 25
    iteration = 0
    while (
        float(rel_res_val) > rel_tol and float(res_val) > tol and iteration < max_iters
    ):
        dofs = _linear_incremental_solver(
            problem,
            raw_res_vec,
            bc_res_vec,
            raw_matrix,
            dofs,
            solver_options,
            cache=linear_cache,
        )
        raw_res_vec, bc_res_vec, raw_matrix = newton_update_helper(dofs)
        res_val = jnp.linalg.norm(bc_res_vec)
        rel_res_val = res_val / res_val_initial
        iteration += 1

    _set_cached_warm_start(linear_cache, "warm_start_fwd", dofs)
    return problem.unflatten_fn_sol_list(dofs)


def _implicit_vjp(
    problem, sol_list, params, v_list, adjoint_solver_options, linear_cache=None
):
    def constraint_fn(dofs, params):
        problem.set_params(params)
        res_fn = _get_flatten_fn(problem.compute_residual, problem)
        res_fn = _apply_bc(res_fn, problem)
        return res_fn(dofs)

    def constraint_fn_sol_to_sol(sol_list, params):
        dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
        con_vec = constraint_fn(dofs, params)
        return problem.unflatten_fn_sol_list(con_vec)

    def get_partial_params_c_fn(sol_list):
        def partial_params_c_fn(params):
            return constraint_fn_sol_to_sol(sol_list, params)

        return partial_params_c_fn

    def get_vjp_constraint_fn_params(params, sol_list):
        partial_c_fn = get_partial_params_c_fn(sol_list)

        def vjp_linear_fn(v_list):
            _, f_vjp = jax.vjp(partial_c_fn, params)
            (val,) = f_vjp(v_list)
            return val

        return vjp_linear_fn

    problem.set_params(params)
    problem.newton_update(sol_list)
    v_vec = jax.flatten_util.ravel_pytree(v_list)[0]
    effective_solver_options, dirichlet_mode = _prepare_forward_solver_options(
        adjoint_solver_options
    )
    cached_adjoint = _get_cached_warm_start(
        linear_cache,
        "warm_start_adj",
        problem.num_total_dofs_all_vars,
    )
    if dirichlet_mode in {"row_column_elimination", "reduced_free_dof"}:
        A, reduced_v_vec, free_rows = _prepare_reduced_adjoint_system(problem, v_vec)
        reduced_x0 = None
        if cached_adjoint is not None:
            reduced_x0 = cached_adjoint[free_rows]
        free_adjoint_vec = _linear_solve(
            A,
            reduced_v_vec,
            reduced_x0,
            effective_solver_options,
            cache=linear_cache,
            cache_key="linear_adj",
        )
        adjoint_full = np.zeros(
            problem.num_total_dofs_all_vars, dtype=np.asarray(free_adjoint_vec).dtype
        )
        adjoint_full[free_rows] = np.asarray(free_adjoint_vec)
        adjoint_vec = jnp.asarray(adjoint_full)
    else:
        A = _build_scipy_matrix(problem, dirichlet_mode="row_elimination")
        adjoint_vec = _linear_solve(
            A.transpose(),
            v_vec,
            cached_adjoint,
            effective_solver_options,
            cache=linear_cache,
            cache_key="linear_adj",
        )
    _set_cached_warm_start(linear_cache, "warm_start_adj", adjoint_vec)
    vjp_linear_fn = get_vjp_constraint_fn_params(params, sol_list)
    vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
    vjp_result = jax.tree_util.tree_map(lambda x: -x, vjp_result)
    return vjp_result


def _local_ad_wrapper(
    problem, solver_options, adjoint_solver_options, linear_cache=None
):
    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        sol_list = _local_solver(problem, solver_options, linear_cache=linear_cache)
        return sol_list

    def f_fwd(params):
        sol_list = fwd_pred(params)
        return sol_list, (params, sol_list)

    def f_bwd(res, v):
        params, sol_list = res
        vjp_result = _implicit_vjp(
            problem,
            sol_list,
            params,
            v,
            adjoint_solver_options,
            linear_cache=linear_cache,
        )
        return (vjp_result,)

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred


class SteadyStateHeatProblem(_JaxFemProblem):
    def custom_init(self, neumann_values=()):
        self._neumann_values = tuple(float(v) for v in neumann_values)

    def get_tensor_map(self):
        def tensor_map(u_grad, conductivity, source):
            del source
            return conductivity * u_grad

        return tensor_map

    def get_mass_map(self):
        def mass_map(T, x, conductivity, source):
            del T
            del x
            del conductivity
            return -source

        return mass_map

    def get_surface_maps(self):
        def make_surface_map(flux: float):
            def surface_map(u, x):
                # Neumann BC convention: n . (k grad T) = flux
                return -jnp.ones_like(u) * flux

            return surface_map

        return [make_surface_map(flux) for flux in self._neumann_values]

    def set_params(self, params):
        conductivity, source = params
        self.internal_vars = [conductivity, source]


@dataclass
class SteadyStateHeatJaxRuntime:
    mesh: StructuredMesh
    dirichlet_bc: Optional[Dict[str, float]]
    neumann_bc: Optional[Dict[str, float]]
    solver_options: Dict[str, Any]
    adjoint_solver_options: Dict[str, Any]
    linear_cache: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        ensure_heat_dependencies()
        if _HEAT_IMPORT_ERROR is not None:
            raise ImportError(
                "Failed to import JAX-FEM heat backend."
            ) from _HEAT_IMPORT_ERROR

        dirichlet_bc_info = normalize_dirichlet_bc(self.dirichlet_bc, self.mesh)
        location_fns, neumann_values = normalize_neumann_bc(self.neumann_bc, self.mesh)

        self.problem = SteadyStateHeatProblem(
            mesh=self.mesh.as_problem_mesh(),
            vec=1,
            dim=self.mesh.dim,
            ele_type=self.mesh.ele_type,
            dirichlet_bc_info=dirichlet_bc_info,
            location_fns=location_fns or None,
            additional_info=(neumann_values,),
        )
        self._solve = _local_ad_wrapper(
            self.problem,
            dict(self.solver_options),
            dict(self.adjoint_solver_options),
            linear_cache=self.linear_cache,
        )
        self._num_quads = self.problem.fes[0].num_quads
        self._cells = jnp.asarray(np.asarray(self.mesh.cells))

    def solve_with_vjp(self, k_map, q_map):
        def wrapped(k_values, q_values):
            return self.solve(k_values, q_values)

        return jax.vjp(wrapped, k_map, q_map)

    def solve(self, k_map, q_map):
        conductivity = self._cellwise_param(k_map)
        source = self._cellwise_param(q_map)
        solution = self._solve((conductivity, source))[0]
        nodal_temperature = solution[:, 0]
        cell_temperature = jnp.mean(nodal_temperature[self._cells], axis=1)
        if getattr(self.mesh, "shape", ()):
            return cell_temperature.reshape(self.mesh.shape)
        return cell_temperature

    def metadata(self):
        metadata = {
            "shape": self.mesh.shape,
            "dim": self.mesh.dim,
            "ele_type": self.mesh.ele_type,
            "num_cells": int(self.mesh.cells.shape[0]),
            "num_nodes": int(self.mesh.points.shape[0]),
            "solver_options": dict(self.solver_options),
        }
        if hasattr(self.mesh, "spacing"):
            metadata["spacing"] = self.mesh.spacing
        else:
            metadata["spacing"] = None
            metadata["domain_lengths"] = tuple(
                float(v) for v in self.mesh.domain_lengths
            )
        return metadata

    def _cellwise_param(self, value_map):
        values = jnp.reshape(value_map, (-1, 1, 1))
        return jnp.broadcast_to(values, (values.shape[0], self._num_quads, 1))
