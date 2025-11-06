"""
Date: 2024-10-10 19:50:23
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-03-07 00:36:41
FilePath: /MAPS/core/fdfd/solver.py
"""

import cupy as cp
import numpy as np
import scipy.sparse.linalg as spl
import torch
from pyMKL import pardisoSolver
from pyutils.general import TimerCtx, logger
from torch import Tensor

from core.utils import print_stat
from thirdparty.ceviche.constants import *
from thirdparty.ceviche.utils import make_sparse

try:
    from .pydiso_solver import MKLPardisoSolver as PydisoSolver

    HAS_PYDISO = True
except:
    HAS_PYDISO = False

try:
    # from pyMKL import pardisoSolver
    from .pardiso_solver import pardisoSolver

    HAS_MKL = True
    # print('using MKL for direct solvers')
except:
    HAS_MKL = False

ENABLE_TIMER = False


def coo_torch2cupy(A):
    A = A.data.coalesce()
    Avals_cp = cp.asarray(A.values())
    Aidx_cp = cp.asarray(A.indices())
    # return cp.sparse.csr_matrix((Avals_cp, Aidx_cp), shape=A.shape)
    return cp.sparse.coo_matrix((Avals_cp, Aidx_cp))


# ---------------------- Sparse Solver Copied from Ceviche ----------------------
DEFAULT_ITERATIVE_METHOD = "bicg"

# dict of iterative methods supported (name: function)
ITERATIVE_METHODS = {
    "bicg": spl.bicg,
    "bicgstab": spl.bicgstab,
    "cg": spl.cg,
    "cgs": spl.cgs,
    "gmres": spl.gmres,
    "lgmres": spl.lgmres,
    "qmr": spl.qmr,
    "gcrotmk": spl.gcrotmk,
}

# convergence tolerance for iterative solvers.
ATOL = 1e-8

""" ========================== SOLVER FUNCTIONS ========================== """


def _solve_iterative_with_nn_init(
    A, b, neural_solver, eps, iterative_method=DEFAULT_ITERATIVE_METHOD, **kwargs
):
    """Iterative solver with neural network initial guess"""

    # 1. Get initial guess from neural network
    if neural_solver is not None:
        # Convert b to a PyTorch tensor if it's a numpy array
        if isinstance(b, np.ndarray):
            b_torch = (
                torch.from_numpy(b).to(eps.device).to(torch.complex64)
            )  # Ensure the tensor is on the same device as eps
        else:
            b_torch = b
        x0 = neural_solver(
            eps.unsqueeze(0),  # Add batch dimension
            b_torch.reshape(eps.shape).unsqueeze(
                0
            ),  # Reshape source term to match eps shape
        )
        if isinstance(x0, tuple):
            x0 = x0[-1]  # Get final prediction if model has error correction
        # Reshape NN output to match linear system size
        x0 = x0.permute(0, 2, 3, 1).contiguous()
        x0 = torch.view_as_complex(x0)
        x0 = x0.squeeze(0).flatten()
        x0 = x0.detach().cpu().numpy()
        print("now we have the initial guess", flush=True)
    else:
        x0 = None  # Use default zero initial guess

    # 2. Get the solver function
    try:
        solver_fn = ITERATIVE_METHODS[iterative_method]
    except:
        raise ValueError(
            f"iterative method {iterative_method} not found.\n supported methods are:\n {ITERATIVE_METHODS}"
        )

    # 3. Call iterative solver with neural network initial guess
    x, info = solver_fn(A, b, x0=x0, atol=ATOL, **kwargs)

    # 4. Check convergence
    if info != 0:
        logger.warning(f"Iterative solver did not converge! Info: {info}")

    return x


# Update solve_linear to include the new solver option
def solve_linear(
    A,
    b,
    solver_type="direct",
    neural_solver=None,
    eps=None,
    iterative_method=None,
    symmetry=False,
    clear: bool = True,
    double: bool = True,
    use_pydiso: bool = False,
    **kwargs,
):
    """Master function to call different solvers"""

    if solver_type == "direct":
        return _solve_direct(
            A, b, symmetry=symmetry, clear=clear, double=double, use_pydiso=use_pydiso
        )
    elif solver_type == "iterative":
        return _solve_iterative(A, b, iterative_method=iterative_method, **kwargs)
    elif solver_type == "iterative_nn":
        return _solve_iterative_with_nn_init(
            A, b, neural_solver, eps, iterative_method=iterative_method, **kwargs
        )
    else:
        raise ValueError(f"Solver type {solver_type} not supported")


def _solve_direct(
    A, b, symmetry=False, clear: bool = True, double: bool = True, use_pydiso=False
):
    """Direct solver"""

    if HAS_MKL:
        # prefered method using MKL. Much faster (on Mac at least)
        if symmetry:
            mtype = 6
            matrix_type = "complex_symmetric"
        else:
            mtype = 13
            matrix_type = "complex_unsymmetric"
        if use_pydiso and HAS_PYDISO:
            pSolve = PydisoSolver(A, matrix_type=matrix_type, factor=False)
        elif HAS_MKL:
            pSolve = pardisoSolver(A, mtype=mtype, double=double)

        else:
            raise ImportError(
                "MKL and Pydiso both are not found, cannot use direct solver."
            )
        pSolve.factor()
        x = pSolve.solve(b).reshape(-1)

        if clear:
            pSolve.clear()
            return x, None
        else:
            return x, pSolve
    else:
        # scipy solver.
        return spl.spsolve(A, b)


def _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD, **kwargs):
    """Iterative solver"""

    # error checking on the method name (https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
    try:
        solver_fn = ITERATIVE_METHODS[iterative_method]
    except:
        raise ValueError(
            "iterative method {} not found.\n supported methods are:\n {}".format(
                iterative_method, ITERATIVE_METHODS
            )
        )

    # call the solver using scipy's API
    x, info = solver_fn(A, b, atol=ATOL, **kwargs)
    return x


def _solve_cuda(A, b, **kwargs):
    """You could put some other solver here if you're feeling adventurous"""
    raise NotImplementedError("Please implement something fast and exciting here!")


# sparse_solve_cupy = SparseSolveCupy.apply
SCALE = 1e15


def compare_As(A, B, atol=1e-5):
    ## return structure match and value match
    if A.shape != B.shape:
        return False, False

    # Compare structure: row pointers and column indices
    structure_same = np.array_equal(A.indptr, B.indptr) and np.array_equal(
        A.indices, B.indices
    )

    # If structure differs, values can't be meaningfully compared
    if not structure_same:
        return False, False

    # Compare data values
    if atol == 0.0:
        values_same = np.array_equal(A.data, B.data)
    else:
        values_same = np.allclose(A.data, B.data, atol=atol)

    return structure_same, values_same


class SparseSolveTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        entries_a,
        indices_a,
        eps_matrix: Tensor,  # for Ez, this is the diagonal of A, for Hz, this should be a sparse matrix
        omega: float,
        b: np.ndarray | Tensor,
        solver_instance,
        slice_name,
        mode,
        temp,
        Pl,
        Pr,
        neural_solver,
        numerical_solver,
        shape,
        use_autodiff=False,
        eps: Tensor | None = None,  # 2D array of eps_r, can be used in neural solver
        pol: str = "Ez",  # Ez or Hz
        double: bool = False,
        _solver_cache: dict | None = None,
        _A_cache: dict | None = None,
    ):
        ### entries_a: values of the sparse matrix A
        ### indices_a: row/column indices of the sparse matrix A
        ### eps_diag: For Ez diagonal of A, e.g., -omega**2 * epr_0 * eps_r
        ctx.use_autodiff = use_autodiff
        if use_autodiff:
            assert (
                numerical_solver == "none"
            ), f"numerical_solver {numerical_solver} is not supported when use_autodiff is True"
        Jz = b / (1j * omega)
        if isinstance(Jz, np.ndarray) and neural_solver is not None:
            if neural_solver["fwd_solver"] is not None:
                Jz = torch.from_numpy(Jz).to(neural_solver["fwd_solver"].device)
            else:
                pass  # we will not use Jz
        if Jz.dtype == torch.complex128:
            Jz = Jz.to(torch.complex64)
        if isinstance(b, Tensor):
            b = b.cpu().numpy()
        N = np.prod(shape)

        A = make_sparse(entries_a, indices_a, (N, N))

        # epsilon = torch.tensor([EPSILON_0,], dtype=eps_diag.dtype, device=eps_diag.device)
        # omega = torch.tensor([omega,], dtype=eps_diag.dtype, device=eps_diag.device)
        # eps = (eps_diag / (-EPSILON_0 * omega**2)).reshape(shape) # here there is an assumption that the eps_vec is a square matrix
        Jz = Jz.reshape(shape)

        if Pl is not None and Pr is not None:
            A = Pl @ A @ Pr
            b = Pl @ b
            symmetry = True
            ctx.precond_A = A
        else:
            symmetry = False

        if not double:
            b = (b / SCALE).astype(np.complex64)
            A = (A / SCALE).astype(np.complex64)

        if _A_cache is not None and _A_cache.get("A", None) is not None:
            structure_match, value_match = compare_As(A, _A_cache["A"], atol=1e-5)
        else:
            structure_match, value_match = False, False

        if numerical_solver == "solve_direct":
            with TimerCtx(enable=ENABLE_TIMER, desc="forward"):
                pSolve = (
                    _solver_cache.get("pSolve_fwd", None)
                    if _solver_cache is not None
                    else None
                )

                if pSolve is None or not structure_match:
                    ## if there is no cached pSolve, or the structure of A is different, we need to create a new solver
                    x, pSolve = solve_linear(
                        A,
                        b,
                        symmetry=symmetry,
                        clear=not _solver_cache and not symmetry,
                        double=double,
                        use_pydiso=HAS_PYDISO,
                    )
                    # print("[fwd] creating a new direct solver", flush=True)
                elif pSolve is not None:
                    # print("now we reuse the forward solver", flush=True)
                    if structure_match and value_match:
                        ## we share the entire solver, just solving different b
                        x = pSolve.solve(b).reshape(-1)
                    elif HAS_PYDISO and structure_match:
                        ## we refactor the matrix A to share the symbolic factorization
                        pSolve.refactor(A)
                        x = pSolve.solve(b)
                        # print("[fwd] refactoring the direct solver", flush=True)
                    else:
                        ## unexpected condition, just recreate the solver anyway
                        x, pSolve = solve_linear(
                            A,
                            b,
                            symmetry=symmetry,
                            clear=not _solver_cache and not symmetry,
                            double=double,
                            use_pydiso=HAS_PYDISO,
                        )
                else:
                    raise ValueError("Unexpected solver condition")

                if _solver_cache is not None:
                    _solver_cache["pSolve_fwd"] = pSolve
                if _A_cache is not None:
                    _A_cache["A"] = A

                # x = solve_linear(A, b, iterative_method="lgmres", symmetry=symmetry, rtol=1e-2)
            # print(f"my solve time (symmetry={symmetry}): {t.interval}")
        elif numerical_solver == "solve_direct_gpu":
            assert symmetry and not double
            with TimerCtx(enable=ENABLE_TIMER, desc="forward"):
                if not double:
                    b = (b / SCALE).astype(np.complex64)
                    A = (A / SCALE).astype(np.complex64)

                with torch.cuda.stream(torch.cuda.default_stream()):
                    A = A.tocoo().tocsr().sorted_indices()
                    x = spsolve_cudss(
                        A, b, device=eps_matrix.device, mtype=1 if symmetry else 0
                    )
                torch.cuda.synchronize()

                # print("forward x", np.max(x))
        elif numerical_solver == "none":
            assert neural_solver is not None
            # print("we are now using pure neural solver", flush=True)
            fwd_solver = neural_solver["fwd_solver"]
            x = fwd_solver(
                eps.unsqueeze(0),
                Jz.unsqueeze(0),
            )
            if isinstance(x, tuple):
                x = x[0]
            x = x.permute(0, 2, 3, 1).contiguous()
            x = torch.view_as_complex(x)
            x = x.flatten()
        elif numerical_solver == "solve_iterative":
            if neural_solver is not None:
                fwd_solver = neural_solver["fwd_solver"]
                solver_type = "iterative_nn"
            else:
                fwd_solver = None
                solver_type = "iterative"
            x = solve_linear(
                A,
                b,
                solver_type=solver_type,
                neural_solver=fwd_solver,
                eps=eps,
                iterative_method=(
                    "bicgstab" if symmetry else "lgmres"
                ),  # or other methods
                symmetry=symmetry,
                maxiter=1000,
                rtol=1e-2,
            )
        else:
            raise ValueError(f"numerical_solver {numerical_solver} not supported")
        if Pl is not None and Pr is not None:
            x = Pr @ x
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(torch.complex128).to(eps_matrix.device)
        ctx.entries_a = entries_a
        ctx.indices_a = np.flip(indices_a, axis=0)
        ctx.A = A
        ctx.save_for_backward(x, eps_matrix)
        ctx.solver_instance = solver_instance
        ctx.slice_name = slice_name
        ctx.mode = mode
        ctx.temp = temp
        ctx.Pl = Pl
        ctx.Pr = Pr
        ctx.shape = shape
        ctx.pol = pol
        ctx.adj_solver = (
            neural_solver["adj_solver"] if neural_solver is not None else None
        )
        ctx.numerical_solver = numerical_solver
        ctx.eps = eps
        ctx.omega = omega
        ctx.double = double
        ctx._solver_cache = _solver_cache
        ctx._A_cache = _A_cache
        return x

    @staticmethod
    def backward(ctx, grad):
        if ctx.use_autodiff:
            return tuple([None] * 20)
        else:
            adj_solver = ctx.adj_solver
            numerical_solver = ctx.numerical_solver
            eps = ctx.eps
            shape = ctx.shape
            x, eps_matrix = ctx.saved_tensors
            entries_a = ctx.entries_a
            indices_a = ctx.indices_a
            solver_instance = ctx.solver_instance
            slice_name = ctx.slice_name
            mode = ctx.mode
            temp = ctx.temp
            N = np.prod(shape)
            Pl, Pr = ctx.Pl, ctx.Pr

            grad = grad.cpu().numpy().astype(np.complex128)
            adj_src = grad.conj()
            if (slice_name != "Norm" and mode != "Norm" and temp != "Norm") or (
                slice_name != "adj" and mode != "adj" and temp != "adj"
            ):
                solver_instance.adj_src[(slice_name, mode, temp)] = (
                    torch.from_numpy(adj_src).to(torch.complex128).to(eps_matrix.device)
                )
            ## this adj_src = "-v" in ceviche
            # print_stat(adj_src, "my adjoint source")
            # print(f"my adjoint A_t", A_t)
            if Pl is not None and Pr is not None:
                # A_t = Pr.T @ A_t @ Pl.T
                A_t = ctx.precond_A
                adj_src = Pr.T @ adj_src
                symmetry = True
            else:
                A_t = make_sparse(entries_a, indices_a, (N, N))
                symmetry = False

            if not ctx.double:
                A_t = (A_t / SCALE).astype(np.complex64)
                adj_src = (adj_src / SCALE).astype(np.complex64)

            _A_cache = ctx._A_cache
            ## compare A_t.T with cached A
            if _A_cache is not None and _A_cache.get("A", None) is not None:
                structure_match, value_match = compare_As(A_t, _A_cache["A"], atol=1e-5)
            else:
                structure_match, value_match = False, False

            if numerical_solver == "solve_direct":
                with TimerCtx(enable=ENABLE_TIMER, desc="adjoint"):
                    if ctx._solver_cache["pSolve_fwd"] is None or not structure_match:
                        ## no cached fwd solver, or structure is different, we need to create a new adj solver
                        adj, pSolve = solve_linear(
                            A_t,
                            adj_src,
                            symmetry=symmetry,
                            clear=not ctx._solver_cache,
                            double=ctx.double,
                        )
                        # print("[bwd] creating a new direct solver", flush=True)
                        # if cache mode, we need to cache this adj solver as well

                    elif ctx._solver_cache["pSolve_fwd"] is not None:
                        if structure_match and value_match:
                            pSolve = ctx._solver_cache["pSolve_fwd"]
                            adj = pSolve.solve(adj_src).reshape(-1)
                            # print("[bwd] reusing the direct solver", flush=True)
                        elif HAS_PYDISO and structure_match:
                            pSolve = ctx._solver_cache["pSolve_fwd"]
                            pSolve.refactor(A_t)
                            adj = pSolve.solve(adj_src)
                            # print("[bwd] refactoring the direct solver", flush=True)
                        else:
                            ## unexpected condition, just recreate the solver anyway
                            adj, pSolve = solve_linear(
                                A_t,
                                adj_src,
                                symmetry=symmetry,
                                clear=not ctx._solver_cache,
                                double=ctx.double,
                            )
                    else:
                        raise ValueError("Unexpected solver condition")

                    if ctx._solver_cache is not None:
                        ctx._solver_cache["pSolve_fwd"] = pSolve
                    if _A_cache is not None:
                        _A_cache["A"] = A_t

            elif numerical_solver == "solve_direct_gpu":
                assert symmetry and not ctx.double
                with TimerCtx(enable=ENABLE_TIMER, desc="adjoint"):
                    if not ctx.double:
                        adj_src = (adj_src / SCALE).astype(np.complex64)
                        A_t = (A_t / SCALE).astype(np.complex64)

                    with torch.cuda.stream(torch.cuda.default_stream()):
                        A_t = A_t.tocoo().tocsr().sorted_indices()
                        adj = spsolve_cudss(
                            A_t,
                            adj_src,
                            device=eps_matrix.device,
                            mtype=1 if symmetry else 0,
                        )
                    torch.cuda.synchronize()

            elif numerical_solver == "none":
                assert adj_solver is not None
                # print("we are now using pure neural solver for adjoint", flush=True)
                if isinstance(adj_src, np.ndarray):
                    adj_src = (
                        torch.from_numpy(adj_src)
                        .to(torch.complex64)
                        .to(eps_matrix.device)
                    )
                adj_src = adj_src.reshape(shape)
                adj = adj_solver(
                    eps.unsqueeze(0),
                    adj_src.unsqueeze(0),
                )
                if isinstance(adj, tuple):
                    adj = adj[-1]
                adj = adj.permute(0, 2, 3, 1).contiguous()
                adj = torch.view_as_complex(adj)
                adj = adj.flatten()
                adj = adj / 1e23
            elif numerical_solver == "solve_iterative":
                adj = solve_linear(
                    A_t,
                    adj_src,
                    solver_type="iterative_nn",
                    neural_solver=adj_solver,
                    eps=eps,
                    iterative_method="lgmres",  # or other methods
                    symmetry=symmetry,
                    maxiter=1000,
                    rtol=1e-2,
                )
            else:
                raise ValueError(f"numerical_solver {numerical_solver} not supported")
            if Pl is not None and Pr is not None:
                adj = Pl.T @ adj
            if not isinstance(adj, torch.Tensor):
                adj = torch.from_numpy(adj).to(torch.complex128).to(eps_matrix.device)
            solver_instance.adj_field[(slice_name, mode, temp)] = adj

            if ctx.pol == "Ez":
                grad_epsilon = -(adj.mul_(x).real.to(eps_matrix.device))
            elif ctx.pol == "Hz":
                indices = eps_matrix.indices()
                grad_epsilon = -(
                    adj[indices[0]].mul_(x[indices[1]]).real.to(eps_matrix.device)
                )
                grad_epsilon = torch.sparse_coo_tensor(
                    indices, grad_epsilon, eps_matrix.shape
                )
            else:
                raise ValueError(f"pol {ctx.pol} not supported")
            ## this grad_epsilon = adj * x in ceviche

            # print(f"my grad eps", grad_epsilon)

            # if b.requires_grad:
            #     grad_b = solve_linear(A_t, grad)
            #     grad_b = torch.from_numpy(grad_b).to(torch.complex128).to(b.device)
            # else:
            if (slice_name != "Norm" and mode != "Norm" and temp != "Norm") or (
                slice_name != "adj" and mode != "adj" and temp != "adj"
            ):
                solver_instance.gradient[(slice_name, mode, temp)] = grad_epsilon.to(
                    torch.float32
                ).detach()
            grad_b = None
            return (
                None,
                None,
                grad_epsilon,
                None,
                grad_b,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )


class SparseSolveTorch(torch.nn.Module):
    def __init__(self, shape, neural_solver, numerical_solver, use_autodiff=False):
        super(SparseSolveTorch, self).__init__()
        self.shape = shape
        self.adj_src = (
            {}
        )  # now the adj_src is a dictionary in which the key is (port_name, mode) with same wl, different wl have different simulation objects
        self.adj_field = {}
        self.gradient = {}
        self.neural_solver = neural_solver
        self.numerical_solver = numerical_solver
        self.use_autodiff = use_autodiff
        self._solver_cache = {"pSolve_fwd": None, "pSolve_adj": None}
        self._A_cache = {"A": None}  # csr matrix

        self.set_cache_mode(False)

    def set_shape(self, shape):
        self.shape = shape

    def set_cache_mode(self, mode: bool):
        ## whether to cache the pSolve
        self._cache_mode = mode
        if not mode:
            self.clear_solver_cache()

    def clear_solver_cache(self):
        if self._solver_cache["pSolve_fwd"] is not None:
            self._solver_cache["pSolve_fwd"].clear()
            self._solver_cache["pSolve_fwd"] = None

        if self._solver_cache["pSolve_adj"] is not None:
            self._solver_cache["pSolve_adj"].clear()
            self._solver_cache["pSolve_adj"] = None
        self._A_cache = {"A": None}

    def forward(
        self,
        entries_a,
        indices_a,
        eps_matrix: Tensor,
        omega: float,
        b: np.ndarray | Tensor,
        slice_name,
        mode,
        temp,
        Pl=None,
        Pr=None,
        eps: Tensor | None = None,
        pol: str = "Ez",
        double: bool = True,
    ):
        x = SparseSolveTorchFunction.apply(
            entries_a,
            indices_a,
            eps_matrix,
            omega,
            b,
            self,
            slice_name,
            mode,
            temp,
            Pl,
            Pr,
            self.neural_solver,
            self.numerical_solver,
            self.shape,
            self.use_autodiff,
            eps,
            pol,
            double,
            self._solver_cache if self._cache_mode else None,
            self._A_cache,
        )
        return x


sparse_solve_torch = SparseSolveTorchFunction.apply
