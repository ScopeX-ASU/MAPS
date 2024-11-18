"""
Date: 2024-10-10 19:50:23
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-10 19:50:53
FilePath: /MAPS/core/models/fdfd/solver.py
"""

import cupy as cp
import scipy.sparse as sp
import torch
from cupyx.scipy.sparse.linalg import spsolve, factorized
from ceviche.solvers import solve_linear
import numpy as np
from torch import Tensor
from core.models.fdfd.utils import torch_sparse_to_scipy_sparse
from ceviche.utils import make_sparse
import numpy as np
from pyutils.general import print_stat
from pyMKL import pardisoSolver
from core.utils import print_stat
import scipy.sparse.linalg as spl
try:
    from pyMKL import pardisoSolver
    HAS_MKL = True
    # print('using MKL for direct solvers')
except:
    HAS_MKL = False
def coo_torch2cupy(A):
    A = A.data.coalesce()
    Avals_cp = cp.asarray(A.values())
    Aidx_cp = cp.asarray(A.indices())
    # return cp.sparse.csr_matrix((Avals_cp, Aidx_cp), shape=A.shape)
    return cp.sparse.coo_matrix((Avals_cp, Aidx_cp))

# ---------------------- Sparse Solver Copied from Ceviche ----------------------
DEFAULT_ITERATIVE_METHOD = 'bicg'

# dict of iterative methods supported (name: function)
ITERATIVE_METHODS = {
    'bicg': spl.bicg,
    'bicgstab': spl.bicgstab,
    'cg': spl.cg,
    'cgs': spl.cgs,
    'gmres': spl.gmres,
    'lgmres': spl.lgmres,
    'qmr': spl.qmr,
    'gcrotmk': spl.gcrotmk
}

# convergence tolerance for iterative solvers.
ATOL = 1e-8

""" ========================== SOLVER FUNCTIONS ========================== """

def solve_linear(A, b, iterative_method=False):
    """ Master function to call the others """

    if iterative_method and iterative_method is not None:
        # if iterative solver string is supplied, use that method
        return _solve_iterative(A, b, iterative_method=iterative_method)
    elif iterative_method and iterative_method is None:
        # if iterative_method is supplied as None, use the default
        return _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD)
    else:
        # otherwise, use a direct solver
        return _solve_direct(A, b)

def _solve_direct(A, b):
    """ Direct solver """

    if HAS_MKL:
        # prefered method using MKL. Much faster (on Mac at least)
        pSolve = pardisoSolver(A, mtype=13)
        pSolve.factor()
        x = pSolve.solve(b)
        pSolve.clear()
        return x
    else:
        # scipy solver.
        return spl.spsolve(A, b)

def _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD):
    """ Iterative solver """

    # error checking on the method name (https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
    try:
        solver_fn = ITERATIVE_METHODS[iterative_method]
    except:
        raise ValueError("iterative method {} not found.\n supported methods are:\n {}".format(iterative_method, ITERATIVE_METHODS))

    # call the solver using scipy's API
    x, info = solver_fn(A, b, atol=ATOL)
    return x

# ---------------------- Sparse Solver Copied from Ceviche ----------------------

# Custom PyTorch sparse solver exploiting a CuPy backend
# See https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
# class SparseSolveCupy(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, A, b):
#         # Sanity check
#         if A.ndim != 2 or (A.shape[0] != A.shape[1]):
#             raise ValueError("A should be a square 2D matrix.")
#         # Transfer data to CuPy
#         A_cp = coo_torch2cupy(A)
#         b_cp = cp.asarray(b.data)
#         # Solver the sparse system
#         ctx.factorisedsolver = None
#         if (b.ndim == 1) or (b.shape[1] == 1):
#             # cp.sparse.linalg.spsolve only works if b is a vector but is fully on GPU
#             x_cp = spsolve(A_cp, b_cp)
#         else:
#             # Make use of a factorisation (only the solver is then on the GPU)
#             # We store it in ctx to reuse it in the backward pass
#             ctx.factorisedsolver = factorized(A_cp)
#             x_cp = ctx.factorisedsolver(b_cp)
#         # Transfer (dense) result back to PyTorch
#         x = torch.as_tensor(x_cp, device=b.device)
#         if A.requires_grad or b.requires_grad:
#             # Not sure if the following is needed / helpful
#             x.requires_grad = True
#         else:
#             # Free up memory
#             ctx.factorisedsolver = None
#         # Save context for backward pass
#         ctx.save_for_backward(A, b, x)
#         return x

#     @staticmethod
#     def backward(ctx, grad):
#         # Recover context
#         A, b, x = ctx.saved_tensors
#         # Compute gradient with respect to b
#         if ctx.factorisedsolver is None:
#             gradb = SparseSolve.apply(A.t(), grad)
#         else:
#             # Re-use factorised solver from forward pass
#             grad_cp = cp.asarray(grad.data)
#             gradb_cp = ctx.factorisedsolver(grad_cp, trans="T")
#             gradb = torch.as_tensor(gradb_cp, device=b.device)
#         # The gradient with respect to the (dense) matrix A would be something like
#         # -gradb @ x.T but we are only interested in the gradient with respect to
#         # the (non-zero) values of A
#         gradAidx = A.indices()
#         mgradbselect = -gradb.index_select(0, gradAidx[0, :])
#         xselect = x.index_select(0, gradAidx[1, :])
#         mgbx = mgradbselect * xselect
#         if x.dim() == 1:
#             gradAvals = mgbx
#         else:
#             gradAvals = torch.sum(mgbx, dim=1)
#         gradAs = torch.sparse_coo_tensor(gradAidx, gradAvals, A.shape).to_sparse_csr()
#         # gradAs = torch.sparse_csr_tensor(gradAs)
#         return gradAs, gradb


# sparse_solve_cupy = SparseSolveCupy.apply


class SparseSolveTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, entries_a, indices_a, eps_diag: Tensor, b: np.ndarray | Tensor, solver_instance, port_name, mode):
        ### entries_a: values of the sparse matrix A
        ### indices_a: row/column indices of the sparse matrix A
        ### eps_diag: diagonal of A, e.g., -omega**2 * epr_0 * eps_r
        if isinstance(b, Tensor):
            b = b.cpu().numpy()
        A = make_sparse(entries_a, indices_a, (eps_diag.shape[0], eps_diag.shape[0]))
        # x = _solve_direct(A, b)
        x = solve_linear(A, b)

        x = torch.from_numpy(x).to(torch.complex128).to(eps_diag.device)
        ctx.entries_a = entries_a
        ctx.indices_a = np.flip(indices_a, axis=0)
        ctx.save_for_backward(x, eps_diag)
        ctx.solver_instance = solver_instance
        ctx.port_name = port_name
        ctx.mode = mode
        return x

    @staticmethod
    def backward(ctx, grad):
        x, eps_diag = ctx.saved_tensors
        entries_a = ctx.entries_a
        indices_a = ctx.indices_a
        solver_instance = ctx.solver_instance
        port_name = ctx.port_name
        mode = ctx.mode
        A_t = make_sparse(entries_a, indices_a, (eps_diag.shape[0], eps_diag.shape[0]))
        grad = grad.cpu().numpy().astype(np.complex128)
        adj_src = grad.conj()
        if (port_name != "Norm" and mode != "Norm") or (port_name != "adj" and mode != "adj"):
            solver_instance.adj_src[(port_name, mode)] = torch.from_numpy(adj_src).to(torch.complex128).to(eps_diag.device)
        ## this adj_src = "-v" in ceviche
        # print_stat(adj_src, "my adjoint source")
        # print(f"my adjoint A_t", A_t)
        adj = solve_linear(A_t, adj_src)
        adj = torch.from_numpy(adj).to(torch.complex128).to(eps_diag.device)

        grad_epsilon = -adj.mul_(x).to(eps_diag.device).real
        ## this grad_epsilon = adj * x in ceviche

        # print(f"my grad eps", grad_epsilon)

        # if b.requires_grad:
        #     grad_b = solve_linear(A_t, grad)
        #     grad_b = torch.from_numpy(grad_b).to(torch.complex128).to(b.device)
        # else:

        grad_b = None
        return None, None, grad_epsilon, grad_b, None, None, None
    
class SparseSolveTorch(torch.nn.Module):
    def __init__(self):
        super(SparseSolveTorch, self).__init__()
        self.adj_src = {} # now the adj_src is a dictionary in which the key is (port_name, mode) with same wl, different wl have different simulation objects

    def forward(self, entries_a, indices_a, eps_diag: Tensor, b: np.ndarray | Tensor, port_name, mode):
        x = SparseSolveTorchFunction.apply(entries_a, indices_a, eps_diag, b, self, port_name, mode)
        return x

sparse_solve_torch = SparseSolveTorchFunction.apply
