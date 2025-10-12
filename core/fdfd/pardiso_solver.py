from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library

standard_library.install_aliases()
from builtins import object
from ctypes import POINTER, byref, c_int, c_longlong

import numpy as np
import scipy.sparse as sp
from numpy import ctypeslib
from pyMKL import mkl_get_version, pardiso, pardisoinit

"""
mtype options
  1 -> real and structurally symmetric
  2 -> real and symmetric positive definite
 -2 -> real and symmetric indefinite
  3 -> complex and structurally symmetric
  4 -> complex and Hermitian positive definite
 -4 -> complex and Hermitian indefinite
  6 -> complex and symmetric
 11 -> real and nonsymmetric
 13 -> complex and nonsymmetric


phase options
 11 -> Analysis
 12 -> Analysis, numerical factorization
 13 -> Analysis, numerical factorization, solve, iterative refinement
 22 -> Numerical factorization
 23 -> Numerical factorization, solve, iterative refinement
 33 -> Solve, iterative refinement
331 -> like phase=33, but only forward substitution
332 -> like phase=33, but only diagonal substitution (if available)
333 -> like phase=33, but only backward substitution
  0 -> Release internal memory for L and U matrix number mnum
 -1 -> Release all internal memory for all matrices
"""


class pardisoSolver(object):
    """Wrapper class for Intel MKL Pardiso solver."""

    def __init__(self, A, mtype=11, double: bool = True, verbose=False):
        """
        Parameters
        ----------
        A : scipy.sparse.csr.csr_matrix
            sparse matrix in csr format.
        mtype : int, optional
            flag specifying the matrix type. The possible types are:

            - 1 : real and structurally symmetric (not supported)
            - 2 : real and symmetric positive definite
            - -2 : real and symmetric indefinite
            - 3 : complex and structurally symmetric (not supported)
            - 4 : complex and Hermitian positive definite
            - -4 : complex and Hermitian indefinite
            - 6 : complex and symmetric
            - 11 : real and nonsymmetric (default)
            - 13 : complex and nonsymmetric
        verbose : bool, optional
            flag for verbose output. Default is False.

        Returns
        -------
        None

        """

        self.mtype = mtype
        if mtype in [1, 3]:
            msg = "mtype = 1/3 - structurally symmetric matrices not supported"
            raise NotImplementedError(msg)
        elif mtype in [2, -2, 4, -4, 6, 11, 13]:
            pass
        else:
            msg = "Invalid mtype: mtype={}".format(mtype)
            raise ValueError(msg)

        self.n = A.shape[0]

        if mtype in [4, -4, 6, 13]:
            # Complex matrix
            if double:
                self.dtype = np.complex128
            else:
                self.dtype = np.complex64
        elif mtype in [2, -2, 11]:
            # Real matrix
            if double:
                self.dtype = np.float64
            else:
                self.dtype = np.float32
        self.ctypes_dtype = ctypeslib.ndpointer(self.dtype)

        # If A is symmetric, store only the upper triangular portion
        if mtype in [2, -2, 4, -4, 6]:
            A = sp.triu(A, format="csr")
        elif mtype in [11, 13]:
            A = A.tocsr()

        if not A.has_sorted_indices:
            A.sort_indices()

        self.a = A.data
        self.ia = A.indptr
        self.ja = A.indices

        self._MKL_a = self.a.ctypes.data_as(self.ctypes_dtype)
        self._MKL_ia = self.ia.ctypes.data_as(POINTER(c_int))
        self._MKL_ja = self.ja.ctypes.data_as(POINTER(c_int))

        # Hardcode some parameters for now...
        self.maxfct = 1
        self.mnum = 1
        self.perm = 0

        if verbose:
            self.msglvl = 1
        else:
            self.msglvl = 0

        # Initialize handle to data structure
        self.pt = np.zeros(64, np.int64)
        self._MKL_pt = self.pt.ctypes.data_as(POINTER(c_longlong))

        # Initialize parameters
        self.iparm = np.zeros(64, dtype=np.int32)
        self._MKL_iparm = self.iparm.ctypes.data_as(POINTER(c_int))

        # Initialize pardiso
        pardisoinit(self._MKL_pt, byref(c_int(self.mtype)), self._MKL_iparm)

        verstring = mkl_get_version()
        # Set iparm
        if "11.3.3" in verstring:
            self.iparm[1] = 0
        else:
            self.iparm[1] = 3  # Use parallel nested dissection for reordering
        self.iparm[23] = 1  # Use parallel factorization
        self.iparm[34] = 1  # Zero base indexing
        if double:
            self.iparm[27] = 0
        else:
            self.iparm[27] = 1

    def clear(self):
        """
        Clear the memory allocated from the solver.
        """
        self.run_pardiso(phase=-1)

    def factor(self):
        out = self.run_pardiso(phase=12)

    def solve(self, rhs, transposed: bool = False):
        x = self.run_pardiso(phase=33, rhs=rhs, transposed=transposed)
        return x

    def run_pardiso(self, phase, rhs=None, transposed: bool = False):
        """
        Run specified phase of the Pardiso solver.

        Parameters
        ----------
        phase : int
            Flag setting the analysis type of the solver:

            -  11 : Analysis
            -  12 : Analysis, numerical factorization
            -  13 : Analysis, numerical factorization, solve, iterative refinement
            -  22 : Numerical factorization
            -  23 : Numerical factorization, solve, iterative refinement
            -  33 : Solve, iterative refinement
            - 331 : like phase=33, but only forward substitution
            - 332 : like phase=33, but only diagonal substitution (if available)
            - 333 : like phase=33, but only backward substitution
            -   0 : Release internal memory for L and U matrix number mnum
            -  -1 : Release all internal memory for all matrices
        rhs : ndarray, optional
            Right hand side of the equation `A x = rhs`. Can either be a vector
            (array of dimension 1) or a matrix (array of dimension 2). Default
            is None.

        Returns
        -------
        x : ndarray
            Solution of the system `A x = rhs`, if `rhs` is provided. Is either
            a vector or a column matrix.

        """

        if rhs is None:
            nrhs = 0
            x = np.zeros(1)
            rhs = np.zeros(1)
        else:
            if rhs.ndim == 1:
                nrhs = 1
            elif rhs.ndim == 2:
                nrhs = rhs.shape[1]
            else:
                msg = (
                    "Right hand side must either be a 1 or 2 dimensional "
                    + "array. Higher order right hand sides are not supported."
                )
                raise NotImplementedError(msg)
            rhs = rhs.astype(self.dtype).flatten(order="f")
            x = np.zeros(nrhs * self.n, dtype=self.dtype)

        if transposed:
            self._MKL_iparm[11] = 2
        else:
            self._MKL_iparm[11] = 0
        MKL_rhs = rhs.ctypes.data_as(self.ctypes_dtype)
        MKL_x = x.ctypes.data_as(self.ctypes_dtype)
        ERR = 0

        pardiso(
            self._MKL_pt,  # pt
            byref(c_int(self.maxfct)),  # maxfct
            byref(c_int(self.mnum)),  # mnum
            byref(c_int(self.mtype)),  # mtype
            byref(c_int(phase)),  # phase
            byref(c_int(self.n)),  # n
            self._MKL_a,  # a
            self._MKL_ia,  # ia
            self._MKL_ja,  # ja
            byref(c_int(self.perm)),  # perm
            byref(c_int(nrhs)),  # nrhs
            self._MKL_iparm,  # iparm
            byref(c_int(self.msglvl)),  # msglvl
            MKL_rhs,  # b
            MKL_x,  # x
            byref(c_int(ERR)),
        )  # error

        if nrhs > 1:
            x = x.reshape((self.n, nrhs), order="f")
        return x
