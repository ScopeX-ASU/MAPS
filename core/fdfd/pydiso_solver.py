"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2025-11-05 21:21:42
LastEditors: Jiaqi Gu (jiaqigu@asu.edu)
LastEditTime: 2025-11-05 21:23:10
FilePath: /MAPS_local/core/fdfd/pydiso_solver.py
"""

import warnings

import numpy as np
import scipy.sparse as sp
from pydiso._mkl_solver import (
    PardisoError,
    PardisoWarning,
    _err_messages,
    _PardisoHandle_int_t,
    _PardisoHandle_long_t,
    get_mkl_int64_size,
    get_mkl_int_size,
    get_mkl_max_threads,
    get_mkl_pardiso_max_threads,
    get_mkl_version,
    set_mkl_pardiso_threads,
    set_mkl_threads,
)

MATRIX_TYPES = {
    "real_structurally_symmetric": 1,
    "real_symmetric_positive_definite": 2,
    "real_symmetric_indefinite": -2,
    "complex_structurally_symmetric": 3,
    "complex_hermitian_positive_definite": 4,
    "complex_hermitian_indefinite": -4,
    "complex_symmetric": 6,
    "real_nonsymmetric": 11,
    "complex_nonsymmetric": 13,
}
"""dict : matrix type string keys and corresponding integer value to describe
the different supported matrix types.
"""


class PardisoTypeConversionWarning(PardisoWarning, sp.SparseEfficiencyWarning):
    pass


class MKLPardisoSolver:

    def __init__(self, A, matrix_type=None, factor=True, verbose=False):
        """An interface to the Intel MKL pardiso sparse matrix solver.

        This is a solver class for a scipy sparse matrix using the Pardiso sparse
        solver in the Intel Math Kernel Library.

        It will factorize the sparse matrix in three steps: a symbolic
        factorization stage, a numerical factorization stage, and a solve stage.

        The purpose is to construct a sparse factorization that can be repeatedly
        called to solve for multiple right-hand sides.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            A sparse matrix preferably in a CSR format.
        matrix_type : str, int, or None, optional
            A string describing the matrix type, or its corresponding integer code.
            If None, then assumed to be nonsymmetric matrix.
        factor : bool, optional
            Whether to perform the factorization stage upon instantiation of the class.
        verbose : bool, optional
            Enable verbose output from the pardiso solver.

        Notes
        -----

        The supported matrix types are: real symmetric positive definite, real
        symmetric indefinite, real structurally symmetric, real nonsymmetric,
        complex hermitian positive definite, complex hermitian indefinite, complex
        symmetric, complex structurally symmetric, and complex nonsymmetric.
        The solver supports both single and double precision matrices.

        Examples
        --------

        Solve a symmetric positive definite system by first forming a simple 5 point
        laplacian stencil with a zero boundary condition. Then we create a known
        solution vector to compare to result with.

        >>> import scipy.sparse as sp
        >>> from pydiso.mkl_solver import MKLPardisoSolver
        >>> nx, ny = 5, 7
        >>> Dx = sp.diags((-1, 1), (-1, 0), (nx+1, nx))
        >>> Dy = sp.diags((-1, 1), (-1, 0), (ny+1, ny))
        >>> A = sp.kron(sp.eye(nx), Dy.T @ Dy) + sp.kron(Dx.T @ Dx, sp.eye(ny))
        >>> x = np.linspace(-10, 10, nx*ny)
        >>> b = A @ x

        Next we create the solver object using pardiso

        >>> Ainv = MKLPardisoSolver(A, matrix_type='real_symmetric_positive_definite')
        >>> x_solved = Ainv.solve(b)
        >>> np.allclose(x, x_solved)
        True
        """
        if not sp.issparse(A):
            raise TypeError(
                f"type(A)={type(A).__name__} must be a sparse array or sparse matrix."
            )

        if A.ndim != 2:
            raise ValueError(f"A.ndim={A.ndim} must be to 2.")

        n_row, n_col = A.shape
        if n_row != n_col:
            raise ValueError(f"A with shape {A.shape} is not a square matrix.")
        self.shape = n_row, n_col

        data_dtype = A.dtype
        if not (
            np.issubdtype(data_dtype, np.single)
            or np.issubdtype(data_dtype, np.double)
            or np.issubdtype(data_dtype, np.csingle)
            or np.issubdtype(data_dtype, np.cdouble)
        ):
            raise ValueError(
                f"Unrecognized matrix data type, {data_dtype}. Must be single or double precision of real or complex values."
            )
        self._data_dtype = data_dtype

        is_complex = np.issubdtype(data_dtype, np.complexfloating)

        if matrix_type is None:
            if is_complex:
                matrix_type = MATRIX_TYPES["complex_nonsymmetric"]
            else:
                matrix_type = MATRIX_TYPES["real_nonsymmetric"]

        if not (matrix_type in MATRIX_TYPES or matrix_type in MATRIX_TYPES.values()):
            raise TypeError(f"Unrecognized matrix_type: {matrix_type}")
        if matrix_type in MATRIX_TYPES:
            matrix_type = MATRIX_TYPES[matrix_type]

        if matrix_type in [1, 2, -2, 11]:
            if is_complex:
                raise ValueError(
                    f"Complex matrix dtype and matrix_type={matrix_type} are inconsistent, expected a real matrix"
                )
        else:
            if not is_complex:
                raise ValueError(
                    f"Real matrix dtype and matrix_type={matrix_type} are inconsistent, expected a complex matrix"
                )

        self.matrix_type = matrix_type

        A = self._validate_csr_matrix(A)

        max_a_ind_itemsize = max(A.indptr.itemsize, A.indices.itemsize)
        mkl_int_size = get_mkl_int_size()
        mkl_int64_size = get_mkl_int64_size()

        target_int_size = (
            mkl_int_size if max_a_ind_itemsize <= mkl_int_size else mkl_int64_size
        )
        self._ind_dtype = np.dtype(f"i{target_int_size}")

        data, indptr, indices = self._validate_matrix_dtypes(A)
        self._data = data
        self._indptr = indptr
        self._indices = indices

        if target_int_size == mkl_int_size:
            HandleClass = _PardisoHandle_int_t
        else:
            HandleClass = _PardisoHandle_long_t
        self._handle = HandleClass(
            self._data_dtype,
            self.shape[0],
            matrix_type,
            maxfct=1,
            mnum=1,
            msglvl=verbose,
        )

        self._analyze()
        self._factored = False
        if factor:
            self.factor()

    def refactor(self, A):
        """Reuse a symbolic factorization with a new matrix.

        Note
        ----
        Must have the same non-zero pattern as the initial `A` matrix.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            A sparse matrix preferably in a CSR format.
        """
        # Assumes that the matrix A has the same non-zero pattern and ordering
        # as the initial A matrix

        if not sp.issparse(A):
            raise TypeError("A is not a sparse matrix.")
        if A.shape != self.shape:
            raise ValueError("A is not the same size as the previous matrix.")

        A = self._validate_csr_matrix(A)
        data, indptr, indices = self._validate_matrix_dtypes(A)
        if len(data) != len(self._data):
            raise ValueError("new A matrix does not have the same number of non zeros.")

        self._data = data
        self.factor()

    def __call__(self, b):
        return self.solve(b)

    def solve(self, b, x=None, transpose=False):
        """Solves the equation AX=B using the factored A matrix

        Parameters
        ----------
        b : numpy.ndarray
            array of shape 1D or 2D for the right hand side of the equation
            (of the same data type as A).
        x : numpy.ndarray, optional
            A pre-allocated output array (of the same data type as A).
            If None, a new array is constructed.
        transpose : bool, optional
            If True, it will solve A^TX=B using the factored A matrix.

        Returns
        -------
        numpy.ndarray
            array containing the solution (in Fortran ordering)

        Notes
        -----
        The data will be copied if not contiguous in all cases. If multiple rhs
        are given, the input arrays will be copied if not in a contiguous
        Fortran order.
        """
        if b.dtype != self._data_dtype:
            warnings.warn(
                "rhs does not have the same data type as A",
                PardisoTypeConversionWarning,
            )
            b = b.astype(self._data_dtype)
        b = np.atleast_1d(b)
        b_was_1d = b.ndim == 1
        if b_was_1d:
            b = b[:, None]
        if b.ndim != 2:
            raise ValueError(f"b.ndim={b.ndim} must be 1 or 2.")
        if b.shape[0] != self.shape[0]:
            raise ValueError(
                f"incorrect length of b, expected {self.shape[0]}, got {b.shape[0]}"
            )
        b = np.require(b, requirements="F")

        if x is None:
            x = np.empty_like(b)
            x_was_1d = b_was_1d
        else:
            if x.dtype != self._data_dtype:
                warnings.warn(
                    "output does not have the same data type as A",
                    PardisoTypeConversionWarning,
                )
                x = x.astype(self._data_dtype)
            x = np.atleast_1d(x)
            x_was_1d = x.ndim == 1
            if x_was_1d:
                x = x[:, None]
            if x.ndim != 2:
                raise ValueError(f"x.ndim={x.ndim} must be 1 or 2.")
            if x.shape[0] != self.shape[0]:
                raise ValueError(
                    f"incorrect length of x, expected {self.shape[0]}, got {x.shape[0]}"
                )
            x = np.require(x, requirements="F")

        if b.shape[1] != x.shape[1]:
            raise ValueError(
                f"Inconsistent shapes of right hand side, {b.shape} and output vector, {x.shape}"
            )

        if x is b or (x.base is not None and (x.base is b.base)):
            raise ValueError("x and b cannot point to the same memory")

        if not self._factored:
            self.factor()

        self._handle.set_iparm(11, 2 if transpose else 0)

        phase = 33
        error = self._handle.call_pardiso(
            phase, self._data, self._indptr, self._indices, b, x
        )
        if error:
            raise PardisoError("Solve step error, " + _err_messages[error])
        if x_was_1d:
            x = x[:, 0]
        return x

    @property
    def perm(self):
        """Fill-reducing permutation vector used inside pardiso."""
        return np.array(self._handle.perm)

    @property
    def iparm(self):
        """Parameter options for the pardiso solver."""
        return np.array(self._handle.iparm)

    def _validate_csr_matrix(self, mat):
        if self.matrix_type in [-2, 2, -4, 4, 6]:
            # only grab the upper triangle.
            mat = sp.triu(mat, format="csr")

        if mat.format != "csr":
            warnings.warn(
                "Converting %s matrix to CSR format." % mat.__class__.__name__,
                PardisoTypeConversionWarning,
                stacklevel=3,
            )
            mat = mat.tocsr()

        mat.sort_indices()
        mat.sum_duplicates()
        return mat

    def _validate_matrix_dtypes(self, mat):
        data = np.require(mat.data, self._data_dtype, requirements="C")
        indptr = np.require(mat.indptr, self._ind_dtype, requirements="C")
        indices = np.require(mat.indices, self._ind_dtype, requirements="C")
        return data, indptr, indices

    def set_iparm(self, i, val):
        if i > 63 or i < 0:
            raise IndexError(f"index {i} is out of bounds for size 64 array")
        if i not in [
            1,
            3,
            4,
            5,
            7,
            9,
            10,
            11,
            12,
            17,
            18,
            20,
            23,
            24,
            26,
            30,
            33,
            34,
            35,
            36,
            38,
            42,
            55,
            59,
        ]:
            raise ValueError(f"cannot set parameter {i} of the iparm array")

        self._handle.set_iparm(i, val)

    @property
    def nnz(self):
        return self._handle.iparm[17]

    def _analyze(self):
        phase = 11
        xb_dummy = np.empty([1, 1], dtype=self._data_dtype)
        error = self._handle.call_pardiso(
            phase, self._data, self._indptr, self._indices, xb_dummy, xb_dummy
        )
        if error:
            raise PardisoError("Analysis step error, " + _err_messages[error])

    def factor(self):
        phase = 22
        self._factored = False
        xb_dummy = np.empty([1, 1], dtype=self._data_dtype)
        error = self._handle.call_pardiso(
            phase, self._data, self._indptr, self._indices, xb_dummy, xb_dummy
        )

        if error:
            raise PardisoError("Factor step error, " + _err_messages[error])

        self._factored = True

    def clear(self):
        phase = -1
        self._factored = False
        xb_dummy = np.empty([1, 1], dtype=self._data_dtype)
        error = self._handle.call_pardiso(
            phase, self._data, self._indptr, self._indices, xb_dummy, xb_dummy
        )

        if error:
            raise PardisoError("Factor step error, " + _err_messages[error])
