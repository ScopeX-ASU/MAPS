"""
Date: 2024-10-10 21:17:31
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-11 13:55:08
FilePath: /MAPS/core/models/fdfd/utils.py
"""

"""
Date: 2024-10-10 21:17:31
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-11 02:18:01
FilePath: /MAPS/core/models/fdfd/utils.py
"""

import torch
import scipy.sparse as sp
from torch_sparse import spspmm, spmm

# def get_entries_indices(csr_matrix):
#     # takes sparse matrix and returns the entries and indeces in form compatible with 'make_sparse'
#     shape = csr_matrix.shape
#     coo_matrix = csr_matrix.tocoo()
#     entries = csr_matrix.data
#     cols = coo_matrix.col
#     rows = coo_matrix.row
#     indices = npa.vstack((rows, cols))
#     return entries, indices


def get_entries_indices(coo_matrix):
    # takes sparse matrix and returns the entries and indeces in form compatible with 'make_sparse'
    entries = coo_matrix.data
    cols = coo_matrix.col
    rows = coo_matrix.row
    indices = torch.vstack((rows, cols))
    return entries, indices


def torch_sparse_to_scipy_sparse(A):
    A = A.coalesce()
    return sp.coo_matrix(
        (A.values().cpu().numpy(), A.indices().cpu().numpy()), shape=tuple(A.shape)
    )


def real_sparse_mm(A, B):
    A = A.coalesce()
    B = B.coalesce()
    indices, values = spspmm(
        A.indices(),
        A.values(),
        B.indices(),
        B.values(),
        A.shape[0],
        A.shape[1],
        B.shape[1],
    )
    return torch.sparse_coo_tensor(indices, values, A.shape, device=A.device)


def sparse_mm(A, B):
    A = A.coalesce()
    B = B.coalesce()
    A_values = A.values()
    B_values = B.values()
    if torch.is_complex(A_values):
        A_real = A_values.real
        A_imag = A_values.imag
        B_real = B_values.real
        B_imag = B_values.imag

    indices, values_rr = spspmm(
        A.indices(),
        A_real,
        B.indices(),
        B_real,
        A.shape[0],
        A.shape[1],
        B.shape[1],
    )
    _, values_ri = spspmm(
        A.indices(),
        A_real,
        B.indices(),
        B_imag,
        A.shape[0],
        A.shape[1],
        B.shape[1],
    )
    _, values_ir = spspmm(
        A.indices(),
        A_imag,
        B.indices(),
        B_real,
        A.shape[0],
        A.shape[1],
        B.shape[1],
    )
    _, values_ii = spspmm(
        A.indices(),
        A_imag,
        B.indices(),
        B_imag,
        A.shape[0],
        A.shape[1],
        B.shape[1],
    )
    values = values_rr - values_ii + 1j * (values_ri + values_ir)
    return torch.sparse_coo_tensor(indices, values, A.shape, device=A.device)


def real_sparse_mv(A, x):
    A = A.coalesce()
    return spmm(A.indices(), A.values(), A.shape[0], A.shape[1], x[..., None]).squeeze(
        -1
    )


def sparse_mv(A, x):
    A = A.coalesce()
    A_values = A.values()
    if torch.is_complex(A_values):
        A_real = A_values.real
        A_imag = A_values.imag

    values_rr = spmm(A.indices(), A_real, A.shape[0], A.shape[1], x[..., None].real)
    values_ri = spmm(A.indices(), A_real, A.shape[0], A.shape[1], x[..., None].imag)
    values_ir = spmm(A.indices(), A_imag, A.shape[0], A.shape[1], x[..., None].real)
    values_ii = spmm(A.indices(), A_imag, A.shape[0], A.shape[1], x[..., None].imag)
    values = values_rr - values_ii + 1j * (values_ri + values_ir)
    return values.squeeze(-1)
