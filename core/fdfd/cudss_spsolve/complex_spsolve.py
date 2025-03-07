"""
Date: 2025-03-02 22:44:08
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-03-07 00:29:08
FilePath: /MAPS/core/fdfd/cudss_spsolve/complex_spsolve.py
"""

import scipy.sparse as sp
import torch

try:
    import cudss_spsolve
except ImportError:
    print("Cannot import cudss_spsolve")


__all__ = ["spsolve_cudss"]


def spsolve_cudss(A, b, device="cuda:0", mtype: int = 0) -> torch.Tensor:
    # mtype = 0,1,2,3,4
    # CUDSS_MTYPE_GENERAL,
    # CUDSS_MTYPE_SYMMETRIC,
    # CUDSS_MTYPE_HERMITIAN,
    # CUDSS_MTYPE_SPD,
    # CUDSS_MTYPE_HPD
    ## assert device must be cuda
    assert isinstance(mtype, int) and mtype in {0, 1, 2, 3, 4}, (
        "mtype must be 0,1,2,3,4"
    )
    if isinstance(A, sp.coo_matrix):
        A = A.tocsr()

    if isinstance(A, torch.Tensor):
        if A.layout == torch.sparse_coo:
            val = A.values().to(device).to(torch.complex128)
            A = A.to_sparse_csr()
        elif A.layout == torch.sparse_csr:
            val = A.to_sparse_coo().values().to(device).to(torch.complex128)
        else:
            raise ValueError("A must be torch.sparse_coo_tensor or torch.sparse_csr_tensor")


    if isinstance(A, sp.csr_matrix):
        A = A.sorted_indices()
        dtype = A.data.dtype
        row = torch.tensor(A.indptr, dtype=torch.int32, device=device)
        col = torch.tensor(A.indices, dtype=torch.int32, device=device)
        val = torch.tensor(A.data, dtype=torch.complex128, device=device)
        b = torch.tensor(b, dtype=torch.complex128, device=device)
    elif isinstance(A, torch.Tensor) and A.layout == torch.sparse_csr:
        assert isinstance(b, torch.Tensor), "b must be torch.Tensor"
        dtype = A.dtype
        A = A.to_sparse_coo().coalesce().to_sparse_csr()
        row = A.crow_indices().to(device).to(torch.int32)
        col = A.col_indices().to(device).to(torch.int32)
        # val = A.values().to(device).to(torch.complex128)
        b = b.to(device).to(torch.complex128)
    else:
        raise ValueError("A must be scipy.sparse.csr_matrix or torch.sparse_csr_tensor")

    x = cudss_spsolve.cudss_spsolve(
        row,
        col,
        torch.view_as_real(val).flatten().contiguous(),
        torch.view_as_real(b).flatten().contiguous(),
        mtype,
    )
    x = torch.view_as_complex(x.reshape(A.shape[0], 2))
    if isinstance(A, sp.csr_matrix):
        x = x.cpu().numpy().astype(dtype)
    elif isinstance(A, torch.Tensor) and A.layout == torch.sparse_csr:
        x = x.to(A.dtype)
    return x
