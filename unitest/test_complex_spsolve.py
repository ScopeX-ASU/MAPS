"""
Date: 2025-03-02 22:44:08
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-03-07 00:35:42
FilePath: /MAPS/unitest/test_complex_spsolve.py
"""

import numpy as np
import scipy.sparse as sp
import torch
from ceviche.fdfd import fdfd_ez
from pyutils.general import TimerCtx
from pyutils.torch_train import set_torch_deterministic

from core.fdfd.cudss_spsolve.complex_spsolve import spsolve_cudss


def test():
    ## create csr sparse tensor tensor
    N = 10
    device = torch.device("cuda")
    dtype = torch.complex128
    set_torch_deterministic(41)
    eps_r = np.ones((N, N), dtype=np.float32)
    sim = fdfd_ez(2 * 3.14 * 3e8 / (1.55e-6), 0.05e-6, eps_r, [5, 5])
    entries_a, indices_a = sim._make_A(sim._grid_to_vec(sim.eps_r))
    A = sp.csr_matrix((entries_a, indices_a), shape=(N * N, N * N))
    A = (A + A.T) / 2
    b = torch.randn(N * N, dtype=dtype, device=device) * 1e16
    A /= 1e17
    b /= 1e17
    print(A)
    # exit(0)

    row = torch.tensor(A.indptr, dtype=torch.int32, device=device)
    col = torch.tensor(A.indices, dtype=torch.int32, device=device)
    val = torch.tensor(A.data, dtype=dtype, device=device)
    print(row, col)
    # row = torch.randint(0, N, (N+1,), dtype=torch.int32, device=device)
    # col = torch.randint(0, N, (nnz,), dtype=torch.int32, device=device)
    # val = torch.randn(nnz, dtype=torch.complex64, device=device)
    A_torch = torch.sparse_csr_tensor(
        row, col, val, size=(N * N, N * N), dtype=dtype, device=device
    )
    print(A_torch)

    x = spsolve_cudss(A_torch, b, device=device, mtype=1)

    torch.cuda.synchronize()
    with TimerCtx():
        x = spsolve_cudss(A_torch, b, device=device, mtype=1)
        torch.cuda.synchronize()

    print("x", x)
    torch.cuda.synchronize()
    b2 = A @ x.cpu().numpy()
    torch.cuda.synchronize()
    print("Ax", b2)
    print("b", b)
    b = b.cpu().numpy()
    err = np.linalg.norm(b2 - b) / np.linalg.norm(b)
    print(err)
    max_err = np.max(np.abs(b2 - b))
    print(max_err)
    assert np.allclose(b, b2, rtol=1e-2)


test()
