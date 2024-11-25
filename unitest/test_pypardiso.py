'''
Date: 2024-10-10 13:30:26
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-11-16 01:16:24
FilePath: /MAPS/unitest/test_pypardiso.py
'''
import time
import pypardiso
import numpy as np
import scipy.sparse as sp
# from core.models.fdfd.solver import sparse_solve
import torch
try:
    from pyMKL import pardisoSolver
except:
    print("pyMKL not installed")
# from pyutils.general import TimerCtx


def pypardiso_spsolve_complex(A, b):
    b = (
        b.astype(np.complex128)[..., np.newaxis]
        .view(np.float64)
        .transpose()
        .reshape(-1)
    )
    A = sp.vstack((sp.hstack((A.real, -A.imag)), sp.hstack((A.imag, A.real))))
    # A = sp.rand(600, 600, density=0.05, format="csr")
    # b = np.random.rand(600)
    # print(A.shape, b.shape)
    # print(A, b)
    # print(A.dtype, b.dtype)
    x = pypardiso.spsolve(A, b)
    N = x.shape[0] // 2
    x = x[:N] + 1j * x[N : 2 * N]
    print(x.shape)
    return x

def pardiso_spsolve(A, b, mtypes=13):
    pSolve = pardisoSolver(A, mtype=mtypes) # Matrix is complex unsymmetric due to SC-PML
    pSolve.factor()
    x = pSolve.solve(b)
    pSolve.clear()
    return x

# def cupy_spsolve(A, b):
#     x = sparse_solve(A, b) # Matrix is complex unsymmetric due to SC-PML
#     return x

A = sp.rand(100, 100, density=0.5, format="csr", dtype=np.complex128)
A = A + A.T + A*0.001
# b = np.random.rand(100)
# x = pypardiso.spsolve(A, b)
# print(x)

# A = sp.rand(300, 300, density=0.005, format="csr", dtype=np.complex128)

b = np.random.rand(A.shape[0], 1).astype(np.complex128)
beg = time.time()
x_scipy = sp.linalg.spsolve(A, b, use_umfpack=True)
end = time.time()
print("Scipy Complex:", end-beg, "seconds")

beg = time.time()
x_pypardiso = pypardiso_spsolve_complex(A, b)
end = time.time()
print("Pypardiso Real:", end-beg, "seconds")
# print(x)
# print(A@x)
# print(b)
beg = time.time()
x_pardiso = pardiso_spsolve(A, b)
end = time.time()
print("Pardiso Complex:", end-beg, "seconds")

beg = time.time()
x_pardiso_sym = pardiso_spsolve(A, b, mtypes=6)
end = time.time()
print("Pardiso Complex(sym):", end-beg, "seconds")

print(x_pardiso)
print(x_pardiso_sym)
print(np.allclose(x_pardiso, x_pardiso_sym))
# Acoo = A.tocoo()

# Apt = torch.sparse_coo_tensor(A.nonzero(), A.data, A.shape)
# print(Apt)
# b = torch.from_numpy(b).to(torch.complex128)
# beg = time.time()
# x_pypardiso = cupy_spsolve(Apt, b)
# end = time.time()
# print("Cupy Real:", end-beg, "seconds")

# np.allclose(x_scipy, x_pypardiso)
# np.allclose(x_scipy, x_pardiso)