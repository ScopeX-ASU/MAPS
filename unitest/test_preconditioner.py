"""
Date: 2024-12-19 03:41:31
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-12-20 04:46:22
FilePath: /MAPS/unitest/test_preconditioner.py
"""

from core.fdfd.fdfd import fdfd_ez
import numpy as np
from thirdparty.ceviche.ceviche.constants import C_0
from pyutils.general import TimerCtx

def test():
    wl = 1.55
    omega = 2 * np.pi * C_0 / (wl * 1e-6)
    grid_step = 50
    dl = grid_step * 1e-6
    eps = np.ones((300, 300), dtype=np.complex128) + 11
    # eps[10:100,10:100] = 11
    NPML = [10, 10]
    simulation = fdfd_ez(
        omega,
        dl,
        eps,
        NPML,
        neural_solver=None,
        numerical_solver="solve_iterative",
        # numerical_solver="solve_direct",
        use_autodiff=False,
        sym_precond=True,
        # sym_precond=False,
    )
    src = np.ones_like(eps, dtype=np.complex128)

    Hx, Hy, Ez = simulation.solve(src, port_name="fwd", mode="fwd")
    with TimerCtx() as t:   
        for _ in range(3):
            Hx, Hy, Ez = simulation.solve(src, port_name="fwd", mode="fwd")
    print(t.interval/3)


test()
