"""
Date: 2024-12-19 03:41:31
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-06 19:06:18
FilePath: /MAPS/unitest/test_preconditioner.py
"""

import numpy as np
from pyutils.general import TimerCtx

from core.fdfd.fdfd import fdfd_ez
from thirdparty.ceviche.constants import C_0, MICRON_UNIT


def profile(simulation, src):
    import cProfile
    import pstats

    with cProfile.Profile() as profile:
        Hx, Hy, Ez = simulation.solve(src, port_name="fwd", mode="fwd")
    profile_result = pstats.Stats(profile)
    # profile_result.sort_stats(pstats.SortKey.TIME)
    profile_result.sort_stats(pstats.SortKey.CUMULATIVE)
    profile_result.print_stats(30)


def test():
    wl = 1.55
    omega = 2 * np.pi * C_0 / (wl * MICRON_UNIT)
    grid_step = 20
    dl = grid_step * MICRON_UNIT
    eps = np.ones((600, 600), dtype=np.complex128) + 11
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

    for _ in range(3):
        Hx, Hy, Ez = simulation.solve(src, port_name="fwd", mode="fwd")
    profile(simulation, src)
    # with TimerCtx() as t:
    #     for _ in range(10):
    #         Hx, Hy, Ez = simulation.solve(src, port_name="fwd", mode="fwd")
    # print(t.interval/10)


test()
