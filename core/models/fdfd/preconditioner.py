"""
Date: 2024-11-15 23:38:50
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-11-16 15:05:21
FilePath: /MAPS/core/models/fdfd/preconditioner.py
"""

import numpy as np
import scipy.sparse as sp

## sc-pml and the nonuniform grid are both examples of diagonal scaling operators...we can symmetrize them both


def create_symmetrizer(Sxb, Syb):
    """
    #usage should be symmetrized_A = Pl@A@Pr
    https://github.com/zhaonat/py-maxwell-fd3d/blob/main/pyfd3d/preconditioner.py
    """
    sxb = Sxb.flatten(order="F")
    syb = Syb.flatten(order="F")

    numerator = np.sqrt((sxb * syb))

    M = len(numerator)

    denominator = 1 / numerator

    Pl = sp.spdiags(numerator, 0, M, M)
    Pr = sp.spdiags(denominator, 0, M, M)

    return Pl, Pr
