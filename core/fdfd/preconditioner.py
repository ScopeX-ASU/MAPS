"""
Date: 2024-11-15 23:38:50
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-12-20 04:50:46
FilePath: /MAPS/core/fdfd/preconditioner.py
"""

import numpy as np
import scipy.sparse as sp

## sc-pml and the nonuniform grid are both examples of diagonal scaling operators...we can symmetrize them both


def create_symmetrizer(sxf, syf):
    """
    #usage should be symmetrized_A = Pl@A@Pr
    https://github.com/zhaonat/py-maxwell-fd3d/blob/main/pyfd3d/preconditioner.py
    """

    sxf = sxf[:, None]
    syf = syf[None, :]

    numerator = np.sqrt((syf * sxf)).flatten()

    M = len(numerator)

    denominator = 1 / numerator

    Pl = sp.spdiags(numerator, 0, M, M)
    Pr = sp.spdiags(denominator, 0, M, M)

    return Pl, Pr
