"""
Date: 2024-11-15 23:38:50
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-12-20 04:50:46
FilePath: /MAPS/core/fdfd/preconditioner.py
"""

import numpy as np
import scipy.sparse as sp

## sc-pml and the nonuniform grid are both examples of diagonal scaling operators...we can symmetrize them both


def create_symmetrizer(x_widths, y_widths):
    """Return ``Pl`` and ``Pr`` such that ``Pl @ A @ Pr`` is symmetric.

    The widths must be the effective widths used by the relevant forward
    derivative.  They are primal widths for Ez and dual widths for Hz.
    """

    x_widths = np.asarray(x_widths, dtype=np.complex128)
    y_widths = np.asarray(y_widths, dtype=np.complex128)
    if x_widths.ndim != 1 or y_widths.ndim != 1:
        raise ValueError("grid widths must be one-dimensional")

    numerator = np.sqrt((x_widths[:, None] * y_widths[None, :])).flatten()

    M = len(numerator)

    denominator = 1 / numerator

    Pl = sp.spdiags(numerator, 0, M, M)
    Pr = sp.spdiags(denominator, 0, M, M)

    return Pl, Pr
