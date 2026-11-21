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

    Args:
        x_widths: One-dimensional array of effective x-cell widths. For an Ez
            formulation these are the primal-grid widths; for an Hz formulation
            they are the corresponding dual widths. Expected shape: ``[Nx]``.
        y_widths: One-dimensional array of effective y-cell widths using the
            same primal/dual convention. Expected shape: ``[Ny]``.

    Returns:
        A tuple ``(Pl, Pr)`` of SciPy sparse diagonal matrices with shape
        ``[Nx * Ny, Nx * Ny]`` and complex128 dtype. The diagonal factors are
        defined by

        .. math::

            P_l = \operatorname{diag}(\sqrt{w_x w_y}), \\
            P_r = \operatorname{diag}(1 / \sqrt{w_x w_y}).

        The transformed operator ``Pl @ A @ Pr`` is the symmetrized system
        when ``A`` was assembled using the matching effective widths.

    Raises:
        ValueError: If either width input is not one-dimensional.

    Example:
        For a uniform ``4 x 3`` grid with unit cell widths::

            Pl, Pr = create_symmetrizer(np.ones(4), np.ones(3))
            assert Pl.shape == (12, 12)
            assert Pr.shape == (12, 12)

    Notes:
        The explicit complex128 conversion is intentional because the effective
        widths can contain complex PML scaling factors.
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
