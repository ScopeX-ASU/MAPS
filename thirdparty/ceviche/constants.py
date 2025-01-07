"""
Date: 2024-11-09 00:15:23
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-06 15:45:52
FilePath: /MAPS/thirdparty/ceviche/constants.py
"""
from numpy import sqrt

"""
This file contains constants that are used throghout the codebase
"""

EPSILON_0 = 8.85418782e-12        # vacuum permittivity
MU_0 = 1.25663706e-6              # vacuum permeability
C_0 = 1 / sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum
ETA_0 = sqrt(MU_0 / EPSILON_0)    # vacuum impedance
Q_e = 1.602176634e-19             # funamental charge
MICRON_UNIT = 1e-6