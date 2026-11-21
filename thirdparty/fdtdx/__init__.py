"""
Date: 2026-08-21 21:35:26
LastEditors: Jiaqi Gu && jqgu@utexas.edu
LastEditTime: 2026-09-01 11:35:50
FilePath: /MAPS/thirdparty/fdtdx/__init__.py
"""

"""MAPS-specific extensions layered on top of the public :mod:`fdtdx` package.

The public FDTD-X engine remains usable when the optional adjoint source
extension is unavailable.  Optional extensions are registered only when their
imports succeed; capability flags expose their availability to MAPS runtime
code without turning forward FDTD execution into an all-or-nothing dependency.
"""

import fdtdx

from .core.physics import install_physical_power_normalization

# Install numerical overrides before importing MAPS source extensions so their
# module-level bindings use the corrected physical-area convention.
install_physical_power_normalization()

try:
    from .objects.sources.distributed import (
        CurrentSourceBlock,
        DistributedCurrentSource,
    )
except ImportError:
    CurrentSourceBlock = None
    DistributedCurrentSource = None

from .fdtd.initialization import place_objects
from .fdtd.stop_condition import DetectorDecayAfterSourceCondition
from .fdtd.symmetry import unfold_grid_slice
from .objects.sources.broadband_mode import BroadbandModePlaneSource
from .objects.sources.chebyshev import (
    ChebyshevDistributedCurrentSource,
    FactoredChebyshevCurrentSource,
    chebyshev_sample_frequencies,
)
from .objects.sources.chebyshev_mode import ChebyshevModePlaneSource
from .objects.sources.linear_polarization import GaussianPlaneSourceTidy3d
from .objects.sources.normalization import discrete_source_spectrum

if DistributedCurrentSource is not None:
    fdtdx.DistributedCurrentSource = DistributedCurrentSource

fdtdx.GaussianPlaneSourceTidy3d = GaussianPlaneSourceTidy3d
fdtdx.BroadbandModePlaneSource = BroadbandModePlaneSource
fdtdx.ChebyshevDistributedCurrentSource = ChebyshevDistributedCurrentSource
fdtdx.FactoredChebyshevCurrentSource = FactoredChebyshevCurrentSource
fdtdx.ChebyshevModePlaneSource = ChebyshevModePlaneSource
fdtdx.chebyshev_sample_frequencies = chebyshev_sample_frequencies
fdtdx.discrete_source_spectrum = discrete_source_spectrum
fdtdx.place_objects = place_objects
fdtdx.DetectorDecayAfterSourceCondition = DetectorDecayAfterSourceCondition
fdtdx.unfold_grid_slice = unfold_grid_slice

FDTDX_ADJOINT_AVAILABLE = DistributedCurrentSource is not None

__all__ = [
    "CurrentSourceBlock",
    "DistributedCurrentSource",
    "FDTDX_ADJOINT_AVAILABLE",
    "GaussianPlaneSourceTidy3d",
    "BroadbandModePlaneSource",
    "ChebyshevDistributedCurrentSource",
    "FactoredChebyshevCurrentSource",
    "ChebyshevModePlaneSource",
    "chebyshev_sample_frequencies",
    "discrete_source_spectrum",
    "place_objects",
    "DetectorDecayAfterSourceCondition",
    "unfold_grid_slice",
    "fdtdx",
]
