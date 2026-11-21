"""Physical-normalization overrides for the public FDTD-X package."""

from .power_normalization import install_physical_power_normalization

__all__ = ["install_physical_power_normalization"]
