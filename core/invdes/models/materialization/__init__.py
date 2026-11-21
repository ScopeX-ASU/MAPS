"""Torch-native materialization helpers for differentiable device design.

The package exposes the canonical HR-to-LR resolution conversion used by the
MAPS parametrization pipeline. The conversion is differentiable and preserves
the invariant that LR material is derived from the HR realization rather than
being independently parameterized.
"""

from .resolution import convert_resolution_tensor

__all__ = ["convert_resolution_tensor"]
