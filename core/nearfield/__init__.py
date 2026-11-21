"""PyTorch near- and far-field projection utilities."""

from .accelerated import angular_spectrum_torch, project_farfield_separable
from .compatible import (
    FieldMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionData,
    FieldProjectionKSpaceMonitor,
    FieldProjectionSurface,
    FieldProjector,
)
from .dispatch import BackendDecision, project, select_backend
from .monitors import from_maps_fields
from .observations import (
    cartesian_observation_plane,
    direction_cosines,
    integrated_directional_power,
    spherical_components,
)
from .projector import (
    C_0,
    EPSILON_0,
    MU_0,
    Surface,
    concatenate_surfaces,
    direction_vectors_spherical,
    directional_power,
    make_planar_surface,
    project_farfield,
    project_nearfield,
    projection_adjoint,
)

__all__ = [
    "BackendDecision",
    "FieldMonitor",
    "FieldProjectionSurface",
    "FieldProjectionAngleMonitor",
    "FieldProjectionCartesianMonitor",
    "FieldProjectionKSpaceMonitor",
    "FieldProjectionData",
    "FieldProjector",
    "angular_spectrum_torch",
    "project_farfield_separable",
    "project",
    "select_backend",
    "from_maps_fields",
    "cartesian_observation_plane",
    "direction_cosines",
    "integrated_directional_power",
    "spherical_components",
    "C_0",
    "EPSILON_0",
    "MU_0",
    "Surface",
    "concatenate_surfaces",
    "direction_vectors_spherical",
    "directional_power",
    "make_planar_surface",
    "projection_adjoint",
    "project_farfield",
    "project_nearfield",
]
