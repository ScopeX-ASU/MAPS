from numbers import Real
from typing import Dict, Tuple

import torch
from torch import Tensor

from .mesh import StructuredMesh, make_face_location_fn


def normalize_dirichlet_bc(bc, mesh):
    normalized = _normalize_boundary_dict(bc, mesh, "dirichlet_bc")
    if not normalized:
        raise ValueError(
            "At least one Dirichlet boundary is required for a well-posed steady-state heat solve."
        )

    location_fns = []
    vecs = []
    value_fns = []
    for face, value in normalized.items():
        location_fns.append(make_face_location_fn(face, mesh))
        vecs.append(0)
        value_fns.append(_constant_value_fn(value))
    return [location_fns, vecs, value_fns]


def normalize_neumann_bc(bc, mesh):
    normalized = _normalize_boundary_dict(bc, mesh, "neumann_bc")
    if not normalized:
        return [], ()

    location_fns = []
    values = []
    for face, value in normalized.items():
        location_fns.append(make_face_location_fn(face, mesh))
        values.append(float(value))
    return location_fns, tuple(values)


def _normalize_boundary_dict(bc, mesh, field_name):
    if bc is None:
        return {}
    if not isinstance(bc, dict):
        raise TypeError(
            f"{field_name} must be a dict mapping boundary faces to scalar values."
        )

    out = {}
    valid_faces = set(mesh.face_keys)
    for raw_face, raw_value in bc.items():
        face = str(raw_face).lower()
        if face not in valid_faces:
            raise ValueError(
                f"Unsupported boundary face {raw_face!r} for {mesh.dim}D heat solve."
            )
        out[face] = _coerce_scalar_value(raw_value, field_name, face)
    return out


def _coerce_scalar_value(value, field_name, face):
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise TypeError(
                f"{field_name}[{face!r}] only supports scalar tensors in V1."
            )
        return float(value.detach().cpu().item())
    if isinstance(value, Real):
        return float(value)
    raise TypeError(f"{field_name}[{face!r}] must be a real scalar or scalar tensor.")


def _constant_value_fn(value):
    def value_fn(point):
        return value

    return value_fn
