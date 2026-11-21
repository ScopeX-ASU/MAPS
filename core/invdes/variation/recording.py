"""Fixed sparse recording envelopes for bounded fabrication variations."""

import os
from contextlib import contextmanager


@contextmanager
def sparse_recording_context(model, sampler, materializer):
    """Expand design detectors for forward and immediate adjoint evaluation.

    Args:
        model: Optimization model owning native detector profiles.
        sampler: Sampler providing theoretical parameter support.
        materializer: Geometry adapter implementing recording_bounds(support).

    Yields:
        None, with expanded design-region detector profiles installed.

    Raises:
        ValueError: If sparse geometry variation lacks finite support.
        NotImplementedError: If a custom adapter cannot bound its geometry.

    Notes:
        Ports and nominal visualization slabs remain fixed. Original profiles
        are restored on exit, including after failures. Disabled or dense
        recording follows the existing path without detector modifications.
    """
    if os.environ.get("MAPS_SPARSE_PHASOR", "0") != "1":
        yield
        return
    if callable(sampler.strategy) and sampler.config.get("scenarios") is None:
        raise ValueError(
            "Sparse recording cannot infer support from a custom sampling strategy; use explicit scenarios"
        )
    support = {
        k: v for k, v in sampler.support_bounds().items() if not k.startswith("source.")
    }
    if not support or all(lo == hi == 0 for lo, hi in support.values()):
        yield
        return
    if not hasattr(materializer, "recording_bounds"):
        raise NotImplementedError(
            "Sparse variation requires a materializer with recording_bounds"
        )
    bounds = materializer.recording_bounds(support)
    saved = []
    try:
        for name, profile in model.device.port_sources_dict.items():
            if not name.endswith("_field_monitor"):
                continue
            for key, value in list(profile.items()):
                if len(key) != 3 or key[2] not in bounds:
                    continue
                region_bounds = bounds[key[2]]
                detector = value[0].aset(
                    "partial_grid_shape", tuple(hi - lo for lo, hi in region_bounds)
                )
                updated = list(value)
                updated[0], updated[4] = detector, region_bounds
                saved.append((profile, key, value))
                profile[key] = updated
        yield
    finally:
        for profile, key, value in saved:
            profile[key] = value
