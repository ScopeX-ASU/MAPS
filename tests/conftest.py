"""Shared fixtures and test hooks for the repository-level pytest suite."""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register MAPS-specific pytest markers when the configuration is loaded."""
    config.addinivalue_line("markers", "unit: fast deterministic local contract tests")
    config.addinivalue_line("markers", "integration: multi-component workflow tests")
    config.addinivalue_line(
        "markers", "gradient: autograd or adjoint correctness tests"
    )
    config.addinivalue_line("markers", "numerical: scientific regression tests")
    config.addinivalue_line("markers", "gpu: accelerator-dependent tests")
    config.addinivalue_line("markers", "slow: intentionally expensive tests")


@pytest.fixture
def seed_everything() -> Callable[[int], None]:
    """Return a helper that seeds supported random-number generators.

    The helper always seeds Python's ``random`` module. NumPy and PyTorch are
    seeded when those optional runtime packages are importable. CUDA RNG state
    is seeded through PyTorch when CUDA is available.
    """

    def _seed(seed: int = 0) -> None:
        random.seed(seed)

        try:
            import numpy as np
        except ImportError:
            np = None
        if np is not None:
            np.random.seed(seed)

        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    return _seed


@pytest.fixture
def require_cuda() -> object:
    """Return PyTorch when CUDA is available, otherwise skip the test.

    Returns:
        object: Imported ``torch`` module with CUDA available.

    Raises:
        pytest.skip.Exception: If PyTorch is unavailable or CUDA is unavailable.
    """
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch is not installed")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    return torch
