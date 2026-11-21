"""Fast tests that prove the repository-level pytest foundation works."""

from __future__ import annotations

import random

import pytest


@pytest.mark.unit
def test_seed_everything_is_reproducible(seed_everything) -> None:
    """Repeated seeding produces the same Python random sequence."""
    seed_everything(12345)
    first = [random.random() for _ in range(4)]

    seed_everything(12345)
    second = [random.random() for _ in range(4)]

    assert second == first


@pytest.mark.unit
def test_seed_everything_changes_sequence(seed_everything) -> None:
    """Different seeds produce different Python random sequences."""
    seed_everything(1)
    first = [random.random() for _ in range(4)]

    seed_everything(2)
    second = [random.random() for _ in range(4)]

    assert second != first


@pytest.mark.gpu
def test_cuda_fixture_smoke(require_cuda) -> None:
    """Exercise the CUDA fixture when an accelerator is available."""
    assert require_cuda.cuda.is_available()
