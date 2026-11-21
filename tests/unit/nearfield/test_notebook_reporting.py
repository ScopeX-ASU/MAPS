"""Numerical contracts for notebook error reporting, independent of projection."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def reporting():
    """Load reporting helpers without making examples a production dependency."""
    path = Path(__file__).resolve().parents[3] / "examples/nearfield/notebook_utils.py"
    spec = importlib.util.spec_from_file_location("nearfield_notebook_reporting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_error_report_masks_null_phase_and_uses_global_scale(reporting):
    reference = np.array([[1 + 0j, 0, 0], [0, 0, 0]])
    candidate = np.array([[1j, 0, 0], [0.001j, 0, 0]])
    result = reporting.error_summary(reference, candidate)
    assert result["relative_complex_l2"] == pytest.approx(np.sqrt(2 + 1e-6))
    assert result["phase_rms_rad"] == pytest.approx(np.pi / 2)
    assert result["strong_relative_l2"] == pytest.approx(np.sqrt(2))
    assert result["weak_absolute_over_peak"] == pytest.approx(0.001)
    assert result["strong_count"] == result["weak_count"] == 1


def test_error_report_marks_undefined_phase_and_empty_regions(reporting):
    reference = np.array([[1 + 0j, 0, 0]])
    result = reporting.error_summary(reference, np.zeros_like(reference))
    assert result["phase_rms_rad"] is None
    assert result["weak_absolute_over_peak"] is None
    assert result["relative_complex_l2"] == pytest.approx(1)
