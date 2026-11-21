"""Dependency-provenance tests for the public FDTD-X backend."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import pytest

MANIFEST_PATH = Path("requirements-fdtdx-test.txt")


def _fdtdx_requirement() -> str:
    lines = [
        line.strip()
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    fdtdx_lines = [line for line in lines if line.startswith("fdtdx @ ")]
    assert (
        len(fdtdx_lines) == 1
    ), "Expected exactly one fdtdx VCS requirement in the manifest."
    return fdtdx_lines[0]


def _manifest_commit_id(requirement: str) -> str:
    prefix = "fdtdx @ git+https://github.com/ymahlau/fdtdx.git@"
    assert requirement.startswith(
        prefix
    ), "The FDTDX test manifest must pin the public upstream repository by commit."
    commit_id = requirement[len(prefix) :]
    assert commit_id and not any(
        char in commit_id for char in "/# "
    ), "The FDTDX manifest pin must resolve to one explicit commit/ref token."
    return commit_id


@pytest.mark.fdtdx
@pytest.mark.unit
def test_fdtdx_regression_manifest_uses_single_vcs_pin() -> None:
    """The regression environment must keep one canonical public FDTD-X pin."""
    requirement = _fdtdx_requirement()
    _manifest_commit_id(requirement)


@pytest.mark.fdtdx
@pytest.mark.unit
def test_installed_fdtdx_matches_manifest_commit() -> None:
    """The installed FDTD-X distribution must match the manifest's exact VCS commit."""
    requirement = _fdtdx_requirement()
    expected_commit = _manifest_commit_id(requirement)

    direct_url_text = metadata.distribution("fdtdx").read_text("direct_url.json")
    assert (
        direct_url_text is not None
    ), "fdtdx must be installed from the pinned VCS URL."
    direct_url = json.loads(direct_url_text)
    vcs_info = direct_url.get("vcs_info", {})

    assert vcs_info.get("vcs") == "git"
    assert vcs_info.get("commit_id") == expected_commit
