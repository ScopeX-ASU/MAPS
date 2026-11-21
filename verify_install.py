from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from importlib import metadata

EXPECTED_FDTDX_REPO = "https://github.com/ymahlau/fdtdx.git"
EXPECTED_FDTDX_COMMIT = "77b4bc523a8e98cb7dd388e99481a0000f71dd4d"
FDTDX_INSTALL_REMEDIATION = (
    'python -m pip install --force-reinstall --no-deps '
    f'"fdtdx @ git+{EXPECTED_FDTDX_REPO}@{EXPECTED_FDTDX_COMMIT}"'
)


@dataclass(frozen=True)
class Check:
    label: str
    module: str
    required: bool = True


def module_version(module):
    return getattr(module, "__version__", None) or getattr(module, "version", None)


def glibc_version() -> str | None:
    try:
        return os.confstr("CS_GNU_LIBC_VERSION").split()[-1]
    except Exception:
        return None


def version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def check_host_prerequisites() -> list[str]:
    current_glibc = glibc_version()
    if current_glibc is not None:
        print(f"Host glibc: {current_glibc}")
        if version_tuple(current_glibc) < version_tuple("2.32"):
            print(
                "INFO host glibc is older than 2.32; torch_scatter and torch_sparse "
                "must come from source builds instead of the prebuilt PyG wheels"
            )
    else:
        print("Host glibc: unavailable")
    return []


def run_import_checks(checks: list[Check]) -> list[str]:
    failures: list[str] = []
    for check in checks:
        try:
            module = importlib.import_module(check.module)
            version = module_version(module)
            print(f"OK   {check.label}: import {check.module} ({version})")
        except Exception as exc:
            if check.required:
                failures.append(f"{check.label}: {type(exc).__name__}: {exc}")
                print(f"FAIL {check.label}: {type(exc).__name__}: {exc}")
            else:
                print(f"SKIP {check.label}: {type(exc).__name__}: {exc}")
    return failures


def check_fdtdx_provenance(required: bool) -> list[str]:
    """Verify fdtdx comes from the pinned upstream commit rather than PyPI 0.6.2."""
    try:
        dist = metadata.distribution("fdtdx")
    except metadata.PackageNotFoundError:
        if required:
            msg = (
                "fdtdx provenance: package 'fdtdx' is not installed. "
                f"Install pinned commit with: {FDTDX_INSTALL_REMEDIATION}"
            )
            print(f"FAIL {msg}")
            return [msg]
        print("SKIP fdtdx provenance: fdtdx is not installed")
        return []

    direct_url_text = dist.read_text("direct_url.json")
    if not direct_url_text:
        msg = (
            f"fdtdx provenance: installed fdtdx ({dist.version}) has no VCS/source "
            "direct_url.json metadata and appears to be the PyPI release "
            "(fdtdx==0.6.2), which is 68 commits behind the required MAPS baseline "
            f"({EXPECTED_FDTDX_COMMIT[:8]}). Fix with:\n  {FDTDX_INSTALL_REMEDIATION}"
        )
        print(f"FAIL {msg}")
        return [msg]

    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        msg = f"fdtdx provenance: invalid direct_url.json ({exc}). Fix with:\n  {FDTDX_INSTALL_REMEDIATION}"
        print(f"FAIL {msg}")
        return [msg]

    vcs_info = direct_url.get("vcs_info") or {}
    commit_id = vcs_info.get("commit_id")
    source_desc = direct_url.get("url", "unknown")

    if commit_id is None:
        parsed = urllib.parse.urlparse(source_desc)
        if parsed.scheme == "file":
            repo_path = pathlib.Path(urllib.parse.unquote(parsed.path))
            if (repo_path / ".git").exists():
                completed = subprocess.run(
                    ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                )
                if completed.returncode == 0:
                    commit_id = completed.stdout.strip()

    if commit_id != EXPECTED_FDTDX_COMMIT:
        msg = (
            f"fdtdx provenance: expected git commit {EXPECTED_FDTDX_COMMIT}, "
            f"but found {commit_id!r} from {source_desc}. "
            "Note that PyPI release fdtdx==0.6.2 is NOT compatible even though "
            "the pinned commit still reports package version 0.6.2. Fix with:\n"
            f"  {FDTDX_INSTALL_REMEDIATION}"
        )
        print(f"FAIL {msg}")
        return [msg]

    print(
        f"OK   fdtdx provenance: verified git commit {commit_id[:12]} ({source_desc})"
    )
    return []


def run_smoke_commands(require_optional_3d: bool) -> list[str]:
    failures: list[str] = []
    commands = [
        (
            "editable import path",
            [
                sys.executable,
                "-c",
                "import core, data, gpu_subpixel, thirdparty; print('editable-ok')",
            ],
        ),
        (
            "fdfd solver import",
            [
                sys.executable,
                "-c",
                "from core.fdfd import solver; print('fdfd-solver-ok')",
            ],
        ),
        (
            "meep mpi import",
            [sys.executable, "-c", "import meep as mp; print(mp.__version__)"],
        ),
        (
            "tidy3d mode plugin import",
            [
                sys.executable,
                "-c",
                "from tidy3d.plugins.mode import ModeSolver; print(ModeSolver.__name__)",
            ],
        ),
    ]
    if require_optional_3d:
        commands.extend(
            [
                (
                    "heat solver import",
                    [
                        sys.executable,
                        "-c",
                        "from core.heat import problem; print('heat-problem-ok')",
                    ],
                ),
                (
                    "fdtdx compat and adjoint layer",
                    [
                        sys.executable,
                        "-c",
                        (
                            "import fdtdx; "
                            "from fdtdx.core.grid import RectilinearGrid; "
                            "import thirdparty.fdtdx as compat; "
                            "assert compat.FDTDX_ADJOINT_AVAILABLE; "
                            "print('fdtdx-compat-ok')"
                        ),
                    ],
                ),
            ]
        )

    for label, command in commands:
        print(f"RUN  {label}: {' '.join(command)}")
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            failures.append(
                f"{label}: exit code {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
            print(f"FAIL {label}: exit code {completed.returncode}")
        else:
            print(f"OK   {label}: {completed.stdout.strip()}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify MAPS installation.")
    parser.add_argument(
        "--require-optional-3d",
        action="store_true",
        help="Fail if the optional JAX/fdtdx/heat stack is not installed.",
    )
    parser.add_argument(
        "--require-app",
        action="store_true",
        help="Fail if the optional Gradio and layout stack is not installed.",
    )
    args = parser.parse_args()

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    checks = [
        Check("numpy", "numpy"),
        Check("scipy", "scipy"),
        Check("matplotlib", "matplotlib"),
        Check("h5py", "h5py"),
        Check("torch", "torch"),
        Check("torchvision", "torchvision"),
        Check("cupy", "cupy"),
        Check("einops", "einops"),
        Check("meep", "meep"),
        Check("nlopt", "nlopt"),
        Check("pydiso", "pydiso"),
        Check("pypardiso", "pypardiso"),
        Check("accelerate", "accelerate"),
        Check("mlflow", "mlflow"),
        Check("mmengine", "mmengine"),
        Check("mmcv-lite", "mmcv"),
        Check("multimethod", "multimethod"),
        Check("ryaml", "ryaml"),
        Check("scienceplots", "scienceplots"),
        Check("svglib", "svglib"),
        Check("pyutils", "pyutils"),
        Check("torch_scatter", "torch_scatter"),
        Check("torch_sparse", "torch_sparse"),
        Check("wandb", "wandb"),
        Check("timm", "timm"),
        Check("kornia", "kornia"),
        Check("tidy3d", "tidy3d"),
        Check("gdstk", "gdstk", required=args.require_app),
        Check("gdspy", "gdspy", required=args.require_app),
        Check("gdsfactory", "gdsfactory", required=args.require_app),
        Check("klayout", "klayout", required=args.require_app),
        Check("jax", "jax", required=args.require_optional_3d),
        Check("jaxlib", "jaxlib", required=args.require_optional_3d),
        Check("jax_fem", "jax_fem", required=args.require_optional_3d),
        Check("meshio", "meshio", required=args.require_optional_3d),
        Check("gmsh", "gmsh", required=args.require_optional_3d),
        Check("basix", "basix", required=args.require_optional_3d),
        Check("fdtdx", "fdtdx", required=args.require_optional_3d),
    ]

    failures = []
    failures.extend(check_host_prerequisites())
    failures.extend(run_import_checks(checks))
    failures.extend(check_fdtdx_provenance(args.require_optional_3d))
    failures.extend(run_smoke_commands(args.require_optional_3d))

    if failures:
        print("\nVerification failed:\n")
        for failure in failures:
            print(f"- {failure}\n")
        return 1

    print("\nVerification succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
