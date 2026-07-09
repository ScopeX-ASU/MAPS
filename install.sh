#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIFORGE_VERSION="26.3.2-3"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/pkgs/miniforge3}"
ENV_NAME="${ENV_NAME:-maps_test}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/environment.yaml}"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-2.9.1+cu128.html"
TORCH_SCATTER_REPO_URL="https://github.com/rusty1s/pytorch_scatter.git"
TORCH_SPARSE_REPO_URL="https://github.com/rusty1s/pytorch_sparse.git"
PYUTILITY_REPO_URL="https://github.com/JeremieMelo/pyutility.git"
TORCH_SCATTER_VERSION="2.1.2"
TORCH_SPARSE_VERSION="0.6.18"
INSTALL_OPTIONAL_3D="${INSTALL_OPTIONAL_3D:-0}"
INSTALL_OPTIONAL_APP="${INSTALL_OPTIONAL_APP:-1}"
VERIFY_OPTIONAL_3D="${VERIFY_OPTIONAL_3D:-$INSTALL_OPTIONAL_3D}"
SKIP_SYSTEM_PACKAGES="${SKIP_SYSTEM_PACKAGES:-0}"
HOST_GLIBC_VERSION=""
PYG_INSTALL_MODE="wheel"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

conda_cmd() {
  local conda_bin="$MINIFORGE_DIR/bin/conda"
  [[ -x "$conda_bin" ]] || die "Expected Conda executable not found at $conda_bin"
  "$conda_bin" "$@"
}

mamba_cmd() {
  local mamba_bin="$MINIFORGE_DIR/bin/mamba"
  [[ -x "$mamba_bin" ]] || return 1
  "$mamba_bin" "$@"
}

pkg_mgr_cmd() {
  if [[ -x "$MINIFORGE_DIR/bin/mamba" ]]; then
    mamba_cmd "$@"
  else
    conda_cmd "$@"
  fi
}

version_lt() {
  [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" != "$2" ]]
}

run_in_env() {
  pkg_mgr_cmd run --no-capture-output -n "$ENV_NAME" "$@"
}

conda_install_env_packages() {
  if [[ -x "$MINIFORGE_DIR/bin/mamba" ]]; then
    mamba_cmd install -n "$ENV_NAME" -y -c conda-forge "$@"
  elif command -v mamba >/dev/null 2>&1; then
    mamba install -n "$ENV_NAME" -y -c conda-forge "$@"
  else
    conda_cmd install -n "$ENV_NAME" -y -c conda-forge "$@"
  fi
}

pip_install() {
  log "pip install: $*"
  run_in_env python -m pip install "$@"
}

verify_import() {
  local module="$1"
  log "Verifying import: $module"
  run_in_env python -c "import ${module}; print(${module}.__name__)" >/dev/null
}

has_python_module() {
  local module_name="$1"
  run_in_env python -c "import ${module_name}" >/dev/null 2>&1
}

has_python_distribution() {
  local dist_name="$1"
  local module_name="$2"
  local expected_version="$3"

  run_in_env python - "$dist_name" "$module_name" "$expected_version" <<'PY' >/dev/null
import importlib
import sys
from importlib import metadata

dist_name, module_name, expected_version = sys.argv[1:4]

try:
    module = importlib.import_module(module_name)
    version = metadata.version(dist_name)
except Exception:
    raise SystemExit(1)

normalized = version.split('+', 1)[0]
if normalized != expected_version:
    raise SystemExit(1)

print(module.__name__, version)
PY
}

check_host_glibc() {
  local minimum_glibc="2.32"

  require_cmd getconf
  HOST_GLIBC_VERSION="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"

  if version_lt "$HOST_GLIBC_VERSION" "$minimum_glibc"; then
    PYG_INSTALL_MODE="source"
    log "Host glibc $HOST_GLIBC_VERSION is older than $minimum_glibc; torch-scatter and torch-sparse will be installed from source"
    return
  fi

  log "Host glibc $HOST_GLIBC_VERSION satisfies the PyG wheel requirement (>= $minimum_glibc)"
}

install_git_repo_from_source() {
  local repo_url="$1"
  local repo_name="$2"
  local build_dir
  local repo_dir

  require_cmd git
  build_dir="$(mktemp -d)"
  repo_dir="$build_dir/$repo_name"

  log "Cloning $repo_url into $repo_dir"
  git clone --depth 1 --recursive "$repo_url" "$repo_dir"

  if [[ -f "$repo_dir/.gitmodules" ]]; then
    log "Updating git submodules for $repo_name"
    git -C "$repo_dir" submodule update --init --recursive
  fi

  log "Installing $repo_name from source without build isolation"
  run_in_env bash -lc "cd '$repo_dir' && python -m pip install . --no-build-isolation"

  rm -rf "$build_dir"
}

install_pyutility() {
  local build_dir
  local repo_dir

  if has_python_module pyutils; then
    log "pyutils already installed and importable; skipping pyutility setup"
    return
  fi

  require_cmd git
  build_dir="$(mktemp -d)"
  repo_dir="$build_dir/pyutility"

  log "Cloning $PYUTILITY_REPO_URL into $repo_dir"
  git clone --depth 1 "$PYUTILITY_REPO_URL" "$repo_dir"

  log "Installing pyutility via setup.sh"
  run_in_env bash -lc "cd '$repo_dir' && ./setup.sh"

  rm -rf "$build_dir"
}

install_pyg_extensions() {
  if [[ "$PYG_INSTALL_MODE" == "source" ]]; then
    if has_python_distribution torch_scatter torch_scatter "$TORCH_SCATTER_VERSION"; then
      log "torch_scatter==$TORCH_SCATTER_VERSION already installed and importable; skipping source build"
    else
      install_git_repo_from_source "$TORCH_SCATTER_REPO_URL" "pytorch_scatter"
    fi

    if has_python_distribution torch_sparse torch_sparse "$TORCH_SPARSE_VERSION"; then
      log "torch_sparse==$TORCH_SPARSE_VERSION already installed and importable; skipping source build"
    else
      install_git_repo_from_source "$TORCH_SPARSE_REPO_URL" "pytorch_sparse"
    fi
    return
  fi

  log "Installing torch-scatter and torch-sparse from the PyG wheel index without build isolation"
  pip_install --no-build-isolation \
    -f "$PYG_WHEEL_URL" \
    torch-scatter \
    torch-sparse
}

ensure_system_packages() {
  if [[ "$SKIP_SYSTEM_PACKAGES" == "1" ]]; then
    log "Skipping Ubuntu system packages because SKIP_SYSTEM_PACKAGES=1"
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    log "Installing Ubuntu system packages"
    sudo apt-get update
    sudo apt-get install -y \
      build-essential \
      curl \
      git \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1
  else
    log "Skipping apt packages because apt-get is not available"
  fi
}

install_miniforge() {
  if [[ -x "$MINIFORGE_DIR/bin/conda" ]]; then
    log "Miniforge already present at $MINIFORGE_DIR"
    return
  fi

  local installer="/tmp/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"
  local url="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh"

  log "Downloading Miniforge ${MINIFORGE_VERSION}"
  mkdir -p "$(dirname "$MINIFORGE_DIR")"
  curl -L "$url" -o "$installer"

  log "Installing Miniforge into $MINIFORGE_DIR"
  bash "$installer" -b -p "$MINIFORGE_DIR"
}

init_conda() {
  log "Initializing Conda"
  # shellcheck disable=SC1091
  source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
  conda_cmd config --set auto_activate_base false
}

verify_env_identity() {
  log "Verifying target environment identity for $ENV_NAME"
  run_in_env python - "$ENV_NAME" <<'PY'
import pathlib
import sys

prefix = pathlib.Path(sys.prefix).resolve()
executable = pathlib.Path(sys.executable).resolve()
expected_env = sys.argv[1]

if prefix.name != expected_env:
    raise SystemExit(
        f"Expected conda env '{expected_env}', but sys.prefix resolved to '{prefix}' and sys.executable to '{executable}'"
    )

print('sys.prefix', prefix)
print('sys.executable', executable)
PY
}

create_or_update_env() {
  if pkg_mgr_cmd env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    log "Updating existing environment $ENV_NAME from $ENV_FILE"
    pkg_mgr_cmd env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
  else
    log "Creating environment $ENV_NAME from $ENV_FILE"
    pkg_mgr_cmd env create -n "$ENV_NAME" -f "$ENV_FILE"
  fi

  log "Ensuring required MKL/Pardiso solver packages are installed through Conda"
  conda_install_env_packages pydiso=0.2.0 mkl-devel

  verify_env_identity

  log "Verifying target Conda environment"
  run_in_env python --version
  run_in_env python -m pip --version
  run_in_env python - <<'PY'
import numpy, scipy, matplotlib, h5py, nlopt, pydiso
import meep as mp
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('matplotlib', matplotlib.__version__)
print('h5py', h5py.__version__)
print('nlopt', nlopt.__version__)
print('pydiso', pydiso.__version__)
print('meep', mp.__version__)
PY
}

install_pytorch_stack() {
  log "Installing PyTorch CUDA 12.8 wheels before Torch extensions"
  pip_install --index-url "$PYTORCH_INDEX_URL" \
    "torch==2.9.1+cu128" \
    "torchvision==0.24.1+cu128"
  verify_import torch
  verify_import torchvision
}

install_core_pip_packages() {
  log "Installing core pip packages required by the 2D FDFD and training stack"
  pip_install \
    "accelerate==1.13.0" \
    "angler==0.0.15" \
    "autograd==1.8.0" \
    "cupy-cuda12x==14.0.1" \
    "einops==0.8.0" \
    "imageio==2.37.3" \
    "kornia==0.8.1" \
    "mlflow==3.4.0" \
    "mlflow-skinny==3.4.0" \
    "mmcv-lite==2.2.0" \
    "mmengine==0.10.5" \
    "opencv-python==4.10.0.84" \
    "pymkl==0.0.3" \
    "pytest==8.3.4" \
    "pyyaml==6.0.3" \
    "ryaml==0.4.0" \
    "scikit-learn==1.5.2" \
    "tensorflow==2.20.0" \
    "tensorly==0.8.1" \
    "tensorly-torch==0.5.0" \
    "tidy3d==2.11.2" \
    "tidy3d-extras" \
    "timm==1.0.16" \
    "tqdm==4.67.3" \
    "wandb==0.22.2" \
    "build==1.5.0" \
    "pre-commit==4.6.0"

  install_pyg_extensions

  install_pyutility

  verify_import cupy
  verify_import einops
  verify_import pydiso
  verify_import mmengine
  verify_import tidy3d
  verify_import torch_scatter
  verify_import torch_sparse
  verify_import pyutils
}

install_optional_app_packages() {
  if [[ "$INSTALL_OPTIONAL_APP" != "1" ]]; then
    log "Skipping optional app and layout packages"
    return
  fi

  log "Installing optional app, geometry, and layout packages"
  pip_install \
    "gdsfactory==9.14.1" \
    "gdspy==1.6.13" \
    "gdstk==0.9.55" \
    "gradio==6.8.0" \
    "klayout==0.30.3"

  verify_import gradio
  verify_import gdstk
  verify_import klayout
}

install_optional_3d_packages() {
  if [[ "$INSTALL_OPTIONAL_3D" != "1" ]]; then
    log "Skipping optional JAX/fdtdx/heat stack"
    return
  fi

  log "Installing optional JAX, heat, Tidy3D, and fdtdx packages"
  pip_install \
    "equinox==0.13.6" \
    "fdtdx==0.6.2" \
    "fenics-basix==0.10.0" \
    "gmsh==4.15.2" \
    "jax==0.9.2" \
    "jax-cuda12-pjrt==0.9.2" \
    "jax-cuda12-plugin==0.9.2" \
    "jax-fem==0.0.11" \
    "jaxlib==0.9.2" \
    "loguru==0.7.3" \
    "meshio==5.3.5" \
    "moviepy==2.2.0" \
    "optax==0.2.8" \
    "rich==14.1.0" \
    "trimesh==4.11.5"

  verify_import jax
  verify_import fdtdx
}

install_repo_editable() {
  log "Installing this repository in editable mode"
  run_in_env bash -lc "cd '$REPO_ROOT' && python -m pip install --no-build-isolation -e ."
  run_in_env python - <<'PY'
import app
import core
import data
import drc
import thirdparty
print('editable-install-ok')
PY
}

run_verification() {
  log "Running repository verification"
  local args=()
  if [[ "$VERIFY_OPTIONAL_3D" == "1" ]]; then
    args+=("--require-optional-3d")
  fi
  if [[ "$INSTALL_OPTIONAL_APP" == "1" ]]; then
    args+=("--require-app")
  fi
  run_in_env bash -lc "cd '$REPO_ROOT' && python verify_install.py ${args[*]}"
}

main() {
  require_cmd curl
  check_host_glibc
  ensure_system_packages
  install_miniforge
  init_conda
  create_or_update_env
  install_pytorch_stack
  install_core_pip_packages
  install_optional_app_packages
  install_optional_3d_packages
  install_repo_editable
  run_verification
  log "Installation completed successfully for environment $ENV_NAME"
}

main "$@"
