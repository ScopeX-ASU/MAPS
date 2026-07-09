# Reproducible Installation for MAPS_fdtdx

This document describes the installation procedure that was derived from the repository itself and from the currently working `python311venv` environment referenced by the repository owner. It is split into a reproducible default path for the 2D FDFD and training workflows, plus optional packages for the 3D `fdtdx`, JAX heat-solver, and Gradio app workflows.

## Scope and assumptions

- OS: Linux
- NVIDIA driver: already installed and working
- CUDA runtime: 12.8 already installed and working
- Starting point: no Conda, no Python packages
- Repository root: this directory

The current validation host reports `glibc 2.28`. On that host, the prebuilt PyG wheels fail to import with `GLIBC_2.32 not found`, so the installer now falls back to source builds for the PyG extensions.

## Files generated for installation

- `install.sh`: end-to-end bootstrap and installation script
- `environment.yml`: Conda environment definition for the compiled Conda-managed base stack
- `verify_install.py`: import and runtime smoke test
- `pyproject.toml`: minimal packaging metadata so `pip install -e .` works

## What was inspected

The dependency set below was derived from these sources in this repository:

- `requirements.txt`
- `environment.yml` from the existing working environment export
- `app/requirements-app.txt`
- `README.md` and `app/README.md`
- direct imports under `app/`, `core/`, `data/`, `drc/`, top-level scripts, and `thirdparty/`
- `core/fdfd/cudss_spsolve/setup.py`

No root `pyproject.toml`, `setup.py`, `setup.cfg`, `Dockerfile`, or CI workflow existed before this change.

## Required Python version

- Validated working version: `Python 3.11.10`
- Evidence:
  - repository `environment.yml` pinned `python=3.11.10`
  - user-provided working interpreter was `/home/jiaqigu/pkgs/miniforge3/envs/python311venv/bin/python`
  - the current working environment reports `Python 3.11.10`

## Dependency classification

### Conda-installed packages

These are installed through Conda because they either ship compiled libraries or have a more reliable MPI/compiled distribution path there:

- `python=3.11.10`
- `numpy=2.4.3`
- `scipy=1.17.1`
- `matplotlib=3.10.8`
- `h5py=3.14.0`
- `ipython=8.37.0`
- `pillow=11.3.0`
- `nlopt=2.8.0`
- `mkl-devel`
- `pymeep=*=mpi_mpich_*`
- `mpich`
- `pydiso=0.2.0`
- `pypardiso=0.4.7`
- build helpers: `pip`, `setuptools`, `wheel`, `cmake`, `ninja`, `pkg-config`

Why `pymeep` is Conda-managed:

- the repository uses `meep` in multiple simulation paths
- the existing working environment resolves to `meep 1.33.0`
- the user explicitly created the working environment with `pymeep=*=mpi_mpich_*`
- the MPI-enabled Conda build is the most concrete evidence in this repository for a working Meep installation
- the exact `pymeep=1.33.0=*_mpi_mpich_*` build string was not solvable from the current `conda-forge` metadata during validation, so the generated environment file keeps the user-provided MPI selector and relies on verification to confirm the resolved Meep version

### Pip-installed core packages

These are required for the default validated 2D FDFD plus training and inference stack:

- `torch==2.9.1+cu128`
- `torchvision==0.24.1+cu128`
- `accelerate==1.13.0`
- `angler==0.0.15`
- `autograd==1.8.0`
- `cupy-cuda12x==14.0.1`
- `einops==0.8.0`
- `imageio==2.37.3`
- `kornia==0.8.1`
- `mlflow==3.4.0`
- `mlflow-skinny==3.4.0`
- `mmcv-lite==2.2.0`
- `mmengine==0.10.5`
- `multimethod>=1.12`
- `opencv-python==4.10.0.84`
- `pymkl==0.0.3`
- `pytest==8.3.4`
- `pyyaml==6.0.3`
- `scienceplots==2.1.1`
- `ryaml==0.4.0`
- `scikit-learn==1.5.2`
- `svglib==1.5.1`
- `tensorflow==2.20.0`
- `tensorly==0.8.1`
- `tensorly-torch==0.5.0`
- `tidy3d==2.11.2`
- `tidy3d-extras`
- `timm==1.0.16`
- `torch-scatter` installed from `https://data.pyg.org/whl/torch-2.9.1+cu128.html` with `--no-build-isolation`, or built from source from `https://github.com/rusty1s/pytorch_scatter.git` on older `glibc` hosts
- `torch-sparse` installed from `https://data.pyg.org/whl/torch-2.9.1+cu128.html` with `--no-build-isolation`, or built from source from `https://github.com/rusty1s/pytorch_sparse.git` on older `glibc` hosts
- `pyutility` installed by cloning `https://github.com/JeremieMelo/pyutility.git` and running `./setup.sh`
- `tqdm==4.67.3`
- `wandb==0.22.2`

Why the install order matters:

1. `torch` and `torchvision` must be installed before `torch-scatter`, `torch-sparse`, and `pyutility` because those packages depend on the active Torch ABI.
2. `torch-scatter` and `torch-sparse` are installed after Torch and without build isolation. On hosts with `glibc >= 2.32`, the installer uses the PyG wheel index `https://data.pyg.org/whl/torch-2.9.1+cu128.html`. On older hosts, it clones `pytorch_scatter` and `pytorch_sparse` from GitHub and installs them from source.
3. `pymeep` must be created in the Conda environment before pip packages are layered on top, because it pulls the MPI-enabled binary distribution.
4. `pyutility` must be installed by cloning `https://github.com/JeremieMelo/pyutility.git` and running `./setup.sh`, per the repository owner’s instruction, because `pip install torchonn-pyutils --no-build-isolation` does not work properly.
5. `pydiso` is installed through Conda with `mkl-devel`, matching the validated fix `mamba install pydiso --channel conda-forge mkl-devel`, because the clean-machine pip build failed with missing `mkl-sdl` metadata.

### Optional pip packages for the app and layout workflows

Installed by default in `install.sh` unless `INSTALL_OPTIONAL_APP=0` is set:

- `gdsfactory==9.14.1`
- `gdspy==1.6.13`
- `gdstk==0.9.55`
- `gradio==6.8.0`
- `klayout==0.30.3`

### Optional pip packages for the 3D `fdtdx` and JAX heat workflows
*Currently depend on private internal fdtdx-adjoint library to support adjoint 3D FDTD inverse design*
*Public fdtdx does not support adjoint solve*
Installed only when `INSTALL_OPTIONAL_3D=1`:

- `fdtdx==0.6.2`
- `jax==0.9.2`
- `jaxlib==0.9.2`
- `jax-cuda12-pjrt==0.9.2`
- `jax-cuda12-plugin==0.9.2`
- `jax-fem==0.0.11`
- `meshio==5.3.5`
- `gmsh==4.15.2`
- `fenics-basix==0.10.0`
- supporting exact versions used in the working environment: `equinox==0.13.6`, `loguru==0.7.3`, `moviepy==2.2.0`, `optax==0.2.8`, `rich==14.1.0`, `trimesh==4.11.5`

The repository owner explicitly noted that `fdtdx` is not required for the 2D FDFD path. `tidy3d` and `tidy3d-extras` are treated as part of the default install because they are also used in the 2D workflows.

## MPI libraries

- Required by Meep: `mpich`
- Installed through Conda alongside `pymeep=*_mpi_mpich_*`

## CUDA-dependent libraries

- `torch==2.9.1+cu128`
- `torchvision==0.24.1+cu128`
- `cupy-cuda12x==14.0.1`
- `pydiso==0.2.0` via Conda together with `mkl-devel`
- default path: `tidy3d==2.11.2`, `tidy3d-extras`
- optional 3D path: `jax-cuda12-pjrt==0.9.2`, `jax-cuda12-plugin==0.9.2`, `fdtdx==0.6.2`

## External system libraries

Installed by `install.sh` through `apt`:

- `build-essential`
- `curl`
- `git`
- `libgl1`
- `libglib2.0-0`
- `libsm6`
- `libxext6`
- `libxrender1`

These are included because OpenCV, plotting, and wheel fallback builds commonly require them on clean Ubuntu systems.

## Host runtime prerequisite

- The validated PyG wheels from `https://data.pyg.org/whl/torch-2.9.1+cu128.html` import successfully only on systems with `glibc >= 2.32`.
- When `install.sh` detects an older host `glibc`, it switches to source installs for `pytorch_scatter` and `pytorch_sparse` instead of failing immediately.
- This keeps the Torch ABI constraint intact while avoiding the known `GLIBC_2.32 not found` wheel import failure on older enterprise Linux bases.

## Editable local packages

- Root repository: installed with `pip install -e .`
- No separate editable install is required for `thirdparty/ceviche` or `thirdparty/PreFab`; they are imported through the repository package tree and become available through the root editable install.

## Local compiled extensions

No local C/CUDA extension build is required for the generated installation path.

## Needs confirmation

The following items were present in code or environment exports, but they are not included in the default installation path because the repository evidence was incomplete or the workflow is clearly optional:

- `sax==0.16.4`: imported only in circuit-simulation code; no direct installation instructions in the repo
- `petsc4py`: imported only in limited heat-related code paths; no pinned version found in the inspected manifests
- `neuraloperator`: present in the working environment export, but the code imports `neuralop`; not needed for the validated default path
- any alternate non-Conda installation path for `pydiso==0.2.0`: the validated clean-machine fix was to install it with `mamba install pydiso --channel conda-forge mkl-devel`

## Step-by-step usage

Run the full default installation:

```bash
bash install.sh
```

Run with the optional 3D stack enabled:

```bash
INSTALL_OPTIONAL_3D=1 bash install.sh
```

Skip the Gradio app and layout stack:

```bash
INSTALL_OPTIONAL_APP=0 bash install.sh
```

Use a different environment name:

```bash
ENV_NAME=maps_test bash install.sh
```

## Verification

The installer ends by running:

```bash
python verify_install.py
```

To require the optional 3D stack during verification:

```bash
python verify_install.py --require-optional-3d
```

If verification fails on `torch_scatter` or `torch_sparse` with `GLIBC_2.32 not found`, the source-build fallback either did not run or did not complete successfully.
