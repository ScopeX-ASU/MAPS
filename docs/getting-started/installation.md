# Installation and Runtime Prerequisites

This getting-started page summarizes the reproducible installation path already provided by the `v0.0.2_fdtdx` baseline. The detailed dependency rationale remains in the root `README_install.md`; this page focuses on the commands and choices a new MAPS developer needs before running the verified examples.

## Validated platform

The baseline installation documentation was validated for:

```text
OS: Linux
Python: 3.11.10
NVIDIA driver: installed and working
CUDA runtime: 12.8
```

The installation stack is GPU-oriented. Some MAPS components can run on CPU, but the verified Bending data-generation example explicitly selects a CUDA device.

## Recommended installation

From the repository root:

```bash
bash install.sh
```

The installer bootstraps Miniforge when needed, creates a Conda environment, installs the compiled Conda-managed dependencies, layers the PyTorch/Python stack, installs the repository in editable mode, and runs the verification script.

The install script defaults to:

```text
ENV_NAME=maps_test
INSTALL_OPTIONAL_3D=0
INSTALL_OPTIONAL_APP=1
```

## Activate the environment

After installation, initialize/activate the Miniforge environment using the path selected by the installer. With the defaults:

```bash
source "$HOME/pkgs/miniforge3/etc/profile.d/conda.sh"
conda activate maps_test
```

If `ENV_NAME` or `MINIFORGE_DIR` was overridden during installation, use those values instead.

## Core environment

The checked-in `environment.yml` defines the compiled/base layer and pins:

```text
python = 3.11.10
numpy = 2.4.3
scipy = 1.17.1
matplotlib = 3.10.8
h5py = 3.14.0
nlopt = 2.8.0
pydiso = 0.2.0
pypardiso = 0.4.7
```

It also installs:

```text
mkl-devel
pymeep with the mpi_mpich build selector
mpich
```

plus standard build tools such as CMake and Ninja.

Do not treat `environment.yml` as the complete installation by itself. The full installer adds the PyTorch and repository-specific pip dependencies in the required order.

## PyTorch/CUDA stack

The validated default uses:

```text
torch 2.9.1 + CUDA 12.8
torchvision 0.24.1 + CUDA 12.8
cupy-cuda12x 14.0.1
```

Torch is installed before `torch-scatter`, `torch-sparse`, and `pyutility` because those packages depend on the active Torch ABI.

## PyG extensions and glibc

The installer checks the host glibc version.

For:

```text
glibc >= 2.32
```

it uses the PyG wheel index matching Torch 2.9.1/CUDA 12.8.

For older glibc hosts, it switches `torch-scatter` and `torch-sparse` to source builds to avoid the known `GLIBC_2.32 not found` import failure.

This fallback is particularly relevant on older enterprise Linux systems.

## `pyutility`

MAPS imports utilities through the `pyutils` package. The validated installation clones:

```text
https://github.com/JeremieMelo/pyutility.git
```

and runs its `setup.sh` rather than relying on the alternate pip package path.

The root installer performs this automatically.

## Default 2D/FDFD path

The default installation is intended to support the validated 2D FDFD and training/inverse-design workflows without installing FDTD-X/JAX.

This includes the dependencies needed by the Bending tutorials:

- PyTorch/CUDA;
- Ceviche/MAPS FDFD code;
- MKL/PARDISO-related sparse solver dependencies;
- Meep/MPI dependencies used by repository workflows;
- Tidy3D dependencies used elsewhere in the default stack;
- plotting, HDF5, configuration, logging, and experiment-tracking packages.

After the default install, start with:

- [First device: Bending](first-device.md)
- [First FDFD workflow](first-fdfd-workflow.md)
- [First inverse-design run](first-inverse-design.md)

## Optional 3D FDTD-X/JAX stack

The `fdtdx` backend is optional for the default 2D FDFD path.

To install the validated optional 3D stack:

```bash
INSTALL_OPTIONAL_3D=1 bash install.sh
```

The baseline installation documentation includes versions for:

```text
fdtdx @ git+https://github.com/ymahlau/fdtdx.git@77b4bc523a8e98cb7dd388e99481a0000f71dd4d
jax = 0.9.2
jaxlib = 0.9.2
jax-cuda12-pjrt = 0.9.2
jax-cuda12-plugin = 0.9.2
jax-fem = 0.0.11
```

plus supporting mesh/optimization packages.

!!! warning "Do not install PyPI `fdtdx==0.6.2`"
    MAPS requires exact upstream commit `77b4bc523a8e98cb7dd388e99481a0000f71dd4d` (68 commits ahead of tag `v0.6.2`), even though package metadata at that commit still reports version `0.6.2`. Installing `fdtdx==0.6.2` from PyPI lacks `RectilinearGrid`, generalized mode-overlap detectors, and symmetry fixes, and will be rejected by `verify_install.py`.

Use this option before working with the documented [FDTD API](../api/fdtd.md) and [FDTD differentiable solver](../api/fdtd-solver.md).

## Optional app/layout stack

The installer enables the app/layout dependencies by default.

To skip them:

```bash
INSTALL_OPTIONAL_APP=0 bash install.sh
```

The optional stack includes packages such as Gradio, gdsfactory, gdspy, gdstk, and KLayout.

## Use a different environment name

```bash
ENV_NAME=maps_dev bash install.sh
```

Then activate:

```bash
conda activate maps_dev
```

## Skip system package installation

The installer can install common Ubuntu system dependencies with `apt-get`. On a managed machine where those packages are already provisioned or `sudo` is unavailable:

```bash
SKIP_SYSTEM_PACKAGES=1 bash install.sh
```

This does not remove the underlying system-library requirements; it only skips the installer's attempt to install them.

## Verify the installation

The installer ends by running:

```bash
python verify_install.py
```

You can rerun this manually after activating the environment.

To require the optional 3D stack:

```bash
python verify_install.py --require-optional-3d
```

Do not proceed to debugging a simulation if the corresponding installation verification fails first.

## Editable repository installation

The repository is installed with:

```bash
pip install -e .
```

so changes in the working tree are visible without reinstalling MAPS after every Python edit.

The `thirdparty/ceviche` and `thirdparty/fdtdx` code is consumed through the repository tree and does not require a separate editable installation in the validated setup.

## Installation profiles

A useful mental model is:

| Profile | Command | Intended use |
| --- | --- | --- |
| Default | `bash install.sh` | 2D FDFD, training, inverse design, app/layout |
| FDFD without app | `INSTALL_OPTIONAL_APP=0 bash install.sh` | Headless/core FDFD development |
| Full 3D | `INSTALL_OPTIONAL_3D=1 bash install.sh` | FDTD-X and JAX-based 3D/heat workflows |
| Managed host | `SKIP_SYSTEM_PACKAGES=1 bash install.sh` | System packages provisioned externally |

## Troubleshooting boundary

### `torch_scatter` / `torch_sparse` glibc errors

If an import reports:

```text
GLIBC_2.32 not found
```

the PyG wheel is incompatible with the host libc. The install script is designed to detect this and use source builds. Check the install log if that fallback did not occur.

### FDTD-X import or version provenance failures

Confirm that installation used:

```bash
INSTALL_OPTIONAL_3D=1 bash install.sh
```

or reinstall the pinned upstream commit directly (do not use PyPI `fdtdx==0.6.2`):

```bash
python -m pip install --force-reinstall --no-deps \
  "fdtdx @ git+https://github.com/ymahlau/fdtdx.git@77b4bc523a8e98cb7dd388e99481a0000f71dd4d"
```

and rerun:

```bash
python verify_install.py --require-optional-3d
```

### CUDA/device errors in the Bending data-generation example

`data/fdfd/generate_bending.py` calls `torch.cuda.set_device(gpu_id)` directly. Confirm that:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

reports a usable CUDA device before running that example.

## Source of truth

For dependency rationale, exact optional package lists, and host-specific notes, consult the root:

```text
README_install.md
```

For the actual installation procedure, `install.sh` is authoritative. If this getting-started page and the installer ever disagree about executable behavior, follow the installer and update the documentation.
