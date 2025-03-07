"""
Date: 2025-03-02 22:41:36
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-03-02 23:48:46
FilePath: /MAPS/core/fdfd/cudss_spsolve/setup.py
"""
import os
from distutils import log
from distutils.dir_util import remove_tree

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension
from torchonn import __version__

here = os.path.abspath(os.path.dirname(__file__))


tokens = str(torch.__version__).split(".")
torch_major_version = "-DTORCH_MAJOR_VERSION=%d" % (int(tokens[0]))
torch_minor_version = "-DTORCH_MINOR_VERSION=%d" % (int(tokens[1]))


def add_prefix(filename):
    return os.path.join(".", filename)

CUDSS_DIR = os.environ.get("CUDSS_DIR", None)
if CUDSS_DIR is None:
    raise RuntimeError("CUDSS_DIR environment variable is not set.")
else:
    print(f"CUDSS_DIR: {CUDSS_DIR}")

ext_modules = []
extra_compile_args = {
    "cxx": ["-g", "-fopenmp", "-O2", torch_major_version, torch_minor_version],
    "nvcc": [
        "-O3",
        "-arch=sm_60",
        "-gencode=arch=compute_60,code=sm_60",
        "-gencode=arch=compute_61,code=sm_61",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_75,code=compute_75",
        "-gencode=arch=compute_86,code=compute_86",
        "--use_fast_math",
        f"-I{CUDSS_DIR}/include",
        f"-L{CUDSS_DIR}/lib64",
        "-lcudss",
        # f"-I${CUDSS_DIR}/include -L${CUDSS_DIR}/lib -l:libcudss.so.0",
    ],
}

if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        "cudss_spsolve",
        [
            add_prefix("complex_spsolve.cpp"),
            add_prefix("complex_spsolve_cuda_kernel.cu"),
        ],
        extra_compile_args=extra_compile_args,
        libraries=["cudss"],  # Explicitly link cuDSS
        library_dirs=[f"{CUDSS_DIR}/lib64"],  # Ensure linker finds libcudss.so
        include_dirs=[f"{CUDSS_DIR}/include"],
    )
    ext_modules.append(extension)

setup(
    name="cudss_spsolve",
    version=__version__,
    description="Pytorch-centric Optical Neural Network Library",
    long_description_content_type="text/markdown",
    url="https://github.com/JeremieMelo/pytorch-onn",
    author="Jiaqi Gu",
    author_email="jqgu@utexas.edu",
    license="MIT",
    install_requires=[
        "numpy>=1.19.2",
        "setuptools>=52.0.0",
        "torch>=1.13.0",
    ],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    # package_dir={"torchonn": "."},
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
