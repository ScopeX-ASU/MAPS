'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-29 04:15:29
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-17 02:48:39
'''

from setuptools import setup, find_packages
from pyutils.version import __version__

setup(
    name="pyutils",
    version=__version__,
    description="Python/Pytorch Utility",
    url="https://github.com/JeremieMelo/pyutility",
    author="Jiaqi Gu",
    author_email="jqgu@utexas.edu",
    license="MIT",
    install_requires=[
        "numpy>=1.19.2",
        "torchvision>=0.9.0.dev20210130",
        "tqdm>=4.56.0",
        "setuptools>=61.0.0",
        "torch>=1.8.0",
        "matplotlib>=3.3.4",
        "svglib>=1.1.0",
        "scipy>=1.5.4",
        "scikit-learn>=0.24.1",
        # "torchsummary>=1.5.1",
        "pyyaml>=5.1.1",
        "tensorflow>=2.5.0",
    ],
    python_requires=">=3",
    include_package_data=True,
    packages=find_packages(),
)
