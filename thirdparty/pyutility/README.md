<!--
 * @Date: 2024-06-16 13:06:59
 * @LastEditors: Jiaqi Gu && jiaqigu@asu.edu
 * @LastEditTime: 2024-06-17 02:43:28
 * @FilePath: /pyutility/README.md
-->
# pyutils
A python/pytorch utility library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## News
- v0.0.2 available. Added new datasets and quantization!
- v0.0.1 available. Feedbacks are highly welcomed!

## Installation
```bash
conda install scopex/label/ScopeX::torchonn-pyutils
```
```bash
pip install torchonn-pyutils --no-build-isolation
```
or install from cloned codes from github if you would like to modify the code
```bash
git clone https://github.com/JeremieMelo/pyutility.git
cd pyutility
pip3 install --editable .
```

## Usage
```bash
import pyutils
```

## Features
- Support pytorch training utility and datasets.

## TODOs
- [ ] Support lr_scheduler
- [ ] Support trainer

## Dependencies
- Python >= 3.6
- PyTorch >= 1.8.0
- Tensorflow >= 2.5.0
- Others are listed in requirements.txt


## Files
| File      | Description |
| ----------- | ----------- |
| datasets/ | Defines different datasets and builder |
| loss/ | Defines different loss functions/criterions |
| optimizer/ | Defines different optimizers |
| lr_scheduler/ | Defines different learning rate schedulers |
| quant/ | Defines different weight/activation quantizers |
| activation.py      | Activation functions |
| compute.py   | functions related to computing |
| config.py   | Hierarchical yaml configuration file parser |
| distribution_sampler.py   | Sample from customized distributions |
| general.py   | Common helper functions |
| initializer.py   | Initialization methods for PyTorch Parameters |
| loss.py   | Loss functions for PyTorch model training |
| quantize.py   | Quantization functions |
| torch_train.py   | Helper functions for torch training |
| typing.py | Defines common types |


## Contact
Jiaqi Gu (jqgu@utexas.edu)
