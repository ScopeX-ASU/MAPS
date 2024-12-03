"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 02:06:54
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 02:06:54
"""

import typing

__all__ = [
    "Logger",
    "Dataset",
    "DataLoader",
    "Optimizer",
    "Scheduler",
    "Criterion",
]

Logger = None
Dataset = None
DataLoader = None
Optimizer = None
Scheduler = None
Criterion = None


# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
if typing.TYPE_CHECKING:
    from pyutils.general import Logger
    from torch.utils.data import DataLoader, Dataset
    from torch.optim.optimizer import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler as Scheduler
    from torch.nn.modules.loss import _Loss as Criterion
