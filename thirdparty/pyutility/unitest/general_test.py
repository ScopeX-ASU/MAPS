"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-08 23:37:46
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-08 23:37:46
"""
from pyutils.general import AverageMeter


def test_averagemeter():
    acc = AverageMeter("Acc", ":.4f")
    acc.update(10, 1)
    acc.update(20, 1)
    acc.update(40, 1)
    print(acc)


if __name__ == "__main__":
    test_averagemeter()
