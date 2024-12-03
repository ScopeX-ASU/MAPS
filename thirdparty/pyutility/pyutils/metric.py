'''
Date: 2024-03-25 20:44:27
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-25 20:44:27
FilePath: /pyutility/pyutils/metric.py
'''
"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 01:45:28
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 01:45:28
"""

import torch

__all__ = [
    "accuracy",
    "top_k_acc",
]


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)



class PerformanceTracker(object):
    """Computes and stores the performance metrics"""

    def __init__(self, total, pos_set={1}):
        self.total = total
        self.pos_set = pos_set
        self.reset()

    def reset(self):
        self.cnt_TP = 0
        self.cnt_FN = 0
        self.cnt_FP = 0
        self.cnt_TN = 0
        self.correct = 0
        self.running_total = 0

    def update(self, pred, label):
        if isinstance(label, torch.Tensor):
            with torch.no_grad():
                correct_mask = pred == label
                wrong_mask = ~correct_mask
                self.correct += correct_mask.sum().item()
                pos_mask = sum(label == i for i in self.pos_set).bool()
                neg_mask = ~pos_mask
                self.cnt_TP += (correct_mask & pos_mask).sum().item()
                self.cnt_TN += (correct_mask & neg_mask).sum().item()
                self.cnt_FN += (wrong_mask & pos_mask).sum().item()
                self.cnt_FP += (wrong_mask & neg_mask).sum().item()
                self.running_total += label.numel()
        else:
            if pred == label:
                self.correct += 1
                if label in self.pos_set:
                    self.cnt_TP += 1
                else:
                    self.cnt_TN += 1
            else:
                if label in self.pos_set:
                    self.cnt_FN += 1
                else:
                    self.cnt_FP += 1
            self.running_total += 1

