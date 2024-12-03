"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 01:32:01
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 01:32:01
"""
import torch
from torch.optim.optimizer import Optimizer, required
import math

__all__ = ["SMTP"]


class SMTP(Optimizer):
    r"""Implements SMTP algorithm. ZOO
    It has been proposed in `Stochastic Momentum Three Points`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum (default: 0)
        obj_fn (callable, required): objective function
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. SMTP: A STOCHASTIC DERIVATIVE FREE OPTIMIZATION METHOD WITH MOMENTUM ICLR 2020
        https://arxiv.org/pdf/1905.13278.pdf

    """

    def __init__(self, params, lr=1e-3, momentum=0, obj_fn=required, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 <= momentum < 1:
            raise ValueError("Invalid momentum: {}".format(momentum))
        self.obj_fn = obj_fn

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SMTP, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SMTP, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["v_k"] = 0
                    state["z_k"] = p.data.clone()

                state["step"] += 1
                lr = state["lr"]
                beta = group["momentum"]
                z_k = state["z_k"]
                v_k = state["v_k"]

                s_k = torch.randn_like(p)
                s_k /= s_k.norm(p=2)
                v_k_plus_1_p = beta * v_k + s_k
                v_k_plus_1_n = beta * v_k - s_k

                x_k_plus_1_p = p - lr * v_k_plus_1_p
                x_k_plus_1_n = p - lr * v_k_plus_1_n

                z_k_plus_1_p = x_k_plus_1_p - lr * beta / (1 - beta) * x_k_plus_1_p
                z_k_plus_1_n = x_k_plus_1_n - lr * beta / (1 - beta) * v_k_plus_1_n

                sorted_obj = sorted(
                    [
                        (p.data, v_k, z_k, self.obj_fn(z_k)),
                        (x_k_plus_1_p, v_k_plus_1_p, z_k_plus_1_p, self.obj_fn(z_k_plus_1_p)),
                        (x_k_plus_1_n, v_k_plus_1_n, z_k_plus_1_n, self.obj_fn(z_k_plus_1_n)),
                    ],
                    key=lambda x: x[3],
                )

                p.data.copy_(sorted_obj[0][0])
                state["v_k"] = sorted_obj[0][1]
                state["z_k"] = sorted_obj[0][2]
                loss = sorted_obj[0][3]

                if group["weight_decay"] != 0:
                    p.data.add_(-lr * group["weight_decay"] * p.data)
                    # grad.add_(group['weight_decay'], p.data)

        return loss
