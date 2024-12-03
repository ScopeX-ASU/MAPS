"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 01:31:12
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 01:31:12
"""
import torch
from torch.optim.optimizer import Optimizer, required
import math

__all__ = ["GLD"]


class GLD(Optimizer):
    r"""Implements GLD-search and GLD-fast algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _GLD\: Gradientless Descent: High-Dimensional Zeroth-Order Optimization ICLR 2020
        https://arxiv.org/abs/1911.06317

    """

    def __init__(
        self, params, lr=1e-3, max_r=8, min_r=1, obj_fn=required, weight_decay=0, max_cond=2, mode="search"
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 <= min_r <= max_r:
            raise ValueError("Invalid searching radius: ({},{})".format(min_r, max_r))
        self.mode = mode
        if mode not in {"search", "fast"}:
            raise ValueError("Invalid mode: {}. Only support 'search' and 'fast'".format(mode))
        if mode == "fast" and max_cond <= 0:
            raise ValueError("Invalid condition number bound: {}.".format(max_cond))
        self.obj_fn = obj_fn
        self.search = {"search": self.GLD_search, "fast": self.GLD_fast_search}[mode]

        K = (int(math.log2(max_r / min_r)) + 1) if mode == "search" else (int(math.log2(max_cond)) + 1)
        defaults = dict(
            lr=lr, max_r=max_r, min_r=min_r, weight_decay=weight_decay, max_cond=max_cond, mode=mode, K=K
        )
        super(GLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GLD, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('amsgrad', False)

    def GLD_search(self, obj_fn, K, R, p):
        obj_start = obj_min = obj_fn(p).item()
        v_min = 0
        for k in range(K):
            r_k = 2 ** (-k) * R
            v_k = torch.randn_like(p)
            v_k /= v_k.norm(p=2) / r_k
            p1 = p + v_k
            obj_k = obj_fn(p1).item()
            if obj_k <= obj_min:
                obj_min = obj_k
                v_min = v_k.clone()
                # p_min = p1.clone()
        if obj_min <= obj_start:
            p.data.add_(v_min)
        return obj_min

    def GLD_fast_search(self, obj_fn, K, R, p):
        obj_start = obj_min = obj_fn(p).item()
        v_min = 0
        for k in range(-K, K + 1):
            r_k = 2 ** (-k) * R
            v_k = torch.randn_like(p)
            v_k /= v_k.norm(p=2) / r_k
            p1 = p + v_k
            obj_k = obj_fn(p1).item()
            if obj_k <= obj_min:
                obj_min = obj_k
                v_min = v_k.clone()
                # p_min = p1.clone()
        if obj_min <= obj_start:
            p.data.add_(v_min)
        return obj_min

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

                state["step"] += 1
                K, R = group["k"], group["max_r"]
                if self.mode == "fast":
                    Q = group["max_cond"]
                    H = int(p.numel() * Q * math.log2(Q))
                    R /= 2 ** (state["step"] // H)
                loss = self.search(self.obj_fn, K, R, p)

                if group["weight_decay"] != 0:
                    grad.add_(p.data, alpha=group["weight_decay"])

        return loss
