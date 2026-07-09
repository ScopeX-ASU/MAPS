"""
Closure-compatible line search functions for expensive FDTD / topology optimization.

Expected closure behavior:
    loss = closure()
        - zero_grad()
        - forward()
        - backward()
        - optionally normalize p.grad
        - return scalar loss

Important:
    These functions DO NOT call torch.autograd.grad(func(), x, ...)
    because your closure already computes x.grad.

They call closure once per trial point and read x.grad directly.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch

Tensor = torch.Tensor


def _dot(a: Tensor, b: Tensor) -> Tensor:
    return torch.dot(a.reshape(-1), b.reshape(-1))


def _clone_grad(x: Tensor) -> Tensor:
    if x.grad is None:
        return torch.zeros_like(x)
    return x.grad.detach().clone()


def _restore_x_and_grad(x: Tensor, x0: Tensor, g0: Optional[Tensor]) -> None:
    with torch.no_grad():
        x.copy_(x0)
        if g0 is not None:
            if x.grad is None:
                x.grad = g0.detach().clone()
            else:
                x.grad.copy_(g0)


def _eval_closure_at(
    func: Callable[[], Tensor],
    x: Tensor,
    x0: Tensor,
    alpha: float,
    d: Tensor,
) -> Tuple[float, Tensor]:
    """
    Evaluate closure at x = x0 + alpha * d.

    Returns:
        f: Python float loss
        g: cloned gradient at trial point
    """
    with torch.no_grad():
        x.copy_(x0 + alpha * d)

    loss = func()
    f = float(loss.detach())
    g = _clone_grad(x)
    return f, g


def _check_descent(g0: Tensor, d: Tensor) -> float:
    gtd0 = float(_dot(g0, d))
    return gtd0


def Armijo(
    func: Callable[[], Tensor],
    x: Tensor,
    g: Tensor,
    d: Tensor,
    lr: float,
    rho: float,
    c1: float,
    iter: int,
    f0: Optional[float] = None,
    restore: bool = False,
) -> float:
    """
    Closure-compatible Armijo backtracking.

    Args:
        func:
            Closure that evaluates loss and writes x.grad.
        x:
            Parameter tensor.
        g:
            Current gradient at original x.
        d:
            Search direction. Should be descent: dot(g, d) < 0.
        lr:
            Initial step size.
        rho:
            Backtracking factor, e.g. 0.5.
        c1:
            Armijo sufficient decrease constant, e.g. 1e-4.
        iter:
            Max line-search trials.
        f0:
            Optional current loss at original x. If None, this function calls func()
            once at the original x.
        restore:
            If True, restore x and grad to original point before returning.
            If False, leave x and x.grad at accepted trial. If failed, restore.

    Returns:
        Accepted alpha. Returns 0.0 if no acceptable step is found.
    """
    x0 = x.detach().clone()
    g0 = g.detach().clone()

    if f0 is None:
        loss0 = func()
        f0 = float(loss0.detach())
        g0 = _clone_grad(x)

    gtd0 = _check_descent(g0, d)

    if gtd0 >= 0:
        _restore_x_and_grad(x, x0, g0)
        return 0.0

    alpha = float(lr)

    for _ in range(iter):
        f_new, g_new = _eval_closure_at(func, x, x0, alpha, d)

        if f_new <= f0 + c1 * alpha * gtd0:
            if restore:
                _restore_x_and_grad(x, x0, g0)
            # otherwise leave x at accepted point and x.grad at accepted grad
            return alpha

        alpha *= rho

    # Failed: restore original point.
    _restore_x_and_grad(x, x0, g0)
    return 0.0


def Curvature(
    func: Callable[[], Tensor],
    x: Tensor,
    g: Tensor,
    d: Tensor,
    lr: float,
    rho: float,
    c2: float,
    iter: int,
    f0: Optional[float] = None,
    restore: bool = False,
) -> float:
    """
    Closure-compatible curvature-only backtracking.

    Accepts alpha if:
        g(alpha)^T d >= c2 * g(0)^T d

    Usually used as part of Wolfe, not recommended alone for your FDTD case.
    """
    del f0

    x0 = x.detach().clone()
    g0 = g.detach().clone()
    gtd0 = _check_descent(g0, d)

    if gtd0 >= 0:
        _restore_x_and_grad(x, x0, g0)
        return 0.0

    alpha = float(lr)

    for _ in range(iter):
        _, g_new = _eval_closure_at(func, x, x0, alpha, d)
        gtd_new = float(_dot(g_new, d))

        if gtd_new >= c2 * gtd0:
            if restore:
                _restore_x_and_grad(x, x0, g0)
            return alpha

        alpha *= rho

    _restore_x_and_grad(x, x0, g0)
    return 0.0


def Weak_Wolfe(
    func: Callable[[], Tensor],
    x: Tensor,
    t: float,
    d: Tensor,
    c1: float,
    c2: float,
    tolerance_change: float = 1e-3,
    max_ls: int = 10,
    f0: Optional[float] = None,
    g0: Optional[Tensor] = None,
    rho: float = 0.5,
    restore: bool = False,
) -> float:
    """
    Closure-compatible weak Wolfe backtracking.

    Conditions:
        f(alpha) <= f0 + c1 * alpha * g0^T d
        g(alpha)^T d >= c2 * g0^T d

    This version is intentionally simple and cheap:
        - no cubic zoom
        - no torch.autograd.grad
        - one closure call per trial

    For expensive 3D FDTD, use max_ls=2 or 3 first.
    """
    del tolerance_change

    x0 = x.detach().clone()

    if f0 is None or g0 is None:
        loss0 = func()
        f0 = float(loss0.detach())
        g0 = _clone_grad(x)
    else:
        f0 = float(f0)
        g0 = g0.detach().clone()

    gtd0 = float(_dot(g0, d))

    if gtd0 >= 0:
        _restore_x_and_grad(x, x0, g0)
        return 0.0

    alpha = float(t)

    for _ in range(max_ls):
        f_new, g_new = _eval_closure_at(func, x, x0, alpha, d)
        gtd_new = float(_dot(g_new, d))

        armijo_ok = f_new <= f0 + c1 * alpha * gtd0
        curvature_ok = gtd_new >= c2 * gtd0

        if armijo_ok and curvature_ok:
            if restore:
                _restore_x_and_grad(x, x0, g0)
            return alpha

        alpha *= rho

    _restore_x_and_grad(x, x0, g0)
    return 0.0


def Strong_Wolfe(
    func: Callable[[], Tensor],
    x: Tensor,
    t: float,
    d: Tensor,
    c1: float,
    c2: float,
    tolerance_change: float = 1e-3,
    max_ls: int = 10,
    f0: Optional[float] = None,
    g0: Optional[Tensor] = None,
    rho: float = 0.5,
    restore: bool = False,
) -> float:
    """
    Closure-compatible strong Wolfe backtracking.

    Conditions:
        f(alpha) <= f0 + c1 * alpha * g0^T d
        |g(alpha)^T d| <= c2 * |g0^T d|

    This is a simplified strong-Wolfe backtracking, not a full cubic zoom
    implementation. It is designed for expensive closures where redundant
    evaluations are unacceptable.

    One closure call per trial.
    No torch.autograd.grad.
    """
    del tolerance_change

    x0 = x.detach().clone()

    if f0 is None or g0 is None:
        loss0 = func()
        f0 = float(loss0.detach())
        g0 = _clone_grad(x)
    else:
        f0 = float(f0)
        g0 = g0.detach().clone()

    gtd0 = float(_dot(g0, d))

    if gtd0 >= 0:
        _restore_x_and_grad(x, x0, g0)
        return 0.0

    alpha = float(t)

    for _ in range(max_ls):
        f_new, g_new = _eval_closure_at(func, x, x0, alpha, d)
        gtd_new = float(_dot(g_new, d))

        armijo_ok = f_new <= f0 + c1 * alpha * gtd0
        curvature_ok = abs(gtd_new) <= c2 * abs(gtd0)

        if armijo_ok and curvature_ok:
            if restore:
                _restore_x_and_grad(x, x0, g0)
            return alpha

        alpha *= rho

    _restore_x_and_grad(x, x0, g0)
    return 0.0


def Goldstein(
    func: Callable[[], Tensor],
    x: Tensor,
    t: float,
    d: Tensor,
    c: float,
    tolerance_change: float = 1e-3,
    max_ls: int = 10,
    f0: Optional[float] = None,
    g0: Optional[Tensor] = None,
    rho: float = 0.5,
    restore: bool = False,
) -> float:
    """
    Closure-compatible Goldstein backtracking.

    Goldstein condition for descent direction:
        f0 + (1 - c) * alpha * g0^T d <= f(alpha)
        f(alpha) <= f0 + c * alpha * g0^T d

    Since g0^T d < 0, the first inequality prevents alpha from becoming
    too small; the second prevents too large.

    In expensive nonconvex FDTD optimization this can reject many useful
    steps, so Armijo is often safer.
    """
    del tolerance_change

    x0 = x.detach().clone()

    if f0 is None or g0 is None:
        loss0 = func()
        f0 = float(loss0.detach())
        g0 = _clone_grad(x)
    else:
        f0 = float(f0)
        g0 = g0.detach().clone()

    gtd0 = float(_dot(g0, d))

    if gtd0 >= 0:
        _restore_x_and_grad(x, x0, g0)
        return 0.0

    alpha = float(t)

    for _ in range(max_ls):
        f_new, _ = _eval_closure_at(func, x, x0, alpha, d)

        lower = f0 + (1.0 - c) * alpha * gtd0
        upper = f0 + c * alpha * gtd0

        goldstein_ok = (f_new >= lower) and (f_new <= upper)

        if goldstein_ok:
            if restore:
                _restore_x_and_grad(x, x0, g0)
            return alpha

        # If f_new is too high, step is too large.
        # If f_new is too low, Goldstein says step may be too small,
        # but for expensive nonconvex topology optimization we avoid expansion
        # by default and accept Armijo-style behavior less aggressively.
        alpha *= rho

    _restore_x_and_grad(x, x0, g0)
    return 0.0


def No_Line_Search(
    func: Callable[[], Tensor],
    x: Tensor,
    t: float,
    d: Tensor,
    restore: bool = False,
) -> float:
    """
    Apply fixed step x <- x + t d.

    If restore=True, only tests the trial point then restores.
    """
    x0 = x.detach().clone()
    g0 = _clone_grad(x)

    with torch.no_grad():
        x.add_(d, alpha=float(t))

    func()

    if restore:
        _restore_x_and_grad(x, x0, g0)

    return float(t)


# Optional aliases matching common naming styles.
armijo = Armijo
curvature = Curvature
weak_wolfe = Weak_Wolfe
strong_wolfe = Strong_Wolfe
goldstein = Goldstein
