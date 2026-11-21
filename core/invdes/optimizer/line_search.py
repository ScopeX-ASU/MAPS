"""Closure-compatible line-search methods for expensive FDTD/topology-optimization closures.

The public routines evaluate a callable ``func`` that performs forward evaluation,
backpropagation, and writes the gradient into ``x.grad``. Trial points are generated
as ``x = x0 + alpha * d`` and the gradient at each trial point is read directly from
``x.grad`` rather than recomputed with ``torch.autograd.grad``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch

Tensor = torch.Tensor


def _dot(a: Tensor, b: Tensor) -> Tensor:
    """Return the flattened inner product of two tensors.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.

    Returns:
        torch.Tensor: Scalar dot product after flattening both tensors.
    """
    return torch.dot(a.reshape(-1), b.reshape(-1))


def _clone_grad(x: Tensor) -> Tensor:
    """Return a detached copy of ``x.grad`` or a zero tensor when no gradient exists.

    Args:
        x (torch.Tensor): Tensor whose gradient is inspected.

    Returns:
        torch.Tensor: Detached cloned gradient, or ``zeros_like(x)`` when ``x.grad`` is None.
    """
    if x.grad is None:
        return torch.zeros_like(x)
    return x.grad.detach().clone()


def _restore_x_and_grad(x: Tensor, x0: Tensor, g0: Optional[Tensor]) -> None:
    """Restore a parameter tensor and, when supplied, its gradient in-place.

    Args:
        x (torch.Tensor): Tensor to restore.
        x0 (torch.Tensor): Saved tensor value.
        g0 (torch.Tensor | None): Saved gradient value, or None to leave the gradient absent.

    Returns:
        None. ``x`` and ``x.grad`` are modified in-place.
    """
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
    """Evaluate the closure at ``x0 + alpha * d`` and capture its gradient.

    Args:
        func (Callable[[], torch.Tensor]): Closure that evaluates a scalar loss and writes
            the corresponding gradient to ``x.grad``.
        x (torch.Tensor): Parameter tensor updated in-place for the trial evaluation.
        x0 (torch.Tensor): Original parameter value from which the trial point is formed.
        alpha (float): Trial step length.
        d (torch.Tensor): Search direction.

    Returns:
        tuple[float, torch.Tensor]: Python scalar loss and detached cloned gradient at the trial point.
    """
    with torch.no_grad():
        x.copy_(x0 + alpha * d)

    loss = func()
    f = float(loss.detach())
    g = _clone_grad(x)
    return f, g


def _check_descent(g0: Tensor, d: Tensor) -> float:
    """Compute the directional derivative ``g0^T d`` used to test descent directions.

    Args:
        g0 (torch.Tensor): Current gradient.
        d (torch.Tensor): Candidate search direction.

    Returns:
        float: Directional derivative. Negative values indicate a descent direction.
    """
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
    """Perform closure-compatible Armijo backtracking.

    Args:
        func (Callable[[], torch.Tensor]): Closure that computes scalar loss and writes ``x.grad``.
        x (torch.Tensor): Parameter tensor updated at each trial point.
        g (torch.Tensor): Gradient at the original point. Used to test descent and restore state.
        d (torch.Tensor): Search direction. Expected to satisfy ``g^T d < 0``.
        lr (float): Initial trial step length. Defaults to the caller-supplied value.
        rho (float): Multiplicative backtracking factor. Defaults to the caller-supplied value.
        c1 (float): Armijo sufficient-decrease constant. Defaults to the caller-supplied value.
        iter (int): Maximum number of trial evaluations. Defaults to the caller-supplied value.
        f0 (float | None): Loss at the original point. Defaults to None, in which case ``func``
            is evaluated once before backtracking.
        restore (bool): Whether to restore the original ``x`` and gradient after an accepted step.
            Defaults to False.

    Returns:
        float: Accepted step length, or ``0.0`` when the direction is not descending or all trials fail.
        On success with ``restore=False``, ``x`` remains at the accepted point; failed searches restore
        the original point.
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
            return alpha

        alpha *= rho

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
    """Perform curvature-only backtracking for a descent direction.

    Args:
        func (Callable[[], torch.Tensor]): Closure that computes the trial loss and gradient.
        x (torch.Tensor): Parameter tensor updated in-place for each trial.
        g (torch.Tensor): Gradient at the original point. Expected to define a descent direction with ``d``.
        d (torch.Tensor): Search direction.
        lr (float): Initial trial step length. Defaults to the caller-supplied value.
        rho (float): Multiplicative backtracking factor. Defaults to the caller-supplied value.
        c2 (float): Curvature acceptance coefficient. Defaults to the caller-supplied value.
        iter (int): Maximum number of trial evaluations. Defaults to the caller-supplied value.
        f0 (float | None): Unused loss argument retained for API compatibility. Defaults to None.
        restore (bool): Whether to restore the original point after an accepted step. Defaults to False.

    Returns:
        float: Accepted step length, or ``0.0`` when no curvature-satisfying trial is found.
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
    """Perform simplified weak-Wolfe backtracking using one closure call per trial.

    Args:
        func (Callable[[], torch.Tensor]): Closure that computes scalar loss and writes ``x.grad``.
        x (torch.Tensor): Parameter tensor updated at trial points.
        t (float): Initial trial step length. Defaults to the caller-supplied value.
        d (torch.Tensor): Search direction; ``g0^T d`` must be negative.
        c1 (float): Armijo sufficient-decrease coefficient. Defaults to the caller-supplied value.
        c2 (float): Weak-Wolfe curvature coefficient. Defaults to the caller-supplied value.
        tolerance_change (float): Compatibility argument; currently ignored. Defaults to 1e-3.
        max_ls (int): Maximum number of backtracking trials. Defaults to 10.
        f0 (float | None): Current loss. Defaults to None, which triggers one baseline closure evaluation
            when either ``f0`` or ``g0`` is missing.
        g0 (torch.Tensor | None): Current gradient. Defaults to None, which triggers baseline gradient evaluation.
        rho (float): Multiplicative backtracking factor. Defaults to 0.5.
        restore (bool): Whether to restore the original parameter/gradient after acceptance. Defaults to False.

    Returns:
        float: Accepted step length, or ``0.0`` if the descent condition or both Wolfe conditions cannot be met.
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
    """Perform simplified strong-Wolfe backtracking without cubic zoom.

    Args:
        func (Callable[[], torch.Tensor]): Closure that computes scalar loss and writes ``x.grad``.
        x (torch.Tensor): Parameter tensor updated at trial points.
        t (float): Initial trial step length. Defaults to the caller-supplied value.
        d (torch.Tensor): Search direction; ``g0^T d`` must be negative.
        c1 (float): Armijo sufficient-decrease coefficient. Defaults to the caller-supplied value.
        c2 (float): Strong-Wolfe curvature coefficient. Defaults to the caller-supplied value.
        tolerance_change (float): Compatibility argument; currently ignored. Defaults to 1e-3.
        max_ls (int): Maximum number of backtracking trials. Defaults to 10.
        f0 (float | None): Current loss. Defaults to None, which triggers baseline closure evaluation.
        g0 (torch.Tensor | None): Current gradient. Defaults to None, which triggers baseline gradient evaluation.
        rho (float): Multiplicative backtracking factor. Defaults to 0.5.
        restore (bool): Whether to restore the original parameter/gradient after acceptance. Defaults to False.

    Returns:
        float: Accepted step length satisfying both sufficient decrease and absolute curvature,
        or ``0.0`` if none is found.
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
    """Perform Goldstein backtracking for a descent direction.

    Args:
        func (Callable[[], torch.Tensor]): Closure that computes scalar loss and writes ``x.grad``.
        x (torch.Tensor): Parameter tensor updated at trial points.
        t (float): Initial trial step length. Defaults to the caller-supplied value.
        d (torch.Tensor): Search direction; ``g0^T d`` must be negative.
        c (float): Goldstein interval coefficient. Defaults to the caller-supplied value.
        tolerance_change (float): Compatibility argument; currently ignored. Defaults to 1e-3.
        max_ls (int): Maximum number of backtracking trials. Defaults to 10.
        f0 (float | None): Current loss. Defaults to None, which triggers baseline closure evaluation.
        g0 (torch.Tensor | None): Current gradient. Defaults to None, which triggers baseline gradient evaluation.
        rho (float): Multiplicative backtracking factor. Defaults to 0.5.
        restore (bool): Whether to restore the original parameter/gradient after acceptance. Defaults to False.

    Returns:
        float: Accepted step length satisfying the implemented Goldstein interval, or ``0.0`` on failure.
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
    """Apply a fixed-step update and optionally restore the original state.

    Args:
        func (Callable[[], torch.Tensor]): Closure evaluated once after the fixed step.
        x (torch.Tensor): Parameter tensor updated in-place.
        t (float): Fixed step length. Defaults to the caller-supplied value.
        d (torch.Tensor): Search direction.
        restore (bool): Whether to restore the original parameter/gradient after evaluation. Defaults to False.

    Returns:
        float: The applied step length ``t``.
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
