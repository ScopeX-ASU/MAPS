import warnings
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer

from .line_search import Armijo, Curvature, Goldstein, Strong_Wolfe, Weak_Wolfe

__all__ = ("BASIC_NCG",)


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.reshape(-1), b.reshape(-1))


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float) -> torch.Tensor:
    return num / den.clamp_min(eps)


class BASIC_NCG(Optimizer):
    """
    Closure-compatible nonlinear conjugate gradient optimizer.

    This version is modified for expensive FDTD / topology optimization closures
    of the form:

        def closure():
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            optionally_normalize_grad()
            return loss

    Important differences from the original NCG-Optimizer BASIC:
        - line_search='None' means fixed-step update, not exact quadratic line search.
        - no internal torch.autograd.grad calls.
        - closure is called once at the current point before line search.
        - f0 and g0 are passed into line search to avoid redundant current-point evals.
        - line search is called with restore=True; final update is applied once here.
        - CG beta formulas are made more standard.
        - descent-direction safeguard / restart is added.
        - optional beta clamp and periodic restart are added.
    """

    def __init__(
        self,
        params,
        eps: float = 1e-12,
        method: str = "PRP",
        line_search: str = "Armijo",
        c1: float = 1e-4,
        c2: float = 0.1,
        c: float = 0.35,
        lr: float = 1.0,
        rho: float = 0.5,
        max_ls: int = 3,
        beta_max: float = 10.0,
        restart_interval: Optional[int] = None,
        prp_plus: bool = True,
    ):
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if method not in [
            "FR",
            "PRP",
            "HS",
            "CD",
            "DY",
            "LS",
            "HZ",
            "HS-DY",
        ]:
            raise ValueError(f"Invalid method: {method}")

        if line_search not in [
            "Armijo",
            "Curvature",
            "Strong_Wolfe",
            "Weak_Wolfe",
            "Goldstein",
            "None",
        ]:
            raise ValueError(f"Invalid line search: {line_search}")

        if line_search == "None":
            warnings.warn(
                "line_search='None' uses fixed-step NCG, not exact line search."
            )

        if not (0.0 < c1 < 1.0):
            raise ValueError(f"Invalid c1 value: {c1}")

        if not (c1 < c2 < 1.0):
            raise ValueError(f"Invalid c2 value: {c2}")

        if not (0.0 < c < 0.5):
            raise ValueError(f"Invalid c value: {c}")

        if lr < 0.0:
            raise ValueError(f"Invalid lr value: {lr}")

        if not (0.0 < rho < 1.0):
            raise ValueError(f"Invalid rho value: {rho}")

        if int(max_ls) != max_ls or max_ls <= 0:
            raise ValueError(f"Invalid max_ls value: {max_ls}")

        if beta_max <= 0:
            raise ValueError(f"Invalid beta_max value: {beta_max}")

        defaults = dict(
            eps=eps,
            method=method,
            line_search=line_search,
            c1=c1,
            c2=c2,
            c=c,
            lr=lr,
            rho=rho,
            max_ls=int(max_ls),
            beta_max=beta_max,
            restart_interval=restart_interval,
            prp_plus=prp_plus,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def _compute_beta(
        self,
        method: str,
        g: torch.Tensor,
        g_prev: torch.Tensor,
        d_prev: torch.Tensor,
        eps: float,
        beta_max: float,
        prp_plus: bool,
    ) -> torch.Tensor:
        """
        Compute nonlinear CG beta.

        g:
            current gradient
        g_prev:
            previous gradient
        d_prev:
            previous search direction
        """
        y = g - g_prev

        gg = _dot(g, g)
        gprev_gprev = _dot(g_prev, g_prev).clamp_min(eps)
        dprev_y = _dot(d_prev, y)
        dprev_gprev = _dot(d_prev, g_prev)
        g_y = _dot(g, y)

        if method == "FR":
            # Fletcher-Reeves
            beta = gg / gprev_gprev

        elif method == "PRP":
            # Polak-Ribiere-Polyak
            beta = g_y / gprev_gprev
            if prp_plus:
                beta = torch.clamp(beta, min=0.0)

        elif method == "HS":
            # Hestenes-Stiefel
            beta = _safe_div(g_y, dprev_y, eps)

        elif method == "CD":
            # Conjugate Descent
            beta = _safe_div(-gg, dprev_gprev, eps)

        elif method == "DY":
            # Dai-Yuan
            beta = _safe_div(gg, dprev_y, eps)

        elif method == "LS":
            # Liu-Storey
            beta = _safe_div(-g_y, dprev_gprev, eps)

        elif method == "HZ":
            # Hager-Zhang
            # beta_HZ = (y - 2 d ||y||^2 / d^T y)^T g / d^T y
            yy = _dot(y, y)
            denom = dprev_y.clamp_min(eps)
            beta = _dot(y - 2.0 * d_prev * yy / denom, g) / denom

        elif method == "HS-DY":
            beta_hs = _safe_div(g_y, dprev_y, eps)
            beta_dy = _safe_div(gg, dprev_y, eps)
            beta = torch.maximum(
                torch.zeros_like(beta_hs),
                torch.minimum(beta_hs, beta_dy),
            )

        else:
            raise ValueError(f"Invalid method: {method}")

        # Avoid pathological huge beta in nonconvex topology optimization.
        beta = torch.nan_to_num(beta, nan=0.0, posinf=beta_max, neginf=0.0)
        beta = torch.clamp(beta, min=0.0 if prp_plus else -beta_max, max=beta_max)
        return beta

    def _line_search_alpha(
        self,
        closure,
        p: torch.Tensor,
        f0: float,
        g0: torch.Tensor,
        d: torch.Tensor,
        state: dict,
        group: dict,
    ) -> float:
        line_search = group["line_search"]
        c1 = group["c1"]
        c2 = group["c2"]
        c = group["c"]
        lr = group["lr"]
        rho = group["rho"]
        eps = group["eps"]
        max_ls = group["max_ls"]

        # Use previous accepted alpha as next initial guess when available.
        alpha0 = float(state.get("alpha", lr))
        if alpha0 <= 0.0:
            alpha0 = lr

        if line_search == "None":
            return float(alpha0)

        if line_search == "Armijo":
            return float(
                Armijo(
                    closure,
                    p,
                    g0,
                    d,
                    alpha0,
                    rho,
                    c1,
                    max_ls,
                    f0=f0,
                    restore=True,
                )
            )

        if line_search == "Curvature":
            return float(
                Curvature(
                    closure,
                    p,
                    g0,
                    d,
                    alpha0,
                    rho,
                    c2,
                    max_ls,
                    f0=f0,
                    restore=True,
                )
            )

        if line_search == "Strong_Wolfe":
            return float(
                Strong_Wolfe(
                    closure,
                    p,
                    alpha0,
                    d,
                    c1,
                    c2,
                    tolerance_change=eps,
                    max_ls=max_ls,
                    f0=f0,
                    g0=g0,
                    rho=rho,
                    restore=True,
                )
            )

        if line_search == "Weak_Wolfe":
            return float(
                Weak_Wolfe(
                    closure,
                    p,
                    alpha0,
                    d,
                    c1,
                    c2,
                    tolerance_change=eps,
                    max_ls=max_ls,
                    f0=f0,
                    g0=g0,
                    rho=rho,
                    restore=True,
                )
            )

        if line_search == "Goldstein":
            return float(
                Goldstein(
                    closure,
                    p,
                    alpha0,
                    d,
                    c,
                    tolerance_change=eps,
                    max_ls=max_ls,
                    f0=f0,
                    g0=g0,
                    rho=rho,
                    restore=True,
                )
            )

        raise ValueError(f"Invalid line search: {line_search}")

    def step(self, closure=None):
        """
        Performs one NCG optimization step.

        The closure must:
            - zero gradients
            - evaluate forward objective
            - call backward()
            - return scalar loss

        Returns:
            loss at the original point before the step.
        """
        if closure is None:
            raise RuntimeError("BASIC NCG requires a closure.")

        # Evaluate current loss and current gradient once.
        with torch.enable_grad():
            loss = closure()

        f0 = float(loss.detach())

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                eps = group["eps"]
                method = group["method"]
                beta_max = group["beta_max"]
                restart_interval = group["restart_interval"]
                prp_plus = group["prp_plus"]

                g = p.grad.detach().clone()
                state = self.state[p]

                grad_norm = torch.linalg.vector_norm(g)
                if grad_norm < eps:
                    continue

                if len(state) == 0:
                    state["g_prev"] = g.clone()
                    state["d"] = -g.clone()
                    state["alpha"] = float(group["lr"])
                    state["step"] = 0
                    state["beta"] = torch.tensor(0.0, dtype=g.dtype, device=g.device)
                else:
                    g_prev = state["g_prev"]
                    d_prev = state["d"]

                    # Optional periodic restart.
                    do_restart = False
                    if restart_interval is not None and restart_interval > 0:
                        if state["step"] % restart_interval == 0:
                            do_restart = True

                    if do_restart:
                        beta = torch.tensor(0.0, dtype=g.dtype, device=g.device)
                        d = -g
                    else:
                        beta = self._compute_beta(
                            method=method,
                            g=g,
                            g_prev=g_prev,
                            d_prev=d_prev,
                            eps=eps,
                            beta_max=beta_max,
                            prp_plus=prp_plus,
                        )
                        d = -g + beta * d_prev

                        # Descent safeguard. If not descent, restart to steepest descent.
                        if _dot(g, d) >= 0:
                            beta = torch.tensor(0.0, dtype=g.dtype, device=g.device)
                            d = -g

                    state["beta"] = beta
                    state["d"] = d.detach().clone()
                    state["g_prev"] = g.clone()

                d = state["d"]

                # Final descent safeguard.
                if _dot(g, d) >= 0:
                    d = -g
                    state["d"] = d.detach().clone()
                    state["beta"] = torch.tensor(0.0, dtype=g.dtype, device=g.device)

                # Line search uses current f0/g and restores p.
                alpha = self._line_search_alpha(
                    closure=closure,
                    p=p,
                    f0=f0,
                    g0=g,
                    d=d,
                    state=state,
                    group=group,
                )

                # If line search fails, skip or take a conservative fallback.
                if alpha <= 0.0:
                    # Conservative fallback: no update.
                    # For topology optimization, skipping is often safer than a bad step.
                    state["alpha"] = float(group["lr"])
                    state["step"] += 1
                    continue

                # Apply the accepted step exactly once.
                with torch.no_grad():
                    p.add_(d, alpha=alpha)

                state["alpha"] = float(alpha)
                state["step"] += 1

        return loss
