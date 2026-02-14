# lm_optimizer.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer

logger = logging.getLogger(__name__)

SolveMethod = Literal["qr", "cholesky", "solve"]
DampingMode = Literal["standard", "fletcher"]


class DampingStrategy:
    """Interface aligned with training.py usage."""

    def reset(self) -> None:
        raise NotImplementedError

    def initialize_step(self, loss: Tensor) -> None:
        raise NotImplementedError

    def apply(self, JJ: Tensor) -> Tensor:
        raise NotImplementedError

    def on_successful_update(self, loss: Tensor) -> None:
        raise NotImplementedError

    def on_unsuccessful_update(self, loss: Tensor) -> None:
        raise NotImplementedError

    def stop_attempts(self, loss: Tensor) -> bool:
        return False

    def stop_training(self, loss: Tensor) -> bool:
        return False

    def get_current_damping(self) -> Tensor:
        raise NotImplementedError


class StandardDampingStrategy(DampingStrategy):
    """
    A reasonable faithful "standard LM" damping strategy with the same hooks used in training.py.

    - apply(JJ): JJ + lambda * I      (standard)
      or JJ + lambda * diag(JJ)       (fletcher)

    - successful update: lambda *= dec_factor
    - unsuccessful update: lambda *= inc_factor

    Includes optional attempt/training stopping when lambda saturates.
    """

    def __init__(
        self,
        starting_value: float = 1e-3,
        dec_factor: float = 0.1,
        inc_factor: float = 10.0,
        min_value: float = 1e-12,
        max_value: float = 1e12,
        mode: DampingMode = "standard",
        conditional_stopping: bool = True,
        auto_reset: bool = False,
    ) -> None:
        self.starting_value = float(starting_value)
        self.dec_factor = float(dec_factor)
        self.inc_factor = float(inc_factor)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.mode = mode
        self.conditional_stopping = bool(conditional_stopping)
        self.auto_reset = bool(auto_reset)

        self._damping: Optional[Tensor] = None
        self.reset()

    def reset(self) -> None:
        self._damping = torch.tensor(
            self.starting_value, dtype=torch.get_default_dtype()
        )

    def initialize_step(self, loss: Tensor) -> None:
        # Keep as-is. training.py calls this at each step.
        _ = loss

    def get_current_damping(self) -> Tensor:
        assert self._damping is not None
        return self._damping

    def apply(self, JJ: Tensor) -> Tensor:
        assert self._damping is not None
        lam = self._damping.to(device=JJ.device, dtype=JJ.dtype)

        if JJ.ndim != 2 or JJ.shape[0] != JJ.shape[1]:
            raise ValueError(f"JJ must be square, got {tuple(JJ.shape)}")

        n = JJ.shape[0]
        I = torch.eye(n, device=JJ.device, dtype=JJ.dtype)

        if self.mode == "standard":
            return JJ + lam * I
        elif self.mode == "fletcher":
            diag = torch.diag(torch.diagonal(JJ))
            return JJ + lam * diag
        else:
            raise ValueError(f"Unknown damping mode: {self.mode}")

    def on_successful_update(self, loss: Tensor) -> None:
        _ = loss
        assert self._damping is not None
        self._damping = torch.clamp(
            self._damping * self.dec_factor, self.min_value, self.max_value
        )

    def on_unsuccessful_update(self, loss: Tensor) -> None:
        _ = loss
        assert self._damping is not None
        self._damping = torch.clamp(
            self._damping * self.inc_factor, self.min_value, self.max_value
        )

    def stop_attempts(self, loss: Tensor) -> bool:
        _ = loss
        if not self.conditional_stopping:
            return False
        assert self._damping is not None
        return bool(self._damping >= self.max_value)

    def stop_training(self, loss: Tensor) -> bool:
        _ = loss
        assert self._damping is not None
        if self._damping >= self.max_value:
            if self.auto_reset:
                self.reset()
                return False
            return True
        return False


@dataclass
class _ParamBackup:
    tensors: List[Tensor]

    @staticmethod
    @torch.no_grad()
    def capture(params: Sequence[Tensor]) -> "_ParamBackup":
        return _ParamBackup([p.detach().clone() for p in params])

    @torch.no_grad()
    def restore(self, params: Sequence[Tensor]) -> None:
        for p, old in zip(params, self.tensors):
            p.copy_(old)


def _trainable_params_from_groups(param_groups) -> List[Tensor]:
    ps: List[Tensor] = []
    for g in param_groups:
        for p in g["params"]:
            if isinstance(p, Tensor) and p.requires_grad:
                ps.append(p)
    return ps


def _flatten_grads_like_training_py(
    grads: Sequence[Optional[Tensor]], params: Sequence[Tensor]
) -> Tensor:
    flat: List[Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros_like(p).reshape(-1))
        else:
            flat.append(g.reshape(-1))
    if len(flat) == 0:
        return torch.empty(0)
    return torch.cat(flat, dim=0)


class LevenbergMarquardt(Optimizer):
    """
    LM as a standard torch optimizer, modeled after training.py's LevenbergMarquardtModule,
    but for scalar loss closures.

    You provide:
      closure() -> scalar loss (Tensor 0-d)

    Internally:
      residual = sqrt(loss)    (scalar)
      Jacobian row J = d residual / d params  (1 x P)
      num_residuals is always 1, so it's typically underdetermined (1 < P).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1.0,
        attempts_per_step: int = 10,
        solve_method: SolveMethod = "qr",
        damping_strategy: Optional[DampingStrategy] = None,
        sqrt_eps: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if attempts_per_step < 1:
            raise ValueError("attempts_per_step must be >= 1")
        if solve_method not in ("qr", "cholesky", "solve"):
            raise ValueError("solve_method must be one of: 'qr', 'cholesky', 'solve'")

        defaults = dict(
            lr=float(lr),
            attempts_per_step=int(attempts_per_step),
            solve_method=solve_method,
            sqrt_eps=float(sqrt_eps),
        )
        super().__init__(list(params), defaults)

        self.damping_strategy = damping_strategy or StandardDampingStrategy()

    @torch.no_grad()
    def _solve(self, matrix: Tensor, rhs: Tensor, solve_method: SolveMethod) -> Tensor:
        # Direct copy of the logic in training.py::_solve
        rhs = rhs.to(device=matrix.device, dtype=matrix.dtype)
        if solve_method == "qr":
            q, r = torch.linalg.qr(matrix)
            y = torch.matmul(q.transpose(-2, -1), rhs)
            return torch.linalg.solve_triangular(r, y, upper=True)
        elif solve_method == "cholesky":
            L = torch.linalg.cholesky(matrix)
            y = torch.linalg.solve_triangular(L, rhs, upper=False)
            return torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
        elif solve_method == "solve":
            return torch.linalg.solve(matrix, rhs)
        else:
            raise ValueError(
                f"Invalid solve_method '{solve_method}'. Choose from 'qr', 'cholesky', 'solve'."
            )

    @torch.no_grad()
    def _apply_updates(self, updates_flat: Tensor) -> None:
        """
        Apply updates in the same spirit as training.py::_apply_updates:
            flat_params.add_(-lr * updates)
        Here we don't rebind params to a single flat tensor; we slice updates into param shapes.
        """
        # Apply learning rate group-wise, keeping ordering consistent.
        offset = 0
        for group in self.param_groups:
            lr = float(group["lr"])
            for p in group["params"]:
                if not isinstance(p, Tensor) or not p.requires_grad:
                    continue
                n = p.numel()
                upd = updates_flat[offset : offset + n].view_as(p)
                p.add_(-lr * upd)
                offset += n

        if offset != updates_flat.numel():
            raise RuntimeError("Update vector length mismatch with parameters.")

    def step(self, closure: Callable[[], Tensor]) -> Tensor:
        if closure is None:
            raise RuntimeError(
                "LevenbergMarquardt requires a closure() that returns the scalar loss."
            )

        params = _trainable_params_from_groups(self.param_groups)
        if len(params) == 0:
            raise RuntimeError("No trainable parameters found (requires_grad=True).")

        # ---- Compute J, JJ, rhs, loss (modeled after training.py) ----
        with torch.enable_grad():
            loss = closure(backward=False)
            if loss.ndim != 0:
                raise ValueError(
                    f"closure() must return scalar loss; got shape {tuple(loss.shape)}"
                )
            if torch.any(loss < 0):
                loss = loss + loss.detach().abs() + 1e-6
                # raise ValueError("Loss must be non-negative to form residual sqrt(loss).")

            self.damping_strategy.initialize_step(loss)

            # Scalar residual (num_residuals = 1)
            sqrt_eps = torch.tensor(
                float(self.param_groups[0]["sqrt_eps"]),
                device=loss.device,
                dtype=loss.dtype,
            )
            residual = torch.sqrt(loss + sqrt_eps)  # scalar

            # Jacobian row wrt params: J_flat shape (P,)
            grads = torch.autograd.grad(
                residual,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
            J_flat = _flatten_grads_like_training_py(grads, params)  # (P,)

            # Shapes to match training.py's linear algebra (column vectors / matrices)
            num_residuals = 1
            num_params = int(J_flat.numel())
            overdetermined = num_residuals >= num_params

            # Build GN pieces exactly in the two cases:
            if overdetermined:
                # J is (1,P), residuals is (1,1)
                # JJ = J'J is (P,P) = outer(J_flat, J_flat)
                # rhs = J' r is (P,1) = J_flat * residual
                JJ = torch.outer(J_flat, J_flat)
                rhs = (J_flat * residual).reshape(-1, 1)
                J = None
            else:
                # JJ = J J' is (1,1) = dot(J_flat, J_flat)
                # rhs = residuals is (1,1) = residual
                JJ = (J_flat @ J_flat).reshape(1, 1)
                rhs = residual.reshape(1, 1)
                J = J_flat.reshape(1, -1)  # (1,P) for J.t() @ updates later

            # Normalize for numerical stability (training.py does 1/batch_size)
            batch_size = 1
            normalization_factor = 1.0 / batch_size
            JJ = JJ * normalization_factor
            rhs = rhs * normalization_factor

        # Backup params (training.py does backup_parameters on accept)
        backup = _ParamBackup.capture(params)

        stop_training = False
        attempt = 0
        self.damping_strategy.initialize_step(loss)

        # ---- Attempt loop (modeled after training.py) ----
        while True:
            params_updated = False

            try:
                with torch.no_grad():
                    JJ_damped = self.damping_strategy.apply(JJ)

                    # Compute updates:
                    # - Overdetermined: updates = (J'J + damp)^-1 * (J' r)
                    # - Underdetermined: updates = (JJ' + damp)^-1 * r ; then map back: J' * updates
                    updates = self._solve(
                        JJ_damped,
                        rhs,
                        solve_method=self.param_groups[0]["solve_method"],
                    )

                    if not overdetermined:
                        assert J is not None
                        updates = J.t().matmul(updates)  # (P,1)

                    updates = updates.view(-1)

                    # Check if updates are finite
                    if torch.all(torch.isfinite(updates)):
                        params_updated = True
                        self._apply_updates(updates)

            except Exception as e:
                logger.warning(f"An exception occurred during LM solve/update: {e}")

            if attempt < int(self.param_groups[0]["attempts_per_step"]):
                attempt += 1

                if params_updated:
                    # Evaluate new loss
                    with torch.enable_grad():
                        new_loss = closure(backward=False)
                        if new_loss.ndim != 0:
                            # restore and hard-fail
                            backup.restore(params)
                            raise ValueError(
                                f"closure() must return scalar loss; got shape {tuple(new_loss.shape)}"
                            )

                    if new_loss < loss:
                        # Accept
                        loss = new_loss
                        self.damping_strategy.on_successful_update(loss)
                        # Backup accepted params like training.py
                        backup = _ParamBackup.capture(params)
                        break

                    # Reject: restore
                    backup.restore(params)

                # Adjust damping for next attempt
                self.damping_strategy.on_unsuccessful_update(loss)

                stop_attempts = self.damping_strategy.stop_attempts(loss)
                stop_training = self.damping_strategy.stop_training(loss)
                if stop_training or stop_attempts:
                    break
            else:
                break

        # Store logs in optimizer state (optional)
        self.state["lm_logs"] = {
            "damping": self.damping_strategy.get_current_damping().detach().clone(),
            "attempts": attempt,
            "stop_training": stop_training,
        }

        return loss
