# InvDes Optimizer Reference

This page records the behavior of the optimizer implementations used by `core.invdes.builder.make_optimizer`.

## Dispatch surface

The optimizer factory owns the public name-to-class dispatch. The supported family includes the custom `Adam`, `Adahessian`, `Nesterov`, `Muon`, `NCGOptimizer`, and line-search-based optimizers present under `core/invdes/optimizer`.

The runtime passes the optimization model parameters to the selected optimizer. For `InvDesign`, the optimizer must support a closure because the expensive photonic simulation and objective evaluation occur inside the closure.

## Closure semantics

A closure is expected to recompute the forward objective and gradients from the current parameter state. The `InvDesign` closure returns `-results["obj"]`, so the optimizer's minimization direction corresponds to maximizing the reported figure of merit.

The closure may also perform gradient normalization before returning. Optimizers that reevaluate the closure during line search therefore trigger fresh electromagnetic forward/adjoint evaluations by design.

## `Adahessian`

`Adahessian` implements a Hutchinson-style approximation of the Hessian trace/diagonal contribution. It maintains exponential moving averages for the gradient and squared Hessian estimate and supports configurable `hessian_power`, `update_each`, and `n_samples` values.

The implementation uses a deterministic `torch.Generator` for the Rademacher probes and rebinds the generator to the active device when necessary. Optional convolution-kernel averaging replaces the raw Hessian estimate with a spatially averaged magnitude for 4-D parameter tensors.

Its `step(closure=None)` evaluates the closure when supplied, refreshes the Hessian estimate, applies decoupled-style weight decay, updates first and second moments, and performs the parameter update.

## `Adam`

The custom `Adam` implementation is derived from the PyTorch optimizer interface but adds an `init_v` configuration for non-zero initialization of the second-moment accumulator. It preserves the standard Adam options used by the codebase, including `amsgrad`, `maximize`, `foreach`, `capturable`, `differentiable`, `fused`, and decoupled weight decay.

This optimizer is the default `InvDesign` choice in the current orchestration configuration.

## `Nesterov`

The Nesterov optimizer provides a momentum-accelerated first-order update and is intended for designs where the acceleration behavior of classical Nesterov momentum is preferable to Adam-family adaptive moments.

## `Muon`

Muon is the matrix-oriented optimizer in this package. It applies the package's orthogonalized momentum/update rule to compatible parameter tensors and therefore has different shape/update assumptions from scalar-elementwise Adam-style optimizers.

## `NCGOptimizer`

`NCGOptimizer` implements nonlinear conjugate-gradient optimization. Its state tracks the previous gradient/search direction needed to construct a new conjugate direction rather than maintaining Adam-style moment buffers.

Because the implementation uses objective reevaluation, closure support is especially important when this optimizer is selected for photonic inverse design.

## `line_search.py`

The line-search utilities are explicitly closure-compatible for expensive simulation-backed objectives. They do **not** call `torch.autograd.grad` merely to obtain a trial gradient because the closure itself already performs `zero_grad()`, forward propagation, backward propagation, and optional gradient normalization.

### Trial evaluation

`_eval_closure_at` moves the parameter tensor to `x0 + alpha * d`, invokes the closure, converts the scalar loss to a Python float, and clones the resulting gradient from `x.grad`.

### Armijo search

`Armijo` checks the descent condition `g^T d < 0` and backtracks the step size by `rho` until the sufficient-decrease condition controlled by `c1` is met or the configured trial limit is exhausted. It can optionally restore the original parameter/gradient state.

This design is important for FDTD/topology optimization because each trial point can trigger a full forward/adjoint solve.

## Practical selection guidance

Use the factory configuration rather than constructing optimizer objects ad hoc so the optimizer, learning-rate scheduler, and sharpness scheduler remain consistent with the `InvDesign` orchestration contract.

For standard device optimization, the current default is custom `Adam`. For second-order information, `Adahessian` provides curvature-aware updates at additional closure/autograd cost. For direct search-direction methods, `Nesterov` and `NCGOptimizer` expose momentum or conjugacy behavior, respectively. Line-search utilities are most appropriate when reliable step acceptance is more important than minimizing the number of expensive objective evaluations.
