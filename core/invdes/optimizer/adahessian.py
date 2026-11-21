import torch


class Adahessian(torch.optim.Optimizer):
    """Implement AdaHessian with Hutchinson Hessian-trace estimates.

    The optimizer maintains exponential moving averages of the gradient and an
    estimated Hessian diagonal/trace contribution, then scales the parameter
    update by a configurable power of the Hessian statistic.

    Arguments:
        params (iterable): Parameters or parameter-group dictionaries accepted by
            :class:`torch.optim.Optimizer`.
        lr (float): Learning rate. Defaults to 0.1.
        betas (tuple[float, float]): Exponential decay factors for gradient and
            Hessian statistics. Defaults to ``(0.9, 0.999)``.
        eps (float): Denominator stabilizer. Defaults to ``1e-8``.
        weight_decay (float): AdamW-style decoupled weight decay. Defaults to 0.0.
        hessian_power (float): Exponent applied to the estimated Hessian statistic.
            Defaults to 1.0.
        update_each (int): Recompute the Hessian estimate every this many optimizer
            steps. Defaults to 1.
        n_samples (int): Number of Hutchinson Rademacher samples per Hessian update.
            Defaults to 1.
        average_conv_kernel (bool): For 4-D tensors, spatially average the absolute
            Hessian statistic before the update. Defaults to False.

    Raises:
        ValueError: If ``lr``, ``eps``, ``betas``, or ``hessian_power`` is outside
            the supported range.
    """

    def __init__(
        self,
        params,
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        hessian_power=1.0,
        update_each=1,
        n_samples=1,
        average_conv_kernel=False,
    ):
        """Initialize optimizer defaults and per-parameter Hessian state.

        Args:
            params: Iterable of parameters or parameter-group dictionaries.
            lr (float): Learning rate. Defaults to 0.1.
            betas (tuple[float, float]): Gradient and Hessian EMA coefficients.
                Defaults to ``(0.9, 0.999)``.
            eps (float): Denominator stabilizer. Defaults to ``1e-8``.
            weight_decay (float): Decoupled weight decay coefficient. Defaults to 0.0.
            hessian_power (float): Hessian statistic exponent in ``[0, 1]``.
                Defaults to 1.0.
            update_each (int): Number of optimizer steps between Hessian recomputations.
                Defaults to 1.
            n_samples (int): Number of Hutchinson samples per recomputation. Defaults to 1.
            average_conv_kernel (bool): Whether to spatially average 4-D Hessian estimates.
                Defaults to False.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid Hessian power value: {hessian_power}")

        self.n_samples = n_samples
        self.update_each = update_each
        self.average_conv_kernel = average_conv_kernel

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(2147483647)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
        )
        super(Adahessian, self).__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def get_params(self):
        """Yield all trainable parameters in all optimizer parameter groups.

        Returns:
            iterator: Parameters whose ``requires_grad`` flag is true.
        """
        return (
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        )

    def zero_hessian(self):
        """Clear stored Hessian estimates when their scheduled update is due.

        Parameters whose Hessian estimate is still a scalar initialization value
        are left unchanged; tensor estimates are zeroed only when the associated
        ``hessian step`` counter is divisible by ``update_each``.
        """
        for p in self.get_params():
            if (
                not isinstance(p.hess, float)
                and self.state[p]["hessian step"] % self.update_each == 0
            ):
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """Estimate and accumulate Hessian statistics with Hutchinson sampling.

        For scheduled parameters with existing gradients, this method samples
        Rademacher vectors and uses ``torch.autograd.grad`` to estimate the
        Hessian-vector products. The resulting ``z * (H @ z)`` estimates are
        accumulated into each parameter's ``p.hess`` field and averaged over
        ``n_samples`` samples.
        """
        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if (
                self.state[p]["hessian step"] % self.update_each == 0
            ):  # compute the trace only each `update_each` step
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if (
            self.generator.device != params[0].device
        ):  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        grads = [p.grad for p in params]

        for i in range(self.n_samples):
            zs = [
                torch.randint(0, 2, p.size(), generator=self.generator, device=p.device)
                * 2.0
                - 1.0
                for p in params
            ]  # Rademacher distribution {-1.0, 1.0}
            h_zs = torch.autograd.grad(
                grads,
                params,
                grad_outputs=zs,
                only_inputs=True,
                retain_graph=i < self.n_samples - 1,
            )
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += (
                    h_z * z / self.n_samples
                )  # approximate the expected values of z*(H@z)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one AdaHessian parameter update.

        Args:
            closure (Callable[[], torch.Tensor] | None): Optional closure that
                reevaluates the objective and returns the loss. Defaults to None.

        Returns:
            torch.Tensor | None: The closure's loss when a closure is supplied;
            otherwise None.

        Notes:
            Each step refreshes scheduled Hessian estimates, updates first- and
            second-moment state with the configured ``betas``, applies decoupled
            weight decay, and scales the update by the bias-corrected Hessian
            statistic raised to ``hessian_power / 2``.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = (
                        torch.abs(p.hess)
                        .mean(dim=[2, 3], keepdim=True)
                        .expand_as(p.hess)
                        .clone()
                    )

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - group["lr"] * group["weight_decay"])

                state = self.state[p]

                # State initialization
                if len(state) == 1:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p.data
                    )  # Exponential moving average of gradient values
                    state["exp_hessian_diag_sq"] = torch.zeros_like(
                        p.data
                    )  # Exponential moving average of Hessian diagonal square values

                exp_avg, exp_hessian_diag_sq = (
                    state["exp_avg"],
                    state["exp_hessian_diag_sq"],
                )
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(
                    p.hess, p.hess, value=1 - beta2
                )

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                k = group["hessian_power"]
                denom = (
                    (exp_hessian_diag_sq / bias_correction2)
                    .pow_(k / 2)
                    .add_(group["eps"])
                )

                # make update
                step_size = group["lr"] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
