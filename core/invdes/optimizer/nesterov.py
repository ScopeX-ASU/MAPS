##
# @file   NesterovAcceleratedGradientOptimizer.py
# @author Yibo Lin
# @date   Aug 2018
# @brief  Nesterov's accelerated gradient method proposed by e-place.
#

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required

__all__ = ["NesterovAcceleratedGradientOptimizer"]


class NesterovAcceleratedGradientOptimizer(Optimizer):
    """
    Follow the Nesterov implementation of the e-place algorithm, but keep
    optimizer state as one flattened tensor per parameter group so groups may
    contain multiple parameter tensors.
    """

    def __init__(
        self,
        params,
        lr: float = required,
        constraint_fn=None,
        alg: str = "bb",
        block_tile_size=None,
        block_blur_kernel_size: int = 3,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr,
            obj_eval_count=0,
            u_k=None,
            v_k=None,
            g_k=None,
            obj_k=None,
            a_k=None,
            alpha_k=None,
            v_k_1=None,
            g_k_1=None,
            obj_k_1=None,
            v_kp1=None,
        )
        super().__init__(params, defaults)

        if constraint_fn is None:

            def dummy_constraint_fn(*args, **kwargs):
                return None

            constraint_fn = dummy_constraint_fn

        self.constraint_fn = constraint_fn
        self.alg = alg
        self.block_tile_size = block_tile_size
        self.block_blur_kernel_size = block_blur_kernel_size
        assert self.alg in [
            "bb",
            "bb_static",
            "bb_static_blockwise",
            "nobb",
        ], f"Unsupported alg: {self.alg}"

    def __setstate__(self, state):
        super().__setstate__(state)

    def _params_of_group(self, group):
        return group["params"]

    def _numel(self, group):
        if "numel_cache" not in group:
            group["numel_cache"] = sum(p.numel() for p in self._params_of_group(group))
        return group["numel_cache"]

    def _gather_flat_grad_group(self, group):
        views = []
        for p in self._params_of_group(group):
            if p.grad is None:
                view = p.new_zeros(p.numel())
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_param_group(self, group):
        views = []
        for p in self._params_of_group(group):
            pdata = p.data
            if torch.is_complex(pdata):
                pdata = torch.view_as_real(pdata)
            views.append(pdata.view(-1))
        return torch.cat(views, 0)

    def _clone_param_group(self, group):
        return [
            p.data.clone(memory_format=torch.contiguous_format)
            for p in self._params_of_group(group)
        ]

    def _set_param_group(self, group, params_data):
        for p, pdata in zip(self._params_of_group(group), params_data):
            p.data.copy_(pdata)

    def _set_flat_param_group(self, group, flat_param):
        offset = 0
        for p in self._params_of_group(group):
            target = p.data
            numel = target.numel()
            if torch.is_complex(target):
                target = torch.view_as_real(target)
            target.copy_(flat_param[offset : offset + numel].view_as(target))
            offset += numel
        assert offset == self._numel(group)

    def _apply_constraint(self, group, flat_param):
        params = self._params_of_group(group)
        if len(params) == 1:
            constrained = flat_param.view_as(params[0].data)
            self.constraint_fn(constrained)
            return constrained.reshape(-1)

        old_params = self._clone_param_group(group)
        self._set_flat_param_group(group, flat_param)
        try:
            try:
                self.constraint_fn(*params)
            except TypeError:
                self.constraint_fn()
            constrained = self._gather_flat_param_group(group).clone()
        finally:
            self._set_param_group(group, old_params)
        return constrained

    def _obj_and_grad_fn(self, group, flat_param, closure):
        old_params = self._clone_param_group(group)
        try:
            self._set_flat_param_group(group, flat_param)
            for p in self._params_of_group(group):
                p.grad = None
            obj = closure()
            grad = self._gather_flat_grad_group(group).detach().clone()
        finally:
            self._set_param_group(group, old_params)
        return obj.detach().clone(), grad

    def step(self, closure):
        closure = torch.enable_grad()(closure)
        if self.alg == "bb_static":
            return self.step_bb_static(closure)
        elif self.alg == "bb_static_blockwise":
            return self.step_bb_static_blockwise(closure)
        elif self.alg == "bb":
            return self.step_bb(closure)
        elif self.alg == "nobb":
            return self.step_nobb(closure)
        else:
            raise ValueError(f"Unsupported alg: {self.alg}")

    def _normalize_tile_size(self):
        tile_size = self.block_tile_size
        if tile_size is None:
            return None
        if isinstance(tile_size, int):
            return (tile_size, tile_size)
        if len(tile_size) == 1:
            return (int(tile_size[0]), int(tile_size[0]))
        if len(tile_size) >= 2:
            return (int(tile_size[0]), int(tile_size[1]))
        return None

    def _build_group_blocks(self, group):
        if "block_specs" in group:
            return group["block_specs"]

        tile_size = self._normalize_tile_size()
        block_specs = []
        offset = 0

        for p in self._params_of_group(group):
            pdata = p.data
            if torch.is_complex(pdata):
                pdata = torch.view_as_real(pdata)
            shape = tuple(pdata.shape)
            numel = pdata.numel()

            if tile_size is None or len(shape) != 2:
                block_specs.append(
                    dict(
                        kind="flat",
                        indices=[
                            torch.arange(
                                offset,
                                offset + numel,
                                device=pdata.device,
                                dtype=torch.long,
                            )
                        ],
                        alpha_shape=(1,),
                        param_shape=shape,
                        offset=offset,
                        numel=numel,
                    )
                )
                offset += numel
                continue

            tile_h = max(1, tile_size[0])
            tile_w = max(1, tile_size[1])
            height, width = shape
            row_ranges = []
            col_ranges = []
            indices = []

            for row_start in range(0, height, tile_h):
                row_stop = min(height, row_start + tile_h)
                row_ranges.append((row_start, row_stop))
            for col_start in range(0, width, tile_w):
                col_stop = min(width, col_start + tile_w)
                col_ranges.append((col_start, col_stop))

            for row_start, row_stop in row_ranges:
                for col_start, col_stop in col_ranges:
                    rows = torch.arange(
                        row_start, row_stop, device=pdata.device, dtype=torch.long
                    )[:, None]
                    cols = torch.arange(
                        col_start, col_stop, device=pdata.device, dtype=torch.long
                    )[None, :]
                    tile_idx = offset + (rows * width + cols).reshape(-1)
                    indices.append(tile_idx)

            block_specs.append(
                dict(
                    kind="tiled_2d",
                    indices=indices,
                    alpha_shape=(len(row_ranges), len(col_ranges)),
                    param_shape=shape,
                    offset=offset,
                    numel=numel,
                )
            )

            offset += numel

        group["block_specs"] = block_specs
        return block_specs

    def _smooth_alpha_map(self, alpha_map):
        kernel_size = int(self.block_blur_kernel_size or 0)
        if kernel_size <= 1:
            return alpha_map
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        return F.avg_pool2d(
            alpha_map[None, None],
            kernel_size=kernel_size,
            stride=1,
            padding=pad,
        )[0, 0]

    def _compute_block_step_size(self, s_b, y_b, prev_alpha):
        y_dot_y = y_b.dot(y_b)
        if torch.isclose(y_dot_y, torch.zeros_like(y_dot_y)):
            step_size = prev_alpha
        else:
            bb_short_step_size = s_b.dot(y_b) / y_dot_y
            gdiff_norm = y_b.norm(p=2)
            if torch.isclose(gdiff_norm, torch.zeros_like(gdiff_norm)):
                lip_step_size = prev_alpha
            else:
                lip_step_size = s_b.norm(p=2) / gdiff_norm
            step_size = torch.where(
                bb_short_step_size > 0,
                bb_short_step_size,
                torch.minimum(lip_step_size, prev_alpha),
            )

        if (not torch.isfinite(step_size)) or step_size <= 0:
            step_size = prev_alpha
        return step_size

    def _scaled_step_size(self, group, step_size):
        return group["lr"] * step_size

    def _compute_blockwise_step(self, group, s_k, y_k, alpha_k):
        block_specs = self._build_group_blocks(group)
        next_alpha = torch.empty_like(alpha_k)
        step_vec = torch.empty_like(s_k)
        alpha_offset = 0

        for spec in block_specs:
            indices = spec["indices"]
            n_blocks = len(indices)
            prev_alpha_chunk = alpha_k[alpha_offset : alpha_offset + n_blocks]
            next_alpha_chunk = torch.empty_like(prev_alpha_chunk)

            for block_id, idx in enumerate(indices):
                s_b = s_k.index_select(0, idx)
                y_b = y_k.index_select(0, idx)
                step_size = self._compute_block_step_size(
                    s_b, y_b, prev_alpha_chunk[block_id]
                )
                next_alpha_chunk[block_id] = step_size

            if spec["kind"] == "tiled_2d":
                alpha_grid = next_alpha_chunk.view(spec["alpha_shape"])
                alpha_map = F.interpolate(
                    alpha_grid[None, None],
                    size=spec["param_shape"],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                alpha_map = self._smooth_alpha_map(alpha_map)
                step_vec[spec["offset"] : spec["offset"] + spec["numel"]] = (
                    alpha_map.reshape(-1)
                )
            else:
                step_vec.index_copy_(
                    0,
                    indices[0],
                    torch.ones_like(indices[0], dtype=s_k.dtype) * next_alpha_chunk[0],
                )

            next_alpha[alpha_offset : alpha_offset + n_blocks] = next_alpha_chunk
            alpha_offset += n_blocks

        return step_vec, next_alpha

    def _init_group_state(self, group, closure):
        if group["v_k"] is not None:
            return

        v_k = self._gather_flat_param_group(group).detach().clone()
        obj_k, g_k = self._obj_and_grad_fn(group, v_k, closure)
        u_k = v_k.clone()
        v_k_1 = v_k - group["lr"] * g_k
        v_k_1.copy_(self._apply_constraint(group, v_k_1))
        obj_k_1, g_k_1 = self._obj_and_grad_fn(group, v_k_1, closure)

        if self.alg == "bb_static_blockwise":
            num_blocks = sum(
                len(spec["indices"]) for spec in self._build_group_blocks(group)
            )
            alpha_k = torch.full(
                (num_blocks,),
                fill_value=group["lr"],
                dtype=v_k.dtype,
                device=v_k.device,
            )
        else:
            denom = (g_k - g_k_1).norm(p=2)
            if torch.isclose(denom, torch.zeros_like(denom)):
                alpha_k = torch.as_tensor(
                    group["lr"], dtype=v_k.dtype, device=v_k.device
                )
            else:
                alpha_k = (v_k - v_k_1).norm(p=2) / denom

        group["u_k"] = u_k
        group["v_k"] = v_k
        group["g_k"] = g_k
        group["obj_k"] = obj_k
        group["a_k"] = torch.ones(1, dtype=v_k.dtype, device=v_k.device)
        group["alpha_k"] = alpha_k.reshape(-1)
        group["v_k_1"] = v_k_1
        group["g_k_1"] = g_k_1
        group["obj_k_1"] = obj_k_1
        group["v_kp1"] = torch.zeros_like(v_k)
        self._set_flat_param_group(group, v_k)

    def step_nobb(self, closure=None):
        loss = None

        for group in self.param_groups:
            self._init_group_state(group, closure)

            u_k = group["u_k"]
            v_k = group["v_k"]
            g_k = group["g_k"]
            obj_k = group["obj_k"]
            a_k = group["a_k"]
            alpha_k = group["alpha_k"]
            v_k_1 = group["v_k_1"]
            g_k_1 = group["g_k_1"]
            obj_k_1 = group["obj_k_1"]
            v_kp1 = group["v_kp1"]

            a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
            coef = (a_k - 1) / a_kp1
            backtrack_cnt = 0
            max_backtrack_cnt = 10

            while True:
                prev_alpha = alpha_k.clone()
                u_kp1 = v_k - self._scaled_step_size(group, alpha_k) * g_k
                v_kp1.copy_(u_kp1 + coef * (u_kp1 - u_k))
                v_kp1.copy_(self._apply_constraint(group, v_kp1))

                f_kp1, g_kp1 = self._obj_and_grad_fn(group, v_kp1, closure)
                denom = torch.sum((g_kp1 - g_k) ** 2)
                if torch.isclose(denom, torch.zeros_like(denom)):
                    alpha_kp1 = alpha_k.clone()
                else:
                    alpha_kp1 = torch.sqrt(
                        torch.sum((v_kp1 - v_k) ** 2) / denom
                    ).reshape_as(alpha_k)

                backtrack_cnt += 1
                group["obj_eval_count"] += 1

                alpha_k.copy_(alpha_kp1)
                if (
                    alpha_kp1.item() > 0.95 * prev_alpha.item()
                    or backtrack_cnt >= max_backtrack_cnt
                ):
                    break

            v_k_1.copy_(v_k)
            g_k_1.copy_(g_k)
            obj_k_1.copy_(obj_k)

            u_k.copy_(u_kp1)
            v_k.copy_(v_kp1)
            g_k.copy_(g_kp1)
            obj_k.copy_(f_kp1)
            a_k.copy_(a_kp1)

            self._set_flat_param_group(group, v_k)

        return loss

    def step_bb(self, closure=None):
        loss = None

        for group in self.param_groups:
            self._init_group_state(group, closure)

            u_k = group["u_k"]
            v_k = group["v_k"]
            a_k = group["a_k"]
            alpha_k = group["alpha_k"]
            v_k_1 = group["v_k_1"]

            obj_k, g_k = self._obj_and_grad_fn(group, v_k, closure)
            obj_k_1, g_k_1 = self._obj_and_grad_fn(group, v_k_1, closure)
            group["obj_k"].copy_(obj_k)
            group["obj_k_1"].copy_(obj_k_1)

            a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
            coef = (a_k - 1) / a_kp1

            with torch.no_grad():
                s_k = v_k - v_k_1
                y_k = g_k - g_k_1
                y_dot_y = y_k.dot(y_k)
                if torch.isclose(y_dot_y, torch.zeros_like(y_dot_y)):
                    step_size = alpha_k.squeeze(0)
                else:
                    bb_short_step_size = s_k.dot(y_k) / y_dot_y
                    gdiff_norm = y_k.norm(p=2)
                    if torch.isclose(gdiff_norm, torch.zeros_like(gdiff_norm)):
                        lip_step_size = alpha_k.squeeze(0)
                    else:
                        lip_step_size = s_k.norm(p=2) / gdiff_norm
                    step_size = (
                        bb_short_step_size
                        if bb_short_step_size > 0
                        else torch.minimum(lip_step_size, alpha_k.squeeze(0))
                    )

            u_kp1 = v_k - self._scaled_step_size(group, step_size) * g_k
            v_kp1 = group["v_kp1"]
            v_kp1.copy_(u_kp1 + coef * (u_kp1 - u_k))
            v_kp1.copy_(self._apply_constraint(group, v_kp1))
            group["obj_eval_count"] += 1

            v_k_1.copy_(v_k)
            alpha_k.fill_(step_size)
            u_k.copy_(u_kp1)
            v_k.copy_(v_kp1)
            group["g_k"].copy_(g_k)
            group["g_k_1"].copy_(g_k_1)
            a_k.copy_(a_kp1)

            self._set_flat_param_group(group, v_k)

        return loss

    def step_bb_static(self, closure=None):
        """
        BB-style step for objectives whose closure has no hidden state drift.

        Reuses the previous step's cached evaluation at ``v_k_1``:
            v_k_1 <- previous v_k
            g_k_1 <- previous g_k
            obj_k_1 <- previous obj_k

        Therefore each new step only evaluates the current ``v_k`` once.
        """
        loss = None

        for group in self.param_groups:
            self._init_group_state(group, closure)

            u_k = group["u_k"]
            v_k = group["v_k"]
            a_k = group["a_k"]
            alpha_k = group["alpha_k"]
            v_k_1 = group["v_k_1"]
            g_k_1 = group["g_k_1"]
            obj_k_1 = group["obj_k_1"]

            obj_k, g_k = self._obj_and_grad_fn(group, v_k, closure)
            group["obj_eval_count"] += 1

            a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
            coef = (a_k - 1) / a_kp1

            with torch.no_grad():
                s_k = v_k - v_k_1
                y_k = g_k - g_k_1
                y_dot_y = y_k.dot(y_k)
                if torch.isclose(y_dot_y, torch.zeros_like(y_dot_y)):
                    step_size = alpha_k.squeeze(0)
                else:
                    bb_short_step_size = s_k.dot(y_k) / y_dot_y
                    gdiff_norm = y_k.norm(p=2)
                    if torch.isclose(gdiff_norm, torch.zeros_like(gdiff_norm)):
                        lip_step_size = alpha_k.squeeze(0)
                    else:
                        lip_step_size = s_k.norm(p=2) / gdiff_norm
                    step_size = (
                        bb_short_step_size
                        if bb_short_step_size > 0
                        else torch.minimum(lip_step_size, alpha_k.squeeze(0))
                    )

            u_kp1 = v_k - self._scaled_step_size(group, step_size) * g_k
            v_kp1 = group["v_kp1"]
            v_kp1.copy_(u_kp1 + coef * (u_kp1 - u_k))
            v_kp1.copy_(self._apply_constraint(group, v_kp1))

            v_k_1.copy_(v_k)
            g_k_1.copy_(g_k)
            obj_k_1.copy_(obj_k)
            alpha_k.fill_(step_size)
            u_k.copy_(u_kp1)
            v_k.copy_(v_kp1)
            group["g_k"].copy_(g_k)
            group["obj_k"].copy_(obj_k)
            a_k.copy_(a_kp1)

            self._set_flat_param_group(group, v_k)

        return loss

    def step_bb_static_blockwise(self, closure=None):
        """
        Blockwise BB step for static objectives.

        Each optimizer group is expected to correspond to one design region.
        Within a group, every 2D parameter tensor is partitioned into fixed tiles.
        Edge tiles use their natural smaller size; no padding is introduced.
        Non-2D tensors fall back to one block per tensor.
        """
        loss = None

        for group in self.param_groups:
            self._init_group_state(group, closure)

            u_k = group["u_k"]
            v_k = group["v_k"]
            a_k = group["a_k"]
            alpha_k = group["alpha_k"]
            v_k_1 = group["v_k_1"]
            g_k_1 = group["g_k_1"]
            obj_k_1 = group["obj_k_1"]

            obj_k, g_k = self._obj_and_grad_fn(group, v_k, closure)
            group["obj_eval_count"] += 1

            a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
            coef = (a_k - 1) / a_kp1

            with torch.no_grad():
                s_k = v_k - v_k_1
                y_k = g_k - g_k_1
                step_vec, next_alpha = self._compute_blockwise_step(
                    group, s_k, y_k, alpha_k
                )

            u_kp1 = v_k - self._scaled_step_size(group, step_vec) * g_k
            v_kp1 = group["v_kp1"]
            v_kp1.copy_(u_kp1 + coef * (u_kp1 - u_k))
            v_kp1.copy_(self._apply_constraint(group, v_kp1))

            v_k_1.copy_(v_k)
            g_k_1.copy_(g_k)
            obj_k_1.copy_(obj_k)
            alpha_k.copy_(next_alpha)
            u_k.copy_(u_kp1)
            v_k.copy_(v_kp1)
            group["g_k"].copy_(g_k)
            group["obj_k"].copy_(obj_k)
            a_k.copy_(a_kp1)

            self._set_flat_param_group(group, v_k)

        return loss
