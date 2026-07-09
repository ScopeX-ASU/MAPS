"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2026-05-29 15:21:57
LastEditors: Jiaqi Gu (jiaqigu@asu.edu)
LastEditTime: 2026-06-01 23:28:40
FilePath: /MAPS_fdtdx/core/heat/solver.py
"""

import time

import torch
from torch import Tensor, nn

from core.utils import _jax_to_torch, _torch_to_jax


def _runtime_torch_dtype(runtime, fallback_dtype: torch.dtype) -> torch.dtype:
    solver_options = getattr(runtime, "solver_options", {}) or {}
    jax_options = solver_options.get("jax_solver", {}) or {}
    solve_dtype = str(jax_options.get("solve_dtype", "")).lower()
    if solve_dtype in {"float64", "fp64", "double"}:
        return torch.float64
    if solve_dtype in {"float32", "fp32", "single"}:
        return torch.float32
    return fallback_dtype


class HeatSolveTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k_map, q_map, runtime):
        if k_map.is_cuda:
            torch.cuda.set_device(k_map.device)
        runtime_dtype = _runtime_torch_dtype(runtime, k_map.dtype)
        runtime_k_map = k_map.to(dtype=runtime_dtype)
        runtime_q_map = q_map.to(dtype=runtime_dtype)
        temperature, vjp_fn = runtime.solve_with_vjp(
            _torch_to_jax(runtime_k_map),
            _torch_to_jax(runtime_q_map),
        )
        ctx.runtime = runtime
        ctx.vjp_fn = vjp_fn
        ctx.runtime_dtype = runtime_dtype
        ctx.k_dtype = k_map.dtype
        ctx.q_dtype = q_map.dtype
        ctx.k_device = k_map.device
        out = _jax_to_torch(temperature).to(device=k_map.device, dtype=k_map.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        start_time = time.time()
        print("HEAT adjoint simulation started...")
        if grad_output.is_cuda:
            torch.cuda.set_device(grad_output.device)
        runtime_grad_output = grad_output.contiguous().to(dtype=ctx.runtime_dtype)
        grad_k, grad_q = ctx.vjp_fn(_torch_to_jax(runtime_grad_output))
        grad_k_t = _jax_to_torch(grad_k).to(device=ctx.k_device, dtype=ctx.k_dtype)
        grad_q_t = _jax_to_torch(grad_q).to(device=ctx.k_device, dtype=ctx.q_dtype)
        end_time = time.time()
        print(f"HEAT adjoint solver takes {end_time - start_time:.4f} seconds")
        return grad_k_t, grad_q_t, None


class HeatSolveTorch(nn.Module):
    def forward(self, k_map, q_map, runtime):
        return HeatSolveTorchFunction.apply(k_map, q_map, runtime)
