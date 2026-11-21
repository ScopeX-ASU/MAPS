"""Optional Triton backend for exact near-field projection.

Importing this module does not require Triton to be installed. The kernel
accepts CUDA float32 geometry and complex64 fields; its first-order adjoint
recomputes the native PyTorch expression, preserving gradients at the cost of
backward speed.
"""

from __future__ import annotations

import math

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from .projector import MU_0, Surface, project_nearfield

try:
    import triton
    import triton.language as tl
except ImportError:  # The backend remains an optional dependency.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _cmul(ar, ai, br, bi):
        """Multiply two complex values represented by real/imaginary pairs."""
        return ar * br - ai * bi, ar * bi + ai * br

    @triton.jit
    def _kernel_nearfield_tiled(
        OBS,
        POS,
        JR,
        JI,
        MR,
        MI,
        WEIGHT,
        OMEGA,
        KVAL,
        EPSILON: tl.constexpr,
        PERMEABILITY: tl.constexpr,
        ERROR,
        OUT,
        M: tl.constexpr,
        P: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        """Reduce exact dyadic E/H contributions over a source/target tile.

        Args:
            OBS: Flattened target coordinates in meters.
            POS: Flattened source coordinates in meters.
            JR: Real electric surface currents, frequency/source/component order.
            JI: Imaginary electric surface currents in the same order.
            MR: Real magnetic surface currents in that order.
            MI: Imaginary magnetic surface currents in that order.
            WEIGHT: Source area weights in square meters.
            OMEGA: Angular frequencies in rad/s.
            KVAL: Wave numbers in rad/m.
            EPSILON: Absolute permittivity in F/m.
            PERMEABILITY: Absolute permeability in H/m.
            ERROR: Device flag for coincident source and target samples.
            OUT: Interleaved real/imaginary E/H output storage.
            M: Number of targets.
            P: Number of source samples per frequency.
            BLOCK_O: Target tile size.
            BLOCK_S: Source reduction tile size.

        Notes:
            Contributions accumulate locally; no full source/target matrix is
            stored. Coincidence detection shares the same kernel traversal.
        """
        oi = tl.program_id(0) * BLOCK_O + tl.arange(0, BLOCK_O)
        fi = tl.program_id(1)
        source_lanes = tl.arange(0, BLOCK_S)
        ox = tl.load(OBS + oi * 3 + 0, oi < M, 0)[:, None]
        oy = tl.load(OBS + oi * 3 + 1, oi < M, 0)[:, None]
        oz = tl.load(OBS + oi * 3 + 2, oi < M, 0)[:, None]
        omega = tl.load(OMEGA + fi)
        k = tl.load(KVAL + fi)
        erx = tl.zeros((BLOCK_O,), tl.float32)
        eix = tl.zeros((BLOCK_O,), tl.float32)
        ery = tl.zeros((BLOCK_O,), tl.float32)
        eiy = tl.zeros((BLOCK_O,), tl.float32)
        erz = tl.zeros((BLOCK_O,), tl.float32)
        eiz = tl.zeros((BLOCK_O,), tl.float32)
        hrx = tl.zeros((BLOCK_O,), tl.float32)
        hix = tl.zeros((BLOCK_O,), tl.float32)
        hry = tl.zeros((BLOCK_O,), tl.float32)
        hiy = tl.zeros((BLOCK_O,), tl.float32)
        hrz = tl.zeros((BLOCK_O,), tl.float32)
        hiz = tl.zeros((BLOCK_O,), tl.float32)
        for source_start in range(0, P, BLOCK_S):
            pi = source_start + source_lanes
            source_mask = pi < P
            sx = tl.load(POS + pi * 3 + 0, source_mask, 0)[None, :]
            sy = tl.load(POS + pi * 3 + 1, source_mask, 0)[None, :]
            sz = tl.load(POS + pi * 3 + 2, source_mask, 0)[None, :]
            dx, dy, dz = ox - sx, oy - sy, oz - sz
            radius_raw = tl.sqrt(dx * dx + dy * dy + dz * dz)
            coincidence = (radius_raw == 0) & source_mask[None, :] & (oi < M)[:, None]
            coincidence_count = tl.sum(tl.sum(coincidence.to(tl.int32), axis=1), axis=0)
            if coincidence_count > 0:
                tl.atomic_or(ERROR, 1, sem="relaxed")
            radius = tl.where(source_mask[None, :] & (radius_raw > 0), radius_raw, 1.0)
            invr = 1.0 / radius
            ux, uy, uz = dx * invr, dy * invr, dz * invr
            kr = k * radius
            inv_kr = 1.0 / kr
            gr = tl.cos(kr) * invr / (4.0 * math.pi)
            gi = tl.sin(kr) * invr / (4.0 * math.pi)
            ar, ai = 1.0 - inv_kr * inv_kr, inv_kr
            br, bi = -1.0 + 3.0 * inv_kr * inv_kr, -3.0 * inv_kr
            base = (fi * P + pi) * 3
            load_mask = source_mask
            jxr = tl.load(JR + base + 0, load_mask, 0)[None, :]
            jxi = tl.load(JI + base + 0, load_mask, 0)[None, :]
            jyr = tl.load(JR + base + 1, load_mask, 0)[None, :]
            jyi = tl.load(JI + base + 1, load_mask, 0)[None, :]
            jzr = tl.load(JR + base + 2, load_mask, 0)[None, :]
            jzi = tl.load(JI + base + 2, load_mask, 0)[None, :]
            mxr = tl.load(MR + base + 0, load_mask, 0)[None, :]
            mxi = tl.load(MI + base + 0, load_mask, 0)[None, :]
            myr = tl.load(MR + base + 1, load_mask, 0)[None, :]
            myi = tl.load(MI + base + 1, load_mask, 0)[None, :]
            mzr = tl.load(MR + base + 2, load_mask, 0)[None, :]
            mzi = tl.load(MI + base + 2, load_mask, 0)[None, :]
            jp_r, jp_i = ux * jxr + uy * jyr + uz * jzr, ux * jxi + uy * jyi + uz * jzi
            mp_r, mp_i = ux * mxr + uy * myr + uz * mzr, ux * mxi + uy * myi + uz * mzi
            ajxr, ajxi = _cmul(ar, ai, jxr, jxi)
            bjpr, bjpi = _cmul(br, bi, jp_r, jp_i)
            ajyr, ajyi = _cmul(ar, ai, jyr, jyi)
            ajzr, ajzi = _cmul(ar, ai, jzr, jzi)
            amxr, amxi = _cmul(ar, ai, mxr, mxi)
            bmpr, bmpi = _cmul(br, bi, mp_r, mp_i)
            amyr, amyi = _cmul(ar, ai, myr, myi)
            amzr, amzi = _cmul(ar, ai, mzr, mzi)
            djxr, djxi = _cmul(gr, gi, ajxr + ux * bjpr, ajxi + ux * bjpi)
            djyr, djyi = _cmul(gr, gi, ajyr + uy * bjpr, ajyi + uy * bjpi)
            djzr, djzi = _cmul(gr, gi, ajzr + uz * bjpr, ajzi + uz * bjpi)
            dmxr, dmxi = _cmul(gr, gi, amxr + ux * bmpr, amxi + ux * bmpi)
            dmyr, dmyi = _cmul(gr, gi, amyr + uy * bmpr, amyi + uy * bmpi)
            dmzr, dmzi = _cmul(gr, gi, amzr + uz * bmpr, amzi + uz * bmpi)
            qgr, qgi = -gr * invr - k * gi, -gi * invr + k * gr
            jcxr, jcxi = uy * jzr - uz * jyr, uy * jzi - uz * jyi
            jcyr, jcyi = uz * jxr - ux * jzr, uz * jxi - ux * jzi
            jczr, jczi = ux * jyr - uy * jxr, ux * jyi - uy * jxi
            mcxr, mcxi = uy * mzr - uz * myr, uy * mzi - uz * myi
            mcyr, mcyi = uz * mxr - ux * mzr, uz * mxi - ux * mzi
            mczr, mczi = ux * myr - uy * mxr, ux * myi - uy * mxi
            qx_r, qx_i = _cmul(qgr, qgi, mcxr, mcxi)
            qy_r, qy_i = _cmul(qgr, qgi, mcyr, mcyi)
            qz_r, qz_i = _cmul(qgr, qgi, mczr, mczi)
            qjx_r, qjx_i = _cmul(qgr, qgi, jcxr, jcxi)
            qjy_r, qjy_i = _cmul(qgr, qgi, jcyr, jcyi)
            qjz_r, qjz_i = _cmul(qgr, qgi, jczr, jczi)
            scale_e = omega * PERMEABILITY
            er_block_r, er_block_i = -scale_e * djxi - qx_r, scale_e * djxr - qx_i
            ey_block_r, ey_block_i = -scale_e * djyi - qy_r, scale_e * djyr - qy_i
            ez_block_r, ez_block_i = -scale_e * djzi - qz_r, scale_e * djzr - qz_i
            scale_h = omega * EPSILON
            hx_block_r, hx_block_i = qjx_r - scale_h * dmxi, qjx_i + scale_h * dmxr
            hy_block_r, hy_block_i = qjy_r - scale_h * dmyi, qjy_i + scale_h * dmyr
            hz_block_r, hz_block_i = qjz_r - scale_h * dmzi, qjz_i + scale_h * dmzr
            weight = tl.load(WEIGHT + pi, source_mask, 0)[None, :]
            erx += tl.sum(weight * er_block_r, axis=1)
            eix += tl.sum(weight * er_block_i, axis=1)
            ery += tl.sum(weight * ey_block_r, axis=1)
            eiy += tl.sum(weight * ey_block_i, axis=1)
            erz += tl.sum(weight * ez_block_r, axis=1)
            eiz += tl.sum(weight * ez_block_i, axis=1)
            hrx += tl.sum(weight * hx_block_r, axis=1)
            hix += tl.sum(weight * hx_block_i, axis=1)
            hry += tl.sum(weight * hy_block_r, axis=1)
            hiy += tl.sum(weight * hy_block_i, axis=1)
            hrz += tl.sum(weight * hz_block_r, axis=1)
            hiz += tl.sum(weight * hz_block_i, axis=1)
        out_base = (fi * M + oi) * 6
        tl.store(OUT + out_base + 0, erx, oi < M)
        tl.store(OUT + out_base + 1, eix, oi < M)
        tl.store(OUT + out_base + 2, ery, oi < M)
        tl.store(OUT + out_base + 3, eiy, oi < M)
        tl.store(OUT + out_base + 4, erz, oi < M)
        tl.store(OUT + out_base + 5, eiz, oi < M)
        hbase = (fi * M + oi) * 6 + 6 * M * tl.num_programs(1)
        tl.store(OUT + hbase + 0, hrx, oi < M)
        tl.store(OUT + hbase + 1, hix, oi < M)
        tl.store(OUT + hbase + 2, hry, oi < M)
        tl.store(OUT + hbase + 3, hiy, oi < M)
        tl.store(OUT + hbase + 4, hrz, oi < M)
        tl.store(OUT + hbase + 5, hiz, oi < M)


class _TritonNearfield(Function):
    """Triton forward with recomputing PyTorch adjoint."""

    @staticmethod
    def forward(
        ctx,
        positions,
        normals,
        weights,
        electric,
        magnetic,
        frequencies,
        observations,
        eps_r,
        mu_r,
        block_o,
        block_s,
    ):
        """Launch the complex64 tiled forward and save fixed-size input state.

        Args:
            ctx: Autograd context used to retain inputs for recomputation.
            positions: Real source positions ``(P,3)`` in meters.
            normals: Outward unit normals ``(P,3)``.
            weights: Source areas ``(P,)`` in square meters.
            electric: Complex electric fields ``(F,P,3)`` in V/m.
            magnetic: Complex magnetic fields ``(F,P,3)`` in A/m.
            frequencies: Frequency vector ``(F,)`` in Hz.
            observations: Real target positions ``(M,3)`` in meters.
            eps_r: Positive relative permittivity tensor.
            mu_r: Positive relative permeability tensor.
            block_o: Target tile size.
            block_s: Source tile size.

        Returns:
            Complex64 electric and magnetic output fields ``(F,M,3)``.

        Raises:
            RuntimeError: If Triton is unavailable.
            ValueError: If a source and observation coincide.
        """
        if triton is None:
            raise RuntimeError("direct_triton requires the optional 'triton' package")
        currents_j = torch.linalg.cross(
            normals[None].to(electric.dtype), magnetic, dim=-1
        ).contiguous()
        currents_m = -torch.linalg.cross(
            normals[None].to(electric.dtype), electric, dim=-1
        ).contiguous()
        jr = torch.view_as_real(currents_j).contiguous()
        mr = torch.view_as_real(currents_m).contiguous()
        frequencies = frequencies.contiguous()
        omega = 2 * math.pi * frequencies
        eps_abs = 8.8541878128e-12 * eps_r
        mu_abs = MU_0 * mu_r
        k = omega * torch.sqrt(eps_abs * mu_abs)
        f_count, p_count = electric.shape[:2]
        m_count = observations.shape[0]
        out = torch.empty(
            (2, f_count, m_count, 3, 2), dtype=torch.float32, device=positions.device
        )
        error = torch.zeros((), dtype=torch.int32, device=positions.device)
        block = int(block_o)
        _kernel_nearfield_tiled[(triton.cdiv(m_count, block), f_count)](
            observations.contiguous(),
            positions.contiguous(),
            jr[..., 0].contiguous(),
            jr[..., 1].contiguous(),
            mr[..., 0].contiguous(),
            mr[..., 1].contiguous(),
            weights.contiguous(),
            omega.contiguous(),
            k.contiguous(),
            float(eps_abs.detach()),
            float(mu_abs.detach()),
            error,
            out,
            m_count,
            p_count,
            block,
            int(block_s),
        )
        electric_out = torch.view_as_complex(out[0].contiguous())
        magnetic_out = torch.view_as_complex(out[1].contiguous())
        if error.item():
            raise ValueError(
                "observation points must not coincide with surface samples"
            )
        ctx.save_for_backward(
            positions,
            normals,
            weights,
            electric,
            magnetic,
            frequencies,
            observations,
            eps_r,
            mu_r,
        )
        return electric_out, magnetic_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_e, grad_h):
        """Recompute a bounded Torch graph and apply its first-order VJP.

        Args:
            ctx: Saved input tensors and tile configuration.
            grad_e: Electric output cotangent, shape ``(F,M,3)``.
            grad_h: Magnetic output cotangent, shape ``(F,M,3)``.

        Returns:
            Cotangents in forward-input order, with None for nondifferentiable
            inputs and tile sizes. Higher-order differentiation is disabled.
        """
        saved = ctx.saved_tensors
        originals = list(saved)
        with torch.enable_grad():
            variables = []
            replacements = []
            for index, value in enumerate(originals):
                want_grad = ctx.needs_input_grad[index]
                fresh = value.detach().requires_grad_(want_grad)
                replacements.append(fresh)
                if want_grad:
                    variables.append(fresh)
            (
                positions,
                normals,
                weights,
                electric,
                magnetic,
                frequencies,
                observations,
                eps_r,
                mu_r,
            ) = replacements
            surface = Surface(
                positions, normals, weights, electric, magnetic, frequencies
            )
            e_ref, h_ref = project_nearfield(
                surface,
                observations,
                eps_r,
                mu_r,
                source_chunk_size=256,
                observation_chunk_size=128,
                checkpoint_chunks=True,
            )
            grads = torch.autograd.grad(
                (e_ref, h_ref),
                variables,
                (grad_e, grad_h),
                allow_unused=True,
                create_graph=False,
            )
        iterator = iter(grads)
        result = [next(iterator) if ctx.needs_input_grad[i] else None for i in range(9)]
        return (*result, None, None)


def project_nearfield_triton(
    surface: Surface,
    observation_points_m: torch.Tensor,
    eps_r: float | torch.Tensor = 1.0,
    mu_r: float | torch.Tensor = 1.0,
    *,
    observation_block_size: int = 32,
    source_block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate exact homogeneous E/H with a fused Triton CUDA kernel.

    Args:
        surface: Sampled source surface. Fields must be complex64 and geometry
            float32, all on a CUDA device.
        observation_points_m: Real float32 CUDA points, shape ``(M,3)``.
        eps_r: Positive scalar relative permittivity.
        mu_r: Positive scalar relative permeability.
        observation_block_size: Number of observations accumulated by one CTA.
        source_block_size: Number of source samples reduced by one kernel tile.

    Returns:
        Complex64 electric and magnetic fields, each shaped ``(F,M,3)``.

    Raises:
        RuntimeError: If Triton is unavailable or CUDA is not selected.
        TypeError: If the dtype/device combination is unsupported.
        ValueError: If positions, material parameters, or block size are invalid.

    Notes:
        The forward pass streams source tiles through observation tiles and
        never forms a full source-by-observation array. The custom adjoint recomputes
        the native PyTorch kernel in chunks, so backward memory stays bounded
        but is not Triton-accelerated. First-order gradients cover fields,
        geometry, weights, observations, frequencies, and tensor-valued
        material parameters. Higher-order gradients are not supported.
    """
    if triton is None:
        raise RuntimeError("direct_triton requires the optional 'triton' package")
    tensors = (
        surface.positions_m,
        surface.normals,
        surface.weights_m2,
        surface.electric_fields,
        surface.magnetic_fields,
        surface.frequencies_hz,
        observation_points_m,
    )
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("direct_triton requires all inputs on CUDA")
    if (
        surface.electric_fields.dtype != torch.complex64
        or surface.magnetic_fields.dtype != torch.complex64
    ):
        raise TypeError("direct_triton currently supports complex64 fields only")
    if any(
        t.dtype != torch.float32
        for t in (
            surface.positions_m,
            surface.normals,
            surface.weights_m2,
            surface.frequencies_hz,
            observation_points_m,
        )
    ):
        raise TypeError("direct_triton currently supports float32 geometry only")
    if (
        observation_points_m.ndim != 2
        or observation_points_m.shape[-1] != 3
        or observation_points_m.shape[0] == 0
    ):
        raise ValueError("observation_points_m must have shape (M,3) with M > 0")
    if (
        observation_points_m.is_complex()
        or not torch.isfinite(observation_points_m).all()
    ):
        raise ValueError("observation_points_m must be finite and real")
    if (
        observation_block_size <= 0
        or observation_block_size > 256
        or observation_block_size & (observation_block_size - 1)
    ):
        raise ValueError("observation_block_size must be a power of two in [1,256]")
    if (
        source_block_size <= 0
        or source_block_size > 128
        or source_block_size & (source_block_size - 1)
    ):
        raise ValueError("source_block_size must be a power of two in [1,128]")
    if any(isinstance(value, complex) for value in (eps_r, mu_r)) or any(
        isinstance(value, torch.Tensor) and value.is_complex()
        for value in (eps_r, mu_r)
    ):
        raise TypeError("eps_r and mu_r must be real")
    eps = torch.as_tensor(eps_r, dtype=torch.float32, device=surface.positions_m.device)
    mu = torch.as_tensor(mu_r, dtype=torch.float32, device=surface.positions_m.device)
    if (
        eps.numel() != 1
        or mu.numel() != 1
        or not torch.isfinite(eps).all()
        or not torch.isfinite(mu).all()
        or eps <= 0
        or mu <= 0
    ):
        raise ValueError("eps_r and mu_r must be finite positive scalars")
    return _TritonNearfield.apply(
        surface.positions_m,
        surface.normals,
        surface.weights_m2,
        surface.electric_fields,
        surface.magnetic_fields,
        surface.frequencies_hz,
        observation_points_m,
        eps,
        mu,
        observation_block_size,
        source_block_size,
    )
