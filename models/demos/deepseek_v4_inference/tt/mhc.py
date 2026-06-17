# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT-NN Manifold-Constrained Hyper-Connections (mHC) residual mixing (roadmap 2.7).

Validated against `reference_model.Block._hc_pre` / `_hc_post` / `hc_split_sinkhorn`.
mHC widens the residual stream into `hc` parallel copies and, per token, learns how
to (a) blend them into the sublayer input (`pre`) and (b) write the sublayer output
back across the streams (`post` + a doubly-stochastic `comb` mixing matrix):

    xf      = concat(streams)                              [b, S, hc*d]
    mixes   = (xf W_fn^T) * rsqrt(mean(xf^2))              [b, S, mix_hc]
    pre     = sigmoid(s0*mixes[:hc]   + base[:hc])         [b, S, hc]
    post    = 2*sigmoid(s2*mixes[hc:2hc] + base[hc:2hc])   [b, S, hc]
    comb    = sinkhorn(s1*mixes[2hc:] + base[2hc:])        [b, S, hc, hc]  (doubly stochastic)
    y       = sum_k pre_k * stream_k                       (pre-combine -> sublayer input)
    stream'_j = post_j * h + sum_k comb_{k,j} * stream_k   (post-expand <- sublayer output)

Streams are carried as a python list of `hc` tensors `[b, 1, S, d]` so the per-token
`comb` mixing is an explicit weighted sum (avoids a 5-D contraction). The fused mix
projection is split into three matmuls at construction (`fn[:hc] | fn[hc:2hc] |
fn[2hc:]`) so no tile-unaligned device slice of the mix vector is needed at runtime.
Mirrors the tt-train C++ ManifoldHyperConnections.

NOTE (flagged in the reference): the element split / `scale`-index mapping
(`pre<-scale[0]`, `comb<-scale[1]`, `post<-scale[2]`) is reconstructed from the paper
and should be reconciled with the official kernel.py before loading checkpoints.
"""

from __future__ import annotations

import torch

import ttnn


def _split_fn(fn: torch.Tensor, base: torch.Tensor, hc: int):
    """Split the fused mix weight [mix_hc, hc*d] and bias [mix_hc] into pre/post/comb.

    Returns ((fn_pre, fn_post, fn_comb), (base_pre, base_post, base_comb)) where the
    row ranges match `hc_split_sinkhorn`'s `mixes[..., :hc] / [hc:2hc] / [2hc:]`.
    """
    fn_pre, fn_post, fn_comb = fn[:hc], fn[hc:2 * hc], fn[2 * hc:2 * hc + hc * hc]
    base_pre, base_post, base_comb = base[:hc], base[hc:2 * hc], base[2 * hc:2 * hc + hc * hc]
    return (fn_pre, fn_post, fn_comb), (base_pre, base_post, base_comb)


class TtHyperConnections:
    def __init__(self, device, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor,
                 hc: int, eps: float, iters: int):
        self.device = device
        self.hc = hc
        self.eps = eps
        self.iters = iters
        # scale is a 3-vector parameter; fixed at inference -> python floats.
        self.s_pre, self.s_comb, self.s_post = (float(v) for v in scale.tolist())

        to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        (fn_pre, fn_post, fn_comb), (b_pre, b_post, b_comb) = _split_fn(fn, base, hc)
        # store fn transposed for xf @ W^T (each its own matmul -> no runtime mix slice)
        self.fn_pre_t = to_dev(fn_pre.t().contiguous())     # [hc*d, hc]
        self.fn_post_t = to_dev(fn_post.t().contiguous())   # [hc*d, hc]
        self.fn_comb_t = to_dev(fn_comb.t().contiguous())   # [hc*d, hc*hc]
        self.base_pre = to_dev(b_pre.reshape(1, 1, 1, hc))
        self.base_post = to_dev(b_post.reshape(1, 1, 1, hc))
        self.base_comb = to_dev(b_comb.reshape(1, 1, 1, hc * hc))

    def _rsqrt(self, xf: "ttnn.Tensor") -> "ttnn.Tensor":
        ms = ttnn.mean(ttnn.multiply(xf, xf), dim=-1, keepdim=True)   # [b, 1, S, 1]
        return ttnn.rsqrt(ttnn.add(ms, self.eps))

    def _sinkhorn(self, logits: "ttnn.Tensor") -> "ttnn.Tensor":
        """[b, S, hc, hc] -> doubly-stochastic via alternating col/row normalize."""
        m = ttnn.exp(logits)
        for _ in range(self.iters):
            m = ttnn.div(m, ttnn.sum(m, dim=-2, keepdim=True))   # column normalize (over k)
            m = ttnn.div(m, ttnn.sum(m, dim=-1, keepdim=True))   # row normalize (over j)
        return m

    def mix(self, streams: list) -> tuple:
        """streams: list of `hc` tensors [b,1,S,d] -> (pre[b,1,S,hc], post[b,1,S,hc], comb[b,S,hc,hc])."""
        hc = self.hc
        xf = ttnn.concat(streams, dim=-1)                # [b, 1, S, hc*d]
        rsqrt = self._rsqrt(xf)
        proj = lambda fn_t: ttnn.multiply(ttnn.matmul(xf, fn_t), rsqrt)

        pre = ttnn.sigmoid(ttnn.add(ttnn.multiply(proj(self.fn_pre_t), self.s_pre), self.base_pre))
        post = ttnn.multiply(
            ttnn.sigmoid(ttnn.add(ttnn.multiply(proj(self.fn_post_t), self.s_post), self.base_post)), 2.0
        )
        comb_raw = ttnn.add(ttnn.multiply(proj(self.fn_comb_t), self.s_comb), self.base_comb)  # [b,1,S,hc*hc]
        b, _, s, _ = comb_raw.shape
        comb = self._sinkhorn(ttnn.reshape(comb_raw, (b, s, hc, hc)))
        return pre, post, comb

    def combine(self, streams: list, pre: "ttnn.Tensor") -> "ttnn.Tensor":
        """Pre-combine streams into the sublayer input: y = sum_k pre_k * stream_k -> [b,1,S,d]."""
        y = ttnn.multiply(streams[0], pre[:, :, :, 0:1])
        for k in range(1, self.hc):
            y = ttnn.add(y, ttnn.multiply(streams[k], pre[:, :, :, k:k + 1]))
        return y

    def expand(self, h: "ttnn.Tensor", streams: list, post: "ttnn.Tensor",
               comb: "ttnn.Tensor") -> list:
        """Post-expand the sublayer output `h` [b,1,S,d] back across the `hc` streams.

        stream'_j = post_j * h + sum_k comb_{k,j} * stream_k.  Returns list of hc [b,1,S,d].
        """
        b, _, s, _ = h.shape
        out = []
        for j in range(self.hc):
            acc = ttnn.multiply(h, post[:, :, :, j:j + 1])
            for k in range(self.hc):
                ckj = ttnn.reshape(comb[:, :, k:k + 1, j:j + 1], (b, 1, s, 1))   # [b,1,S,1] per-token weight
                acc = ttnn.add(acc, ttnn.multiply(streams[k], ckj))
            out.append(acc)
        return out
