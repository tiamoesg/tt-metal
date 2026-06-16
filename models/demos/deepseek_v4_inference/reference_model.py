# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-runnable PyTorch reference for DeepSeek-V4 inference.

This is the architecture-faithful ground truth that the TT-NN (Tenstorrent) port
validates against -- the same role `reference/` plays for the deepseek_v3 demo. It
is adapted from the official DeepSeek-V4 `model.py`, with the production concerns
removed so it runs anywhere:
  * FP8/FP4 quantization, Hadamard rotation, scale formats     -> plain bf16/fp32
  * tensor/expert/pipeline parallelism (world_size)            -> single process
  * the incremental KV-cache decode state machine              -> full-sequence
    forward; `generate()` is autoregressive by recompute (O(S^2), correct, simple)

What it keeps (the V4 architecture):
  * Hybrid attention per layer via `compress_ratios`: dense (0), CSA (4: compress +
    lightning-indexer top-k sparse selection), HCA (128: heavy compress, dense)
  * Token-level KV compression (overlapping two-series for CSA, single-series for HCA)
  * Lightning indexer (low-rank q, per-head ReLU-weighted scores) + top-k selection
  * Sliding-window branch, attention sink, partial RoPE with the "-i" output trick
  * Manifold-Constrained Hyper-Connections (mHC) residual mixing
  * DeepSeekMoE FFN (sqrtsoftplus gate, routed + shared SwiGLU experts)

Correctness over speed: the sparse attention is implemented as a dense masked
reference (a [S, G] keep-mask), not a gather kernel. The TT port replaces this with
a real top-k gather (see README).

NOTE: a few fusion details the paper defers to its kernels (e.g. the exact element
split inside `hc_split_sinkhorn`) are reconstructed faithfully but flagged inline;
reconcile with the official kernel.py before loading their checkpoints.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.demos.deepseek_v4_inference.config import V4Config


# --------------------------------------------------------------------------------------
# Norm + RoPE
# --------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def precompute_freqs_cis(dim: int, seqlen: int, original_seq_len: int, base: float, factor: float,
                         beta_fast: int, beta_slow: int) -> torch.Tensor:
    """YaRN-scaled rotary frequencies as complex exponentials, shape [seqlen, dim/2]."""

    def correction_dim(num_rotations, d, b, max_seq):
        return d * math.log(max_seq / (num_rotations * 2 * math.pi)) / (2 * math.log(b))

    def correction_range(low_rot, high_rot, d, b, max_seq):
        low = math.floor(correction_dim(low_rot, d, b, max_seq))
        high = math.ceil(correction_dim(high_rot, d, b, max_seq))
        return max(low, 0), min(high, d - 1)

    def ramp(lo, hi, d):
        if lo == hi:
            hi += 0.001
        f = (torch.arange(d, dtype=torch.float32) - lo) / (hi - lo)
        return torch.clamp(f, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - ramp(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate the last dim of `x` ([..., S, rope_dim]) by `freqs_cis` ([S, rope_dim/2]).

    `inverse=True` rotates by -theta (the V4 "-i" output trick)."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    fc = freqs_cis.conj() if inverse else freqs_cis
    # broadcast freqs over leading dims: x is [..., S, rope_dim] -> complex [..., S, rope_dim/2]
    while fc.dim() < xc.dim():
        fc = fc.unsqueeze(0)
    return torch.view_as_real(xc * fc).flatten(-2).to(x.dtype)


def partial_rope(x: torch.Tensor, freqs_cis: torch.Tensor, rope_dim: int, inverse: bool = False) -> torch.Tensor:
    """Apply RoPE to the last `rope_dim` dims of x, leave the prefix unchanged."""
    if rope_dim == 0:
        return x
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    return torch.cat([nope, apply_rotary_emb(rope, freqs_cis, inverse)], dim=-1)


# --------------------------------------------------------------------------------------
# KV compression (CSA overlapping two-series; HCA single-series)
# --------------------------------------------------------------------------------------
class Compressor(nn.Module):
    """Pool every `ratio` tokens into one compressed entry via softmax-gated pooling.

    overlap=True (CSA, ratio==4): each entry pools a 2*ratio window (current + previous
    segment) under a joint softmax. overlap=False (HCA): non-overlapping single series.
    """

    def __init__(self, cfg: V4Config, ratio: int, head_dim: int):
        super().__init__()
        self.ratio = ratio
        self.head_dim = head_dim
        self.overlap = ratio == 4
        coff = 1 + int(self.overlap)
        self.wkv = nn.Linear(cfg.dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(cfg.dim, coff * head_dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(ratio, coff * head_dim))  # learnable positional bias
        self.norm = RMSNorm(head_dim, cfg.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        r, d = self.ratio, self.head_dim
        g = s // r
        x = x[:, : g * r]  # drop tail that doesn't fill a segment (reference simplification)
        kv = self.wkv(x).float()
        score = self.wgate(x).float()
        kv = kv.unflatten(1, (g, r))          # [b, g, r, coff*d]
        score = score.unflatten(1, (g, r)) + self.ape
        if self.overlap:
            kv = self._overlap(kv, 0.0)        # [b, g, 2r, d]
            score = self._overlap(score, float("-inf"))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)  # [b, g, d]
        return self.norm(kv.to(x.dtype))

    def _overlap(self, t: torch.Tensor, pad: float) -> torch.Tensor:
        # t: [b, g, r, 2d] -> [b, g, 2r, d]: previous segment's first half + current second half.
        b, g, r, _ = t.shape
        d = self.head_dim
        out = t.new_full((b, g, 2 * r, d), pad)
        out[:, :, r:] = t[:, :, :, d:]        # current segment, "b" series
        out[:, 1:, :r] = t[:, :-1, :, :d]     # previous segment, "a" series
        return out


# --------------------------------------------------------------------------------------
# Lightning indexer (DSA top-k selection)
# --------------------------------------------------------------------------------------
class Indexer(nn.Module):
    def __init__(self, cfg: V4Config, ratio: int):
        super().__init__()
        self.ratio = ratio
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.topk = cfg.index_topk
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(cfg.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wproj = nn.Linear(cfg.dim, self.n_heads, bias=False)
        self.compressor = Compressor(cfg, ratio, self.head_dim)

    def forward(self, x: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
        """Returns a [b, S, G] boolean keep-mask of the top-k compressed blocks per query."""
        b, s, _ = x.shape
        keys = self.compressor(x)                         # [b, G, c^I]
        g = keys.shape[1]
        q = self.wq(qr).unflatten(-1, (self.n_heads, self.head_dim))  # [b, S, H, c^I]
        weights = self.wproj(x) * (self.scale * self.n_heads ** -0.5)  # [b, S, H]
        # I_{t,s} = sum_h w_{t,h} * ReLU(q_{t,h} . key_s)
        score = torch.einsum("bshd,bgd->bshg", q, keys).relu()
        score = (score * weights.unsqueeze(-1)).sum(dim=2)            # [b, S, G]
        # compressed-causal: block s visible iff s < (t+1)//ratio
        t = torch.arange(s, device=x.device)
        causal = (torch.arange(g, device=x.device)[None, :] < ((t[:, None] + 1) // self.ratio))  # [S, G]
        score = score.masked_fill(~causal[None], float("-inf"))
        k = min(self.topk, g)
        idx = score.topk(k, dim=-1).indices                          # [b, S, k]
        keep = torch.zeros(b, s, g, dtype=torch.bool, device=x.device)
        keep.scatter_(-1, idx, True)
        keep &= causal[None]
        return keep


# --------------------------------------------------------------------------------------
# Attention (dense / CSA / HCA) with sliding window, sink, partial RoPE
# --------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: V4Config, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.ratio = cfg.compress_ratios[layer_id]
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = cfg.rope_head_dim
        self.window = cfg.window_size
        self.scale = self.head_dim ** -0.5

        self.wq_a = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        # uncompressed per-token KV (used for sliding window + dense layers)
        self.wkv = nn.Linear(cfg.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        # grouped low-rank output projection
        self.wo_a = nn.Linear(self.n_heads * self.head_dim, cfg.o_groups * cfg.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(cfg.o_groups * cfg.o_lora_rank, cfg.dim, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads))

        if self.ratio > 0:
            self.compressor = Compressor(cfg, self.ratio, self.head_dim)
            self.indexer = Indexer(cfg, self.ratio) if self.ratio == 4 else None

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, freqs_comp: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim)).transpose(1, 2)  # [b, H, S, d]
        q = partial_rope(q, freqs, self.rope_dim)

        kv_unc = self.kv_norm(self.wkv(x))                                # [b, S, d] (per-token, uncompressed)
        kv_unc = partial_rope(kv_unc, freqs, self.rope_dim)

        t = torch.arange(s, device=x.device)
        if self.ratio == 0:
            keys = kv_unc                                                # [b, S, d]
            mask = (t[:, None] >= t[None, :])                            # full causal [S, S]
            mask = mask[None].expand(b, s, s)
        else:
            comp = self.compressor(x)                                    # [b, G, d]
            comp = partial_rope(comp, freqs_comp, self.rope_dim)         # blocks at positions s*ratio
            g = comp.shape[1]
            # sliding-window over recent uncompressed tokens
            win_mask = (t[:, None] >= t[None, :]) & (t[None, :] > t[:, None] - self.window)  # [S, S]
            if self.indexer is not None:
                comp_keep = self.indexer(x, qr)                          # [b, S, G] (top-k + causal)
            else:
                causal = (torch.arange(g, device=x.device)[None, :] < ((t[:, None] + 1) // self.ratio))
                comp_keep = causal[None].expand(b, s, g)
            keys = torch.cat([kv_unc, comp], dim=1)                      # [b, S+G, d]
            mask = torch.cat([win_mask[None].expand(b, s, s), comp_keep], dim=-1)  # [b, S, S+G]

        logits = torch.einsum("bhsd,bkd->bhsk", q, keys) * self.scale    # [b, H, S, K]
        logits = logits.masked_fill(~mask[:, None], float("-inf"))
        # softmax with per-head attention sink in the denominator
        m = logits.amax(dim=-1, keepdim=True)
        m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
        e = (logits - m).exp()
        denom = e.sum(-1, keepdim=True) + (self.attn_sink[None, :, None, None] - m).exp()
        w = e / denom
        o = torch.einsum("bhsk,bkd->bhsd", w, keys)                      # [b, H, S, d]
        o = partial_rope(o, freqs, self.rope_dim, inverse=True)         # "-i" output trick
        o = o.transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim)
        return self.wo_b(self.wo_a(o))


# --------------------------------------------------------------------------------------
# mHC residual mixing
# --------------------------------------------------------------------------------------
def _sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    m = torch.exp(logits)
    for _ in range(iters):
        m = m / m.sum(dim=-2, keepdim=True)  # column normalize (T_c)
        m = m / m.sum(dim=-1, keepdim=True)  # row normalize (T_r)
    return m


def hc_split_sinkhorn(mixes, scale, base, hc, iters):
    """Split the fused mHC mix vector into A (pre), C (post), B (comb) with constraints.

    ASSUMPTION (reconcile with official kernel.py for checkpoint loading): the mix
    vector is ordered [pre(hc) | post(hc) | comb(hc*hc)] and `scale` indexes
    (pre, comb, post)."""
    pre = torch.sigmoid(scale[0] * mixes[..., :hc] + base[:hc])
    post = 2 * torch.sigmoid(scale[2] * mixes[..., hc:2 * hc] + base[hc:2 * hc])
    comb_raw = scale[1] * mixes[..., 2 * hc:2 * hc + hc * hc] + base[2 * hc:2 * hc + hc * hc]
    comb = _sinkhorn(comb_raw.unflatten(-1, (hc, hc)), iters)
    return pre, post, comb


class Block(nn.Module):
    def __init__(self, cfg: V4Config, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.hc = cfg.hc_mult
        self.iters = cfg.hc_sinkhorn_iters
        self.eps = cfg.hc_eps
        self.attn = Attention(cfg, layer_id)
        self.ffn = MoE(cfg)
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        mix_hc = (2 + self.hc) * self.hc
        hc_dim = self.hc * cfg.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim).normal_(std=0.02))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim).normal_(std=0.02))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.full((3,), 0.01))
        self.hc_ffn_scale = nn.Parameter(torch.full((3,), 0.01))

    def _hc_pre(self, x, fn, scale, base):
        # x: [b, s, hc, d] -> combined [b, s, d], plus post/comb for hc_post
        shape = x.shape
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(xf, fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, scale, base, self.hc, self.iters)
        y = (pre.unsqueeze(-1) * x.view(shape)).sum(dim=2)
        return y.to(x.dtype), post, comb

    @staticmethod
    def _hc_post(x, residual, post, comb):
        # x: [b, s, d], residual: [b, s, hc, d] -> [b, s, hc, d]
        return post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )

    def forward(self, x, freqs, freqs_comp):
        residual = x
        h, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.attn(self.attn_norm(h), freqs, freqs_comp)
        x = self._hc_post(h, residual, post, comb)

        residual = x
        h, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.ffn(self.ffn_norm(h))
        x = self._hc_post(h, residual, post, comb)
        return x


# --------------------------------------------------------------------------------------
# DeepSeekMoE FFN
# --------------------------------------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, dim, inter, limit):
        super().__init__()
        self.w1, self.w2, self.w3 = (nn.Linear(dim, inter, bias=False), nn.Linear(inter, dim, bias=False),
                                     nn.Linear(dim, inter, bias=False))
        self.limit = limit

    def forward(self, x):
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.limit > 0:
            gate, up = gate.clamp(max=self.limit), up.clamp(-self.limit, self.limit)
        return self.w2((F.silu(gate) * up).to(x.dtype))


class Gate(nn.Module):
    def __init__(self, cfg: V4Config):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.dim).normal_(std=0.02))
        self.topk = cfg.n_activated_experts
        self.route_scale = cfg.route_scale

    def forward(self, x):
        scores = F.linear(x.float(), self.weight.float())
        if self.cfg.score_func == "softmax":
            scores = scores.softmax(-1)
        elif self.cfg.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).sqrt()
        idx = scores.topk(self.topk, dim=-1).indices
        weights = scores.gather(-1, idx)
        if self.cfg.score_func != "softmax":
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, idx


class MoE(nn.Module):
    def __init__(self, cfg: V4Config):
        super().__init__()
        self.cfg = cfg
        self.gate = Gate(cfg)
        self.experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit) for _ in range(cfg.n_routed_experts)]
        )
        self.shared = nn.ModuleList(
            [Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit) for _ in range(cfg.n_shared_experts)]
        )

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        weights, idx = self.gate(x)                 # [N, topk], [N, topk]
        y = sum(s(x) for s in self.shared)
        counts = torch.bincount(idx.flatten(), minlength=self.cfg.n_routed_experts)
        for e, expert in enumerate(self.experts):
            if counts[e] == 0:
                continue
            tok, slot = torch.where(idx == e)
            y[tok] += expert(x[tok]) * weights[tok, slot, None]
        return y.view(shape)


# --------------------------------------------------------------------------------------
# Transformer
# --------------------------------------------------------------------------------------
class DeepSeekV4(nn.Module):
    def __init__(self, cfg: V4Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        # mHC head reduction: hc copies -> 1 via sigmoid-gated weighted sum
        mix_hc = cfg.hc_mult
        self.hc_head_fn = nn.Parameter(torch.empty(mix_hc, cfg.hc_mult * cfg.dim).normal_(std=0.02))
        self.hc_head_base = nn.Parameter(torch.zeros(mix_hc))
        self.hc_head_scale = nn.Parameter(torch.full((1,), 0.01))
        # RoPE caches: main (token positions) and compressed (theta differs)
        self.register_buffer(
            "freqs",
            precompute_freqs_cis(cfg.rope_head_dim, cfg.max_seq_len, cfg.original_seq_len, cfg.rope_theta,
                                 cfg.rope_factor, cfg.beta_fast, cfg.beta_slow),
            persistent=False,
        )
        self.register_buffer(
            "freqs_comp",
            precompute_freqs_cis(cfg.rope_head_dim, cfg.max_seq_len, cfg.original_seq_len,
                                 cfg.compress_rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow),
            persistent=False,
        )

    def _hc_head(self, x):
        # x: [b, s, hc, d] -> [b, s, d]
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.cfg.hc_eps)
        mixes = F.linear(xf, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.cfg.hc_eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, s = tokens.shape
        x = self.embed(tokens)                                   # [b, s, d]
        x = x.unsqueeze(2).expand(b, s, self.cfg.hc_mult, self.cfg.dim).contiguous()  # hc streams
        freqs = self.freqs[:s]
        # compressed-block RoPE positions are s*ratio; the buffer is sliced per layer below
        for layer in self.layers:
            ratio = max(layer.attn.ratio, 1)
            freqs_comp = self.freqs_comp[: s * 0 + (s // ratio) * ratio : ratio][: s // ratio] if ratio > 1 else freqs
            x = layer(x, freqs, freqs_comp)
        x = self._hc_head(x)
        return self.head(self.norm(x))                           # [b, s, vocab]

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy autoregressive generation by recompute (reference; no KV cache)."""
        for _ in range(max_new_tokens):
            logits = self.forward(tokens[:, -self.cfg.max_seq_len :])
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens
