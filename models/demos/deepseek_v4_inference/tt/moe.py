# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT-NN DeepSeekMoE FFN (roadmap step 2.8).

Validated against `reference_model.MoE` / `Gate` / `Expert`. A sqrtsoftplus (or
softmax / sigmoid) gate routes each token to its top-k of `n_routed_experts` SwiGLU
experts, plus always-on shared experts:

    scores  = score_func(x W_gate^T)                       [N, E]
    keep    = top-k(scores)                                (k = n_activated_experts)
    rweight = (keep ? scores : 0) ; if not softmax: / sum  ; * route_scale
    y       = sum_e rweight_e * Expert_e(x) + sum shared(x)
    Expert  : W2( silu(clamp W1 x) * clamp(W3 x) )         (SwiGLU, optional clamp)

This is the **dense-equivalent** of the token-choice routing the reference does with
gather/scatter: every token flows through every routed expert, weighted by `rweight`
(0 for unselected). Numerically identical, and the correct PCC gate before the
production swap to a DRAM-streamed top-k expert gather (b1 `deepseek_moe_gate` +
expert weights streamed from DRAM, never SRAM-pinned). Mirrors the tt-train C++
DeepSeekMoE.
"""

from __future__ import annotations

import torch

import ttnn

_NEG = -1.0e9


class TtExpert:
    def __init__(self, device, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, limit: float):
        to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.w1_t = to_dev(w1.t().contiguous())   # [dim, inter]
        self.w2_t = to_dev(w2.t().contiguous())   # [inter, dim]
        self.w3_t = to_dev(w3.t().contiguous())   # [dim, inter]
        self.limit = limit

    def __call__(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        gate = ttnn.matmul(x, self.w1_t)
        up = ttnn.matmul(x, self.w3_t)
        if self.limit > 0:
            gate = ttnn.clamp(gate, -1.0e30, self.limit)
            up = ttnn.clamp(up, -self.limit, self.limit)
        return ttnn.matmul(ttnn.multiply(ttnn.silu(gate), up), self.w2_t)


class TtGate:
    def __init__(self, device, weight: torch.Tensor, topk: int, route_scale: float, score_func: str):
        self.weight_t = ttnn.from_torch(
            weight.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )                                          # [dim, n_routed_experts]
        self.topk = topk
        self.route_scale = route_scale
        self.score_func = score_func

    def __call__(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        """x: [b, 1, S, dim] -> dense routing weights [b, 1, S, E] (0 for unselected)."""
        logits = ttnn.matmul(x, self.weight_t)     # [b, 1, S, E]
        if self.score_func == "softmax":
            scores = ttnn.softmax(logits, dim=-1)
        elif self.score_func == "sigmoid":
            scores = ttnn.sigmoid(logits)
        else:  # sqrtsoftplus
            scores = ttnn.sqrt(ttnn.softplus(logits))

        e = scores.shape[-1]
        k = min(self.topk, e)
        vals, _ = ttnn.topk(scores, k=k, dim=-1, largest=True, sorted=True)
        threshold = ttnn.repeat(vals[:, :, :, k - 1:k], ttnn.Shape((1, 1, 1, e)))  # k-th largest
        keep = ttnn.ge(ttnn.subtract(scores, threshold), 0.0)                      # 0/1 [b,1,S,E]
        weights = ttnn.multiply(scores, keep)
        if self.score_func != "softmax":
            denom = ttnn.add(ttnn.sum(weights, dim=-1, keepdim=True), 1.0e-20)
            weights = ttnn.div(weights, denom)
        return ttnn.multiply(weights, self.route_scale)


class TtMoE:
    def __init__(self, device, gate_weight, experts, shared, topk, route_scale, score_func,
                 dim, inter, limit):
        """`experts`/`shared`: lists of (w1, w2, w3) torch weight triples."""
        self.gate = TtGate(device, gate_weight, topk, route_scale, score_func)
        self.experts = [TtExpert(device, *w, limit) for w in experts]
        self.shared = [TtExpert(device, *w, limit) for w in shared]

    def __call__(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        """x: [b, 1, S, dim] -> [b, 1, S, dim]."""
        rweights = self.gate(x)                                  # [b, 1, S, E]
        y = self.shared[0](x)
        for s in self.shared[1:]:
            y = ttnn.add(y, s(x))
        for e, expert in enumerate(self.experts):
            y = ttnn.add(y, ttnn.multiply(expert(x), rweights[:, :, :, e:e + 1]))  # weight broadcast over dim
        return y
