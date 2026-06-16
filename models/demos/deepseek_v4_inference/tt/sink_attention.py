# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT-NN shared-KV MQA + attention sink (roadmap step 2.4, the math core).

Validated against `reference_model.masked_sink_attention`. This is the *math* the
production top-k DRAM-gather decode kernel realizes: a query attends over the kept
compressed/window KV (here selected by a 0/1 keep mask rather than physically
gathered), with a per-head attention sink added to the softmax denominator so
fully-masked rows are numerically safe (-> 0).

    L = (q . kv) / sqrt(hd) + (keep-1)*BIG          # masked logits
    w = exp(L - m) / ( sum exp(L - m) + exp(sink - m) )     (m = rowmax, for stability)
    o = w @ kv

The production realization (next sub-step) replaces the dense [S, K] mask with a
gather of only the top-k blocks from DRAM (extend deepseek_v3_b1 flash_mla +
kv_cache_update). Mirrors the tt-train C++ shared_kv_mqa_attention.
"""

from __future__ import annotations

import ttnn

_BIG = 1.0e9


def sink_attention(q: "ttnn.Tensor", keys: "ttnn.Tensor", keep: "ttnn.Tensor",
                   sink: "ttnn.Tensor", scale: float) -> "ttnn.Tensor":
    """q: [b,H,S,hd], keys (=values): [b,1,K,hd], keep: [b,1,S,K] (0/1), sink: [1,H,1,1].
    Returns [b,H,S,hd]."""
    b, heads, s, hd = q.shape
    keys_h = ttnn.repeat(keys, ttnn.Shape((1, heads, 1, 1)))             # [b, H, K, hd] (MQA share)
    logits = ttnn.multiply(ttnn.matmul(q, ttnn.transpose(keys_h, -2, -1)), scale)  # [b, H, S, K]

    bias = ttnn.multiply(ttnn.subtract(keep, 1.0), _BIG)                # 0 (keep) / -BIG (drop), [b,1,S,K]
    logits = ttnn.add(logits, bias)                                    # broadcast over H

    m = ttnn.max(logits, dim=-1, keepdim=True)                         # [b, H, S, 1] row max (stability)
    e = ttnn.exp(ttnn.subtract(logits, m))
    sink_term = ttnn.exp(ttnn.subtract(sink, m))                      # exp(sink_h - m), broadcast over S,K->1
    denom = ttnn.add(ttnn.sum(e, dim=-1, keepdim=True), sink_term)
    w = ttnn.div(e, denom)
    return ttnn.matmul(w, keys_h)                                      # [b, H, S, hd]
