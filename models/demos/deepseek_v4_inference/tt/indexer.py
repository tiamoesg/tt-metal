# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT-NN device implementation of the V4 lightning indexer + top-k (roadmap step 2.3).

Validated against `reference_model.Indexer`. Cheaply scores how relevant each
compressed KV block is to each query and selects the top-k per query:

    keys     = Compressor(x)                                   [b, G, c^I]
    q^I      = (qr W^IUQ) split into H heads                   [b, H, S, c^I]
    w^I      = x W^w * (scale * H^-0.5)                        [b, H, S, 1]
    I_{t,s}  = sum_h w^I_{t,h} * ReLU(q^I_{t,h} . key_s)       [b, 1, S, G]   (scores)
    keep     = top-k(I) AND (s < (t+1)//ratio)                 [b, 1, S, G]   (mask, stop-grad)

`scores()` is the continuous, PCC-checkable part; `select()` does the causal + top-k
(bf16-fragile, validated by agreement). Mirrors the tt-train C++ LightningIndexer.
"""

from __future__ import annotations

import torch

import ttnn

from models.demos.deepseek_v4_inference.tt.compressor import TtCompressor

_NEG = -1.0e9


def _split_heads(x, n_heads, head_dim):
    """[b, 1, S, H*hd] -> [b, H, S, hd] (head-major, like MLA split_heads)."""
    b, _, s, _ = x.shape
    x = ttnn.reshape(x, (b, s, n_heads, head_dim))
    return ttnn.permute(x, (0, 2, 1, 3))


def compressed_causal_keep(seq: int, groups: int, ratio: int) -> torch.Tensor:
    """[1, 1, S, G] 0/1: block s visible to query t iff s < (t+1)//ratio."""
    t = torch.arange(seq)
    s = torch.arange(groups)
    keep = (s[None, :] < ((t[:, None] + 1) // ratio)).float()
    return keep.reshape(1, 1, seq, groups)


class TtIndexer:
    def __init__(self, device, wq: torch.Tensor, wproj: torch.Tensor, compressor: TtCompressor,
                 n_heads: int, head_dim: int, scale: float, ratio: int, topk: int):
        self.device = device
        self.compressor = compressor
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.weight_scale = scale * n_heads ** -0.5
        self.ratio = ratio
        self.topk = topk
        to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.wq_t = to_dev(wq.t().contiguous())        # [q_lora, H*c^I]
        self.wproj_t = to_dev(wproj.t().contiguous())  # [dim, H]

    def scores(self, x: "ttnn.Tensor", qr: "ttnn.Tensor") -> "ttnn.Tensor":
        """x: [b,1,S,d], qr: [b,1,S,q_lora] -> scores [b, 1, S, G]."""
        b, _, s, _ = x.shape
        keys = self.compressor(x)                              # [b, 1, G, c^I]
        g = keys.shape[2]
        qi = _split_heads(ttnn.matmul(qr, self.wq_t), self.n_heads, self.head_dim)  # [b, H, S, c^I]
        wi = _split_heads(ttnn.multiply(ttnn.matmul(x, self.wproj_t), self.weight_scale), self.n_heads, 1)  # [b,H,S,1]
        keys_h = ttnn.repeat(keys, ttnn.Shape((1, self.n_heads, 1, 1)))             # [b, H, G, c^I]
        dots = ttnn.matmul(qi, ttnn.transpose(keys_h, -2, -1))                      # [b, H, S, G]
        weighted = ttnn.multiply(ttnn.relu(dots), wi)                              # broadcast [b,H,S,1] over G
        return ttnn.sum(weighted, dim=1, keepdim=True)                             # [b, 1, S, G]

    def select(self, scores: "ttnn.Tensor") -> "ttnn.Tensor":
        """scores [b,1,S,G] -> 0/1 keep mask [b,1,S,G] (top-k AND compressed-causal)."""
        b, _, s, g = scores.shape
        k = min(self.topk, g)
        causal = ttnn.from_torch(
            compressed_causal_keep(s, g, self.ratio).expand(b, 1, s, g).contiguous(),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        masked = ttnn.where(causal, scores, _NEG)
        vals, _ = ttnn.topk(masked, k=k, dim=-1, largest=True, sorted=True)
        threshold = ttnn.repeat(vals[:, :, :, k - 1 : k], ttnn.Shape((1, 1, 1, g)))  # k-th largest
        selected = ttnn.ge(ttnn.subtract(masked, threshold), 0.0)
        return ttnn.multiply(selected, causal)
