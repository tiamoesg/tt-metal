# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT-NN grouped low-rank attention output projection (roadmap step 2.6).

Validated against `reference_model.Attention`'s `self.wo_b(self.wo_a(o))`. The V4
output projection is factored low-rank with `o_groups` parallel ranks:

    o   : [b, S, H*head_dim]      (concatenated head outputs, post "-i" RoPE)
    a   = o  W_a^T                [b, S, o_groups * o_lora_rank]
    out = a  W_b^T                [b, S, dim]

Two plain matmuls; mirrors the tt-train C++ GroupedOutputProjection.
"""

from __future__ import annotations

import torch

import ttnn


class TtOutputProjection:
    def __init__(self, device, wo_a: torch.Tensor, wo_b: torch.Tensor):
        to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # torch Linear weight is [out, in]; store transposed for x @ W^T
        self.wo_a_t = to_dev(wo_a.t().contiguous())   # [H*head_dim, o_groups*o_lora_rank]
        self.wo_b_t = to_dev(wo_b.t().contiguous())   # [o_groups*o_lora_rank, dim]

    def __call__(self, o: "ttnn.Tensor") -> "ttnn.Tensor":
        """o: [b, 1, S, H*head_dim] -> [b, 1, S, dim]."""
        return ttnn.matmul(ttnn.matmul(o, self.wo_a_t), self.wo_b_t)
