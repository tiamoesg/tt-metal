# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 inference configuration.

`V4Config.full()` mirrors the published DeepSeek-V4-Pro config.json. `V4Config.small()`
is a tiny CPU-runnable shape for the reference model's smoke tests. The per-layer
`compress_ratios` schedule drives the hybrid attention: 4 -> CSA (compress + DSA
sparse selection), 128 -> HCA (heavy compress, dense), 0 -> dense full attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class V4Config:
    # vocab / model
    vocab_size: int = 129280
    dim: int = 7168
    moe_inter_dim: int = 3072
    n_layers: int = 61
    n_hash_layers: int = 3
    norm_eps: float = 1e-6

    # MoE
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"  # sqrtsoftplus | softmax | sigmoid
    route_scale: float = 2.5
    swiglu_limit: float = 10.0

    # attention (MLA-style low-rank Q + grouped low-rank O)
    n_heads: int = 128
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1536
    o_groups: int = 16
    o_lora_rank: int = 1024
    window_size: int = 128  # sliding-window of recent uncompressed tokens

    # lightning indexer (DSA)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 1024

    # mHC (Manifold-Constrained Hyper-Connections)
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # RoPE / YaRN
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    compress_rope_theta: float = 160000.0

    # runtime
    max_seq_len: int = 4096
    max_batch_size: int = 1

    # per-layer compression schedule (len == n_layers): 4=CSA, 128=HCA, 0=dense
    compress_ratios: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def qk_nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @classmethod
    def full(cls) -> "V4Config":
        ratios = [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4]
        ratios += [128, 4] * ((61 - len(ratios)) // 2)
        ratios = (ratios + [0])[:61]
        return cls(compress_ratios=tuple(ratios))

    @classmethod
    def small(cls) -> "V4Config":
        """A tiny config that runs on CPU and exercises CSA, HCA and dense layers."""
        return cls(
            vocab_size=256,
            dim=128,
            moe_inter_dim=128,
            n_layers=4,
            n_hash_layers=0,
            n_routed_experts=8,
            n_shared_experts=1,
            n_activated_experts=2,
            n_heads=4,
            head_dim=64,
            rope_head_dim=16,
            q_lora_rank=96,
            o_groups=2,
            o_lora_rank=64,
            window_size=8,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=4,
            hc_mult=4,
            max_seq_len=64,
            compress_ratios=(0, 4, 128, 0),  # dense, CSA, HCA, dense
        )

    def __post_init__(self):
        if self.compress_ratios and len(self.compress_ratios) != self.n_layers:
            raise ValueError(
                f"compress_ratios has {len(self.compress_ratios)} entries, expected n_layers={self.n_layers}"
            )
