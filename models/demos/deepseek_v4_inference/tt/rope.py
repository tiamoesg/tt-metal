# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT-NN device implementation of DeepSeek-V4 partial RoPE (roadmap step 2.1).

Validated against `reference_model.partial_rope` (the CPU ground truth).

Convention note (important): DeepSeek's `apply_rotary_emb` rotates *interleaved* pairs
(x[2i], x[2i+1]) -- the GPT-NeoX/complex convention -- NOT the half-split convention
that ttnn's fused `rotary_embedding_llama` kernel uses. To match the reference exactly
with the same weights, we implement the rotation explicitly as

    out = x ⊙ cos  +  (x · R) ⊙ sin            (inverse / "-i": sin -> -sin)

where R is the constant block-diagonal matrix of 2x2 blocks [[0, 1], [-1, 0]] that maps
(x[2i], x[2i+1]) -> (-x[2i+1], x[2i]). cos/sin are the per-pair angles repeated twice
across the rope dimension. This is exact, tile-friendly (a constant rope_dim x rope_dim
matmul + elementwise), and avoids any size-2 pair slicing.

Only the last `rope_dim` of the head are rotated; the leading "no-PE" prefix passes
through. The partial split boundary (full - rope_dim) must be tile-aligned.
"""

from __future__ import annotations

import torch

import ttnn


def build_rotate_matrix(rope_dim: int) -> torch.Tensor:
    """R [rope_dim, rope_dim]: (x · R)[2i] = -x[2i+1], (x · R)[2i+1] = x[2i]."""
    r = torch.zeros(rope_dim, rope_dim)
    for i in range(0, rope_dim, 2):
        r[i, i + 1] = 1.0   # contributes to out[2i+1] = x[2i]
        r[i + 1, i] = -1.0  # contributes to out[2i]   = -x[2i+1]
    return r


def cos_sin_from_freqs(freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """freqs_cis [S, rope_dim/2] (complex) -> (cos, sin) [S, rope_dim], angles repeated per pair."""
    cos = freqs_cis.real.repeat_interleave(2, dim=-1)
    sin = freqs_cis.imag.repeat_interleave(2, dim=-1)
    return cos, sin


class TtRope:
    """Applies partial RoPE on device. cos/sin/R are uploaded once at construction.

    cos, sin: torch [S, rope_dim] (host) -> stored as ttnn [1, 1, S, rope_dim].
    Pass `cos`/`sin` built at the desired positions (contiguous token positions for
    queries / window KV; strided s*ratio for compressed blocks)."""

    def __init__(self, device, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int):
        self.device = device
        self.rope_dim = rope_dim
        to_dev = lambda t: ttnn.from_torch(
            t.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.cos = to_dev(cos)  # [1, 1, S, rope_dim]
        self.sin = to_dev(sin)
        self.rotate = ttnn.from_torch(
            build_rotate_matrix(rope_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, x: "ttnn.Tensor", inverse: bool = False) -> "ttnn.Tensor":
        """x: [B, H, S, full] -> same shape, last rope_dim dims rotated."""
        full = x.shape[-1]
        if full == self.rope_dim:
            nope, rope = None, x
        else:
            nope = x[:, :, :, : full - self.rope_dim]
            rope = x[:, :, :, full - self.rope_dim :]

        rot = ttnn.matmul(rope, self.rotate)                       # x · R
        term_cos = ttnn.multiply(rope, self.cos)                   # broadcasts [1,1,S,r] over B,H
        term_sin = ttnn.multiply(rot, self.sin)
        out_rope = ttnn.subtract(term_cos, term_sin) if inverse else ttnn.add(term_cos, term_sin)
        return out_rope if nope is None else ttnn.concat([nope, out_rope], dim=-1)
