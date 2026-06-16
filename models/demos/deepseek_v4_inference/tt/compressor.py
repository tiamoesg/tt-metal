# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT-NN device implementation of the V4 KV compressor (roadmap step 2.2).

Validated against `reference_model.Compressor`. Pools every `ratio` tokens into one
compressed entry via softmax-gated pooling:

    kv = wkv(x); score = wgate(x)                          [b, s, coff*head_dim]
    kv, score = unflatten to segments [b, g, ratio, ...] ; score += ape
    (CSA only, overlap) widen each segment to a 2*ratio window:
        previous segment's "a" half + current segment's "b" half  (segment 0 padded)
    kv = (kv * softmax(score over the segment dim)).sum(segment) ; RMSNorm

`overlap=True` is CSA (ratio==4); single-series is HCA. Mirrors the reference exactly
op-for-op so the PCC test aligns. Segments fold into the height dim ([b, g, ratio, .])
exactly as the tt-train C++ KVCompressor does; ttnn pads the small `ratio` dim.
"""

from __future__ import annotations

import torch

import ttnn

_NEG = -1.0e9  # softmax-suppressed padding (avoids -inf -> NaN on device)


def overlap_window(a_half, b_half, pad_seg, *, slice_g, concat_g, concat_seg):
    """Build the 2*ratio window from the two half-series (backend-agnostic).

    a_half/b_half: [b, g, ratio, d]. Returns [b, g, 2*ratio, d] where segment i =
    [previous segment's a-half (padded for i=0) | current segment's b-half].
    The slice/concat callables let this same logic be unit-tested with plain lists.
    """
    a_prev = concat_g(pad_seg, slice_g(a_half))  # prepend pad segment, drop last -> shift by one
    return concat_seg(a_prev, b_half)


class TtCompressor:
    def __init__(self, device, wkv: torch.Tensor, wgate: torch.Tensor, ape: torch.Tensor,
                 norm_weight: torch.Tensor, ratio: int, head_dim: int, eps: float = 1e-6):
        self.device = device
        self.ratio = ratio
        self.head_dim = head_dim
        self.eps = eps
        self.overlap = ratio == 4
        self.coff = 1 + int(self.overlap)

        def to_dev(t, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device)

        # store weights transposed for x @ W^T (torch Linear weight is [out, in])
        self.wkv_t = to_dev(wkv.t().contiguous())          # [dim, coff*head_dim]
        self.wgate_t = to_dev(wgate.t().contiguous())
        self.ape = to_dev(ape.reshape(1, 1, ratio, self.coff * head_dim))  # broadcast over (b, g)
        self.norm_weight = to_dev(norm_weight)

    def __call__(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        """x: [b, 1, s, dim] -> compressed [b, 1, g, head_dim]  (g = s // ratio)."""
        b, _, s, _ = x.shape
        r, d = self.ratio, self.head_dim
        g = s // r

        kv = ttnn.matmul(x, self.wkv_t)                                  # [b, 1, s, coff*d]
        score = ttnn.matmul(x, self.wgate_t)
        kv = ttnn.reshape(kv, (b, g, r, self.coff * d))                  # segments
        score = ttnn.add(ttnn.reshape(score, (b, g, r, self.coff * d)), self.ape)

        if self.overlap:
            kv = self._overlap(kv, 0.0)                                  # [b, g, 2r, d]
            score = self._overlap(score, _NEG)
        weights = ttnn.softmax(score, dim=2)                            # over the segment dim
        pooled = ttnn.sum(ttnn.multiply(kv, weights), dim=2)            # [b, g, d] (or coff*d if no overlap)
        pooled = ttnn.reshape(pooled, (b, 1, g, d))
        return ttnn.rms_norm(pooled, weight=self.norm_weight, epsilon=self.eps)

    def _overlap(self, t: "ttnn.Tensor", pad: float) -> "ttnn.Tensor":
        b, g, r, _ = t.shape
        d = self.head_dim
        a_half = t[:, :, :, :d]
        b_half = t[:, :, :, d:]
        pad_seg = ttnn.full((b, 1, r, d), pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return overlap_window(
            a_half, b_half, pad_seg,
            slice_g=lambda x: x[:, : g - 1],
            concat_g=lambda p, x: ttnn.concat([p, x], dim=1),
            concat_seg=lambda p, x: ttnn.concat([p, x], dim=2),
        )
