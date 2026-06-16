# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test: TT-NN sink attention vs the CPU reference (roadmap step 2.4 gate, math core).

Requires a Tenstorrent device:
    pytest models/demos/deepseek_v4_inference/tt/test_sink_attention.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v4_inference.reference_model import masked_sink_attention
from models.demos.deepseek_v4_inference.tt.sink_attention import sink_attention


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def test_sink_attention_matches_reference(device):
    torch.manual_seed(0)
    b, h, s, k, hd = 1, 4, 32, 32, 64
    scale = hd ** -0.5

    q = torch.randn(b, h, s, hd) * 0.2          # small so bf16 exp is well-behaved
    keys = torch.randn(b, k, hd) * 0.2
    keep = (torch.randn(b, s, k) > 0.0).float()  # ~half kept
    keep[:, 0, :] = 0.0                           # one fully-masked row -> sink must give ~0
    sink = torch.zeros(h)

    ref = masked_sink_attention(q, keys, keep, sink, scale)  # [b, H, S, hd]

    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(
        sink_attention(
            to_dev(q),
            to_dev(keys.unsqueeze(1)),       # [b,1,K,hd]
            to_dev(keep.unsqueeze(1)),       # [b,1,S,K]
            to_dev(sink.reshape(1, h, 1, 1)),
            scale,
        )
    )

    pcc = _pcc(ref, out)
    assert pcc > 0.99, f"sink attention PCC {pcc:.4f}"
    # the fully-masked row must be ~0 (sink safety)
    assert out[:, :, 0, :].abs().max() < 1e-2, "fully-masked row should be ~0"
