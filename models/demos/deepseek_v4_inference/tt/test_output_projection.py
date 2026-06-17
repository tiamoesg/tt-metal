# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test: TT-NN grouped output projection vs the CPU reference (roadmap 2.6 gate).

Requires a Tenstorrent device:
    pytest models/demos/deepseek_v4_inference/tt/test_output_projection.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v4_inference.config import V4Config
from models.demos.deepseek_v4_inference.reference_model import Attention
from models.demos.deepseek_v4_inference.tt.output_projection import TtOutputProjection


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def test_output_projection_matches_reference(device):
    torch.manual_seed(0)
    cfg = V4Config.small()
    attn = Attention(cfg, layer_id=0).eval()        # ratio 0 layer; only wo_a/wo_b used here

    b, s = 1, 32
    o = torch.randn(b, s, cfg.n_heads * cfg.head_dim)
    ref = attn.wo_b(attn.wo_a(o))                    # [b, S, dim]

    tt = TtOutputProjection(device, attn.wo_a.weight, attn.wo_b.weight)
    o_tt = ttnn.from_torch(o.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(tt(o_tt)).squeeze(1)

    pcc = _pcc(ref, out)
    assert pcc > 0.99, f"output projection PCC {pcc:.4f}"
