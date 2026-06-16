# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test: TT-NN lightning indexer vs the CPU reference (roadmap step 2.3 gate).

Requires a Tenstorrent device:
    pytest models/demos/deepseek_v4_inference/tt/test_indexer.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v4_inference.config import V4Config
from models.demos.deepseek_v4_inference.reference_model import Indexer
from models.demos.deepseek_v4_inference.tt.compressor import TtCompressor
from models.demos.deepseek_v4_inference.tt.indexer import TtIndexer


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def _build(device):
    torch.manual_seed(0)
    cfg = V4Config.small()
    ratio = 4
    ref = Indexer(cfg, ratio).eval()
    comp = ref.compressor
    tt_comp = TtCompressor(device, comp.wkv.weight, comp.wgate.weight, comp.ape, comp.norm.weight,
                           ratio, cfg.index_head_dim, cfg.norm_eps)
    tt = TtIndexer(device, ref.wq.weight, ref.wproj.weight, tt_comp,
                   cfg.index_n_heads, cfg.index_head_dim, ref.scale, ratio, cfg.index_topk)
    return cfg, ref, tt


def test_indexer_scores_match_reference(device):
    cfg, ref, tt = _build(device)
    b, s = 1, 32
    x = torch.randn(b, s, cfg.dim)
    qr = torch.randn(b, s, cfg.q_lora_rank)

    ref_scores, _ = ref.scores(x, qr)  # [b, S, G]
    x_tt = ttnn.from_torch(x.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    qr_tt = ttnn.from_torch(qr.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(tt.scores(x_tt, qr_tt)).squeeze(1)  # [b, S, G]

    pcc = _pcc(ref_scores, out)
    assert pcc > 0.99, f"indexer scores PCC {pcc:.4f}"


def test_indexer_selection_agrees(device):
    cfg, ref, tt = _build(device)
    b, s = 1, 32
    x = torch.randn(b, s, cfg.dim)
    qr = torch.randn(b, s, cfg.q_lora_rank)

    ref_keep = ref(x, qr).float()  # [b, S, G] bool
    x_tt = ttnn.from_torch(x.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    qr_tt = ttnn.from_torch(qr.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_keep = ttnn.to_torch(tt.select(tt.scores(x_tt, qr_tt))).squeeze(1)

    # bf16 can flip a few borderline top-k picks; require high agreement, not identity.
    agree = (ref_keep == (tt_keep > 0.5)).float().mean()
    assert agree > 0.95, f"selection agreement {float(agree):.4f}"
