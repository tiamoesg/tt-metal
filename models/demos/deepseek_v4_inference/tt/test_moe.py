# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test: TT-NN DeepSeekMoE vs the CPU reference (roadmap 2.8 gate).

The TT module is the dense-equivalent of the reference's gather/scatter token-choice
routing (every token through every expert, weighted by the gate; 0 for unselected),
so the two should match to PCC. Exercised for both `sqrtsoftplus` and `softmax` gates.

Requires a Tenstorrent device:
    pytest models/demos/deepseek_v4_inference/tt/test_moe.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v4_inference.config import V4Config
from models.demos.deepseek_v4_inference.reference_model import MoE
from models.demos.deepseek_v4_inference.tt.moe import TtMoE


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def _triples(experts):
    return [(e.w1.weight, e.w2.weight, e.w3.weight) for e in experts]


@pytest.mark.parametrize("score_func", ["sqrtsoftplus", "softmax"])
def test_moe_matches_reference(device, score_func):
    torch.manual_seed(0)
    cfg = V4Config.small()
    cfg.score_func = score_func
    ref = MoE(cfg).eval()

    b, s = 1, 32
    x = torch.randn(b, s, cfg.dim) * 0.5
    out_ref = ref(x)                                  # [b, S, dim]

    tt = TtMoE(device, ref.gate.weight, _triples(ref.experts), _triples(ref.shared),
               cfg.n_activated_experts, cfg.route_scale, score_func,
               cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit)
    x_tt = ttnn.from_torch(x.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(tt(x_tt)).squeeze(1)

    pcc = _pcc(out_ref, out)
    assert pcc > 0.99, f"MoE PCC {pcc:.4f} ({score_func})"
