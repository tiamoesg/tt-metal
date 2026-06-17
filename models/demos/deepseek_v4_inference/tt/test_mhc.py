# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test: TT-NN mHC residual mixing vs the CPU reference (roadmap 2.7 gate).

Drives the full pre-combine -> sublayer -> post-expand cycle (with an identity
sublayer h = y) so `pre`, `post`, and the Sinkhorn `comb` are all exercised, then
compares the rewritten `hc` streams to `reference_model.Block`'s `_hc_pre`/`_hc_post`.

Requires a Tenstorrent device:
    pytest models/demos/deepseek_v4_inference/tt/test_mhc.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v4_inference.config import V4Config
from models.demos.deepseek_v4_inference.reference_model import Block
from models.demos.deepseek_v4_inference.tt.mhc import TtHyperConnections


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def test_mhc_pre_post_cycle_matches_reference(device):
    torch.manual_seed(0)
    cfg = V4Config.small()
    block = Block(cfg, layer_id=0).eval()
    hc, d = cfg.hc_mult, cfg.dim

    b, s = 1, 32
    x = torch.randn(b, s, hc, d) * 0.5

    # reference: pre-combine -> identity sublayer (h = y) -> post-expand
    y, post, comb = block._hc_pre(x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    ref_out = block._hc_post(y, x, post, comb)        # [b, S, hc, d]

    # tt: same sequence on streams carried as a list of hc [b,1,S,d]
    hcn = TtHyperConnections(device, block.hc_attn_fn, block.hc_attn_base, block.hc_attn_scale,
                             hc, cfg.hc_eps, cfg.hc_sinkhorn_iters)
    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    streams = [to_dev(x[:, :, k, :].unsqueeze(1).contiguous()) for k in range(hc)]  # hc x [b,1,S,d]

    pre, post_tt, comb_tt = hcn.mix(streams)
    y_tt = hcn.combine(streams, pre)
    out_streams = hcn.expand(y_tt, streams, post_tt, comb_tt)
    out = torch.stack([ttnn.to_torch(t).squeeze(1) for t in out_streams], dim=2)  # [b, S, hc, d]

    pcc = _pcc(ref_out, out)
    assert pcc > 0.99, f"mHC pre/post cycle PCC {pcc:.4f}"
