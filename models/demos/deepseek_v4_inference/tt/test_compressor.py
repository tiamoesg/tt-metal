# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test: TT-NN KV compressor vs the CPU reference (roadmap step 2.2 gate).

Requires a Tenstorrent device:
    pytest models/demos/deepseek_v4_inference/tt/test_compressor.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v4_inference.config import V4Config
from models.demos.deepseek_v4_inference.reference_model import Compressor
from models.demos.deepseek_v4_inference.tt.compressor import TtCompressor


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


@pytest.mark.parametrize("ratio", [4, 8], ids=["csa_overlap", "hca_single"])
def test_compressor_matches_reference(device, ratio):
    torch.manual_seed(0)
    cfg = V4Config.small()
    head_dim = 64
    b, s = 1, 32  # divisible by both 4 and 8

    ref = Compressor(cfg, ratio, head_dim).eval()
    x = torch.randn(b, s, cfg.dim)
    ref_out = ref(x)  # [b, g, head_dim]

    tt = TtCompressor(device, ref.wkv.weight, ref.wgate.weight, ref.ape, ref.norm.weight,
                      ratio, head_dim, cfg.norm_eps)
    x_tt = ttnn.from_torch(x.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(tt(x_tt)).squeeze(1)  # [b, g, head_dim]

    pcc = _pcc(ref_out, out)
    # NOTE: bf16 softmax-pool vs fp32 reference -- bump intermediates to fp32 if this is tight.
    assert pcc > 0.99, f"compressor PCC {pcc:.4f} (ratio={ratio})"
