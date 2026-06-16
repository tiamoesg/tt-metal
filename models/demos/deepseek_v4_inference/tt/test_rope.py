# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test: TT-NN partial RoPE vs the CPU reference (roadmap step 2.1 gate).

Requires a Tenstorrent device (uses the standard `device` pytest fixture):
    pytest models/demos/deepseek_v4_inference/tt/test_rope.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v4_inference.reference_model import partial_rope as ref_partial_rope
from models.demos.deepseek_v4_inference.reference_model import precompute_freqs_cis
from models.demos.deepseek_v4_inference.tt.rope import TtRope, cos_sin_from_freqs


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse_minus_i"])
@pytest.mark.parametrize("ratio", [1, 4], ids=["contiguous", "strided_s_times_ratio"])
def test_partial_rope_matches_reference(device, inverse, ratio):
    B, H, S, full, rope_dim = 1, 4, 32, 64, 32  # nope=32, rope=32 (both tile-aligned)
    torch.manual_seed(0)
    x = torch.randn(B, H, S, full)

    # Contiguous token positions (queries / window) or strided s*ratio (compressed blocks).
    n_pos = S if ratio == 1 else S // ratio
    freqs = precompute_freqs_cis(rope_dim, S * ratio, 0, 10000.0, 1.0, 32, 1)  # original_seq_len=0 -> no YaRN
    freqs = freqs[::ratio][:n_pos]
    x = x[:, :, :n_pos]

    ref = ref_partial_rope(x, freqs, rope_dim, inverse=inverse)

    cos, sin = cos_sin_from_freqs(freqs)
    tt_rope = TtRope(device, cos, sin, rope_dim)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(tt_rope(x_tt, inverse=inverse))

    pcc = _pcc(ref, out)
    assert pcc > 0.99, f"partial RoPE PCC {pcc:.4f} (inverse={inverse}, ratio={ratio})"
