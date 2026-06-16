# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests for the DeepSeek-V4 inference reference model.

No Tenstorrent device required -- run with:
    pytest models/demos/deepseek_v4_inference/test_reference.py
"""

import pytest

torch = pytest.importorskip("torch")

from models.demos.deepseek_v4_inference.config import V4Config
from models.demos.deepseek_v4_inference.reference_model import DeepSeekV4


def _build():
    torch.manual_seed(0)
    cfg = V4Config.small()
    return cfg, DeepSeekV4(cfg).eval()


def test_full_config_schedule_valid():
    cfg = V4Config.full()
    assert len(cfg.compress_ratios) == cfg.n_layers
    assert set(cfg.compress_ratios) <= {0, 4, 128}


def test_forward_shape_and_finite():
    cfg, model = _build()
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = model(tokens)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(logits).all(), "logits must be finite (the attention sink should keep empty rows safe)"


def test_causality_prefix_invariance():
    """A causal model's logits at position t must not change when later tokens change."""
    cfg, model = _build()
    base = torch.randint(0, cfg.vocab_size, (1, 24))
    alt = base.clone()
    alt[0, -1] = (alt[0, -1] + 1) % cfg.vocab_size  # perturb only the last token
    lb = model(base)
    la = model(alt)
    # everything before the last position must be identical
    assert torch.allclose(lb[:, :-1], la[:, :-1], atol=1e-4)


def test_generate_runs():
    cfg, model = _build()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(prompt, max_new_tokens=4)
    assert out.shape == (1, 12)


@pytest.mark.parametrize("score_func", ["sqrtsoftplus", "softmax", "sigmoid"])
def test_moe_score_functions(score_func):
    torch.manual_seed(0)
    cfg = V4Config.small()
    cfg.score_func = score_func
    model = DeepSeekV4(cfg).eval()
    logits = model(torch.randint(0, cfg.vocab_size, (1, 16)))
    assert torch.isfinite(logits).all()
