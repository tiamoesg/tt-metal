// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Shared-Key-Value Multi-Query Attention with an attention sink, the core
// attention of DeepSeek-V4 CSA and HCA (§2.3.1 eq 19, §2.3.2 eq 26, §2.3.3
// eq 27). Each compressed KV entry serves as both attention key and value
// (shared across all query heads -- MQA), and a learnable per-head sink logit is
// added to the softmax denominator:
//
//   L_{h,t,s} = (q_{t,h} . kv_s) / sqrt(c)
//   weight_{h,t,s} = exp(L_{h,t,s}) / ( sum_s' exp(L_{h,t,s'}) + exp(z'_h) )
//   o_{t,h} = sum_s weight_{h,t,s} * kv_s
//
// The sink term exp(z'_h) lets a query head route attention mass "nowhere" (and
// keeps fully-masked rows numerically safe). DeepSeek applies per-head RMSNorm to
// the queries and KV entries *before* this op, which bounds the logits L -- so we
// compute exp(L) directly without the usual max-subtraction, matching the paper.
//
//   q:           [B, H, S, c]   per-head queries
//   kv:          [B, 1, G, c]   compressed KV entries (shared K = V)
//   sink_logits: [1, H, 1, 1]   learnable per-head sink logits z'_h
//   keep_mask:   [B, 1, S, G]   optional 0/1 selection mask (1 = attend)
//   returns:     [B, H, S, c]
autograd::TensorPtr shared_kv_mqa_attention(
    const autograd::TensorPtr& q,
    const autograd::TensorPtr& kv,
    const autograd::TensorPtr& sink_logits,
    const std::optional<autograd::TensorPtr>& keep_mask = std::nullopt);

}  // namespace ttml::ops
