// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sink_attention.hpp"

#include <cmath>
#include <core/ttnn_all_includes.hpp>

#include "ops/binary_ops.hpp"
#include "ops/matmul_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr shared_kv_mqa_attention(
    const autograd::TensorPtr& q,
    const autograd::TensorPtr& kv,
    const autograd::TensorPtr& sink_logits,
    const std::optional<autograd::TensorPtr>& keep_mask) {
    const auto q_shape = q->get_value().logical_shape().to_array_4D();
    const uint32_t batch = q_shape[0];
    const uint32_t heads = q_shape[1];
    const uint32_t seq = q_shape[2];
    const uint32_t head_dim = q_shape[3];
    const uint32_t groups = kv->get_value().logical_shape().to_array_4D()[2];
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    // Share the compressed KV entries across all query heads (MQA).
    auto kv_bc = ops::broadcast_to(kv, ttnn::Shape({batch, heads, groups, head_dim}));  // [B, H, G, c]

    // Scaled logits L = (q . kv^T) / sqrt(c).
    auto logits = ops::mul(ops::matmul_op(q, kv_bc, /* transpose_a */ false, /* transpose_b */ true), scale);

    // Apply the selection mask as an additive bias (0 for kept, large negative for
    // dropped). The mask carries no gradient, so it is folded in as a constant.
    if (keep_mask.has_value()) {
        constexpr float kBig = 1.0e4F;
        auto m = keep_mask.value()->get_value();                    // [B, 1, S, G]
        auto bias = ttnn::multiply(ttnn::subtract(m, 1.0F), kBig);  // 0 / -kBig
        bias = ttnn::repeat(bias, ttnn::Shape({1, heads, 1, 1}));   // -> [B, H, S, G]
        logits = ops::add(logits, bias);
    }

    // Sink-augmented softmax over the G compressed blocks. RMSNorm upstream keeps
    // the logits bounded, so exp is taken directly (no max-subtraction).
    auto numerator = ops::exp(logits);                                    // [B, H, S, G]
    auto denom = ops::sum(numerator, /* dim */ -1, /* keep_dim */ true);  // [B, H, S, 1]
    auto sink = ops::broadcast_to(ops::exp(sink_logits), ttnn::Shape({batch, heads, seq, 1U}));
    denom = ops::add(denom, sink);  // + exp(z'_h)
    auto weights = ops::div(numerator, ops::broadcast_to(denom, ttnn::Shape({batch, heads, seq, groups})));

    // Weighted sum of the (shared) value entries.
    return ops::matmul_op(weights, kv_bc, /* transpose_a */ false, /* transpose_b */ false);  // [B, H, S, c]
}

}  // namespace ttml::ops
