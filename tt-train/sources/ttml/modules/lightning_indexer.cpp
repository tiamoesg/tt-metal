// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightning_indexer.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <core/ttnn_all_includes.hpp>
#include <stdexcept>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/matmul_op.hpp"
#include "ops/permute_op.hpp"
#include "ops/reshape_op.hpp"
#include "ops/unary_ops.hpp"
#include "ttnn/operations/reduction/topk/topk.hpp"

namespace ttml::modules {

namespace {

// Move (heads, seq) into [B, H, S, *] layout, identical to MLA's split_heads:
// [B, 1, S, H*D] -> [B, S, H, D] -> permute(0,2,1,3) -> [B, H, S, D].
const ttnn::SmallVector<int64_t> kHeadsToFront = {0, 2, 1, 3};

// Compressed-causal keep mask [1, 1, S, G]: query token t (in compressed block
// floor(t/m)) may attend to compressed block s iff s < floor(t/m). 1 = allowed.
tt::tt_metal::Tensor compressed_causal_keep(
    uint32_t seq, uint32_t groups, uint32_t rate, ttnn::distributed::MeshDevice* device) {
    std::vector<float> data(static_cast<size_t>(seq) * groups, 0.0F);
    for (uint32_t t = 0; t < seq; ++t) {
        const uint32_t allowed = t / rate;  // blocks [0, allowed) are visible
        for (uint32_t s = 0; s < groups; ++s) {
            data[static_cast<size_t>(t) * groups + s] = (s < allowed) ? 1.0F : 0.0F;
        }
    }
    return core::from_vector(data, ttnn::Shape({1, 1, seq, groups}), device);
}

}  // namespace

LightningIndexer::LightningIndexer(const LightningIndexerConfig& config) : m_config(config) {
    if (m_config.dim == 0U || m_config.index_query_dim == 0U || m_config.index_head_dim == 0U ||
        m_config.num_index_heads == 0U) {
        throw std::invalid_argument(
            "LightningIndexerConfig: dim, index_query_dim, index_head_dim and "
            "num_index_heads must all be set.");
    }

    create_name("lightning_indexer");
    m_w_dq = std::make_shared<LinearLayer>(m_config.dim, m_config.index_query_dim, /* has_bias */ false);
    m_w_iuq = std::make_shared<LinearLayer>(
        m_config.index_query_dim, m_config.index_head_dim * m_config.num_index_heads, /* has_bias */ false);
    m_w_w = std::make_shared<LinearLayer>(m_config.dim, m_config.num_index_heads, /* has_bias */ false);
    register_module(m_w_dq, "w_dq");
    register_module(m_w_iuq, "w_iuq");
    register_module(m_w_w, "w_w");
}

autograd::TensorPtr LightningIndexer::index_scores(
    const autograd::TensorPtr& query_hidden, const autograd::TensorPtr& index_keys) {
    const auto qshape = query_hidden->get_value().logical_shape().to_array_4D();
    const uint32_t batch = qshape[0];
    const uint32_t seq = qshape[2];
    const uint32_t groups = index_keys->get_value().logical_shape().to_array_4D()[2];
    const uint32_t heads = m_config.num_index_heads;
    const uint32_t head_dim = m_config.index_head_dim;

    // Low-rank indexer queries: c^Q = h W^DQ, q^I = c^Q W^IUQ -> per-head [B, H, S, c^I].
    auto cq = (*m_w_dq)(query_hidden);  // [B, 1, S, d_c]
    auto qi = (*m_w_iuq)(cq);           // [B, 1, S, H * c^I]
    auto qi_h = ops::permute(ops::reshape(qi, ttnn::Shape({batch, seq, heads, head_dim})), kHeadsToFront);

    // Per-head weights w^I = h W^w -> [B, H, S, 1].
    auto wi = (*m_w_w)(query_hidden);  // [B, 1, S, H]
    auto wi_h = ops::permute(ops::reshape(wi, ttnn::Shape({batch, seq, heads, 1U})), kHeadsToFront);

    // Index keys are shared across heads: broadcast [B, 1, G, c^I] -> [B, H, G, c^I].
    auto keys = ops::broadcast_to(index_keys, ttnn::Shape({batch, heads, groups, head_dim}));

    // I_{t,s} = sum_h w^I_{t,h} * ReLU(q^I_{t,h} . K_s).
    auto dots = ops::matmul_op(qi_h, keys, /* transpose_a */ false, /* transpose_b */ true);  // [B, H, S, G]
    auto activated = ops::relu(dots);
    auto weighted = ops::mul(activated, ops::broadcast_to(wi_h, ttnn::Shape({batch, heads, seq, groups})));
    return ops::sum(weighted, /* dim */ 1, /* keep_dim */ true);  // [B, 1, S, G]
}

tt::tt_metal::Tensor LightningIndexer::selection_mask(const autograd::TensorPtr& index_scores_tensor) const {
    const auto scores = index_scores_tensor->get_value();  // [B, 1, S, G]
    const auto shape = scores.logical_shape().to_array_4D();
    const uint32_t batch = shape[0];
    const uint32_t seq = shape[2];
    const uint32_t groups = shape[3];
    auto* device = &autograd::ctx().get_device();

    uint32_t k = std::min(m_config.top_k, groups);
    if (k == 0U) {
        k = 1U;
    }

    // Compressed-causal keep mask, broadcast over the batch.
    auto causal = compressed_causal_keep(seq, groups, m_config.compression_rate, device);  // [1, 1, S, G]
    auto causal_b = ttnn::repeat(causal, ttnn::Shape({batch, 1, 1, 1}));                   // [B, 1, S, G]

    // Mask out non-causal blocks before selecting, so they can never be chosen.
    auto masked_scores = ttnn::where(causal_b, scores, /* other */ -1e9F);

    // Top-k threshold = k-th largest score per query (sorted descending).
    auto topk_out = ttnn::topk(masked_scores, k, /* dim */ -1, /* largest */ true, /* sorted */ true);
    const ttsl::SmallVector<uint32_t> start = {0U, 0U, 0U, k - 1U};
    const ttsl::SmallVector<uint32_t> end = {batch, 1U, seq, k};
    const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};
    auto threshold = ttnn::slice(topk_out[0], start, end, step);                 // [B, 1, S, 1]
    auto threshold_b = ttnn::repeat(threshold, ttnn::Shape({1, 1, 1, groups}));  // [B, 1, S, G]

    // keep = (score >= threshold) AND causal.
    auto selected = ttnn::ge(ttnn::subtract(masked_scores, threshold_b), 0.0F);  // 1 / 0
    return ttnn::multiply(selected, causal_b);                                   // [B, 1, S, G]
}

}  // namespace ttml::modules
