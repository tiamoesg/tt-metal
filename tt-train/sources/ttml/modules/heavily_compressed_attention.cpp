// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "heavily_compressed_attention.hpp"

#include <fmt/format.h>

#include <core/ttnn_all_includes.hpp>
#include <stdexcept>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/attention_masks.hpp"
#include "ops/concat_op.hpp"
#include "ops/partial_rope_op.hpp"
#include "ops/permute_op.hpp"
#include "ops/reshape_op.hpp"
#include "ops/rope_op.hpp"
#include "ops/sink_attention.hpp"

namespace ttml::modules {

namespace {
const ttnn::SmallVector<int64_t> kHeadsToFront = {0, 2, 1, 3};  // [B,S,H,c] -> [B,H,S,c]
}  // namespace

HeavilyCompressedAttention::HeavilyCompressedAttention(const HeavilyCompressedAttentionConfig& config) :
    m_config(config) {
    if (m_config.dim == 0U || m_config.num_heads == 0U || m_config.head_dim == 0U || m_config.query_comp_dim == 0U ||
        m_config.compression_rate == 0U || m_config.num_groups == 0U || m_config.group_inter_dim == 0U) {
        throw std::invalid_argument("HeavilyCompressedAttentionConfig: all fields must be set.");
    }

    create_name("heavily_compressed_attention");

    m_compressor = std::make_shared<KVCompressor>(KVCompressorConfig{
        .dim = m_config.dim, .compressed_dim = m_config.head_dim, .compression_rate = m_config.compression_rate});
    m_w_dq = std::make_shared<LinearLayer>(m_config.dim, m_config.query_comp_dim, /* has_bias */ false);
    m_w_uq = std::make_shared<LinearLayer>(
        m_config.query_comp_dim, m_config.num_heads * m_config.head_dim, /* has_bias */ false);
    m_q_norm = std::make_shared<RMSNormLayer>(m_config.head_dim);
    m_kv_norm = std::make_shared<RMSNormLayer>(m_config.head_dim);
    m_sink_logits =
        autograd::create_tensor(core::zeros(ttnn::Shape({1, m_config.num_heads, 1, 1}), &autograd::ctx().get_device()));
    m_out_proj = std::make_shared<GroupedOutputProjection>(GroupedOutputProjectionConfig{
        .num_heads = m_config.num_heads,
        .head_dim = m_config.head_dim,
        .num_groups = m_config.num_groups,
        .group_inter_dim = m_config.group_inter_dim,
        .out_dim = m_config.dim});

    register_module(m_compressor, "compressor");
    register_module(m_w_dq, "w_dq");
    register_module(m_w_uq, "w_uq");
    register_module(m_q_norm, "q_norm");
    register_module(m_kv_norm, "kv_norm");
    register_tensor(m_sink_logits, "sink_logits");
    register_module(m_out_proj, "out_proj");

    if (m_config.sliding_window > 0U) {
        m_w_win = std::make_shared<LinearLayer>(m_config.dim, m_config.head_dim, /* has_bias */ false);
        m_win_norm = std::make_shared<RMSNormLayer>(m_config.head_dim);
        register_module(m_w_win, "w_win");
        register_module(m_win_norm, "win_norm");
    }

    if (m_config.rope_head_dim > 0U) {
        if (m_config.rope_head_dim >= m_config.head_dim || m_config.rope_max_seq == 0U) {
            throw std::invalid_argument(
                "HeavilyCompressedAttention: rope_head_dim must be < head_dim and rope_max_seq must be set.");
        }
        m_rope_params = ops::build_rope_params(m_config.rope_max_seq, m_config.rope_head_dim, m_config.rope_theta);
        m_use_rope = true;
    }
}

autograd::TensorPtr HeavilyCompressedAttention::operator()(const autograd::TensorPtr& hidden) {
    const auto shape = hidden->get_value().logical_shape().to_array_4D();
    const uint32_t batch = shape[0];
    const uint32_t seq = shape[2];
    const uint32_t heads = m_config.num_heads;
    const uint32_t head_dim = m_config.head_dim;

    auto* device = &autograd::ctx().get_device();

    // Compressed, normalized shared KV entries: [B, 1, G, c].
    auto kv = (*m_kv_norm)((*m_compressor)(hidden));
    const uint32_t groups = kv->get_value().logical_shape().to_array_4D()[2];

    // Low-rank query generation, split into heads, then per-head RMSNorm.
    auto q = (*m_w_uq)((*m_w_dq)(hidden));  // [B, 1, S, n_h*c]
    auto q_heads = ops::permute(ops::reshape(q, ttnn::Shape({batch, seq, heads, head_dim})), kHeadsToFront);
    q_heads = (*m_q_norm)(q_heads);  // [B, n_h, S, c]

    // Partial RoPE (§2.3.3): queries at token positions, compressed blocks at the
    // strided positions s*ratio; the output is later inverse-rotated (the -i trick).
    if (m_use_rope) {
        const auto query_rope = ops::rope_params_prefix(m_rope_params, seq);
        q_heads = ops::partial_rope(q_heads, query_rope, /* token_position */ 0);
        const auto block_rope = ops::rope_params_strided(m_rope_params, m_config.compression_rate, groups);
        kv = ops::partial_rope(kv, block_rope, /* token_position */ 0);
    }

    // Compressed-causal keep mask over the G compressed blocks.
    auto compressed_mask = ops::compressed_causal_keep(seq, groups, m_config.compression_rate, device);  // [1,1,S,G]

    autograd::TensorPtr attn;
    if (m_config.sliding_window == 0U) {
        // Dense (causal) shared-KV MQA with attention sink, over compressed blocks.
        auto causal = autograd::create_tensor(compressed_mask);
        attn = ops::shared_kv_mqa_attention(q_heads, kv, m_sink_logits, causal);  // [B, n_h, S, c]
    } else {
        // Sliding-window branch: recent uncompressed tokens (RoPE'd at their token
        // positions), concatenated with the compressed entries (§2.3.3).
        auto win_kv = (*m_win_norm)((*m_w_win)(hidden));  // [B, 1, S, c]
        if (m_use_rope) {
            win_kv = ops::partial_rope(win_kv, ops::rope_params_prefix(m_rope_params, seq), 0);
        }
        auto combined_kv = ops::concat({win_kv, kv}, /* dim */ 2);  // [B, 1, S+G, c]

        // Combined keep mask [B, 1, S, S+G] = [sliding-window-causal | compressed-causal].
        auto window_mask = ops::sliding_window_keep(seq, m_config.sliding_window, device);  // [1,1,S,S]
        auto combined_raw = ttnn::concat(std::vector<ttnn::Tensor>{window_mask, compressed_mask}, /* dim */ 3);
        auto combined_mask = autograd::create_tensor(ttnn::repeat(combined_raw, ttnn::Shape({batch, 1, 1, 1})));
        attn = ops::shared_kv_mqa_attention(q_heads, combined_kv, m_sink_logits, combined_mask);  // [B, n_h, S, c]
    }

    // The "-i" output trick: inverse-rotate the output's last dims at query
    // positions so the value-carried absolute positions become relative.
    if (m_use_rope) {
        attn = ops::partial_rope(attn, ops::rope_params_inverse(ops::rope_params_prefix(m_rope_params, seq)), 0);
    }
    return (*m_out_proj)(attn);  // [B, 1, S, d]
}

}  // namespace ttml::modules
