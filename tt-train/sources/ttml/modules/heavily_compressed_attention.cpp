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
#include "ops/permute_op.hpp"
#include "ops/reshape_op.hpp"
#include "ops/sink_attention.hpp"

namespace ttml::modules {

namespace {

const ttnn::SmallVector<int64_t> kHeadsToFront = {0, 2, 1, 3};  // [B,S,H,c] -> [B,H,S,c]

// Compressed-causal keep mask [1, 1, S, G]: query t may attend to compressed
// block s iff s < floor(t/m') (a query cannot see its own or later blocks; the
// sliding-window branch -- not yet wired -- covers the local context).
tt::tt_metal::Tensor compressed_causal_keep(
    uint32_t seq, uint32_t groups, uint32_t rate, ttnn::distributed::MeshDevice* device) {
    std::vector<float> data(static_cast<size_t>(seq) * groups, 0.0F);
    for (uint32_t t = 0; t < seq; ++t) {
        const uint32_t allowed = t / rate;
        for (uint32_t s = 0; s < groups; ++s) {
            data[static_cast<size_t>(t) * groups + s] = (s < allowed) ? 1.0F : 0.0F;
        }
    }
    return core::from_vector(data, ttnn::Shape({1, 1, seq, groups}), device);
}

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
}

autograd::TensorPtr HeavilyCompressedAttention::operator()(const autograd::TensorPtr& hidden) {
    const auto shape = hidden->get_value().logical_shape().to_array_4D();
    const uint32_t batch = shape[0];
    const uint32_t seq = shape[2];
    const uint32_t heads = m_config.num_heads;
    const uint32_t head_dim = m_config.head_dim;

    // Compressed, normalized shared KV entries: [B, 1, G, c].
    auto kv = (*m_kv_norm)((*m_compressor)(hidden));
    const uint32_t groups = kv->get_value().logical_shape().to_array_4D()[2];

    // Low-rank query generation, split into heads, then per-head RMSNorm.
    auto q = (*m_w_uq)((*m_w_dq)(hidden));  // [B, 1, S, n_h*c]
    auto q_heads = ops::permute(ops::reshape(q, ttnn::Shape({batch, seq, heads, head_dim})), kHeadsToFront);
    q_heads = (*m_q_norm)(q_heads);  // [B, n_h, S, c]

    // Dense (causal) shared-KV MQA with attention sink.
    auto causal = autograd::create_tensor(
        compressed_causal_keep(seq, groups, m_config.compression_rate, &autograd::ctx().get_device()));
    auto attn = ops::shared_kv_mqa_attention(q_heads, kv, m_sink_logits, causal);  // [B, n_h, S, c]

    return (*m_out_proj)(attn);  // [B, 1, S, d]
}

}  // namespace ttml::modules
